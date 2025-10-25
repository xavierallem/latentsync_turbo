# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from omegaconf import OmegaConf
import torch
from diffusers import AutoencoderKL, DDIMScheduler, DPMSolverMultistepScheduler
from latentsync.models.unet import UNet3DConditionModel
from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
from accelerate.utils import set_seed
from latentsync.whisper.audio2feature import Audio2Feature
from latentsync.utils.telemetry import TelemetrySession
from DeepCache import DeepCacheSDHelper


def main(config, args):
    if not os.path.exists(args.video_path):
        raise RuntimeError(f"Video path '{args.video_path}' not found")
    if not os.path.exists(args.audio_path):
        raise RuntimeError(f"Audio path '{args.audio_path}' not found")

    # Check if the GPU supports float16
    is_fp16_supported = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 7
    dtype = torch.float16 if is_fp16_supported else torch.float32

    print(f"Input video path: {args.video_path}")
    print(f"Input audio path: {args.audio_path}")
    print(f"Loaded checkpoint path: {args.inference_ckpt_path}")

    scheduler = DDIMScheduler.from_pretrained("configs")
    if getattr(args, "use_dpm_solver", False):
        scheduler = DPMSolverMultistepScheduler.from_config(
            scheduler.config,
            lower_order_final=True,
        )
        if getattr(scheduler.config, "steps_offset", 0) != 0:
            scheduler.register_to_config(steps_offset=0)
        if getattr(scheduler.config, "solver_order", None) != 1:
            scheduler.register_to_config(solver_order=1)
        scheduler.solver_order = 1

    scheduler_name = type(scheduler).__name__
    print(f"Scheduler: {scheduler_name} (use_dpm_solver={getattr(args, 'use_dpm_solver', False)})")

    if config.model.cross_attention_dim == 768:
        whisper_model_path = "checkpoints/whisper/small.pt"
    elif config.model.cross_attention_dim == 384:
        whisper_model_path = "checkpoints/whisper/tiny.pt"
    else:
        raise NotImplementedError("cross_attention_dim must be 768 or 384")

    audio_encoder = Audio2Feature(
        model_path=whisper_model_path,
        device="cuda",
        num_frames=config.data.num_frames,
        audio_feat_length=config.data.audio_feat_length,
    )

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=dtype)
    vae.config.scaling_factor = 0.18215
    vae.config.shift_factor = 0

    unet, _ = UNet3DConditionModel.from_pretrained(
        OmegaConf.to_container(config.model),
        args.inference_ckpt_path,
        device="cpu",
    )

    unet = unet.to(dtype=dtype)

    pipeline = LipsyncPipeline(
        vae=vae,
        audio_encoder=audio_encoder,
        unet=unet,
        scheduler=scheduler,
    ).to("cuda")

    # use DeepCache
    if args.enable_deepcache:
        helper = DeepCacheSDHelper(pipe=pipeline)
        helper.set_params(cache_interval=3, cache_branch_id=0)
        helper.enable()

    if args.seed != -1:
        set_seed(args.seed)
    else:
        torch.seed()

    print(f"Initial seed: {torch.initial_seed()}")

    telemetry = TelemetrySession()
    with telemetry:
        pipeline(
            video_path=args.video_path,
            audio_path=args.audio_path,
            video_out_path=args.video_out_path,
            num_frames=config.data.num_frames,
            num_inference_steps=args.inference_steps,
            guidance_scale=args.guidance_scale,
            weight_dtype=dtype,
            width=config.data.resolution,
            height=config.data.resolution,
            mask_image_path=config.data.mask_image_path,
            temp_dir=args.temp_dir,
        )

    metrics = telemetry.metrics()

    temp_dir = Path(args.temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_stem = Path(args.video_path).stem or "latentsync"
    metrics_path = temp_dir / f"{video_stem}_{timestamp}_metrics.json"

    metrics_record = {
        "timestamp": timestamp,
        "video_path": args.video_path,
        "audio_path": args.audio_path,
        "output_path": args.video_out_path,
        "guidance_scale": args.guidance_scale,
        "inference_steps": args.inference_steps,
        "use_dpm_solver": getattr(args, "use_dpm_solver", False),
        "seed": int(args.seed) if hasattr(args, "seed") else None,
        "metrics": metrics,
    }

    metrics_path.write_text(json.dumps(metrics_record, indent=2))
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--unet_config_path", type=str, default="configs/unet.yaml")
    parser.add_argument("--inference_ckpt_path", type=str, required=True)
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--video_out_path", type=str, required=True)
    parser.add_argument("--inference_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--temp_dir", type=str, default="temp")
    parser.add_argument("--seed", type=int, default=1247)
    parser.add_argument("--enable_deepcache", action="store_true")
    parser.set_defaults(use_dpm_solver=True)
    parser.add_argument("--use_dpm_solver", action="store_true")
    parser.add_argument(
        "--use_ddim_scheduler",
        action="store_false",
        dest="use_dpm_solver",
        help="Fallback to DDIM scheduler instead of DPM-Solver.",
    )
    args = parser.parse_args()

    config = OmegaConf.load(args.unet_config_path)

    main(config, args)

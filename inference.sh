#!/bin/bash

python -m scripts.inference \
    --unet_config_path "configs/unet/stage2_efficient.yaml" \
    --inference_ckpt_path "checkpoints/latentsync_unet.pt" \
    --inference_steps 17 \
    --guidance_scale 1.7 \
    --enable_deepcache \
    --video_path "assets/demo1_video.mp4" \
    --audio_path "assets/demo1_audio.wav" \
    --video_out_path "video_out.mp4" \
    #--use_ddim_scheduler \
    #--use_quantization

    # --use_flash_attention
    # --use_ddim_scheduler
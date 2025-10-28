# LatentSync Inference Results Analysis

This document analyzes the performance metrics from various inference configurations on the demo video (`assets/demo1_video.mp4` with `assets/demo1_audio.wav`). Tested on Latentsync 1.5 , will scale well for 1.6

## Summary Table

| Config                  | Resolution | Inference Steps | Scheduler | Elapsed Time (s) | Peak GPU Memory (MiB) | SyncNet Confidence | AV Offset |
|-------------------------|------------|-----------------|-----------|------------------|------------------------|---------------------|-----------|
| Baseline (DDIM)        | 512       | 25             | DDIM     | 2076            | 13404                 | 2.06               | 0        |
| Baseline 15 Steps      | 512       | 15             | DDIM     | 1547            | 13407                 | 1.99               | 0        |
| Baseline 25 Steps (256)| 256       | 25             | DDIM     | 91              | 7637                  | 9.75               | 0        |
| DPM 15 Steps           | 512       | 15             | DPM      | 1552            | 13407                 | 2.03               | 0        |
| DPM tweek              | 512       | 15             | DPM      | 1185            | 12386                 | 2.01               | 0        |
| DPM FA                 | 512       | 15             | DPM      | 1773            | 13402                 | 2.04               | 0        |
| DPM 17 Steps           | 256       | 17             | DPM      | 80              | 7640                  | 9.78               | 0        |
| 4-bit Quant + DPM 25   | 256       | 25             | DPM      | 65              | 7542                  | 1.12               | -10      |
| 8-bit Quant + DPM 17   | 256       | 17             | DPM      | 82              | 7640                  | 9.84               | 0        |


### Note:- Whisper Tiny Quants

## Video Results

Here are playable examples of the generated videos for key configurations. Videos are attached as GitHub assets for better performance.

<table class="center">
  <tr style="font-weight: bolder;text-align:center;">
    <td width="50%"><b>Best Overall: 8-bit Quant + DPM 17 (256px)</b></td>
    <td width="50%"><b>Fastest: DPM 17 Steps (256px)</b></td>
  </tr>
  <tr>
    <td>
      <video src="https://github.com/user-attachments/assets/video_Q8_E.mp4" controls preload="metadata" width="100%"></video>
    </td>
    <td>
      <video src="https://github.com/user-attachments/assets/video_DPM_E.mp4" controls preload="metadata" width="100%"></video>
    </td>
  </tr>
  <tr style="font-weight: bolder;text-align:center;">
    <td width="50%"><b>Baseline: DDIM 25 Steps (256px)</b></td>
    <td width="50%"><b>Optimized: DPM Tweek (512px)</b></td>
  </tr>
  <tr>
    <td>
      <video src="https://github.com/user-attachments/assets/video_out_baseline_25step_E.mp4" controls preload="metadata" width="100%"></video>
    </td>
    <td>
      <video src="https://github.com/user-attachments/assets/video_out_DPM_tweek.mp4" controls preload="metadata" width="100%"></video>
    </td>
  </tr>
</table>

*Highest SyncNet confidence (9.84) | Good quality + fastest inference | Traditional approach comparison | Torch optimizations for 512px*

## Key Findings

- **Resolution Impact**: 256x256 configs use ~40% less memory (7.5-7.6 GB vs. 13.4 GB) and are ~20-25x faster, with significantly higher SyncNet confidence (9.75-9.84 vs. 1.99-2.06) across schedulers.
- **Scheduler Impact**: DPM-Solver is ~20x faster than DDIM at 512, but at 256, DDIM is still slow (91s) compared to DPM (80-82s).
- **Step Count**: 15-17 steps with DPM at 256 maintain high quality.
- **Quantization**: 8-bit at 256 resolution achieves best quality (SyncNet 9.84). 4-bit reduces memory but quality drops (SyncNet 1.12, AV offset -10).
- **Memory Usage**: 512 configs peak at ~13.4 GB; 256 at ~7.5-7.6 GB.
- **Quality Metrics**: Higher SyncNet confidence at 256 indicates better lip-sync, regardless of scheduler.
- **Optimizations at 512**: DPM tweek (torch optimizations) reduces time by ~25% (1185s vs 1552s) over standard DPM at similar quality. DPM FA (custom Flash Attention 2) increases memory slightly but maintains quality, though slower (1773s).
- **CUDA Runtime Optimizations**: Enabled TF32 precision (`torch.backends.cuda.matmul.allow_tf32 = True`, `torch.backends.cudnn.allow_tf32 = True`) and cuDNN benchmarking (`torch.backends.cudnn.benchmark = True`) in `inference.py` for faster matrix multiplications and optimized kernel selection.

## Additional Tests and Attempts

The following experiments were attempted but not included in the main results table due to issues or incomplete implementation:

1. **Faster Whisper Integration**: Attempted to use Faster Whisper for audio processing to improve speed, but encountered several incompatibilities:
   - **API shape mismatch**: The existing Audio2Feature class calls `model.transcribe` and reads `segment["encoder_embeddings"]`, which comes from OpenAI Whisper's Python implementation. Faster Whisper's `transcribe` returns segments without encoder states—you'd have to call `encode` yourself and manage chunking, padding, and batching manually.
   - **Different I/O types**: Faster Whisper is built on CTranslate2. It expects audio to be preprocessed with its feature extractor, produce StorageView buffers, and tends to emit FP32 outputs even if you requested FP16. That doesn't line up with the raw torch.Tensor pipeline the current code assumes.
   - **Dependency footprint**: The rest of the pipeline relies on the original Whisper checkpoints (.pt) and Torch modules. Faster Whisper ships GGML/CTranslate2 weights, so you'd need to convert models and keep both runtimes in sync.
   - **Deployment differences**: Faster Whisper's sweet spot is CPU or mixed CPU/GPU with quantization. This inference pipeline already leans on CUDA half precision, so the gains can be marginal unless you refit the audio path around its strengths.
   This was abandoned in favor of the standard Hugging Face Whisper implementation.

2. **SageAttention Incorporation**: Tried integrating SageAttention for faster inference, but it failed because the Temporal model uses 88-dimensional attention heads, which are not supported by SageAttention. This limits its applicability to models with standard head dimensions for instant switch and replacement as SageAttention.

3. **Flash Attention 3 Compilation**: Could not compile Flash Attention 3 due to high resource requirements (significant compute and time, potentially days). However, if pre-built binaries become available, it could greatly increase speed compared to Flash Attention 2 in PyTorch, especially on Hopper GPUs.

4. **TensorRT Optimization**: Considered using TensorRT for accelerated inference, but it requires complex model optimization, calibration, and conversion processes, which were not implemented due to the added complexity and time investment.

## Potential Improvements

Future optimizations to consider for enhancing performance and quality:

1. **Torch Profiling**: Use PyTorch profiling tools to identify bottlenecks in the architecture, such as which components (e.g., UNet, audio encoder, SyncNet) consume the most time or memory during inference.

2. **Super-Resolution Upscaling**: Apply RealESRGAN or similar upscaling after generating low-resolution output videos to improve visual quality without increasing inference cost.

3. **Voice Activity Detection (VAD)**: Integrate VAD for audio preprocessing to filter out silent segments, reducing processing time and improving focus on relevant audio features.

4. **Quantization-Aware Training**: Quantize the model during training (e.g., using QAT techniques) to achieve better accuracy at lower precisions, potentially enabling 4-bit inference without quality loss.

## Conclusions

- **Recommended Config**: DPM-Solver with 17 steps + 8-bit quantization at 256 resolution for optimal speed, memory, and quality.
- **Trade-offs**: Use 256 for efficiency and quality; 512 for potentially higher detail but higher cost and lower lip-sync accuracy. At 512, DPM tweek offers speed gains without quality loss. Hence it's scalable with proper setip
- **Hardware**: Tested on RTX 4070 (12GB VRAM); 256 configs fit better.

All runs used guidance_scale=1.7, seed=1247, and the same input video/audio.

## Related Documents

- [Lip-Sync Improvement Proof-of-Concept](lip_sync_improvement_poc.md): Detailed analysis and proposals for enhancing lip-sync quality, including VAD integration and model optimization strategies.
- [AWS Deployment Framework](deployment.md): Complete guide for deploying LatentSync in production on AWS EKS with GPU support, monitoring, and auto-scaling.
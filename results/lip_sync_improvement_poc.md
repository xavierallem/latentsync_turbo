# Proof of Concept: Improving Lip-Sync in LatentSync

This document outlines a structured approach to enhance lip-sync quality by swapping or augmenting the current SyncNet-based audio-visual alignment with more advanced models.

## Candidate Replacements

- **SyncFormer / AV-HuBERT**: Transformer-based aligners that outperform classic SyncNet on LRS3 dataset, offering better temporal alignment.
- **Wav2Lip-HD / TalkLip**: Upgraded variants of Wav2Lip with higher-capacity encoders for more realistic mouth movements.

**Note**: But I doubt it'll be necessary since the lip syncing is fair enough and we have realtime constraints. A good GPU however will greatly help but have to investigate and test further if they will be realtime capable (I think not). Still for realtime many less quality models can be compared.

## Part 2: Quality Enhancement

### Current Model Limitations

Through experimentation with various configs (see `results/README.md`), key limitations identified:
- **Resolution Dependency**: 512x512 outputs have low SyncNet confidence (~2.0) and poor lip-sync, while 256x256 achieves ~9.8, indicating the model struggles with high-res details.
- **Quantization Sensitivity**: 4-bit quantization degrades quality significantly (SyncNet 1.12, AV offset -10), while 8-bit maintains it.
- **Scheduler Variance**: DDIM produces inconsistent results across resolutions; DPM is more stable but still resolution-bound.
- **Audio Processing**: Current Whisper integration lacks voice activity detection, processing silent segments unnecessarily.
- **CUDA Runtime Optimizations**: Added functions to optimize CUDA runtime as much as PyTorch allows, such as efficient tensor operations and memory management in DPM tweek configs.

###  Improvements

1. **Voice Activity Detection (VAD) Integration**: Filter audio to process only speech segments, reducing noise and improving focus on relevant features.
2. **Post-Processing Upscaling**: Apply RealESRGAN or native ffmpeg (bilinear or nearest neighbor interpolation) to 256x256 outputs for higher perceived quality without re-running inference.
3. **Enhanced Audio Encoder**: Swap to AV-HuBERT for better audio-visual alignment, potentially improving SyncNet scores by 20-30% based on literature.

### Implement One Improvement as Proof-of-Concept

**Chosen: Voice Activity Detection (VAD) Integration**

**Implementation**:
- Add `pyannote.audio` or `silero-vad` for VAD.
- Modify `latentsync/whisper/audio2feature.py` to detect speech segments and process only those, and add a layer to process which can help reduce some costs.
- Update `inference.py` to pass filtered audio.



**Reasoning and Expected Impact**:
- **Reasoning**: Silent segments introduce noise in embeddings, reducing alignment accuracy. VAD focuses processing on speech, improving efficiency and quality.
- **Expected Impact**: 10-15% improvement in SyncNet confidence by eliminating irrelevant audio, faster processing (less data), and better lip-sync in videos with pauses.


# Apollo-mod — Project State

## What this is
A fork of JusperLee/Apollo for fine-tuning the Apollo audio restoration model on a single consumer GPU against a static paired LQ/HQ dataset. Upstream targets 8-GPU distributed training with on-the-fly codec simulation; this fork replaces all of that with a self-contained, single-GPU pipeline.

## Core goal
Enable hobbyists/researchers to fine-tune Apollo on their own paired audio data without needing a multi-GPU cluster or HDF5 infrastructure.

## Scope
**In scope:**
- Training pipeline (`train.py`, `paired_datamodule.py`)
- Model modifications (`look2hear/models/apollo.py`)
- Loss/discriminator improvements (`gan_losses.py`, `frequencydis.py`)
- Export tooling (`export_for_uvr.py`)
- Config management (`configs/`)

**Out of scope:**
- Distributed/multi-GPU support (stripped intentionally)
- HDF5 / on-the-fly codec simulation (replaced by static pairs)
- Upstream Apollo paper reproduction

## Key design decisions
- Single-GPU: all DDP, sync_dist, all_gather removed
- Pretrained weights: loaded from local .ckpt/.pth or HuggingFace fallback
- BSNet layers 0–1 frozen; layers 2–3 at 0.3× LR; layers 4–5 + heads at full LR
- TF32 (`high`) precision for throughput
- HF perceptual boost in loss: 3× penalty above 50% Nyquist
- Buffered windows/tensors in loss and model to avoid per-step reallocation
- fp16-safe STFT: cast to float32, back to input dtype

## Checkpoint plan
- [x] Strip DDP / multi-GPU from training system
- [x] Implement PairedAudioDataModule (static WAV pairs)
- [x] Auto-preprocessing pipeline in train.py
- [x] Pretrained weight loading (Lightning + bare state dict)
- [x] Selective layer freezing with per-group LRs
- [x] Validation audio saving
- [x] HF perceptual weighting in freq_MAE
- [x] Buffer Hann windows in model + loss
- [x] fp16-safe STFT/iSTFT
- [x] Inverse window-size weighting in discriminator
- [x] torch.compile support via --compile flag
- [ ] (open) — awaiting user direction

## Current state
New session — project freshly initialized. Codebase appears complete per README. Awaiting user's task.

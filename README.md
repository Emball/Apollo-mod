# Apollo-mod

> A community mod of [JusperLee/Apollo](https://github.com/JusperLee/Apollo) — a GAN-based audio restoration model that recovers high-frequency detail lost to MP3 compression, restoring lossy audio toward lossless quality.

---

## What is Apollo?

Apollo is a research model from Tsinghua University / Tencent AI Lab. It takes MP3-compressed music as input and predicts the original, uncompressed audio. It works by splitting the signal into explicit frequency bands and modeling the relationships between them — preserving low frequencies while reconstructing degraded mid and high frequencies. The model is trained with a GAN framework (generator + frequency discriminator) and evaluated on MUSDB18-HQ and MoisesDB.

The original repo is a research codebase: it assumes multi-GPU clusters, the Weights & Biases logger, MUSDB/MoisesDB dataset pipelines, and HDF5 preprocessing. Running it on a single consumer GPU required significant rework.

---

## What This Mod Does

This fork reworks Apollo for **single-GPU fine-tuning on your own audio pairs** — without cloud infrastructure, without proprietary datasets, and without a CUDA-heavy setup process. It adds a terminal UI, a self-contained data pipeline, and a suite of training stability improvements.

### Goals

- Fine-tune Apollo on **any paired LQ/HQ WAV dataset** you provide (e.g. your own MP3 → FLAC pairs)
- Run on a **single consumer GPU** (tested down to ~11 GB VRAM)
- Make checkpoints portable: load HuggingFace weights, `.pth`/`.bin` serialized models, or Lightning `.ckpt` files interchangeably

---

## Changes from the Original

### New Files

| File | Purpose |
|---|---|
| `paired_datamodule.py` | Drop-in replacement datamodule that loads your own LQ/HQ WAV pairs instead of MUSDB/MoisesDB + HDF5 |
| `install.sh` / `install.bat` | One-command installer using `uv` — creates a `.venv`, installs all pinned dependencies, no conda required |
| `requirements.txt` | Pinned dependency list for reproducible installs |
| `configs/README.txt` | Human-readable reference for every config key, what it does, and when to change it |
| `configs/apollo_uni.yaml` | Config variant for the larger "universal" model (`feature_dim=384`) |

### Modified Files

#### `inference.py` — Complete Rewrite

The original script was ~25 lines and only worked with HuggingFace weights on CUDA. The mod rewrites it to:

- Accept **local `.ckpt` (Lightning), `.pth`/`.bin` (serialized), or HuggingFace** weights via `--weights`
- Handle **any audio format** (WAV, FLAC, MP3, OGG) with automatic resampling to 44,100 Hz
- Process **files of any length** using overlapping 30-second chunks with crossfade stitching — avoiding OOM on long tracks
- Auto-select CPU or GPU (`--device auto`)
- Expose all model hyperparameters (`--feature_dim`, `--layer`, `--sr`, `--win`) as CLI flags so both base and universal models work without code changes

#### `train.py` — Major Extension

The original training script was ~120 lines targeting 8-GPU DDP with WandB. The mod extends it to:

- **Optimization bootstrap** (`apply_optimizations`): centralizes TF32, cuDNN benchmark, CUDA allocator expandable segments, and Triton kernel caching — applied once at startup from `cfg.optimizations`
- **Integrated data preparation**: on first run, automatically chunks your `data/train/` and `data/val/` folders into fixed-length overlapping WAV pairs saved to `chunks/`. Skips gracefully if chunks already exist. Supports cached augmentation variants written at chunk time
- **`adamw_8bit` optimizer**: uses bitsandbytes 8-bit AdamW for the generator, cutting optimizer state memory by ~50% with negligible quality impact
- **Gradient accumulation**: `grad_accum_steps` simulates larger batch sizes without extra VRAM
- **Separate LR schedule**: generator (`lr_g`) and discriminator (`lr_d`) have independent learning rates; defaults are more conservative than the original (`1e-5` / `1e-6`) to avoid catastrophic forgetting when fine-tuning from pretrained weights
- Switched logger from **WandB → TensorBoard** (no account or API key needed)
- Switched training from **8-GPU DDP → single GPU** (`devices: [0]`, removed `sync_batchnorm`)

#### `configs/apollo.yaml` — Restructured

| Area | Original | Mod |
|---|---|---|
| Dataset | `MusdbMoisesdbDataModule` + HDF5 | `PairedAudioDataModule` + raw WAV pairs |
| Optimizer | Two separate `AdamW` blocks | Single `optimizer` block with `adamw_8bit` and per-group LRs |
| Logger | WandB | TensorBoard |
| Devices | `[0,1,2,3,4,5,6,7]` | `[0]` |
| Precision | fp32 | `16-mixed` |
| Checkpoint filename | `{epoch}-{val_loss:.4f}` | `step={step:06d}-val_loss={val_loss:.4f}` |
| Checkpoint saving | Top-5 + last | All (save_top_k: -1) for full training history |
| `patience` | 20 | 2000 (effectively disabled early stopping for long fine-tunes) |
| New: `optimizations` block | — | TF32, cuDNN benchmark, expandable segments, Triton cache |
| New: `training` block | — | Layer freezing, HF boost, val audio saving, grad accumulation |
| New: `augmentation` block | — | Live/cached augmentation pipeline (gain, polarity, pitch shift, noise, MP3 degradation) |

#### `look2hear/models/apollo.py` — STFT Stability Fixes

- Registers the Hann window as a **persistent buffer** (`register_buffer`) instead of creating a new tensor on every forward pass — avoids redundant allocation and ensures the window is always on the right device
- Casts input to **float32 before STFT/iSTFT** with a cast-back afterward, because cuFFT does not support half precision for non-power-of-two FFT sizes (would crash in `16-mixed` training)
- Makes `.real` and `.imag` views **contiguous** before `torch.cat` — prevents stride assertion errors in `torch.compile` / inductor

#### `look2hear/system/audio_litmodule.py` — Single-GPU + Gradient Checkpointing

- Removed `sync_dist=True` from all `self.log()` calls (causes hangs without DDP)
- Removed `all_gather` in validation (multi-GPU only)
- Removed WandB-specific logger calls
- Added **gradient checkpointing**: wraps `BSNet` layers and `FrequencyDiscriminator` with `torch.utils.checkpoint`, recomputing activations during backward instead of storing them — trades ~30% compute for large VRAM savings
- Added **validation audio saving**: every `val_save_interval` epochs, a fixed set of validation chunks is run through the model and saved to disk so you can track improvement by ear alongside the loss curve

#### `look2hear/losses/gan_losses.py` and `look2hear/discriminators/frequencydis.py`

Minor compatibility fixes for mixed-precision training and single-GPU execution.

---

## Workflows

### 1. Installation

```bash
git clone https://github.com/Emball/Apollo-mod.git
cd Apollo-mod

# Linux / macOS
chmod +x install.sh && ./install.sh

# Windows
install.bat
```

The installer uses `uv` (installed automatically if absent) to create a `.venv` with Python 3.11 and all pinned dependencies.

---

### 2. Prepare Your Data

Organize your audio pairs:

```
data/
  train/
    lq/   ← MP3-degraded or compressed audio (WAV format)
    hq/   ← Original lossless audio (WAV format, same filenames)
  val/
    lq/
    hq/
```

Filenames must match between `lq/` and `hq/`. Files must be WAV. On the first training run, the trainer will automatically chunk these into fixed-length overlapping segments and save them to `chunks/train/` and `chunks/val/`. Subsequent runs skip this step.

---

### 3. Fine-Tune

```bash
# Activate the venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate.bat       # Windows

python train.py --conf_dir=configs/apollo.yaml
```

TensorBoard logs are written to the `exp.dir` folder. Launch with:

```bash
tensorboard --logdir ./experiments
```

---

### 4. Run Inference

```bash
# From HuggingFace (no local weights needed)
python inference.py --in_wav input.mp3 --out_wav output.wav

# From a Lightning checkpoint saved during fine-tuning
python inference.py \
  --in_wav  input.mp3 \
  --out_wav output.wav \
  --weights experiments/my_run/step=001200-val_loss=0.0312.ckpt

# Universal model (feature_dim=384)
python inference.py \
  --in_wav  input.mp3 \
  --out_wav output.wav \
  --weights models/apollo_model_uni.ckpt \
  --feature_dim 384

# CPU-only machine
python inference.py --in_wav input.mp3 --out_wav output.wav --device cpu
```

Long files are automatically processed in 30-second overlapping chunks with crossfade stitching.

---

## VRAM Requirements (approximate)

| Config | VRAM |
|---|---|
| Base model (`feature_dim=256`), `batch_size=2`, `segment_sec=4`, gradient checkpointing on | ~8–10 GB |
| Base model, `batch_size=1`, `segment_sec=3`, gradient checkpointing on | ~6–8 GB |
| Universal model (`feature_dim=384`), `batch_size=1` | ~10–12 GB |

Enable `gradient_checkpointing: true` and `precision: 16-mixed` in the config to reduce VRAM at the cost of ~30% longer training steps.

---

## Upstream

This mod is based on [JusperLee/Apollo](https://github.com/JusperLee/Apollo) by Kai Li and Yi Luo (Tsinghua University / Tencent AI Lab), published at ICASSP 2025.

```bibtex
@inproceedings{li2025apollo,
  title={Apollo: Band-sequence Modeling for High-Quality Music Restoration in Compressed Audio},
  author={Li, Kai and Luo, Yi},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2025},
  organization={IEEE}
}
```

Licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

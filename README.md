# Apollo-mod

> A community mod of [JusperLee/Apollo](https://github.com/JusperLee/Apollo) — a GAN-based audio restoration model. Codec compression (MP3, AAC, etc.) is the core target, but the architecture generalises to broader restoration tasks: bandwidth extension, noise reduction, and other forms of audio degradation.

---

## What is Apollo?

Apollo is a research model from Tsinghua University / Tencent AI Lab (ICASSP 2025). It takes degraded audio as input and predicts the clean original. It works by splitting the signal into explicit frequency bands and modeling relationships between them — preserving low frequencies while reconstructing degraded mid and high frequencies. The model uses a GAN framework (generator + frequency discriminator) and was originally evaluated on MUSDB18-HQ and MoisesDB with codec compression as the primary degradation type.

The architecture is not codec-specific. The same frequency-band approach applies to any task where degradation is concentrated in mid/high frequencies — codec compression (MP3, AAC, Opus) is the core target, but bandwidth extension, noise reduction, and other restoration tasks are reasonable fine-tuning targets given appropriate paired training data.

The original repo is a research codebase: it assumes multi-GPU clusters, the Weights & Biases logger, MUSDB/MoisesDB dataset pipelines, and HDF5 preprocessing. Running it on a single consumer GPU required significant rework.

---

## What This Mod Does

This fork reworks Apollo for **single-GPU fine-tuning on your own audio pairs** — without cloud infrastructure, without proprietary datasets, and without a CUDA-heavy setup process. It adds a terminal UI, a self-contained data pipeline, and a suite of training stability improvements.

### Goals

- Fine-tune Apollo on **any paired LQ/HQ WAV dataset** you provide — codec-degraded pairs are the primary use case but any degradation type works
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
| `configs/apollo_uni.yaml` | Config variant for the larger "universal" model (`feature_dim=384`) |
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
    lq/   ← Degraded audio (WAV format) — codec-compressed, noisy, bandwidth-limited, etc.
    hq/   ← Clean reference audio (WAV format, same filenames)
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
# Default model (apollo — downloads pytorch_model.bin on first run)
python inference.py --in_wav input.mp3 --out_wav output.wav

# Lew vocal enhancer v2 (auto-downloads on first run)
python inference.py --in_wav input.mp3 --out_wav output.wav --weights lew_v2

# Lew universal model (feature_dim set automatically)
python inference.py --in_wav input.mp3 --out_wav output.wav --weights lew_uni

# Local fine-tune checkpoint
python inference.py \
  --in_wav  input.mp3 \
  --out_wav output.wav \
  --weights experiments/my_run/step=001200-val_loss=0.0312.ckpt

# CPU-only machine
python inference.py --in_wav input.mp3 --out_wav output.wav --device cpu
```

Available shortnames (all auto-download into `models/` and are reused on subsequent runs):

| Shortname | File | `feature_dim` |
|---|---|---|
| `apollo` | `pytorch_model.bin` — official base model | 256 |
| `lew` | `apollo_model.ckpt` — Lew vocal enhancer v1 | 256 |
| `lew_v2` | `apollo_model_v2.ckpt` — Lew vocal enhancer v2 | 256 |
| `lew_uni` | `apollo_model_uni.ckpt` — Lew universal model | 384 |

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

## Config Reference

Two configs are provided: `configs/apollo.yaml` (base model, `feature_dim=256`, recommended for most fine-tuning) and `configs/apollo_uni.yaml` (universal model, `feature_dim=384`, larger, needs more VRAM).

### `exp`

| Key | Description |
|---|---|
| `dir` | Root folder for all experiment outputs (checkpoints, logs, val audio). |
| `name` | Subfolder name for this run. Change this to avoid overwriting a previous run. |

### `optimizations`

| Key | Description |
|---|---|
| `tf32` | Enable TF32 matmuls on Ampere+ GPUs (3000/4000 series). Negligible quality loss, meaningful speed boost. Leave `true` unless you need full float32 precision. |
| `cudnn_benchmark` | Let cuDNN benchmark conv algorithms on the first batch and pick the fastest one. Leave `true` for fixed input shapes. Turn off if input sizes vary wildly between runs. |
| `expandable_segments` | Reduces CUDA allocator fragmentation, helps avoid OOMs mid-run. Leave `true`. |
| `triton_cache` | Cache compiled Triton kernels to disk so they don't recompile every run. Cache lives in `.triton_cache/`. Leave `true`. |

### `training`

| Key | Description |
|---|---|
| `n_layers_to_freeze` | Number of BSNet layers to freeze (BN front-end is always frozen). Higher = less VRAM, faster steps, less catastrophic forgetting, less adaptability. Recommended: 4 for base on 11 GB VRAM, 3 for universal. |
| `hf_boost` | Extra loss weight on high frequencies. `1.0` = flat. ~`1.5` pushes the model toward treble detail. Don't go above `2.0` or it will over-sharpen. |
| `val_save_interval` | Save rendered validation audio to disk every N epochs. The same fixed set of chunks is used every time so you can track improvement by ear. |
| `val_audio_pairs` | How many val chunks to render per save interval. Picked on the first val run and locked in. |
| `grad_accum_steps` | Accumulate gradients over N batches before stepping. Simulates a larger effective batch size without extra VRAM. `grad_accum_steps: 2` + `batch_size: 2` = effective batch of 4. |

### `datas`

| Key | Description |
|---|---|
| `train_dir` | Where chunked training pairs are stored. Auto-generated from `data/train/` on first run. Delete to force re-chunking (e.g. after changing `segment_sec`). |
| `eval_dir` | Same as above for validation data. |
| `sr` | Sample rate. Apollo expects `44100`. Don't change this. |
| `segment_sec` | Length of audio chunks in seconds. Longer = more context, more VRAM. Changing this requires deleting `chunks/` and re-chunking. |
| `batch_size` | Chunks per training step. With `segment_sec: 4` on an 11 GB card, `batch_size: 2` is about the limit. |
| `num_workers` | CPU workers prefetching data. Set to number of free CPU cores, max ~8. |
| `pin_memory` | Allocate DataLoader batches in page-locked RAM for faster CPU→GPU transfers. Leave `true` unless RAM-constrained. |
| `val_bootstrap_chunks` | If no `data/val/` folder exists, this many chunks are copied from the training set for validation, picked round-robin across songs. |

**Augmentation** — two categories: `live` (applied each epoch by the dataloader, cheap ops) and `cached` (baked into chunk files at chunking time, amortizes expensive ops like pitch shift). All augmentations are applied identically to both LQ and HQ so the model never sees a mismatch.

`cached.cached_variants` — number of augmented variants written per chunk in addition to the clean original. e.g. `variants: 2` produces 3 files per chunk: original + 2 augmented copies.

Live uses `prob` (per-sample probability); cached uses `fraction` (fraction of chunks that receive the augmentation across the dataset).

| Augmentation | Description |
|---|---|
| `mono_channel` | Alternates L/R by sample index (even=L, odd=R) instead of feeding both near-identical stereo channels as unique data. Guarantees a clean 50/50 split every epoch regardless of shuffle order. |
| `pitch_shift` | Transparent pitch shift. `semitones_max` sets the max shift in either direction. |
| `noise` | Matched Gaussian noise added to both LQ and HQ. `sigma: 0.002` is very subtle. |
| `mp3_degradation` | Random CBR MP3 re-encode on LQ only. `kbps_min`/`kbps_max` set the bitrate range. |
| `gain` | Random gain shift ±`db_max` dB. Output clamped to `[-1, 1]`. |
| `polarity` | Polarity flip (multiply by −1). Essentially free. |

> **Note:** Chunks are always saved as 16-bit PCM WAV. Only WAV files are accepted in `data/` — MP3/FLAC/etc will be rejected.

### `model`

| Key | Description |
|---|---|
| `feature_dim` | Model width. `256` = base, `384` = universal. Must match pretrained weights. |
| `layer` | Number of BSNet layers. Always `6` for pretrained Apollo weights. |
| `win` | STFT window size in ms. Always `20` for pretrained Apollo weights. |

### `optimizer`

| Key | Description |
|---|---|
| `type` | `adamw` (standard 32-bit, no extra deps), `adamw_8bit` (8-bit via bitsandbytes, cuts optimizer VRAM ~75%, requires `bitsandbytes`), or `cpu_offload` (momentum states in CPU RAM, saves ~200–400 MB VRAM). |
| `lr_g` | Learning rate for the generator. |
| `lr_d` | Learning rate for the discriminator. |
| `weight_decay` | Applied to both optimizers. |
| `betas_g` | Adam β1, β2 for the generator. |
| `betas_d` | Adam β1, β2 for the discriminator. Lower β1 (0.5) is standard for GAN discriminators. |

### `scheduler_g` / `scheduler_d`

| Key | Description |
|---|---|
| `step_size` | Decay the LR every N epochs. |
| `gamma` | Multiply LR by this value each step. `0.98` = 2% reduction per step. |

### `system`

| Key | Description |
|---|---|
| `gradient_checkpointing` | Recompute activations during backward instead of storing them. Saves 30–40% VRAM at the cost of ~30% more compute. Useful for pushing `segment_sec` or `batch_size` higher. |

### `early_stopping`

| Key | Description |
|---|---|
| `patience` | Stop if `val_loss` doesn't improve for this many val checks. Set very high in the default configs to effectively disable it — lower it if you want it to trigger. |

### `checkpoint`

| Key | Description |
|---|---|
| `save_top_k` | How many checkpoints to keep. `-1` = keep all. Set to e.g. `5` to keep only the best 5 by `val_loss`. |

### `trainer`

| Key | Description |
|---|---|
| `devices` | Which GPU index to use. `[0]` = first GPU. |
| `max_epochs` | Hard cap on training epochs. |
| `precision` | `16-mixed` = fp16 mixed precision (recommended). Use `32` for debugging. |
| `fast_dev_run` | Set `true` to run 1 train + 1 val batch then exit. Good for sanity-checking a new setup. |
| `val_check_interval` | (`apollo_uni` only) Run validation every N steps instead of every epoch. Useful when epochs are very long. |

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

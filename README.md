# Apollo-mod

> A community mod of [JusperLee/Apollo](https://github.com/JusperLee/Apollo) — a GAN-based audio restoration model. Codec compression is the core target, but the architecture generalises to broader restoration tasks: bandwidth extension, noise reduction, clipping restoration, and other audio degradation types.

---

## What is Apollo?

Apollo is a research model from Tsinghua University / Tencent AI Lab (ICASSP 2025). It takes degraded audio as input and predicts the clean original by splitting the signal into explicit frequency bands and modeling relationships between them — preserving low frequencies while reconstructing degraded mid and high frequencies. The model uses a GAN framework (generator + frequency discriminator) and was originally trained on MUSDB18-HQ and MoisesDB with MP3 compression at 24–128 kbps as the primary degradation type.

The original repo is a research codebase: it assumes multi-GPU clusters, Weights & Biases, MUSDB/MoisesDB dataset pipelines, and HDF5 preprocessing. Running it on a single consumer GPU required significant rework.

---

## What This Mod Does

This fork reworks Apollo for **single-GPU fine-tuning on your own audio pairs** — no cloud infrastructure, no proprietary datasets, no complex setup. Drop in your LQ/HQ pairs, run `apollo.bat`, and train.

### Goals

- Fine-tune Apollo on **any paired LQ/HQ dataset** — WAV, MP3, or FLAC input accepted
- Run on a **single consumer GPU** (tested on RTX 2080 Ti, 11 GB VRAM)
- Load weights from HuggingFace, `.pth`/`.bin` serialized models, or Lightning `.ckpt` files interchangeably
- Automatically handle encoder delay alignment for MP3 training pairs

---

## Installation

```bash
git clone https://github.com/Emball/Apollo-mod.git
cd Apollo-mod

# Windows
apollo.bat

# Linux / macOS
chmod +x apollo.sh && ./apollo.sh
```

On first run this creates a `.venv` with Python 3.11 and installs all dependencies. On subsequent runs it drops you into an activated shell where `train.py` and `inference.py` are ready to use.

> **Windows note:** If `apollo.bat` fails with a uv error, your real `uv.exe` may be shadowed by a stub at `C:\Windows\System32\uv`. The bat will find the real one at `%USERPROFILE%\.local\bin\uv.exe` automatically on recent versions.

---

## Prepare Your Data

Data lives under `data/<config_name>/` — the folder name is derived automatically from your config filename (e.g. `configs/apollo_stfl.yaml` → `data/apollo_stfl/`).

```
data/apollo_stfl/
  train/
    LQ/   ← degraded audio (WAV, MP3, or FLAC — filenames must match HQ)
    HQ/   ← clean reference audio
  val/
    LQ/
    HQ/
```

On first run, `train.py` automatically chunks these into fixed-length overlapping segments saved to `chunks/<config_name>/`. Delete `chunks/` to force re-chunking (required after changing `segment_sec` or augmentation settings).

If you have no `val/` folder, the trainer will bootstrap one by copying chunks from training data across all songs.

### MP3 Input and Alignment

MP3, FLAC, and WAV are all accepted directly. If your LQ files are MP3s encoded from the HQ WAVs, set `align_data` in the config to trim the encoder delay:

```yaml
datas:
  align_data: 1057   # iTunes encoder delay — exact sample count, applied at decode time
```

`align_data: true` attempts auto-detection via the LAME header with xcorr fallback. `align_data: false` disables alignment. An integer value applies a fixed trim: positive trims LQ, negative trims HQ.

---

## Training

```bash
python train.py --conf_dir configs/apollo_uni.yaml
```

TensorBoard logs are written to `runs/<name>/logs/`. Monitor with:

```bash
tensorboard --logdir ./runs
```

Validation audio is saved to `runs/<name>/val_audio/` at intervals so you can track improvement by ear alongside the loss curve.

---

## Inference

```bash
# Named model (auto-downloads on first run)
python inference.py --in_wav input.mp3 --out_wav output.wav --weights lew_v2

# Local fine-tune with config (reads feature_dim, sr, chunk size from yaml)
python inference.py --in_wav input.mp3 --out_wav output.wav \
    --weights runs/apollo_stfl/checkpoints/step=001200-val_loss=0.0312.ckpt \
    --conf_dir configs/apollo_stfl.yaml

# Manual feature_dim override
python inference.py --in_wav input.mp3 --out_wav output.wav \
    --weights models/my_model.ckpt --feature_dim 384
```

Output is written chunk-by-chunk to disk as inference runs — you can drag the output file into Audacity immediately and preview completed chunks while the rest processes.

Available named models (all auto-download to `models/`):

| Shortname | File | `feature_dim` |
|---|---|---|
| `apollo` | `pytorch_model.bin` — official base model | 256 |
| `lew` | `apollo_model.ckpt` — Lew vocal enhancer v1 | 256 |
| `lew_v2` | `apollo_model_v2.ckpt` — Lew vocal enhancer v2 | 256 |
| `lew_uni` | `apollo_model_uni.ckpt` — Lew universal model | 384 |

---

## VRAM Requirements (approximate)

| Config | VRAM |
|---|---|
| Base (`feature_dim=256`), `batch_size=2`, `segment_sec=4`, gradient checkpointing on | ~8–10 GB |
| Universal (`feature_dim=384`), `batch_size=2`, `segment_sec=4`, gradient checkpointing on | ~10–11 GB |

Always use `gradient_checkpointing: true` and `precision: 16-mixed` on consumer cards.

---

## Config Reference

Two base configs: `configs/apollo.yaml` (`feature_dim=256`) and `configs/apollo_uni.yaml` (`feature_dim=384`). Copy and rename for each fine-tune — the name drives the data and run folders automatically.

### `exp`

| Key | Description |
|---|---|
| `dir` | Root folder for run outputs. Default `./runs`. |
| `name` | Subfolder for this run. Derived from config filename if null — `apollo_stfl.yaml` → `apollo_stfl`. |

### `optimizations`

| Key | Description |
|---|---|
| `tf32` | TF32 matmuls. Only benefits Ampere+ GPUs (RTX 3000/4000). Harmless on older cards. |
| `cudnn_benchmark` | Benchmarks cuDNN conv algorithms on first batch and picks the fastest. Leave `true` for fixed input shapes. |
| `expandable_segments` | Reduces CUDA allocator fragmentation. Leave `true`. |
| `triton_cache` | Caches compiled Triton kernels to `.triton_cache/` — saves 30–60s on startup after first run. |
| `ram_limit_fraction` | Fraction of total system RAM at which the training process kills itself cleanly before the OS crashes. Default `0.90`. Lower to `0.85` on 16 GB systems. |

### `training`

| Key | Description |
|---|---|
| `n_layers_to_freeze` | Freeze the first N BSNet layers and the BN front-end. Apollo has 6 layers total. `4` is recommended for fine-tuning — only the back half and output heads train, which is where codec-specific adaptation matters. |
| `hf_boost` | Extra loss weight on high frequencies. `1.0` = flat. `1.5` pushes the model toward treble detail. Don't exceed `2.0`. |
| `val_save_interval` | Save rendered validation audio every N validation runs. |
| `val_audio_pairs` | How many LQ/HQ/enhanced triplets to save per interval. |
| `grad_accum_steps` | Accumulate gradients over N batches before stepping. Simulates a larger batch without extra VRAM. |

### `datas`

| Key | Description |
|---|---|
| `sr` | Sample rate. Fixed at `44100`. Do not change. |
| `segment_sec` | Chunk length in seconds. Changing requires deleting `chunks/`. |
| `batch_size` | Chunks per training step. `2` is the limit on 11 GB VRAM with the universal model. |
| `num_workers` | DataLoader worker processes. On 16 GB RAM, use `4`. More workers = more RAM consumed simultaneously. |
| `pin_memory` | Page-locks DataLoader buffers for faster CPU→GPU transfers. Set `false` on 16 GB systems — pinned memory can't be swapped and causes OS crashes under pressure. |
| `val_bootstrap_chunks` | Chunks copied from training set if no val data exists. |
| `align_data` | `true` = auto-detect MP3 encoder delay, `false` = off, integer = fixed sample trim (positive trims LQ, negative trims HQ). |

**Augmentation** — `live` ops run each epoch in the DataLoader workers (cheap). `cached` ops are baked into chunk files at chunking time (amortizes expensive work). All augmentations apply identically to both LQ and HQ.

| Augmentation | Live/Cached | Description |
|---|---|---|
| `gain` | Live | Random ±`db_max` dB shift. Never hard-clamps — scales both signals down together if clipping would occur, preserving any existing flat-top distortion as valid training signal. |
| `polarity` | Live | Multiply by −1. Essentially free. |
| `noise` | Live | Matched Gaussian noise on both LQ and HQ. Same soft-clamp behavior as gain. |
| `mono_channel` | Live | Feeds one channel at a time (L or R), alternating by sample index. Avoids the model seeing near-identical stereo pairs as unique data. |
| `pitch_shift` | Cached | Transparent pitch shift ±`semitones_max`. **Turn off for codec restoration** — warps frequency relationships the model is trying to learn. |
| `mp3_degradation` | Cached | Random CBR MP3 re-encode on LQ only. Move to cached (not live) to avoid spawning an ffmpeg process per chunk per epoch. |

`cached.cached_variants` — number of augmented files written per chunk in addition to the clean original.

### `model`

| Key | Description |
|---|---|
| `feature_dim` | Model width. `256` = base, `384` = universal. Must match pretrained weights. |
| `layer` | BSNet layer count. Always `6` for pretrained Apollo weights. |
| `win` | STFT window in ms. Always `20` for pretrained Apollo weights. |

### `optimizer`

| Key | Description |
|---|---|
| `type` | `adamw` (32-bit), `adamw_8bit` (8-bit via bitsandbytes, cuts optimizer VRAM ~75%), or `cpu_offload` (momentum in CPU RAM). |
| `lr_g` | Generator learning rate. `1e-5` is conservative for fine-tuning. |
| `lr_d` | Discriminator learning rate. Keep 10× lower than `lr_g`. |
| `weight_decay` | L2 regularization, applied to both. |
| `betas_g` / `betas_d` | Adam momentum terms. Lower β1 for discriminator (0.5) is standard GAN practice. |

### `scheduler_g` / `scheduler_d`

StepLR: multiply LR by `gamma` every `step_size` epochs. `gamma: 0.98` = 2% reduction per step. Very gentle.

### `system`

| Key | Description |
|---|---|
| `gradient_checkpointing` | Recompute activations during backward instead of storing them. Saves 30–40% VRAM at ~30% compute cost. Keep `true` on consumer cards. |

### `early_stopping`

| Key | Description |
|---|---|
| `patience` | Stop if `val_loss` doesn't improve for this many validation checks. Default is very high (effectively disabled). Set to `20–30` for active early stopping. |

### `checkpoint`

| Key | Description |
|---|---|
| `save_top_k` | `-1` = keep all checkpoints. Set to e.g. `5` to keep only the best 5 by `val_loss`. |

### `trainer`

| Key | Description |
|---|---|
| `devices` | GPU index. `[0]` = first GPU. |
| `max_epochs` | Hard epoch cap. Early stopping will usually trigger first. |
| `precision` | `16-mixed` = fp16 mixed precision. Use `32` for debugging only. |
| `val_check_interval` | Validate every N steps. With ~1300 steps/epoch on a 20-song dataset, `500` = validating roughly twice per epoch. |
| `fast_dev_run` | `true` = run 1 train + 1 val batch then exit. Use to sanity-check a new config without waiting. |

---

## Upstream

Based on [JusperLee/Apollo](https://github.com/JusperLee/Apollo) by Kai Li and Yi Luo (Tsinghua University / Tencent AI Lab), ICASSP 2025.

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

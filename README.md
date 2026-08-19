# Apollo-mod

> A community mod of [JusperLee/Apollo](https://github.com/JusperLee/Apollo) — a GAN-based audio restoration model. Codec compression is the core target, but the architecture generalises to broader restoration tasks: bandwidth extension, noise reduction, clipping restoration, and other audio degradation types.

---

## What is Apollo?

Apollo is a research model from Tsinghua University / Tencent AI Lab (ICASSP 2025). It takes degraded audio as input and predicts the clean original by splitting the signal into explicit frequency bands and modeling relationships between them. The original model was trained on MUSDB18-HQ and MoisesDB with MP3 compression at 24–128 kbps as the primary degradation type.

The original repo assumes multi-GPU clusters, Weights & Biases, and proprietary dataset pipelines. This fork reworks it for single-GPU fine-tuning on your own audio pairs.

---

## What This Mod Does

- Fine-tune Apollo on **any paired LQ/HQ dataset** — WAV, MP3, or FLAC input accepted
- Run on a **single consumer GPU** (tested on RTX 2080 Ti, 11 GB VRAM)
- Load weights from HuggingFace, `.pth`/`.bin` serialized models, or Lightning `.ckpt` files
- Automatically handle encoder delay alignment for MP3 training pairs
- Isolated timestamped run folders with automatic resume

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

On first run this creates a `.venv` with Python 3.11 and installs all dependencies. Subsequent runs activate the env directly.

> **Windows note:** If `apollo.bat` fails with a uv error, your real `uv.exe` may be shadowed by a stub at `C:\Windows\System32\uv`. The bat will find the real one at `%USERPROFILE%\.local\bin\uv.exe` automatically on recent versions.

---

## Prepare Your Data

Data lives under `data/<config_name>/` — the folder name is derived from your config filename automatically (e.g. `configs/apollo_stfl.yaml` → `data/apollo_stfl/`).

```
data/apollo_stfl/
  train/
    LQ/   ← degraded audio (WAV, MP3, or FLAC — filenames must match HQ)
    HQ/   ← clean reference audio
  val/
    LQ/
    HQ/
```

On first run, `train.py` auto-chunks these into fixed-length segments saved to `chunks/<config_name>/`. Delete `chunks/` to force re-chunking (required after changing `segment_sec`).

### MP3 Input and Alignment

MP3, FLAC, and WAV are all accepted. If your LQ files are MP3s encoded from the HQ WAVs, set `align_data` in your config to trim the encoder delay at decode time — no overhead, no separate preprocessing step:

```yaml
datas:
  align_data: 1057   # iTunes encoder delay; exact sample count
```

`align_data: true` attempts auto-detection via the LAME header with xcorr fallback. `false` disables it. A positive integer trims LQ; negative trims HQ.

---

## Training

```bash
python train.py --conf_dir configs/apollo_uni.yaml
```

Each run creates a timestamped folder under `runs/<name>/`. To resume from where you left off, set `resume: true` in your config — train.py finds the most recent checkpoint automatically. Ctrl+C saves a checkpoint before exiting and resumes from it cleanly next time.

Validation audio (LQ + HQ + restored triplets) is saved to `runs/<name>/<timestamp>/val_audio/` every val run so you can track improvement by ear. Monitor loss with:

```bash
tensorboard --logdir ./runs
```

---

## Inference

```bash
# Local fine-tune with config (reads feature_dim, chunk size, etc. from yaml)
python inference.py --in_wav input.mp3 --out_wav output.wav \
    --weights runs/apollo_stfl/20260819_143022/checkpoints/001200-val_loss=-24.41.ckpt \
    --conf_dir configs/apollo_stfl.yaml

# Manual feature_dim override
python inference.py --in_wav input.mp3 --out_wav output.wav \
    --weights models/apollo_model_uni.ckpt --feature_dim 384
```

Output is written chunk-by-chunk to disk as inference runs — drag the output file into Audacity immediately to preview completed sections while the rest processes. Output is 32-bit float WAV.

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
| `dir` | Root for run outputs. Default `./runs`. |
| `name` | Subfolder for this run. Derived from config filename if null. |

### `optimizations`

| Key | Description |
|---|---|
| `tf32` | TF32 matmuls. Only benefits Ampere+ GPUs (RTX 3000/4000). Harmless on older cards. |
| `cudnn_benchmark` | Benchmarks cuDNN conv algorithms on first batch. Leave `true` for fixed input shapes. |
| `expandable_segments` | Reduces CUDA allocator fragmentation. Leave `true`. |
| `triton_cache` | Caches compiled Triton kernels — saves 30–60s on startup after first run. |
| `ram_limit_fraction` | Fraction of system RAM at which the process kills itself cleanly. Default `0.90`. On 16 GB systems where idle RAM is already high (e.g. 15+ GB), set to `0.98` or `1.0`. |

### `training`

| Key | Description |
|---|---|
| `n_layers_to_freeze` | Freeze the first N BSNet layers. Apollo has 6 total. `4` is recommended for fine-tuning. |
| `hf_boost` | Extra loss weight on high frequencies. `1.0` = flat. Don't exceed `2.0`. |
| `val_audio_pairs` | Val audio samples saved **per song** each val run. |
| `grad_accum_steps` | Accumulate gradients over N steps. Simulates a larger batch without extra VRAM. |

### `datas`

| Key | Description |
|---|---|
| `sr` | Sample rate. Fixed at `44100`. Do not change. |
| `segment_sec` | Chunk length in seconds. Changing requires deleting `chunks/`. |
| `batch_size` | Chunks per step. `2` is the practical limit on 11 GB VRAM with the universal model. |
| `num_workers` | DataLoader workers. On 16 GB RAM, use `4`. More workers = more simultaneous RAM consumption. |
| `pin_memory` | Page-locks DataLoader buffers. Set `false` on 16 GB — pinned memory can't swap and causes OS crashes under pressure. |
| `val_bootstrap_chunks` | Chunks copied from training set if no val data exists. |
| `align_data` | `true` = auto-detect delay, `false` = off, integer = fixed sample trim. |

**Augmentation** — `live` ops run each epoch in DataLoader workers. `cached` ops are baked into chunk files at prep time.

| Augmentation | Live/Cached | Notes |
|---|---|---|
| `gain` | Live | Never hard-clamps — scales both signals down if clipping would occur. |
| `polarity` | Live | Multiply by −1. Free. |
| `noise` | Live | Matched noise on both LQ and HQ. Same soft-clamp behavior as gain. |
| `mono_channel` | Live | One channel per sample, alternating L/R by index. |
| `pitch_shift` | Cached | **Off for codec restoration** — warps the frequency relationships the model is trying to learn. |
| `mp3_degradation` | Cached | CBR MP3 re-encode on LQ only. Keep cached, not live — live spawns ffmpeg per chunk per epoch. |

### `resume`

| Key | Description |
|---|---|
| `resume` | `true` = find most recent checkpoint and continue. `false` = new timestamped run. |

### `model`

| Key | Description |
|---|---|
| `feature_dim` | `256` = base, `384` = universal. Must match pretrained weights. |
| `layer` | Always `6` for pretrained Apollo weights. |
| `win` | Always `20` for pretrained Apollo weights. |

### `optimizer`

| Key | Description |
|---|---|
| `type` | `adamw` (32-bit), `adamw_8bit` (8-bit via bitsandbytes, cuts optimizer VRAM ~75%), or `cpu_offload`. |
| `lr_g` | Generator learning rate. `1e-5` is conservative for fine-tuning. |
| `lr_d` | Discriminator learning rate. Keep 10× lower than `lr_g`. |

### `system`

| Key | Description |
|---|---|
| `gradient_checkpointing` | Recomputes activations during backward. Saves 30–40% VRAM at ~30% compute cost. Keep `true` on consumer cards. |

### `early_stopping`

| Key | Description |
|---|---|
| `patience` | Stop if `val_loss` doesn't improve for this many validation checks. Set `20–30` for active early stopping. |

### `checkpoint`

| Key | Description |
|---|---|
| `save_top_k` | `-1` = keep all. Set to e.g. `5` to keep only the best 5 by `val_loss`. |

### `trainer`

| Key | Description |
|---|---|
| `val_check_interval` | Validate every N steps. Also controls val audio save frequency. |
| `limit_val_batches` | Cap val batches per run. `50` keeps overhead to ~90 seconds without sacrificing loss accuracy. |
| `max_epochs` | Hard epoch cap. Early stopping usually triggers first. |
| `precision` | `16-mixed` = fp16 mixed precision. Required on consumer cards. |

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

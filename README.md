# Apollo-mod

A custom fine-tuning fork of [JusperLee/Apollo](https://github.com/JusperLee/Apollo), a GAN-based audio restoration model targeting codec-compressed audio.

---

## What is Apollo?

Apollo is a research model from Tsinghua University / Tencent AI Lab (ICASSP 2025). It restores degraded audio by splitting the signal into frequency bands and modeling relationships between them. The original was trained on MUSDB18-HQ and MoisesDB with MP3 compression as the primary degradation type, but it has proven to be broadly useful for a variety of different restoration tasks.

This fork reworks it for single-GPU fine-tuning on your own paired audio datasets. The aim is to help the community explore the true potential of the architecture, unlocking the ability for custom finetunes on consumer hardware.

---

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Emball/Apollo-mod.git
cd Apollo-mod
```

2. Run the setup script:
```bash
# Windows
apollo.bat

# Linux / macOS
chmod +x apollo.sh && ./apollo.sh
```

On first run this creates a `.venv` with all dependencies. Subsequent runs open the TUI directly.

---

## Prepare Your Data

Place your audio files under `data/<apollo_name>/`, where the folder name matches your config file (e.g. `configs/apollo_name.yaml`).

```
data/apollo_stfl/
  train/
    LQ/    -> degraded audio (filenames must match HQ)
    HQ/    -> clean reference audio
  val/
    LQ/
    HQ/
```

Your validation data should ideally be at least half the size of your train set, with matching degradation to the train set for the most accurate training progress calculations.

On first run, `train.py` automatically converts any non-WAV sources to 32-bit float WAV in-place via FFmpeg, then chunks them into fixed-length segments under `chunks/<apollo_name>/`. If chunk parameters change in your config, the cache is invalidated and re-chunked automatically.

WAV and FLAC are fully supported. MP3 sources are converted automatically -- the encoder delay (`align_data: 1057` for iTunes-encoded files) is baked in during conversion.

---

## Training

Run `apollo.bat` (Windows) or `./apollo.sh` (Linux/macOS) to open the TUI. Select **Train**, choose your config, and training starts immediately with live output in the terminal. Ctrl+C stops training cleanly, saves a checkpoint, and returns to the menu.

To run directly from the command line:

```bash
train --conf_dir configs/apollo_name.yaml
```

Each run creates a timestamped folder under `runs/<n>/<timestamp>/`. Set `resume: true` in your config to continue from the most recent checkpoint automatically.

Validation audio (LQ, HQ and restored triplets) is saved to `runs/<n>/<timestamp>/val_audio/` every val run for a fixed set of reference chunks. Comparing these files over the course of training is the best way to monitor progress.

After each val run the console prints:

```
  [val] msstft=0.3421  sfr=0.991  hfmae=0.0312  sdr=0.052  sisdr=25.341  (42.3s)
```

- **msstft** -- multi-scale log-STFT loss. Lower is better.
- **sfr** -- spectral flatness ratio in the 8-22kHz band. Rising above ~1.05 (flagged as `noise^`) is an early overfitting signal.
- **hfmae** -- mean absolute log-magnitude error in the 13-19kHz transition band. Most direct signal for MP3 rolloff fine-tuning.
- **sdr** -- Signal-to-Distortion Ratio. Less noisy than sisdr.
- **sisdr** -- waveform fidelity. Treat as a secondary signal.

All five are logged to TensorBoard.

### Checkpoints

All checkpoints are kept. Each is named with full stats and a rank badge:

```
[1]-step=001200-sisdr=-24.801-msstft=0.2913-sfr=0.988-hfmae=0.0241-sdr=0.052.ckpt
[2]-step=001100-sisdr=-24.650-msstft=0.2934-sfr=0.991-hfmae=0.0258-sdr=0.048.ckpt
```

`[1]` = best by weighted composite of perceptual metrics. Rank badges update after every save.

---

## Inference

Open the TUI and select **Inference** to pick a config, model, and input file interactively. The model picker shows the **Latest checkpoint** and the **Best checkpoint** as separate options. Batch processing runs all files in the input folder sequentially.

Non-WAV inputs (MP3, FLAC, etc.) are converted to 32-bit float WAV in-place before processing to ensure consistent decoder behavior across tools.

To run directly from the command line:

```bash
inference --in_wav "degraded.mp3" --out_wav "restored.wav" --conf_dir configs/apollo_stfl.yaml
```

Or with explicit weights:

```bash
inference --in_wav "degraded.mp3" --out_wav "restored.wav" --weights models/apollo_model_uni.ckpt --feature_dim 384
```

Output format is 32-bit float WAV.

### Spectral Merge

The TUI ensemble picker lets you blend the original input with the enhanced output in the frequency domain. Options: no ensemble, low-end preserve (max magnitude below 700 Hz), low-end + transition blend, or custom JSON. Available via CLI:

```bash
# Low-end preservation
inference ... --low_end_preserve

# Custom band control
inference ... --ensemble '[{"lo":0,"hi":700,"mode":"max_fft","weight":1.0},{"lo":15000,"hi":22050,"mode":"avg","weight":0.6}]'

# Auxiliary checkpoint blending
inference ... --aux_weights runs/other/checkpoints/best.ckpt --aux_ensemble '[{"lo":8000,"hi":22050,"mode":"enhanced","weight":1.0}]'
```

Blend modes: `max_fft`, `min_fft`, `avg`, `original`, `enhanced`. Each band takes a `weight` (0-1) blending between the mode result and pure enhanced output. Phase always comes from the enhanced output.

---

## Config Reference

Two base configs are included: `configs/apollo.yaml` and `configs/apollo_uni.yaml`. Copy and rename for each fine-tune. Local configs are not tracked by git.

### exp

| Key | Description |
|---|---|
| `dir` | Root directory for run outputs. Default `./runs`. |
| `name` | Subfolder name for this run. Derived from config filename if not set. |
| `resume` | `true` = resume from most recent checkpoint. `false` = start a new run. |

### optimizations

| Key | Description |
|---|---|
| `tf32` | TF32 matmuls. Only benefits Ampere+ GPUs (RTX 3000/4000 series). Disable for GAN training. |
| `cudnn_benchmark` | Benchmarks cuDNN conv algorithms on first batch. Leave `true` for fixed input shapes. |
| `expandable_segments` | Reduces CUDA allocator fragmentation. Leave `true`. |
| `triton_cache` | Caches compiled Triton kernels. Saves 30-60s on startup after first run. |
| `ram_limit_fraction` | Fraction of system RAM at which the process exits cleanly. Default `0.95`. |

### training

| Key | Description |
|---|---|
| `n_layers_to_freeze` | Freeze the first N BSNet layers. Apollo has 6 total. `4` is recommended for synthetic/noisy degradation; `0` for clean/consistent degradation. |
| `val_songs` | Number of songs saved per val run as LQ/HQ/Restored triplets. |
| `val_rotate_every` | `auto` = derive rotation cadence from total configured steps. Integer = switch every N val runs. |
| `grad_accum_steps` | Accumulate gradients over N steps to simulate a larger batch without extra VRAM. |

### datas

| Key | Description |
|---|---|
| `sr` | Sample rate. Fixed at `44100`. |
| `segment_sec` | Chunk length in seconds. |
| `batch_size` | Chunks per step. `1` is a safe starting point. |
| `num_workers` | DataLoader workers. `2-4` recommended on 16 GB RAM. |
| `pin_memory` | Set `false` on 16 GB systems. |
| `align_data` | Fixed sample offset for encoder delay. Baked into WAV at conversion time. iTunes MP3s use `1057`. Set `false` to disable. |

### Augmentation

`live` augmentations run each epoch in the DataLoader workers. `cached` augmentations are baked into chunk files at prep time.

| Augmentation | Type | Notes |
|---|---|---|
| `stereo_alternation` | Live | Alternates L/R by sample index. Balanced stereo exposure without random channel collapse. |
| `gain` | Live | Random gain shift applied identically to LQ and HQ. Never hard-clamps. |
| `polarity` | Live | Randomly flips signal polarity. |
| `noise` | Live | Matched Gaussian noise added to both LQ and HQ. Avoid with fragile/synthetic degradation. |
| `pitch_shift` | Cached | Disabled recommended for codec restoration. |
| `mp3_degradation` | Cached | CBR MP3 re-encode on LQ only. |

### loss_g (band weight)

| Key | Description |
|---|---|
| `band_weight_shape` | `gaussian` (default) or `trapezoid`. |
| `band_weight_center_hz` | Gaussian: center frequency of the penalty bump in Hz. Default `15000`. |
| `band_weight_sigma_hz` | Gaussian: width of the bump (1-sigma) in Hz. Default `3000`. |
| `band_weight_lo_hz` | Trapezoid: low edge of the boosted band in Hz. |
| `band_weight_hi_hz` | Trapezoid: high edge of the boosted band in Hz. |
| `band_weight_ramp_hz` | Trapezoid: soft ramp width at each edge in Hz. |
| `band_weight_gain` | Peak gain above baseline. `0` = flat loss. Start at `0` and enable after a baseline run. |

### discriminator

| Key | Description |
|---|---|
| `window_weight_boost` | Biases discriminator toward HF detail. Leave `false` for mildly degraded sources. |

### model

| Key | Description |
|---|---|
| `feature_dim` | `256` = base, `384` = universal. Must match pretrained weights. |
| `layer` | Always `6` for pretrained Apollo weights. |
| `win` | Always `20` for pretrained Apollo weights. |

### optimizer

| Key | Description |
|---|---|
| `type` | `adamw`, `adamw_8bit` (cuts optimizer VRAM ~75%), or `cpu_offload`. |
| `lr_g` | Generator learning rate. `3e-6` recommended for fine-tuning. |
| `lr_d` | Discriminator learning rate. Keep ~10x lower than `lr_g`. |
| `betas_g` | Generator Adam betas. Default `[0.9, 0.999]`. |
| `betas_d` | Discriminator Adam betas. Default `[0.5, 0.99]`. |

### system

| Key | Description |
|---|---|
| `gradient_checkpointing` | Recomputes activations during backward. Saves 30-40% VRAM at ~30% compute cost. |

### checkpoint and trainer

| Key | Description |
|---|---|
| `val_check_interval` | Validate every N training steps. |
| `limit_val_batches` | Cap val batches per run. `100` gives good coverage at 3s chunk size. |
| `max_epochs` | Hard epoch cap. Early stopping usually triggers before this. |
| `precision` | `16-mixed` for fp16 mixed precision. |
| `patience` | Early stopping patience in val runs. |

---

## Credits

Based on [JusperLee/Apollo](https://github.com/JusperLee/Apollo) by Kai Li and Yi Luo (Tsinghua University / Tencent AI Lab), ICASSP 2025.

```bibtex
@inproceedings{li2025apollo,
  title={Apollo: Band-sequence Modeling for High-Quality Music Restoration in Compressed Audio},
  author={Li, Kai and Luo, Yi},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2025},
  organization={IEEE}\n}
```

Licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

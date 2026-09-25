# Apollo-mod

A custom fine-tuning fork of [JusperLee/Apollo](https://github.com/JusperLee/Apollo), a GAN-based audio restoration model targeting degraded audio.

---

## What is Apollo?

Apollo is a research model from Tsinghua University / Tencent AI Lab (ICASSP 2025). It restores degraded audio by splitting the signal into frequency bands and modeling relationships between them. The original was trained on MUSDB18-HQ and MoisesDB with MP3 compression as the primary degradation type, but it has proven to be broadly useful for a variety of different restoration tasks.

## What's Different?

This fork reworks it for single-GPU fine-tuning on your own paired audio datasets. The aim is to help the community explore the true potential of the architecture, unlocking the ability for custom finetunes on consumer hardware.

For a detailed breakdown of every change relative to the original codebase, see the [Changes and Improvements](https://github.com/Emball/Apollo-mod/wiki/Changes-and-improvements) wiki page.

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

On first run, sources are converted to WAV and chunked into fixed-length segments under `usr/cache/chunks/`. The cache is keyed on source file contents and chunk parameters, so changing either automatically triggers a rebuild. The same cache is shared across configs that use the same dataset and parameters.

16-bit WAV and FLAC are fully supported. MP3 is supported but not recommended as it requires manual delay compensation in your config.

### Validation Set Guidelines

Before training starts, two 10-second clips are extracted from each val song and cached. Clips are selected from the highest-energy regions of each track (skipping intros and outros), so you're always scoring on representative material. On the first val run, `val_songs` of those clips are locked permanently. Every val run scores SDR, SFR, and VISQOL on those same clips, and writes the LQ/HQ/Restored triplets to `val_audio/` so you can listen and judge quality directly. Because the same clips are used throughout, scores are directly comparable across every checkpoint. If your val set is larger than `val_songs`, the active window rotates on `val_rotate_every` steps and the console prints a summary of the best metrics from the previous window before switching.

After each val run the console prints:

```
  [val] visqol=3.821  sdr=10.234  sfr=0.968  sisdr=12.458  (18.8s)
```

- **visqol** — perceptual quality score (0-5). Correlates with human listening judgement. Higher is better. Requires `visqol-python` (`uv pip install "visqol-python[all]"`); silently skipped and reported as 0.0 if not installed.
- **sdr** — Signal-to-Distortion Ratio. Primary checkpoint monitor. Higher is better.
- **sfr** — spectral flatness ratio in the 8-22kHz band. Rising is expected for MP3 restoration (the model is reconstructing frequencies the codec removed). Watch for abrupt changes, not absolute values.
- **sisdr** — waveform fidelity, noisy. Logged for reference.

All four are logged to TensorBoard.

**What to put in your val set:**

- Use your most representative and challenging material.
- Match the degradation type exactly to your training data.
- A few songs of similar character is better than many songs of mixed difficulty.

---

## Training

Run `apollo.bat` (Windows) or `./apollo.sh` (Linux/macOS) to open the TUI. Select **Train**, choose your config, and training starts immediately with live output in the terminal. Ctrl+C stops training cleanly, saves a checkpoint, and returns to the menu.

Pass `--dev` to include configs from `dev/` alongside the standard `configs/` directory in the TUI config picker:

```bash
apollo.bat --dev
```

Before training begins, a baseline val pass runs on the pretrained weights so you have a reference point:

```
[baseline] sisdr=22.140  (pretrained, before any training)
```

Training lines show speed and the most recent val metrics:

```
  24.9%  step=400  400/1604  1.38 it/s  visqol=3.821  sdr=10.234  sfr=0.968  sisdr=12.458
```

To run directly from the command line:

```bash
train --conf_dir configs/apollo_name.yaml
```

Each run creates a timestamped folder under `runs/<name>/<timestamp>/`. Set `resume: true` in your config to continue from the most recent checkpoint automatically.

### Checkpoints

All checkpoints are kept. Each is named with full stats and a rank badge:

```
[1]-step=001200-val_loss=-12.470-val_visqol=3.940-val_sdr=10.143-val_sfr=0.973.ckpt
[2]-step=001100-val_loss=-12.398-val_visqol=3.891-val_sdr=10.120-val_sfr=0.975.ckpt
```

`[1]` = best by `val_sdr`. The rank badges are updated after every new checkpoint save. Offline `evaluate.py` re-ranks the full set and can add additional metrics.

---

## Inference

Open the TUI and select **Inference** to pick a config, model, and input file interactively. The model picker shows the **Latest checkpoint** (what training resumes from) and the **Best checkpoint** (rank `[1]` by the weighted composite score) as separate options. The TUI remembers your last-used settings per config. Batch processing runs all files in the input folder sequentially without prompting between files.

### Spectral Merge (Ensemble Inference)

The inference engine can blend the original input with the enhanced output in the frequency domain. This is useful for preserving low-end content that Apollo sometimes struggles with, or for blending outputs from multiple checkpoints.

**TUI ensemble picker:** After selecting your output path, you'll see an ensemble picker with options:
- **No ensemble** — standard enhanced-only output
- **Low-end preserve** — applies `max_fft` below 700 Hz (original and enhanced blended by bin-wise maximum)
- **Low-end + transition blend** — `max_fft` below 700 Hz + weighted average in the 15-22kHz transition band
- **Custom JSON** — full band control via JSON input

**Command-line flags:**

```bash
# Quick low-end preservation (crossover at 700 Hz)
inference --in_wav input.wav --out_wav output.wav --conf_dir configs/apollo_stfl.yaml --low_end_preserve

# Custom crossover frequency
inference ... --low_end_preserve --low_end_hz 1000

# Full ensemble control via JSON
inference ... --ensemble '[{"lo":0,"hi":700,"mode":"max_fft","weight":1.0},{"lo":15000,"hi":22050,"mode":"avg","weight":0.6}]'

# Auxiliary checkpoint blending
inference ... --aux_weights runs/other/checkpoints/best.ckpt --aux_ensemble '[{"lo":8000,"hi":22050,"mode":"enhanced","weight":1.0}]'
```

**Blend modes per band:**
- `max_fft` — bin-wise maximum magnitude (takes whichever of original or enhanced has more energy)
- `min_fft` — bin-wise minimum magnitude
- `avg` — linear average of magnitudes
- `original` — bypass, use input only
- `enhanced` — bypass, use model output only

Each band also has a `weight` (0-1) that blends between the mode result and pure enhanced output.

---

## Config Reference

Two base configs are included: `configs/apollo.yaml` and `configs/apollo_uni.yaml`. Copy and rename for each fine-tune. Local configs are not tracked by git.

### Choosing Your Config: Lessons from Real Runs

**Synthetic / noisy degradation (e.g., WMA→MP3 stacking, inconsistent encoding):**
- Freeze 4 layers (`n_layers_to_freeze: 4`)
- Keep batch size at 1 (`batch_size: 1`, `grad_accum_steps: 1`)
- Disable noise augmentation (`noise.enabled: false`)
- Set `val_check_interval` to 200-300 for cleaner metrics
- Disable TF32 (`tf32: false`)
- Start with `band_weight_gain: 0`, enable after baseline

**Real / consistent degradation (e.g., iTunes MP3s, single encoder fingerprint):**
- Unfreeze all layers (`n_layers_to_freeze: 0`)
- Can increase batch size to 2-4 (`batch_size: 2`, `grad_accum_steps: 2`)
- Noise augmentation is safe (`noise.enabled: true`)
- TF32 is optional but safer to disable (`tf32: false`)
- Start with `band_weight_gain: 0`, enable after baseline

**General:**
- Start with `band_weight_gain: 0` for a baseline run, then enable at 1.0-1.5 after you have a reference point
- Use `val_songs` to match your validation set size
- SI-SDR is noisy — use VISQOL as the primary signal; SFR as a canary
- SFR climbing is not always a problem — it often means the model is synthesizing HF content. Watch for trend changes, not a hard threshold.

### exp

| Key | Description |
|---|---|
| `dir` | Root directory for run outputs. Default `./runs`. |
| `name` | Subfolder name for this run. Derived from config filename if not set. |
| `resume` | `true` = resume from most recent checkpoint. `false` = start a new run. |

### optimizations

| Key | Description |
|---|---|
| `tf32` | TF32 matmuls. Only benefits Ampere+ GPUs (RTX 3000/4000 series). **Disable for GAN training** — spectral loss landscapes are sensitive to numerical error. Default `false`. |
| `cudnn_benchmark` | Benchmarks cuDNN conv algorithms on first batch. Leave `true` for fixed input shapes. |
| `expandable_segments` | Reduces CUDA allocator fragmentation. Leave `true`. |
| `triton_cache` | Caches compiled Triton kernels. Saves 30-60s on startup after first run. |
| `ram_limit_fraction` | Fraction of system RAM at which the process exits cleanly. Default `0.95`. |

### training

| Key | Description |
|---|---|
| `n_layers_to_freeze` | Freeze the first N BSNet layers. Apollo has 6 total. `4` is recommended for synthetic/noisy degradation; `0` for clean/consistent degradation like real iTunes encodes. Ignored when `extra_layers > 0`. |
| `extra_layers` | Append N new randomly-initialized BSNet layers after loading pretrained weights. All pretrained layers are frozen; only the new layers train. New layers are zero-initialized as near-identity so the model starts at pretrained performance. Default `0`. |
| `val_songs` | Number of val clips evaluated per val run. Two 10-second clips are extracted per song before training starts, so with 3 val songs the pool has 6 clips. `val_songs: 6` uses all of them. Locked on the first val run. LQ/HQ/Restored triplets saved to `val_audio/` after each run. |
| `val_rotate_every` | Steps between val clip window rotation. Only meaningful when the pool is larger than `val_songs`. Integer step count or omit to disable. |
| `grad_accum_steps` | Accumulate gradients over N steps to simulate a larger batch without extra VRAM. |

### datas

| Key | Description |
|---|---|
| `sr` | Sample rate. Fixed at `44100`. |
| `segment_sec` | Chunk length in seconds. |
| `batch_size` | Chunks per step. `1` is a safe starting point for noisy/synthetic degradation. |
| `num_workers` | DataLoader workers. `2-4` recommended on 16 GB RAM. |
| `pin_memory` | Set `false` on 16 GB systems. |
| `align_data` | Fixed sample offset for encoder delay. Baked into WAV at conversion time. iTunes MP3s use `1057`. Set `false` to disable. |

### Augmentation

`live` augmentations run each epoch in the DataLoader workers. `cached` augmentations are baked into chunk files at prep time.

| Augmentation | Type | Notes |
|---|---|---|
| `stereo_alternation` | Live | Alternates L/R by sample index. Balanced stereo exposure without random channel collapse. Skipped if `mid_side_isolation` fires on the same chunk. |
| `gain` | Live | Random gain shift applied identically to LQ and HQ. Never hard-clamps. |
| `deep_gain` | Live | Large gain reduction (e.g. -10 to -6 dB) at low probability. Trains robustness to quiet passages. |
| `silence_dip` | Live | Briefly fades signal toward silence with configurable ramp shapes. Trains robustness to dynamic dips. |
| `polarity` | Live | Randomly flips signal polarity. |
| `mid_side_isolation` | Live | Collapses the pair to mid `(L+R)/2` or side `(L-R)/2`, duplicated to both channels. Applied identically to LQ and HQ. Helps the model generalize to stereo field components. Rolls before stereo_alternation; if it fires, stereo_alternation is skipped. |
| `noise` | Live | Matched Gaussian noise added to both LQ and HQ. **Do not use with fragile/synthetic degradation** — it pollutes the gradient signal. Safe for clean/consistent degradation. |
| `pitch_shift` | Cached | Disabled recommended for codec restoration. |
| `mp3_degradation` | Cached | CBR MP3 re-encode on LQ only. |

### loss_g (band weight)

| Key | Description |
|---|---|
| `band_weight_shape` | `gaussian`, `trapezoid`, or `piecewise`. |
| `band_weight_center_hz` | Gaussian only. Center frequency of the penalty bump in Hz. Default `15000`. |
| `band_weight_sigma_hz` | Gaussian only. Width of the bump (1-sigma) in Hz. Default `3000`. |
| `band_weight_lo_hz` | Trapezoid only. Low edge of the boosted band in Hz. Default `4500`. |
| `band_weight_hi_hz` | Trapezoid only. High edge of the boosted band in Hz. Default `18500`. |
| `band_weight_ramp_hz` | Trapezoid only. Width of the soft ramp at each edge in Hz. Default `1500`. |
| `band_weight_breakpoints` | Piecewise only. List of `[hz, weight]` pairs defining the curve; linearly interpolated between points. Weight range is 0–1. |
| `band_weight_gain` | Peak gain above baseline. `0` = flat loss regardless of shape. |

`gaussian` adds a bump centered on one frequency — good for a targeted HF push. `trapezoid` targets a flat band with soft edges — useful for a known codec rolloff zone. `piecewise` takes arbitrary `[hz, weight]` breakpoints and interpolates linearly — use this when you have actual null-test data showing where the codec does damage and want to match the weighting to that curve exactly. Start with `gain: 0` for a baseline run.

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
| `type` | `adamw`, `gefen` (recommended — lower memory, faster steps; `uv pip install gefen`), `gefen_muon` (same as `gefen` for Apollo — no 2D params), `adamw_8bit` (bitsandbytes), or `cpu_offload`. |
| `lr_g` | Generator learning rate. `3e-6` recommended for fine-tuning close to the target distribution; `1e-5` for more aggressive adaptation. |
| `lr_d` | Discriminator learning rate. Keep ~10x lower than `lr_g`. |
| `betas_g` | Generator Adam betas. Default `[0.9, 0.999]`. |
| `betas_d` | Discriminator Adam betas. Default `[0.5, 0.99]` — classic GAN recommendation. |

### system

| Key | Description |
|---|---|
| `gradient_checkpointing` | Recomputes activations during backward. Saves 30-40% VRAM at ~30% compute cost. |
| `visqol_fraction` | Fraction of val songs to score with VISQOL per val run. `1.0` = all. Requires `visqol-python` (`uv pip install "visqol-python[all]"`); silently skipped and reported as 0.0 if not installed. Default `1.0`. |
| `target_band_loss_enabled` | Adds a configurable-range band loss to live val metrics. Off by default. When enabled, appears as `tbl=` in console and checkpoint filenames. |
| `target_band_loss_lo_hz` | Low edge of the target band in Hz. Default `13000`. |
| `target_band_loss_hi_hz` | High edge of the target band in Hz. Default `19000`. |

### checkpoint and trainer

| Key | Description |
|---|---|
| `val_check_interval` | Validate every N training steps. |
| `limit_val_batches` | Cap on the fixed chunk batch used for `val_loss` (sisdr) computation. `100` is typical. |
| `max_epochs` | Hard epoch cap. Early stopping usually triggers before this. |
| `precision` | `16-mixed` for fp16 mixed precision. |
| `patience` | Early stopping patience in val runs. `100` is reasonable; set higher or disable for exploratory runs. |

---

## Lessons Learned (Quick Reference)

- **Frozen layers** — `4` for synthetic/noisy degradation; `0` for clean/consistent degradation.
- **Augmentation** — noise augmentation harms fragile/synthetic degradation; safe for clean degradation.
- **Batch size** — smaller (`1`) stabilizes training on noisy degradation; larger (`2-4`) works for clean degradation.
- **TF32** — disable for GAN training on spectral data; precision loss matters.
- **Training step order** — discriminator first, fresh feature maps for generator loss. Do not reuse feature maps.
- **SI-SDR** — noisy and misleading for restoration tasks. Use VISQOL and SDR as primary signals; SFR as a canary.
- **SFR** — watch for trend changes, not absolute threshold. Rising SFR with stable VISQOL is fine.
- **Encoder fingerprint** — real encoder output beats synthetic approximation. Match the encoder version exactly.

---

## Credits

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

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

Place your audio files under `data/apollo_name/`, where the folder name matches your config file (e.g. `configs/apollo_name.yaml`) and the `name` value in your config.

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

16-bit WAV and FLAC are fully supported. MP3 is supported but not recommended as it requires manual delay compensation in your config, which can be difficult to reliably calculate. For delay-controlled degradation, you can have the LQ side generated from your HQ files instead (see Generate LQ below); the generator supports very complex encoding configurations.

### Generate LQ

For HQ songs that do not have an accompanying pre-prepared LQ file, Apollo-mod can build the degraded half of a pair for you using its compression and degradation tools. Within `utils/degrade/`, you can create custom configurations, or 'recipes', for the type of codec degradation you want, with broad support for layering different codecs, randomizing bitrates, setting percentages and likelihoods, and other encoding settings.

A recipe is a JSON file that describes the degradation you want as odds instead of one fixed chain. Apollo-mod rolls against the recipe for each generated chunk, so the model sees a range of the degradation, which can make it more robust to small variations you can't account for otherwise. Supported codecs are MP3 (LAME or Fraunhofer), WMA, AAC, Vorbis, Opus, and MP2. `utils/degrade/stfl.json` is the included example (WMA, then several MP3 passes, finishing at 192 kbps CBR). Copy it and change the numbers to match your own material.

To use it, put a clean song in `HQ/` with no matching file in `LQ/` and turn on the [compression](#compression) block in your config, pointing it at your recipe. The generated LQ passes are automatically aligned no matter how severe the compression. **Songs that already have an LQ file are used as they are, so real pairs and generated ones can share a folder.** This also works for `val/`.

This only helps when the degradation you are restoring can be reproduced by a codec chain. **For anything else, supply your own LQ files.**

### Validation Guidelines

**What to put in your val set:**

- Use your most representative and challenging material.
- Match the degradation type exactly to your training data.
- A few songs of similar character is better than many songs of mixed difficulty.

### Validation System

On the first val run, `val_songs` clips (two per val song, see below) are selected from the full val set and locked in training until the val set changes. `val_rotate_every` allows the model to select a different set of val clips from the full val folder based on the configured step count. 

Each val song is cut into two 10-second clips taken from its highest-energy sections to follow guidelines for metric calculations. Every val run scores VISQOL, SI-SDR, and HFNR on those same clips, and writes the LQ/HQ/Restored triplets to `runs/apollo_name/<timestamp>/val_audio/` so you can listen and judge quality directly after every validation. 

Because the same clips are used throughout, scores are directly comparable across checkpoints for as long as the clips stay the same. If your val set is larger than `val_songs` and `val_rotate_every` is set, the active window rotates on that step interval and the console prints a summary of the best metrics from the previous window before switching. `visqol` is not comparable across windows; use `visqol_gain` for that.

### Validation Metrics

After each val run the training progress line (see Training) shows the latest metrics. All of them are logged to TensorBoard.

- **visqol** — Perceptual quality score (0-5) that approximates human perception. Higher is better.
- **sisdr** — Waveform fidelity. Higher is better. 
- **hfnr** — High-frequency noise ratio. Rising is expected for MP3 restoration, so lower may not always be better.
- **visqol_gain** — ViSQOL of the restored clips minus ViSQOL of the LQ clips, over the same clips. Logged to TensorBoard only. Unlike `visqol`, it stays comparable when the val window rotates.

---

## Training

Run `apollo.bat` (Windows) or `./apollo.sh` (Linux/macOS) to open the TUI. Select **Train**, choose your config, and training starts immediately with live output in the terminal. Ctrl+C saves an `interrupted` checkpoint at the next safe point and returns to the menu; a second Ctrl+C quits immediately without saving.

Before a fresh run begins, a baseline val pass runs on the pretrained weights so you have a reference point. Its metrics appear on the progress line until the first real val run.

Training lines show speed and the most recent val metrics:

```
  24.9%  step=400  400/1604  1.38 it/s  visqol=3.913  sisdr=14.458  hfnr=0.968
```

Each run creates a timestamped folder under `runs/apollo_name/<timestamp>/`. Set `resume: true` in your config to continue from the most recent checkpoint automatically or set `resume_checkpoint` to the exact checkpoint you wish to resume training from. If neither is set, the model will start training from the base model.

### Checkpoints

Each checkpoint is named with full stats:

```
step=001200-sisdr=12.470-visqol=3.940-hfnr=0.973.ckpt
step=001100-sisdr=12.398-visqol=3.891-hfnr=0.975.ckpt
```

The best `keep_best` checkpoints by VISQOL and the latest `keep_latest` are kept (`10` and `3` in the shipped configs). Interrupted checkpoints are never removed. Set `keep_best: 0` to keep everything.

Select **Evaluate** in the TUI to rank every checkpoint in a run on the val clips used in training. Metrics already in a checkpoint's filename are reused. Pass `--recompute` on the command line to rescore every checkpoint under the current val protocol, which is needed to compare checkpoints saved before protocol 2.
## Inference

Open the TUI and select **Inference** to pick a config, model, and input file interactively. The model picker shows the **Latest checkpoint** (what training resumes from) and the **Best checkpoint** (highest VISQOL, with SI-SDR and then HFNR closest to 1.0 as tiebreakers) as separate options. The TUI remembers your last-used settings per config. Batch processing runs all files in the input folder sequentially without prompting between files.

### Spectral Merge (Ensemble Inference)

The inference engine can blend the original input with the enhanced output in the frequency domain. This is useful for preserving low-end content that Apollo sometimes struggles with, or for blending outputs from multiple checkpoints.

**TUI ensemble picker:** After selecting your output path, you'll see an ensemble picker with options:
- **No ensemble** — standard enhanced-only output
- **Low-end preserve** — applies `max_fft` below 700 Hz (original and enhanced blended by bin-wise maximum)
- **Low-end + transition blend** — `max_fft` below 700 Hz + weighted average in the 15-22kHz transition band
- **Custom JSON** — full band control via JSON input

**Blend modes per band:**
- `max_fft` — bin-wise maximum magnitude (takes whichever of original or enhanced has more energy)
- `min_fft` — bin-wise minimum magnitude
- `avg` — linear average of magnitudes
- `original` — bypass, use input only
- `enhanced` — bypass, use model output only

Each band also has a `weight` (0-1) that blends between the mode result and pure enhanced output.

---

## Config Reference

Two base configs are included: `apollo.yaml` and `apollo_uni.yaml` as well as a few for custom projects like `apollo_stfl2` and `apollo_stfl_new.yaml`. Copy and rename for each fine-tune and change parameters accordingly.

### exp

| Key | Description |
|---|---|
| `dir` | Root directory for run outputs. Default `./runs`. |
| `name` | Subfolder name for this run. Derived from config filename if not set. |
| `resume_checkpoint` | Path to a specific `.ckpt` file to resume from. Overrides `resume`. `null` = use `resume` behavior. |
| `resume` | `true` = resume from most recent checkpoint. `false` = start a new run. |

### optimizations

| Key | Description |
|---|---|
| `cudnn_benchmark` | Benchmarks cuDNN conv algorithms on first batch. Leave `true` for fixed input shapes. |
| `expandable_segments` | Reduces CUDA allocator fragmentation. Leave `true`. |
| `triton_cache` | Caches compiled Triton kernels. Saves 30-60s on startup after first run. |
| `ram_limit_fraction` | Fraction of system RAM at which the process exits cleanly. Default `0.98`. |

### training

| Key | Description |
|---|---|
| `n_layers_to_freeze` | Freeze the first N BSNet layers. Apollo has 6 total. `4` is recommended for training the pretrained weights a new type of degradation as the model collapses quick attempting to train all layers at once on a small dataset. If you have a bigger one or feel like taking risks, you can experiment with lower values or progressively unfreezing more layers after the later ones are already mature. Changing the value on resume is supported: saved optimizer state is carried over for every layer that still matches, and newly unfrozen layers start fresh (the console prints `[resume] The generator trainable parameters changed ...`). |
| `val_songs` | Number of val clips evaluated per val run. Each val song produces two 10-second clips, so with 3 val songs the pool has 6 clips and `val_songs: 6` uses all of them. Locked on the first val run and held fixed for the entire run. LQ/HQ/Restored triplets saved to `val_audio/` after each run. |
| `val_rotate_every` | Steps between val song window rotation which selects new pairs from the full folder. Only meaningful when the val set is larger than `val_songs`. For the clearest view possible, disabling this may be ideal unless you want to experiment with testing the model on different types of val inputs. Integer step count, or `auto`/omit to disable. |
| `grad_accum_steps` | Accumulate gradients over N steps to simulate a larger batch without extra VRAM. |

### datas

| Key | Description |
|---|---|
| `sr` | Sample rate. Fixed at `44100`. Changing this value requires training from scratch. |
| `segment_sec` | Chunk length in seconds. If you have the VRAM, larger chunk lengths can potentially increase quality. |
| `batch_size` | Chunks per step. If your dataset is extremely small, a value of `1` would be your best shot. For larger datasets, you can try 2-4 or raise until you OOM.|
| `num_workers` | DataLoader workers. Can experiment for potential speed benefits. `2-4` tested as working on 16 GB RAM. |
| `pin_memory` | Recommended to set as `false` on 16 GB or low RAM systems as it can cause instability. You can try enabling it if you have 32 GB or more. |
| `align_data` | Fixed sample offset for encoder delay if you know the exact value that the LQ files are misaligned by. Baked into WAV at conversion time. Set `false` to disable. |

### compression

Generates the LQ side for songs that have no LQ file, as described in [Generate LQ](#generate-lq). Set under `datas`. Requires `ffmpeg` with the encoders the recipe uses. A recipe can also mix Fraunhofer (`acmenc`, an older MP3 encoder much more common in the early days of the internet) and LAME MP3 passes within one chain; set `external_codecs.acmenc` in the recipe to your `acmenc` executable to use it.

| Key | Description |
|---|---|
| `enabled` | Turns LQ generation on. |
| `recipe` | Path to a recipe JSON with a `random` block, e.g. `utils/degrade/stfl.json`. |
| `seed` | Seed for chain sampling. Each chunk is seeded from the seed, file name, and chunk index. |
| `pad_sec` | Seconds of surrounding audio encoded with each chunk and discarded after alignment. |
| `workers` | Threads used while chunking. `0` uses CPU count minus 1. |

Chunks whose chain fails the recipe's alignment or silence checks are dropped, never written with a clean LQ. A summary of compressed and rejected chunks is printed after chunking. The compression settings are part of the chunk cache key.

### Augmentation

`live` augmentations run each epoch in the DataLoader workers. `cached` augmentations are baked into chunk files at prep time.

| Augmentation | Type | Notes |
|---|---|---|
| `stereo_alternation` | Live | Alternates L/R channels respectively by sample index instead of feeding the network a mono mixdown. Adds data variety by balancing stereo exposure. Skipped if `mid_side_isolation` fires on the same chunk. |
| `gain` | Live | Random gain shift applied identically to LQ and HQ. Never hard-clamps. |
| `deep_gain` | Live | Large gain reduction (e.g. -10 to -6 dB) at low probability. Trains robustness to quiet passages. |
| `silence_dip` | Live | Briefly fades signal toward silence with configurable ramp shapes. Trains robustness to dynamic dips. |
| `polarity` | Live | Randomly flips signal polarity. Cheat code for free data. You may get better results using controlled inversion on the data level, however, as in by inverting the second half a song where certain passages repeat and are identical. |
| `mid_side_isolation` | Live | Collapses the pair to mid `(L+R)/2` or side `(L-R)/2`, duplicated to both channels. Applied identically to LQ and HQ. Helps the model generalize to stereo field components. This could potentially cause the model to overgeneralize inputs for non-mid-side-isolated audio. Rolls before stereo_alternation; if it fires, stereo_alternation is skipped. |

### model

| Key | Description |
|---|---|
| `feature_dim` | `256` = base, `384` = universal. Must match pretrained weights. |
| `layer` | Always `6` for pretrained Apollo weights. |
| `win` | Always `20` for pretrained Apollo weights. |

### optimizer

| Key | Description |
|---|---|
| `type` | `adamw` (default), `adamw_8bit` (lower memory footprint; needs bitsandbytes), `gefen` and `gefen_muon` (lower memory, faster steps; need `uv pip install gefen`). |
| `lr_g` | Generator learning rate. |
| `lr_d` | Discriminator learning rate. |
| `betas_g` | Generator Adam betas. Default `[0.9, 0.999]`. |
| `betas_d` | Discriminator Adam betas. Default `[0.5, 0.99]` |

### system

| Key | Description |
|---|---|
| `gradient_checkpointing` | Recomputes activations during backward. Saves 30-40% VRAM at ~30% compute cost. |

### checkpoint and trainer

| Key | Description |
|---|---|
| `val_check_interval` | Validate every N training steps, compute metrics and save preview audio. |
| `limit_val_batches` | Cap on the fixed chunk batch used for the computation. `1.0` uses all of it. |
| `keep_best` | Under `checkpoint`. Number of best checkpoints (by VISQOL) to keep. `0` keeps all. |
| `keep_latest` | Under `checkpoint`. Number of most recent checkpoints to keep. |
| `max_epochs` | Hard epoch cap for training. |
| `precision` | `16-mixed` for fp16 mixed precision. |

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

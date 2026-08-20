# Apollo-mod

A community fine-tuning fork of [JusperLee/Apollo](https://github.com/JusperLee/Apollo), a GAN-based audio restoration model targeting codec-compressed audio.

[GitHub](https://github.com/Emball/Apollo-mod)

---

## What is Apollo?

Apollo is a research model from Tsinghua University / Tencent AI Lab (ICASSP 2025). It restores degraded audio by splitting the signal into frequency bands and modeling relationships between them. The original was trained on MUSDB18-HQ and MoisesDB with MP3 compression as the primary degradation type.

This fork reworks it for single-GPU fine-tuning on your own paired audio datasets.

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

On first run this creates a `.venv` with all dependencies. Subsequent runs activate it directly.

Note: On Windows, if `apollo.bat` fails with a uv error, your real `uv.exe` may be shadowed by a system stub at `C:\Windows\System32\uv`. Recent versions of the script handle this automatically.

---

## Prepare Your Data

Place your audio files under `data/<config_name>/`, where the folder name matches your config file (e.g. `configs/apollo_stfl.yaml` uses `data/apollo_stfl/`).

```
data/apollo_stfl/
  train/
    LQ/    degraded audio (WAV, MP3, or FLAC; filenames must match HQ)
    HQ/    clean reference audio
  val/
    LQ/
    HQ/
```

On first run, `train.py` automatically chunks these into fixed-length segments under `chunks/<config_name>/`. If you change `segment_sec` in your config, delete the `chunks/` folder to force a re-chunk.

### MP3 Alignment

If your LQ files are MP3s encoded from the HQ WAVs, set `align_data` in your config to trim the encoder delay at decode time:

```yaml
datas:
  align_data: 1057   # iTunes/AAC encoder delay in samples
```

| Value | Behavior |
|---|---|
| `1057` (or any integer) | Trim that exact number of samples from LQ at decode time |
| `true` | Auto-detect delay via LAME header, with xcorr fallback |
| `false` | No alignment |

A negative integer trims HQ instead of LQ.

---

## Training

```bash
python train.py --conf_dir configs/apollo_uni.yaml
```

Each run creates a timestamped folder under `runs/<name>/`. Set `resume: true` in your config to continue from the most recent checkpoint automatically. Ctrl+C saves a checkpoint before exiting.

Validation audio (LQ + HQ + restored triplets) is saved to `runs/<name>/<timestamp>/val_audio/` every val run. Monitor training with:

```bash
tensorboard --logdir ./runs
```

---

## Inference

```bash
# With a config file (reads model settings automatically)
python inference.py --in_wav input.mp3 --out_wav output.wav \
    --weights runs/apollo_stfl/20260819_143022/checkpoints/001200-val_loss=-24.41.ckpt \
    --conf_dir configs/apollo_stfl.yaml

# With manual settings
python inference.py --in_wav input.mp3 --out_wav output.wav \
    --weights models/apollo_model_uni.ckpt --feature_dim 384
```

Output is written to disk chunk-by-chunk as inference runs. You can open the output file in Audacity immediately to preview completed sections while the rest processes. Output format is 32-bit float WAV.

---

## VRAM Requirements

| Config | VRAM |
|---|---|
| Base (`feature_dim=256`), `batch_size=2`, `segment_sec=4`, gradient checkpointing on | ~8-10 GB |
| Universal (`feature_dim=384`), `batch_size=2`, `segment_sec=4`, gradient checkpointing on | ~10-11 GB |

Always use `gradient_checkpointing: true` and `precision: 16-mixed` on consumer GPUs.

---

## Config Reference

Two base configs are included: `configs/apollo.yaml` (`feature_dim=256`) and `configs/apollo_uni.yaml` (`feature_dim=384`). Copy and rename for each fine-tune.

### exp

| Key | Description |
|---|---|
| `dir` | Root directory for run outputs. Default `./runs`. |
| `name` | Subfolder name for this run. Derived from config filename if not set. |
| `resume` | `true` = resume from most recent checkpoint. `false` = start a new run. |

### optimizations

| Key | Description |
|---|---|
| `tf32` | TF32 matmuls. Only benefits Ampere+ GPUs (RTX 3000/4000 series). |
| `cudnn_benchmark` | Benchmarks cuDNN conv algorithms on first batch. Leave `true` for fixed input shapes. |
| `expandable_segments` | Reduces CUDA allocator fragmentation. Leave `true`. |
| `triton_cache` | Caches compiled Triton kernels. Saves 30-60s on startup after first run. |
| `ram_limit_fraction` | Fraction of system RAM at which the process exits cleanly. Default `0.95`. |

### training

| Key | Description |
|---|---|
| `n_layers_to_freeze` | Freeze the first N BSNet layers. Apollo has 6 total. `4` is recommended for fine-tuning. |
| `hf_boost` | Extra loss weight on high frequencies. `1.0` = flat. Do not exceed `2.0`. |
| `val_audio_pairs` | Number of val audio samples saved per val run. |
| `grad_accum_steps` | Accumulate gradients over N steps to simulate a larger batch without extra VRAM. |

### datas

| Key | Description |
|---|---|
| `sr` | Sample rate. Fixed at `44100`. Do not change. |
| `segment_sec` | Chunk length in seconds. Changing this requires deleting `chunks/`. |
| `batch_size` | Chunks per step. `2` is the practical limit on 11 GB VRAM with the universal model. |
| `num_workers` | DataLoader workers. `2-4` is recommended on 16 GB RAM. |
| `pin_memory` | Set `false` on 16 GB systems. Pinned memory cannot be swapped and causes instability under memory pressure. |
| `align_data` | See MP3 Alignment above. |

### Augmentation

`live` augmentations run each epoch in the DataLoader workers. `cached` augmentations are baked into chunk files at prep time.

| Augmentation | Type | Notes |
|---|---|---|
| `stereo_alternation` | Live | Alternates between L and R channels by sample index. Gives the model balanced stereo exposure without random channel clumping. |
| `gain` | Live | Random gain shift applied identically to LQ and HQ. Never hard-clamps. |
| `polarity` | Live | Randomly flips signal polarity. Free operation. |
| `noise` | Live | Matched Gaussian noise added to both LQ and HQ. |
| `pitch_shift` | Cached | Disabled for codec restoration. Warps frequency relationships the model is trying to learn. |
| `mp3_degradation` | Cached | CBR MP3 re-encode on LQ only. Use cached, not live. Live mode spawns an ffmpeg process per chunk per epoch. |

### model

| Key | Description |
|---|---|
| `feature_dim` | `256` = base, `384` = universal. Must match your pretrained weights. |
| `layer` | Always `6` for pretrained Apollo weights. |
| `win` | Always `20` for pretrained Apollo weights. |

### optimizer

| Key | Description |
|---|---|
| `type` | `adamw`, `adamw_8bit` (cuts optimizer VRAM ~75% via bitsandbytes), or `cpu_offload`. |
| `lr_g` | Generator learning rate. `1e-5` is a safe starting point for fine-tuning. |
| `lr_d` | Discriminator learning rate. Keep around 10x lower than `lr_g`. |

### system

| Key | Description |
|---|---|
| `gradient_checkpointing` | Recomputes activations during backward pass. Saves 30-40% VRAM at ~30% compute cost. Recommended on consumer GPUs. |

### checkpoint and trainer

| Key | Description |
|---|---|
| `save_top_k` | Number of checkpoints to keep, ranked by `val_loss`. Set to `5` to avoid unbounded disk growth. |
| `val_check_interval` | Validate every N training steps. |
| `limit_val_batches` | Cap val batches per run. `25` is a good balance between speed and coverage. |
| `max_epochs` | Hard epoch cap. Early stopping usually triggers before this. |
| `precision` | `16-mixed` for fp16 mixed precision. Required on consumer GPUs. |

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

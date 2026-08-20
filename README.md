# Apollo-mod

A custom fine-tuning fork of [JusperLee/Apollo](https://github.com/JusperLee/Apollo), a GAN-based audio restoration model targeting degraded audio.

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

On first run this creates a `.venv` with all dependencies. Subsequent runs activate it directly.

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

On first run, `train.py` automatically chunks these into fixed-length segments under `chunks/<apollo_name>/`. If chunk parameters change in your config, the cache is invalidated and re-chunked automatically.

16-bit WAV and FLAC are fully supported. MP3 is supported but not recommended as it requires manual delay compensation in your config.

---

## Training

Run the apollo.bat/.sh script to auto-launch the venv console. Then run:

```bash
python train.py --conf_dir configs/apollo_name.yaml
```

Each run creates a timestamped folder under `runs/<name>/<timestamp>/`. Set `resume: true` in your config to continue from the most recent checkpoint automatically. Ctrl+C saves a checkpoint before exiting.

Validation audio (LQ, HQ and restored triplets) is saved to `runs/<name>/<timestamp>/val_audio/` every val run for a pre-determined set of samples. Comparing these files over the course of training is the best way to monitor progress.

---

## Inference

Run the apollo.bat/.sh script to auto-launch the venv console. Then run:

```bash
python inference.py --in_wav "degraded.mp3" --out_wav "restored.wav" --conf_dir configs/apollo_stfl.yaml
```

Or run with manual CLI args:

```bash
python inference.py --in_wav "degraded.mp3" --out_wav "restored.wav" --weights models/apollo_model_uni.ckpt --feature_dim 384
```

When `--conf_dir` is provided without `--weights`, inference automatically selects the best checkpoint from your run folder based on val_loss. Output is written to disk chunk-by-chunk as inference runs. You can open the output file in Audacity immediately to preview completed sections while the rest processes. Output format is 32-bit float WAV.

---

## Config Reference

Two base configs are included: `configs/apollo.yaml` and `configs/apollo_uni.yaml`. Copy and rename for each fine-tune. Below is a comprehensive documentation of every parameter:

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
| `triton_cache` | Caches compiled Triton kernels. Saves 30-60s on startup after first run, but may cause dependency hassles to get working. |
| `ram_limit_fraction` | Fraction of system RAM at which the process exits cleanly. Default `0.95`. |

### training

| Key | Description |
|---|---|
| `n_layers_to_freeze` | Freeze the first N BSNet layers. Apollo has 6 total. `4` is recommended for fine-tuning models that are codec-degradation based, but more may be required for more advanced restoration tasks. |
| `hf_boost` | Extra loss weight on high frequencies introduced to improve reconstruction. `1.0` = flat. Do not exceed `2.0`. |
| `val_audio_pairs` | Number of val audio samples saved per val run. |
| `grad_accum_steps` | Accumulate gradients over N steps to simulate a larger batch without extra VRAM. With a batch size of 1 and `grad_accum_steps` at 2, this simulates a batch size of 2 with the memory footprint of 1, at the cost of speed. |

### datas

| Key | Description |
|---|---|
| `sr` | Sample rate. Fixed at `44100`. Do not change unless you know what you're doing. |
| `segment_sec` | Chunk length in seconds. Increasing this value can potentially improve model quality. However, it drastically impacts performance and might cause issues when fine-tuning from the known Apollo base models. |
| `batch_size` | Chunks per step. `1` is a good starting point to test where your VRAM sits before attempting to increase it. If you have a newer generation GPU with more than 11 GB of memory, starting with `2` for testing is likely the better move. |
| `num_workers` | DataLoader workers. `2-4` is recommended on 16 GB RAM. Increasing beyond that seems to cause severe lag and memory issues. If you have more RAM, you can likely push it closer to your core count. |
| `pin_memory` | Set `false` on 16 GB systems. Pinned memory cannot be swapped and causes instability under memory pressure. Worth experimenting with if you have more memory. |
| `align_data` | Offset the MP3 chunks by a set number of samples either backwards or forwards. This can be used to correct for delay introduced by the encoding process of LQ data, but only really works if the delay is consistent across your entire dataset. |

### Augmentation

`live` augmentations run each epoch in the DataLoader workers, and as such do make a marginal impact on training speed. `cached` augmentations are baked into chunk files at prep time and increase pre-processing duration.

| Augmentation | Type | Notes |
|---|---|---|
| `stereo_alternation` | Live | Alternates between L and R channels by sample index. Gives the model balanced stereo exposure without random channel clumping. This improves data variety and model versatility. |
| `gain` | Live | Random gain shift applied identically to LQ and HQ. Never hard-clamps. |
| `polarity` | Live | Randomly flips signal polarity. Practically free data. |
| `noise` | Live | Matched Gaussian noise added to both LQ and HQ. |
| `pitch_shift` | Cached | Disabled is recommended for codec restoration, as it warps frequency relationships the model is trying to learn. Potentially worth experimenting with for other types of restoration tasks. |
| `mp3_degradation` | Cached | CBR MP3 re-encode on LQ only introduced to help in cases where the target data of your fine-tune is heavily layered in compression. Live mode spawns an ffmpeg process per chunk per epoch so is generally not recommended. |

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

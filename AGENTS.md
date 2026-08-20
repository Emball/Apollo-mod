# Apollo-mod — Agent Reference

## Project

Custom fine-tuning fork of [JusperLee/Apollo](https://github.com/JusperLee/Apollo), a GAN-based audio restoration model targeting codec-compressed audio. Reworked for single-GPU fine-tuning on user-supplied paired LQ/HQ datasets.

---

## Architecture

| File | Role |
|---|---|
| `look2hear/models/apollo.py` | Generator. BSNet + Roformer layers. STFT/iSTFT always in float32. |
| `look2hear/discriminators/frequencydis.py` | Frequency discriminator. Hann windows cached as plain tensors in `_hann_cache` (not `register_buffer` — keeps them out of checkpoint state_dict). Mono input auto-expanded to stereo via `.expand()`. |
| `look2hear/system/audio_litmodule.py` | Lightning module. Manual optimization, gradient checkpointing (BSNet only), val audio saving, RAM/CUDA OOM watchdogs, val index locking. |
| `look2hear/losses/gan_losses.py` | GAN losses. Hann windows and HF weight tensors in `_hann_cache`/`_weight_cache` dicts (not buffers). |
| `paired_datamodule.py` | Loads chunked LQ/HQ WAV pairs. Live + cached augmentation pipeline. Val dataloader shuffles; dataset index passed through batch for stable audio monitoring. |
| `train.py` | Entry point. Auto-chunks `data/` on first run, then trains. Run isolation via timestamped subdirs, resume logic, chunk cache manifest. |
| `inference.py` | Entry point. Chunked inference, streams output to disk. Auto-selects best checkpoint by val_loss when `--weights` is omitted. |
| `configs/apollo.yaml` | Base config (`feature_dim=256`). |
| `configs/apollo_uni.yaml` | Universal config (`feature_dim=384`). |

---

## Data Layout

```
data/<name>/
  train/  LQ/  HQ/     raw pairs; WAV, MP3, FLAC; filenames must match
  val/    LQ/  HQ/

chunks/<name>/          auto-generated; invalidated automatically when segment_sec/align_data/fixed_delay changes
  train/  LQ/  HQ/
  val/    LQ/  HQ/

runs/<name>/
  <timestamp>/          one folder per run
    checkpoints/        .ckpt files; filename encodes val_loss for auto-selection
    logs/               TensorBoard events
    val_audio/
      step_NNNNNN/      LQ + HQ + restored triplets per val run
```

`<name>` is derived from the config filename stem. Override with `exp.name`.

---

## Commands

```bash
# Install / activate
apollo.bat              # Windows
./apollo.sh             # Linux / macOS

# Train
python train.py --conf_dir configs/apollo_uni.yaml

# Inference (auto-selects best checkpoint)
python inference.py --in_wav input.mp3 --out_wav output.wav --conf_dir configs/apollo_stfl.yaml

# Inference (explicit checkpoint)
python inference.py --in_wav input.mp3 --out_wav output.wav --weights path/to/checkpoint.ckpt --conf_dir configs/apollo_stfl.yaml

# Monitor
tensorboard --logdir ./runs
```

---

## Key Config Knobs

| Key | Notes |
|---|---|
| `exp.resume` | `true` = resume from most recent checkpoint; `false` = fresh run |
| `datas.segment_sec` | Chunk length in seconds. Changing this auto-invalidates the chunk cache and re-chunks. |
| `datas.align_data` | `true` (auto-detect delay), `false` (off), or integer (fixed sample offset: positive trims LQ, negative trims HQ) |
| `datas.num_workers` | Keep at 4 or below on 16 GB RAM |
| `datas.pin_memory` | Set `false` on 16 GB systems |
| `training.n_layers_to_freeze` | Freeze front N BSNet layers. Apollo has 6 total. |
| `training.grad_accum_steps` | Simulate larger batch without extra VRAM |
| `training.val_audio_pairs` | Total val audio chunks saved per val run |
| `system.gradient_checkpointing` | BSNet only. Large VRAM saving at ~30% compute cost. |
| `optimizer.type` | `adamw`, `adamw_8bit`, or `cpu_offload` |
| `optimizations.ram_limit_fraction` | Fraction of system RAM at which the process self-terminates cleanly. Default `0.95`. |
| `trainer.val_check_interval` | Validate every N steps |
| `trainer.limit_val_batches` | Cap val batches per run. `25` out of 50 is a good balance. |
| `checkpoint.save_top_k` | Keep top N checkpoints by val_loss. Set `5` to avoid unbounded disk growth. |

---

## Alignment (`align_data`)

Runs at chunk time for each LQ/HQ pair.

- **Fixed offset** (`align_data: 1057`): trims exactly N samples from LQ via `frame_offset`. iTunes-encoded MP3s have a consistent 1057-sample encoder delay.
- **Auto** (`align_data: true`): tries LAME header via mutagen; falls back to xcorr. Unreliable on music — use fixed offset when known.
- Negative values trim HQ instead of LQ.

---

## Val Audio Monitoring

- Saves LQ + HQ + restored triplets to `val_audio/step_NNNNNN/` every val run.
- After the first real val run, a set of dataset indices is locked and persisted in the checkpoint. On resume, the same reference chunks are used — no re-randomization.
- Val dataloader shuffles. `limit_val_batches: 25` draws a random subset each run. Locked chunks appear in the output when drawn in the current shuffle.
- Files named `<SongName>_s<start_sample>_restored/lq/hq.wav`.

---

## Augmentation Notes

| Augmentation | Notes |
|---|---|
| `stereo_alternation` | Picks L or R by sample index (even → L, odd → R). Returns `(1, samples)` for both LQ and HQ. `prob: 1.0` alternates systematically — does not randomly collapse to mono. |
| `gain` | Applied identically to LQ and HQ. Never hard-clamps — scales both down if push would clip. |
| `polarity` | Flips signal polarity on both. Practically free. |
| `noise` | Matched Gaussian noise added to both. |
| `pitch_shift` | Disabled for codec restoration — warps frequency relationships. |
| `mp3_degradation` | LQ only. Use cached mode; live spawns an ffmpeg process per chunk per epoch. |

`normalize_pair` scales by joint peak — pre-existing flat-top clipping preserved as valid training signal.

---

## Performance Notes

- `torch.cuda.empty_cache()` removed from training loop. Primary cause of sub-1 it/s on Windows/WDDM. Only called in OOM recovery now.
- Hann windows cached as plain tensors in `_hann_cache` dicts. Not `register_buffer` — avoids checkpoint bloat.
- Gradient checkpointing applied to BSNet only. Checkpointing the discriminator caused a double-forward and killed feature-matching gradients.
- `on_epoch=True` PL logging replaced with manual scalar accumulator to prevent GPU tensor caching across epochs.
- Scheduler stepping moved to `on_train_epoch_end` to prevent double-step on resume.

---

## Inference Notes

- `--conf_dir` reads `feature_dim`, `sr`, `win`, `layer`, `segment_sec` from a training yaml. Explicit CLI args always override config values.
- When `--weights` is omitted, inference scans all run folders for the checkpoint with the best val_loss in its filename.
- Input normalized by peak before inference; rescaled back after. Preserves transient headroom.
- Output is 32-bit float WAV.
- `overlap_sec` must be >= 0 and < `chunk_sec`; validated at startup.

---

## Known Constraints

- `sr=44100` required. Do not change.
- cuFFT does not support fp16 for non-power-of-two FFT sizes — STFT always casts to float32.
- Single GPU only.
- `apollo.bat` on Windows: `C:\Windows\System32\uv` stub may shadow the real uv at `%USERPROFILE%\.local\bin\uv.exe`.
- torch and torchaudio pinned to `2.1.2+cu121`.

---

## Security

- All `torch.load` calls use `weights_only=True`.
- Remote model downloads are not hash-verified. Only load `.ckpt` files from trusted sources.

---

## Version

Current: 0.2.4.7

# Apollo-mod — Agent Reference

## Project

Community fork of [JusperLee/Apollo](https://github.com/JusperLee/Apollo) — a GAN-based audio restoration model targeting codec-compressed audio. Reworked for single-GPU fine-tuning on user-supplied paired LQ/HQ datasets.

## Architecture

- `look2hear/models/apollo.py` — Generator. BSNet + Roformer layers. STFT/iSTFT always in float32.
- `look2hear/discriminators/frequencydis.py` — Frequency discriminator. Hann windows cached as plain tensors in `_hann_cache` dict (not `register_buffer` — keeps them out of checkpoint state_dict).
- `look2hear/system/audio_litmodule.py` — Lightning module. Manual optimization, gradient checkpointing, val audio saving, RAM/CUDA OOM watchdogs.
- `look2hear/losses/gan_losses.py` — GAN losses. Hann windows and HF weight tensors cached in `_hann_cache`/`_weight_cache` dicts (not buffers — no checkpoint pollution).
- `paired_datamodule.py` — Loads chunked LQ/HQ WAV pairs. Live + cached augmentation pipeline.
- `train.py` — Entry point. Auto-chunks `data/` → `chunks/` on first run, then trains. Includes run isolation (timestamped subdirs), resume logic, and alignment.
- `inference.py` — Entry point. Chunked inference, streams output to disk for mid-run preview. Normalizes input to match training amplitude range.
- `configs/apollo.yaml` — Base config (`feature_dim=256`).
- `configs/apollo_uni.yaml` — Universal config (`feature_dim=384`).

## Data Layout

```
data/<name>/
  train/  LQ/  HQ/     ← raw pairs; WAV, MP3, FLAC accepted; filenames must match
  val/    LQ/  HQ/

chunks/<name>/          ← auto-generated; invalidated automatically when segment_sec/align_data/fixed_delay changes
  train/  LQ/  HQ/
  val/    LQ/  HQ/

runs/<name>/
  <timestamp>/          ← one folder per run (fresh) or resumed run
    checkpoints/        ← Lightning .ckpt files; sorted by val_loss in filename
    logs/               ← TensorBoard events
    val_audio/
      step_NNNNNN/      ← LQ + HQ + restored triplets, one folder per val run
```

`<name>` is derived from the config filename stem (e.g. `apollo_stfl2.yaml` → `apollo_stfl2`). Override with `exp.name`. Loose files dropped in `data/` root with `_LQ`/`_HQ` postfixes are auto-moved to `data/<name>/` on train start.

## Commands

```bash
# Install / activate
apollo.bat              # Windows (uses uv; real uv.exe may be at %USERPROFILE%\.local\bin\uv.exe if PATH stub interferes)

# Train
python train.py --conf_dir configs/apollo_uni.yaml

# Inference
python inference.py --in_wav input.mp3 --out_wav output.wav \
    --weights models/SFTL.ckpt --conf_dir configs/SFTL.yaml

# Monitor
tensorboard --logdir ./runs
```

## Key Config Knobs

- `exp.name` — dataset/run namespace; derived from config filename if null
- `exp.dir` — run output root; default `./runs`
- `resume` — `true` = pick up from most recent checkpoint in `runs/<name>/`; `false` = fresh timestamped run
- `datas.segment_sec` — chunk length in seconds; changing requires deleting `chunks/`
- `datas.align_data` — `true` (auto-detect delay), `false` (off), or integer (fixed sample offset: positive trims LQ, negative trims HQ)
- `datas.num_workers` — DataLoader workers; keep ≤4 on 16 GB RAM to avoid progressive memory pressure
- `datas.pin_memory` — set `false` on 16 GB; pinned memory can't be swapped and causes OS crashes
- `training.n_layers_to_freeze` — freeze front BSNet layers to reduce catastrophic forgetting
- `training.grad_accum_steps` — simulate larger batch without extra VRAM
- `training.val_audio_pairs` — val audio samples per song saved each val run
- `system.gradient_checkpointing` — large VRAM saving at ~30% compute cost
- `optimizer.type` — `adamw`, `adamw_8bit`, or `cpu_offload`
- `optimizations.ram_limit_fraction` — fraction of system RAM at which the process self-terminates cleanly (default 0.90; set 0.98+ on 16 GB if idle RAM is already high)
- `trainer.val_check_interval` — validate every N steps; also controls val audio save frequency
- `trainer.limit_val_batches` — cap val batches per run to keep overhead low (e.g. 50)

## Alignment (`align_data`)

Runs at chunk time for each LQ/HQ pair.

- **Fixed offset** (`align_data: 1057`): trims exactly N samples from LQ at decode time via `frame_offset` — fast, no decode overhead. iTunes-encoded MP3s have a consistent 1057-sample encoder delay.
- **Auto** (`align_data: true`): tries LAME header via mutagen; falls back to 2048-sample pattern xcorr in an 8192-sample search window. Unreliable on music — use fixed offset when known.
- Negative values trim HQ instead of LQ.

## Run Isolation and Resume

Each fresh run creates `runs/<name>/<timestamp>/`. On resume (`resume: true`), train.py finds the most recently modified timestamped subfolder with a `checkpoints/` dir and picks the newest `.ckpt` inside it. Ctrl+C saves a checkpoint named by step/loss into `checkpoints/` before exiting — picked up automatically on next resume.

## Val Audio

- Saves LQ + HQ + restored triplets to `runs/<name>/<timestamp>/val_audio/step_NNNNNN/` every val run.
- `val_audio_pairs` controls total chunks saved (not per-song); the selection guarantees at least one chunk per song where possible.
- Val dataloader shuffles — `limit_val_batches` (e.g. 25 of 50) draws a random subset each run. Set to roughly half the val set for good song coverage without running everything.
- **Index locking**: after the first real val run, a set of dataset indices is locked and persisted in the checkpoint. On resume, the same chunks are monitored — no re-randomization. Locked indices only appear in the audio output when they're drawn in the current shuffled val batch.
- Files named `<SongName>_s<start_sample>_restored/lq/hq.wav`.

## Performance Notes

- **`torch.cuda.empty_cache()` in training loop**: removed. Was the primary cause of <1 it/s on Windows/WDDM. Only called in OOM recovery path now.
- **Hann window allocation**: previously created 7+ GPU tensors per discriminator forward call. Now cached in `_hann_cache` dicts (plain tensors, not `register_buffer` — avoids checkpoint bloat). Same fix in `gan_losses.py` for HF weight tensors.
- **Gradient checkpointing**: applied to BSNet layers only. Previously also checkpointed `FrequencyDiscriminator`, which caused a double-forward and killed feature-matching gradients. Discriminator is not checkpointed.
- **`on_epoch=True` logging**: accumulates GPU tensors until epoch end (PL issue #4556). Replaced with manual `_val_loss_sum / _val_loss_count` scalar accumulator.
- **Scheduler stepping**: moved from `is_last_batch` guard inside `training_step` to `on_train_epoch_end`. Prevents double-step on resume which was causing the loss jump.
- **Train DataLoader**: `persistent_workers=True` when `num_workers > 0`, `prefetch_factor=2`. Workers survive across epochs, prefetch overlaps disk I/O with GPU compute.
- **Val DataLoader `num_workers=0`**: val set is small; worker spawn overhead exceeded I/O savings.

## Augmentation Notes

- `mono_channel`: Apollo processes one channel at a time. This augmentation picks L or R **deterministically by sample index** (even index → L, odd → R), returning a `(1, samples)` tensor for both LQ and HQ. With `prob: 1.0` every sample is single-channel, alternating L/R systematically across the dataset each epoch. This is intentional — it gives the model balanced exposure to both stereo channels without randomly clumping. Do not interpret `prob: 1.0` as "always collapses to mono" — it alternates.
- Gain and noise never hard-clamp. If a push would clip, both signals scale down together, preserving any pre-existing flat-top clipping as valid training signal.
- `normalize_pair` scales by joint peak — clipped source files handled correctly.
- MP3 degradation (LQ only): use cached, not live — live spawns an ffmpeg process per chunk per epoch.
- Pitch shift: turn off for codec restoration — warps the frequency relationships the model is trying to learn.

## Inference Notes

- `--conf_dir` reads `feature_dim`, `sr`, `win`, `layer`, `segment_sec` from a training yaml. Explicit CLI args always win over config values.
- Output written chunk-by-chunk (soundfile append mode) — drag into Audacity to preview mid-run.
- Input normalized by peak before inference; rescaled back after. Preserves transient headroom — clipped-looking MP3 transients are restored relative to the training distribution.
- Output is 32-bit float WAV — no integer ceiling, no clipping at ±1.0.
- MP3/FLAC input accepted.
- `overlap_sec` must be ≥ 0 and < `chunk_sec`; validated at startup with a clear error.

## Known Constraints

- `sr=44100` required. Do not change.
- cuFFT does not support fp16 for non-power-of-two FFT sizes — STFT always casts to float32.
- Single GPU only — `sync_dist` and `all_gather` removed.
- `apollo.bat` on Windows: `C:\Windows\System32\uv` stub may shadow the real uv.
- torch and torchaudio pinned to `2.1.2+cu121` — pytorch-lightning will otherwise silently upgrade torch.
- If `.git` is lost after moving the project: `git init && git remote add origin <url> && git fetch && git read-tree HEAD && git checkout-index -a -f`. Use `git show HEAD:<file> > <file>` for individual files.

## Chunk Cache Invalidation

A `.manifest.json` is written to `chunks/<name>/` after each successful chunk run. It records `segment_sec`, `align_data`, and `fixed_delay`. On next train start, if the manifest doesn't match the current config, the chunk directory is deleted and re-chunked automatically. Delete the manifest manually to force a full re-chunk.

## Security

- All `torch.load` calls use `weights_only=True`. Checkpoints must deserialize to a plain dict — arbitrary pickle execution is rejected.
- Remote model downloads (pretrained weights) are not hash-verified. Only load `.ckpt` files from trusted sources.

## Version

Current: 0.2.4.0

# Apollo-mod — Agent Reference

## Project

Community fork of [JusperLee/Apollo](https://github.com/JusperLee/Apollo) — a GAN-based audio restoration model targeting codec-compressed audio. Reworked for single-GPU fine-tuning on user-supplied paired LQ/HQ datasets.

## Architecture

- `look2hear/models/apollo.py` — Generator. BSNet + Roformer layers. STFT/iSTFT always in float32.
- `look2hear/discriminators/frequencydis.py` — Frequency discriminator. Hann windows pre-registered as buffers (not allocated per forward).
- `look2hear/system/audio_litmodule.py` — Lightning module. Manual optimization, gradient checkpointing, val audio saving, RAM/CUDA OOM watchdogs.
- `look2hear/losses/gan_losses.py` — GAN losses. HF boost and hann windows registered as buffers, not allocated per forward.
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

chunks/<name>/          ← auto-generated; delete to force re-chunk
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
- `val_audio_pairs` is per-song: `3` gives 3 chunks from each val song.
- Index is locked after the first real val run (skips sanity check) and re-locked on every resume to get fresh song coverage.
- Files named `<SongName>_s<start_sample>_restored/lq/hq.wav`.

## Performance Notes

- **Discriminator forward**: hann windows are pre-registered as buffers — creating them in `forward()` was causing progressive CUDA allocator slowdown. Same fix in `gan_losses.py`.
- **Gradient checkpointing discriminator**: previously ran `FrequencyDiscriminator.forward` twice per call (once for output, once for hiddens). Fixed to return both in a single checkpoint pass — halved discriminator cost.
- **`on_epoch=True` logging**: using this for `val_loss` accumulates GPU tensors until epoch end. Replaced with a manual `_val_loss_sum / _val_loss_count` accumulator that resets per val run.
- **`persistent_workers=False`** on training DataLoader: persistent workers on Windows gradually grow their memory footprint, causing progressive slowdown. Workers now restart each epoch.
- **Val DataLoader `num_workers=0`**: val loads are small and fast enough on the main process; worker overhead exceeded savings.

## Augmentation Notes

- Gain and noise never hard-clamp. If a push would clip, both signals scale down together, preserving any pre-existing flat-top clipping as valid training signal.
- `normalize_pair` scales by joint peak — clipped source files handled correctly.
- MP3 degradation (LQ only): use cached, not live — live spawns an ffmpeg process per chunk per epoch.
- Pitch shift: turn off for codec restoration — warps the frequency relationships the model is trying to learn.

## Inference Notes

- `--conf_dir` reads `feature_dim`, `sr`, `win`, `layer`, `segment_sec` from a training yaml.
- Output written chunk-by-chunk (soundfile append mode) — drag into Audacity to preview mid-run.
- Input normalized to training amplitude range before inference; rescaled back after.
- Output is 32-bit float WAV to avoid integer clipping at ±1.0.
- MP3/FLAC input accepted.

## Known Constraints

- `sr=44100` required. Do not change.
- cuFFT does not support fp16 for non-power-of-two FFT sizes — STFT always casts to float32.
- Single GPU only — `sync_dist` and `all_gather` removed.
- `apollo.bat` on Windows: `C:\Windows\System32\uv` stub may shadow the real uv.
- torch and torchaudio pinned to `2.1.2+cu121` — pytorch-lightning will otherwise silently upgrade torch.
- If `.git` is lost after moving the project: `git init && git remote add origin <url> && git fetch && git read-tree HEAD && git checkout-index -a -f`. Use `git show HEAD:<file> > <file>` for individual files.

## Version

Current: 0.1.8.2

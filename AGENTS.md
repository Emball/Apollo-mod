# Apollo-mod — Agent Reference

## Project

Community fork of [JusperLee/Apollo](https://github.com/JusperLee/Apollo) — a GAN-based audio restoration model targeting codec-compressed audio. Reworked for single-GPU fine-tuning on user-supplied paired LQ/HQ datasets.

## Architecture

- `look2hear/models/apollo.py` — Generator. BSNet + Roformer layers. STFT/iSTFT always in float32.
- `look2hear/discriminators/frequencydis.py` — Frequency discriminator.
- `look2hear/system/audio_litmodule.py` — Lightning module. Train/val steps, gradient checkpointing, val audio saving.
- `look2hear/losses/` — GAN losses + matrix loss.
- `paired_datamodule.py` — Loads chunked LQ/HQ WAV pairs. Live + cached augmentation pipeline.
- `train.py` — Entry point. Auto-chunks `data/` → `chunks/` on first run, then trains.
- `inference.py` — Entry point. Chunked inference, streams output to disk for mid-run preview.
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

runs/<name>/            ← checkpoints, val audio, logs
```

`<name>` is derived from the config filename stem (e.g. `apollo_stfl2.yaml` → `apollo_stfl2`). Override with `exp.name` in the config. Loose files dropped in `data/` root with `_LQ`/`_HQ` postfixes are auto-moved to `data/<name>/` on train start.

## Commands

```bash
# Install / activate
apollo.bat              # Windows (uses uv; real uv.exe may be at %USERPROFILE%\.local\bin\uv.exe if PATH stub interferes)

# Train
python train.py --conf_dir configs/apollo_uni.yaml

# Inference
python inference.py --in_wav input.wav --out_wav output.wav --weights lew_v2
python inference.py --in_wav input.wav --out_wav output.wav \
    --weights models/SFTL.ckpt --conf_dir configs/SFTL.yaml

# Monitor
tensorboard --logdir ./runs
```

## Key Config Knobs

- `exp.name` — dataset/run namespace; derived from config filename if null
- `exp.dir` — run output root; default `./runs`
- `datas.segment_sec` — chunk length in seconds; changing requires deleting `chunks/`
- `datas.align_data` — `true` (auto-detect delay), `false` (off), or integer (fixed sample offset: positive trims LQ, negative trims HQ)
- `training.n_layers_to_freeze` — freeze front BSNet layers to reduce catastrophic forgetting
- `training.grad_accum_steps` — simulate larger batch without extra VRAM
- `system.gradient_checkpointing` — large VRAM saving at ~30% compute cost
- `optimizer.type` — `adamw`, `adamw_8bit`, or `cpu_offload`
- `augmentation.live` / `augmentation.cached` — per-epoch vs baked-at-chunk-time augmentation blocks

## Alignment (`align_data`)

`_align_pair` in `train.py` runs at chunk time for each LQ/HQ pair, before slicing.

- **Fixed offset** (`align_data: 1057`): trims exactly N samples from LQ (or HQ if negative). Use this for iTunes-encoded MP3s — encoder delay is a consistent 1057 samples.
- **Auto** (`align_data: true`): tries to read the LAME header delay via mutagen; falls back to a small-window pattern xcorr (2048-sample pattern, 8192-sample search window). The xcorr fallback is unreliable on music — use fixed offset when you know it.
- Alignment runs for both train and val splits.

## Augmentation Notes

- Gain and noise augmentation never hard-clamp. If a gain push would clip, both LQ and HQ are scaled down together to just below 0 dBFS, preserving any pre-existing flat-top clipping distortion as valid training signal.
- MP3 degradation applies to LQ only at chunk load time.
- `normalize_pair` scales by joint peak of both signals — clipped source files are handled correctly without re-clipping.

## Inference Notes

- `--conf_dir` reads `feature_dim`, `sr`, `win`, `layer`, `segment_sec` from a training yaml. Explicit CLI flags override.
- Output is written chunk-by-chunk to disk (soundfile append mode) — drag the output file into Audacity during inference to preview.
- Default chunk size matches training `segment_sec` (4s). The original 30s default caused OOM on long files.
- MP3/FLAC input accepted at inference; WAV-only restriction is training-only (chunking pipeline).

## Known Constraints

- `sr=44100` is required. Do not change.
- cuFFT does not support fp16 for non-power-of-two FFT sizes — STFT always casts to float32.
- Single GPU only — `sync_dist` and `all_gather` removed.
- `apollo.bat` on Windows: `C:\Windows\System32\uv` stub may shadow the real uv. Real installs at `%USERPROFILE%\.local\bin\uv.exe` or `%LOCALAPPDATA%\uv\uv.exe`.
- torch and torchaudio pinned to `2.1.2+cu121` in `requirements.txt` — pytorch-lightning will otherwise silently upgrade torch and break torchaudio.
- If the `.git` folder is lost (e.g. from moving the project), use `git init && git remote add origin <url> && git fetch && git read-tree HEAD && git checkout-index -a -f` to reconnect without touching untracked files. `git show HEAD:<file> > <file>` is the reliable fallback for individual files.

## Version

Current: 0.1.4.7

# Apollo-mod — Agent Reference

Read `README.md` for usage, config reference, data layout, commands, and augmentation options. This file covers internal architecture, constraints, and non-obvious behavior that isn't user-facing.

---

## Architecture

| File | Role |
|---|---|
| `look2hear/models/apollo.py` | Generator. BSNet + Roformer layers. STFT/iSTFT always in float32. |
| `look2hear/discriminators/frequencydis.py` | Frequency discriminator. Hann windows cached as plain tensors in `_hann_cache` (not `register_buffer` — keeps them out of checkpoint state_dict). Mono input auto-expanded to stereo via `.expand()`. |
| `look2hear/system/audio_litmodule.py` | Lightning module. Manual optimization, gradient checkpointing (BSNet only), val audio saving, RAM/CUDA OOM watchdogs, val index locking. |
| `look2hear/losses/gan_losses.py` | GAN losses. Hann windows and HF weight tensors in `_hann_cache`/`_weight_cache` dicts (not buffers). |
| `paired_datamodule.py` | Loads chunked LQ/HQ WAV pairs. Live + cached augmentation pipeline. Val dataloader shuffles; dataset index passed through batch for stable audio monitoring. |
| `train.py` | Entry point. Auto-chunks `data/` on first run. Run isolation via timestamped subdirs, resume logic, chunk cache manifest. |
| `inference.py` | Entry point. Chunked inference, streams output to disk. Auto-selects best checkpoint by val_loss when `--weights` is omitted. |
| `configs/apollo.yaml` | Base config (`feature_dim=256`). |
| `configs/apollo_uni.yaml` | Universal config (`feature_dim=384`). |

---

## Internal Behavior

**Chunk cache:** A `.manifest.json` is written after each successful chunk run recording `segment_sec`, `align_data`, and `fixed_delay`. On train start, if the manifest doesn't match the current config the chunk dir is deleted and re-chunked automatically.

**Run isolation:** Each fresh run creates `runs/<name>/<timestamp>/`. Resume finds the most recently modified timestamped subfolder with a `checkpoints/` dir and picks the newest `.ckpt` inside.

**Val index locking:** After the first real val run, a set of dataset indices is locked and persisted in the checkpoint. On resume, the same reference chunks are used. Val dataloader shuffles each run — locked chunks only appear in audio output when drawn in the current batch.

**Alignment:** Runs at chunk time. Fixed offset (`align_data: 1057`) trims via `frame_offset` — no decode overhead. Auto mode tries LAME header via mutagen, falls back to xcorr — unreliable on music. Negative values trim HQ instead of LQ.

**Augmentation internals:** `normalize_pair` scales by joint peak — pre-existing flat-top clipping is preserved as valid training signal, not clamped away. `stereo_alternation` picks L or R by sample index (even → L, odd → R) — `prob: 1.0` alternates systematically, does not randomly collapse to mono.

**Inference normalization:** Input normalized by peak before chunked inference, rescaled back after. Preserves transient headroom — hot MP3 transients above 1.0 at decode time are handled correctly.

---

## Performance Notes

- `torch.cuda.empty_cache()` removed from training loop. Was the primary cause of sub-1 it/s on Windows/WDDM. Only called in OOM recovery now.
- Hann windows in `_hann_cache` dicts — not `register_buffer`. Avoids checkpoint bloat and the key-stripping dance on load.
- Gradient checkpointing on BSNet only. Checkpointing the discriminator caused a double-forward and dead feature-matching gradients.
- `on_epoch=True` PL logging replaced with manual scalar accumulator to prevent GPU tensor accumulation across epochs (PL issue #4556).
- Scheduler stepping moved to `on_train_epoch_end` — prevents double-step on resume which caused a visible loss jump.

---

## Known Constraints

- `sr=44100` required. Do not change.
- cuFFT does not support fp16 for non-power-of-two FFT sizes — STFT always casts to float32.
- Single GPU only — `sync_dist` and `all_gather` removed.
- `apollo.bat` on Windows: `C:\Windows\System32\uv` stub may shadow the real uv at `%USERPROFILE%\.local\bin\uv.exe`.
- torch and torchaudio pinned to `2.1.2+cu121`.
- All `torch.load` calls use `weights_only=True`. Remote downloads are not hash-verified — only load `.ckpt` files from trusted sources.

---

## Version

Current: 0.2.4.9

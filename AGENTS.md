# Apollo-mod — Agent Reference

Read `README.md` for usage, config reference, data layout, commands, and augmentation options. This file covers internal architecture, constraints, and non-obvious behavior that isn't user-facing.

The GitHub wiki (`Changes-and-improvements` page, repo `Emball/Apollo-mod.wiki.git`) documents every meaningful change made in this fork relative to upstream `JusperLee/Apollo`, with rationale. Sync it alongside this file when changes to documented behavior land.

---

## Architecture

| File | Role |
|---|---|
| `look2hear/models/apollo.py` | Generator. BSNet + Roformer layers. STFT/iSTFT always in float32. |
| `look2hear/discriminators/frequencydis.py` | Frequency discriminator. Hann windows cached as plain tensors in `_hann_cache` (not `register_buffer` -- keeps them out of checkpoint state_dict). Mono input auto-expanded to stereo via `.expand()`. `window_weight_boost` config param (default `false`) enables inverse-size window weighting -- only useful for severely degraded sources, causes HF artifact reproduction on mildly degraded audio. |
| `look2hear/system/audio_litmodule.py` | Lightning module. Manual optimization, gradient checkpointing (BSNet only), val audio saving, RAM/CUDA OOM watchdogs, val index locking, rotation schedule, background write thread. |
| `look2hear/losses/gan_losses.py` | GAN losses. Hann windows and weight tensors in `_hann_cache`/`_weight_cache` dicts (not buffers). Gaussian band weight curve replaces the old flat `hf_boost` step function -- peaks at `band_weight_center_hz` with `band_weight_sigma_hz` width. Set `band_weight_gain: 0` for flat loss. `hf_band_mae()` exposed as standalone function for val metrics. |
| `paired_datamodule.py` | Loads chunked LQ/HQ WAV pairs. Live + cached augmentation pipeline. Val dataloader shuffles; dataset index passed through batch for stable audio monitoring. |
| `train.py` | Entry point. FFmpeg-based in-place source conversion, auto-chunks on first run, run isolation, resume logic, chunk cache manifest. `RankBadger` callback renames checkpoints with `[rank]` prefix after each save. Baseline val pass runs before training on fresh runs. |
| `inference.py` | Entry point. Chunked inference, streams output to disk. Auto-selects latest checkpoint when `--weights` is omitted. Spectral merge engine (`_spectral_merge`) blends original and enhanced in the STFT domain (4096-point Hann, OLA) per user-defined frequency bands. Modes: `max_fft`, `min_fft`, `avg`, `original`, `enhanced`. `--low_end_preserve` applies `max_fft` below `--low_end_hz` (default 700 Hz). `--ensemble` accepts a JSON list of band specs for full control. `--aux_weights` / `--aux_conf_dir` / `--aux_ensemble` add a second checkpoint blended at specified bands. Phase always comes from the primary enhanced output; only magnitude is blended. |
| `tui.py` | Keyboard-navigated TUI (Rich). Primary interface, launched by `apollo.bat`/`apollo.sh`. Shows Latest and Best checkpoint options separately in inference. Ctrl+C during training saves a checkpoint and returns to menu. Inference screen includes an ensemble picker (`_pick_ensemble`) between output selection and run -- offers no ensemble, low-end preserve, low-end + transition blend preset, and custom JSON input. Selected ensemble flags are forwarded to inference.py. |
| `evaluate.py` | Offline checkpoint evaluator. Launched from TUI (Evaluate screen) or standalone (`python evaluate.py --conf_dir ...`). Reads metrics already encoded in checkpoint filenames; only runs inference for missing ones. Adds SDR and optional VISQOL (requires `pip install pyvisqol`). VISQOL scores cached in `<ckpt_dir>/.eval_cache.json`. Ranking weights: visqol=0.40, hfmae=0.25, msstft=0.20, sfr=0.10, sdr=0.05 -- separate from RankBadger's training-time weights. |
| `configs/apollo.yaml` | Base config (`feature_dim=256`). |
| `configs/apollo_uni.yaml` | Universal config (`feature_dim=384`). |
| `degrade_audio.bat` | Windows batch script for synthetic degradation pipeline. Single file or `/bulk` folder mode. Chain: WMA 128k → LAME q5 → FhG 192k → LAME q5 → FhG 192k → LAME q2 → FhG 192k (10 steps total). Requires FFmpeg on PATH and `acmenc.exe` at `C:\Portable\acmenc\`. Output written to `<outdir>\<basename>_degraded.mp3`; temp files cleaned on success. |

---

## Internal Behavior

**Chunk cache:** A `.manifest.json` is written after each successful chunk run recording `segment_sec`, `overlap`, and `fixed_delay`. On train start, if the manifest doesn't match the current config the chunk dir is deleted and re-chunked automatically.

**Source conversion:** On chunk prep, any non-WAV source files (MP3, FLAC, etc.) are converted in-place to 32-bit float WAV via FFmpeg in parallel threads. The alignment trim (`align_data`) is baked in during this pass via FFmpeg's `atrim` filter. Already-WAV files are never re-converted. Conversion is skipped entirely if all sources are already WAV.

**Run isolation:** Each fresh run creates `runs/<name>/<timestamp>/`. Resume finds the most recently modified timestamped subfolder with a `checkpoints/` dir and picks the newest `.ckpt` by mtime.

**Val fixed evaluation set:** On the first real val run, `_lock_val_fixed_indices` groups all seen dataset indices by song, then samples `limit_val_batches / num_songs` chunks from each song (stratified). The result is locked into `_val_fixed_indices` -- all subsequent val runs skip any chunk not in this set, so the loss is always computed on the exact same fixed chunks. Both `_val_fixed_indices` and the rotation schedule are persisted in the checkpoint and restored on resume.

**Val audio rotation:** Saves exactly `val_songs` songs × 3 files (LQ/HQ/Restored) = N×3 files per val run. At training start a rotation schedule is computed so that every val song gets equal coverage by end of training. Each song's chunk is picked once at schedule-build time (`_lock_val_refs`) -- one specific non-silent chunk, never re-picked -- and stored in `_val_locked_refs`, so the same audio is always compared across val steps for a given song. `val_rotate_every: auto` derives the cadence from total configured steps; an integer overrides it manually. The schedule and locked refs are checkpointed and resume-stable. File writes run in a background thread so training resumes immediately. Old configs using `val_audio_pairs` still work via fallback.

**Val perceptual metrics:** Five metrics computed after each val run: `sisdr` (legacy, used for checkpoint selection and early stopping), `msstft` (multi-scale log-STFT loss, 3 window sizes), `sfr` (spectral flatness ratio 8-22kHz -- rising above 1.05 is an early overfitting signal), `hf_band_mae` (mean absolute log-magnitude error in the 13-19kHz transition band -- primary signal for MP3 rolloff fine-tuning), `sdr` (Signal-to-Distortion Ratio -- cheap signal, less noisy than sisdr). All five are logged to TensorBoard. The background write thread is joined in `on_validation_epoch_end` before Lightning reads metrics for checkpoint naming, so all appear in filenames.

**Checkpoint filenames:** Format is `[rank]-step={step:06d}-{val_loss:.3f}-{val_msstft:.4f}-{val_sfr:.3f}-{val_hfmae:.4f}-{val_sdr:.3f}.ckpt`. `[rank]` is prepended by the `RankBadger` callback after each save. Rank is a weighted composite (msstft=0.40, hfmae=0.30, sfr=0.15, sisdr=0.10, sdr=0.05) of min-max normalized metrics -- Rank 1 = best. All checkpoints are kept (`save_top_k=-1` enforced in code). `evaluate.py` uses a separate weighting that includes VISQOL (visqol=0.40, hfmae=0.25, msstft=0.20, sfr=0.10, sdr=0.05).

**Baseline eval:** On fresh runs (no resume), `trainer.validate()` is called on the pretrained weights before `trainer.fit()`. Prints `[baseline] sisdr=XX.XXX` so improvement is immediately visible against the starting point.

**StepPrinter:** TQDM is disabled. Prints one line per optimizer step. `it/s` counts every batch (including accumulation batches) for consistency with pre-accumulation baselines. Val time is excluded from the rate. Last val metrics are shown inline on every training line once available, ordered `msstft, sfr, hfmae, sisdr` (perceptual signals first, sisdr last since it's the noisiest and least perceptually meaningful).

**Band weight:** `MultiFrequencyGenLoss` applies a penalty curve over STFT bins, shape controlled by `band_weight_shape`: `"gaussian"` (default, raised curve peaking at `band_weight_center_hz` with width `band_weight_sigma_hz`) or `"trapezoid"` (flat-topped between `band_weight_lo_hz`/`band_weight_hi_hz` with `band_weight_ramp_hz` soft edges -- useful for targeting a specific rolloff/transition band). `band_weight_gain=0` is perfectly flat regardless of shape. Replaces the old `hf_boost` + `hf_threshold_ratio` step function. Config keys `hf_boost` and `hf_threshold_ratio` are accepted for back-compat but ignored.

**Alignment:** `align_data` accepts an integer offset in samples -- positive trims LQ, negative trims HQ. Baked into WAV files at conversion time via FFmpeg, never applied at chunk or training time. iTunes-encoded MP3s have a consistent 1057-sample encoder delay. Set `false` to disable.

**Augmentation internals:** `normalize_pair` scales by joint peak -- pre-existing flat-top clipping is preserved as valid training signal. `stereo_alternation` picks L or R by sample index (even → L, odd → R) -- `prob: 1.0` alternates systematically.

**Inference normalization:** Input normalized by peak before chunked inference, rescaled back after. Preserves transient headroom.

**CRITICAL -- never call `torch.cuda.empty_cache()` in any training or validation hook.** Only acceptable inside an OOM recovery handler (`except torch.cuda.OutOfMemoryError`).

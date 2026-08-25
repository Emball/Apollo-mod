# Apollo-mod — Agent Reference

Read `README.md` for usage, config reference, data layout, commands, and augmentation options. This file covers internal architecture, constraints, and non-obvious behavior that isn't user-facing.

## Coding Rules

Unless explicitly instructed to make a change on a specific branch only, all changes must be applied to both `main` and `diagnostic/revert-training-step`.

## README Editing Guidelines

Write from the user's perspective. If the user can't act on the information and it doesn't change what they do, leave it out.

What does not belong in the README:
- Implementation details: what library, what algorithm, how it's parallelized, internal mechanics
- Project-specific config values presented as universal defaults (e.g. a hardcoded encoder delay for one dataset)
- Advice sections, lessons learned, or "choosing your config" guidance -- that belongs in the wiki
- Anything that flatters the implementation ("fast", "automatically", "seamlessly", "cleanly")
- M-dashes

Do not add, expand, or rewrite README sections without a specific instruction to do so. When adding new content, match the tone and density of the surrounding text exactly -- do not editorialize.

The GitHub wiki (`Changes-and-improvements` page, repo `Emball/Apollo-mod.wiki.git`) documents every meaningful change made in this fork relative to upstream `JusperLee/Apollo`, with rationale. Sync it alongside this file when changes to documented behavior land.

---

## Architecture

## Folder Structure

```
apollo.bat / apollo.sh   -- launchers (install deps, start TUI)
core/                    -- training/inference engine
  train.py               -- training entry point
  inference.py           -- inference entry point
  evaluate.py            -- offline checkpoint evaluator
  paired_datamodule.py   -- LQ/HQ data loading
  look2hear/             -- model, discriminator, losses, metrics, system
utils/                   -- TUI and tools
  tui.py                 -- keyboard-navigated launcher (primary interface)
  degrade_audio.py       -- synthetic degradation pipeline
  degrade/               -- degradation JSON configs
configs/                 -- training YAML configs (apollo.yaml, apollo_uni.yaml)
dev/                     -- internal docs and dev-only configs (stfl, stfl-og, stfl2)
  AGENTS.md
  apollo_stfl.yaml       -- active stfl run config (diagnostic branch)
  apollo_stfl-og.yaml    -- original stfl config (reference)
  apollo_stfl2.yaml      -- stfl2 experimental config
data/                    -- training source audio (LQ/ + HQ/ pairs per split)
chunks/                  -- legacy placeholder (unused; chunks live in usr/cache/chunks/)
models/                  -- pretrained / downloaded weights
runs/                    -- training run output (checkpoints, logs)
input/ output/           -- inference I/O staging dirs
```

## Architecture

| File | Role |
|---|---|
| `core/look2hear/models/apollo.py` | Generator. BSNet + Roformer layers. STFT/iSTFT always in float32. |
| `core/look2hear/discriminators/frequencydis.py` | Frequency discriminator. Hann windows cached as plain tensors in `_hann_cache` (not `register_buffer` — keeps them out of checkpoint state_dict). Mono input auto-expanded to stereo via `.expand()`. `window_weight_boost` config param (default `false`) enables inverse-size window weighting — only useful for severely degraded sources, causes HF artifact reproduction on mildly degraded audio. |
| `core/look2hear/system/audio_litmodule.py` | Lightning module. Manual optimization, gradient checkpointing (BSNet only), val audio saving, RAM/CUDA OOM watchdogs, val index locking, rotation schedule, background write thread. Training step follows original two-pass structure (discriminator first, fresh feature maps for generator loss). TF32 disabled. AMP handled by Lightning's built-in precision plugin, not manual autocast. |
| `core/look2hear/losses/gan_losses.py` | GAN losses. Hann windows and weight tensors in `_hann_cache`/`_weight_cache` dicts (not buffers). Gaussian band weight curve replaces the old flat `hf_boost` step function — peaks at `band_weight_center_hz` with `band_weight_sigma_hz` width. Set `band_weight_gain: 0` for flat loss. `hf_band_mae()` exposed as standalone function for val metrics. |
| `core/paired_datamodule.py` | Loads chunked LQ/HQ WAV pairs. Live + cached augmentation pipeline. Val dataloader shuffles; dataset index passed through batch for stable audio monitoring. |
| `core/train.py` | Entry point. Source files decoded to `usr/cache/<md5>.wav` (FFmpeg, parallel, skips if already cached). Chunks written to `usr/cache/chunks/<key>/<split>/` keyed on source md5s + chunk params + aug config — shared across configs that use the same dataset. `cfg.datas.train_dir`/`eval_dir` are overwritten with the resolved absolute cache paths after `prepare_data()`. Pretrained checkpoint selection is `feature_dim`-aware: `feature_dim=256` → `apollo_model.ckpt` / `pytorch_model.bin`; `feature_dim=384` → `apollo_model_uni.ckpt` / `pytorch_model_uni.bin`. Auto-scan checks `models/` first in the correct dim-matched order, falls back to HuggingFace hub with the matching file. Override with `weights_path` in the config. `RankBadger` callback renames checkpoints with `[rank]` prefix after each save. Baseline val pass runs before training on fresh runs. Checkpoint and early stopping monitor `val_composite` (weighted perceptual composite), not `val_loss`. All paths resolved from `_REPO_ROOT` (two levels up from `core/`). |
| `core/inference.py` | Entry point. Chunked inference, streams output to disk. Auto-selects latest checkpoint when `--weights` is omitted. `_ensure_wav` converts any non-WAV input to 32-bit float WAV in-place via FFmpeg before loading — eliminates encoder-delay mismatches between tools. `_spectral_merge` blends original and enhanced in the STFT domain (4096-point Hann, OLA) per user-defined frequency bands. Chunks shorter than `n_fft` are zero-padded before STFT and trimmed back after iSTFT — prevents crash on short tail chunks. Modes: `max_fft`, `min_fft`, `avg`, `original`, `enhanced`. `--low_end_preserve` applies `max_fft` below `--low_end_hz` (default 700 Hz). `--ensemble` accepts a JSON list of band specs for full control. `--aux_weights` / `--aux_conf_dir` / `--aux_ensemble` add a second checkpoint blended at specified bands. Phase always comes from the primary enhanced output; only magnitude is blended. |
| `utils/tui.py` | Keyboard-navigated TUI (Rich). Primary interface, launched by `apollo.bat`/`apollo.sh`. Shows Latest and Best checkpoint options separately in inference. Ctrl+C during training saves a checkpoint and returns to menu. Inference screen includes an ensemble picker (`_pick_ensemble`) between output selection and run — offers no ensemble, low-end preserve, low-end + transition blend preset, and custom JSON input. Selected ensemble flags are forwarded to inference.py. Utilities screen includes "Update Apollo" — runs `git pull --ff-only`, then relaunches itself in place via `os.execv`. `ROOT` resolves to repo root (parent of `utils/`); `core/` is added to `sys.path` for lazy imports. |
| `core/evaluate.py` | Offline checkpoint evaluator. Launched from TUI (Evaluate screen) or standalone (`python core/evaluate.py --conf_dir ...`). Reads metrics already encoded in checkpoint filenames; only runs inference for missing ones. Adds SDR and optional VISQOL (requires `pip install pyvisqol`). VISQOL scores cached in `<ckpt_dir>/.eval_cache.json`. Ranking weights: visqol=0.40, hfmae=0.25, msstft=0.20, sfr=0.10, sdr=0.05 — separate from RankBadger's training-time weights. |
| `configs/apollo.yaml` | Base config (`feature_dim=256`). |
| `configs/apollo_uni.yaml` | Universal config (`feature_dim=384`). |
| `utils/align_audio.py` | LQ/HQ temporal alignment tool. Global sinc resample corrects end-to-end speed drift (PPM), then chunked cross-correlation micro-aligns with Hann crossfade. Low-confidence chunks are interpolated from neighbours. Outputs peak-normalized 24-bit WAV. Launched from TUI Utilities ("Align audio") or standalone (`python utils/align_audio.py --hq ... --lq ... --out ...`). Requires `scipy`. |
| `utils/degrade_audio.py` | Synthetic degradation pipeline. Chain of codec/filter steps defined in a JSON config under `utils/degrade/` -- step types: `wma_encode`, `mp3_lame`, `lowpass`, `highpass` (all ffmpeg, shippable with no setup), and `mp3_fhg` (Fraunhofer IIS via `acmenc` -- requires `external_codecs.acmenc` set to a local `acmenc.exe` path; acmenc is a custom install, not shipped). Compressed steps auto-decode to WAV before the next step; the config only lists degradation passes, not the plumbing. Single file or `--bulk` folder mode. `utils/degrade/default.json` is ffmpeg-only (WMA 128k → LAME q5 → LAME q2); `utils/degrade/fhg_original.json` reproduces the original 7-pass FhG chain and requires acmenc. Launched from TUI Utilities (`Degrade audio`) or standalone. |

---

## Internal Behavior

**Chunk cache:** Chunks live in `usr/cache/chunks/<key>/<split>/LQ|HQ` — keyed on an MD5 hash of (source file contents + `segment_sec` + `overlap` + `fixed_delay` + augmentation config). Any training config that requests the same dataset with the same parameters reuses the cached chunks without re-chunking. The cache is global: different configs can share a chunk set if their source data and params match. A `.manifest.json` is written after each successful chunk run for legacy skip-if-present logic inside `_chunk_split`.

**Source conversion:** On chunk prep, any non-WAV source files (MP3, FLAC, etc.) are converted in-place to 32-bit float WAV via FFmpeg in parallel threads. The alignment trim (`align_data`) is baked in during this pass via FFmpeg's `atrim` filter. Already-WAV files are never re-converted. Conversion is skipped entirely if all sources are already WAV.

**Run isolation:** Each fresh run creates `runs/<name>/<timestamp>/`. Resume finds the most recently modified timestamped subfolder with a `checkpoints/` dir and picks the newest `.ckpt` by mtime.

**Val fixed evaluation set:** On the first real val run, `_lock_val_fixed_indices` groups all seen dataset indices by song, then samples `limit_val_batches / num_songs` chunks from each song (stratified). The result is locked into `_val_fixed_indices` — all subsequent val runs skip any chunk not in this set, so the loss is always computed on the exact same fixed chunks. Both `_val_fixed_indices` and the rotation schedule are persisted in the checkpoint and restored on resume.

**Val audio rotation:** Saves exactly `val_songs` songs × 3 files (LQ/HQ/Restored) = N×3 files per val run. At training start a rotation schedule is computed so that every val song gets equal coverage by end of training. Each song's chunk is picked once at schedule-build time (`_lock_val_refs`) — one specific non-silent chunk, never re-picked — and stored in `_val_locked_refs`, so the same audio is always compared across val steps for a given song. `val_rotate_every: auto` derives the cadence from total configured steps; an integer overrides it manually. The schedule and locked refs are checkpointed and resume-stable. File writes run in a background thread so training resumes immediately. Old configs using `val_audio_pairs` still work via fallback.

**Val perceptual metrics (diagnostic branch):** Four metrics computed live after each val run: `visqol` (perceptual quality score via pyvisqol — primary quality signal; lazy-loaded, silently skipped if not installed; runs on `visqol_fraction` of val pairs, default 1.0), `sdr` (Signal-to-Distortion Ratio), `sfr` (spectral flatness ratio 8-22kHz — rising above 1.05 is an early overfitting/artifact signal), `sisdr` (legacy, noisy). Optional `target_band_loss` (configurable Hz range, off by default) appends `tbl=` to console and checkpoint names when enabled. All metrics logged to TensorBoard. `msstft` and `hf_band_mae` removed from live val; available as offline helpers via `evaluate.py` for legacy checkpoint scoring.

**Checkpoint monitoring (diagnostic branch):** Two separate weighting systems exist:

1. **Checkpoint save trigger** — `AudioLightningModule.on_validation_epoch_end()` logs `val_composite` using weights `visqol=0.50, sdr=0.25, sfr=0.15, sisdr=0.10`. The `ModelCheckpoint` callback monitors `val_composite` (min) to decide whether to save.

2. **Filename rank badge** — `RankBadger` in `train.py` renames checkpoints after save using the same weights. `[1]` = lowest composite score (best).

All checkpoints are kept (`save_top_k=-1` enforced in code). `evaluate.py` uses a separate weighting that includes msstft/hfmae for legacy checkpoint compatibility — offline ranking only.

**Checkpoint filenames (diagnostic branch):** Format is `[rank]-step={step:06d}-sisdr={val_loss:.3f}-visqol={val_visqol:.3f}-sdr={val_sdr:.3f}-sfr={val_sfr:.3f}.ckpt`. `tbl={val_tbl:.4f}` is appended when `target_band_loss_enabled: true`. `[rank]` is prepended by the `RankBadger` callback after each save.

**Baseline eval:** On fresh runs (no resume), `trainer.validate()` is called on the pretrained weights before `trainer.fit()`. Prints `[baseline] sisdr=XX.XXX` so improvement is immediately visible against the starting point.

**StepPrinter (diagnostic branch):** TQDM is disabled. Prints one line per optimizer step. `it/s` counts every batch (including accumulation batches) for consistency with pre-accumulation baselines. Val time is excluded from the rate. Last val metrics shown inline once available: `visqol, sdr, sfr, sisdr` (perceptual zoom-out first, sisdr last). `tbl=` appended when target band loss is enabled.

**Band weight:** `MultiFrequencyGenLoss` applies a penalty curve over STFT bins, shape controlled by `band_weight_shape`: `"gaussian"` (default, raised curve peaking at `band_weight_center_hz` with width `band_weight_sigma_hz`) or `"trapezoid"` (flat-topped between `band_weight_lo_hz`/`band_weight_hi_hz` with `band_weight_ramp_hz` soft edges — useful for targeting a specific rolloff/transition band). `band_weight_gain=0` is perfectly flat regardless of shape. Replaces the old `hf_boost` + `hf_threshold_ratio` step function. Config keys `hf_boost` and `hf_threshold_ratio` are accepted for back-compat but ignored.

**Alignment:** `align_data` accepts an integer offset in samples — positive trims LQ, negative trims HQ. Baked into WAV files at conversion time via FFmpeg, never applied at chunk or training time. iTunes-encoded MP3s have a consistent 1057-sample encoder delay. Set `false` to disable.

**Augmentation internals:** `normalize_pair` scales by joint peak — pre-existing flat-top clipping is preserved as valid training signal. `stereo_alternation` picks L or R by sample index (even → L, odd → R) — `prob: 1.0` alternates systematically.

**Inference normalization:** Input normalized by peak before chunked inference, rescaled back after. Preserves transient headroom.

**Training step structure (diagnostic branch):** Two-pass discriminator-first structure restored from original working code:

1. `optimizer_d.zero_grad()` then discriminator forward on `output.detach()` → `loss_d` → `manual_backward` → clip → `optimizer_d.step()`
2. `optimizer_g.zero_grad()` then discriminator forward on `output` (live, not detached) → compute fresh `targets_feature_maps` → `loss_g` → `manual_backward` → clip → `optimizer_g.step()`

Feature matching loss uses fresh feature maps from the generator step, not stale ones from the discriminator update. TF32 disabled globally (`torch.set_float32_matmul_precision("highest")`; `allow_tf32=False`). AMP managed by Lightning's `precision_plugin`; no manual `torch.amp.autocast` wrapping. Gradient clipping uses manual `scaler.unscale_(optimizer)` + `torch.nn.utils.clip_grad_norm_()`, not `self.clip_gradients()`.

**CRITICAL — never call `torch.cuda.empty_cache()` in any training or validation hook.** Only acceptable inside an OOM recovery handler (`except torch.cuda.OutOfMemoryError`).

**Startup update check:** `apollo.bat` runs `git pull --ff-only` before installing dependencies, if the working dir is a git checkout and `git` is on PATH. Failure is non-fatal — a warning is printed and it continues with the local copy.

**Dev mode:** `apollo.bat --dev` sets `APOLLO_DEV=1` before launching the TUI. When set, `_list_configs()` in `tui.py` includes `dev/*.yaml` in the config picker alongside `configs/*.yaml`. The flag is stripped before any other arg parsing — `apollo.bat --dev train ...` works as expected.

**Pretrained checkpoint loading:** `BaseModel.from_pretrain()` allow-lists OmegaConf's `DictConfig`/`ListConfig` classes via `torch.serialization.add_safe_globals` before `torch.load(..., weights_only=True)`, since the upstream HuggingFace checkpoint's `infos` dict contains a pickled `DictConfig`.

---

## Current Branch State

| Branch | Contains |
|--------|----------|
| `main` | Spectral merge engine, TUI ensemble picker, aux checkpoint blending, `_ensure_wav` in-place MP3→WAV conversion, short-chunk STFT fix, noise augmentation disabled. Dev mode (`--dev` flag). stfl configs live in `dev/` on diagnostic only. |
| `diagnostic/revert-training-step` | All of main plus: training step revert (discriminator-first, fresh feature maps), configurable TF32, AMP via Lightning precision plugin, VISQOL live metrics, stfl/stfl-og/stfl2 configs in `dev/`. |

The diagnostic branch contains four training fixes that have not yet been validated. If they resolve the stfl regression, they will be merged to main.

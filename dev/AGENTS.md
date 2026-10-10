# Apollo-mod — Agent Reference

Read `README.md` for usage, config reference, data layout, commands, and augmentation options. This file covers internal architecture, constraints, and non-obvious behavior that isn't user-facing.

## Coding Rules

All changes go to `main`. The `diagnostic/revert-training-step` branch has been deleted — all its fixes are now in main.

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

## Project Context

The target is a set of 2003 leaked tracks whose surviving files are all heavily degraded: a WMA rip passed through several MP3 encodes, the last at 192 kbps. The full chain and its history are unknown, so `utils/degrade/stfl.json` is an estimate of it. Only some songs have a later studio or HQCD release to pair against; those are aligned by hand and used as real LQ/HQ pairs. Other songs are kept HQ-only and their LQ is generated from randomized chains (see compression), which gives exact alignment and variety instead of an exact match to the unknown chain. Hand-aligned real pairs with mix differences made models worse, so real pairs are limited to ones that align cleanly.

---

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
  compression.py         -- shared codec-chain sampling, encoding, alignment
  degrade/               -- degradation JSON configs
configs/                 -- training YAML configs
  apollo.yaml            -- base config (feature_dim=256)
  apollo_uni.yaml        -- universal config (feature_dim=384)
  apollo_stfl.yaml       -- stfl active run config
  apollo_stfl-og.yaml    -- original stfl config (reference baseline, resume: false)
  apollo_stfl2.yaml      -- stfl2 experimental (deep_gain, silence_dip, all layers unfrozen)
  apollo_stfl_new.yaml   -- stfl_new run (HQ-only songs get generated LQ)
dev/                     -- internal docs (not user-facing)
  AGENTS.md
data/                    -- training source audio (LQ/ + HQ/ pairs per split)
models/                  -- pretrained / downloaded weights
runs/                    -- training run output (checkpoints, logs)
input/ output/           -- inference I/O staging dirs
```

## Architecture

| File | Role |
|---|---|
| `core/look2hear/models/apollo.py` | Generator. BSNet + Roformer layers. Band-split STFT/iSTFT always in float32. Time dimension processed by `ICB` (stacked depthwise-separable Conv1d) — scales linearly with sequence length, no attention blowup. Band dimension processed by `Roformer` (fixed small count). |
| `core/look2hear/discriminators/frequencydis.py` | Frequency discriminator. Hann windows cached as plain tensors in `_hann_cache` (not `register_buffer` — keeps them out of checkpoint state_dict). Mono input auto-expanded to stereo via `.expand()`. `window_weight_boost` (default `false`) — only useful for severely degraded sources. |
| `core/look2hear/system/audio_litmodule.py` | Lightning module. Manual optimization, gradient checkpointing (BSNet only), 10-second clip val inference (OLA-chunked), live perceptual metrics (VISQOL/SI-SDR/HFNR), LQ/HQ/Restored audio saving, RAM/CUDA OOM watchdogs. Training step: discriminator-first, two-pass, fresh feature maps for generator loss. TF32 disabled globally. AMP via Lightning's precision plugin. |
| `core/look2hear/losses/gan_losses.py` | GAN losses. Band weight curve shapes: `"gaussian"` (raised bump at `band_weight_center_hz`, width `band_weight_sigma_hz`), `"trapezoid"` (flat-topped between `band_weight_lo_hz`/`band_weight_hi_hz` with `band_weight_ramp_hz` edges). `band_weight_gain=0` = perfectly flat. |
| `core/paired_datamodule.py` | Loads chunked LQ/HQ WAV pairs. Live augmentation pipeline. Val dataloader reads the extracted 10 s clips (`ChunkedPairDataset` in val mode, restricted to the locked clips via `set_val_subset`). |
| `core/train.py` | Entry point. Chunk cache keyed on source md5s + chunk params. Pretrained checkpoint selection is `feature_dim`-aware. `val_metric_songs` and `val_metric_samples` accepted as fallback aliases for `val_songs`. Baseline val pass on fresh runs (skipped if RAM headroom < 1.5 GB). Checkpoint monitor: `visqol`, `mode: max`, no early stopping. Legacy checkpoint filenames are renamed to the `sisdr`/`visqol`/`hfnr` scheme at resume (`_migrate_legacy_ckpt_names`). The live progress line prints `Validating...` during a val run. |
| `core/inference.py` | Chunked OLA inference. Auto-selects latest checkpoint. `_ensure_wav` converts input to WAV before loading. `_spectral_merge` blends original/enhanced in STFT domain. `--low_end_preserve`, `--ensemble` JSON, `--aux_weights` blending all supported. |
| `utils/tui.py` | Keyboard-navigated TUI. Checkpoints are ranked by `_ckpt_score` = `(visqol, sisdr, -hfnr)`; duplicates per step are dropped (`_dedup_ckpts`), unscored ones (no visqol) rank last, and `migrate_checkpoints` renames legacy filenames on launch. Latest/Best checkpoint options in inference picker. Ctrl+C during training saves checkpoint. Ensemble picker after output path selection. "Update Apollo" in Utilities runs `git pull --ff-only`. `--dev` flag adds `dev/*.yaml` to config picker. |
| `core/evaluate.py` | Offline checkpoint evaluator. Reads metrics from filenames; runs inference only for missing ones. Composite weights: visqol 0.60, hfnr 0.25, sisdr 0.15. VISQOL via `visqol-python`. Scores cached in `<ckpt_dir>/.eval_cache.json`. |
| `utils/align_audio.py` | LQ/HQ temporal alignment. Global sinc resample for speed drift, chunked cross-correlation micro-alignment. |
| `utils/degrade_audio.py` | Synthetic degradation pipeline via JSON configs. Fixed-chain mode (`chain`, own encoders, supports `mp3_fhg` and filters, no `codec` steps) or `--randomize` mode (`random` block, all encoding via `compression.py`). Randomized mode cuts each file into `--segment_sec` segments (down to 3 s), gives each its own chain, writes one LQ/HQ pair per segment plus `_chains.jsonl`. TUI flow lives in `_screen_randomized`. |
| `utils/compression.py` | Shared codec-chain module used by `degrade_audio.py --randomize` and the cached `compression` augmentation: chain sampling from a `random` recipe (`sample_chain`), encoders (`wma_encode`, `mp3_lame`, `codec` steps for aac/vorbis/opus/mp2 via ffmpeg, and `mp3_fhg` via `acmenc`), encoder availability check (also verifies `acmenc` when the recipe can draw it), per-window delay measurement and alignment (`degrade_window`). Padding is encoded with each window and cropped after alignment so codec edge artifacts never reach the pair. The shipped recipe is `utils/degrade/stfl.json` (WMA, LAME and Fraunhofer MP3 passes); `other_codecs` is supported by the sampler but not used by it. |

---

## Internal Behavior

**Chunk cache:** `usr/cache/chunks/<key>/<split>/` — keyed on MD5 of (source contents + `segment_sec` + overlap + `fixed_delay` + aug config + compression recipe contents). Shared across configs with matching dataset and params. `.manifest.json` written after each successful run. File digests are reused from `usr/cache/md5_index.json` while a file's path, size and mtime are unchanged. After the val clips are locked, the val dataloader reads only those clips (`PairedAudioDataModule.set_val_subset`).

**Source conversion:** Non-WAV sources converted to 32-bit float WAV via FFmpeg in parallel threads. `align_data` integer offset baked in via `atrim`. Already-WAV files are never re-converted.

**Run isolation:** Each fresh run creates `runs/<name>/<timestamp>/`. Resume finds the most recently modified timestamped subfolder with a `checkpoints/` dir.

**Val system — clip-based metric computation:** At the first val run, `_lock_val_songs` picks `val_songs` songs from the full val set and locks their file paths in `_val_song_refs`. `train.py::_extract_val_clips` cuts two non-overlapping 10-second highest-RMS clips per val song (`<stem>_clip0/_clip1`, ViSQOL input guidelines) and caches them; each clip is its own entry in `_val_song_refs`, so `val_songs` counts clips. `_visqol_score` pads 0.5 s of silence on both sides. Every subsequent val run runs OLA chunked inference on those same locked clips and scores VISQOL, SI-SDR, and HFNR against the corresponding HQ reference clip. Results are averaged across songs. Clip offsets, paths, and all rotation state are checkpointed and resume-stable. When the rotation window changes (see below), the next clips are scored. LQ/HQ/Restored triplets of the scored clips are written to `runs/<name>/<timestamp>/val_audio/` inline during `_compute_val_metrics` — no second inference pass, no background thread.

**Val perceptual metrics:** Three metrics after each val run, logged under the bare names used in checkpoint filenames: `visqol` (ViSQOL via `visqol-python`; lazy-loaded, silently skipped if not installed; runs on every val run on `visqol_fraction` of locked clips; primary monitor), `sisdr` (SI-SDR over the fixed chunk batch in `validation_step`; noisy, tiebreaker), `hfnr` (high-frequency noise ratio, spectral flatness ratio 8-22kHz; canary for HF noise injection, not a quality metric; rising is expected for MP3 restoration since the model reconstructs frequencies the codec removed). Checkpoint filename: `step={step:06d}-sisdr=..-visqol=..-hfnr=..`. A checkpoint with `visqol=-1.000` was saved before any VISQOL score existed.

**VISQOL:** Uses the `visqol-python` package (pip-installable, pure-Python port of ViSQOL v3.3.3, Windows-compatible). Install: `uv pip install "visqol-python[all]"`. Added to `requirements.txt`. If not installed, `visqol=-1.000` is logged with no crash. Both `audio_litmodule.py` and `evaluate.py` use the same `_get_visqol_api()` / `_visqol_score()` helpers.

**Band weight:** `MultiFrequencyGenLoss` applies a penalty curve over STFT bins. Two shapes: `"gaussian"` (bump centered at `band_weight_center_hz`) and `"trapezoid"` (flat band with soft ramps). Any other `band_weight_shape` raises. `band_weight_gain=0` is flat regardless of shape.

**Compression augmentation:** `augmentation.cached.compression` only. A `compression` or `mp3_degradation` key under `live` raises an error, and `cached.mp3_degradation` raises a pointer to `compression`. Applied to the LQ chunk during chunking in `core/train.py` (`_CompressionAug`). Songs with a file in `LQ/` are never encoded. A song present only in `HQ/` (needs the `LQ/` + `HQ/` layout, compression enabled) is chunked with LQ as a copy of HQ and every chunk's LQ is generated; `align_data` offsets are not applied to it. The same rule applies to `data/<name>/val/`: `_extract_val_clips` cuts each HQ-only song's two highest-RMS 10 s clips from HQ and encodes each through its own chain (seeded `<seed>:<stem>_val:<clip>`, so reruns are identical), with the surrounding audio as context. A clip whose compression fails is dropped, never written with a clean LQ. The val cache key includes the recipe digest only when the val set has HQ-only songs. A split with only `HQ/` (or only `_HQ` files) is accepted. Legacy `fraction` and `exclude` keys are ignored with a warning. Each chain draws an MP3 encoder mode (`mp3_encoder.mode` weights over `lame`, `fhg`, `mixed`; `mp3_fhg` block in the recipe): all LAME, all `mp3_fhg` including the final pass, or a random mix with at least one of each (needs 2+ MP3 passes, else mixed is not drawn). Without `mp3_encoder` the sampler's draws are unchanged. `acmenc` path comes from the YAML `acmenc` key or the recipe's `external_codecs.acmenc`, is excluded from the chunk cache digest, and a recipe that can draw it fails the preflight if the path is unset or missing. `acmenc` input is always re-decoded to 16-bit WAV first. Delay is measured per chunk, since chain order changes the total encoder delay (WMA adds one 2048-sample frame). Verified on real material at 3 s chunks: residual lag under 0.1 sample across WMA-first, WMA-last, WMA-middle and double-WMA chains.

**Mid/side isolation augmentation:** Live augmentation option (`mid_side_isolation` block). When enabled, randomly collapses the LQ/HQ pair to mid `(L+R)/2` or side `(L-R)/2`, duplicated to both channels (reduced to one channel when `stereo_alternation` is enabled, so batch shapes match). Applied identically to LQ and HQ. Rolls before `stereo_alternation` — if mid/side fires, stereo_alternation is skipped for that chunk. Only runs on stereo input. Off by default. Config keys: `enabled`, `prob_mid`, `prob_side`.

**Training step structure:** Two-pass discriminator-first: (1) discriminator forward on detached output → `loss_d` → backward → step; (2) generator forward on live output → fresh feature maps (discriminator params frozen, real-side maps computed under `no_grad`) → `loss_g` → backward → step. TF32 disabled globally. AMP via Lightning `precision_plugin`, not manual autocast. Gradient clipping via manual `unscale_` + `clip_grad_norm_`.

**CRITICAL — never call `torch.cuda.empty_cache()` in any training or validation hook.** Only acceptable inside an OOM recovery handler.

**Alignment:** `align_data` integer offset baked into WAV at conversion time. Positive trims LQ, negative trims HQ. `false` to disable.

**Augmentation — mid/side priority:** `mid_side_isolation` rolls first. If it fires, `stereo_alternation` is skipped. If not, `stereo_alternation` runs normally. They cannot both apply to the same chunk.

**Startup update check:** `apollo.bat` runs `git pull --ff-only` on startup. Failure is non-fatal.

**Dev mode:** `apollo.bat --dev` sets `APOLLO_DEV=1`. `tui.py` includes `dev/*.yaml` in the config picker.

**Checkpoint IO:** all checkpoint loads go through `look2hear.utils.ckpt_io.load_checkpoint` (`weights_only=True` first, full unpickle fallback with a warning). Downloads stream to `<dest>.part` and are moved into place only when complete (`download_file`). The resume patch rewrites the checkpoint atomically. Ctrl+C sets a flag; `train.py` saves `step=NNNNNN-interrupted.ckpt` at the next batch end and stops (second Ctrl+C force-quits). The TUI pause (`PAUSED`) is acknowledged by `PAUSED.ack`.

**Gefen optimizer:** `type: gefen` in the optimizer config. Drop-in AdamW replacement with ~8x lower optimizer-state memory and faster optimizer steps. Requires `pip install gefen`; builds CUDA kernels via JIT on first run (needs `nvcc` in PATH). Falls back to AdamW32bit with a warning if not installed. Recommended over `adamw_8bit` for new runs.

**GefenMuon optimizer:** `type: gefen_muon`. Routes 2D parameters (Linear weights — attention layers) to GefenMuon (no second-moment state, momentum-only) and all other parameters (Conv1d, norms, biases) to Gefen. Gefen's paper recommends this split for fine-tuning over plain Gefen. Implemented via `_ComboOpt` wrapper that exposes a unified `param_groups`, `step()`, `zero_grad()`, `state_dict()`, and `load_state_dict()` so PyTorch schedulers and Lightning checkpointing work transparently. If no 2D params exist, falls back to Gefen-only; if no non-2D params exist, uses GefenMuon-only.

**Randomized degradation (`--randomize`):** Sources are cut into `segment_sec` (30 s) segments with `pad_sec` of context each side. Each segment gets its own chain from `sample_chain()`, seeded by `seed` + segment name, so reruns reproduce. The chain runs on the padded segment, one integer delay is measured per segment (0-4 kHz cross-correlation, `measure_offset`), and the pair is cropped to the unpadded window. Segments that are near-silent or below `min_corr` are dropped. Output is one LQ/HQ file pair per segment under `<output>/LQ` and `<output>/HQ`, plus `_chains.jsonl` with the chain, lag and correlation per segment. Existing pairs are skipped on rerun. ffmpeg's WMA path measured a constant -2048 sample lag in testing, so the per-segment measurement is required, not optional. Sources must be 32/44.1/48 kHz (MP3 rates). `--segment_sec 3` is supported: tested on 28 real 3 s chunks from two songs, residual lag after alignment was at most 0.09 samples in 200 Hz-4 kHz, in the 4-8 kHz envelope, and between chunk halves (no drift). The first segment of a file has no leading pad, so the WMA delay shortens it; the length floor is `min(min_segment_sec, 0.9 * segment_sec)`.

**Config key aliases:** `val_songs` is the current key. `val_metric_songs`, `val_metric_samples`, and `val_audio_pairs` are accepted as fallbacks in `train.py` for old configs.

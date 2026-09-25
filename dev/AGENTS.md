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
configs/                 -- training YAML configs
  apollo.yaml            -- base config (feature_dim=256)
  apollo_uni.yaml        -- universal config (feature_dim=384)
  apollo_stfl.yaml       -- stfl active run config
  apollo_stfl-og.yaml    -- original stfl config (reference baseline, resume: false)
  apollo_stfl2.yaml      -- stfl2 experimental (deep_gain, silence_dip, all layers unfrozen)
  apollo_stfl_new.yaml   -- stfl_new run with piecewise band weighting
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
| `core/look2hear/system/audio_litmodule.py` | Lightning module. Manual optimization, gradient checkpointing (BSNet only), 30-second clip val inference (OLA-chunked), live perceptual metrics (VISQOL/SDR/SFR), LQ/HQ/Restored audio saving, RAM/CUDA OOM watchdogs. Training step: discriminator-first, two-pass, fresh feature maps for generator loss. TF32 disabled globally. AMP via Lightning's precision plugin. |
| `core/look2hear/losses/gan_losses.py` | GAN losses. Band weight curve shapes: `"gaussian"` (raised bump at `band_weight_center_hz`, width `band_weight_sigma_hz`), `"trapezoid"` (flat-topped between `band_weight_lo_hz`/`band_weight_hi_hz` with `band_weight_ramp_hz` edges), `"piecewise"` (arbitrary breakpoints list `[[hz, weight], ...]`, linear interpolation). `band_weight_gain=0` = perfectly flat. |
| `core/paired_datamodule.py` | Loads chunked LQ/HQ WAV pairs. Live augmentation pipeline. Val dataloader uses `ChunkedPairDataset` (pre-extracted 10s clips, `batch_size=1`). Chunk dataloader uses `ChunkedPairDataset`. |
| `core/train.py` | Entry point. Chunk cache keyed on source md5s + chunk params. Val clips pre-extracted via `_extract_val_clips` (2×10s per song, highest-RMS windows). Baseline val pass on fresh runs cached to `usr/cache/baseline/`. Pretrained checkpoint selection is `feature_dim`-aware. `val_metric_songs` and `val_metric_samples` accepted as fallback aliases for `val_songs`. `extra_layers: N` appends N new zero-initialized BSNet blocks after loading pretrained weights, freezing all pretrained params. Checkpoint monitor: `val_sdr`, `mode: max`. |
| `core/inference.py` | Chunked OLA inference. Auto-selects latest checkpoint. `_ensure_wav` converts input to WAV before loading. `_spectral_merge` blends original/enhanced in STFT domain. `--low_end_preserve`, `--ensemble` JSON, `--aux_weights` blending all supported. |
| `utils/tui.py` | Keyboard-navigated TUI. Latest/Best checkpoint options in inference picker. Ctrl+C during training saves checkpoint. Ensemble picker after output path selection. "Update Apollo" in Utilities runs `git pull --ff-only`. `--dev` flag adds `dev/*.yaml` to config picker. |
| `core/evaluate.py` | Offline checkpoint evaluator. Reads metrics from filenames; runs inference only for missing ones. VISQOL via `visqol-python`. Scores cached in `<ckpt_dir>/.eval_cache.json`. |
| `utils/align_audio.py` | LQ/HQ temporal alignment. Global sinc resample for speed drift, chunked cross-correlation micro-alignment. |
| `utils/degrade_audio.py` | Synthetic degradation pipeline via JSON configs. |

---

## Internal Behavior

**Chunk cache:** `usr/cache/chunks/<key>/<split>/` — keyed on MD5 of (source contents + `segment_sec` + overlap + `fixed_delay` + aug config). Shared across configs with matching dataset and params. `.manifest.json` written after each successful run.

**Source conversion:** Non-WAV sources converted to 32-bit float WAV via FFmpeg in parallel threads. `align_data` integer offset baked in via `atrim`. Already-WAV files are never re-converted.

**Run isolation:** Each fresh run creates `runs/<name>/<timestamp>/`. Resume finds the most recently modified timestamped subfolder with a `checkpoints/` dir.

**Val system — clip-based metric computation:** Val clips are pre-extracted from the val set during `prepare_data` and cached under `usr/cache/baseline/`. Two 10-second clips are extracted per song, selected by highest-RMS energy windows (skipping the first and last 5s of each track) using the HQ mono mix for RMS scoring. The two clips per song are non-overlapping with at least 10s separation. Files are named `{stem}_clip0.wav` and `{stem}_clip1.wav` at 44.1kHz stereo. `val_songs` controls how many of these files are used per val window (default 6, covering all clips from 3 songs). At the first val run, `_val_fixed_indices` locks the selected clip files permanently. Every subsequent val run runs OLA chunked inference on those same locked clips and scores SDR, SFR, and VISQOL. Results are averaged. `_val_fixed_indices`, `_val_run_count`, and `_cached_val_visqol` are all checkpointed and resume-stable. LQ/HQ/Restored triplets are written to `runs/<name>/<timestamp>/val_audio/` inline — no second inference pass.

**VISQOL alternation:** `_val_run_count` is incremented each val pass. VISQOL is computed on odd-numbered runs only; even runs carry forward the last real score as `_cached_val_visqol`. When VISQOL is skipped, `val_visqol=-1.000` is written to the checkpoint filename so gaps are identifiable. The evaluate script can rescore `-1.000` checkpoints post-hoc.

**Baseline val cache:** On fresh runs (not resume), a baseline val pass fires before training begins. Results are cached to `usr/cache/baseline/<key>.json`, keyed on the MD5 of the pretrained weights + the val clip cache key. Subsequent fresh runs with the same pretrain and val data skip the baseline pass and log a cache hit.

**Val perceptual metrics:** Four metrics after each val run: `val_visqol` (ViSQOL via `visqol-python`; lazy-loaded, silently skipped if not installed; runs on `visqol_fraction` of locked songs), `val_sdr` (Signal-to-Distortion Ratio — primary monitor), `val_sfr` (spectral flatness ratio 8–22kHz — useful as a canary for HF noise injection, not as a primary quality metric; rising is expected for MP3 restoration since the model reconstructs frequencies the codec removed), `val_loss` (negated SI-SDR, computed over the fixed chunk batch in `validation_step`). All logged to TensorBoard. Checkpoint monitor: `val_sdr`, `mode: max`.

**VISQOL:** Uses the `visqol-python` package (pip-installable, pure-Python port of ViSQOL v3.3.3, Windows-compatible). Install: `uv pip install "visqol-python[all]"`. Added to `requirements.txt`. If not installed, `val_visqol=0.000` with no crash. Both `audio_litmodule.py` and `evaluate.py` use the same `_get_visqol_api()` / `_visqol_score()` helpers. `_visqol_score` runs in audio mode (requires 48kHz — resampled internally from 44.1kHz), mono-mixes both signals, and prepends/appends 0.5s of silence padding per ViSQOL's input guidelines before calling `measure_from_arrays`. Max score in audio mode is ~4.75.

**Band weight:** `MultiFrequencyGenLoss` applies a penalty curve over STFT bins. Three shapes: `"gaussian"` (bump centered at `band_weight_center_hz`), `"trapezoid"` (flat band with soft ramps), `"piecewise"` (list of `[hz, weight]` breakpoints, linearly interpolated — use to match the exact codec damage curve for your material). `band_weight_gain=0` is flat regardless of shape. `apollo_stfl_new.yaml` uses piecewise shape tuned for MP3 restoration: flat at 0.15 below 550Hz, peaks at 1.0 from 12–14kHz, drops to 0.1 above 16.5kHz.

**Mid/side isolation augmentation:** Live augmentation option (`mid_side_isolation` block). When enabled, randomly collapses the LQ/HQ pair to mid `(L+R)/2` or side `(L-R)/2`, duplicated to both channels. Applied identically to LQ and HQ. Rolls before `stereo_alternation` — if mid/side fires, stereo_alternation is skipped for that chunk. Only runs on stereo input. Off by default. Config keys: `enabled`, `prob_mid`, `prob_side`.

**Training step structure:** Two-pass discriminator-first: (1) discriminator forward on detached output → `loss_d` → backward → step; (2) generator forward on live output → fresh feature maps → `loss_g` → backward → step. Feature matching uses fresh feature maps from the generator step. TF32 disabled globally. AMP via Lightning `precision_plugin`, not manual autocast. Gradient clipping via manual `unscale_` + `clip_grad_norm_`.

**CRITICAL — never call `torch.cuda.empty_cache()` in any training or validation hook.** Only acceptable inside an OOM recovery handler.

**Alignment:** `align_data` integer offset baked into WAV at conversion time. Positive trims LQ, negative trims HQ. `false` to disable.

**Augmentation — mid/side priority:** `mid_side_isolation` rolls first. If it fires, `stereo_alternation` is skipped. If not, `stereo_alternation` runs normally. They cannot both apply to the same chunk.

**Startup update check:** `apollo.bat` runs `git pull --ff-only` on startup. Failure is non-fatal.

**Dev mode:** `apollo.bat --dev` sets `APOLLO_DEV=1`. `tui.py` includes `dev/*.yaml` in the config picker.

**Pretrained checkpoint loading:** `BaseModel.from_pretrain()` allow-lists OmegaConf types via `add_safe_globals` before `torch.load(..., weights_only=True)`.

**Extra layers (`extra_layers`):** When `extra_layers: N` is set in the training config (default 0), `append_extra_layers()` fires after pretrained weights load (fresh runs only — resume skips it). It freezes all pretrained parameters (BN front-end, all original BSNet layers, output heads), then appends N new BSNet blocks to `model.net`. New blocks are zero-initialized as near-identity: `Roformer.output` and `Roformer.MLP_output` are zeroed (so attention acts as identity via its residuals), and the final conv of each `ConvActNorm1d` in ICB is zeroed (so time processing ≈ zero at init). Optimizer state is only allocated for the new trainable parameters. `n_layers_to_freeze` is ignored when `extra_layers > 0`.

**Gefen optimizer:** `type: gefen` in the optimizer config. Drop-in AdamW replacement with ~8x lower optimizer-state memory and faster optimizer steps. Requires `pip install gefen`; builds CUDA kernels via JIT on first run (needs `nvcc` in PATH). Falls back to AdamW32bit with a warning if not installed. Recommended over `adamw_8bit` for new runs.

**GefenMuon optimizer:** `type: gefen_muon`. Routes 2D parameters (Linear weights — attention layers) to GefenMuon (no second-moment state, momentum-only) and all other parameters (Conv1d, norms, biases) to Gefen. Gefen's paper recommends this split for fine-tuning over plain Gefen. Implemented via `_ComboOpt` wrapper that exposes a unified `param_groups`, `step()`, `zero_grad()`, `state_dict()`, and `load_state_dict()` so PyTorch schedulers and Lightning checkpointing work transparently. If no 2D params exist, falls back to Gefen-only; if no non-2D params exist, uses GefenMuon-only.

**Config key aliases:** `val_songs` is the current key. `val_metric_songs`, `val_metric_samples`, and `val_audio_pairs` are accepted as fallbacks in `train.py` for old configs.

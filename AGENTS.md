# Apollo-mod

Audio restoration trainer built on Apollo (look2hear), adding GAN discriminator, ViSQOL-guided validation, and a full TUI.

## Architecture

```
core/
  train.py                    -- Hydra entry point; all training logic
  evaluate.py                 -- Offline checkpoint evaluator (TUI + CLI)
  paired_datamodule.py        -- LQ/HQ paired audio data pipeline
  look2hear/
    models/apollo.py          -- Apollo backbone (BSNet layers)
    losses/gan_losses.py      -- MultiFrequencyGenLoss / MultiFrequencyDisLoss
    system/audio_litmodule.py -- LightningModule; val loop computes visqol + hfnr
utils/
  tui.py                      -- Rich TUI; training, inference, evaluate, utilities
configs/                      -- One YAML per experiment (ground truth, user-owned)
```

## Metrics

| Metric | Key | Better | Notes |
|--------|-----|--------|-------|
| ViSQOL | `visqol` | higher | Primary; perceptual score (logged positive, not negated) |
| SI-SDR | `sisdr` | higher | Waveform quality (logged positive; negation stripped at log time) |
| HFNR   | `hfnr`  | lower  | High-Frequency Noise Ratio; lower = cleaner HF output |

SDR removed as redundant. MS-STFT removed. `sfr` renamed to `hfnr`. All metric keys are unprefixed (no `val_` prefix anywhere).

## Checkpoint Filename Format

```
step={step:06d}-sisdr={sisdr:.3f}-visqol={visqol:.3f}-hfnr={hfnr:.3f}
```

Legacy filenames with `val_loss=`, `val_visqol=`, `val_sfr=`, `val_sdr=`, or bare `sdr=` are auto-migrated on startup by `_migrate_legacy_ckpt_names` (train.py, evaluate.py, tui.py).

## Best Checkpoint Selection (TUI)

Composite score tuple: `(visqol, sisdr, -hfnr)` — visqol primary, sisdr tiebreaker, hfnr final tiebreaker (negated: lower = better). `_dedup_ckpts` removes per-step duplicates before scoring. `_ckpt_score` normalises sisdr with `abs()` for backwards compat with legacy negative-valued filenames.

## Config Keys (required in all configs)

- `early_stopping`: removed; do not add back
- `checkpoint.monitor`: always `visqol`
- `trainer.limit_val_batches`: always present (default `1.0`)
- `system.val_songs` / `system.val_rotate_every`: always `${training.val_songs}` / `${training.val_rotate_every}`
- All augmentation keys must be present even if disabled (`mid_side_isolation`, `deep_gain`, `silence_dip`)
- `system.target_band_loss_enabled` / `target_band_loss_lo_hz` / `target_band_loss_hi_hz` / `target_band_loss_weight`: always present; weight defaults to `1.0`. This is a **training loss** (direct MAE on generator output vs HQ in the specified band), not a val metric.
- `training.extra_layers`: number of new BSNet layers to append on top of frozen pretrained layers (default `0`)
- `training.extra_layers_init_scale`: `0.5` = copy last pretrained layer's output projections at half strength (recommended); `0.0` = zero-init. Only scales `band_net.output`, `band_net.MLP_output`, and each `seq_net.blocks[i].conv[-1]` — internal weights are untouched, so the effect does not compound across multiple new layers.

## Extra Layers System

When `extra_layers > 0`, all pretrained layers are frozen and N new trainable BSNet blocks are appended. On fresh start, new layers are initialised via `init_scale`. On resume, checkpoint weights overwrite the init. `append_extra_layers` always runs when `extra_layers > 0` (both fresh and resume) so architecture matches before checkpoint load.

## Target Band Loss

`target_band_loss_enabled: true` adds a direct MAE penalty on the generator for the Hz range `[lo_hz, hi_hz]` on every training step (not validation). Used in curriculum phases to target specific damaged frequency zones. Phase history is documented via comments in `apollo_stfl2.yaml`.

## Utilities (TUI)

- **Merge checkpoints**: blends two checkpoints at a configurable ratio. Shared keys are interpolated; keys unique to one model pass through untouched (supports partial merges between models with different layer counts). Optimizer state is cleared in the merged output.

## Running

```bash
# TUI (recommended)
python utils/tui.py

# Train directly
python core/train.py --config-path ../configs --config-name apollo

# Offline evaluate
python core/evaluate.py --conf_dir configs/apollo.yaml
```

## Data Layout

```
data/<exp_name>/train/LQ  HQ
data/<exp_name>/val/LQ    HQ
chunks/<exp_name>/        # auto-generated cache; keyed by content hash
```

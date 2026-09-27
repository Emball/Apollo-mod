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


## Target Band Loss

`target_band_loss_enabled: true` adds a direct MAE penalty on the generator for the Hz range `[lo_hz, hi_hz]` on every training step (not validation). Used in curriculum phases to target specific damaged frequency zones. Phase history is documented via comments in `apollo_stfl2.yaml`.

## Utilities (TUI)

- **Merge checkpoints**: blends two checkpoints at a configurable ratio. Shared keys are interpolated; keys unique to one model pass through untouched (supports partial merges between models with different layer counts). Optimizer state is cleared in the merged output.

## Experiment Identities

| Config | Dataset | Target degradation |
|--------|---------|-------------------|
| `apollo_stfl` / `apollo_stfl-og` | Eminem *Straight From The Lab* / *Encore* 2003 leaks. Mix of real LQ/HQ aligned pairs and synthetic approximation: WMA 128 → 192 MP3 VBR x5 → 192 MP3 CBR. `stfl-og` is the original run; `stfl` (and `stfl_new`) use less synthetic data and more carefully hand-aligned real leak pairs. | Hard HF cutoff from multi-gen lossy encoding; Napster/Limewire-era file sharing artifact chain. |
| `apollo_stfl2` | Koolos leaks (2011) — WAV files encoded through various iTunes MP3 encoder versions at 128–192 kbps. | **No hard cutoff** (iTunes encoder behaviour); single or 2-generation iTunes MP3; distinct spectral signature from the STFL chain. Fine-tuned specifically for this. |
| `apollo_stfl_new` | New run currently in training. Less synthetic data than stfl-og; majority real aligned STFL pairs. | Same target as stfl/stfl-og but cleaner training pairs. |
| `apollo_uni` | General-purpose base model (pretrained, not fine-tuned here). Used as the weight source for all fine-tunes. | — |

Do not conflate these experiments. Each model is specialised and not interchangeable.

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

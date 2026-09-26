# Apollo-mod

Audio restoration trainer built on Apollo (look2hear), adding GAN discriminator, VISQOL-guided validation, and a full TUI.

## Architecture

```
core/
  train.py                  -- Hydra entry point; all training logic
  evaluate.py               -- Offline checkpoint evaluator (TUI + CLI)
  paired_datamodule.py      -- LQ/HQ paired audio data pipeline
  look2hear/
    models/apollo.py        -- Apollo backbone
    losses/gan_losses.py    -- MultiFrequencyGenLoss / MultiFrequencyDisLoss
    system/audio_litmodule.py -- LightningModule; val loop computes visqol + hfnr
utils/
  tui.py                    -- Rich TUI; training, inference, evaluate screens
configs/                    -- One YAML per experiment (ground truth, user-owned)
```

## Metrics

| Metric | Key | Better | Notes |
|--------|-----|--------|-------|
| ViSQOL | `val_visqol` | higher | Primary; perceptual score |
| SI-SDR | `val_loss` (neg) | higher | Waveform quality |
| HFNR | `val_hfnr` | lower | High-Frequency Noise Ratio; canary for HF artifact injection |

SDR removed as redundant with SI-SDR. MS-STFT removed (superseded by ViSQOL). `sfr` renamed to `hfnr` throughout.

## Checkpoint Filename Format

```
step={step:06d}-sisdr={val_loss:.3f}-visqol={val_visqol:.3f}-hfnr={val_hfnr:.3f}
```

## Best Checkpoint Selection (TUI)

Composite score tuple: `(visqol, sisdr, -hfnr)` — visqol primary, sisdr tiebreaker, hfnr final tiebreaker (negated: lower noise = better).

## Config Keys (required in all configs)

- `early_stopping`: removed; do not add back
- `checkpoint.monitor`: always `val_visqol`
- `trainer.limit_val_batches`: always present (default `1.0`)
- `system.val_songs` / `system.val_rotate_every`: always `${training.val_songs}` / `${training.val_rotate_every}`
- All augmentation keys must be present even if disabled (`mid_side_isolation`, `deep_gain`, `silence_dip`)

## Running

```bash
# Train
python core/train.py --config-path ../configs --config-name apollo

# TUI
python utils/tui.py

# Offline evaluate
python core/evaluate.py --conf_dir configs/apollo.yaml
```

## Data Layout

```
data/<exp_name>/train/LQ  HQ
data/<exp_name>/val/LQ    HQ
chunks/<exp_name>/train/LQ  HQ   # auto-generated
chunks/<exp_name>/val/LQ    HQ
```

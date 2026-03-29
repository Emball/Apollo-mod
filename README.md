# Apollo — Single-GPU Fine-Tuning Fork

A fork of [JusperLee/Apollo](https://github.com/JusperLee/Apollo) adapted for fine-tuning on a **single consumer GPU** against a **static paired dataset** of known codec degradation. The upstream repo targets 8-GPU distributed training on dynamically-generated codec data; this fork strips that out and replaces it with a self-contained pipeline oriented around a fixed LQ/HQ audio pair collection.

Original paper: *Apollo: Band-sequence Modeling for High-Quality Music Restoration in Compressed Audio* (Li et al., 2024).

---

## What's different from upstream

### Training pipeline (`train.py`)

The original `train.py` is a thin launcher that assumes data is already prepared and a multi-GPU DDP environment is available. This fork's `train.py` is substantially rewritten:

- **Auto-preprocessing pipeline.** Drop raw paired audio into `data/LQ/` and `data/HQ/` (matched by filename stem) and the script chunks, resamples, and force-converts to stereo automatically before training begins. If `chunks/` is already populated it skips this step entirely.
- **Eval bootstrapping.** If `eval/` is empty, 3 chunk pairs are randomly moved out of `chunks/` into `eval/` at startup so there is always a held-out validation set without any manual setup.
- **Pretrained weight loading.** Accepts a local `.ckpt` or `.pth` via `--weights_path`. Handles both Lightning checkpoint format (keys prefixed `audio_model.`) and bare state dict format automatically. Falls back to downloading the official weights from HuggingFace if no path is given.
- **Selective layer freezing with per-group learning rates.** BSNet layers 0–1 are frozen to preserve low-level pretrained representations. Layers 2–3 (newly unfrozen relative to a more conservative baseline) train at `0.3×` base LR to protect their pretrained features. Layers 4–5, BN front-ends, and output heads train at the full base LR.
- **Single-GPU / single-process target.** `sync_dist`, `all_gather`, DDP strategy, and `sync_batchnorm` are all removed. `torch.set_float32_matmul_precision` set to `"high"` (TF32) rather than `"highest"` for a meaningful throughput gain with negligible quality difference.
- **Validation audio saving.** At each validation run, restored, LQ, and HQ samples from the first batch are written to `Exps/<name>/samples/step_XXXXXX/` for listening-based monitoring without needing a separate inference script.
- **Resume from last checkpoint.** Pass `--resume` to automatically find and resume from the most recent checkpoint in the experiment directory.

### Data module (`paired_datamodule.py`)

Replaces the original `MusdbMoisesdbDataModule` (which streams from HDF5 files and synthesises codec degradation on the fly) with `PairedAudioDataModule`, which reads pre-chunked stereo WAV pairs from `chunks/LQ/` and `chunks/HQ/`. Training pairs are assumed to already represent the target degradation — no on-the-fly codec simulation is performed.

### Generator loss (`look2hear/losses/gan_losses.py`)

- **High-frequency perceptual weighting.** `freq_MAE` now applies a frequency-dependent weight to the multi-scale STFT magnitude error. Bins above 50% of Nyquist (~11 kHz at 44100 Hz) are penalised at `3×` the rate of low/mid bins (`HF_BOOST = 3.0`, `HF_THRESHOLD_RATIO = 0.5`). This compensates for the natural energy imbalance that causes unweighted MAE to effectively ignore the high end.
- **Buffered windows and weight tensors.** All 7 Hann windows and their corresponding per-bin weight tensors are pre-registered as `nn.Module` buffers in `MultiFrequencyGenLoss.__init__`. The original code reallocated these on every forward pass (14 host-to-device transfers per step). They now live on the correct device from startup and are never reallocated.
- `freq_MAE` is now an instance method `_freq_MAE` on `MultiFrequencyGenLoss` rather than a module-level function, so it has access to the buffered tensors.

### Discriminator (`look2hear/discriminators/frequencydis.py`)

- **Inverse window-size weighting.** `MultiFrequencyDiscriminator` now scales each sub-discriminator's output by a weight inversely proportional to its window size before aggregating. The smallest windows (32, 64) — which have the finest frequency resolution in the high end — contribute most to the discriminator signal. Weights are normalised so the overall loss scale is unchanged relative to the original equal-weight scheme.

### Model (`look2hear/models/apollo.py`)

- **Buffered Hann window.** The Hann window used for STFT/iSTFT in the forward pass is registered as a buffer (`self.hann_win`) rather than being created fresh each call.
- **fp16-safe STFT.** Input is explicitly cast to `float32` before `torch.stft` (cuFFT does not support fp16 for non-power-of-two FFT sizes). The output is cast back to the input dtype before returning, keeping the rest of the forward pass in fp16 during mixed-precision training.
- **iSTFT complex cast.** `est_spec` is cast to `torch.complex64` before `torch.istft` to avoid dtype mismatches under mixed precision.

### Training system (`look2hear/system/audio_litmodule.py`)

- Removed `sync_dist=True` from all `self.log` calls (causes hangs on single GPU).
- Removed `all_gather` in validation and test epoch end hooks (multi-GPU only).
- Removed WandB-specific logger calls (`self.logger.experiment.log`).
- Discriminator forward passes in `training_step` restructured to cache `target_outputs` and `targets_feature_maps` from the discriminator update and reuse them in the generator update, eliminating one full discriminator forward pass per step.
- Explicit `requires_grad_(True/False)` toggling around the generator update to avoid computing unnecessary gradients through the discriminator during the generator loss backward pass.
- Validation step saves audio samples to disk if `sample_output_dir` is set (wired up by `train.py`).

### Configuration

Two configs are provided:

| Config | Model | `feature_dim` | `batch_size` | Notes |
|---|---|---|---|---|
| `configs/apollo.yaml` | Base Apollo | 256 | 2 | Lighter, faster per step |
| `configs/apollo_uni.yaml` | Apollo Universal | 384 | 1 | Wider model, step-based validation |

Both configs are set for single-GPU training (`devices: [0]`), `16-mixed` precision, TensorBoard logging, and `num_workers: 2` (safe default for Windows `spawn`-based multiprocessing with 16 GB RAM).

### New files

| File | Purpose |
|---|---|
| `paired_datamodule.py` | Dataset/datamodule for pre-chunked LQ/HQ WAV pairs |
| `export_for_uvr.py` | Export a trained Lightning checkpoint to a UVR-compatible `.pth` |
| `requirements.txt` | Pinned dependencies for the WinPython environment |

---

## Directory layout expected at training time

```
apollo-mod/
├── data/
│   ├── LQ/          ← raw degraded audio (any supported format)
│   └── HQ/          ← raw clean audio (matched filenames)
├── chunks/          ← auto-populated by train.py from data/
│   ├── LQ/
│   └── HQ/
├── eval/            ← auto-bootstrapped from chunks/ if empty
│   ├── LQ/
│   └── HQ/
├── configs/
│   ├── apollo.yaml
│   └── apollo_uni.yaml
└── train.py
```

If `chunks/` is already populated, `data/` is not required. Supported input formats: `.wav`, `.flac`, `.mp3`, `.aac`, `.ogg`, `.m4a`.

---

## Usage

**Standard run (Universal model, recommended):**
```
python train.py --conf_dir configs/apollo_uni.yaml --weights_path ./apollo_model_uni.ckpt
```

**Base model:**
```
python train.py --conf_dir configs/apollo.yaml --weights_path ./apollo_model.pth
```

**Resume from last checkpoint:**
```
python train.py --conf_dir configs/apollo_uni.yaml --resume
```

**Export to UVR after training:**
```
python export_for_uvr.py --ckpt ./Exps/Apollo_Universal/checkpoints/best.ckpt --out ./my_model_uvr.pth
```

Logs and checkpoints are written to `Exps/<exp.name>/`. Launch TensorBoard with:
```
tensorboard --logdir Exps/
```

---

## Tuning reference

| Parameter | Location | Default | Notes |
|---|---|---|---|
| `HF_BOOST` | `gan_losses.py` | `3.0` | Extra penalty weight for bins above `HF_THRESHOLD_RATIO` of Nyquist |
| `HF_THRESHOLD_RATIO` | `gan_losses.py` | `0.5` | Fraction of Nyquist above which HF boost applies (~11 kHz) |
| `n_layers_to_freeze` | `train.py` | `2` | Number of BSNet layers (from layer 0) to keep frozen |
| Base LR multiplier for layers 2–3 | `train.py` | `0.3` | Scale factor applied to newly-unfrozen mid layers |
| `optimizer_g.lr` | config yaml | `1e-5` | Generator base learning rate |
| `optimizer_d.lr` | config yaml | `1e-6` | Discriminator learning rate |
| `batch_size` | config yaml | `1–2` | Reduce to `1` if VRAM is tight |
| `num_workers` | config yaml | `2` | Keep at `2` on Windows; `spawn`-based workers are expensive |

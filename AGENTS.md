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
| `tui.py` | Keyboard-navigated TUI (Rich). Launched by `apollo.bat`/`apollo.sh` with no arguments. Wraps train/inference/utilities with persistent per-config state in `.tui_state.json`. Ctrl+C during training sends SIGINT to the subprocess, triggers a clean checkpoint save, and returns to the menu. |
| `eval_checkpoints.py` | Standalone utility. Evaluates all checkpoints against the val set, prints a ranked table (SI-SDR, msstft, sfr). `--rename` rewrites filenames with accurate loss values. |
| `configs/apollo.yaml` | Base config (`feature_dim=256`). |
| `configs/apollo_uni.yaml` | Universal config (`feature_dim=384`). |

---

## Internal Behavior

**Chunk cache:** A `.manifest.json` is written after each successful chunk run recording `segment_sec`, `align_data`, and `fixed_delay`. On train start, if the manifest doesn't match the current config the chunk dir is deleted and re-chunked automatically.

**Run isolation:** Each fresh run creates `runs/<name>/<timestamp>/`. Resume finds the most recently modified timestamped subfolder with a `checkpoints/` dir and picks the newest `.ckpt` inside.

**Val fixed evaluation set:** On the first real val run, `_lock_val_fixed_indices` groups all seen dataset indices by song, then samples `limit_val_batches / num_songs` chunks from each song (stratified). This ensures equal representation across songs and by extension bit rates. The result is locked into `_val_fixed_indices` — all subsequent val runs skip any chunk not in this set, so the loss is always computed on the exact same fixed chunks and is perfectly comparable across checkpoints. Both `_val_fixed_indices` and `_val_locked_refs` are persisted in the checkpoint and restored on resume.

**Val audio refs:** `_val_locked_refs` stores file paths and sample offsets for `val_audio_pairs` reference chunks (one per song where possible), drawn from the fixed index set. Every val epoch, `_save_val_audio` loads them directly from disk and saves LQ/HQ/Restored triplets.

**Planned — hard example mining:** Future enhancement to replace the random fixed-set selection with a difficulty-ranked selection. On first val run, score all chunks by SI-SDR loss, lock the N hardest as the fixed evaluation set. Harder chunks give a more sensitive signal for detecting genuine model improvement vs easy material that saturates early. Implementation deferred — current dataset is homogeneous enough that random selection is sufficient.

**Perceptual val metrics:** After each val run, two additional metrics are computed alongside SI-SDR and stored on the module as `_last_val_msstft` and `_last_val_sfr`. `StepPrinter` reads these and emits one consolidated `[val]` line. `msstft` is multi-scale log-STFT loss across three window sizes (2048/1024/512). `sfr` is spectral flatness ratio in the 8-22kHz band — values above 1.05 indicate high-frequency noise injection (early overfitting signal). SI-SDR remains the checkpoint selection and early stopping metric. All three log to TensorBoard.

**Step printer:** TQDM is disabled. `StepPrinter` in `train.py` prints one line per batch with wall-clock measured it/s. Val time is excluded from the rate calculation so it/s stays accurate before and after val runs. On resume mid-epoch, the timer starts from the first batch of the new session.

**Alignment:** Runs at chunk time. `align_data` accepts an integer offset in samples — positive trims LQ, negative trims HQ. Applied via `frame_offset` at decode time, no extra memory cost. iTunes-encoded MP3s have a consistent 1057-sample encoder delay. Set `false` to disable.

**Augmentation internals:** `normalize_pair` scales by joint peak — pre-existing flat-top clipping is preserved as valid training signal, not clamped away. `stereo_alternation` picks L or R by sample index (even → L, odd → R) — `prob: 1.0` alternates systematically, does not randomly collapse to mono.

**Inference normalization:** Input normalized by peak before chunked inference, rescaled back after. Preserves transient headroom — hot MP3 transients above 1.0 at decode time are handled correctly.

# @claude last-modified: 2026-08-25T00:00:00Z
# @claude last-commit: 0.5.1.0 -- replace msstft/hfmae with VISQOL live; add configurable target_band_loss (off by default)
###
# Modified from original Apollo audio_litmodule.py
# Changes:
#   - Removed sync_dist=True from all log calls (causes hangs on single GPU)
#   - Removed all_gather in validation (multi-GPU only)
#   - Removed WandB-specific logger calls
#   - val_save_interval / val_audio_dir: save restored audio from val dataloader
#   - Val saved files: exactly 3 songs x 3 files (LQ/HQ/Restored) = 9 files per run
#   - Rotation schedule: derived automatically from total configured steps so that
#     every val song gets equal coverage by end of training. Manual override via
#     val_rotate_every (int or "auto"). Schedule locked at first val run and
#     checkpointed so resume never changes the sequence.
#   - File writes moved to background thread (training resumes immediately)
#   - Perceptual metrics: msstft, sfr, hf_band_mae added to val logging
#   - Timer fix: val timer now stops after module hook (captures audio save time)
#   - Step fix: val_check_interval interpreted in optimizer steps not batches
###
import os
import threading
import torchaudio
from omegaconf import OmegaConf
import torch
import torch.utils.checkpoint as torch_checkpoint
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections.abc import MutableMapping
from omegaconf import ListConfig

def flatten_dict(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


# ---------------------------------------------------------------------------
# Perceptual metric helpers
# ---------------------------------------------------------------------------

def _ms_log_stft_loss(est: "torch.Tensor", ref: "torch.Tensor") -> float:
    """
    Multi-scale log-magnitude STFT loss. Lower = better match to HQ.
    Not used in live training as of 0.5.1.0 (superseded by VISQOL) -- kept
    for evaluate.py's offline legacy-checkpoint scoring.
    """
    windows = [512, 1024, 2048]
    total = 0.0
    for n_fft in windows:
        hop = n_fft // 4
        win = torch.hann_window(n_fft, device=est.device)
        def _mag(x):
            return torch.stft(x.reshape(-1, x.shape[-1]),
                              n_fft=n_fft, hop_length=hop, win_length=n_fft,
                              window=win, return_complex=True).abs()
        e_mag = _mag(est)
        r_mag = _mag(ref)
        eps = 1e-7
        total += torch.mean(torch.abs(torch.log(e_mag + eps) - torch.log(r_mag + eps))).item()
    return total / len(windows)


def _hf_band_mae_cpu(est: "torch.Tensor", ref: "torch.Tensor",
                     sr: int = 44100,
                     lo_hz: float = 13000.0,
                     hi_hz: float = 19000.0) -> float:
    """
    Mean absolute log-magnitude error in the 13-19 kHz transition band.
    Not used in live training as of 0.5.1.0 (superseded by configurable
    target_band_loss) -- kept for evaluate.py's offline legacy-checkpoint scoring.
    """
    return _target_band_mae(est, ref, sr=sr, lo_hz=lo_hz, hi_hz=hi_hz)


def _spectral_flatness_ratio(est: "torch.Tensor", ref: "torch.Tensor", sr: int = 44100) -> float:
    """
    Spectral flatness ratio in the 8-22 kHz band: est_flatness / ref_flatness.
    > 1.0 means the restored signal is noisier than HQ in the high band.
    Rising over training = overfitting / noise injection. Canary signal only.
    """
    n_fft = 2048
    hop   = 512
    win   = torch.hann_window(n_fft, device=est.device)
    bin_lo = int(8000  / (sr / n_fft))
    bin_hi = min(int(22000 / (sr / n_fft)), n_fft // 2)

    def _flatness(x):
        mag = torch.stft(x.reshape(-1, x.shape[-1]),
                         n_fft=n_fft, hop_length=hop, win_length=n_fft,
                         window=win, return_complex=True).abs()
        band = mag[:, bin_lo:bin_hi, :].clamp(min=1e-10)
        log_mean   = band.log().mean()
        arith_mean = band.mean().log()
        return (log_mean - arith_mean).exp().item()

    eps = 1e-8
    return (_flatness(est) + eps) / (_flatness(ref) + eps)


def _target_band_mae(est: "torch.Tensor", ref: "torch.Tensor",
                     sr: int = 44100,
                     lo_hz: float = 13000.0,
                     hi_hz: float = 19000.0) -> float:
    """
    Mean absolute log-magnitude error in a configurable frequency band.
    Lower = better. Disabled by default; enabled via cfg.metrics.target_band_loss.
    """
    n_fft = 2048
    hop   = n_fft // 4
    win   = torch.hann_window(n_fft, device=est.device)
    bin_lo = int(lo_hz / (sr / n_fft))
    bin_hi = min(int(hi_hz / (sr / n_fft)), n_fft // 2)

    def _mag(x):
        return torch.stft(x.reshape(-1, x.shape[-1]),
                          n_fft=n_fft, hop_length=hop, win_length=n_fft,
                          window=win, return_complex=True).abs()

    eps = 1e-7
    e_mag = _mag(est)[:, bin_lo:bin_hi, :]
    r_mag = _mag(ref)[:, bin_lo:bin_hi, :]
    return torch.mean(torch.abs(torch.log(e_mag + eps) - torch.log(r_mag + eps))).item()


# VISQOL loader -- lazy, cached, gracefully absent
_visqol_api = None
_visqol_available = None

def _get_visqol_api():
    # Uses the "visqol-python" package (pure-Python port, pip-installable on
    # Windows -- the official google/visqol pybind11 bindings only ship for
    # Linux/Mac and require a Bazel build from source).
    global _visqol_api, _visqol_available
    if _visqol_available is False:
        return None
    if _visqol_api is not None:
        return _visqol_api
    try:
        from visqol import VisqolApi
        api = VisqolApi()
        api.create(mode="audio")
        _visqol_api = api
        _visqol_available = True
        return _visqol_api
    except Exception as e:
        print(f"[visqol] Not available ({e}) -- VISQOL will be skipped.")
        _visqol_available = False
        return None


def _visqol_score(est: "torch.Tensor", ref: "torch.Tensor", sr: int = 44100) -> float | None:
    """
    Perceptual VISQOL MOS-LQO score. Higher = better (1-5 scale).
    Runs on mono mix, resampled to 48kHz. Returns None if VISQOL unavailable.
    """
    api = _get_visqol_api()
    if api is None:
        return None
    try:
        import librosa, numpy as np
        mono_est = est.mean(0).cpu().numpy().astype(np.float64)
        mono_ref = ref.mean(0).cpu().numpy().astype(np.float64)
        if sr != 48000:
            mono_est = librosa.resample(mono_est, orig_sr=sr, target_sr=48000)
            mono_ref = librosa.resample(mono_ref, orig_sr=sr, target_sr=48000)
        result = api.measure_from_arrays(mono_ref, mono_est, 48000)
        return float(result.moslqo)
    except Exception as e:
        print(f"[visqol] Measure failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Main module
# ---------------------------------------------------------------------------

class AudioLightningModule(pl.LightningModule):
    def __init__(
        self,
        model=None,
        discriminator=None,
        optimizer=None,
        loss_func=None,
        metrics=None,
        scheduler=None,
        val_save_interval=5,
        val_audio_dir=None,
        val_songs=3,                # number of val songs to evaluate (full-song metrics + saved audio)
        val_rotate_every="auto",    # "auto" or int steps between preview-clip position rotations
        gradient_checkpointing=False,
        grad_accum_steps=1,
        # Configurable target band loss (off by default)
        target_band_loss_enabled=False,
        target_band_loss_lo_hz=13000.0,
        target_band_loss_hi_hz=19000.0,
        # VISQOL: fraction of val audio pairs to score (0.0 = off, 1.0 = all)
        visqol_fraction=1.0,
    ):
        super().__init__()
        self.audio_model      = model
        self.discriminator    = discriminator
        self.optimizer        = list(optimizer)
        self.loss_func        = loss_func
        self.metrics          = metrics
        self.scheduler        = list(scheduler)
        self.val_save_interval    = val_save_interval
        self.val_audio_dir        = val_audio_dir
        self.val_songs            = val_songs
        self.val_rotate_every     = val_rotate_every
        self.target_band_loss_enabled = target_band_loss_enabled
        self.target_band_loss_lo_hz   = target_band_loss_lo_hz
        self.target_band_loss_hi_hz   = target_band_loss_hi_hz
        self.visqol_fraction          = max(0.0, min(1.0, float(visqol_fraction)))

        # Val fixed-index lock (for val_loss / SI-SDR computation in validation_step)
        self._val_fixed_indices = None   # set[int] locked after first real val run
        self._val_seen_indices  = []     # accumulator during first run

        # Full-song val refs: locked once for metric computation, never changes.
        self._val_song_refs: dict = {}   # song_key -> (lq_path, hq_path)

        # Pending preview tensors from _compute_val_metrics, consumed by _save_val_audio
        self._pending_preview_data: list  = []

        # Gradient accumulation state
        self.grad_accum_steps = max(1, grad_accum_steps)
        self._accum_loss_g    = None
        self._accum_loss_d    = None
        self._accum_step      = 0

        # Val loss accumulator
        self._val_loss_sum   = 0.0
        self._val_loss_count = 0

        # Last val metric values (read by StepPrinter in train.py)
        self._last_val_sisdr  = None
        self._last_val_sdr    = None
        self._last_val_sfr    = None
        self._last_val_visqol = None
        self._last_val_tbl    = None   # target_band_loss (None when disabled)

        # Background write thread tracking
        self._write_thread: threading.Thread | None = None

        if gradient_checkpointing:
            self._enable_gradient_checkpointing()

        self.default_monitor     = "val_loss"
        self.validation_step_outputs = []
        self.test_step_outputs   = []
        self.automatic_optimization = False

    # ------------------------------------------------------------------
    # Gradient checkpointing
    # ------------------------------------------------------------------

    def _enable_gradient_checkpointing(self):
        """
        Wrap BSNet layers with torch.utils.checkpoint so intermediate activations
        are recomputed during backward instead of stored -- trades ~30% compute for
        large VRAM savings.

        FrequencyDiscriminator is deliberately NOT checkpointed: feature matching
        requires a single forward pass that returns both output and hidden feature
        maps with live gradients.
        """
        import look2hear.models.apollo as _apollo_mod
        original_bsnet_forward = _apollo_mod.BSNet.forward

        def checkpointed_bsnet_forward(self_bsnet, input):
            if not torch.is_grad_enabled():
                return original_bsnet_forward(self_bsnet, input)
            return torch_checkpoint.checkpoint(
                original_bsnet_forward, self_bsnet, input, use_reentrant=False
            )

        _apollo_mod.BSNet.forward = checkpointed_bsnet_forward
        print("[gradient_checkpointing] Enabled for BSNet layers only.")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, wav):
        return self.audio_model(wav)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_nb):
        ori_data, codec_data = batch
        optimizer_g, optimizer_d = self.optimizers()

        is_last_accum = ((self._accum_step + 1) % self.grad_accum_steps == 0)

        if self._accum_step % self.grad_accum_steps == 0:
            optimizer_g.zero_grad()
            optimizer_d.zero_grad()

        # --- Generator forward (under Lightning's AMP context) ---
        output = self(codec_data)

        # --- Discriminator update ---
        # output detached so D grads don't flow into the generator.
        est_outputs_d, _ = self.discriminator(output.detach(), sample_rate=44100)
        target_outputs, _ = self.discriminator(ori_data, sample_rate=44100)
        loss_d = self.loss_func["d"](target_outputs, est_outputs_d) / self.grad_accum_steps

        self._accum_loss_d = (self._accum_loss_d or 0.0) + loss_d.detach()
        self.manual_backward(loss_d)

        if is_last_accum:
            self.clip_gradients(optimizer_d, gradient_clip_val=5, gradient_clip_algorithm="norm")
            optimizer_d.step()

        # --- Generator update ---
        # Fresh discriminator forward on real audio for targets_feature_maps --
        # separate pass, live gradients, not reused from D step.
        est_outputs, est_feature_maps = self.discriminator(output, sample_rate=44100)
        _, targets_feature_maps = self.discriminator(ori_data, sample_rate=44100)
        loss_g = self.loss_func["g"](
            est_outputs, est_feature_maps, targets_feature_maps, output, ori_data
        ) / self.grad_accum_steps

        self._accum_loss_g = (self._accum_loss_g or 0.0) + loss_g.detach()
        self.manual_backward(loss_g)

        if is_last_accum:
            self.clip_gradients(optimizer_g, gradient_clip_val=5, gradient_clip_algorithm="norm")
            optimizer_g.step()

            self.log("train_loss_d", float(self._accum_loss_d), on_step=True, prog_bar=True, logger=True)
            self.log("train_loss_g", float(self._accum_loss_g), on_step=True, prog_bar=True, logger=True)

            self._accum_loss_g = None
            self._accum_loss_d = None

        self._accum_step += 1

    def on_train_epoch_end(self):
        if self._accum_loss_g is not None:
            try:
                optimizer_g, optimizer_d = self.optimizers()
                self.clip_gradients(optimizer_d, gradient_clip_val=5, gradient_clip_algorithm="norm")
                optimizer_d.step()
                self.clip_gradients(optimizer_g, gradient_clip_val=5, gradient_clip_algorithm="norm")
                optimizer_g.step()
                self.log("train_loss_d", float(self._accum_loss_d), on_step=False, prog_bar=False, logger=True)
                self.log("train_loss_g", float(self._accum_loss_g), on_step=False, prog_bar=False, logger=True)
            except AssertionError:
                pass
            self._accum_loss_g = None
            self._accum_loss_d = None
        self._accum_step = 0

        scheduler_g, scheduler_d = self.lr_schedulers()
        scheduler_g.step()
        scheduler_d.step()

    # ------------------------------------------------------------------
    # Validation step
    # ------------------------------------------------------------------

    def validation_step(self, batch, batch_nb):
        ori_data, codec_data, ds_idx, song_key = batch
        ds_idx   = int(ds_idx[0])
        song_key = song_key[0] if isinstance(song_key, (list, tuple)) else song_key

        if self.trainer.sanity_checking:
            est_sources = self(codec_data)
            loss = self.metrics(est_sources, ori_data)
            return {"val_loss": loss}

        # First run: collect all seen indices for locking later
        if self._val_fixed_indices is None:
            self._val_seen_indices.append(ds_idx)

        # Once locked, skip chunks not in the fixed set
        if self._val_fixed_indices is not None and ds_idx not in self._val_fixed_indices:
            return {"val_loss": None}

        est_sources = self(codec_data)
        loss = self.metrics(est_sources, ori_data)

        self._val_loss_sum   += float(loss)
        self._val_loss_count += 1
        self.validation_step_outputs.append(float(loss))

        return {"val_loss": loss}

    # ------------------------------------------------------------------
    # Val index locking
    # ------------------------------------------------------------------

    def _lock_val_fixed_indices(self):
        """Stratified sample -- equal chunks per song -- locked for all future runs."""
        import random

        dataset = self.trainer.datamodule.data_val
        seen    = self._val_seen_indices
        if not seen:
            return

        by_song = {}
        for ds_idx in seen:
            pair_idx, _ = dataset.index[ds_idx]
            _, hq_path  = dataset.pairs[pair_idx]
            stem  = os.path.splitext(os.path.basename(hq_path))[0]
            parts = stem.rsplit("_", 1)
            key   = parts[0] if len(parts) == 2 and parts[1].isdigit() else stem
            by_song.setdefault(key, []).append(ds_idx)

        num_songs  = len(by_song)
        lv         = self.trainer.limit_val_batches
        total_n    = lv if isinstance(lv, int) else int(lv * len(dataset))
        per_song   = max(1, total_n // num_songs)

        fixed = set()
        for key, indices in by_song.items():
            k = min(per_song, len(indices))
            fixed.update(random.sample(indices, k))

        self._val_fixed_indices = fixed
        print(f"[val] Locked {len(fixed)} fixed indices -- {per_song} per song across {num_songs} songs.")

    # ------------------------------------------------------------------
    # Full-song val ref locking + preview rotation
    # ------------------------------------------------------------------

    def _lock_val_songs(self, by_song: dict) -> None:
        """Lock one LQ/HQ file path per val song. Fixed for the entire run."""
        dataset = self.trainer.datamodule.data_val
        self._val_song_refs = {}
        for song_key, indices in list(by_song.items())[:self.val_songs]:
            pair_idx, _ = dataset.index[indices[0]]
            lq_path, hq_path = dataset.pairs[pair_idx]
            self._val_song_refs[song_key] = (lq_path, hq_path)
        print(f"[val] Locked {len(self._val_song_refs)} songs for full-song metric evaluation.")



    # ------------------------------------------------------------------
    # Full-song inference helper
    # ------------------------------------------------------------------

    def _infer_full_song(self, lq_path: str) -> torch.Tensor:
        """
        Load a full LQ file and run chunked OLA inference over it.
        Returns restored [2, T] float32 CPU tensor, normalized to the input peak.
        """
        lq_full, _ = torchaudio.load(lq_path)
        if lq_full.shape[0] == 1:
            lq_full = lq_full.repeat(2, 1)
        peak    = lq_full.abs().max().clamp(min=1e-8)
        lq_norm = lq_full / peak

        try:
            chunk_sec = float(self.trainer.datamodule.segment_sec)
        except Exception:
            chunk_sec = 3.0

        chunk_samples   = max(1, int(round(chunk_sec * 44100)))
        overlap_samples = max(0, int(round(min(0.5, chunk_sec * 0.25) * 44100)))
        hop_samples     = max(1, chunk_samples - overlap_samples)

        T       = lq_norm.shape[-1]
        out_buf = torch.zeros(2, T)
        wt_buf  = torch.zeros(T)

        start = 0
        while start < T:
            end = min(start + chunk_samples, T)
            n   = end - start
            inp = lq_norm[..., start:end].unsqueeze(0).to(self.device)
            out = self.audio_model(inp)
            if out.ndim == 3:
                out = out[0]
            out = out.float().cpu()[..., :n]
            w = torch.hann_window(n, periodic=False) if n >= 2 else torch.ones(n)
            out_buf[..., start:end] += out * w
            wt_buf[start:end]       += w
            start += hop_samples

        return (out_buf / wt_buf.clamp(min=1e-8)).clamp(-1.0, 1.0)

    # ------------------------------------------------------------------
    # Val audio saving (background thread)
    # ------------------------------------------------------------------

    def _save_val_audio(self):
        """
        Save full-song LQ/HQ/Restored triplets from _pending_preview_data.
        Tensors are already computed by _compute_val_metrics -- no extra inference.
        """
        if self.val_audio_dir is None or not self._pending_preview_data:
            return

        epoch_dir = os.path.join(self.val_audio_dir, f"step_{self.global_step:06d}")
        os.makedirs(epoch_dir, exist_ok=True)

        write_jobs = [(song_key, lq_t.clone(), hq_t.clone(), restored_t.clone())
                      for song_key, lq_t, hq_t, restored_t in self._pending_preview_data]
        self._pending_preview_data = []

        if self._write_thread is not None and self._write_thread.is_alive():
            self._write_thread.join(timeout=120)

        def _write_files():
            for tag, lq_s, hq_s, out_s in write_jobs:
                try:
                    torchaudio.save(os.path.join(epoch_dir, f"{tag}_LQ.wav"),       lq_s,  44100)
                    torchaudio.save(os.path.join(epoch_dir, f"{tag}_HQ.wav"),       hq_s,  44100)
                    torchaudio.save(os.path.join(epoch_dir, f"{tag}_Restored.wav"), out_s, 44100)
                except Exception as ex:
                    print(f"[val] Write error {tag}: {ex}")

        self._write_thread = threading.Thread(target=_write_files, daemon=True)
        self._write_thread.start()

    def _compute_val_metrics(self):
        """
        Run full-song OLA inference on each locked metric song.
        Computes val_sdr / val_sfr / val_visqol over complete files.
        Stores restored tensors in _pending_preview_data for _save_val_audio().
        """
        if not self._val_song_refs:
            return

        import look2hear.losses as _ll
        from paired_datamodule import normalize_pair

        _sdr_fn = _ll.MultiSrcNegSDR("snr", zero_mean=True)

        sfr_sum = sdr_sum = visqol_sum = tbl_sum = 0.0
        count = visqol_count = tbl_count = 0

        song_items = list(self._val_song_refs.items())
        do_visqol  = set(range(len(song_items))) if self.visqol_fraction >= 1.0 else \
                     set(range(max(1, round(len(song_items) * self.visqol_fraction))))

        preview_data = []

        self.audio_model.eval()
        with torch.no_grad():
            for i, (song_key, (lq_path, hq_path)) in enumerate(song_items):
                try:
                    lq_full, _ = torchaudio.load(lq_path)
                    hq_full, _ = torchaudio.load(hq_path)
                    if lq_full.shape[0] == 1: lq_full = lq_full.repeat(2, 1)
                    if hq_full.shape[0] == 1: hq_full = hq_full.repeat(2, 1)

                    peak    = lq_full.abs().max().clamp(min=1e-8)
                    lq_norm = (lq_full / peak).clamp(-1.0, 1.0)
                    hq_norm = (hq_full / peak).clamp(-1.0, 1.0)

                    restored = self._infer_full_song(lq_path)

                    e = restored[0:1]
                    r = hq_norm[0:1]

                    sfr_sum += _spectral_flatness_ratio(e, r)
                    sdr_sum += -float(_sdr_fn(e.unsqueeze(0), r.unsqueeze(0)).mean())
                    count   += 1

                    if i in do_visqol:
                        v = _visqol_score(e, r)
                        if v is not None:
                            visqol_sum   += v
                            visqol_count += 1

                    if self.target_band_loss_enabled:
                        tbl_sum   += _target_band_mae(e, r,
                                                      lo_hz=self.target_band_loss_lo_hz,
                                                      hi_hz=self.target_band_loss_hi_hz)
                        tbl_count += 1

                    preview_data.append((song_key, lq_norm, hq_norm, restored))

                except Exception as ex:
                    print(f"[val] Error on {song_key}: {ex}")

        self.audio_model.train()

        if count > 0:
            self._last_val_sdr = sdr_sum / count
            self._last_val_sfr = sfr_sum / count
        if visqol_count > 0:
            self._last_val_visqol = visqol_sum / visqol_count
        if tbl_count > 0:
            self._last_val_tbl = tbl_sum / tbl_count

        self._pending_preview_data = preview_data

    # ------------------------------------------------------------------
    # Validation epoch end
    # ------------------------------------------------------------------

    def on_validation_epoch_end(self):
        self._last_val_sisdr  = None
        self._last_val_sdr    = None
        self._last_val_sfr    = None
        self._last_val_visqol = None
        self._last_val_tbl    = None

        if self._val_loss_count > 0:
            avg_val_loss = self._val_loss_sum / self._val_loss_count
            self.log("val_loss", avg_val_loss, prog_bar=True, logger=True)
            self._last_val_sisdr = avg_val_loss
        self._val_loss_sum   = 0.0
        self._val_loss_count = 0
        self.log("lr", self.optimizer[0].param_groups[0]["lr"], prog_bar=True)
        self.validation_step_outputs.clear()

        if self.trainer.sanity_checking:
            return

        # --- Lock fixed indices on first real val run ---
        if self._val_fixed_indices is None and self._val_seen_indices:
            self._lock_val_fixed_indices()

        # --- Lock full-song metric refs on first real val run ---
        if not self._val_song_refs and self._val_fixed_indices is not None:
            dataset   = self.trainer.datamodule.data_val
            by_song   = {}
            all_songs = []
            seen_keys = set()
            for ds_idx in self._val_seen_indices or self._val_fixed_indices:
                pair_idx, _ = dataset.index[ds_idx]
                _, hq_path  = dataset.pairs[pair_idx]
                stem  = os.path.splitext(os.path.basename(hq_path))[0]
                parts = stem.rsplit("_", 1)
                key   = parts[0] if len(parts) == 2 and parts[1].isdigit() else stem
                by_song.setdefault(key, []).append(ds_idx)
                if key not in seen_keys:
                    seen_keys.add(key)
                    all_songs.append(key)
            self._lock_val_songs(by_song)
            print(f"[val] {len(self._val_song_refs)} songs locked for full-song metric evaluation.")

        # Full-song metrics pass (also populates _pending_preview_data), then save files
        self._compute_val_metrics()
        self._save_val_audio()

        _sfr    = self._last_val_sfr
        _sdr    = self._last_val_sdr
        _visqol = self._last_val_visqol
        _tbl    = self._last_val_tbl

        self.log("val_sfr",    float(_sfr)    if _sfr    is not None else 0.0, prog_bar=False, logger=True)
        self.log("val_sdr",    float(_sdr)    if _sdr    is not None else 0.0, prog_bar=False, logger=True)
        self.log("val_visqol", float(_visqol) if _visqol is not None else 0.0, prog_bar=False, logger=True)
        if self.target_band_loss_enabled:
            self.log("val_tbl", float(_tbl) if _tbl is not None else 0.0, prog_bar=False, logger=True)

    # ------------------------------------------------------------------
    # Checkpoint persistence
    # ------------------------------------------------------------------

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        checkpoint["val_fixed_indices"] = self._val_fixed_indices
        checkpoint["val_song_refs"]     = self._val_song_refs

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        self._val_fixed_indices = checkpoint.get("val_fixed_indices", None)
        self._val_song_refs     = checkpoint.get("val_song_refs",     {})

        # Strip state_dict keys from older checkpoints that no longer exist in model
        sd = checkpoint.get("state_dict", {})
        unexpected = [k for k in list(sd.keys()) if
                      any(k.startswith(p) for p in [
                          "loss_func.g.hann_", "loss_func.g.weights_",
                          "loss_func.d.hann_", "loss_func.d.weights_",
                      ])]
        for k in unexpected:
            del sd[k]
        checkpoint["state_dict"] = sd

        # Clear seen-indices accumulator on load (we already have fixed indices)
        self._val_seen_indices = []

    # ------------------------------------------------------------------
    # Test
    # ------------------------------------------------------------------

    def test_step(self, batch, batch_nb):
        mixtures, targets = batch
        est_sources = self(mixtures)
        loss = self.metrics(est_sources, targets)
        self.log("test_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.test_step_outputs.append(loss)
        return {"test_loss": loss}

    def on_test_epoch_end(self):
        self.log("lr", self.optimizer[0].param_groups[0]["lr"], on_epoch=True, prog_bar=True)
        self.test_step_outputs.clear()

    # ------------------------------------------------------------------
    # Optimizers
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        if self.scheduler is None:
            return self.optimizer
        if not isinstance(self.scheduler, (list, tuple)):
            self.scheduler = [self.scheduler]
        if not isinstance(self.optimizer, (list, tuple)):
            self.optimizer = [self.optimizer]

        epoch_schedulers = []
        for sched in self.scheduler:
            if not isinstance(sched, dict):
                if isinstance(sched, ReduceLROnPlateau):
                    sched = {"scheduler": sched, "monitor": self.default_monitor}
                epoch_schedulers.append(sched)
            else:
                sched.setdefault("monitor", self.default_monitor)
                sched.setdefault("frequency", 1)
                if sched["interval"] == "batch":
                    sched["interval"] = "step"
                assert sched["interval"] in ["epoch", "step"]
                epoch_schedulers.append(sched)
        return self.optimizer, epoch_schedulers

    @staticmethod
    def config_to_hparams(dic):
        dic = flatten_dict(dic)
        for k, v in dic.items():
            if v is None:
                dic[k] = str(v)
            elif isinstance(v, (list, tuple)):
                dic[k] = torch.tensor(v)
        return dic

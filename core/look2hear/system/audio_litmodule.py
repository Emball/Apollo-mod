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
    global _visqol_api, _visqol_available
    if _visqol_available is False:
        return None
    if _visqol_api is not None:
        return _visqol_api
    try:
        from visqol import visqol_lib_py
        from visqol.pb2 import visqol_config_pb2
        import os as _os
        cfg = visqol_config_pb2.VisqolConfig()
        cfg.audio.sample_rate = 48000
        cfg.options.use_speech_scoring = False
        cfg.options.svr_model_path = _os.path.join(
            _os.path.dirname(visqol_lib_py.__file__), "model", "libsvm_nu_svr_model.txt"
        )
        api = visqol_lib_py.VisqolApi()
        api.Create(cfg)
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
        return float(api.Measure(mono_ref, mono_est).moslqo)
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
        val_songs=3,                # number of songs to save audio for per val run
        val_rotate_every="auto",    # "auto" = derive from total steps; int = manual cadence
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
        self.val_save_interval = val_save_interval
        self.val_audio_dir    = val_audio_dir
        self.val_songs        = val_songs
        self.val_rotate_every = val_rotate_every
        self.target_band_loss_enabled = target_band_loss_enabled
        self.target_band_loss_lo_hz   = target_band_loss_lo_hz
        self.target_band_loss_hi_hz   = target_band_loss_hi_hz
        self.visqol_fraction          = max(0.0, min(1.0, float(visqol_fraction)))

        # Val fixed-index lock (for loss computation)
        self._val_fixed_indices = None   # set[int] locked after first real val run
        self._val_seen_indices  = []     # accumulator during first run

        # Val audio rotation schedule
        # _val_rotation_schedule: list of lists, each inner list is the song keys
        #   for that rotation slot. Computed once, checkpointed.
        # _val_run_count: how many real val runs have happened this training run.
        #   Checkpointed so resume picks up the right slot.
        self._val_rotation_schedule: list = []
        self._val_run_count: int          = 0
        self._val_locked_refs: dict       = {}  # song_key -> (lq_path, hq_path, start, seg_samples)

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

        amp_ctx = torch.amp.autocast("cuda", dtype=torch.float16)

        with amp_ctx:
            output = self(codec_data)

        # Discriminator update
        for p in self.discriminator.parameters():
            p.requires_grad_(True)

        with amp_ctx:
            est_outputs_d, _          = self.discriminator(output.detach(), sample_rate=44100)
            target_outputs, targets_feature_maps = self.discriminator(ori_data, sample_rate=44100)
            loss_d = self.loss_func["d"](target_outputs, est_outputs_d) / self.grad_accum_steps

        self._accum_loss_d = (self._accum_loss_d or 0.0) + loss_d.detach()
        self.manual_backward(loss_d)

        # Detach targets_feature_maps -- fixed reference for feature matching
        targets_feature_maps = [
            [f.detach() for f in fmap] for fmap in targets_feature_maps
        ]
        del est_outputs_d, target_outputs

        # Generator update
        for p in self.discriminator.parameters():
            p.requires_grad_(False)

        with amp_ctx:
            est_outputs, est_feature_maps = self.discriminator(output, sample_rate=44100)
            loss_g = self.loss_func["g"](
                est_outputs, est_feature_maps, targets_feature_maps, output, ori_data
            ) / self.grad_accum_steps

        self._accum_loss_g = (self._accum_loss_g or 0.0) + loss_g.detach()
        self.manual_backward(loss_g)
        del loss_g, loss_d, est_outputs, est_feature_maps, targets_feature_maps, output

        for p in self.discriminator.parameters():
            p.requires_grad_(True)

        if is_last_accum:
            scaler = getattr(self.trainer.precision_plugin, "scaler", None)
            if scaler is not None:
                scaler.unscale_(optimizer_d)
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 5.0)
            optimizer_d.step()

            if scaler is not None:
                scaler.unscale_(optimizer_g)
            torch.nn.utils.clip_grad_norm_(self.audio_model.parameters(), 5.0)
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
                _scaler = getattr(self.trainer.precision_plugin, "scaler", None)
                if _scaler is not None:
                    _scaler.unscale_(optimizer_d)
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 5.0)
                optimizer_d.step()
                if _scaler is not None:
                    _scaler.unscale_(optimizer_g)
                torch.nn.utils.clip_grad_norm_(self.audio_model.parameters(), 5.0)
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
    # Rotation schedule
    # ------------------------------------------------------------------

    def _build_rotation_schedule(self, all_songs: list) -> list:
        """
        Build the full rotation schedule for the entire training run.
        Each slot is a list of val_songs song keys.
        Slots are constructed so every song appears as equally often as possible.
        The cadence (slots between rotations) is derived from total configured
        steps when val_rotate_every="auto", otherwise uses the integer value.
        """
        import random
        import math

        n_songs  = len(all_songs)
        # Cap per_set so we always have at least 2 rotation slots (meaningful rotation).
        # e.g. 3 songs / 3 per_set = 1 slot = no rotation; reduce to 2 per_set = 2 slots.
        per_set  = min(self.val_songs, max(1, n_songs - 1)) if n_songs > 1 else n_songs

        # Derive total val runs from trainer config
        try:
            max_epochs    = self.trainer.max_epochs or 1
            batches       = self.trainer.num_training_batches or 1
            accum         = self.grad_accum_steps
            total_steps   = (max_epochs * batches) // accum
            val_interval  = self.trainer.val_check_interval or 100
            total_val_runs = max(1, total_steps // val_interval)
        except Exception:
            total_val_runs = 50  # safe fallback

        # Derive rotate_every
        if str(self.val_rotate_every).lower() == "auto":
            # How many slots do we need to cover every song at least once?
            n_slots = math.ceil(n_songs / per_set)
            rotate_every = max(1, total_val_runs // n_slots)
        else:
            rotate_every = int(self.val_rotate_every)

        total_slots = max(1, math.ceil(total_val_runs / rotate_every))

        # Build slots by cycling through shuffled song list
        shuffled = all_songs[:]
        random.shuffle(shuffled)
        # Tile enough copies to fill all slots
        tiled = (shuffled * math.ceil((total_slots * per_set) / max(n_songs, 1)))

        schedule = []
        for i in range(total_slots):
            offset = (i * per_set) % len(tiled)
            slot   = tiled[offset : offset + per_set]
            # wrap-around
            if len(slot) < per_set:
                slot += tiled[: per_set - len(slot)]
            schedule.append(slot)

        print(f"[val audio] Rotation schedule: {n_songs} songs, "
              f"{per_set} per set, rotate every {rotate_every} runs, "
              f"{total_slots} slots total.")
        return schedule

    def _current_slot_songs(self) -> list:
        """Return the song keys for the current val run's slot."""
        if not self._val_rotation_schedule:
            return []
        slot_idx = (self._val_run_count - 1) // max(1, self._rotate_every_resolved())
        slot_idx = min(slot_idx, len(self._val_rotation_schedule) - 1)
        return self._val_rotation_schedule[slot_idx]

    def _rotate_every_resolved(self) -> int:
        """Return the resolved integer rotate_every cadence."""
        if not self._val_rotation_schedule:
            return 1
        # Derive from schedule length vs total slots (stored in _val_rotate_cadence)
        return getattr(self, "_val_rotate_cadence", 5)

    # ------------------------------------------------------------------
    # Val audio reference building
    # ------------------------------------------------------------------

    def _lock_val_refs(self) -> None:
        """
        Called once when the rotation schedule is first built.
        Picks one stable non-silent chunk per song from the fixed index set
        and stores it in _val_locked_refs. These refs never change for the
        lifetime of the run, so the same audio is always compared across
        val steps for a given song.
        """
        import random
        import torchaudio as _ta

        dataset = self.trainer.datamodule.data_val
        fixed   = self._val_fixed_indices or set()
        _SILENCE_RMS = 0.01

        by_song = {}
        for ds_idx in fixed:
            pair_idx, _ = dataset.index[ds_idx]
            _, hq_path  = dataset.pairs[pair_idx]
            stem  = os.path.splitext(os.path.basename(hq_path))[0]
            parts = stem.rsplit("_", 1)
            key   = parts[0] if len(parts) == 2 and parts[1].isdigit() else stem
            by_song.setdefault(key, []).append(ds_idx)

        locked = {}
        for song_key, candidates in by_song.items():
            shuffled = candidates[:]
            random.shuffle(shuffled)
            chosen = shuffled[0]
            for ds_idx in shuffled:
                pair_idx, s = dataset.index[ds_idx]
                lq_p, _ = dataset.pairs[pair_idx]
                try:
                    wav, _ = _ta.load(lq_p, frame_offset=s, num_frames=dataset.segment_samples)
                    if wav.pow(2).mean().sqrt().item() >= _SILENCE_RMS:
                        chosen = ds_idx
                        break
                except Exception:
                    pass
            pair_idx, start = dataset.index[chosen]
            lq_path, hq_path = dataset.pairs[pair_idx]
            locked[song_key] = (lq_path, hq_path, start, dataset.segment_samples)

        self._val_locked_refs = locked
        print(f"[val audio] Locked {len(locked)} stable chunk refs.")

    def _build_val_refs_for_slot(self, slot_songs: list) -> list:
        """
        Given a list of song keys for this slot, return the pre-locked refs.
        Each song always uses the exact same chunk across all val runs.
        """
        refs = []
        for song_key in slot_songs:
            if song_key in self._val_locked_refs:
                lq_path, hq_path, start, seg_samples = self._val_locked_refs[song_key]
                refs.append((song_key, lq_path, hq_path, start, seg_samples))
        return refs

    # ------------------------------------------------------------------
    # Val audio saving (background thread)
    # ------------------------------------------------------------------

    def _save_val_audio(self):
        """
        Run current slot's ref chunks through the model, then hand off the
        tensors to a background thread for disk writes + perceptual metrics.
        Training resumes immediately after the inference pass.
        """
        if self.val_audio_dir is None:
            return
        if not self._val_rotation_schedule:
            return

        slot_songs = self._current_slot_songs()
        if not slot_songs:
            return

        refs = self._build_val_refs_for_slot(slot_songs)
        if not refs:
            return

        from paired_datamodule import normalize_pair

        epoch_dir = os.path.join(self.val_audio_dir, f"step_{self.global_step:06d}")
        os.makedirs(epoch_dir, exist_ok=True)

        # Run inference synchronously (GPU needed)
        perc_pairs = []
        self.audio_model.eval()
        with torch.no_grad():
            for song_key, lq_path, hq_path, start, seg_samples in refs:
                try:
                    lq, _ = torchaudio.load(lq_path, frame_offset=start, num_frames=seg_samples)
                    hq, _ = torchaudio.load(hq_path, frame_offset=start, num_frames=seg_samples)

                    if lq.shape[0] == 1: lq = lq.repeat(2, 1)
                    if hq.shape[0] == 1: hq = hq.repeat(2, 1)

                    lq_norm, hq_norm = normalize_pair(lq, hq)

                    inp = lq_norm.unsqueeze(0).to(self.device)
                    out = self.audio_model(inp)
                    if out.ndim == 3:
                        out = out[0]
                    out     = out.float().cpu().clamp(-1.0, 1.0)
                    lq_save = lq_norm.float().clamp(-1.0, 1.0)
                    hq_save = hq_norm.float().clamp(-1.0, 1.0)

                    perc_pairs.append((song_key, lq_save.clone(), hq_save.clone(), out.clone()))
                except Exception as e:
                    print(f"[val audio] Inference error {song_key}: {e}")

        self.audio_model.train()

        # Compute perceptual metrics synchronously -- must finish before
        # on_validation_epoch_end returns so Lightning can log into checkpoint filenames.
        import random as _random
        import look2hear.losses as _ll
        _sdr_fn = _ll.MultiSrcNegSDR("snr", zero_mean=True)

        sfr_sum    = 0.0
        sdr_sum    = 0.0
        visqol_sum = 0.0
        tbl_sum    = 0.0
        visqol_count = 0
        tbl_count    = 0
        count        = 0

        # Determine which pairs to run VISQOL on (fraction-based sampling)
        visqol_indices = set()
        if self.visqol_fraction > 0.0 and len(perc_pairs) > 0:
            n_visqol = max(1, round(len(perc_pairs) * self.visqol_fraction))
            visqol_indices = set(_random.sample(range(len(perc_pairs)), min(n_visqol, len(perc_pairs))))

        for i, (song_key, lq_save, hq_save, out) in enumerate(perc_pairs):
            try:
                e = out[0:1]      if out.ndim     == 2 else out
                r = hq_save[0:1]  if hq_save.ndim == 2 else hq_save
                sfr_sum += _spectral_flatness_ratio(e, r)
                sdr_sum += -float(_sdr_fn(e.unsqueeze(0), r.unsqueeze(0)).mean())
                count   += 1

                if i in visqol_indices:
                    v = _visqol_score(e, r)
                    if v is not None:
                        visqol_sum   += v
                        visqol_count += 1

                if self.target_band_loss_enabled:
                    tbl_sum   += _target_band_mae(e, r,
                                                  lo_hz=self.target_band_loss_lo_hz,
                                                  hi_hz=self.target_band_loss_hi_hz)
                    tbl_count += 1

            except Exception as ex:
                print(f"[val audio] Metric error {song_key}: {ex}")

        if count > 0:
            self._last_val_sfr    = sfr_sum / count
            self._last_val_sdr    = sdr_sum / count
        if visqol_count > 0:
            self._last_val_visqol = visqol_sum / visqol_count
        if tbl_count > 0:
            self._last_val_tbl = tbl_sum / tbl_count

        # Disk writes go async -- training resumes immediately after metrics are logged.
        # Join any previous write thread first.
        if self._write_thread is not None and self._write_thread.is_alive():
            self._write_thread.join(timeout=60)

        def _write_files():
            for song_key, lq_save, hq_save, out in perc_pairs:
                try:
                    torchaudio.save(os.path.join(epoch_dir, f"{song_key}_LQ.wav"),       lq_save, 44100)
                    torchaudio.save(os.path.join(epoch_dir, f"{song_key}_HQ.wav"),       hq_save, 44100)
                    torchaudio.save(os.path.join(epoch_dir, f"{song_key}_Restored.wav"), out,     44100)
                except Exception as ex:
                    print(f"[val audio] Write error {song_key}: {ex}")

        self._write_thread = threading.Thread(target=_write_files, daemon=True)
        self._write_thread.start()

    # ------------------------------------------------------------------
    # Validation epoch end
    # ------------------------------------------------------------------

    def on_validation_epoch_end(self):
        # Reset metric slots -- filled synchronously in _save_val_audio()
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

        # --- Build rotation schedule on first real val run ---
        if not self._val_rotation_schedule and self._val_fixed_indices is not None:
            dataset = self.trainer.datamodule.data_val
            all_songs = []
            seen_keys = set()
            for ds_idx in self._val_fixed_indices:
                pair_idx, _ = dataset.index[ds_idx]
                _, hq_path  = dataset.pairs[pair_idx]
                stem  = os.path.splitext(os.path.basename(hq_path))[0]
                parts = stem.rsplit("_", 1)
                key   = parts[0] if len(parts) == 2 and parts[1].isdigit() else stem
                if key not in seen_keys:
                    seen_keys.add(key)
                    all_songs.append(key)

            self._val_rotation_schedule = self._build_rotation_schedule(all_songs)
            self._lock_val_refs()

            if str(self.val_rotate_every).lower() == "auto":
                import math
                try:
                    max_epochs   = self.trainer.max_epochs or 1
                    batches      = self.trainer.num_training_batches or 1
                    total_steps  = (max_epochs * batches) // self.grad_accum_steps
                    val_interval = self.trainer.val_check_interval or 200
                    total_val_runs = max(1, total_steps // val_interval)
                except Exception:
                    total_val_runs = 50
                n_songs  = len(all_songs)
                per_set  = min(self.val_songs, n_songs)
                n_slots  = math.ceil(n_songs / per_set)
                self._val_rotate_cadence = max(1, total_val_runs // n_slots)
            else:
                self._val_rotate_cadence = int(self.val_rotate_every)

            print(f"[val audio] {len(all_songs)} songs available, "
                  f"rotate every {self._val_rotate_cadence} val runs.")

        # Increment run counter then save audio (metrics computed synchronously inside)
        self._val_run_count += 1
        self._save_val_audio()

        # Log metrics so Lightning can interpolate into filenames.
        _sfr    = self._last_val_sfr
        _sdr    = self._last_val_sdr
        _visqol = self._last_val_visqol
        _tbl    = self._last_val_tbl
        _sisdr  = self._last_val_sisdr

        self.log("val_sfr",    float(_sfr)    if _sfr    is not None else 0.0, prog_bar=False, logger=True)
        self.log("val_sdr",    float(_sdr)    if _sdr    is not None else 0.0, prog_bar=False, logger=True)
        self.log("val_visqol", float(_visqol) if _visqol is not None else 0.0, prog_bar=False, logger=True)
        if self.target_band_loss_enabled:
            self.log("val_tbl", float(_tbl) if _tbl is not None else 0.0, prog_bar=False, logger=True)

    # ------------------------------------------------------------------
    # Checkpoint persistence
    # ------------------------------------------------------------------

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        checkpoint["val_fixed_indices"]      = self._val_fixed_indices
        checkpoint["val_rotation_schedule"]  = self._val_rotation_schedule
        checkpoint["val_run_count"]          = self._val_run_count
        checkpoint["val_rotate_cadence"]     = getattr(self, "_val_rotate_cadence", 5)
        checkpoint["val_locked_refs"]        = self._val_locked_refs

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        self._val_fixed_indices     = checkpoint.get("val_fixed_indices",     None)
        self._val_rotation_schedule = checkpoint.get("val_rotation_schedule", [])
        self._val_run_count         = checkpoint.get("val_run_count",         0)
        self._val_rotate_cadence    = checkpoint.get("val_rotate_cadence",    5)
        self._val_locked_refs       = checkpoint.get("val_locked_refs",       {})

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

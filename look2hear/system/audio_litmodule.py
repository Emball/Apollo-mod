# @claude last-modified: 2026-05-05T06:34:39Z
# @claude last-commit: feat: major update -- TUI, augmentation system, gradient checkpointing, optimization bootstrap
###
# Modified from original Apollo audio_litmodule.py
# Changes:
#   - Removed sync_dist=True from all log calls (causes hangs on single GPU)
#   - Removed all_gather in validation (multi-GPU only)
#   - Removed WandB-specific logger calls
#   - val_save_interval / val_audio_dir: save restored audio from val dataloader
#     every N epochs so what you hear == what the loss measures
###
import os
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

def _ms_log_stft_loss(est: "torch.Tensor", ref: "torch.Tensor") -> float:
    """Multi-scale log-magnitude STFT loss. Lower = better match to HQ."""
    import torch, math
    windows = [512, 1024, 2048]
    total = 0.0
    for n_fft in windows:
        hop = n_fft // 4
        win = torch.hann_window(n_fft, device=est.device)
        def _mag(x):
            # x: (C, T) -> (C, F, frames)
            return torch.stft(x.reshape(-1, x.shape[-1]),
                              n_fft=n_fft, hop_length=hop, win_length=n_fft,
                              window=win, return_complex=True).abs()
        e_mag = _mag(est)
        r_mag = _mag(ref)
        eps = 1e-7
        total += torch.mean(torch.abs(torch.log(e_mag + eps) - torch.log(r_mag + eps))).item()
    return total / len(windows)


def _spectral_flatness_ratio(est: "torch.Tensor", ref: "torch.Tensor", sr: int = 44100) -> float:
    """
    Spectral flatness ratio in the 8-22 kHz band: est_flatness / ref_flatness.
    > 1.0 means the restored signal is noisier than HQ in the high band.
    Rising over training checkpoints = overfitting / noise injection.
    """
    import torch, math
    n_fft = 2048
    hop   = 512
    win   = torch.hann_window(n_fft, device=est.device)
    # frequency bin range for 8-22 kHz
    bin_lo = int(8000  / (sr / n_fft))
    bin_hi = int(22000 / (sr / n_fft))
    bin_hi = min(bin_hi, n_fft // 2)

    def _flatness(x):
        mag = torch.stft(x.reshape(-1, x.shape[-1]),
                         n_fft=n_fft, hop_length=hop, win_length=n_fft,
                         window=win, return_complex=True).abs()
        band = mag[:, bin_lo:bin_hi, :].clamp(min=1e-10)
        log_mean = band.log().mean()
        arith_mean = band.mean().log()
        return (log_mean - arith_mean).exp().item()  # geometric/arithmetic = flatness

    eps = 1e-8
    est_f = _flatness(est) + eps
    ref_f = _flatness(ref) + eps
    return est_f / ref_f


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
        val_audio_pairs=10,
        gradient_checkpointing=False,
        grad_accum_steps=1,
    ):
        super().__init__()
        self.audio_model = model
        self.discriminator = discriminator
        self.optimizer = list(optimizer)
        self.loss_func = loss_func
        self.metrics = metrics
        self.scheduler = list(scheduler)
        self.val_save_interval = val_save_interval
        self.val_audio_dir = val_audio_dir
        self.val_audio_pairs = val_audio_pairs
        self._val_locked_refs    = None  # list of (song_key, lq_path, hq_path, start, seg_samples)
        self._val_fixed_indices  = None  # locked dataset indices for loss computation
        self.gradient_checkpointing = gradient_checkpointing
        self.grad_accum_steps = max(1, grad_accum_steps)
        self._accum_loss_g = None
        self._accum_loss_d = None
        self._accum_step   = 0
        self._val_loss_sum    = 0.0
        self._val_loss_count  = 0
        self._last_val_sisdr  = None
        self._last_val_msstft = None
        self._last_val_sfr    = None
        self._perc_thread     = None

        if gradient_checkpointing:
            self._enable_gradient_checkpointing()

        self.default_monitor = "val_loss"
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.automatic_optimization = False

    def _enable_gradient_checkpointing(self):
        """
        Wrap BSNet layers with torch.utils.checkpoint so intermediate activations
        are recomputed during backward instead of stored -- trades ~30% compute for
        large VRAM savings.

        FrequencyDiscriminator is deliberately NOT checkpointed: feature matching
        requires a single forward pass that returns both output and hidden feature
        maps with live gradients. Checkpointing it caused a double forward (once
        for output, once under no_grad for hiddens) with no correctness benefit.
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

    def forward(self, wav):
        return self.audio_model(wav)

    def training_step(self, batch, batch_nb):
        ori_data, codec_data = batch
        optimizer_g, optimizer_d = self.optimizers()
        scheduler_g, scheduler_d = self.lr_schedulers()

        is_last_accum = ((self._accum_step + 1) % self.grad_accum_steps == 0)

        # Zero grads at the start of each new accumulation window
        if self._accum_step % self.grad_accum_steps == 0:
            optimizer_g.zero_grad()
            optimizer_d.zero_grad()

        # Mixed precision context
        # Explicit autocast ensures every sub-op (STFT, conv, attention) runs
        # in fp16 rather than relying on implicit casting from Lightning alone.
        amp_ctx = torch.amp.autocast("cuda", dtype=torch.float16)

        with amp_ctx:
            output = self(codec_data)

        # Discriminator update
        for p in self.discriminator.parameters():
            p.requires_grad_(True)

        with amp_ctx:
            est_outputs_d, _ = self.discriminator(output.detach(), sample_rate=44100)
            target_outputs, targets_feature_maps = self.discriminator(ori_data, sample_rate=44100)
            loss_d = self.loss_func["d"](target_outputs, est_outputs_d) / self.grad_accum_steps

        self._accum_loss_d = (self._accum_loss_d or 0.0) + loss_d.detach()
        self.manual_backward(loss_d)

        # Detach targets_feature_maps -- fixed reference for feature matching;
        # dropping the D graph here is the primary VRAM saving.
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

        # Optimizer step -- only after accumulating grad_accum_steps batches
        if is_last_accum:
            self.clip_gradients(optimizer_d, gradient_clip_val=5, gradient_clip_algorithm="norm")
            optimizer_d.step()

            self.clip_gradients(optimizer_g, gradient_clip_val=5, gradient_clip_algorithm="norm")
            optimizer_g.step()

            # Log scalar values only -- .item() detaches from GPU, avoiding the
            # on_epoch=True accumulation bug (PL caches GPU tensors until epoch end).
            self.log("train_loss_d", float(self._accum_loss_d), on_step=True, prog_bar=True, logger=True)
            self.log("train_loss_g", float(self._accum_loss_g), on_step=True, prog_bar=True, logger=True)

            self._accum_loss_g = None
            self._accum_loss_d = None

        self._accum_step += 1

    def on_train_epoch_end(self):
        # Flush any partial accumulation window at epoch end so the final
        # batches of an epoch always contribute to a weight update.
        # Wrapped in try/except because AMP's grad scaler will reject the step
        # if training was interrupted mid-window (e.g. early stopping fires),
        # in which case the partial gradients are discarded harmlessly.
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
                # AMP scaler has no inf-checks recorded -- partial window with no
                # completed backward (happens on early stop mid-accumulation).
                pass
            self._accum_loss_g = None
            self._accum_loss_d = None
        self._accum_step = 0

        scheduler_g, scheduler_d = self.lr_schedulers()
        scheduler_g.step()
        scheduler_d.step()

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
            if not hasattr(self, '_val_seen_indices'):
                self._val_seen_indices = []
            self._val_seen_indices.append(ds_idx)

        # Once locked, skip chunks not in the fixed set
        if self._val_fixed_indices is not None and ds_idx not in self._val_fixed_indices:
            return {"val_loss": None}

        est_sources = self(codec_data)
        loss = self.metrics(est_sources, ori_data)

        self._val_loss_sum   += float(loss)
        self._val_loss_count += 1
        self.validation_step_outputs.append(float(loss))

        # Perceptual metrics are computed on locked reference chunks in _save_val_audio,
        # not per-batch, to avoid the CPU STFT overhead on every val step.

        return {"val_loss": loss}

    def _lock_val_fixed_indices(self):
        """Stratified sample -- equal chunks per song -- lock for all future runs."""
        import random

        dataset  = self.trainer.datamodule.data_val
        seen     = getattr(self, '_val_seen_indices', [])
        if not seen:
            return

        # Group seen indices by song
        by_song = {}
        for ds_idx in seen:
            pair_idx, _ = dataset.index[ds_idx]
            _, hq_path  = dataset.pairs[pair_idx]
            stem = os.path.splitext(os.path.basename(hq_path))[0]
            # Strip chunk number suffix (e.g. "Freak_0017" -> "Freak")
            parts = stem.rsplit("_", 1)
            song_key = parts[0] if len(parts) == 2 and parts[1].isdigit() else stem
            by_song.setdefault(song_key, []).append(ds_idx)

        num_songs   = len(by_song)
        lv          = self.trainer.limit_val_batches
        total_n     = lv if isinstance(lv, int) else int(lv * len(dataset))
        per_song    = max(1, total_n // num_songs)

        fixed = set()
        for song_key, indices in by_song.items():
            k = min(per_song, len(indices))
            fixed.update(random.sample(indices, k))

        self._val_fixed_indices = fixed
        print(f"[val] Locked {len(fixed)} fixed indices -- {per_song} per song across {num_songs} songs.")

    def _lock_val_refs(self):
        """Pick one chunk per song from fixed indices, store file paths for direct loading."""
        import random

        dataset = self.trainer.datamodule.data_val
        fixed = self._val_fixed_indices or set()
        if not fixed:
            return

        # Group fixed indices by song
        by_song = {}
        for ds_idx in fixed:
            pair_idx, _ = dataset.index[ds_idx]
            _, hq_path = dataset.pairs[pair_idx]
            song_key = os.path.splitext(os.path.basename(hq_path))[0]
            if song_key not in by_song:
                by_song[song_key] = []
            by_song[song_key].append(ds_idx)

        songs = list(by_song.keys())
        random.shuffle(songs)
        chosen_songs = songs[:self.val_audio_pairs]

        refs = []
        for song_key in chosen_songs:
            ds_idx = random.choice(by_song[song_key])
            pair_idx, start = dataset.index[ds_idx]
            lq_path, hq_path = dataset.pairs[pair_idx]
            seg_samples = dataset.segment_samples
            refs.append((song_key, lq_path, hq_path, start, seg_samples))

        self._val_locked_refs = refs
        print(f"[val audio] Locked {len(refs)} reference chunks: {[r[0] for r in refs]}")

    def _save_val_audio(self):
        """
        Run locked reference chunks through the model and save LQ/HQ/Restored triplets.
        Also computes perceptual metrics (msstft, sfr) on these chunks -- much cheaper
        than computing on every val batch since we only have 4 reference chunks.
        Called from on_validation_epoch_end -- training is paused so full GPU is available.
        """
        if not self._val_locked_refs or self.val_audio_dir is None:
            return

        from paired_datamodule import normalize_pair

        epoch_dir = os.path.join(self.val_audio_dir, f"step_{self.global_step:06d}")
        os.makedirs(epoch_dir, exist_ok=True)

        perc_pairs = []

        self.audio_model.eval()
        with torch.no_grad():
            for song_key, lq_path, hq_path, start, seg_samples in self._val_locked_refs:
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

                    torchaudio.save(os.path.join(epoch_dir, f"{song_key}_LQ.wav"),       lq_save, 44100)
                    torchaudio.save(os.path.join(epoch_dir, f"{song_key}_HQ.wav"),       hq_save, 44100)
                    torchaudio.save(os.path.join(epoch_dir, f"{song_key}_Restored.wav"), out,     44100)

                    # Collect tensors for perceptual metrics (computed in background thread)
                    perc_pairs.append((
                        out[0:1].clone() if out.ndim == 2 else out.clone(),
                        hq_save[0:1].clone() if hq_save.ndim == 2 else hq_save.clone(),
                        song_key,
                    ))

                except Exception as e:
                    print(f"[val audio] Error saving {song_key}: {e}")

        self.audio_model.train()

        # Compute perceptual metrics in a background thread so training resumes immediately
        import threading
        def _compute_perc(pairs, module):
            msstft_sum = 0.0
            sfr_sum    = 0.0
            count      = 0
            for e, r, sk in pairs:
                try:
                    msstft_sum += _ms_log_stft_loss(e, r)
                    sfr_sum    += _spectral_flatness_ratio(e, r)
                    count += 1
                except Exception as _me:
                    print(f"[val metrics] {sk}: {_me}")
            if count > 0:
                module._last_val_msstft = msstft_sum / count
                module._last_val_sfr    = sfr_sum    / count

        if perc_pairs:
            t = threading.Thread(target=_compute_perc, args=(perc_pairs, self), daemon=True)
            t.start()
            # Join before next val run so we don't pile up threads
            if hasattr(self, '_perc_thread') and self._perc_thread is not None:
                self._perc_thread.join(timeout=30)
            self._perc_thread = t

    def on_validation_epoch_end(self):
        # Wait for previous perceptual metric thread before logging new val results
        if self._perc_thread is not None:
            self._perc_thread.join(timeout=30)
            self._perc_thread = None

        self._last_val_sisdr  = None
        self._last_val_msstft = None
        self._last_val_sfr    = None

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

        # First real val run: lock fixed evaluation indices, then audio refs.
        # Only re-lock if still None after collecting seen indices this run.
        # If loaded from checkpoint, _val_fixed_indices is already set -- don't re-lock.
        if self._val_fixed_indices is None and getattr(self, '_val_seen_indices', []):
            self._lock_val_fixed_indices()

        if self._val_locked_refs is None and self._val_fixed_indices is not None:
            self._lock_val_refs()

        self._save_val_audio()

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        checkpoint["val_locked_refs"]   = self._val_locked_refs
        checkpoint["val_fixed_indices"] = self._val_fixed_indices

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        self._val_locked_refs   = checkpoint.get("val_locked_refs", None)
        self._val_fixed_indices = checkpoint.get("val_fixed_indices", None)

        # Strip state_dict keys from older checkpoints that no longer exist in model.
        sd = checkpoint.get("state_dict", {})
        unexpected = [k for k in list(sd.keys()) if
                      any(k.startswith(p) for p in [
                          "loss_func.g.hann_", "loss_func.g.weights_",
                          "loss_func.d.hann_", "loss_func.d.weights_",
                      ])]
        for k in unexpected:
            del sd[k]
        checkpoint["state_dict"] = sd

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

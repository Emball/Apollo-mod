# @claude last-modified: 2026-05-05T06:34:39Z
# @claude last-commit: feat: major update — TUI, augmentation system, gradient checkpointing, optimization bootstrap
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
        self.val_audio_dir = val_audio_dir
        self.val_audio_pairs = val_audio_pairs
        self._val_audio_indices = None  # always re-locks on first val run, even after resume
        self._val_batch_index   = {}    # cleared after locking
        self.gradient_checkpointing = gradient_checkpointing
        self.grad_accum_steps = max(1, grad_accum_steps)
        self._accum_loss_g = None
        self._accum_loss_d = None
        self._accum_step   = 0

        if gradient_checkpointing:
            self._enable_gradient_checkpointing()

        self.default_monitor = "val_loss"
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.automatic_optimization = False
        self._amp_ctx = torch.amp.autocast("cuda", dtype=torch.float16)

    def _enable_gradient_checkpointing(self):
        """
        Wrap BSNet layers and FrequencyDiscriminator sub-networks with
        torch.utils.checkpoint so intermediate activations are recomputed
        during backward instead of stored — trades ~30% compute for large
        VRAM savings.
        """
        import look2hear.models.apollo as _apollo_mod
        import look2hear.discriminators.frequencydis as _dis_mod

        original_bsnet_forward = _apollo_mod.BSNet.forward

        def checkpointed_bsnet_forward(self_bsnet, input):
            if not torch.is_grad_enabled():
                return original_bsnet_forward(self_bsnet, input)
            return torch_checkpoint.checkpoint(
                original_bsnet_forward, self_bsnet, input, use_reentrant=False
            )

        _apollo_mod.BSNet.forward = checkpointed_bsnet_forward

        original_freqdis_forward = _dis_mod.FrequencyDiscriminator.forward

        def checkpointed_freqdis_forward(self_dis, x):
            if not torch.is_grad_enabled():
                return original_freqdis_forward(self_dis, x)

            def _inner(x_):
                out, _ = original_freqdis_forward(self_dis, x_)
                return out

            out = torch_checkpoint.checkpoint(_inner, x, use_reentrant=False)
            with torch.no_grad():
                _, hiddens = original_freqdis_forward(self_dis, x.detach())
            return out, hiddens

        _dis_mod.FrequencyDiscriminator.forward = checkpointed_freqdis_forward
        print("[gradient_checkpointing] Enabled for BSNet layers and FrequencyDiscriminator.")

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
        amp_ctx = self._amp_ctx

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

        # Detach targets_feature_maps — fixed reference for feature matching;
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

        for p in self.discriminator.parameters():
            p.requires_grad_(True)

        # Optimizer step — only after accumulating grad_accum_steps batches
        if is_last_accum:
            self.clip_gradients(optimizer_d, gradient_clip_val=5, gradient_clip_algorithm="norm")
            optimizer_d.step()

            self.clip_gradients(optimizer_g, gradient_clip_val=5, gradient_clip_algorithm="norm")
            optimizer_g.step()

            self.log("train_loss_d", self._accum_loss_d, on_epoch=False, on_step=True, prog_bar=True, logger=True)
            self.log("train_loss_g", self._accum_loss_g, on_epoch=False, on_step=True, prog_bar=True, logger=True)
            self._accum_loss_g = None
            self._accum_loss_d = None

        self._accum_step += 1

        if self.trainer.is_last_batch:
            scheduler_g.step()
            scheduler_d.step()

    def validation_step(self, batch, batch_nb):
        ori_data, codec_data = batch
        est_sources = self(codec_data)
        loss = self.metrics(est_sources, ori_data)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)

        # Save restored audio every val_save_interval epochs so the rendered
        # audio and the loss number are always from the same computation.
        # Only saves for the fixed subset of pairs chosen at first val run.

        # Build a batch_nb -> song_key map on the first real val epoch so
        # on_validation_epoch_end can select diverse indices.
        # Skip during the sanity check — it only sees 2 batches and would
        # lock in a useless index before real validation runs.
        if self._val_audio_indices is None and not self.trainer.sanity_checking:
            if not hasattr(self, '_val_batch_index'):
                self._val_batch_index = {}
            # Derive song key from batch_nb via the val dataset's index table
            try:
                val_dataset = self.trainer.datamodule.data_val
                batch_size = self.trainer.datamodule.batch_size
                sample_idx = min(batch_nb * batch_size, len(val_dataset.index) - 1)
                pair_idx, _ = val_dataset.index[sample_idx]
                lq_path = val_dataset.pairs[pair_idx][0]
                stem = os.path.splitext(os.path.basename(lq_path))[0]
                # Strip trailing _chunk#### suffix so all chunks of a song group together
                import re as _re
                song_key = _re.sub(r'_chunk\d+$', '', stem)
            except Exception:
                song_key = str(batch_nb)
            self._val_batch_index[batch_nb] = song_key

        if (
            self.val_audio_dir is not None
            and self._val_audio_indices is not None
        ):
            save_this = batch_nb in self._val_audio_indices

            if save_this:
                epoch_dir = os.path.join(
                    self.val_audio_dir, f"step_{self.global_step:06d}"
                )
                os.makedirs(epoch_dir, exist_ok=True)
                sr = 44100

                # Derive a human-readable name from the val dataset
                try:
                    val_dataset = self.trainer.datamodule.data_val
                    batch_size = self.trainer.datamodule.batch_size
                    sample_idx = min(batch_nb * batch_size, len(val_dataset.index) - 1)
                    pair_idx, start_sample = val_dataset.index[sample_idx]
                    lq_path = val_dataset.pairs[pair_idx][0]
                    song_name = os.path.splitext(os.path.basename(lq_path))[0]
                    fname_base = f"{song_name}_s{start_sample:06d}"
                except Exception:
                    fname_base = f"pair_{batch_nb:04d}"

                out = est_sources
                if out.ndim == 3:
                    out = out[0]
                elif out.ndim == 1:
                    out = out.unsqueeze(0)
                out = out.float().cpu().clamp(-1.0, 1.0)
                torchaudio.save(os.path.join(epoch_dir, f"{fname_base}_restored.wav"), out, sr)

                # Save LQ and HQ alongside for easy A/B comparison
                try:
                    lq_t = codec_data[0]
                    hq_t = ori_data[0]
                    if lq_t.ndim == 1: lq_t = lq_t.unsqueeze(0)
                    if hq_t.ndim == 1: hq_t = hq_t.unsqueeze(0)
                    torchaudio.save(os.path.join(epoch_dir, f"{fname_base}_lq.wav"), lq_t.float().cpu().clamp(-1.0, 1.0), sr)
                    torchaudio.save(os.path.join(epoch_dir, f"{fname_base}_hq.wav"), hq_t.float().cpu().clamp(-1.0, 1.0), sr)
                except Exception:
                    pass

        return {"val_loss": loss}

    def on_validation_epoch_end(self):
        self.log("lr", self.optimizer[0].param_groups[0]["lr"], on_epoch=True, prog_bar=True)
        self.validation_step_outputs.clear()

        # After the first val epoch we know the full dataset size — randomly
        # pick val_audio_pairs indices with at least 1 from each song, then
        # lock them in for all future epochs.
        if self.trainer.sanity_checking:
            self._val_batch_index = {}
            return

        if self._val_audio_indices is None and not self.trainer.sanity_checking:
            import random, re as _re
            from collections import defaultdict

            # Build song->batch_nb map directly from the full val dataset,
            # not from the limited batches that actually ran (limit_val_batches
            # may have cut off most songs).
            try:
                val_dataset = self.trainer.datamodule.data_val
                batch_size  = self.trainer.datamodule.batch_size
                song_batches = defaultdict(list)
                for sample_idx, (pair_idx, _) in enumerate(val_dataset.index):
                    lq_path  = val_dataset.pairs[pair_idx][0]
                    stem     = os.path.splitext(os.path.basename(lq_path))[0]
                    song_key = _re.sub(r'_chunk\d+$', '', stem)
                    batch_nb = sample_idx // batch_size
                    song_batches[song_key].append(batch_nb)
            except Exception:
                # Fallback: use whatever we observed during val steps
                song_batches = defaultdict(list)
                for batch_nb, song_key in self._val_batch_index.items():
                    song_batches[song_key].append(batch_nb)

            songs = list(song_batches.keys())
            total = sum(len(v) for v in song_batches.values())

            if self.val_audio_pairs == 0:
                self._val_audio_indices = set(range(total))
                print(f"[val audio] Saving all {len(self._val_audio_indices)} val pairs across {len(songs)} songs")
            else:
                # val_audio_pairs = pairs PER SONG, not total
                per_song = max(1, self.val_audio_pairs)
                selected = []
                for s in songs:
                    choices = song_batches[s]
                    selected += random.sample(choices, min(per_song, len(choices)))
                self._val_audio_indices = set(selected)
                print(f"[val audio] Locked {len(self._val_audio_indices)} indices ({per_song} per song across {len(songs)} songs): {sorted(self._val_audio_indices)}")
            self._val_batch_index = {}  # free memory

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

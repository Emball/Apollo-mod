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
        self.val_save_interval = val_save_interval
        self.val_audio_dir = val_audio_dir
        self.val_audio_pairs = val_audio_pairs
        self._val_audio_indices = None
        self.gradient_checkpointing = gradient_checkpointing
        self.grad_accum_steps = max(1, grad_accum_steps)
        self._accum_loss_g = None
        self._accum_loss_d = None
        self._accum_step   = 0
        self._val_loss_sum   = 0.0
        self._val_loss_count = 0

        if gradient_checkpointing:
            self._enable_gradient_checkpointing()

        self.default_monitor = "val_loss"
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.automatic_optimization = False

    def _enable_gradient_checkpointing(self):
        """
        Wrap BSNet layers with torch.utils.checkpoint so intermediate activations
        are recomputed during backward instead of stored — trades ~30% compute for
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

            # Log scalar values only — .item() detaches from GPU, avoiding the
            # on_epoch=True accumulation bug (PL caches GPU tensors until epoch end).
            self.log("train_loss_d", float(self._accum_loss_d), on_step=True, prog_bar=True, logger=True)
            self.log("train_loss_g", float(self._accum_loss_g), on_step=True, prog_bar=True, logger=True)
            self._accum_loss_g = None
            self._accum_loss_d = None

        self._accum_step += 1

    def on_train_epoch_end(self):
        scheduler_g, scheduler_d = self.lr_schedulers()
        scheduler_g.step()
        scheduler_d.step()

    def validation_step(self, batch, batch_nb):
        ori_data, codec_data, ds_idx, song_key = batch
        # ds_idx and song_key are batched tensors/lists — unwrap the scalar
        ds_idx   = int(ds_idx[0])
        song_key = song_key[0] if isinstance(song_key, (list, tuple)) else song_key

        est_sources = self(codec_data)
        loss = self.metrics(est_sources, ori_data)

        self._val_loss_sum   += float(loss)
        self._val_loss_count += 1
        self.validation_step_outputs.append(float(loss))

        if self.trainer.sanity_checking:
            return {"val_loss": loss}

        # On the first real val run, record dataset_idx -> song_key for all batches
        # seen. With shuffle=True + limit_val_batches this covers a random cross-
        # section of songs each run, building up coverage across runs until locked.
        if self._val_audio_indices is None:
            if not hasattr(self, '_val_batch_index'):
                self._val_batch_index = {}
            self._val_batch_index[ds_idx] = song_key

        # Save audio if this dataset index is in the locked set.
        # On the first run (indices not yet locked) save everything seen —
        # on_validation_epoch_end will prune to val_audio_pairs diverse indices.
        if self.val_audio_dir is not None:
            save_this = (self._val_audio_indices is None) or (ds_idx in self._val_audio_indices)

            if save_this:
                epoch_dir = os.path.join(
                    self.val_audio_dir, f"step_{self.global_step:06d}"
                )
                os.makedirs(epoch_dir, exist_ok=True)
                out = est_sources
                if out.ndim == 3:
                    out = out[0]
                elif out.ndim == 1:
                    out = out.unsqueeze(0)
                out = out.float().cpu().clamp(-1.0, 1.0)
                torchaudio.save(
                    os.path.join(epoch_dir, f"{song_key}_{ds_idx:04d}.wav"),
                    out, 44100,
                )

        return {"val_loss": loss}

    def on_validation_epoch_end(self):
        if self._val_loss_count > 0:
            avg_val_loss = self._val_loss_sum / self._val_loss_count
            self.log("val_loss", avg_val_loss, prog_bar=True, logger=True)
        self._val_loss_sum   = 0.0
        self._val_loss_count = 0
        self.log("lr", self.optimizer[0].param_groups[0]["lr"], prog_bar=True)
        self.validation_step_outputs.clear()

        # After the first real val run, lock a diverse set of dataset indices —
        # one guaranteed per song, remainder filled randomly. These are stable
        # dataset positions so they survive shuffle and limit_val_batches.
        if self._val_audio_indices is None and hasattr(self, '_val_batch_index') and not self.trainer.sanity_checking:
            import random
            from collections import defaultdict

            song_batches = defaultdict(list)
            for ds_idx, song_key in self._val_batch_index.items():
                song_batches[song_key].append(ds_idx)

            songs = list(song_batches.keys())
            n = min(self.val_audio_pairs, sum(len(v) for v in song_batches.values()))

            selected = [random.choice(song_batches[s]) for s in songs]
            already  = set(selected)
            remainder = [i for i in self._val_batch_index if i not in already]
            still_needed = max(0, n - len(selected))
            selected += random.sample(remainder, min(still_needed, len(remainder)))

            self._val_audio_indices = set(selected)
            print(f"[val audio] Locked {len(self._val_audio_indices)} dataset indices across {len(songs)} songs: {sorted(self._val_audio_indices)}")

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        # Persist val audio selection state so resume uses the same locked indices.
        checkpoint["val_audio_indices"] = self._val_audio_indices
        checkpoint["val_batch_index"]   = getattr(self, "_val_batch_index", None)

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        # Restore val audio selection state from checkpoint.
        self._val_audio_indices = checkpoint.get("val_audio_indices", None)
        saved_batch_index = checkpoint.get("val_batch_index", None)
        if saved_batch_index is not None:
            self._val_batch_index = saved_batch_index

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

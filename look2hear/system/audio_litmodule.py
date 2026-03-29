###
# Modified from original Apollo audio_litmodule.py
# Changes:
#   - Removed sync_dist=True from all log calls (causes hangs on single GPU)
#   - Removed all_gather in validation (multi-GPU only)
#   - Removed WandB-specific logger calls
#   - Kept everything else identical
###
import os
import torchaudio
from omegaconf import OmegaConf
import torch
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
    ):
        super().__init__()
        self.audio_model = model
        self.discriminator = discriminator
        self.optimizer = list(optimizer)
        self.loss_func = loss_func
        self.metrics = metrics
        self.scheduler = list(scheduler)

        self.default_monitor = "val_loss"
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.automatic_optimization = False
        self.sample_output_dir = None  # set by train.py after instantiation

    def forward(self, wav):
        return self.audio_model(wav)

    def training_step(self, batch, batch_nb):
        ori_data, codec_data = batch
        optimizer_g, optimizer_d = self.optimizers()
        scheduler_g, scheduler_d = self.lr_schedulers()

        # Generator forward pass (shared between both discriminator and generator updates)
        optimizer_g.zero_grad()
        output = self(codec_data)

        # ── Discriminator update ──────────────────────────────────────────────
        # Run discriminator on both generated and real audio.
        # Cache target_outputs and targets_feature_maps here — we reuse them
        # in the generator update below to avoid a redundant forward pass.
        optimizer_d.zero_grad()
        for p in self.discriminator.parameters():
            p.requires_grad_(True)

        est_outputs_d, _ = self.discriminator(output.detach(), sample_rate=44100)
        target_outputs, targets_feature_maps = self.discriminator(ori_data, sample_rate=44100)

        loss_d = self.loss_func["d"](target_outputs, est_outputs_d)
        self.manual_backward(loss_d)
        self.clip_gradients(optimizer_d, gradient_clip_val=5, gradient_clip_algorithm="norm")
        optimizer_d.step()

        # ── Generator update ──────────────────────────────────────────────────
        # Freeze discriminator weights — we only need gradients w.r.t. generator.
        # Reuse cached target_outputs/targets_feature_maps from above (saves one
        # full discriminator forward pass per training step).
        for p in self.discriminator.parameters():
            p.requires_grad_(False)

        est_outputs, est_feature_maps = self.discriminator(output, sample_rate=44100)

        loss_g = self.loss_func["g"](est_outputs, est_feature_maps, targets_feature_maps, output, ori_data)
        self.manual_backward(loss_g)
        self.clip_gradients(optimizer_g, gradient_clip_val=5, gradient_clip_algorithm="norm")
        optimizer_g.step()

        # Unfreeze discriminator for next step
        for p in self.discriminator.parameters():
            p.requires_grad_(True)

        if self.trainer.is_last_batch:
            scheduler_g.step()
            scheduler_d.step()

        self.log("train_loss_d", loss_d, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss_g", loss_g, on_epoch=True, prog_bar=True, logger=True)

    def validation_step(self, batch, batch_nb):
        ori_data, codec_data = batch
        est_sources = self(codec_data)
        loss = self.metrics(est_sources, ori_data)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.validation_step_outputs.append(loss)

        # Save restored audio samples at each validation run (keyed by global step)
        if self.sample_output_dir is not None:
            step = self.global_step
            sample_dir = os.path.join(self.sample_output_dir, f"step_{step:06d}")
            os.makedirs(sample_dir, exist_ok=True)
            restored = est_sources[0].float().cpu()
            lq       = codec_data[0].float().cpu()
            hq       = ori_data[0].float().cpu()
            sr = 44100
            torchaudio.save(os.path.join(sample_dir, f"sample_{batch_nb:03d}_restored.wav"), restored, sr)
            torchaudio.save(os.path.join(sample_dir, f"sample_{batch_nb:03d}_lq.wav"),       lq,       sr)
            torchaudio.save(os.path.join(sample_dir, f"sample_{batch_nb:03d}_hq.wav"),       hq,       sr)

        return {"val_loss": loss}

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.validation_step_outputs).mean()
        self.log("lr", self.optimizer[0].param_groups[0]["lr"], on_epoch=True, prog_bar=True)
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_nb):
        mixtures, targets = batch
        est_sources = self(mixtures)
        loss = self.metrics(est_sources, targets)
        self.log("test_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.test_step_outputs.append(loss)
        return {"test_loss": loss}

    def on_test_epoch_end(self):
        avg_loss = torch.stack(self.test_step_outputs).mean()
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

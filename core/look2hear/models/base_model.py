# @claude last-modified: 2026-05-05T06:34:39Z
# @claude last-commit: feat: major update -- TUI, augmentation system, gradient checkpointing, optimization bootstrap
###
# Author: Kai Li
# Date: 2021-06-17 23:08:32
# LastEditors: Please set LastEditors
# LastEditTime: 2022-05-26 18:06:22
###
import torch
import torch.nn as nn

from huggingface_hub import PyTorchModelHubMixin


class BaseModel(nn.Module, PyTorchModelHubMixin):
    def __init__(self, sample_rate, in_chan=1):
        super().__init__()
        self._sample_rate = sample_rate
        self._in_chan = in_chan

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def sample_rate(self,):
        return self._sample_rate

    @staticmethod
    def from_pretrain(pretrained_model_conf_or_path, *args, **kwargs):
        from . import get
        from omegaconf import DictConfig, ListConfig

        if hasattr(torch.serialization, "add_safe_globals"):
            from omegaconf.base import ContainerMetadata, Metadata
            from typing import Any
            torch.serialization.add_safe_globals(
                [DictConfig, ListConfig, ContainerMetadata, Metadata, Any, dict, list]
            )
            conf = torch.load(
                pretrained_model_conf_or_path, map_location="cpu", weights_only=True
            )
        else:
            # torch <2.4 — add_safe_globals unavailable; checkpoint is from trusted HuggingFace source
            conf = torch.load(
                pretrained_model_conf_or_path, map_location="cpu", weights_only=False
            )

        model_class = get(conf["model_name"])
        # model_class = get("Conv_TasNet")
        model = model_class(*args, **kwargs)
        model_state = model.state_dict()
        filtered = {
            k: v for k, v in conf["state_dict"].items()
            if k in model_state and v.shape == model_state[k].shape
        }
        skipped = len(conf["state_dict"]) - len(filtered)
        if skipped:
            print(f"[weights] Skipped {skipped} mismatched keys (shape mismatch — expected when feature_dim differs from pretrained checkpoint)")
        model.load_state_dict(filtered, strict=False)
        return model

    def serialize(self):
        import pytorch_lightning as pl  # Not used in torch.hub

        model_conf = dict(
            model_name=self.__class__.__name__,
            state_dict=self.get_state_dict(),
            model_args=self.get_model_args(),
        )
        # Additional infos
        infos = dict()
        infos["software_versions"] = dict(
            torch_version=torch.__version__, pytorch_lightning_version=pl.__version__,
        )
        model_conf["infos"] = infos
        return model_conf

    def get_state_dict(self):
        """In case the state dict needs to be modified before sharing the model."""
        return self.state_dict()

    def get_model_args(self):
        """Should return args to re-instantiate the class."""
        raise NotImplementedError

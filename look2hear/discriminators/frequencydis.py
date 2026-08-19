# @claude last-modified: 2026-05-05T06:34:39Z
# @claude last-commit: feat: major update — TUI, augmentation system, gradient checkpointing, optimization bootstrap
###
# Modified from original Apollo frequencydis.py
# Changes:
#   - MultiFrequencyDiscriminator now applies per-window weights
#     when aggregating discriminator outputs, giving higher weight
#     to small windows (32, 64) which capture high frequency detail
#   - Weights are inversely proportional to window size so smaller
#     windows (higher freq resolution) contribute more to the loss
###

import torch
import torch.nn as nn
import numpy as np

class MultiFrequencyDiscriminator(nn.Module):
    def __init__(self, nch, window):
        super(MultiFrequencyDiscriminator, self).__init__()

        self.nch = nch
        self.window = window
        self.hidden_channels = 8
        self.eps = torch.finfo(torch.float32).eps
        self.discriminators = nn.ModuleList([
            FrequencyDiscriminator(2 * nch, self.hidden_channels)
            for _ in range(len(self.window))
        ])

        # Inverse window size weighting — smaller windows get higher weight
        # since they resolve high frequency content better.
        # e.g. window=32 gets weight proportional to 1/32, window=2048 to 1/2048
        raw_weights = torch.tensor([1.0 / w for w in window], dtype=torch.float32)
        # Normalize so weights sum to len(window) — keeps overall loss scale
        # roughly the same as the original equal-weight version
        self.register_buffer(
            "window_weights",
            raw_weights / raw_weights.mean()
        )

        # Pre-register hann windows as buffers so they're on the right device
        # automatically and never allocated inside the forward pass.
        for w in window:
            self.register_buffer(f"hann_{w}", torch.hann_window(w).float())

    def forward(self, est, sample_rate=44100):
        B, nch, _ = est.shape
        assert nch == self.nch

        # Normalize power
        est = est / (est.pow(2).sum((1, 2)) + self.eps).sqrt().reshape(B, 1, 1)
        est = est.view(-1, est.shape[-1])

        est_outputs = []
        est_feature_maps = []

        for i in range(len(self.discriminators)):
            est_spec = torch.stft(
                est.float(),
                self.window[i],
                self.window[i] // 2,
                window=getattr(self, f"hann_{self.window[i]}"),
                return_complex=True
            )
            est_RI = torch.stack([est_spec.real, est_spec.imag], dim=1)
            est_RI = est_RI.view(
                B, nch * 2, est_RI.shape[-2], est_RI.shape[-1]
            ).type(est.type())

            valid_enc = int(est_RI.shape[2] * sample_rate / 44100)
            est_out, est_feat_map = self.discriminators[i](
                est_RI[:, :, :valid_enc].contiguous()
            )

            # Scale output by window weight — applied to every forward pass (real
            # AND fake) so the weight is consistent across both sides of the D loss.
            weight = self.window_weights[i]
            est_outputs.append(est_out * weight)
            est_feature_maps.append(est_feat_map)

        return est_outputs, est_feature_maps

class FrequencyDiscriminator(nn.Module):
    def __init__(self, in_channels, hidden_channels=512):
        super(FrequencyDiscriminator, self).__init__()

        self.eps = torch.finfo(torch.float32).eps
        self.discriminator = nn.ModuleList()
        self.discriminator += [
            nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(in_channels, hidden_channels, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))),
                nn.LeakyReLU(0.2, True)
            ),
            nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2))),
                nn.LeakyReLU(0.2, True)
            ),
            nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(hidden_channels * 2, hidden_channels * 4, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))),
                nn.LeakyReLU(0.2, True)
            ),
            nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(hidden_channels * 4, hidden_channels * 8, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2))),
                nn.LeakyReLU(0.2, True)
            ),
            nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(hidden_channels * 8, hidden_channels * 16, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))),
                nn.LeakyReLU(0.2, True)
            ),
            nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(hidden_channels * 16, hidden_channels * 32, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2))),
                nn.LeakyReLU(0.2, True)
            ),
            nn.Conv2d(hidden_channels * 32, 1, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        ]

    def forward(self, x):
        hiddens = []
        for layer in self.discriminator:
            x = layer(x)
            hiddens.append(x)
        return x, hiddens[:-1]

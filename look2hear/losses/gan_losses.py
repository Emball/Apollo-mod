# @claude last-modified: 2026-05-05T06:34:39Z
# @claude last-commit: feat: major update — TUI, augmentation system, gradient checkpointing, optimization bootstrap
###
# Modified from original Apollo gan_losses.py
# Changes:
#   - freq_MAE now applies perceptual weighting that penalizes
#     high frequency errors more heavily, since the high end has
#     less energy and contributes less to unweighted MAE
#   - hf_boost is now a constructor arg (configurable via yaml)
#     1.0 = flat (original behavior), 2.0 = double penalty on top band
#   - hf_threshold_ratio is also configurable
###

import torch
from torch.nn.modules.loss import _Loss


class MultiFrequencyDisLoss(_Loss):
    def __init__(self, eps=1e-8):
        super(MultiFrequencyDisLoss, self).__init__()

    def forward(self, target_outputs, est_outputs):
        D_real = 0
        D_fake = 0
        for i in range(len(target_outputs)):
            D_real = D_real + (target_outputs[i] - 1).pow(2).mean() / len(target_outputs)
            D_fake = D_fake + (est_outputs[i]).pow(2).mean() / len(est_outputs)
        return D_real + D_fake


class MultiFrequencyGenLoss(_Loss):
    def __init__(self, eps=1e-8, hf_boost=1.0, hf_threshold_ratio=0.5):
        super(MultiFrequencyGenLoss, self).__init__()
        self.eps = eps
        self.all_win = [32, 64, 128, 256, 512, 1024, 2048]

        # Pre-build hann windows and HF weight tensors as buffers so they live
        # on the right device automatically and are never reallocated per step.
        for win in self.all_win:
            self.register_buffer(f"hann_{win}", torch.hann_window(win))
            n_bins = win // 2 + 1
            hf_cutoff = int(n_bins * hf_threshold_ratio)
            w = torch.ones(1, n_bins, 1)
            w[0, hf_cutoff:, 0] = hf_boost
            self.register_buffer(f"weights_{win}", w)

    def _freq_MAE(self, output, target):
        loss = 0.
        eps = torch.finfo(torch.float32).eps
        device = output.device
        flat_out = output.view(-1, output.shape[-1])
        flat_tgt = target.view(-1, target.shape[-1])

        for win in self.all_win:
            hann    = getattr(self, f"hann_{win}").to(device)
            weights = getattr(self, f"weights_{win}").to(device)

            est_spec    = torch.stft(flat_out, n_fft=win, hop_length=win // 2,
                                     window=hann, return_complex=True)
            target_spec = torch.stft(flat_tgt, n_fft=win, hop_length=win // 2,
                                     window=hann, return_complex=True)

            mag_err      = (est_spec.abs() - target_spec.abs()).abs()
            weighted_err = (mag_err * weights).mean() / (target_spec.abs().mean() + eps)
            loss         = loss + weighted_err

        return loss / len(self.all_win)

    def forward(self, est_outputs, est_feature_maps, targets_feature_maps, output, ori_data):
        G_fake = 0
        feature_matching = 0
        eps = self.eps

        for i in range(len(est_outputs)):
            G_fake = G_fake + (est_outputs[i] - 1).pow(2).mean() / len(est_outputs)
            for j in range(len(est_feature_maps[i])):
                feature_matching = feature_matching + (
                    est_feature_maps[i][j] - targets_feature_maps[i][j].detach()
                ).abs().mean() / (targets_feature_maps[i][j].detach().abs().mean() + eps)

        feature_matching = feature_matching / (len(est_outputs) * len(est_feature_maps[0]))
        freq_loss  = self._freq_MAE(output, ori_data.unsqueeze(1))
        total_loss = freq_loss + G_fake + feature_matching

        return total_loss

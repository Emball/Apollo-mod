# @claude last-modified: 2026-08-21T00:00:00Z
# @claude last-commit: 0.3.0.0 -- gaussian band weight, hf_band_mae metric, val overhaul, timer fix, step fix
###
# Modified from original Apollo gan_losses.py
# Changes:
#   - freq_MAE now applies a gaussian perceptual weight curve centered on the
#     MP3 transition zone (default 15kHz), replacing the flat step-function
#     hf_boost. Much more surgical -- penalizes the actual problem band without
#     over-boosting already-fine frequencies below it or the supersonic range above.
#   - band_weight_center_hz / band_weight_sigma_hz / band_weight_gain are the
#     three knobs replacing hf_boost + hf_threshold_ratio
#   - hf_band_mae() exposed as a standalone function for use in val metrics
###

import torch
import math
from torch.nn.modules.loss import _Loss


def _gaussian_weight(n_bins: int, center_bin: int, sigma_bins: float,
                     gain: float, device: torch.device) -> torch.Tensor:
    """
    1D gaussian weight curve over frequency bins.
    Flat at 1.0 everywhere, with a raised bump of `gain` centered at center_bin
    with std dev sigma_bins. Values are clamped >= 1.0 so the gaussian only
    adds penalty, never reduces it.
    """
    bins = torch.arange(n_bins, dtype=torch.float32, device=device)
    gaussian = gain * torch.exp(-0.5 * ((bins - center_bin) / sigma_bins) ** 2)
    return (1.0 + gaussian).view(1, n_bins, 1)


def hf_band_mae(est: torch.Tensor, ref: torch.Tensor,
                sr: int = 44100,
                lo_hz: float = 13000.0,
                hi_hz: float = 19000.0) -> float:
    """
    Mean absolute log-magnitude error in the transition band (default 13-19 kHz).
    Lower = better restoration of the MP3 rolloff zone.
    Exposed for use in val metrics logging.
    """
    n_fft = 2048
    hop   = 512
    win   = torch.hann_window(n_fft, device=est.device)
    hz_per_bin = sr / n_fft
    bin_lo = int(lo_hz / hz_per_bin)
    bin_hi = min(int(hi_hz / hz_per_bin), n_fft // 2)
    eps = 1e-7

    def _log_mag(x):
        return torch.stft(
            x.reshape(-1, x.shape[-1]),
            n_fft=n_fft, hop_length=hop, win_length=n_fft,
            window=win, return_complex=True
        ).abs().clamp(min=eps).log()

    e_log = _log_mag(est)[:, bin_lo:bin_hi, :]
    r_log = _log_mag(ref)[:, bin_lo:bin_hi, :]
    return (e_log - r_log).abs().mean().item()


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
    def __init__(self,
                 eps=1e-8,
                 # Gaussian band weight parameters
                 band_weight_center_hz=15000.0,  # center of the penalty bump (Hz)
                 band_weight_sigma_hz=3000.0,    # width of the bump (Hz, 1-sigma)
                 band_weight_gain=1.5,           # peak gain above baseline (0 = flat)
                 sr=44100,
                 # Legacy flat-boost params kept for config back-compat but ignored
                 hf_boost=1.0,
                 hf_threshold_ratio=0.5,
                 ):
        super(MultiFrequencyGenLoss, self).__init__()
        self.eps = eps
        self.all_win = [32, 64, 128, 256, 512, 1024, 2048]
        self._sr = sr
        self._center_hz = band_weight_center_hz
        self._sigma_hz  = band_weight_sigma_hz
        self._gain      = band_weight_gain

        # Hann windows cached as plain dicts -- NOT register_buffer so they never
        # pollute state_dict or checkpoints. Fully deterministic; no learned state.
        # Moved to device lazily on first _freq_MAE call.
        self._hann_cache:   dict = {}
        self._weight_cache: dict = {}
        for win in self.all_win:
            self._hann_cache[win] = torch.hann_window(win)
            # weights built lazily on first device move (need device for gaussian)

    def _get_weights(self, win: int, device: torch.device) -> torch.Tensor:
        key = (win, str(device))
        if key not in self._weight_cache:
            n_bins      = win // 2 + 1
            hz_per_bin  = self._sr / win
            center_bin  = self._center_hz / hz_per_bin
            sigma_bins  = self._sigma_hz  / hz_per_bin
            self._weight_cache[key] = _gaussian_weight(
                n_bins, center_bin, sigma_bins, self._gain, device
            )
        return self._weight_cache[key]

    def _freq_MAE(self, output, target):
        loss   = 0.
        eps    = torch.finfo(torch.float32).eps
        device = output.device
        flat_out = output.view(-1, output.shape[-1])
        flat_tgt = target.view(-1, target.shape[-1])

        for win in self.all_win:
            hann = self._hann_cache[win]
            if hann.device != device:
                hann = hann.to(device)
                self._hann_cache[win] = hann

            weights = self._get_weights(win, device)

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

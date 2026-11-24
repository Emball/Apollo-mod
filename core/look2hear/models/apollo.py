# @claude last-modified: 2026-05-05T06:34:39Z
# @claude last-commit: feat: major update -- TUI, augmentation system, gradient checkpointing, optimization bootstrap
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .base_model import BaseModel

class RMSNorm(nn.Module):
    def __init__(self, dimension, groups=1):
        super().__init__()
        
        self.weight = nn.Parameter(torch.ones(dimension))
        self.groups = groups
        self.eps = 1e-5

    def forward(self, input):
        # input size: (B, N, T)
        B, N, T = input.shape
        assert N % self.groups == 0

        input_float = input.reshape(B, self.groups, -1, T).float()
        input_norm = input_float * torch.rsqrt(input_float.pow(2).mean(-2, keepdim=True) + self.eps)

        return input_norm.type_as(input).reshape(B, N, T) * self.weight.reshape(1, -1, 1)
    
class Roformer(nn.Module):
    """
    Transformer with rotary positional embedding.
    """
    def __init__(self, input_size, hidden_size, num_head=8, theta=10000, window=10000, 
                 input_drop=0., attention_drop=0., causal=True):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size // num_head
        self.num_head = num_head
        self.theta = theta  # base frequency for RoPE
        self.window = window
        # pre-calculate rotary embeddings
        cos_freq, sin_freq = self._calc_rotary_emb()
        self.register_buffer("cos_freq", cos_freq)  # win, N
        self.register_buffer("sin_freq", sin_freq)  # win, N
        self.register_buffer("reverse_sign", torch.tensor([-1.0, 1.0]))

        self.attention_drop = attention_drop
        self.causal = causal
        self.eps = 1e-5

        self.input_norm = RMSNorm(self.input_size)
        self.input_drop = nn.Dropout(p=input_drop)
        self.weight = nn.Conv1d(self.input_size, self.hidden_size*self.num_head*3, 1, bias=False)
        self.output = nn.Conv1d(self.hidden_size*self.num_head, self.input_size, 1, bias=False)

        self.MLP = nn.Sequential(RMSNorm(self.input_size),
                                 nn.Conv1d(self.input_size, self.input_size*8, 1, bias=False),
                                 nn.SiLU()
                                )
        self.MLP_output = nn.Conv1d(self.input_size*4, self.input_size, 1, bias=False)

    def _calc_rotary_emb(self):
        freq = 1. / (self.theta ** (torch.arange(0, self.hidden_size, 2)[:(self.hidden_size // 2)] / self.hidden_size))  # theta_i
        freq = freq.reshape(1, -1)  # 1, N//2
        pos = torch.arange(0, self.window).reshape(-1, 1)  # win, 1
        cos_freq = torch.cos(pos*freq)  # win, N//2
        sin_freq = torch.sin(pos*freq)  # win, N//2
        cos_freq = torch.stack([cos_freq]*2, -1).reshape(self.window, self.hidden_size)  # win, N
        sin_freq = torch.stack([sin_freq]*2, -1).reshape(self.window, self.hidden_size)  # win, N

        return cos_freq, sin_freq
    
    def _add_rotary_sequence(self, feature):
        # feature shape: ..., T, N
        T, N = feature.shape[-2:]
        feature_reshape = feature.reshape(-1, T, N)

        cos_freq = self.cos_freq[:T]
        sin_freq = self.sin_freq[:T]
        reverse_sign = self.reverse_sign.to(feature.dtype)
        feature_reshape_neg = (torch.flip(feature_reshape.reshape(-1, N//2, 2), [-1]) * reverse_sign.reshape(1, 1, 2)).reshape(-1, T, N)
        feature_rope = feature_reshape * cos_freq.unsqueeze(0) + feature_reshape_neg * sin_freq.unsqueeze(0)
    
        return feature_rope.reshape(feature.shape)
    
    def forward(self, input):
        # input shape: B, N, T

        B, _, T = input.shape

        weight = self.weight(self.input_drop(self.input_norm(input))).reshape(B, self.num_head, self.hidden_size*3, T).mT
        Q, K, V = torch.split(weight, self.hidden_size, dim=-1)  # B, num_head, T, N
        
        # rotary positional embedding
        Q_rot = self._add_rotary_sequence(Q)
        K_rot = self._add_rotary_sequence(K)

        attention_output = F.scaled_dot_product_attention(Q_rot.contiguous(), K_rot.contiguous(), V.contiguous(), dropout_p=self.attention_drop, is_causal=self.causal)  # B, num_head, T, N
        attention_output = attention_output.mT.reshape(B, -1, T)
        output = self.output(attention_output) + input

        gate, z = self.MLP(output).chunk(2, dim=1)
        output = output + self.MLP_output(F.silu(gate) * z)

        return output
    
class ConvActNorm1d(nn.Module):
    def __init__(self, in_channel, hidden_channel, kernel=7, causal=False):
        super(ConvActNorm1d, self).__init__()
        
        self.in_channel = in_channel
        self.kernel = kernel
        self.causal = causal
        if not causal:
            self.conv = nn.Sequential(nn.Conv1d(in_channel, in_channel, kernel, padding=(kernel-1)//2, groups=in_channel),
                                      RMSNorm(in_channel),
                                      nn.Conv1d(in_channel, hidden_channel, 1),
                                      nn.SiLU(),
                                      nn.Conv1d(hidden_channel, in_channel, 1)
                                     )
        else:
            self.conv = nn.Sequential(nn.Conv1d(in_channel, in_channel, kernel, padding=kernel-1, groups=in_channel),
                                      RMSNorm(in_channel),
                                      nn.Conv1d(in_channel, hidden_channel, 1),
                                      nn.SiLU(),
                                      nn.Conv1d(hidden_channel, in_channel, 1)
                                     )
        
    def forward(self, input):
        
        output = self.conv(input)
        if self.causal:
            output = output[...,:-self.kernel+1]
        return input + output

class ICB(nn.Module):
    def __init__(self, in_channel, kernel=7, causal=False):
        super(ICB, self).__init__()
        
        self.blocks = nn.Sequential(ConvActNorm1d(in_channel, in_channel*4, kernel, causal=causal),
                                    ConvActNorm1d(in_channel, in_channel*4, kernel, causal=causal),
                                    ConvActNorm1d(in_channel, in_channel*4, kernel, causal=causal)
                                    )
        
    def forward(self, input):
        
        return self.blocks(input)

class BSNet(nn.Module):
    def __init__(self, feature_dim, kernel=7):
        super(BSNet, self).__init__()

        self.feature_dim = feature_dim

        self.band_net = Roformer(self.feature_dim, self.feature_dim, num_head=8, window=100, causal=False)
        self.seq_net = ICB(self.feature_dim, kernel=kernel)

    def forward(self, input):
        # input shape: B, nband, N, T

        B, nband, N, T = input.shape

        # band comm
        band_input = input.permute(0,3,2,1).reshape(B*T, -1, nband)
        band_output = self.band_net(band_input)
        band_output = band_output.reshape(B, T, -1, nband).permute(0,3,2,1)

        # sequence modeling
        output = self.seq_net(band_output.reshape(B*nband, -1, T)).reshape(B, nband, -1, T)  # B, nband, N, T

        return output
    
class Apollo(BaseModel):
    def __init__(
        self, 
        sr: int,
        win: int,
        feature_dim: int,
        layer: int
    ):
        super().__init__(sample_rate=sr)
        
        self.sr = sr
        self.win = int(sr * win // 1000)
        self.stride = self.win // 2
        self.enc_dim = self.win // 2 + 1
        self.feature_dim = feature_dim
        self.eps = torch.finfo(torch.float32).eps

        # Pre-register hann window as a buffer so it's not recreated every forward pass
        self.register_buffer("hann_win", torch.hann_window(int(sr * win // 1000)))

        # 80 bands
        bandwidth = int(self.win / 160)
        self.band_width = [bandwidth]*79
        self.band_width.append(self.enc_dim - np.sum(self.band_width))
        self.nband = len(self.band_width)
        self._n_uniform = self.nband if self.band_width[-1] == self.band_width[0] else self.nband - 1

        self.BN = nn.ModuleList([])
        for i in range(self.nband):
            self.BN.append(nn.Sequential(RMSNorm(self.band_width[i]*2+1),
                                         nn.Conv1d(self.band_width[i]*2+1, self.feature_dim, 1))
                          )

        self.net = []
        for _ in range(layer):
            self.net.append(BSNet(self.feature_dim))
        self.net = nn.Sequential(*self.net)
        
        self.output = nn.ModuleList([])
        for i in range(self.nband):
            self.output.append(nn.Sequential(RMSNorm(self.feature_dim),
                                                 nn.Conv1d(self.feature_dim, self.band_width[i]*4, 1),
                                                 nn.GLU(dim=1)
                                                )
                                  )

    def _rms(self, x, weight, dim):
        xf = x.float()
        xn = xf * torch.rsqrt(xf.pow(2).mean(dim, keepdim=True) + 1e-5)
        return xn.type_as(x) * weight

    def _stack(self, modules, getter):
        return torch.stack([getter(m) for m in modules], 0)

    def _band_proj(self, x, norm_w, conv_w, conv_b):
        # x: B, K, C, T with one weight set per band K
        x = self._rms(x, norm_w.unsqueeze(0).unsqueeze(-1), 2)
        return torch.einsum("bkct,kfc->bkft", x, conv_w) + conv_b.unsqueeze(0).unsqueeze(-1)

    def feature_extractor(self, input):
        B, nch, nsample = input.shape

        # cuFFT doesn't support half precision for non-power-of-two sizes,
        # so the STFT runs in float32.
        spec = torch.stft(input.float().view(B*nch, nsample), n_fft=self.win, hop_length=self.stride,
                          window=self.hann_win, return_complex=True)
        T = spec.shape[-1]
        nu, bw = self._n_uniform, self.band_width[0]

        parts = []
        if nu > 0:
            u = spec[:, :nu*bw].reshape(B*nch, nu, bw, T)
            power = (u.abs().pow(2).sum(2) + self.eps).sqrt()  # B, nu, T
            un = u / power.unsqueeze(2)
            feat = torch.cat([un.real, un.imag, torch.log(power).unsqueeze(2)], 2)  # B, nu, 2*bw+1, T
            bn = [self.BN[i] for i in range(nu)]
            parts.append(self._band_proj(
                feat,
                self._stack(bn, lambda m: m[0].weight),
                self._stack(bn, lambda m: m[1].weight.squeeze(-1)),
                self._stack(bn, lambda m: m[1].bias)))
        if nu < self.nband:
            t = spec[:, nu*bw:]
            power = (t.abs().pow(2).sum(1, keepdim=True) + self.eps).sqrt()  # B, 1, T
            tn = t / power
            feat = torch.cat([tn.real, tn.imag, torch.log(power)], 1).unsqueeze(1)  # B, 1, 2*bw_t+1, T
            bn = self.BN[nu]
            parts.append(self._band_proj(
                feat, bn[0].weight.unsqueeze(0), bn[1].weight.squeeze(-1).unsqueeze(0), bn[1].bias.unsqueeze(0)))
        return torch.cat(parts, 1)  # B, nband, N, T

    def _band_head(self, feat, norm_w, conv_w, conv_b, bw):
        # feat: B, K, N, T -> real, imag each B, K, bw, T
        x = self._rms(feat, norm_w.unsqueeze(0).unsqueeze(-1), 2)
        x = torch.einsum("bknt,kon->bkot", x, conv_w) + conv_b.unsqueeze(0).unsqueeze(-1)
        a, g = x.chunk(2, dim=2)
        x = a * torch.sigmoid(g)
        x = x.reshape(x.shape[0], x.shape[1], 2, bw, x.shape[-1])
        return x[:, :, 0], x[:, :, 1]

    def forward(self, input):

        B, nch, nsample = input.shape

        subband_feature = self.feature_extractor(input)
        feature = self.net(subband_feature)

        nu, bw = self._n_uniform, self.band_width[0]
        real, imag = [], []
        if nu > 0:
            out = [self.output[i] for i in range(nu)]
            r, im = self._band_head(
                feature[:, :nu],
                self._stack(out, lambda m: m[0].weight),
                self._stack(out, lambda m: m[1].weight.squeeze(-1)),
                self._stack(out, lambda m: m[1].bias), bw)
            real.append(r.reshape(r.shape[0], nu*bw, -1))
            imag.append(im.reshape(im.shape[0], nu*bw, -1))
        if nu < self.nband:
            head = self.output[nu]
            r, im = self._band_head(
                feature[:, nu:], head[0].weight.unsqueeze(0), head[1].weight.squeeze(-1).unsqueeze(0),
                head[1].bias.unsqueeze(0), self.band_width[nu])
            real.append(r[:, 0])
            imag.append(im[:, 0])
        est_spec = torch.complex(torch.cat(real, 1).float(), torch.cat(imag, 1).float())
        output = torch.istft(est_spec, n_fft=self.win, hop_length=self.stride,
                             window=self.hann_win, length=nsample).view(B, nch, -1).to(input.dtype)

        return output

    def get_model_args(self):
        model_args = {"n_sample_rate": 2}
        return model_args
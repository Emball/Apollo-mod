# @claude last-modified: 2026-05-05T06:34:39Z
# @claude last-commit: feat: major update -- TUI, augmentation system, gradient checkpointing, optimization bootstrap
"""
Paired audio datamodule for Apollo fine-tuning.

Directory layout
----------------
chunks/
    train/
        LQ/  track1_0000.wav  track1_0001.wav  ...
        HQ/  track1_0000.wav  track1_0001.wav  ...
    val/
        LQ/  held_out_0000.wav  ...
        HQ/  held_out_0000.wav  ...

Chunking is handled automatically by train.py's prepare_data() at startup.
Raw source material goes in data/train/ and data/val/ -- see train.py for
accepted input layouts (_LQ/_HQ subdirs, flat postfix files, or pre-normalized
LQ/HQ subdirs).

Val loss is computed from chunks/val/ every validation epoch.
Restored audio is saved to runs/<n>/val_audio/epoch_NNNN/ every
val_save_interval epochs for a fixed subset of val_audio_pairs chunks,
controlled via the training: block in the config yaml.

Augmentation
------------
All augmentation behaviour is controlled by the `augmentation` block in
your config yaml. See configs/README.txt for full documentation.
"""

import io
import os
import random
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule

# Augmentation config dataclasses

@dataclass
class GainAugCfg:
    enabled: bool = True
    prob: float   = 0.5
    db_max: float = 1.5

@dataclass
class SimpleAugCfg:
    enabled: bool = True
    prob: float   = 0.5

@dataclass
class Mp3AugCfg:
    enabled:  bool  = False
    prob:     float = 0.5
    kbps_min: int   = 64
    kbps_max: int   = 256
    target:   str   = "lq"   # "lq" = LQ only (default), "both" = LQ and HQ at same bitrate

@dataclass
class DeepGainAugCfg:
    enabled: bool  = True
    prob:    float = 0.03   # rare -- ~1 in 33 chunks
    db_min:  float = -10.0  # floor
    db_max:  float = -6.0   # ceiling (always a reduction, never a boost)

@dataclass
class SilenceDipAugCfg:
    enabled:             bool  = True
    prob:                float = 0.05    # ~1 in 20 chunks
    max_hold_sec:        float = 2.0     # max silence hold duration
    # Ramp duration distribution: short=70%, medium=20%, long=10%
    short_ramp_ms:       float = 10.0
    short_ramp_max_ms:   float = 50.0
    medium_ramp_ms:      float = 50.0
    medium_ramp_max_ms:  float = 200.0
    long_ramp_ms:        float = 200.0
    long_ramp_max_ms:    float = 1000.0

@dataclass
class MidSideAugCfg:
    enabled:    bool  = False
    prob_mid:   float = 0.1   # prob of replacing pair with mid (L+R)/2 summed to both channels
    prob_side:  float = 0.1   # prob of replacing pair with side (L-R)/2 summed to both channels

@dataclass
class AugmentationCfg:
    enabled:            bool             = True
    gain:               GainAugCfg       = field(default_factory=GainAugCfg)
    deep_gain:          DeepGainAugCfg   = field(default_factory=DeepGainAugCfg)
    polarity:           SimpleAugCfg     = field(default_factory=SimpleAugCfg)
    silence_dip:        SilenceDipAugCfg = field(default_factory=SilenceDipAugCfg)
    mp3_degradation:    Mp3AugCfg        = field(default_factory=Mp3AugCfg)
    stereo_alternation: SimpleAugCfg     = field(default_factory=SimpleAugCfg)
    mid_side_isolation: MidSideAugCfg   = field(default_factory=MidSideAugCfg)

def _get(d, key, default):
    try:
        return d[key]
    except (KeyError, TypeError):
        return default

def _parse_aug_cfg(raw) -> AugmentationCfg:
    """Build AugmentationCfg from an OmegaConf DictConfig, plain dict, or None.
    If the config has a 'live' sub-key (new-style split config), reads from that.
    Falls back to reading the block directly for backwards compatibility."""
    if raw is None:
        return AugmentationCfg()
    # New-style: augmentation.live -- extract the live sub-block
    live = _get(raw, "live", None)
    if live is not None:
        raw = live

    gain_raw = _get(raw, "gain", {})
    dg_raw   = _get(raw, "deep_gain", {})
    pol_raw  = _get(raw, "polarity", {})
    sil_raw  = _get(raw, "silence_dip", {})
    mp3_raw  = _get(raw, "mp3_degradation", {})
    mono_raw = _get(raw, "stereo_alternation", {})
    ms_raw   = _get(raw, "mid_side_isolation", {})

    return AugmentationCfg(
        enabled=_get(raw, "enabled", True),
        gain=GainAugCfg(
            enabled=_get(gain_raw, "enabled", True),
            prob=   _get(gain_raw, "prob",    0.5),
            db_max= _get(gain_raw, "db_max",  1.5),
        ),
        deep_gain=DeepGainAugCfg(
            enabled=_get(dg_raw, "enabled", True),
            prob=   _get(dg_raw, "prob",    0.03),
            db_min= _get(dg_raw, "db_min",  -10.0),
            db_max= _get(dg_raw, "db_max",  -6.0),
        ),
        polarity=SimpleAugCfg(
            enabled=_get(pol_raw, "enabled", True),
            prob=   _get(pol_raw, "prob",    0.5),
        ),
        silence_dip=SilenceDipAugCfg(
            enabled=            _get(sil_raw, "enabled",           True),
            prob=               _get(sil_raw, "prob",              0.05),
            max_hold_sec=       _get(sil_raw, "max_hold_sec",      2.0),
            short_ramp_ms=      _get(sil_raw, "short_ramp_ms",     10.0),
            short_ramp_max_ms=  _get(sil_raw, "short_ramp_max_ms", 50.0),
            medium_ramp_ms=     _get(sil_raw, "medium_ramp_ms",    50.0),
            medium_ramp_max_ms= _get(sil_raw, "medium_ramp_max_ms",200.0),
            long_ramp_ms=       _get(sil_raw, "long_ramp_ms",      200.0),
            long_ramp_max_ms=   _get(sil_raw, "long_ramp_max_ms",  1000.0),
        ),
        mp3_degradation=Mp3AugCfg(
            enabled= _get(mp3_raw, "enabled",  False),
            prob=    _get(mp3_raw, "prob",     0.5),
            kbps_min=_get(mp3_raw, "kbps_min", 64),
            kbps_max=_get(mp3_raw, "kbps_max", 256),
            target=  _get(mp3_raw, "target",   "lq"),
        ),
        stereo_alternation=SimpleAugCfg(
            enabled=_get(mono_raw, "enabled", True),
            prob=   _get(mono_raw, "prob",    1.0),
        ),
        mid_side_isolation=MidSideAugCfg(
            enabled=  _get(ms_raw, "enabled",  False),
            prob_mid= _get(ms_raw, "prob_mid", 0.1),
            prob_side=_get(ms_raw, "prob_side",0.1),
        ),
    )

# Individual augmentation implementations

_ffmpeg_available: Optional[bool] = None

def _check_ffmpeg() -> bool:
    global _ffmpeg_available
    if _ffmpeg_available is None:
        try:
            import ffmpeg
            _ffmpeg_available = True
        except ImportError:
            print("[augmentation] WARNING: ffmpeg-python not installed -- mp3_degradation disabled.")
            print("               Install with: pip install ffmpeg-python")
            _ffmpeg_available = False
    return _ffmpeg_available

def _pitch_shift_tensor(wav: torch.Tensor, semitones: float, sr: int) -> torch.Tensor:
    """Pitch shift via resampling. No external deps, exact same shape guaranteed."""
    original_length = wav.shape[-1]
    shift_factor = 2 ** (semitones / 12)
    virtual_sr = int(round(sr * shift_factor))
    wav = torchaudio.functional.resample(wav, sr, virtual_sr)
    wav = torchaudio.functional.resample(wav, virtual_sr, sr)
    if wav.shape[-1] >= original_length:
        wav = wav[:, :original_length]
    else:
        wav = torch.nn.functional.pad(wav, (0, original_length - wav.shape[-1]))
    return wav.float()

def _mp3_degrade_tensor(wav: torch.Tensor, kbps: int, sr: int) -> torch.Tensor:
    """Encode wav to MP3 at kbps then decode back, with encoder delay compensation.

    MP3 encoding introduces a fixed encoder delay at the start of the decoded audio
    (typically 576 or 1152 samples with LAME). We detect this by prepending a known
    impulse, encoding, decoding, then finding where the impulse lands to measure the
    exact delay introduced at this bitrate. The delay is then stripped from the front
    of the decoded audio so it stays perfectly aligned with HQ.
    """
    import ffmpeg
    import numpy as np

    original_length = wav.shape[-1]
    n_channels = wav.shape[0]

    # --- Measure encoder delay using an impulse probe ---
    # We prepend a short impulse and detect its position after encode/decode.
    # This accounts for any LAME delay regardless of bitrate.
    probe_len = 2048
    impulse = torch.zeros(n_channels, probe_len)
    impulse[:, 0] = 1.0  # single-sample impulse at position 0
    probed = torch.cat([impulse, wav.float()], dim=-1)

    def _encode_decode(tensor):
        pcm_bytes = tensor.numpy().T.tobytes()
        mp3_bytes, _ = (
            ffmpeg
            .input("pipe:", format="f32le", ar=sr, ac=n_channels)
            .output("pipe:", format="mp3", audio_bitrate=f"{kbps}k", codec="libmp3lame")
            .run(input=pcm_bytes, capture_stdout=True, capture_stderr=True, quiet=True)
        )
        pcm_out, _ = (
            ffmpeg
            .input("pipe:", format="mp3")
            .output("pipe:", format="f32le", ar=sr, ac=n_channels)
            .run(input=mp3_bytes, capture_stdout=True, capture_stderr=True, quiet=True)
        )
        samples = np.frombuffer(pcm_out, dtype=np.float32).reshape(-1, n_channels).T
        return torch.from_numpy(samples.copy())

    decoded_probed = _encode_decode(probed)

    # Find the impulse peak in the decoded output to measure actual delay
    probe_region = decoded_probed[0, :probe_len * 2].abs()
    delay = int(probe_region.argmax().item())

    # Now encode/decode just the original audio and strip the measured delay
    decoded = _encode_decode(wav.float())
    decoded = decoded[:, delay:]

    # Trim or pad to exact original length
    if decoded.shape[-1] >= original_length:
        decoded = decoded[:, :original_length]
    else:
        decoded = torch.nn.functional.pad(decoded, (0, original_length - decoded.shape[-1]))

    return decoded.float()
def _silence_dip_envelope(n_samples: int, sr: int, cfg: "SilenceDipAugCfg") -> "torch.Tensor":
    """Build a [1, n_samples] amplitude envelope that dips to zero somewhere inside
    the chunk. Shape: ramp down -> hold at zero -> ramp back up. The dip is placed
    at a random interior position so neither boundary is ever silenced (keeps chunk
    edges clean for the dataloader). Applied identically to LQ and HQ."""
    # Pick ramp type by weighted draw: 70% short, 20% medium, 10% long
    r = random.random()
    if r < 0.70:
        ramp_ms = random.uniform(cfg.short_ramp_ms,  cfg.short_ramp_max_ms)
    elif r < 0.90:
        ramp_ms = random.uniform(cfg.medium_ramp_ms, cfg.medium_ramp_max_ms)
    else:
        ramp_ms = random.uniform(cfg.long_ramp_ms,   cfg.long_ramp_max_ms)

    ramp_samples = max(1, int(ramp_ms / 1000.0 * sr))
    hold_samples = random.randint(1, max(1, int(cfg.max_hold_sec * sr)))
    dip_len      = 2 * ramp_samples + hold_samples

    # Keep the whole dip interior -- at least 1 sample of non-silence at each boundary
    max_start = n_samples - dip_len - 2
    if max_start <= 1:
        # Chunk too short for this dip -- return flat envelope (no-op)
        return torch.ones(1, n_samples)

    start = random.randint(1, max_start)

    env = torch.ones(n_samples)
    # Ramp down: 1 -> 0
    env[start : start + ramp_samples] = torch.linspace(1.0, 0.0, ramp_samples)
    # Hold at zero
    env[start + ramp_samples : start + ramp_samples + hold_samples] = 0.0
    # Ramp up: 0 -> 1
    env[start + ramp_samples + hold_samples : start + dip_len] = torch.linspace(0.0, 1.0, ramp_samples)

    return env.unsqueeze(0)  # (1, n_samples) for broadcasting over channels


def augment_pair(
    lq: torch.Tensor,
    hq: torch.Tensor,
    cfg: AugmentationCfg,
    sr: int = 44100,
    idx: Optional[int] = None,
    in_second_half: bool = False,
    forced_kbps: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply augmentations to an LQ/HQ pair. Shape: (2, samples).
    Inputs are expected to be in [-1, 1] (post normalize_pair).

    stereo_alternation, polarity: applied deterministically based on which
        half of the song this chunk falls in. First half = clean (no aug).
        Second half = augmented. This guarantees that every repeated section
        in a song (chorus, loop, etc.) appears once clean and once augmented,
        making redundant musical content into genuinely distinct training pairs.

    gain: still applied per-chunk with a random draw, since amplitude
        variation within a song is realistic and benefits from fine-grained
        coverage rather than coarse half-level assignment.

    mp3_degradation: applied per-chunk when enabled (unchanged).
    """
    if not cfg.enabled:
        return lq, hq

    # Mid/side isolation: rolls first. If it fires, stereo_alternation is skipped
    # so the model sees the full stereo difference/sum signal rather than a single channel.
    # Teaches the model to work on mid and side signals independently, which is where
    # MP3 joint-stereo does the most damage at low bitrates.
    _mid_side_fired = False
    if cfg.mid_side_isolation.enabled and lq.shape[0] == 2:
        r = random.random()
        if r < cfg.mid_side_isolation.prob_mid:
            mid_lq = (lq[0:1] + lq[1:2]) * 0.5
            mid_hq = (hq[0:1] + hq[1:2]) * 0.5
            lq = mid_lq.expand(2, -1).clone()
            hq = mid_hq.expand(2, -1).clone()
            _mid_side_fired = True
        elif r < cfg.mid_side_isolation.prob_mid + cfg.mid_side_isolation.prob_side:
            side_lq = (lq[0:1] - lq[1:2]) * 0.5
            side_hq = (hq[0:1] - hq[1:2]) * 0.5
            lq = side_lq.expand(2, -1).clone()
            hq = side_hq.expand(2, -1).clone()
            _mid_side_fired = True

    # stereo_alternation: deterministic by song half. Skipped if mid/side fired.
    # First half gets L channel, second half gets R channel, giving the model
    # both stereo perspectives of every repeated musical idea.
    if cfg.stereo_alternation.enabled and not _mid_side_fired:
        ch = 1 if in_second_half else 0
        lq = lq[ch:ch+1]
        hq = hq[ch:ch+1]

    # Gain: per-chunk random draw (realistic intra-song amplitude variance).
    if cfg.gain.enabled and random.random() < cfg.gain.prob:
        db    = random.uniform(-cfg.gain.db_max, cfg.gain.db_max)
        scale = 10 ** (db / 20.0)
        lq    = lq * scale
        hq    = hq * scale
        peak  = max(lq.abs().max(), hq.abs().max())
        if peak > 1.0:
            lq = lq / peak
            hq = hq / peak

    # Polarity inversion: deterministic by song half.
    # Second half only, so every repeated section appears once normal, once inverted.
    if cfg.polarity.enabled and in_second_half:
        lq = -lq
        hq = -hq

    # Deep gain reduction: rare, -6 to -10dB, trains the model that low-ceiling
    # masters and near-silence are valid targets, not noise to fill in.
    # Applied on top of the fine gain above -- independent draw.
    if cfg.deep_gain.enabled and random.random() < cfg.deep_gain.prob:
        db    = random.uniform(cfg.deep_gain.db_min, cfg.deep_gain.db_max)
        scale = 10 ** (db / 20.0)
        lq    = lq * scale
        hq    = hq * scale

    # Silence dip: zero out a random interior region of the chunk via a
    # fade-down / hold / fade-up envelope. Applied identically to LQ and HQ
    # so the pair stays aligned. Teaches the model that silence = silence,
    # preventing hallucinated texture on quiet passages and fades.
    if cfg.silence_dip.enabled and random.random() < cfg.silence_dip.prob:
        env = _silence_dip_envelope(lq.shape[-1], sr, cfg.silence_dip)
        lq  = lq * env
        hq  = hq * env

    # MP3 degradation (unchanged -- per-chunk when enabled).
    if cfg.mp3_degradation.enabled and random.random() < cfg.mp3_degradation.prob:
        if _check_ffmpeg():
            kbps = forced_kbps if forced_kbps is not None else random.randint(
                cfg.mp3_degradation.kbps_min, cfg.mp3_degradation.kbps_max
            )
            lq = _mp3_degrade_tensor(lq, kbps, sr)
            if cfg.mp3_degradation.target == "both":
                hq = _mp3_degrade_tensor(hq, kbps, sr)

    return lq, hq

# Shared helpers

SR = 44100

def load_wav(path: str, target_sr: int = SR) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    elif wav.shape[0] > 2:
        wav = wav[:2]
    return wav

def normalize_pair(lq: torch.Tensor, hq: torch.Tensor):
    scale = max(lq.abs().max(), hq.abs().max())
    if scale > 0:
        lq = lq / scale
        hq = hq / scale
    return lq, hq

def get_matched_pairs(lq_dir: str, hq_dir: str) -> List[Tuple[str, str]]:
    lq_files = {
        os.path.splitext(f)[0]: os.path.join(lq_dir, f)
        for f in os.listdir(lq_dir) if f.endswith(".wav")
    }
    hq_files = {
        os.path.splitext(f)[0]: os.path.join(hq_dir, f)
        for f in os.listdir(hq_dir) if f.endswith(".wav")
    }
    matched = sorted(set(lq_files.keys()) & set(hq_files.keys()))

    unmatched_lq = set(lq_files.keys()) - set(hq_files.keys())
    unmatched_hq = set(hq_files.keys()) - set(lq_files.keys())
    if unmatched_lq:
        print(f"WARNING: LQ files with no HQ match (skipping): {sorted(unmatched_lq)}")
    if unmatched_hq:
        print(f"WARNING: HQ files with no LQ match (skipping): {sorted(unmatched_hq)}")
    if not matched:
        raise RuntimeError(f"No matched pairs found in {lq_dir} and {hq_dir}")

    return [(lq_files[s], hq_files[s]) for s in matched]

# Training dataset -- loads pre-chunked files

class ChunkedPairDataset(Dataset):
    def __init__(self, chunks_dir: str, sr: int = SR, aug_cfg: AugmentationCfg = None, label: str = "Training"):
        self._is_val = (label == "Validation")
        lq_dir = os.path.join(chunks_dir, "LQ")
        hq_dir = os.path.join(chunks_dir, "HQ")
        self.pairs   = get_matched_pairs(lq_dir, hq_dir)
        self.sr      = sr
        self.aug_cfg = aug_cfg or AugmentationCfg()

        # Build a per-chunk second-half flag keyed by (lq_path, hq_path).
        # Chunks are named <stem>_NNNN.wav; we group by stem and mark the
        # upper half of each song's chunks as in_second_half=True. This
        # guarantees every repeated musical section appears once clean and
        # once augmented without relying on coin flips.
        from collections import defaultdict
        import re
        stem_to_indices: dict = defaultdict(list)
        for i, (lq_path, _) in enumerate(self.pairs):
            fname = os.path.splitext(os.path.basename(lq_path))[0]
            # Strip trailing _NNNN chunk index to recover the song stem
            stem = re.sub(r'_\d{4}$', '', fname)
            stem_to_indices[stem].append(i)

        self._in_second_half: list[bool] = [False] * len(self.pairs)
        for stem, indices in stem_to_indices.items():
            indices_sorted = sorted(indices)
            midpoint = len(indices_sorted) // 2
            for i, global_idx in enumerate(indices_sorted):
                self._in_second_half[global_idx] = (i >= midpoint)

        aug = self.aug_cfg
        if label == "Validation":
            print(f"Validation dataset: {len(self.pairs)} 30s chunk pairs")
            return
        print(f"Training dataset : {len(self.pairs)} chunk pairs")
        print(
            f"Augmentation     : enabled={aug.enabled}  "
            f"stereo_alternation={aug.stereo_alternation.enabled}(half-split)  "
            f"polarity={aug.polarity.enabled}(half-split)  "
            f"gain={aug.gain.enabled}(per-chunk, p={aug.gain.prob}, +/-{aug.gain.db_max}dB)  "
            f"deep_gain={aug.deep_gain.enabled}(p={aug.deep_gain.prob}, "
            f"{aug.deep_gain.db_min}..{aug.deep_gain.db_max}dB)  "
            f"silence_dip={aug.silence_dip.enabled}(p={aug.silence_dip.prob}, "
            f"hold<={aug.silence_dip.max_hold_sec}s)  "
            f"mp3={aug.mp3_degradation.enabled}(p={aug.mp3_degradation.prob}, "
            f"{aug.mp3_degradation.kbps_min}-{aug.mp3_degradation.kbps_max}kbps)  "
            f"mid_side_isolation={aug.mid_side_isolation.enabled}"
            f"(p_mid={aug.mid_side_isolation.prob_mid}, p_side={aug.mid_side_isolation.prob_side})"
        )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        lq_path, hq_path = self.pairs[idx]
        lq = load_wav(lq_path, self.sr)
        hq = load_wav(hq_path, self.sr)

        if self._is_val:
            # Val mode: normalize and return idx + song_key so validation_step can do index locking
            lq, hq = normalize_pair(lq, hq)
            song_key = os.path.splitext(os.path.basename(lq_path))[0]
            return hq, lq, idx, song_key

        # No per-chunk normalize_pair here -- chunks are already normalized at
        # the full-song level during _slice_and_save. Normalizing again per-chunk
        # would re-introduce inconsistent gain riding at chunk boundaries.
        lq, hq = augment_pair(
            lq, hq, self.aug_cfg,
            sr=self.sr,
            idx=idx,
            in_second_half=self._in_second_half[idx],
        )
        return hq, lq

# Validation dataset -- full-length files sliced at runtime

class FullLengthPairDataset(Dataset):
    def __init__(self, eval_dir: str, sr: int = SR, segment_sec: float = 2.0):
        lq_dir = os.path.join(eval_dir, "LQ")
        hq_dir = os.path.join(eval_dir, "HQ")
        self.pairs           = get_matched_pairs(lq_dir, hq_dir)
        self.sr              = sr
        self.segment_samples = int(segment_sec * sr)

        self.index = []
        for pair_idx, (lq_path, hq_path) in enumerate(self.pairs):
            lq_info = torchaudio.info(lq_path)
            hq_info = torchaudio.info(hq_path)
            min_len = min(lq_info.num_frames, hq_info.num_frames)
            start = 0
            while start + self.segment_samples <= min_len:
                self.index.append((pair_idx, start))
                start += self.segment_samples

        print(f"Validation dataset: {len(self.pairs)} files -> {len(self.index)} segments")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, str]:
        pair_idx, start = self.index[idx]
        lq_path, hq_path = self.pairs[pair_idx]
        lq, _ = torchaudio.load(lq_path, frame_offset=start, num_frames=self.segment_samples)
        hq, _ = torchaudio.load(hq_path, frame_offset=start, num_frames=self.segment_samples)
        if lq.shape[0] == 1: lq = lq.repeat(2, 1)
        if hq.shape[0] == 1: hq = hq.repeat(2, 1)
        lq_chunk, hq_chunk = normalize_pair(lq, hq)
        song_key = os.path.splitext(os.path.basename(lq_path))[0]
        return hq_chunk, lq_chunk, idx, song_key

# DataModule

class PairedAudioDataModule(LightningDataModule):
    def __init__(
        self,
        train_dir: str,
        eval_dir: str,
        sr: int = SR,
        segment_sec: float = 2.0,
        batch_size: int = 1,
        num_workers: int = 4,
        pin_memory: bool = True,
        augmentation: Optional[dict] = None,
        val_bootstrap_chunks: int = 50,
        **kwargs,  # absorb train-only keys (e.g. align_data) passed via Hydra
    ):
        super().__init__()
        self.train_dir            = train_dir
        self.eval_dir             = eval_dir
        self.sr                   = sr
        self.segment_sec          = segment_sec
        self.batch_size           = batch_size
        self.num_workers          = num_workers
        self.pin_memory           = pin_memory
        self.aug_cfg              = _parse_aug_cfg(augmentation)
        self.val_bootstrap_chunks = val_bootstrap_chunks

        self.data_train: Optional[Dataset] = None
        self.data_val:   Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        if self.data_train is None:
            self.data_train = ChunkedPairDataset(
                chunks_dir=self.train_dir,
                sr=self.sr,
                aug_cfg=self.aug_cfg,
            )
        if self.data_val is None:
            self.data_val = ChunkedPairDataset(
                chunks_dir=self.eval_dir,
                sr=self.sr,
                aug_cfg=None,
                label="Validation",
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
        )
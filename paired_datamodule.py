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
class PitchShiftAugCfg:
    enabled:      bool  = True
    prob:         float = 0.5
    semitones_max: float = 1.5

@dataclass
class NoiseAugCfg:
    enabled: bool  = True
    prob:    float = 0.5
    sigma:   float = 0.002

@dataclass
class Mp3AugCfg:
    enabled:  bool  = False
    prob:     float = 0.5
    kbps_min: int   = 64
    kbps_max: int   = 256
    target:   str   = "lq"   # "lq" = LQ only (default), "both" = LQ and HQ at same bitrate

@dataclass
class AugmentationCfg:
    enabled:      bool            = True
    gain:         GainAugCfg      = field(default_factory=GainAugCfg)
    polarity:     SimpleAugCfg    = field(default_factory=SimpleAugCfg)
    pitch_shift:  PitchShiftAugCfg = field(default_factory=PitchShiftAugCfg)
    noise:        NoiseAugCfg     = field(default_factory=NoiseAugCfg)
    mp3_degradation: Mp3AugCfg   = field(default_factory=Mp3AugCfg)
    stereo_alternation: SimpleAugCfg   = field(default_factory=SimpleAugCfg)

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

    gain_raw    = _get(raw, "gain", {})
    pol_raw     = _get(raw, "polarity", {})
    pitch_raw   = _get(raw, "pitch_shift", {})
    noise_raw   = _get(raw, "noise", {})
    mp3_raw     = _get(raw, "mp3_degradation", {})
    mono_raw    = _get(raw, "stereo_alternation", {})

    return AugmentationCfg(
        enabled=_get(raw, "enabled", True),
        gain=GainAugCfg(
            enabled=_get(gain_raw, "enabled", True),
            prob=   _get(gain_raw, "prob",    0.5),
            db_max= _get(gain_raw, "db_max",  1.5),
        ),
        polarity=SimpleAugCfg(
            enabled=_get(pol_raw, "enabled", True),
            prob=   _get(pol_raw, "prob",    0.5),
        ),
        pitch_shift=PitchShiftAugCfg(
            enabled=       _get(pitch_raw, "enabled",       True),
            prob=          _get(pitch_raw, "prob",          0.5),
            semitones_max= _get(pitch_raw, "semitones_max", 1.5),
        ),
        noise=NoiseAugCfg(
            enabled=_get(noise_raw, "enabled", True),
            prob=   _get(noise_raw, "prob",    0.5),
            sigma=  _get(noise_raw, "sigma",   0.002),
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
def augment_pair(
    lq: torch.Tensor,
    hq: torch.Tensor,
    cfg: AugmentationCfg,
    sr: int = 44100,
    idx: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply random augmentations to an LQ/HQ pair. Shape: (2, samples).
    Inputs are expected to be in [-1, 1] (post normalize_pair).

    stereo_alternation: picks one channel (L or R) randomly and returns a
        (1, samples) tensor for both LQ and HQ. Apollo processes each
        channel independently anyway, so this avoids feeding near-duplicate
        stereo channels as if they were unique data.

    pitch_shift: shifts both LQ and HQ by the same random amount via
        torch-pitch-shift. Output length is guaranteed identical to input.

    noise: adds matched Gaussian noise to both LQ and HQ. Since both
        receive the same noise, the pair relationship is preserved and
        the model learns noise robustness without becoming a denoiser.

    mp3_degradation: applies a random CBR MP3 encode/decode pass to LQ
        only, simulating an additional lossy encoding stage on top of
        existing codec degradation. HQ is untouched. Output length is
        trimmed/padded to original to handle encoder delay.

    gain, polarity: applied identically to both LQ and HQ.
    """
    if not cfg.enabled:
        return lq, hq

    # stereo_alternation
    # Alternate L/R deterministically by sample index so every epoch covers
    # both channels evenly rather than randomly clumping on one side.
    # Applied first so all subsequent augmentations work on the selected channel.
    if cfg.stereo_alternation.enabled and random.random() < cfg.stereo_alternation.prob:
        ch = (idx % 2) if idx is not None else random.randint(0, lq.shape[0] - 1)
        lq = lq[ch:ch+1]
        hq = hq[ch:ch+1]

    # Pitch shift
    if cfg.pitch_shift.enabled and random.random() < cfg.pitch_shift.prob:
        semitones = random.uniform(-cfg.pitch_shift.semitones_max, cfg.pitch_shift.semitones_max)
        lq = _pitch_shift_tensor(lq, semitones, sr)
        hq = _pitch_shift_tensor(hq, semitones, sr)

    # Gain
    if cfg.gain.enabled and random.random() < cfg.gain.prob:
        db    = random.uniform(-cfg.gain.db_max, cfg.gain.db_max)
        scale = 10 ** (db / 20.0)
        lq    = lq * scale
        hq    = hq * scale
        # If the gain push would clip, rescale both down just enough to avoid it
        # rather than hard-clamping -- preserves waveform shape including any
        # pre-existing clipping distortion in the source material
        peak = max(lq.abs().max(), hq.abs().max())
        if peak > 1.0:
            lq = lq / peak
            hq = hq / peak

    # Polarity inversion
    if cfg.polarity.enabled and random.random() < cfg.polarity.prob:
        lq = -lq
        hq = -hq

    # Gaussian noise
    if cfg.noise.enabled and random.random() < cfg.noise.prob:
        noise = torch.randn_like(lq) * cfg.noise.sigma
        lq = lq + noise
        hq = hq + noise
        peak = max(lq.abs().max(), hq.abs().max())
        if peak > 1.0:
            lq = lq / peak
            hq = hq / peak

    # MP3 degradation. target="lq" applies to LQ only (default, standard augmentation).
    # target="both" applies the same bitrate to LQ and HQ so codec artifacts cancel out,
    # forcing the model to focus only on non-codec differences between the streams.
    if cfg.mp3_degradation.enabled and random.random() < cfg.mp3_degradation.prob:
        if _check_ffmpeg():
            kbps = random.randint(cfg.mp3_degradation.kbps_min, cfg.mp3_degradation.kbps_max)
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
    def __init__(self, chunks_dir: str, sr: int = SR, aug_cfg: AugmentationCfg = None):
        lq_dir = os.path.join(chunks_dir, "LQ")
        hq_dir = os.path.join(chunks_dir, "HQ")
        self.pairs   = get_matched_pairs(lq_dir, hq_dir)
        self.sr      = sr
        self.aug_cfg = aug_cfg or AugmentationCfg()
        aug = self.aug_cfg
        print(f"Training dataset : {len(self.pairs)} chunk pairs")
        print(
            f"Augmentation     : enabled={aug.enabled}  "
            f"stereo_alternation={aug.stereo_alternation.enabled}(p={aug.stereo_alternation.prob})  "
            f"gain={aug.gain.enabled}(p={aug.gain.prob}, +/-{aug.gain.db_max}dB)  "
            f"polarity={aug.polarity.enabled}(p={aug.polarity.prob})  "
            f"pitch_shift={aug.pitch_shift.enabled}(p={aug.pitch_shift.prob}, "
            f"+/-{aug.pitch_shift.semitones_max}st)  "
            f"noise={aug.noise.enabled}(p={aug.noise.prob}, sigma={aug.noise.sigma})  "
            f"mp3={aug.mp3_degradation.enabled}(p={aug.mp3_degradation.prob}, "
            f"{aug.mp3_degradation.kbps_min}-{aug.mp3_degradation.kbps_max}kbps)"
        )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        lq_path, hq_path = self.pairs[idx]
        lq = load_wav(lq_path, self.sr)
        hq = load_wav(hq_path, self.sr)
        # No per-chunk normalize_pair here -- chunks are already normalized at
        # the full-song level during _slice_and_save. Normalizing again per-chunk
        # would re-introduce inconsistent gain riding at chunk boundaries.
        lq, hq = augment_pair(lq, hq, self.aug_cfg, sr=self.sr, idx=idx)
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
            self.data_val = FullLengthPairDataset(
                eval_dir=self.eval_dir,
                sr=self.sr,
                segment_sec=self.segment_sec,
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
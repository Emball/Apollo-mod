"""
Paired audio datamodule for Apollo fine-tuning.

Training data (chunks directory):
    chunks/
        LQ/
            track1_0000.wav
            track1_0001.wav
        HQ/
            track1_0000.wav
            track1_0001.wav

Validation data (eval directory):
    eval/
        LQ/
            track1.wav
        HQ/
            track1.wav

Validation loads full-length files and slices them at runtime.
This allows using the real leak/HQ pairs as validation without
needing to pre-chunk them.
"""

import os
import random
from typing import Optional, Tuple, List

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule

# ---------------------------------------------------------------------------
# Augmentation — applied identically to both LQ and HQ
# ---------------------------------------------------------------------------

# Probability each augmentation fires on a given sample
AUG_PROB_GAIN         = 0.5   # random gain ± GAIN_DB_MAX
AUG_PROB_POLARITY     = 0.5   # flip polarity (×-1)
AUG_PROB_STEREO_SWAP  = 0.5   # swap L and R channels

GAIN_DB_MAX = 1.5             # max gain shift in either direction (dB)


def augment_pair(lq: torch.Tensor, hq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply random augmentations to an LQ/HQ pair.
    Every transform is applied identically to both tensors so the
    paired relationship is never broken.
    Shape: (2, samples)
    """
    # -- Gain --
    if random.random() < AUG_PROB_GAIN:
        db    = random.uniform(-GAIN_DB_MAX, GAIN_DB_MAX)
        scale = 10 ** (db / 20.0)
        lq    = lq * scale
        hq    = hq * scale

    # -- Polarity inversion --
    if random.random() < AUG_PROB_POLARITY:
        lq = -lq
        hq = -hq

    # -- Stereo channel swap --
    if random.random() < AUG_PROB_STEREO_SWAP:
        lq = lq.flip(0)
        hq = hq.flip(0)

    return lq, hq


SR = 44100


def load_wav(path: str, target_sr: int = SR) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    # Force stereo
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


# ---------------------------------------------------------------------------
# Training dataset — loads pre-chunked files
# ---------------------------------------------------------------------------

class ChunkedPairDataset(Dataset):
    def __init__(self, chunks_dir: str, sr: int = SR):
        lq_dir = os.path.join(chunks_dir, "LQ")
        hq_dir = os.path.join(chunks_dir, "HQ")
        self.pairs = get_matched_pairs(lq_dir, hq_dir)
        self.sr = sr
        print(f"Training dataset: {len(self.pairs)} chunk pairs")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        lq_path, hq_path = self.pairs[idx]
        lq = load_wav(lq_path, self.sr)
        hq = load_wav(hq_path, self.sr)
        lq, hq = normalize_pair(lq, hq)
        lq, hq = augment_pair(lq, hq)
        # Apollo expects (ori_data, codec_data) i.e. (clean, degraded)
        return hq, lq


# ---------------------------------------------------------------------------
# Validation dataset — loads full-length files, slices at runtime
# ---------------------------------------------------------------------------

class FullLengthPairDataset(Dataset):
    def __init__(self, eval_dir: str, sr: int = SR, segment_sec: float = 2.0):
        lq_dir = os.path.join(eval_dir, "LQ")
        hq_dir = os.path.join(eval_dir, "HQ")
        self.pairs = get_matched_pairs(lq_dir, hq_dir)
        self.sr = sr
        self.segment_samples = int(segment_sec * sr)
        self.hop_samples = self.segment_samples // 2

        # Pre-build index: (pair_idx, start_sample)
        self.index = []
        for pair_idx, (lq_path, hq_path) in enumerate(self.pairs):
            lq = load_wav(lq_path, sr)
            hq = load_wav(hq_path, sr)
            min_len = min(lq.shape[-1], hq.shape[-1])
            start = 0
            while start + self.segment_samples <= min_len:
                self.index.append((pair_idx, start))
                start += self.hop_samples

        # Cache loaded audio to avoid reloading on every __getitem__
        self._cache = {}
        for pair_idx, (lq_path, hq_path) in enumerate(self.pairs):
            lq = load_wav(lq_path, sr)
            hq = load_wav(hq_path, sr)
            min_len = min(lq.shape[-1], hq.shape[-1])
            self._cache[pair_idx] = (lq[:, :min_len], hq[:, :min_len])

        print(f"Validation dataset: {len(self.pairs)} files → {len(self.index)} segments")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        pair_idx, start = self.index[idx]
        lq, hq = self._cache[pair_idx]
        lq_chunk = lq[:, start:start + self.segment_samples]
        hq_chunk = hq[:, start:start + self.segment_samples]
        lq_chunk, hq_chunk = normalize_pair(lq_chunk, hq_chunk)
        return hq_chunk, lq_chunk


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------

class PairedAudioDataModule(LightningDataModule):
    def __init__(
        self,
        train_dir: str,
        eval_dir: str,
        sr: int = SR,
        segment_sec: float = 2.0,
        batch_size: int = 1,
        num_workers: int = 4,
    ):
        super().__init__()
        self.train_dir = train_dir
        self.eval_dir = eval_dir
        self.sr = sr
        self.segment_sec = segment_sec
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        if self.data_train is None:
            self.data_train = ChunkedPairDataset(
                chunks_dir=self.train_dir,
                sr=self.sr,
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
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

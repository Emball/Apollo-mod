# @claude last-modified: 2026-05-05T06:34:39Z
# @claude last-commit: feat: major update — TUI, augmentation system, gradient checkpointing, optimization bootstrap
#!/usr/bin/env python3
"""
Augmentation test script.

Defaults to chunks/train — the same chunked data the dataloader actually sees.
If chunks/train doesn't exist yet, it will be created automatically from
data/train before running (same chunker used by train.py).

Usage:
    python test_augmentations.py
    python test_augmentations.py --split val --n 5
    python test_augmentations.py --src chunks/train --n 3

After running, check augmented_test/ — load pairs in your DAW and verify
LQ and HQ are time-aligned with no drift.
"""

import argparse
import os
import random
import shutil
import sys

import torch
import torchaudio

from paired_datamodule import (
    AugmentationCfg,
    GainAugCfg,
    SimpleAugCfg,
    PitchShiftAugCfg,
    NoiseAugCfg,
    Mp3AugCfg,
    augment_pair,
    load_wav,
    normalize_pair,
    get_matched_pairs,
)

SR = 44100
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def ensure_chunks(split: str, sr: int) -> str:
    """Return path to chunks/<split>, creating them from data/<split> if needed."""
    chunks_dir = os.path.join(SCRIPT_DIR, "chunks", split)
    lq_dir = os.path.join(chunks_dir, "LQ")

    if os.path.isdir(lq_dir) and any(f.endswith(".wav") for f in os.listdir(lq_dir)):
        n = sum(1 for f in os.listdir(lq_dir) if f.endswith(".wav"))
        print(f"chunks/{split} already exists ({n} pairs) — skipping chunking.")
        return chunks_dir

    data_dir = os.path.join(SCRIPT_DIR, "data", split)
    if not os.path.isdir(data_dir):
        print(f"ERROR: Neither chunks/{split} nor data/{split} found.")
        print(f"  Expected chunks at : {chunks_dir}")
        print(f"  Or source data at  : {data_dir}")
        sys.exit(1)

    print(f"chunks/{split} not found — chunking from data/{split} ...")
    sys.path.insert(0, SCRIPT_DIR)
    from train import _chunk_split
    n = _chunk_split(data_dir, chunks_dir, split)
    if n == 0:
        print(f"ERROR: Chunking produced 0 pairs. Check data/{split}/LQ and HQ.")
        sys.exit(1)
    print(f"Chunking done — {n} pairs written to chunks/{split}")
    return chunks_dir


def make_cfg(**kwargs) -> AugmentationCfg:
    """Base config with everything disabled, then enable what's passed via kwargs."""
    defaults = dict(
        enabled=True,
        gain=GainAugCfg(enabled=False),
        polarity=SimpleAugCfg(enabled=False),
        pitch_shift=PitchShiftAugCfg(enabled=False),
        noise=NoiseAugCfg(enabled=False),
        mp3_degradation=Mp3AugCfg(enabled=False),
        mono_channel=SimpleAugCfg(enabled=False),
    )
    defaults.update(kwargs)
    return AugmentationCfg(**defaults)


AUGMENTATION_CONFIGS = {
    "mono_channel": make_cfg(
        mono_channel=SimpleAugCfg(enabled=True, prob=1.0),
    ),
    "gain": make_cfg(
        gain=GainAugCfg(enabled=True, prob=1.0, db_max=1.5),
    ),
    "polarity": make_cfg(
        polarity=SimpleAugCfg(enabled=True, prob=1.0),
    ),
    "pitch_shift": make_cfg(
        pitch_shift=PitchShiftAugCfg(enabled=True, prob=1.0, semitones_max=1.5),
    ),
    "noise": make_cfg(
        noise=NoiseAugCfg(enabled=True, prob=1.0, sigma=0.002),
    ),
    "mp3_degradation": make_cfg(
        mp3_degradation=Mp3AugCfg(enabled=True, prob=1.0, kbps_min=64, kbps_max=256),
    ),
}


def save(tensor: torch.Tensor, path: str, sr: int):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torchaudio.save(path, tensor.float().clamp(-1.0, 1.0), sr)


def main():
    parser = argparse.ArgumentParser(description="Test augmentations on chunked data")
    parser.add_argument("--split", default="train", choices=["train", "val"],
                        help="Which split to use (default: train)")
    parser.add_argument("--src", default=None,
                        help="Override chunk dir directly (skips auto-chunking)")
    parser.add_argument("--out", default="augmented_test", help="Output directory")
    parser.add_argument("--n", type=int, default=3, help="Number of chunks to process")
    parser.add_argument("--sr", type=int, default=SR, help="Sample rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.src:
        chunks_dir = os.path.join(SCRIPT_DIR, args.src)
    else:
        chunks_dir = ensure_chunks(args.split, args.sr)

    lq_dir = os.path.join(chunks_dir, "LQ")
    hq_dir = os.path.join(chunks_dir, "HQ")

    if not os.path.isdir(lq_dir) or not os.path.isdir(hq_dir):
        print(f"ERROR: Could not find LQ/HQ in {chunks_dir}")
        sys.exit(1)

    pairs = get_matched_pairs(lq_dir, hq_dir)
    n = min(args.n, len(pairs))
    selected = random.sample(pairs, n)
    out_dir = os.path.join(SCRIPT_DIR, args.out)

    print(f"\nProcessing {n} chunks from {chunks_dir}")
    print(f"Output → {out_dir}\n")

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    for aug_name, cfg in AUGMENTATION_CONFIGS.items():
        print(f"  [{aug_name}]")
        for lq_path, hq_path in selected:
            stem = os.path.splitext(os.path.basename(lq_path))[0]

            lq = load_wav(lq_path, args.sr)
            hq = load_wav(hq_path, args.sr)
            lq, hq = normalize_pair(lq, hq)

            original_lq_len = lq.shape[-1]
            original_hq_len = hq.shape[-1]

            lq_aug, hq_aug = augment_pair(lq, hq, cfg, sr=args.sr)

            if lq_aug.shape[-1] != original_lq_len:
                print(f"    WARNING: LQ length changed {original_lq_len} → {lq_aug.shape[-1]} ({stem})")
            if hq_aug.shape[-1] != original_hq_len:
                print(f"    WARNING: HQ length changed {original_hq_len} → {hq_aug.shape[-1]} ({stem})")
            if lq_aug.shape[-1] != hq_aug.shape[-1]:
                print(f"    WARNING: LQ/HQ length mismatch: {lq_aug.shape[-1]} vs {hq_aug.shape[-1]} ({stem})")
            else:
                print(f"    OK  {stem}  shape={tuple(lq_aug.shape)}  len={lq_aug.shape[-1]}")

            save(lq_aug, os.path.join(out_dir, aug_name, "LQ", f"{stem}.wav"), args.sr)
            save(hq_aug, os.path.join(out_dir, aug_name, "HQ", f"{stem}.wav"), args.sr)

    print(f"\nDone. Results in {out_dir}/")
    print("Each subfolder is one augmentation in isolation, plus 'all_enabled'.")
    print("Load LQ and HQ from the same subfolder side-by-side in your DAW to check alignment.")


if __name__ == "__main__":
    main()

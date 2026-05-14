#!/usr/bin/env python3
"""
Align heavily degraded audio (LQ) to clean reference (HQ).

Loads paired WAV files, corrects timing drift by cross-correlating
short overlapping chunks, then saves aligned LQ — ready for the
training pipeline's chunking step.

Usage:
  # Single pair
  python align.py --lq voice_lq.wav --hq voice_hq.wav --out voice_aligned.wav

  # Batch: align all pairs in data/train/ and data/val/ in-place
  python align.py --data-dir data
"""

import torch
import torchaudio
import argparse
import os
import sys
import numpy as np
import scipy.ndimage


def _load(path, sr=44100):
    wav, rsr = torchaudio.load(path)
    if rsr != sr:
        wav = torchaudio.functional.resample(wav, rsr, sr)
    return wav, sr


def _save(path, wav, sr):
    torchaudio.save(path, wav.cpu(), sr)


def _xcorr_peak(a, b, max_shift):
    n = a.shape[0] + b.shape[0] - 1
    n_fft = 1 << (n - 1).bit_length()
    A = torch.fft.rfft(a, n_fft)
    B = torch.fft.rfft(b, n_fft)
    corr = torch.fft.irfft(A * B.conj(), n_fft)
    center = b.shape[0] - 1
    lo = max(0, center - max_shift)
    hi = min(n_fft, center + max_shift + 1)
    peak = lo + torch.argmax(corr[lo:hi])
    return peak - center


def _correct_start(lq, hq, sr, max_off=int(0.5 * sr)):
    """Stage 1: find global start offset and pad/trim LQ."""
    win = int(min(2.0 * sr, lq.shape[0], hq.shape[0]))
    off = _xcorr_peak(lq[:win], hq[:win], max_off).item()
    if abs(off) > int(0.001 * sr):
        print(f"    Stage 1: start offset {off:+d} samples", flush=True)
    return off


def _correct_speed(lq, hq, sr, max_off=int(0.1 * sr)):
    """Stage 2: measure end drift after start fix, resample LQ to match speed."""
    win = int(min(2.0 * sr, lq.shape[0], hq.shape[0]))
    off = _xcorr_peak(lq[-win:], hq[-win:], max_off).item()
    if abs(off) < int(0.001 * sr):
        return lq, 0
    drift = off
    print(f"    Stage 2: end drift {drift:+d} samples — correcting speed", flush=True)
    n = lq.shape[-1]
    new_len = int(round(n + drift))
    new_len = max(new_len, 1)
    chans = []
    for ch in range(lq.shape[0]):
        x = lq[ch:ch+1, :n].unsqueeze(0)
        y = torch.nn.functional.interpolate(
            x.float(), size=new_len, mode='linear', align_corners=False
        )
        chans.append(y.squeeze(0))
    lq = torch.cat(chans, dim=0)
    return lq, drift


def _align_chunks(lq_wav, hq_wav, sr, chunk_sec=0.5, overlap=0.5, search_ms=200):
    """Stage 3: cut-and-crossfade chunk alignment for residual local drift."""
    lq = lq_wav.mean(dim=0)
    hq = hq_wav.mean(dim=0)
    n = min(lq.shape[0], hq.shape[0])
    lq = lq[:n]
    hq = hq[:n]

    chunk = int(chunk_sec * sr)
    hop = int(chunk * (1 - overlap))
    max_shift = int(search_ms / 1000 * sr)

    offsets = []
    for start in range(0, n - chunk + 1, hop):
        a = lq[start:start + chunk]
        b = hq[start:start + chunk]
        off = _xcorr_peak(a, b, max_shift)
        offsets.append(off.item())

    if not offsets:
        return lq_wav[:, :n]

    o = torch.tensor(offsets, dtype=torch.float)
    if len(o) > 3:
        o = torch.tensor(scipy.ndimage.median_filter(o.numpy(), size=3))
    o = o.round().long()

    nch = lq_wav.shape[0]
    out = torch.zeros((nch, n), device=lq_wav.device)
    weight = torch.zeros(n, device=lq_wav.device)
    win = torch.hann_window(chunk, device=lq_wav.device)

    for i, start in enumerate(range(0, n - chunk + 1, hop)):
        src_start = start + int(o[i])
        if src_start < 0 or src_start + chunk > n:
            continue
        for ch in range(nch):
            out[ch, start:start + chunk] += lq_wav[ch, src_start:src_start + chunk] * win
        weight[start:start + chunk] += win

    weight = weight.clamp_min(1e-10)
    for ch in range(nch):
        out[ch] /= weight
    return out


def align_pair(lq_wav, hq_wav, sr, chunk_sec=0.5, overlap=0.5, search_ms=200):
    lq = lq_wav.mean(dim=0)
    hq = hq_wav.mean(dim=0)

    # Stage 1: start offset
    start_off = _correct_start(lq, hq, sr)
    n = min(lq.shape[0], hq.shape[0])
    if start_off > 0:
        lq_wav = torch.nn.functional.pad(lq_wav, (start_off, 0))[:, :n]
    elif start_off < 0:
        lq_wav = torch.nn.functional.pad(lq_wav[:, -start_off:], (0, -start_off))[:, :n]
    else:
        lq_wav = lq_wav[:, :n]
    hq_wav = hq_wav[:, :n]
    lq = lq_wav.mean(dim=0)
    hq = hq_wav.mean(dim=0)

    # Stage 2: global speed correction
    lq_wav, drift = _correct_speed(lq, hq, sr)
    n = min(lq_wav.shape[1], hq_wav.shape[1])
    lq_wav = lq_wav[:, :n]
    hq_wav = hq_wav[:, :n]

    # Stage 3: chunked fine alignment
    aligned = _align_chunks(lq_wav, hq_wav, sr, chunk_sec, overlap, search_ms)
    return aligned


def process_pair(lq_path, hq_path, out_path=None, chunk_sec=0.5,
                 overlap=0.5, search_ms=200, sr=44100, in_place=False):
    print(f"  Loading {os.path.basename(lq_path)}", flush=True)
    lq, sr = _load(lq_path, sr)
    hq, _ = _load(hq_path, sr)

    min_len = min(lq.shape[1], hq.shape[1])
    lq = lq[:, :min_len]
    hq = hq[:, :min_len]

    print(f"  Aligning ...", flush=True)
    aligned = align_pair(lq, hq, sr, chunk_sec, overlap, search_ms)

    dst = out_path or lq_path
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    _save(dst, aligned[:, :min_len], sr)
    print(f"  Saved {dst}", flush=True)


def _find_pairs(root):
    pairs = []
    for split in ("train", "val"):
        lq_dir = os.path.join(root, split, "LQ")
        hq_dir = os.path.join(root, split, "HQ")
        if not (os.path.isdir(lq_dir) and os.path.isdir(hq_dir)):
            continue
        lq_files = {os.path.splitext(f)[0]: f
                    for f in os.listdir(lq_dir) if f.endswith(".wav")}
        hq_files = {os.path.splitext(f)[0]: f
                    for f in os.listdir(hq_dir) if f.endswith(".wav")}
        for stem in sorted(set(lq_files) & set(hq_files)):
            pairs.append((
                os.path.join(lq_dir, lq_files[stem]),
                os.path.join(hq_dir, hq_files[stem]),
                os.path.join(lq_dir, lq_files[stem]),
                split
            ))
    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="Align degraded LQ audio to clean HQ reference"
    )
    g = parser.add_argument_group("single-file mode")
    g.add_argument("--lq", help="Path to degraded audio")
    g.add_argument("--hq", help="Path to clean reference")
    g.add_argument("--out", default=None, help="Output path (default: overwrite LQ)")

    g = parser.add_argument_group("batch mode")
    g.add_argument("--data-dir", default=None,
                   help="Path to data/ directory with train+LQ/HQ and val+LQ/HQ")

    g = parser.add_argument_group("alignment settings")
    g.add_argument("--chunk-sec", type=float, default=0.5,
                   help="Analysis chunk length in seconds (default: 0.5)")
    g.add_argument("--overlap", type=float, default=0.5,
                   help="Overlap fraction between chunks (default: 0.5)")
    g.add_argument("--search-ms", type=int, default=200,
                   help="Max offset search range in ms (default: 200)")
    g.add_argument("--sr", type=int, default=44100,
                   help="Target sample rate (default: 44100)")

    args = parser.parse_args()

    if args.data_dir:
        pairs = _find_pairs(args.data_dir)
        if not pairs:
            print("No matched LQ/HQ pairs found under", args.data_dir)
            sys.exit(1)
        print(f"Found {len(pairs)} pairs to align")
        for lq_path, hq_path, out_path, split in pairs:
            print(f"[{split}]")
            process_pair(lq_path, hq_path, out_path,
                         args.chunk_sec, args.overlap, args.search_ms, args.sr)
        print("All pairs aligned.")
    elif args.lq and args.hq:
        process_pair(args.lq, args.hq, args.out,
                     args.chunk_sec, args.overlap, args.search_ms, args.sr)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

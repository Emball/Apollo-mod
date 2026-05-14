#!/usr/bin/env python3
"""
Align heavily degraded audio (LQ) to a clean reference (HQ) by computing
a time-varying delay via STFT-magnitude cross-correlation and applying it
to the full file. Runs on entire data/ directories in-place before training.

Usage:
  # Single pair
  python align.py --lq input_lq.wav --hq input_hq.wav

  # Batch: align all pairs in data/train and data/val in-place
  python align.py --data_dir data

  # Custom directories
  python align.py --lq_dir my/lq --hq_dir my/hq --out_dir my/aligned
"""

import torch
import torchaudio
import numpy as np
from scipy.ndimage import median_filter
import argparse, os, sys, glob


def load_audio(path, target_sr=44100):
    wav, sr = torchaudio.load(path)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav, target_sr


def mono(wav):
    return wav.mean(dim=0)


def stft_mag(x, n_fft=1024, hop_length=256, win_length=1024):
    window = torch.hann_window(win_length, device=x.device)
    z = torch.stft(x, n_fft, hop_length, win_length, window, return_complex=True)
    return 20 * torch.log10(z.abs().clamp_min(1e-10))


@torch.no_grad()
def compute_offsets(lq, hq, sr, win_sec=1.0, hop_sec=0.25, search_ms=200,
                    n_fft=1024, hop_length=256, win_length=1024, device='cpu'):
    """
    Compute a per-sample time-varying offset curve between LQ and HQ.
    """
    lq = lq.to(device)
    hq = hq.to(device)

    window = torch.hann_window(win_length, device=device)
    def _stft(x):
        z = torch.stft(x, n_fft, hop_length, win_length, window, return_complex=True)
        return 20 * torch.log10(z.abs().clamp_min(1e-10))

    lq_mag = _stft(lq)
    hq_mag = _stft(hq)

    min_frames = min(lq_mag.shape[1], hq_mag.shape[1])
    lq_mag = lq_mag[:, :min_frames]
    hq_mag = hq_mag[:, :min_frames]
    min_samples = min_frames * hop_length + win_length
    lq = lq[:min_samples]
    hq = hq[:min_samples]

    win_frames = int(win_sec * sr / hop_length)
    hop_frames = max(1, int(hop_sec * sr / hop_length))
    max_shift = int(search_ms / 1000 * sr / hop_length)

    positions = []
    shifts = []

    for start in range(0, min_frames - win_frames + 1, hop_frames):
        end = start + win_frames
        lc = lq_mag[:, start:end]
        hc = hq_mag[:, start:end]

        lc = (lc - lc.mean(dim=1, keepdim=True)) / (lc.std(dim=1, keepdim=True) + 1e-10)
        hc = (hc - hc.mean(dim=1, keepdim=True)) / (hc.std(dim=1, keepdim=True) + 1e-10)

        n = win_frames * 2 - 1
        lc_pad = torch.nn.functional.pad(lc, (0, n - win_frames))
        hc_pad = torch.nn.functional.pad(hc, (0, n - win_frames))
        corr = torch.fft.irfft(
            torch.fft.rfft(lc_pad) * torch.fft.rfft(hc_pad).conj(), dim=1
        ).mean(dim=0)

        center = win_frames - 1
        lo = max(0, center - max_shift)
        hi = min(n, center + max_shift + 1)
        peak = lo + torch.argmax(corr[lo:hi])
        shift = peak - center

        positions.append(start + win_frames // 2)
        shifts.append(shift.item())

    if not shifts:
        return torch.zeros(min_samples, device=device)

    pos = torch.tensor(positions, dtype=torch.float, device=device)
    sh = torch.tensor(shifts, dtype=torch.float, device=device)
    sh_np = median_filter(sh.cpu().numpy(), size=3)
    sh = torch.tensor(sh_np, dtype=torch.float, device=device)

    all_frames = np.arange(min_frames)
    if len(positions) > 1:
        frame_offsets_np = np.interp(all_frames, pos.cpu().numpy(), sh.cpu().numpy())
    else:
        frame_offsets_np = np.full(min_frames, sh.item())
    frame_offsets = torch.from_numpy(frame_offsets_np).to(device)

    return lq, hq, frame_offsets * hop_length


def apply_offsets(wav, offsets):
    n_channels, n_samples = wav.shape
    out = torch.zeros_like(wav)
    clamp = int(n_samples * 0.15)
    src_pos = torch.arange(n_samples, device=wav.device, dtype=torch.float) + offsets
    src_pos = src_pos.clamp(0, n_samples - 1)

    for ch in range(n_channels):
        idx = src_pos.long()
        frac = src_pos - idx.float()
        idx = idx.clamp(0, n_samples - 2)
        out[ch] = wav[ch, idx] * (1 - frac) + wav[ch, idx + 1] * frac

    return out


def align_pair(lq_path, hq_path, out_lq=None, out_hq=None,
               sr=44100, device='cpu', **kwargs):
    lq, sr = load_audio(lq_path, sr)
    hq, _ = load_audio(hq_path, sr)

    min_len = min(lq.shape[1], hq.shape[1])
    lq = lq[:, :min_len]
    hq = hq[:, :min_len]

    lq_mono = mono(lq)
    hq_mono = mono(hq)

    lq_trimmed, hq_trimmed, offsets = compute_offsets(
        lq_mono, hq_mono, sr, device=device, **kwargs
    )

    min_samp = min(lq.shape[1], hq.shape[1], offsets.shape[0])
    offsets = offsets[:min_samp]
    lq = lq[:, :min_samp]
    hq = hq[:, :min_samp]

    aligned = apply_offsets(lq, offsets)

    out_lq = out_lq or lq_path
    out_hq = out_hq or hq_path

    os.makedirs(os.path.dirname(out_lq) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(out_hq) or '.', exist_ok=True)

    torchaudio.save(out_lq, aligned.cpu(), sr)
    torchaudio.save(out_hq, hq[:, :min_samp].cpu(), sr)

    print(f"  Aligned {os.path.basename(lq_path)}"
          f"  (max offset: {offsets.abs().max().item():.1f} samples, "
          f"mean: {offsets.mean().item():.2f})")


def get_matched_pairs(lq_dir, hq_dir):
    lq_files = {os.path.splitext(f)[0]: f
                for f in os.listdir(lq_dir) if f.endswith('.wav')}
    hq_files = {os.path.splitext(f)[0]: f
                for f in os.listdir(hq_dir) if f.endswith('.wav')}
    matched = sorted(set(lq_files) & set(hq_files))
    return [(os.path.join(lq_dir, lq_files[s]),
             os.path.join(hq_dir, hq_files[s])) for s in matched]


def main():
    parser = argparse.ArgumentParser(
        description='Align degraded LQ audio to clean HQ reference.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--lq', help='Single LQ file')
    group.add_argument('--data_dir', help='Root data/ dir (scans train+LQ/HQ, val+LQ/HQ)')
    group.add_argument('--lq_dir', help='Batch: directory of LQ files')

    parser.add_argument('--hq', help='Single HQ file (required with --lq)')
    parser.add_argument('--hq_dir', help='Batch: directory of HQ files (required with --lq_dir)')
    parser.add_argument('--out_dir', help='Output directory (default: in-place)')
    parser.add_argument('--sr', type=int, default=44100)
    parser.add_argument('--win_sec', type=float, default=1.0,
                        help='Analysis window size in seconds')
    parser.add_argument('--hop_sec', type=float, default=0.25,
                        help='Analysis hop size in seconds')
    parser.add_argument('--search_ms', type=int, default=200,
                        help='Max offset search range in milliseconds')
    parser.add_argument('--device', default='auto',
                        help='cpu, cuda, or auto')

    args = parser.parse_args()
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    kw = dict(win_sec=args.win_sec, hop_sec=args.hop_sec,
              search_ms=args.search_ms, sr=args.sr, device=device)

    if args.lq:
        if not args.hq:
            print("error: --hq is required with --lq", file=sys.stderr)
            sys.exit(1)
        out_lq = None
        out_hq = None
        if args.out_dir:
            os.makedirs(args.out_dir, exist_ok=True)
            out_lq = os.path.join(args.out_dir, os.path.basename(args.lq))
            out_hq = os.path.join(args.out_dir, os.path.basename(args.hq))
        align_pair(args.lq, args.hq, out_lq, out_hq, **kw)
        return

    if args.lq_dir:
        if not args.hq_dir:
            print("error: --hq_dir is required with --lq_dir", file=sys.stderr)
            sys.exit(1)
        pairs = get_matched_pairs(args.lq_dir, args.hq_dir)
        if not pairs:
            print(f"No matched WAV pairs found in {args.lq_dir} / {args.hq_dir}")
            sys.exit(1)
        print(f"Aligning {len(pairs)} pairs from {args.lq_dir} / {args.hq_dir}")
        for lq_path, hq_path in pairs:
            out_lq = out_hq = None
            if args.out_dir:
                out_lq = os.path.join(args.out_dir, 'LQ', os.path.basename(lq_path))
                out_hq = os.path.join(args.out_dir, 'HQ', os.path.basename(hq_path))
            align_pair(lq_path, hq_path, out_lq, out_hq, **kw)
        return

    # --data_dir mode: scan data/train and data/val
    root = args.data_dir
    for split in ('train', 'val'):
        lq_dir = os.path.join(root, split, 'LQ')
        hq_dir = os.path.join(root, split, 'HQ')
        if not (os.path.isdir(lq_dir) and os.path.isdir(hq_dir)):
            print(f"Skipping {split}/ (LQ/HQ dirs not found)")
            continue
        pairs = get_matched_pairs(lq_dir, hq_dir)
        if not pairs:
            print(f"No matched pairs in {split}/")
            continue
        print(f"[{split}/] Aligning {len(pairs)} pairs in-place ...")
        for lq_path, hq_path in pairs:
            align_pair(lq_path, hq_path, **kw)
        print(f"[{split}/] Done\n")

    print("All pairs aligned.")


if __name__ == '__main__':
    main()

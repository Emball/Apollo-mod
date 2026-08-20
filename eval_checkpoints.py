"""
eval_checkpoints.py -- Retroactively evaluate checkpoints against the val set.

Loads each checkpoint, runs it against your val chunks, computes SI-SDR /
msstft / sfr, prints a ranked table, and optionally renames checkpoint files
with the accurate loss value.

Usage:
    python eval_checkpoints.py --conf_dir configs/apollo_stfl2.yaml
    python eval_checkpoints.py --conf_dir configs/apollo_stfl2.yaml --rename
    python eval_checkpoints.py --conf_dir configs/apollo_stfl2.yaml --ckpt_dir runs/apollo_stfl2/20260819_000000/checkpoints
"""

import argparse
import os
import re
import sys

import torch
import torchaudio
from omegaconf import OmegaConf

# Reuse metric functions from the litmodule
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from look2hear.system.audio_litmodule import _ms_log_stft_loss, _spectral_flatness_ratio
import look2hear.models.apollo
import look2hear.losses

_SR = 44100


def _load_model(ckpt_path, feature_dim, sr, win, layer):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if not isinstance(ckpt, dict):
        raise ValueError(f"Checkpoint must be a dict: {ckpt_path}")
    state = ckpt.get("state_dict", ckpt)
    # Strip 'audio_model.' prefix if present
    cleaned = {}
    for k, v in state.items():
        if k.startswith("audio_model."):
            cleaned[k[len("audio_model."):]] = v
        elif not any(k.startswith(p) for p in ("discriminator.", "metrics.", "optimizer")):
            cleaned[k] = v
    model = look2hear.models.apollo.Apollo(
        sr=sr, win=win, feature_dim=feature_dim, layer=layer
    )
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"  [warn] {len(missing)} missing keys")
    return model.cuda().eval()


def _load_val_chunks(val_chunk_dir, limit=None):
    """Load all LQ/HQ chunk pairs from val chunk dir."""
    lq_dir = os.path.join(val_chunk_dir, "LQ")
    hq_dir = os.path.join(val_chunk_dir, "HQ")
    if not os.path.isdir(lq_dir) or not os.path.isdir(hq_dir):
        raise RuntimeError(f"Val chunk dirs not found: {lq_dir}, {hq_dir}")

    files = sorted(f for f in os.listdir(lq_dir) if f.endswith(".wav"))
    if limit:
        # Stratify by song prefix
        from collections import defaultdict
        by_song = defaultdict(list)
        for f in files:
            song = "_".join(f.rsplit("_", 1)[:-1]) if "_" in f else f
            by_song[song].append(f)
        per_song = max(1, limit // len(by_song))
        selected = []
        for song_files in by_song.values():
            selected.extend(song_files[:per_song])
        files = sorted(selected)[:limit]

    chunks = []
    for fname in files:
        lq_path = os.path.join(lq_dir, fname)
        hq_path = os.path.join(hq_dir, fname)
        if os.path.exists(hq_path):
            chunks.append((lq_path, hq_path, fname))
    return chunks


def _normalize_pair(lq, hq):
    peak = max(lq.abs().amax().item(), hq.abs().amax().item())
    if peak > 0:
        lq = lq / peak
        hq = hq / peak
    return lq, hq


def _eval_checkpoint(model, chunks, device):
    sisdr_metric = look2hear.losses.MultiSrcNegSDR("sisdr")
    sisdr_sum = 0.0
    msstft_sum = 0.0
    sfr_sum = 0.0
    n = 0

    with torch.no_grad():
        for lq_path, hq_path, fname in chunks:
            try:
                lq, _ = torchaudio.load(lq_path)
                hq, _ = torchaudio.load(hq_path)
                lq, hq = _normalize_pair(lq, hq)

                # Apollo expects (B, C, T) -- single channel
                if lq.shape[0] == 2:
                    lq = lq[0:1]
                    hq = hq[0:1]
                lq_in = lq.unsqueeze(0).to(device)
                hq_ref = hq.unsqueeze(0).to(device)

                out = model(lq_in)

                # SI-SDR
                sisdr_val = -float(sisdr_metric(out, hq_ref).mean())
                sisdr_sum += sisdr_val

                # Perceptual metrics (CPU)
                e = out.squeeze(0).cpu()
                r = hq_ref.squeeze(0).cpu()
                msstft_sum += _ms_log_stft_loss(e, r)
                sfr_sum += _spectral_flatness_ratio(e, r)
                n += 1
            except Exception as ex:
                print(f"  [skip] {fname}: {ex}")

    if n == 0:
        return None
    return {
        "sisdr":  sisdr_sum  / n,
        "msstft": msstft_sum / n,
        "sfr":    sfr_sum    / n,
        "n":      n,
    }


def main():
    parser = argparse.ArgumentParser(description="Retroactively evaluate Apollo checkpoints")
    parser.add_argument("--conf_dir", required=True, help="Path to yaml config")
    parser.add_argument("--ckpt_dir", default=None, help="Checkpoint folder (auto-detected from config if omitted)")
    parser.add_argument("--limit", type=int, default=100, help="Max val chunks to evaluate per checkpoint (default 100)")
    parser.add_argument("--rename", action="store_true", help="Rename checkpoint files with accurate val_loss")
    parser.add_argument("--pattern", default=None, help="Only evaluate checkpoints matching this substring")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.conf_dir)
    exp_dir  = cfg.exp.dir
    exp_name = cfg.exp.name
    feature_dim = int(cfg.model.feature_dim)
    sr          = int(cfg.model.sr)
    win         = int(cfg.model.win)
    layer       = int(cfg.model.layer)

    # Auto-detect checkpoint dir
    if args.ckpt_dir:
        ckpt_dir = args.ckpt_dir
    else:
        run_root = os.path.join(exp_dir, exp_name)
        if not os.path.isdir(run_root):
            print(f"[error] Run root not found: {run_root}")
            sys.exit(1)
        # Find most recent timestamped run
        runs = sorted(
            (d for d in os.listdir(run_root) if os.path.isdir(os.path.join(run_root, d))),
            reverse=True
        )
        if not runs:
            print(f"[error] No runs found in {run_root}")
            sys.exit(1)
        ckpt_dir = os.path.join(run_root, runs[0], "checkpoints")
        print(f"[eval] Using run: {runs[0]}")

    if not os.path.isdir(ckpt_dir):
        print(f"[error] Checkpoint dir not found: {ckpt_dir}")
        sys.exit(1)

    # Find val chunk dir
    # Convention: chunks/<exp_name>/val/
    chunk_root = os.path.join("chunks", exp_name, "val")
    if not os.path.isdir(chunk_root):
        print(f"[error] Val chunk dir not found: {chunk_root}")
        print("Run train.py once to generate chunks.")
        sys.exit(1)

    print(f"[eval] Loading val chunks from: {chunk_root}")
    chunks = _load_val_chunks(chunk_root, limit=args.limit)
    print(f"[eval] {len(chunks)} chunks selected for evaluation")

    # Find checkpoints
    ckpt_files = sorted(
        f for f in os.listdir(ckpt_dir)
        if f.endswith(".ckpt") and (args.pattern is None or args.pattern in f)
    )
    if not ckpt_files:
        print(f"[error] No checkpoints found in {ckpt_dir}")
        sys.exit(1)

    print(f"[eval] Found {len(ckpt_files)} checkpoints\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []

    for fname in ckpt_files:
        ckpt_path = os.path.join(ckpt_dir, fname)
        print(f"  Evaluating: {fname} ...", end=" ", flush=True)
        try:
            model = _load_model(ckpt_path, feature_dim, sr, win, layer)
            metrics = _eval_checkpoint(model, chunks, device)
            del model
            torch.cuda.empty_cache()
            if metrics:
                print(f"sisdr={metrics['sisdr']:.4f}  msstft={metrics['msstft']:.4f}  sfr={metrics['sfr']:.4f}")
                results.append((fname, ckpt_path, metrics))
            else:
                print("no valid chunks")
        except Exception as ex:
            print(f"ERROR: {ex}")

    if not results:
        print("\n[eval] No results.")
        sys.exit(1)

    # Sort by SI-SDR descending (best first)
    results.sort(key=lambda x: x[2]["sisdr"], reverse=True)

    print("\n" + "=" * 80)
    print(f"{'Rank':<5} {'SI-SDR':>8} {'MSSTFT':>8} {'SFR':>8}  Checkpoint")
    print("-" * 80)
    for rank, (fname, _, m) in enumerate(results, 1):
        sfr_flag = " noise^" if m["sfr"] > 1.05 else ""
        print(f"  {rank:<3} {m['sisdr']:>8.4f} {m['msstft']:>8.4f} {m['sfr']:>8.4f}{sfr_flag}  {fname}")
    print("=" * 80)
    print(f"\nBest checkpoint: {results[0][0]}")
    print(f"  sisdr={results[0][2]['sisdr']:.4f}  msstft={results[0][2]['msstft']:.4f}  sfr={results[0][2]['sfr']:.4f}")

    if args.rename:
        print("\n[eval] Renaming checkpoints with accurate val_loss ...")
        for fname, ckpt_path, m in results:
            sisdr = m["sisdr"]
            # Build new name -- preserve step number if present
            step_match = re.search(r"step[=_](\d+)", fname)
            if step_match:
                step = step_match.group(1)
                new_name = f"step={step}-val_loss={sisdr:.4f}.ckpt"
            else:
                # Fallback: strip old loss suffix and append new
                base = re.sub(r"-val_loss=[^.]+", "", fname.replace(".ckpt", ""))
                new_name = f"{base}-val_loss={sisdr:.4f}.ckpt"
            new_path = os.path.join(os.path.dirname(ckpt_path), new_name)
            if new_path != ckpt_path:
                os.rename(ckpt_path, new_path)
                print(f"  {fname} -> {new_name}")
            else:
                print(f"  {fname} (unchanged)")
        print("[eval] Done.")


if __name__ == "__main__":
    main()

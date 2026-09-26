"""
evaluate.py -- Retroactive checkpoint evaluator for Apollo-mod.

Launched from tui.py as a TUI screen, or standalone:
    python evaluate.py --conf_dir configs/apollo_stfl2.yaml

Metrics computed per checkpoint:
    visqol  -- ViSQOL perceptual score (primary; requires: uv pip install "visqol-python[all]")
    hfnr     -- Spectral flatness ratio (HF noise canary)
    sisdr   -- SI-SDR (scale-invariant waveform quality)
    hfmae   -- HF band MAE (configurable; computed here for legacy checkpoints)

Ranking weights (evaluate.py only -- offline, VISQOL-anchored):
    visqol=0.60, hfnr=0.25, sisdr=0.15

Metrics already encoded in the checkpoint filename are read directly --
only missing metrics trigger inference. This means subsequent evaluate runs
on the same checkpoint set are fast (VISQOL only).

VISQOL scores are cached in <ckpt_dir>/.eval_cache.json so they survive
across sessions. hfmae, hfnr are always re-read from the filename
or recomputed fresh.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

import torch
import torchaudio
from omegaconf import OmegaConf

_CORE_DIR  = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_CORE_DIR)
sys.path.insert(0, _CORE_DIR)
from look2hear.system.audio_litmodule import (
    _hf_noise_ratio, _hf_band_mae_cpu,
    _get_visqol_api, _visqol_score,
)
import look2hear.models.apollo
import look2hear.losses

_SR = 44100

# ---------------------------------------------------------------------------
# Weights (evaluate.py offline ranking -- VISQOL-anchored)
# ---------------------------------------------------------------------------
_EVAL_WEIGHTS = {
    "visqol": 0.60,
    "hfnr":    0.25,
    "sisdr":  0.15,
    "hfmae":  0.00,  # kept for display; excluded from composite (use target_band_loss in config)
}

# Metrics where higher = better (will be inverted during normalization)
_HIGHER_IS_BETTER = {"visqol", "sisdr"}

# Patterns to parse metric values out of checkpoint filenames
_FILENAME_PATS = {
    "sisdr":  re.compile(r"sisdr=(-?[\d.]+)"),
    "hfnr":    re.compile(r"hfnr=(-?[\d.]+)"),
    "hfmae":  re.compile(r"hfmae=(-?[\d.]+)"),
}


# ---------------------------------------------------------------------------
# Legacy checkpoint filename migration
# ---------------------------------------------------------------------------

def _migrate_legacy_ckpt_names_eval(ckpt_dir: str) -> None:
    """Rename legacy-formatted .ckpt files in ckpt_dir to the current scheme."""
    _SKIP = {"last.ckpt", "interrupted.ckpt"}
    for fname in os.listdir(ckpt_dir):
        if not fname.endswith(".ckpt") or fname in _SKIP:
            continue
        stem = fname[:-5]
        new  = stem
        new  = new.replace("val_loss=",   "sisdr=")
        new  = new.replace("val_visqol=", "visqol=")
        new  = new.replace("val_sfr=",    "hfnr=")
        new  = new.replace("val_hfnr=",   "hfnr=")
        new  = re.sub(r"-val_sdr=[\d.]+", "", new)
        new  = re.sub(r"-(?<!si)sdr=[\d.]+", "", new)
        if new == stem:
            continue
        src = os.path.join(ckpt_dir, fname)
        dst = os.path.join(ckpt_dir, new + ".ckpt")
        if os.path.exists(dst):
            try:
                os.unlink(src)
            except Exception:
                pass
            continue
        try:
            os.rename(src, dst)
        except Exception as exc:
            print(f"[migrate] could not rename {fname}: {exc}")


# ---------------------------------------------------------------------------
# VISQOL
# ---------------------------------------------------------------------------

def _visqol_available() -> bool:
    return _get_visqol_api() is not None


def _compute_visqol(ref: torch.Tensor, deg: torch.Tensor, sr: int = _SR) -> Optional[float]:
    """Compute ViSQOL score. Returns None if visqol-python is unavailable.

    Delegates to the same _visqol_score() used during live training, so
    evaluate.py and the training-time metric always agree on how the
    score is computed (mono mix, resampled to 48kHz internally).
    """
    return _visqol_score(deg, ref, sr=sr)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_model(ckpt_path: str, feature_dim: int, sr: int, win: int, layer: int):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if not isinstance(ckpt, dict):
        raise ValueError(f"Checkpoint must be a dict: {ckpt_path}")
    state = ckpt.get("state_dict", ckpt)
    cleaned = {}
    for k, v in state.items():
        if k.startswith("audio_model."):
            cleaned[k[len("audio_model."):]] = v
        elif not any(k.startswith(p) for p in ("discriminator.", "metrics.", "optimizer")):
            cleaned[k] = v
    model = look2hear.models.apollo.Apollo(sr=sr, win=win, feature_dim=feature_dim, layer=layer)
    missing, _ = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"  [warn] {len(missing)} missing keys")
    return model.cuda().eval() if torch.cuda.is_available() else model.eval()


# ---------------------------------------------------------------------------
# Val chunk loading
# ---------------------------------------------------------------------------

def _load_val_chunks(val_chunk_dir: str, limit: Optional[int] = None) -> list:
    """
    Load val chunks with equal distribution across songs (floor/ceil budget split).
    When limit is set, distributes as evenly as possible: each song gets
    floor(limit/n_songs) chunks, with the remainder going to whichever songs
    have the most chunks available. Matches the litmodule _pick_chunks logic.
    """
    import math
    from collections import defaultdict

    lq_dir = os.path.join(val_chunk_dir, "LQ")
    hq_dir = os.path.join(val_chunk_dir, "HQ")
    if not os.path.isdir(lq_dir) or not os.path.isdir(hq_dir):
        raise RuntimeError(f"Val chunk dirs not found: {lq_dir}, {hq_dir}")

    all_files = sorted(f for f in os.listdir(lq_dir) if f.endswith(".wav"))

    # Group by song (strip trailing _NNN chunk index if present)
    by_song: dict = defaultdict(list)
    for f in all_files:
        stem = f[:-4]  # strip .wav
        parts = stem.rsplit("_", 1)
        song = parts[0] if len(parts) == 2 and parts[1].isdigit() else stem
        by_song[song].append(f)

    if limit is None or limit <= 0:
        selected = list(all_files)
    else:
        n_songs   = len(by_song)
        base      = limit // n_songs
        remainder = limit % n_songs
        # Give remainder slots to songs that have the most chunks available
        song_keys = sorted(by_song.keys(), key=lambda k: -len(by_song[k]))
        selected  = []
        for i, song in enumerate(song_keys):
            n = base + (1 if i < remainder else 0)
            selected.extend(by_song[song][:n])

    chunks = []
    for fname in sorted(selected):
        lq_path = os.path.join(lq_dir, fname)
        hq_path = os.path.join(hq_dir, fname)
        if os.path.exists(hq_path):
            chunks.append((lq_path, hq_path, fname))
    return chunks


def _normalize_pair(lq: torch.Tensor, hq: torch.Tensor):
    peak = max(lq.abs().amax().item(), hq.abs().amax().item())
    if peak > 0:
        lq = lq / peak
        hq = hq / peak
    return lq, hq


# ---------------------------------------------------------------------------
# Parse metrics from filename
# ---------------------------------------------------------------------------

def _parse_filename_metrics(fname: str) -> dict:
    clean = re.sub(r"^\[\d+\]-", "", fname)
    vals = {}
    for key, pat in _FILENAME_PATS.items():
        m = pat.search(clean)
        if m:
            v = float(m.group(1))
            # sisdr stored as negative in filename; convert back to positive
            if key == "sisdr":
                v = -v
            vals[key] = v
    return vals


# ---------------------------------------------------------------------------
# Per-checkpoint evaluation
# ---------------------------------------------------------------------------

def _eval_checkpoint(
    model,
    chunks: list,
    device: torch.device,
    run_visqol: bool,
    known_metrics: dict,
) -> dict:
    """
    Compute metrics for one checkpoint. Skips metrics already in known_metrics.
    Returns a dict of all metrics (known + newly computed).
    """
    need_inference = not all(k in known_metrics for k in ("hfnr", "hfmae"))

    metrics = dict(known_metrics)

    if not need_inference and not run_visqol:
        return metrics

    hfnr_sum    = 0.0
    hfmae_sum  = 0.0
    sisdr_sum  = 0.0
    visqol_sum = 0.0
    visqol_n   = 0
    n          = 0

    with torch.no_grad():
        for lq_path, hq_path, fname in chunks:
            try:
                lq, _ = torchaudio.load(lq_path)
                hq, _ = torchaudio.load(hq_path)
                lq, hq = _normalize_pair(lq, hq)
                if lq.shape[0] == 2:
                    lq = lq[0:1]
                    hq = hq[0:1]

                if need_inference or run_visqol:
                    lq_in  = lq.unsqueeze(0).to(device)
                    hq_ref = hq.unsqueeze(0).to(device)
                    out    = model(lq_in)
                    e_gpu  = out.squeeze(0)
                    r_gpu  = hq_ref.squeeze(0)
                    e_cpu  = e_gpu.float().cpu()
                    r_cpu  = r_gpu.float().cpu()

                if need_inference:
                    if "hfnr" not in known_metrics:
                        hfnr_sum    += _hf_noise_ratio(e_cpu, r_cpu)
                    if "hfmae" not in known_metrics:
                        hfmae_sum  += _hf_band_mae_cpu(e_cpu, r_cpu)
                    if "sisdr" not in known_metrics:
                        _sisdr_fn = look2hear.losses.MultiSrcNegSDR("sisdr", zero_mean=True)
                        sisdr_sum  += -float(_sisdr_fn(e_cpu.unsqueeze(0), r_cpu.unsqueeze(0)).mean())

                if run_visqol:
                    v = _compute_visqol(r_cpu, e_cpu)
                    if v is not None:
                        visqol_sum += v
                        visqol_n   += 1

                n += 1
            except Exception as ex:
                print(f"  [skip] {fname}: {ex}")

    if n == 0:
        return metrics

    if need_inference:
        if "hfnr"    not in known_metrics: metrics["hfnr"]    = hfnr_sum    / n
        if "hfmae"  not in known_metrics: metrics["hfmae"]  = hfmae_sum  / n
        if "sisdr"  not in known_metrics: metrics["sisdr"]  = sisdr_sum  / n

    if run_visqol and visqol_n > 0:
        metrics["visqol"] = visqol_sum / visqol_n

    metrics["n"] = n
    return metrics


# ---------------------------------------------------------------------------
# Composite ranking
# ---------------------------------------------------------------------------

def _rank_results(results: list[tuple]) -> list[tuple]:
    """
    results: list of (fname, ckpt_path, metrics_dict)
    Returns same list sorted best-first with 'rank' and 'composite' added to metrics.
    Weights: visqol=0.60, hfnr=0.25, sisdr=0.15
    Falls back gracefully when VISQOL is absent (renormalizes weights).
    """
    keys = list(_EVAL_WEIGHTS.keys())

    # Collect per-metric value ranges for min-max normalization
    ranges = {}
    for key in keys:
        present = [m[key] for _, _, m in results if key in m]
        if present:
            ranges[key] = (min(present), max(present))

    def composite(m: dict) -> float:
        score = 0.0
        total_w = 0.0
        for key, weight in _EVAL_WEIGHTS.items():
            if key not in m or key not in ranges:
                continue
            lo, hi = ranges[key]
            span = hi - lo
            n = 0.0 if span == 0 else (m[key] - lo) / span
            if key in _HIGHER_IS_BETTER:
                n = 1.0 - n  # invert: higher = better -> lower normalized = better
            score   += weight * n
            total_w += weight
        return score / total_w if total_w > 0 else float("inf")

    scored = []
    for fname, path, m in results:
        c = composite(m)
        m = dict(m, composite=c)
        scored.append((fname, path, m))

    scored.sort(key=lambda x: x[2]["composite"])
    for rank, (_, _, m) in enumerate(scored, 1):
        m["rank"] = rank
    return scored


# ---------------------------------------------------------------------------
# VISQOL cache
# ---------------------------------------------------------------------------

def _load_cache(ckpt_dir: str) -> dict:
    path = os.path.join(ckpt_dir, ".eval_cache.json")
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return {}


def _save_cache(ckpt_dir: str, cache: dict) -> None:
    path = os.path.join(ckpt_dir, ".eval_cache.json")
    try:
        Path(path).write_text(json.dumps(cache, indent=2))
    except Exception as ex:
        print(f"  [warn] cache write failed: {ex}")


# ---------------------------------------------------------------------------
# Main evaluation logic (called by TUI and __main__)
# ---------------------------------------------------------------------------

def run_evaluation(
    conf_dir: str,
    ckpt_dir: Optional[str] = None,
    limit: Optional[int] = None,
    run_visqol: bool = False,
    pattern: Optional[str] = None,
    print_fn=print,
) -> list[tuple]:
    """
    Core evaluation routine. Returns ranked results list.
    print_fn: callable for output (allows TUI to redirect).
    limit: total chunks to use across all songs; defaults to val_metric_samples
           from config (or 15 if absent). Equal distribution across songs.
    """
    cfg         = OmegaConf.load(conf_dir)
    exp_dir     = cfg.exp.dir
    exp_name    = cfg.exp.name
    feature_dim = int(cfg.model.feature_dim)
    sr          = int(cfg.model.sr)
    win         = int(cfg.model.win)
    layer       = int(cfg.model.layer)

    # Default limit to val_metric_samples from config so evaluate matches training
    if limit is None:
        limit = int(cfg.training.get("val_metric_samples", 15))

    # Auto-detect checkpoint dir
    if ckpt_dir is None:
        run_root = os.path.join(exp_dir, exp_name)
        if not os.path.isdir(run_root):
            raise RuntimeError(f"Run root not found: {run_root}")
        runs = sorted(
            (d for d in os.listdir(run_root) if os.path.isdir(os.path.join(run_root, d))),
            reverse=True,
        )
        if not runs:
            raise RuntimeError(f"No runs found in {run_root}")
        ckpt_dir = os.path.join(run_root, runs[0], "checkpoints")
        print_fn(f"[eval] Using run: {runs[0]}")

    if not os.path.isdir(ckpt_dir):
        raise RuntimeError(f"Checkpoint dir not found: {ckpt_dir}")

    # Val chunk dir
    chunk_root = os.path.join("chunks", exp_name, "val")
    if not os.path.isdir(chunk_root):
        raise RuntimeError(
            f"Val chunk dir not found: {chunk_root}\n"
            "Run train.py once to generate chunks."
        )

    print_fn(f"[eval] Loading val chunks from: {chunk_root}")
    chunks = _load_val_chunks(chunk_root, limit=limit)
    from collections import defaultdict as _dd
    _by_s: dict = _dd(int)
    for _, _, fn in chunks:
        stem = fn[:-4]; parts = stem.rsplit("_", 1)
        _by_s[parts[0] if len(parts) == 2 and parts[1].isdigit() else stem] += 1
    print_fn(f"[eval] {len(chunks)} chunks across {len(_by_s)} songs "
             f"({min(_by_s.values())}-{max(_by_s.values())} per song)")

    _migrate_legacy_ckpt_names_eval(ckpt_dir)
    ckpt_files = sorted(
        f for f in os.listdir(ckpt_dir)
        if f.endswith(".ckpt") and (pattern is None or pattern in f)
    )
    if not ckpt_files:
        raise RuntimeError(f"No checkpoints found in {ckpt_dir}")

    has_visqol = _visqol_available()
    if run_visqol and not has_visqol:
        print_fn('[eval] WARNING: visqol-python not installed -- skipping VISQOL. Run: uv pip install "visqol-python[all]"')
        run_visqol = False

    cache = _load_cache(ckpt_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []

    print_fn(f"[eval] Found {len(ckpt_files)} checkpoint(s)\n")

    for fname in ckpt_files:
        ckpt_path = os.path.join(ckpt_dir, fname)

        # Parse what's already in the filename
        known = _parse_filename_metrics(fname)

        # Pull VISQOL from cache if available
        cache_key = fname
        if cache_key in cache and "visqol" in cache[cache_key]:
            known["visqol"] = cache[cache_key]["visqol"]

        needs_model = (
            not all(k in known for k in ("hfnr", "hfmae"))
            or (run_visqol and "visqol" not in known)
        )

        parts = []
        for k in ("hfnr", "hfmae", "sisdr"):
            if k in known:
                parts.append(f"{k}={known[k]:.4f}")
        cached_str = "  ".join(parts) if parts else ""

        if needs_model:
            print_fn(f"  Evaluating: {fname} ...", end=" ", flush=True)
            try:
                model   = _load_model(ckpt_path, feature_dim, sr, win, layer)
                metrics = _eval_checkpoint(model, chunks, device, run_visqol, known)
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Cache VISQOL result
                if "visqol" in metrics and "visqol" not in known:
                    cache.setdefault(cache_key, {})["visqol"] = metrics["visqol"]
                    _save_cache(ckpt_dir, cache)

                new_parts = []
                for k in ("hfnr", "hfmae", "sisdr", "visqol"):
                    if k in metrics and k not in known:
                        new_parts.append(f"{k}={metrics[k]:.4f}")
                print_fn("  ".join(new_parts) if new_parts else "ok")
                results.append((fname, ckpt_path, metrics))
            except Exception as ex:
                print_fn(f"ERROR: {ex}")
        else:
            print_fn(f"  Cached:    {fname}  {cached_str}")
            results.append((fname, ckpt_path, known))

    if not results:
        return []

    return _rank_results(results)


# ---------------------------------------------------------------------------
# Table display
# ---------------------------------------------------------------------------

def print_results_table(results: list[tuple], print_fn=print) -> None:
    has_visqol = any("visqol" in m for _, _, m in results)
    cols = ["visqol"] if has_visqol else []
    cols += ["hfmae", "hfnr", "sisdr"]

    header = f"{'Rank':<5}"
    for c in cols:
        header += f" {c.upper():>8}"
    header += "  Checkpoint"
    print_fn("\n" + "=" * 90)
    print_fn(header)
    print_fn("-" * 90)

    for fname, _, m in results:
        rank = m.get("rank", "?")
        row  = f"  {rank:<3}"
        for c in cols:
            if c in m:
                val = m[c]
                hfnr_flag = " ^" if c == "hfnr" and val > 1.05 else "  "
                row += f" {val:>8.4f}{hfnr_flag}" if c == "hfnr" else f" {val:>8.4f}  "
            else:
                row += f" {'--':>8}  "
        row += f" {fname}"
        print_fn(row)

    print_fn("=" * 90)
    if results:
        best_fname, _, best_m = results[0]
        print_fn(f"\nBest: {best_fname}")
        summary = "  ".join(
            f"{k}={best_m[k]:.4f}" for k in ("visqol", "hfmae", "hfnr", "sisdr")
            if k in best_m
        )
        print_fn(f"  {summary}")


# ---------------------------------------------------------------------------
# TUI screen entry point
# ---------------------------------------------------------------------------

def screen_evaluate(state: dict, console, _pick, _run_with_live_output, ROOT: Path) -> None:
    """Entry point called from tui.py."""
    from pathlib import Path as _Path
    import yaml

    configs_dir = ROOT / "configs"
    configs = sorted(configs_dir.glob("*.yaml")) if configs_dir.exists() else []
    if not configs:
        console.clear()
        console.print("[red]No configs found in configs/[/]")
        console.input("Press Enter to return.")
        return

    last_cfg = state.get("evaluate", {}).get("last_config", "")
    start    = next((i for i, c in enumerate(configs) if c.name == last_cfg), 0)
    items    = [c.stem for c in configs]

    idx = _pick("Evaluate -- select config", items, hint="Enter=select  Esc=back", start=start)
    if idx is None:
        return

    cfg_path = configs[idx]
    state.setdefault("evaluate", {})["last_config"] = cfg_path.name

    # VISQOL option
    has_visqol = _visqol_available()
    visqol_label = "Run VISQOL (slow -- perceptual score)" if has_visqol else 'Run VISQOL  [not installed -- uv pip install "visqol-python[all]"]'
    mode_items = [
        "Fast  (read filename metrics + SDR only)",
        visqol_label,
    ]
    midx = _pick("Evaluate -- mode", mode_items, hint="Enter=select  Esc=back")
    if midx is None:
        return
    run_visqol = (midx == 1 and has_visqol)

    console.clear()
    from rich.panel import Panel
    console.print(Panel(
        f"[bold cyan]Evaluating: {cfg_path.stem}[/]\n"
        f"[dim]{'VISQOL enabled' if run_visqol else 'Fast mode (filename + SDR)'}[/]",
        border_style="cyan",
        padding=(0, 2),
    ))

    lines = []
    def _print(*args, **kwargs):
        msg = " ".join(str(a) for a in args)
        lines.append(msg)
        console.print(msg)

    try:
        results = run_evaluation(
            conf_dir   = str(cfg_path),
            run_visqol = run_visqol,
            print_fn   = _print,
        )
        if results:
            print_results_table(results, print_fn=_print)
        else:
            console.print("[yellow]No results.[/]")
    except Exception as ex:
        console.print(f"[red]Error: {ex}[/]")

    console.input("\n[dim]Press Enter to return to menu[/]")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Retroactively evaluate Apollo checkpoints")
    parser.add_argument("--conf_dir",  required=True, help="Path to yaml config")
    parser.add_argument("--ckpt_dir",  default=None,  help="Checkpoint folder (auto-detected if omitted)")
    parser.add_argument("--limit",     type=int, default=None, help="Total val chunks across songs (default: val_metric_samples from config)")
    parser.add_argument("--visqol",    action="store_true",   help="Run VISQOL (requires visqol-python)")
    parser.add_argument("--pattern",   default=None,          help="Only evaluate checkpoints matching this substring")
    args = parser.parse_args()

    try:
        results = run_evaluation(
            conf_dir   = args.conf_dir,
            ckpt_dir   = args.ckpt_dir,
            limit      = args.limit,
            run_visqol = args.visqol,
            pattern    = args.pattern,
        )
        if results:
            print_results_table(results)
        else:
            print("\n[eval] No results.")
            sys.exit(1)
    except RuntimeError as ex:
        print(f"[error] {ex}")
        sys.exit(1)


if __name__ == "__main__":
    main()

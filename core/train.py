# @claude last-modified: 2026-05-05T06:34:39Z
# @claude last-commit: feat: major update -- TUI, augmentation system, gradient checkpointing, optimization bootstrap

# Must be set before torch is imported -- CUDA allocator reads this env var at
# device init time. Setting it inside apply_optimizations() (post-import) has
# no effect because CUDA is already initialised by then.
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import json
from typing import Any, Dict, List, Optional, Tuple
from omegaconf import OmegaConf, open_dict
import argparse
import pytorch_lightning as pl
import torch
import hydra
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from omegaconf import DictConfig

# Optimisation bootstrap -- reads cfg.optimizations and applies everything
# in one place, before any model or trainer code runs.

def apply_optimizations(cfg: DictConfig) -> None:
    """Apply hardware/compiler optimisations declared in cfg.optimizations."""
    opt = cfg.get("optimizations", {})

    # TF32 disabled -- GAN spectral loss landscapes are sensitive to accumulated
    # numerical error. Keep highest precision regardless of config.
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # cuDNN benchmark (fastest conv algo for fixed input shapes)
    cudnn_benchmark = opt.get("cudnn_benchmark", True)
    torch.backends.cudnn.benchmark = cudnn_benchmark

    # CUDA allocator: expandable segments (reduces fragmentation)
    alloc_conf_parts = []
    if opt.get("expandable_segments", True):
        alloc_conf_parts.append("expandable_segments:True")
    if alloc_conf_parts:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ",".join(alloc_conf_parts)

    # Triton kernel cache (compiled kernels persist between runs)
    triton_cache = opt.get("triton_cache", True)
    if triton_cache:
        cache_dir = os.path.join(_REPO_ROOT, ".triton_cache")
        os.makedirs(cache_dir, exist_ok=True)
        os.environ["TRITON_CACHE_DIR"] = cache_dir

    # Misc env flags
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # RAM watchdog -- kills the process cleanly if system RAM crosses the threshold
    # before the OS does it violently. Threshold is a fraction of total RAM (default 90%).
    ram_limit = opt.get("ram_limit_fraction", 0.90)
    _start_ram_watchdog(ram_limit)


def _start_ram_watchdog(limit_fraction: float = 0.90) -> None:
    """Background thread that monitors system RAM and exits cleanly if usage
    crosses limit_fraction of total RAM. Prevents OS-level crashes from
    DataLoader workers or CUDA allocator runaway."""
    import threading
    try:
        import psutil
    except ImportError:
        print_only("[watchdog] psutil not installed -- RAM watchdog disabled. "
                   "Run: pip install psutil")
        return

    total = psutil.virtual_memory().total
    threshold = total * limit_fraction
    threshold_gb = threshold / (1024 ** 3)
    print_only(f"[watchdog] RAM watchdog active -- will exit cleanly above "
               f"{threshold_gb:.1f} GB ({limit_fraction*100:.0f}% of total)")

    def _watch():
        import time
        time.sleep(30)  # wait for DataLoader workers to finish spawning
        while True:
            used = psutil.virtual_memory().used
            if used >= threshold:
                used_gb = used / (1024 ** 3)
                print_only(f"\n[watchdog] SYSTEM RAM CRITICAL: {used_gb:.1f} GB used "
                            f"(threshold {threshold_gb:.1f} GB) -- exiting cleanly to protect OS")
                os._exit(1)
            time.sleep(2)

    t = threading.Thread(target=_watch, daemon=True)
    t.start()

import look2hear.system
import look2hear.datas
import look2hear.losses
import look2hear.models
import look2hear.models.apollo
from look2hear.utils import print_only
import warnings
warnings.filterwarnings("ignore")

def _migrate_legacy_ckpt_names(base_dir: str) -> None:
    """Rename legacy-formatted checkpoint files to the current naming scheme."""
    import re as _re
    _SKIP = {"last.ckpt", "interrupted.ckpt"}
    for root, _, files in os.walk(base_dir):
        for fname in files:
            if not fname.endswith(".ckpt") or fname in _SKIP:
                continue
            stem = fname[:-5]
            new  = stem
            new  = new.replace("val_loss=",   "sisdr=")
            new  = new.replace("val_visqol=", "visqol=")
            new  = new.replace("val_sfr=",    "hfnr=")
            new  = new.replace("val_hfnr=",   "hfnr=")
            new  = _re.sub(r"-(?<!si)sdr=[\d.]+", "", new)
            if new == stem:
                continue
            src = os.path.join(root, fname)
            dst = os.path.join(root, new + ".ckpt")
            if os.path.exists(dst):
                continue
            try:
                os.rename(src, dst)
            except Exception as exc:
                print_only(f"[migrate] could not rename {fname}: {exc}")


# Constants -- chunk size is read from cfg.datas.segment_sec at runtime in prepare_data()
_SR      = 44100
_OVERLAP = 0.5

# Mutable globals set by prepare_data() from the config
_CHUNK_SEC     = 3
_CHUNK_SAMPLES = int(_CHUNK_SEC * _SR)
_HOP_SAMPLES   = int(_CHUNK_SAMPLES * (1 - _OVERLAP))
_SUPPORTED_EXTS = {".wav", ".mp3", ".flac"}

# Models directory -- pretrained weights are looked up here automatically
_REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODELS_DIR = os.path.join(_REPO_ROOT, "models")
_CACHE_DIR  = os.path.join(_REPO_ROOT, "usr", "cache")

# Pretrained model filenames to search for (base -> universal).
# Set download URLs here once you have them; None = skip auto-download.
_PRETRAINED_MODELS = {
    "apollo.ckpt":           None,  # base model   (feature_dim=256)
    "apollo_uni.ckpt":       None,  # universal     (feature_dim=384)
    "apollo_model.ckpt":     None,  # legacy name   (feature_dim=256)
    "apollo_model_uni.ckpt": None,  # legacy name   (feature_dim=384)
    "pytorch_model.bin":     None,  # HF bin format (feature_dim=256)
}

# Data preparation -- runs before training, skips gracefully if already done

def _load_wav_stereo(path: str, frame_offset: int = 0):
    """Load audio (WAV, MP3, FLAC), resample to _SR if needed, force stereo float32.

    frame_offset: skip this many samples at the start of the file.
    For MP3s this is passed directly to torchaudio.load so the decoder
    seeks past the offset rather than decoding and discarding it.
    """
    import torchaudio
    wav, sr = torchaudio.load(path, frame_offset=frame_offset)
    wav = wav.float()
    if sr != _SR:
        wav = torchaudio.functional.resample(wav, sr, _SR)
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    elif wav.shape[0] > 2:
        wav = wav[:2]
    return wav


def _align_pair(lq: "torch.Tensor", hq: "torch.Tensor", stem: str, fixed_delay: int = 0) -> tuple:
    """Trim a fixed sample offset from LQ (positive) or HQ (negative) to compensate for encoder delay."""
    if fixed_delay > 0:
        lq = lq[:, fixed_delay:]
        print_only(f"[align] {stem}: trimmed {fixed_delay} samples from LQ")
    elif fixed_delay < 0:
        hq = hq[:, abs(fixed_delay):]
        print_only(f"[align] {stem}: trimmed {abs(fixed_delay)} samples from HQ")
    min_len = min(lq.shape[-1], hq.shape[-1])
    return lq[:, :min_len], hq[:, :min_len]

def _save_wav_f32(tensor, path: str, sr: int = None) -> None:
    """Save a tensor as 32-bit float WAV. No quantisation — full dynamic range preserved."""
    import torchaudio
    torchaudio.save(path, tensor.float().cpu(), sr or _SR, encoding="PCM_F", bits_per_sample=32)


def _save_chunk_16bit(tensor, path: str):
    """Kept for reference. All chunk writes now use _save_wav_f32."""
    _save_wav_f32(tensor, path)


def _save_chunks_batch(chunks: list, paths: list) -> None:
    """Write multiple chunks as 32-bit float WAV."""
    for tensor, path in zip(chunks, paths):
        _save_wav_f32(tensor, path)



def _slice_and_save(
    lq_wav, hq_wav, stem: str, lq_out: str, hq_out: str,
    cached_aug_fn=None,
    progress_cb=None,
) -> list:
    """Slice a pair into overlapping chunks, write each immediately to avoid
    accumulating all chunks in memory (critical for long source files).

    progress_cb: optional callable(chunks_done, chunks_total) called after each chunk write.
    """
    min_len = min(lq_wav.shape[-1], hq_wav.shape[-1])
    lq_wav  = lq_wav[:, :min_len]
    hq_wav  = hq_wav[:, :min_len]

    song_peak = max(lq_wav.abs().max().item(), hq_wav.abs().max().item())
    if song_peak > 1.0:
        lq_wav = lq_wav / song_peak
        hq_wav = hq_wav / song_peak

    def _write_chunk(t, path):
        _save_wav_f32(t, path)

    # Pre-compute total chunk count for progress reporting without storing chunks
    total_chunks = max(0, (min_len - _CHUNK_SAMPLES) // _HOP_SAMPLES + 1)

    saved = []
    start = 0
    idx   = 0
    while start + _CHUNK_SAMPLES <= min_len:
        lq_c = lq_wav[:, start:start + _CHUNK_SAMPLES]
        hq_c = hq_wav[:, start:start + _CHUNK_SAMPLES]
        fname = f"{stem}_{idx:04d}.wav"

        if cached_aug_fn is not None:
            lq_out_c, hq_out_c = cached_aug_fn(lq_c.clone(), hq_c.clone())
            _write_chunk(lq_out_c, os.path.join(lq_out, fname))
            _write_chunk(hq_out_c, os.path.join(hq_out, fname))
        else:
            _write_chunk(lq_c, os.path.join(lq_out, fname))
            _write_chunk(hq_c, os.path.join(hq_out, fname))

        saved.append(fname)
        if progress_cb:
            progress_cb(idx + 1, total_chunks)

        start += _HOP_SAMPLES
        idx   += 1

    return saved

def _has_wav_pairs(lq_dir: str, hq_dir: str) -> bool:
    """Return True if both dirs exist and share at least one matching stem."""
    if not (os.path.isdir(lq_dir) and os.path.isdir(hq_dir)):
        return False
    lq_stems = {os.path.splitext(f)[0] for f in os.listdir(lq_dir) if os.path.splitext(f)[1].lower() in _SUPPORTED_EXTS}
    hq_stems = {os.path.splitext(f)[0] for f in os.listdir(hq_dir) if os.path.splitext(f)[1].lower() in _SUPPORTED_EXTS}
    return bool(lq_stems & hq_stems)

def _count_wav_pairs(lq_dir: str, hq_dir: str) -> int:
    if not (os.path.isdir(lq_dir) and os.path.isdir(hq_dir)):
        return 0
    lq_stems = {os.path.splitext(f)[0] for f in os.listdir(lq_dir) if os.path.splitext(f)[1].lower() in _SUPPORTED_EXTS}
    hq_stems = {os.path.splitext(f)[0] for f in os.listdir(hq_dir) if os.path.splitext(f)[1].lower() in _SUPPORTED_EXTS}
    return len(lq_stems & hq_stems)

def _normalize_data_dir(src_root: str, split_name: str) -> bool:
    """
    Accepts two input layouts and normalizes both into src_root/LQ + src_root/HQ.
    Returns True if src_root is ready (has matched LQ/HQ content), False otherwise.

    Layout A -- subfolder pairs (existing):
        src_root/song1_LQ/   song1_HQ/
        src_root/song2_LQ/   song2_HQ/
        -> moves audio files from each _LQ/_HQ subdir into src_root/LQ/ + src_root/HQ/

    Layout B -- flat postfix files (new shortcut):
        src_root/song1_LQ.wav   song1_HQ.wav
        src_root/song2_LQ.flac  song2_HQ.flac
        -> moves files into src_root/LQ/ + src_root/HQ/, stripping the _LQ/_HQ suffix

    Layout C -- already normalized (LQ/ and HQ/ subdirs exist):
        src_root/LQ/   src_root/HQ/
        -> nothing to do

    After normalization src_root always looks like:
        src_root/LQ/<stem>.wav ...
        src_root/HQ/<stem>.wav ...
    """
    import shutil

    if not os.path.isdir(src_root):
        return False

    lq_dir = os.path.join(src_root, "LQ")
    hq_dir = os.path.join(src_root, "HQ")

    # Layout C -- already normalized, nothing to do
    if os.path.isdir(lq_dir) and os.path.isdir(hq_dir):
        lq_files = [f for f in os.listdir(lq_dir) if os.path.splitext(f)[1].lower() in _SUPPORTED_EXTS]
        hq_files = [f for f in os.listdir(hq_dir) if os.path.splitext(f)[1].lower() in _SUPPORTED_EXTS]
        if lq_files and hq_files:
            print_only(f"[data/{split_name}] LQ/ + HQ/ already present -- skipping normalization.")
            return True

    entries = os.listdir(src_root)

    # Layout A: _LQ / _HQ subdirectories
    subdirs  = {e for e in entries if os.path.isdir(os.path.join(src_root, e))}
    lq_dirs  = {d[:-3]: d for d in subdirs if d.upper().endswith("_LQ")}
    hq_dirs  = {d[:-3]: d for d in subdirs if d.upper().endswith("_HQ")}
    dir_pairs = sorted(set(lq_dirs) & set(hq_dirs))

    # Layout B: _LQ / _HQ postfix files
    files    = {e for e in entries if os.path.isfile(os.path.join(src_root, e))
                and os.path.splitext(e)[1].lower() in _SUPPORTED_EXTS}
    lq_files_flat = {}
    hq_files_flat = {}
    for fname in files:
        stem, ext = os.path.splitext(fname)
        if stem.upper().endswith("_LQ"):
            lq_files_flat[stem[:-3]] = fname   # strip _LQ to get base stem
        elif stem.upper().endswith("_HQ"):
            hq_files_flat[stem[:-3]] = fname
    file_pairs = sorted(set(lq_files_flat) & set(hq_files_flat))

    if not dir_pairs and not file_pairs:
        print_only(f"[data/{split_name}] WARNING: no _LQ/_HQ pairs found in {src_root}")
        return False

    os.makedirs(lq_dir, exist_ok=True)
    os.makedirs(hq_dir, exist_ok=True)

    # Move Layout A: song_LQ/ -> LQ/<stem_from_dir>.wav (files keep their own names)
    for stem in dir_pairs:
        src_lq = os.path.join(src_root, lq_dirs[stem])
        src_hq = os.path.join(src_root, hq_dirs[stem])
        for fname in sorted(os.listdir(src_lq)):
            if os.path.splitext(fname)[1].lower() in _SUPPORTED_EXTS:
                # Prefix with song stem to avoid collisions between songs
                dest_name = f"{stem}_{fname}"
                shutil.move(os.path.join(src_lq, fname), os.path.join(lq_dir, dest_name))
        for fname in sorted(os.listdir(src_hq)):
            if os.path.splitext(fname)[1].lower() in _SUPPORTED_EXTS:
                dest_name = f"{stem}_{fname}"
                shutil.move(os.path.join(src_hq, fname), os.path.join(hq_dir, dest_name))
        # Remove now-empty source dirs
        try:
            os.rmdir(src_lq)
            os.rmdir(src_hq)
        except OSError:
            pass  # not empty (e.g. had subdirs or other files) -- leave it
        print_only(f"[data/{split_name}]   normalized dir pair: {stem}")

    # Move Layout B: song_LQ.wav -> LQ/song.wav  (strip postfix from filename)
    for stem in file_pairs:
        lq_fname = lq_files_flat[stem]
        hq_fname = hq_files_flat[stem]
        ext_lq = os.path.splitext(lq_fname)[1]
        ext_hq = os.path.splitext(hq_fname)[1]
        shutil.move(os.path.join(src_root, lq_fname), os.path.join(lq_dir, f"{stem}{ext_lq}"))
        shutil.move(os.path.join(src_root, hq_fname), os.path.join(hq_dir, f"{stem}{ext_hq}"))
        print_only(f"[data/{split_name}]   normalized file pair: {stem}")

    total_pairs = len(dir_pairs) + len(file_pairs)
    print_only(f"[data/{split_name}] Normalized {total_pairs} pairs into LQ/ + HQ/")
    return True

def _build_cached_aug_fn(cfg: "DictConfig"):
    """
    Build a callable (lq, hq) -> (lq_aug, hq_aug) from the cached_augmentation
    block in the config, using fraction-based selection instead of per-sample prob.
    Returns None if cached augmentation is disabled or not configured.
    """
    import random as _random
    cached_cfg = getattr(cfg.datas, "augmentation", None)
    if cached_cfg is None:
        return None
    cached_cfg = getattr(cached_cfg, "cached", None)
    if cached_cfg is None or not getattr(cached_cfg, "enabled", False):
        return None

    from paired_datamodule import (
        augment_pair, AugmentationCfg, GainAugCfg, SimpleAugCfg,
        PitchShiftAugCfg, NoiseAugCfg, Mp3AugCfg,
    )

    def _frac(block, key, default=0.0):
        try:
            return float(block[key])
        except (KeyError, TypeError):
            return default

    def _bool(block, key, default=False):
        try:
            return bool(block[key])
        except (KeyError, TypeError):
            return default

    g   = getattr(cached_cfg, "gain",            {})
    pol = getattr(cached_cfg, "polarity",         {})
    ps  = getattr(cached_cfg, "pitch_shift",      {})
    ns  = getattr(cached_cfg, "noise",            {})
    mp3 = getattr(cached_cfg, "mp3_degradation",  {})
    mc  = getattr(cached_cfg, "stereo_alternation",     {})

    aug_cfg = AugmentationCfg(
        enabled=True,
        gain=GainAugCfg(
            enabled=_bool(g,   "enabled", False),
            prob=   _frac(g,   "fraction", 0.0),
            db_max= _frac(g,   "db_max",   1.5),
        ),
        polarity=SimpleAugCfg(
            enabled=_bool(pol, "enabled", False),
            prob=   _frac(pol, "fraction", 0.0),
        ),
        pitch_shift=PitchShiftAugCfg(
            enabled=       _bool(ps, "enabled",       False),
            prob=          _frac(ps, "fraction",       0.0),
            semitones_max= _frac(ps, "semitones_max",  1.5),
        ),
        noise=NoiseAugCfg(
            enabled=_bool(ns, "enabled", False),
            prob=   _frac(ns, "fraction", 0.0),
            sigma=  _frac(ns, "sigma",    0.002),
        ),
        mp3_degradation=Mp3AugCfg(
            enabled= _bool(mp3, "enabled",  False),
            prob=    _frac(mp3, "fraction",  0.5),
            kbps_min=int(_frac(mp3, "kbps_min", 64)),
            kbps_max=int(_frac(mp3, "kbps_max", 256)),
        ),
        stereo_alternation=SimpleAugCfg(
            enabled=_bool(mc, "enabled", False),
            prob=   _frac(mc, "fraction", 1.0),
        ),
    )

    sr = int(getattr(cfg.datas, "sr", 44100))

    def _apply(lq, hq):
        # Single roll per chunk: augment_pair's own prob check decides whether
        # mp3_degradation fires at all, and if so draws one random bitrate from
        # [kbps_min, kbps_max]. No duplication, no stratified variants -- variety
        # across the song comes naturally from each chunk rolling independently.
        return augment_pair(lq, hq, aug_cfg, sr=sr)

    return _apply

_CHUNK_CACHE_DIR = os.path.join(_REPO_ROOT, "usr", "cache", "chunks")


def _source_md5s(src_root: str) -> str:
    """Return a stable hex digest over all LQ + HQ source files in src_root (sorted by name)."""
    import hashlib
    h = hashlib.md5()
    lq_dir = os.path.join(src_root, "LQ")
    hq_dir = os.path.join(src_root, "HQ")
    for d in (lq_dir, hq_dir):
        if not os.path.isdir(d):
            continue
        for fname in sorted(os.listdir(d)):
            p = os.path.join(d, fname)
            if not os.path.isfile(p):
                continue
            h.update(fname.encode())
            fh = hashlib.md5()
            with open(p, "rb") as f:
                for block in iter(lambda: f.read(1 << 20), b""):
                    fh.update(block)
            h.update(fh.hexdigest().encode())
    return h.hexdigest()


def _chunk_cache_key(src_root: str, fixed_delay, aug_cfg, extra: str = "") -> str:
    """Stable cache key: source file md5s + chunk params + aug params."""
    import hashlib, json as _json
    params = {
        "segment_sec": _CHUNK_SEC,
        "overlap": _OVERLAP,
        "fixed_delay": str(fixed_delay),
        "aug": aug_cfg if aug_cfg is None else _json.dumps(
            OmegaConf.to_container(aug_cfg, resolve=True), sort_keys=True
        ),
        "extra": extra,
    }
    h = hashlib.md5()
    h.update(_source_md5s(src_root).encode())
    h.update(_json.dumps(params, sort_keys=True).encode())
    return h.hexdigest()[:16]


def _chunk_cache_lookup(key: str, split: str) -> str | None:
    """Return the cache chunk dir for this key+split if it exists and has WAV files, else None."""
    d = os.path.join(_CHUNK_CACHE_DIR, key, split, "LQ")
    if os.path.isdir(d) and any(f.endswith(".wav") for f in os.listdir(d)):
        return os.path.join(_CHUNK_CACHE_DIR, key, split)
    return None



def _chunk_split(src_root: str, dst_root: str, split_name: str, cached_aug_fn=None, fixed_delay: int = None) -> int:
    """
    Normalize src_root into LQ/ + HQ/ layout (if not already), then chunk all
    matched pairs into dst_root/LQ and dst_root/HQ.
    Returns the number of chunk pairs written. Skips if already chunked.

    Accepted src_root layouts (all auto-detected and normalized):
        A)  src_root/<song>_LQ/  <song>_HQ/     -- subdirectory pairs
        B)  src_root/<song>_LQ.wav  <song>_HQ.wav  -- flat postfix files
        C)  src_root/LQ/  src_root/HQ/           -- already normalized
    """
    lq_out = os.path.join(dst_root, "LQ")
    hq_out = os.path.join(dst_root, "HQ")
    manifest_path = os.path.join(dst_root, ".manifest.json")

    def _manifest_params():
        return {"segment_sec": _CHUNK_SEC, "overlap": _OVERLAP, "fixed_delay": str(fixed_delay)}

    def _manifest_valid():
        if not os.path.isfile(manifest_path):
            return False
        try:
            import json as _json
            with open(manifest_path, "r") as _f:
                return _json.load(_f) == _manifest_params()
        except Exception:
            return False

    # Skip if already chunked with matching parameters
    if os.path.isdir(lq_out) and any(f.endswith(".wav") for f in os.listdir(lq_out)):
        if _manifest_valid():
            n = sum(1 for f in os.listdir(lq_out) if f.endswith(".wav"))
            print_only(f"[data/{split_name}] Already chunked ({n} pairs) -- skipping.")
            return n
        else:
            print_only(f"[data/{split_name}] Chunk parameters changed -- re-chunking.")

    # Normalize the source layout first
    if not _normalize_data_dir(src_root, split_name):
        return 0

    lq_src = os.path.join(src_root, "LQ")
    hq_src = os.path.join(src_root, "HQ")

    # Collect matched pairs by stem
    lq_files = {
        os.path.splitext(f)[0]: f
        for f in os.listdir(lq_src)
        if os.path.splitext(f)[1].lower() in _SUPPORTED_EXTS
    }
    hq_files = {
        os.path.splitext(f)[0]: f
        for f in os.listdir(hq_src)
        if os.path.splitext(f)[1].lower() in _SUPPORTED_EXTS
    }
    matched   = sorted(set(lq_files) & set(hq_files))
    unmatched = (set(lq_files) | set(hq_files)) - set(matched)
    if unmatched:
        print_only(f"[data/{split_name}] WARNING: unmatched files (skipping): {sorted(unmatched)}")
    if not matched:
        print_only(f"[data/{split_name}] ERROR: no matched LQ/HQ pairs after normalization")
        return 0

    os.makedirs(lq_out, exist_ok=True)
    os.makedirs(hq_out, exist_ok=True)

    sep = "=" * 58
    print_only(f"\n[data/{split_name}] {sep}")
    print_only(f"[data/{split_name}] Chunking {len(matched)} pairs -> {dst_root}")
    print_only(f"[data/{split_name}] {sep}\n")

    import threading, concurrent.futures as _cf

    lq_offset = fixed_delay if (fixed_delay is not None and fixed_delay > 0) else 0
    hq_offset = (-fixed_delay) if (fixed_delay is not None and fixed_delay < 0) else 0
    print_lock = threading.Lock()

    # --- Phase 1: WAV conversion to cache ---
    # Non-WAV files (and WAVs with an alignment trim) are decoded to
    # usr/cache/<md5>.wav -- keyed on the MD5 of the original file bytes so
    # re-runs with the same source skip re-conversion instantly. Originals are
    # never modified. Alignment trim is baked in during conversion so chunking
    # never needs to know about offsets. FFmpeg subprocesses are fully parallel.

    def _has_ffmpeg() -> bool:
        try:
            import ffmpeg
            return True
        except ImportError:
            return False

    def _file_md5(path: str) -> str:
        import hashlib
        h = hashlib.md5()
        with open(path, "rb") as _f:
            for _block in iter(lambda: _f.read(1 << 20), b""):
                h.update(_block)
        return h.hexdigest()

    def _cached_wav_path(src: str, trim_samples: int) -> str:
        """Return the usr/cache path for this source + trim combo."""
        md5 = _file_md5(src)
        key = md5 if trim_samples == 0 else f"{md5}_t{trim_samples}"
        os.makedirs(_CACHE_DIR, exist_ok=True)
        return os.path.join(_CACHE_DIR, f"{key}.wav")

    def _to_cached_wav_ffmpeg(src: str, trim_samples: int) -> str:
        """Decode src to usr/cache via ffmpeg. Returns cache path."""
        import ffmpeg, tempfile
        dst = _cached_wav_path(src, trim_samples)
        if os.path.isfile(dst):
            return dst
        fd, tmp_path = tempfile.mkstemp(suffix=".wav", dir=_CACHE_DIR)
        os.close(fd)
        try:
            stream = ffmpeg.input(src)
            if trim_samples > 0:
                trim_sec = trim_samples / _SR
                stream = stream.audio.filter("atrim", start=trim_sec).filter("asetpts", "PTS-STARTPTS")
            (
                stream
                .output(tmp_path, format="wav", acodec="pcm_f32le", ar=_SR, ac=2)
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
        except ffmpeg.Error as e:
            os.unlink(tmp_path)
            stderr = e.stderr.decode(errors="replace") if e.stderr else ""
            raise RuntimeError(f"[ffmpeg] Failed to convert {src}:\n{stderr}") from e
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise
        os.replace(tmp_path, dst)
        print_only(f"[cache] Wrote {os.path.basename(dst)}  ({os.path.basename(src)})")
        return dst

    def _to_cached_wav_torchaudio(src: str, trim_samples: int) -> str:
        """Fallback: decode via torchaudio (GIL-bound for MP3)."""
        import torchaudio
        dst = _cached_wav_path(src, trim_samples)
        if os.path.isfile(dst):
            return dst
        wav, sr = torchaudio.load(src, frame_offset=trim_samples)
        wav = wav.float()
        if sr != _SR:
            wav = torchaudio.functional.resample(wav, sr, _SR)
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)
        elif wav.shape[0] > 2:
            wav = wav[:2]
        torchaudio.save(dst, wav, _SR, encoding="PCM_F", bits_per_sample=32)
        print_only(f"[cache] Wrote {os.path.basename(dst)}  ({os.path.basename(src)})")
        return dst

    use_ffmpeg = _has_ffmpeg()
    # Files needing cache conversion: non-WAV always; WAV with alignment trim also
    # needs a cached trimmed version. WAVs with no trim are used directly.
    needs_cache = {
        s for s in matched
        if os.path.splitext(lq_files[s])[1].lower() != ".wav"
        or os.path.splitext(hq_files[s])[1].lower() != ".wav"
        or lq_offset > 0
        or hq_offset > 0
    }

    wav_paths: dict[str, tuple[str, str]] = {}

    if needs_cache:
        to_wav = _to_cached_wav_ffmpeg if use_ffmpeg else _to_cached_wav_torchaudio
        method = "FFmpeg" if use_ffmpeg else "torchaudio (install ffmpeg-python for faster conversion)"
        print_only(f"[data/{split_name}] Caching {len(needs_cache)} source(s) to WAV via {method}...")
        done_conv = [0]

        def _cache_stem(stem):
            lq_p = os.path.join(lq_src, lq_files[stem])
            hq_p = os.path.join(hq_src, hq_files[stem])
            lq_ext = os.path.splitext(lq_p)[1].lower()
            hq_ext = os.path.splitext(hq_p)[1].lower()
            new_lq = to_wav(lq_p, lq_offset) if (lq_ext != ".wav" or lq_offset > 0) else lq_p
            new_hq = to_wav(hq_p, hq_offset) if (hq_ext != ".wav" or hq_offset > 0) else hq_p
            with print_lock:
                done_conv[0] += 1
                print_only(f"[data/{split_name}]   Cached {done_conv[0]}/{len(needs_cache)}: {stem}")
            return stem, new_lq, new_hq

        n_conv = min(len(needs_cache), 2)  # FFmpeg is disk+CPU heavy; >2 concurrent thrashes the system
        with _cf.ThreadPoolExecutor(max_workers=n_conv) as ex:
            for stem, lq_p, hq_p in ex.map(_cache_stem, sorted(needs_cache)):
                wav_paths[stem] = (lq_p, hq_p)

        for stem in matched:
            if stem not in wav_paths:
                wav_paths[stem] = (
                    os.path.join(lq_src, lq_files[stem]),
                    os.path.join(hq_src, hq_files[stem]),
                )
        print_only(f"[data/{split_name}] Cache pass done.\n")
    else:
        wav_paths = {
            s: (os.path.join(lq_src, lq_files[s]), os.path.join(hq_src, hq_files[s]))
            for s in matched
        }

    # --- Phase 2: Parallel WAV chunking ---
    # Sources are now guaranteed WAV. torchaudio WAV load is near-zero-copy
    # (GIL released during I/O). Numpy slicing and wave.write are also GIL-free.
    n_workers = min(len(matched), 2)  # cap at 2 to avoid OOM on long source files
    total = 0
    chunks_done = [0]
    songs_done = [0]
    n_songs = len(matched)
    _REPORT_EVERY = 50  # print a progress line every N chunks across all songs

    def _chunk_stem(stem):
        lq_wav = _load_wav_stereo(wav_paths[stem][0])
        hq_wav = _load_wav_stereo(wav_paths[stem][1])

        def _progress(done, total_c):
            with print_lock:
                chunks_done[0] += 1
                if chunks_done[0] % _REPORT_EVERY == 0:
                    print_only(
                        f"[data/{split_name}] {songs_done[0]}/{n_songs} songs  "
                        f"{chunks_done[0]} chunks"
                    )

        saved = _slice_and_save(lq_wav, hq_wav, stem, lq_out, hq_out,
                                cached_aug_fn=cached_aug_fn,
                                progress_cb=_progress)
        with print_lock:
            songs_done[0] += 1
            print_only(f"[data/{split_name}]   {stem}: done ({len(saved)} chunks)  [{songs_done[0]}/{n_songs}]")
        return len(saved)

    with _cf.ThreadPoolExecutor(max_workers=n_workers) as pool:
        for n in pool.map(_chunk_stem, matched):
            total += n

    print_only(f"[data/{split_name}] Done -- {total} chunk pairs -> {dst_root}\n")
    import json as _json
    with open(manifest_path, "w") as _f:
        _json.dump(_manifest_params(), _f, indent=2)
    return total

def _extract_val_clips(src_root: str, dst_root: str, clip_sec: float = 10.0, fixed_delay: int = None) -> None:
    """
    Extract two content-rich 10s clips per LQ/HQ pair in src_root, following ViSQOL
    input guidelines:
      - ~8-10s per clip (clip_sec, default 10)
      - Selected by highest-RMS energy windows (not silence, not intro/outro)
      - Mono-mixed for RMS selection only; saved as stereo at model SR
      - Two non-overlapping clips per song, with at least clip_sec gap between them
    Files written: {stem}_clip0.wav, {stem}_clip1.wav in dst_root/LQ/ and dst_root/HQ/.
    """
    import hashlib
    import torch
    import torchaudio

    lq_src = os.path.join(src_root, "LQ")
    hq_src = os.path.join(src_root, "HQ")
    if not os.path.isdir(lq_src) or not os.path.isdir(hq_src):
        return

    os.makedirs(os.path.join(dst_root, "LQ"), exist_ok=True)
    os.makedirs(os.path.join(dst_root, "HQ"), exist_ok=True)

    def _file_md5(path):
        h = hashlib.md5()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        return h.hexdigest()

    def _cached_wav(src):
        md5 = _file_md5(src)
        os.makedirs(_CACHE_DIR, exist_ok=True)
        dst = os.path.join(_CACHE_DIR, f"{md5}.wav")
        if os.path.isfile(dst):
            return dst
        try:
            import ffmpeg, tempfile as _tmp
            fd, tmp = _tmp.mkstemp(suffix=".wav", dir=_CACHE_DIR)
            os.close(fd)
            (ffmpeg.input(src).output(tmp, format="wav", acodec="pcm_f32le", ar=_SR, ac=2)
             .overwrite_output().run(capture_stdout=True, capture_stderr=True))
            os.replace(tmp, dst)
        except Exception:
            wav, sr = torchaudio.load(src)
            wav = wav.float()
            if sr != _SR:
                wav = torchaudio.functional.resample(wav, sr, _SR)
            if wav.shape[0] == 1: wav = wav.repeat(2, 1)
            elif wav.shape[0] > 2: wav = wav[:2]
            torchaudio.save(dst, wav, _SR, encoding="PCM_F", bits_per_sample=32)
        return dst

    def _pick_two_rms_clips(wav: "torch.Tensor", clip_samples: int, sr: int,
                             margin_sec: float = 5.0) -> "list[int]":
        """
        Return start-sample offsets for two non-overlapping highest-RMS windows.
        Skips the first and last margin_sec of the file. Clips must be separated
        by at least clip_samples samples.
        mono wav: shape (T,)
        """
        margin = int(margin_sec * sr)
        n = wav.shape[-1]
        search_start = margin
        search_end   = n - margin - clip_samples
        if search_end <= search_start:
            # File too short for margin -- fall back to start/middle
            mid = max(0, n // 2 - clip_samples // 2)
            return [0, min(mid, max(0, n - clip_samples))]

        # Scan with hop = clip_samples // 4 for reasonable resolution
        hop = max(1, clip_samples // 4)
        offsets = list(range(search_start, search_end + 1, hop))
        if not offsets:
            offsets = [search_start]

        rms_scores = []
        for s in offsets:
            window = wav[s: s + clip_samples]
            rms = float(window.pow(2).mean().sqrt())
            rms_scores.append((rms, s))
        rms_scores.sort(key=lambda x: -x[0])

        # Best clip
        best_rms, best_start = rms_scores[0]
        # Second-best non-overlapping clip (gap >= clip_samples)
        second_start = None
        for rms, s in rms_scores[1:]:
            if abs(s - best_start) >= clip_samples:
                second_start = s
                break
        if second_start is None:
            # Fallback: place second clip as far from first as possible
            candidate = search_end if best_start < (search_start + search_end) // 2 else search_start
            second_start = max(search_start, min(search_end, candidate))

        return sorted([best_start, second_start])

    clip_samples = int(clip_sec * _SR)

    def _find_audio_pairs(lq_dir, hq_dir):
        lq_map = {os.path.splitext(f)[0]: os.path.join(lq_dir, f)
                  for f in os.listdir(lq_dir)
                  if os.path.splitext(f)[1].lower() in _SUPPORTED_EXTS}
        hq_map = {os.path.splitext(f)[0]: os.path.join(hq_dir, f)
                  for f in os.listdir(hq_dir)
                  if os.path.splitext(f)[1].lower() in _SUPPORTED_EXTS}
        matched = sorted(set(lq_map) & set(hq_map))
        return [(lq_map[s], hq_map[s]) for s in matched]

    pairs = _find_audio_pairs(lq_src, hq_src)

    print_only(f"\n[data/val] ==========================================================")
    print_only(f"[data/val] Extracting 2x{clip_sec:.0f}s clips (highest-RMS, ViSQOL-compliant)")
    print_only(f"[data/val] from {len(pairs)} val pair(s)")
    print_only(f"[data/val] ==========================================================\n")
    n = 0
    for song_idx, (lq_path, hq_path) in enumerate(pairs):
        lq_wav_path = _cached_wav(lq_path)
        hq_wav_path = _cached_wav(hq_path)

        lq_wav, _ = torchaudio.load(lq_wav_path)  # already at _SR, stereo
        hq_wav, _ = torchaudio.load(hq_wav_path)

        min_frames = min(lq_wav.shape[-1], hq_wav.shape[-1])

        # Mono mix HQ for RMS selection (reference should be clean per ViSQOL spec)
        hq_mono = hq_wav[:, :min_frames].mean(dim=0)  # (T,)
        starts = _pick_two_rms_clips(hq_mono, clip_samples, _SR)

        stem = os.path.splitext(os.path.basename(lq_path))[0]
        for clip_idx, start in enumerate(starts):
            lq_start = max(0, start - fixed_delay) if fixed_delay and fixed_delay > 0 else start
            hq_start = max(0, start + fixed_delay) if fixed_delay and fixed_delay < 0 else start

            lq_clip = lq_wav[:, lq_start: lq_start + clip_samples]
            hq_clip = hq_wav[:, hq_start: hq_start + clip_samples]

            # Pad to exact clip_samples if the file was shorter
            if lq_clip.shape[-1] < clip_samples:
                lq_clip = torch.nn.functional.pad(lq_clip, (0, clip_samples - lq_clip.shape[-1]))
            if hq_clip.shape[-1] < clip_samples:
                hq_clip = torch.nn.functional.pad(hq_clip, (0, clip_samples - hq_clip.shape[-1]))

            out_name = f"{stem}_clip{clip_idx}.wav"
            torchaudio.save(os.path.join(dst_root, "LQ", out_name), lq_clip, _SR, encoding="PCM_F", bits_per_sample=32)
            torchaudio.save(os.path.join(dst_root, "HQ", out_name), hq_clip, _SR, encoding="PCM_F", bits_per_sample=32)

            print_only(f"[data/val]   {stem} clip{clip_idx}: {clip_sec:.0f}s @ {start//_SR}s  [{song_idx+1}/{len(pairs)}]")
            n += 1

    print_only(f"[data/val] Done -- {n} clip files -> {dst_root}")


def prepare_data(cfg: DictConfig) -> None:
    # Sync chunk size globals from config so _slice_and_save uses the correct sizes.
    global _CHUNK_SEC, _CHUNK_SAMPLES, _HOP_SAMPLES
    _CHUNK_SEC     = float(getattr(cfg.datas, "segment_sec", 3))
    _CHUNK_SAMPLES = int(_CHUNK_SEC * _SR)
    _HOP_SAMPLES   = int(_CHUNK_SAMPLES * (1 - _OVERLAP))
    """
    Auto-preprocessing pipeline called before training.

    Accepts any of these layouts under data/train/ and data/val/:
        A)  <song>_LQ/  <song>_HQ/        -- subdirectory pairs
        B)  <song>_LQ.wav  <song>_HQ.wav  -- flat postfix files
        C)  LQ/  HQ/                       -- already normalized

    Layouts A and B are automatically reorganized into LQ/ + HQ/ in-place,
    then chunked into:
        usr/cache/chunks/<key>/train/LQ|HQ
        usr/cache/chunks/<key>/val/LQ|HQ

    Skips any split that is already chunked.
    """
    # Data source dirs live under data/<name>/train and data/<name>/val
    data_root  = os.path.join(_REPO_ROOT, "data", cfg.exp.name)
    data_train = os.path.join(data_root, "train")
    data_val   = os.path.join(data_root, "val")

    cached_aug_fn = _build_cached_aug_fn(cfg)
    _align_raw    = getattr(cfg.datas, "align_data", False)
    if isinstance(_align_raw, int) and not isinstance(_align_raw, bool) and _align_raw != 0:
        fixed_delay = int(_align_raw)
    else:
        fixed_delay = None

    aug_cfg = getattr(cfg.datas, "augmentation", None)

    # Normalize source dirs so _source_md5s can hash them correctly before chunking.
    _normalize_data_dir(data_train, "train")
    _normalize_data_dir(data_val, "val")

    # Chunks live in usr/cache/chunks/<key>/<split>/ keyed on (source md5s + params).
    # Any config that requests the same dataset with the same params reuses the cache.
    train_key = _chunk_cache_key(data_train, fixed_delay, aug_cfg)
    val_key   = _chunk_cache_key(data_val,   fixed_delay, None)

    train_chunks = _chunk_cache_lookup(train_key, "train")
    if train_chunks:
        print_only(f"[data/train] Cache hit ({train_key[:8]}...) -- skipping chunking.")
    else:
        train_chunks = os.path.join(_CHUNK_CACHE_DIR, train_key, "train")
        _chunk_split(data_train, train_chunks, "train", cached_aug_fn=cached_aug_fn, fixed_delay=fixed_delay)

    # Val clips: two 10s highest-RMS clips per song (ViSQOL-compliant), cached to disk.
    # Reuses the WAV conversion cache from _chunk_split so no re-encoding.
    val_clip_sec = 10.0
    val_clip_key = _chunk_cache_key(data_val, fixed_delay, None, extra=f"valclip_2x{val_clip_sec:.0f}s_rms")
    val_wav_dir  = os.path.join(_CHUNK_CACHE_DIR, val_clip_key, "val")
    val_lq_check = os.path.join(val_wav_dir, "LQ")
    if os.path.isdir(val_lq_check) and any(f.endswith(".wav") for f in os.listdir(val_lq_check)):
        print_only(f"[data/val]   Cache hit ({val_clip_key[:8]}...) -- skipping clip extraction.")
    else:
        _extract_val_clips(data_val, val_wav_dir, clip_sec=val_clip_sec, fixed_delay=fixed_delay)

    # Expose resolved absolute paths back into cfg so the datamodule picks them up.
    with open_dict(cfg):
        cfg.datas.train_dir = train_chunks
        cfg.datas.eval_dir  = val_wav_dir

    # Val bootstrap from train chunks
    # If val is still empty after WAV conversion (no data/val source exists),
    # copy a random selection of train chunks into val -- without removing
    # them from training. Chunks are picked by randomly selecting songs first,
    # then random chunks from those songs, so val covers diverse source material.
    val_lq = os.path.join(val_wav_dir, "LQ")
    val_hq = os.path.join(val_wav_dir, "HQ")
    val_has_files = (
        os.path.isdir(val_lq)
        and any(f.endswith(".wav") for f in os.listdir(val_lq))
    ) if os.path.isdir(val_lq) else False

    if not val_has_files:
        import shutil, random as _random
        n_bootstrap = int(cfg.datas.get("val_bootstrap_chunks", 50))
        train_lq = os.path.join(train_chunks, "LQ")
        train_hq = os.path.join(train_chunks, "HQ")

        if not os.path.isdir(train_lq) or not any(f.endswith(".wav") for f in os.listdir(train_lq)):
            print_only("[data/val] No train chunks available for val bootstrap -- skipping.")
        else:
            # Build stem -> [chunk filenames] map grouped by song
            from collections import defaultdict
            song_chunks = defaultdict(list)
            for fname in sorted(os.listdir(train_lq)):
                if not fname.endswith(".wav"):
                    continue
                # Stem is everything before the last _NNNN chunk index
                parts = fname.rsplit("_", 1)
                song_key = parts[0] if len(parts) == 2 and parts[1].replace(".wav","").isdigit() else fname
                song_chunks[song_key].append(fname)

            songs = list(song_chunks.keys())
            _random.shuffle(songs)

            selected = []
            # Round-robin across songs until we have enough
            song_iters = {s: iter(_random.sample(song_chunks[s], len(song_chunks[s]))) for s in songs}
            while len(selected) < n_bootstrap:
                progress = False
                for song in songs:
                    if len(selected) >= n_bootstrap:
                        break
                    try:
                        selected.append(next(song_iters[song]))
                        progress = True
                    except StopIteration:
                        pass
                if not progress:
                    break

            print_only(f"[data/val] Bootstrapped {len(selected)} val chunks from {len(songs)} training songs (copied, not moved).")

            os.makedirs(val_lq, exist_ok=True)
            os.makedirs(val_hq, exist_ok=True)
            for fname in selected:
                shutil.copy2(os.path.join(train_lq, fname), os.path.join(val_lq, fname))
                shutil.copy2(os.path.join(train_hq, fname), os.path.join(val_hq, fname))

            print_only(f"[data/val] Bootstrapped {len(selected)} val chunks across {len(songs)} songs (copied, not moved).")

def freeze_early_layers(model, n_layers_to_freeze=4):
    """
    Freeze the band-split front-end (BN) and first N BSNet layers.
    Default of 4 keeps VRAM and backprop cost manageable on the universal
    model (feature_dim=384) with an 11 GB card -- only layers 4-5 and the
    output heads are trained, which is where band reconstruction happens
    and where codec-specific adaptation matters most.
    """
    # Freeze band normalization and bottleneck front-end
    for param in model.BN.parameters():
        param.requires_grad = False

    # Freeze first N layers of the BSNet stack
    for i, layer in enumerate(model.net):
        if i < n_layers_to_freeze:
            for param in layer.parameters():
                param.requires_grad = False

    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total  = sum(p.numel() for p in model.parameters())
    print_only(f"Frozen {frozen:,} / {total:,} parameters ({100*frozen/total:.1f}%)")

def append_extra_layers(model, n_extra: int):
    """
    Freeze all pretrained layers (BN + entire net stack) and append n_extra new
    BSNet blocks to model.net that start as near-identity.

    Each new block is zero-initialized so it passes input through unchanged at
    the start of training: Roformer.output and Roformer.MLP_output are zeroed
    (Roformer then acts as identity via its internal residuals), and the final
    conv of each ICB ConvActNorm1d block is zeroed (so seq_net output ≈ 0,
    and BSNet output ≈ band_net output ≈ input). The new blocks are the only
    trainable parameters.
    """
    if n_extra <= 0:
        return

    # Freeze everything pretrained: BN front-end + all existing net layers + output heads
    for param in model.parameters():
        param.requires_grad = False

    # Determine feature_dim from the existing net
    feature_dim = model.feature_dim

    from look2hear.models.apollo import BSNet

    new_layers = nn.ModuleList()
    for _ in range(n_extra):
        block = BSNet(feature_dim)

        # Zero-init Roformer output projections so attention + MLP are identity at init
        nn.init.zeros_(block.band_net.output.weight)
        nn.init.zeros_(block.band_net.MLP_output.weight)

        # Zero-init the last conv in each ConvActNorm1d inside ICB so seq_net ≈ 0
        for can in block.seq_net.blocks:
            # ConvActNorm1d.conv is Sequential; last element is the output Conv1d
            last_conv = can.conv[-1]
            nn.init.zeros_(last_conv.weight)
            if last_conv.bias is not None:
                nn.init.zeros_(last_conv.bias)

        # New block trains freely
        for param in block.parameters():
            param.requires_grad = True

        new_layers.append(block)

    # Extend model.net (nn.Sequential) with the new blocks
    existing = list(model.net.children())
    model.net = nn.Sequential(*(existing + list(new_layers)))

    pretrained_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable_new    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total            = pretrained_frozen + trainable_new
    print_only(f"[extra_layers] Added {n_extra} new BSNet layer(s) (zero-init). "
               f"Pretrained frozen: {pretrained_frozen:,} | New trainable: {trainable_new:,} | Total: {total:,}")


def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    # Apply hardware / compiler optimisations declared in cfg.optimizations
    apply_optimizations(cfg)

    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    # Auto-preprocess raw data and bootstrap eval if needed
    prepare_data(cfg)

    # Recompute val_clip_key here (same formula as prepare_data) for use in the baseline cache.
    _vck_data_val    = os.path.join(_REPO_ROOT, "data", cfg.exp.name, "val")
    _vck_fixed_delay = int(cfg.datas.fixed_align_samples) if getattr(cfg.datas, "fixed_align_samples", None) else None
    _vck_clip_sec    = 10.0
    val_clip_key     = _chunk_cache_key(_vck_data_val, _vck_fixed_delay, None, extra=f"valclip_2x{_vck_clip_sec:.0f}s_rms")

    # Verify chunks exist -- if data/ was empty, provide a clear error
    train_lq   = os.path.join(cfg.datas.train_dir, "LQ")
    if not os.path.isdir(train_lq) or not any(f.endswith(".wav") for f in os.listdir(train_lq)):
        _name = cfg.exp.name
        print_only("")
        print_only("ERROR: No training chunks found.")
        print_only("")
        print_only(f"  Populate data/{_name}/train/ and data/{_name}/val/ with paired audio files:")
        print_only(f"    data/{_name}/train/LQ/   <- degraded audio (MP3, FLAC, WAV)")
        print_only(f"    data/{_name}/train/HQ/   <- clean reference (same filenames)")
        print_only(f"    data/{_name}/val/LQ/")
        print_only(f"    data/{_name}/val/HQ/")
        print_only("")
        print_only("  Or drop _LQ/_HQ files directly in data/ and they will be moved automatically.")
        print_only("")
        raise SystemExit(1)

    # Instantiate datamodule
    print_only(f"Instantiating datamodule <{cfg.datas._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datas)

    # Resolve run directory -- fresh start gets a timestamped subfolder,
    # resume reuses the most recent existing run folder.
    from datetime import datetime as _dt
    _base_dir = os.path.join(cfg.exp.dir, cfg.exp.name)
    ckpt_path = None

    if cfg.get("resume", False):
        # Find the most recently modified run folder that has checkpoints
        _run_dir = None
        if os.path.isdir(_base_dir):
            _migrate_legacy_ckpt_names(_base_dir)
            _subdirs = [
                os.path.join(_base_dir, d)
                for d in os.listdir(_base_dir)
                if os.path.isdir(os.path.join(_base_dir, d, "checkpoints"))
            ]
            if _subdirs:
                _run_dir = max(_subdirs, key=os.path.getmtime)
        if _run_dir is None:
            # No existing runs -- fresh start
            _run_id = _dt.now().strftime("%Y%m%d_%H%M%S")
            _run_dir = os.path.join(_base_dir, _run_id)
            print_only(f"[resume] No existing runs found -- starting fresh run: {_run_id}")
        else:
            _run_id = os.path.basename(_run_dir)
            ckpt_dir = os.path.join(_run_dir, "checkpoints")
            candidates = [
                os.path.join(ckpt_dir, f)
                for f in os.listdir(ckpt_dir)
                if f.endswith(".ckpt") and f != "last.ckpt"
            ]
            if candidates:
                ckpt_path = max(candidates, key=os.path.getmtime)
                print_only(f"[resume] Resuming run {_run_id}")
                print_only(f"[resume] Checkpoint: {os.path.basename(ckpt_path)}")
                print_only("[resume] Skipping pretrain weight loading -- checkpoint takes precedence.")
            else:
                print_only(f"[resume] Run folder found but no checkpoints -- starting from pretrained weights.")
    else:
        _run_id = _dt.now().strftime("%Y%m%d_%H%M%S")
        _run_dir = os.path.join(_base_dir, _run_id)
        print_only(f"[run] New run: {_run_id}")

    os.makedirs(_run_dir, exist_ok=True)

    # Pretrained weights resolution
    # Priority:
    #   1. Explicit --weights_path / cfg.weights_path
    #   2. Auto-scan ./models/ for known filenames (first match wins)
    #   3. Auto-download if a URL is configured in _PRETRAINED_MODELS
    #   4. Fall back to HuggingFace hub (legacy behaviour)
    # Skipped entirely when resuming from a checkpoint (ckpt_path is set above).
    # -----------------------------------------------------------------------
    feature_dim = cfg.model.get("feature_dim", 256)

    def _load_weights(path: str, feature_dim: int):
        """Load a .pth or .ckpt file and return an Apollo model with weights applied."""
        if path.endswith(".ckpt"):
            try:
                ckpt = torch.load(path, map_location="cpu", weights_only=True)
            except Exception:
                print_only("[weights] weights_only=True failed (legacy checkpoint format) -- retrying with weights_only=False")
                ckpt = torch.load(path, map_location="cpu", weights_only=False)
            raw = ckpt["state_dict"]
            if any(k.startswith("audio_model.") for k in raw.keys()):
                model_state = {k.replace("audio_model.", ""): v
                               for k, v in raw.items() if k.startswith("audio_model.")}
                print_only("Detected Lightning checkpoint format (audio_model. prefix)")
            else:
                model_state = raw
                print_only("Detected bare state dict format (no prefix)")
            m = look2hear.models.apollo.Apollo(sr=44100, win=20,
                                               feature_dim=feature_dim, layer=6)
            missing, unexpected = m.load_state_dict(model_state, strict=False)
            if missing:
                print_only(f"Missing keys ({len(missing)}): {missing[:5]}...")
            if unexpected:
                print_only(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
            if not missing:
                print_only("All keys loaded successfully.")
        else:
            m = look2hear.models.BaseModel.from_pretrain(
                path, sr=44100, win=20, feature_dim=feature_dim, layer=6
            )
        return m

    if ckpt_path is not None:
        # Resuming -- Lightning will restore all weights from the checkpoint.
        # Just instantiate a bare model so the system can be constructed;
        # the state dict will be overwritten by trainer.fit(ckpt_path=...).
        print_only("[weights] Resume mode -- skipping pretrain load, instantiating bare model.")
        model = look2hear.models.apollo.Apollo(
            sr=44100, win=20, feature_dim=feature_dim, layer=6
        )
    else:
        local_path = cfg.get("weights_path", None)

        if not local_path:
            # Auto-scan ./models/ -- prefer the checkpoint that matches feature_dim.
            # 384 -> apollo_model_uni.ckpt first; 256 -> apollo_model.ckpt / pytorch_model.bin first.
            os.makedirs(_MODELS_DIR, exist_ok=True)
            is_uni = (feature_dim == 384)
            scan_order = (
                ["apollo_uni.ckpt", "apollo_model_uni.ckpt", "apollo.ckpt", "apollo_model.ckpt", "pytorch_model.bin"]
                if is_uni else
                ["apollo.ckpt", "apollo_model.ckpt", "pytorch_model.bin", "apollo_uni.ckpt", "apollo_model_uni.ckpt"]
            )
            for fname in scan_order:
                candidate = os.path.join(_MODELS_DIR, fname)
                if os.path.isfile(candidate):
                    print_only(f"[weights] Found pretrained model in models/: {fname}")
                    local_path = candidate
                    break
                url = _PRETRAINED_MODELS.get(fname)
                if url is not None:
                    print_only(f"[weights] Downloading {fname} from configured URL...")
                    import urllib.request
                    urllib.request.urlretrieve(url, candidate)
                    print_only(f"[weights] Saved to {candidate}")
                    local_path = candidate
                    break

        if local_path:
            print_only(f"[weights] Loading from: {local_path}")
            model = _load_weights(local_path, feature_dim)
            print_only("[weights] Weights loaded.")
        else:
            # Final fallback -- HuggingFace hub (256-dim only; no uni model exists on HF).
            if feature_dim == 384:
                raise FileNotFoundError(
                    "[weights] feature_dim=384 requires a local universal checkpoint, "
                    "but none was found in models/. "
                    "Place apollo_model_uni.ckpt in the models/ folder, "
                    "or set weights_path in your config to point at it explicitly. "
                    "The JusperLee/Apollo HuggingFace repo does not host a 384-dim checkpoint."
                )
            hf_repo  = "JusperLee/Apollo"
            hf_file  = "pytorch_model.bin"
            print_only(f"[weights] No local model found in models/ -- downloading from HuggingFace ({hf_file})...")
            from huggingface_hub import hf_hub_download
            weights_path = hf_hub_download(repo_id=hf_repo, filename=hf_file)
            print_only(f"[weights] Cached at: {weights_path}")
            model = look2hear.models.BaseModel.from_pretrain(
                weights_path, sr=44100, win=20, feature_dim=feature_dim, layer=6
            )
            print_only("[weights] Pretrained weights loaded.")

    n_extra = cfg.training.get("extra_layers", 0)
    if n_extra > 0 and not is_resume:
        # Freeze all pretrained weights and append new zero-init BSNet layers.
        # append_extra_layers handles all freezing, so skip freeze_early_layers.
        append_extra_layers(model, n_extra)
    else:
        # Standard partial freeze: BN front-end + first N layers, rest trainable.
        # (no-op when resuming since frozen params are restored by the checkpoint too)
        freeze_early_layers(model, n_layers_to_freeze=cfg.training.n_layers_to_freeze)

    # Instantiate discriminator fresh -- learns your artifact type from scratch
    print_only(f"Instantiating Discriminator <{cfg.discriminator._target_}>")
    discriminator = hydra.utils.instantiate(cfg.discriminator)

    # Instantiate optimizers
    print_only(f"Instantiating optimizers")
    opt_cfg = cfg.optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    # Optimizer factory
    # Reads cfg.optimizer.type to select the optimizer:
    #   adamw        -- standard 32-bit AdamW
    #   adamw_8bit   -- 8-bit AdamW via bitsandbytes (pip install bitsandbytes)
    #   cpu_offload  -- 32-bit AdamW with momentum states in CPU RAM
    #   gefen        -- Gefen (drop-in AdamW, ~8x lower optimizer memory, pip install gefen)
    #   gefen_muon   -- GefenMuon for 2D params + Gefen for rest (recommended for fine-tuning)

    class _ComboOpt:
        """Thin wrapper combining two optimizers into one interface for StepLR + Lightning."""
        def __init__(self, opt_2d, opt_rest):
            self.opt_2d   = opt_2d
            self.opt_rest = opt_rest
            # expose param_groups and defaults so StepLR doesn't choke
            self.param_groups = opt_2d.param_groups + opt_rest.param_groups
            self.defaults = opt_2d.defaults

        def step(self, closure=None):
            self.opt_2d.step(closure)
            self.opt_rest.step(closure)

        def zero_grad(self, set_to_none=True):
            self.opt_2d.zero_grad(set_to_none=set_to_none)
            self.opt_rest.zero_grad(set_to_none=set_to_none)

        def state_dict(self):
            return {"opt_2d": self.opt_2d.state_dict(), "opt_rest": self.opt_rest.state_dict()}

        def load_state_dict(self, sd):
            self.opt_2d.load_state_dict(sd["opt_2d"])
            self.opt_rest.load_state_dict(sd["opt_rest"])
            self.param_groups = self.opt_2d.param_groups + self.opt_rest.param_groups

    def _make_optimizer(params, lr, weight_decay, betas):
        opt_type = opt_cfg.get("type", "adamw").lower()

        if opt_type == "gefen_muon":
            try:
                import pkg_resources, packaging as _pkg  # ensure pkg_resources.packaging exists
                if not hasattr(pkg_resources, "packaging"):
                    pkg_resources.packaging = _pkg
                from gefen import Gefen, GefenMuon
                params_2d   = [p for p in params if p.ndim == 2]
                params_rest = [p for p in params if p.ndim != 2]
                opt_2d   = GefenMuon(params_2d,   lr=lr, weight_decay=weight_decay) if params_2d   else None
                opt_rest = Gefen(params_rest, lr=lr, weight_decay=weight_decay, betas=betas, fused=True) if params_rest else None
                if opt_2d is None:
                    print_only(f"[optimizer] Gefen (no 2D params -- gefen_muon falls back to Gefen) -- lr={lr}")
                    return opt_rest
                if opt_rest is None:
                    print_only(f"[optimizer] GefenMuon (all params 2D) -- lr={lr}")
                    return opt_2d
                print_only(f"[optimizer] GefenMuon (2D) + Gefen (rest) -- lr={lr} -- 2D={len(params_2d)} rest={len(params_rest)}")
                return _ComboOpt(opt_2d, opt_rest)
            except ImportError as _e:
                print_only(f"[optimizer] gefen unavailable ({_e}) -- falling back to AdamW32bit")

        if opt_type == "gefen":
            try:
                import pkg_resources, packaging as _pkg  # ensure pkg_resources.packaging exists
                if not hasattr(pkg_resources, "packaging"):
                    pkg_resources.packaging = _pkg
                from gefen import Gefen
                opt = Gefen(params, lr=lr, weight_decay=weight_decay, betas=betas, fused=True)
                print_only(f"[optimizer] Gefen -- lr={lr}")
                return opt
            except ImportError as _e:
                print_only(f"[optimizer] gefen unavailable ({_e}) -- falling back to AdamW32bit")

        if opt_type == "adamw_8bit":
            try:
                import bitsandbytes as bnb
                opt = bnb.optim.AdamW8bit(params, lr=lr, weight_decay=weight_decay, betas=betas)
                print_only(f"[optimizer] AdamW8bit (bitsandbytes) -- lr={lr}")
                return opt
            except ImportError:
                print_only("[optimizer] bitsandbytes not installed -- falling back to AdamW32bit")

        if opt_type == "cpu_offload":
            try:
                opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas, fused=False)
                print_only(f"[optimizer] AdamW32bit with CPU offload -- lr={lr}")
                return opt
            except Exception as e:
                print_only(f"[optimizer] CPU offload failed ({e}), falling back to AdamW32bit")

        opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas)
        print_only(f"[optimizer] AdamW32bit -- lr={lr}")
        return opt

    optimizer_g = _make_optimizer(
        trainable_params,
        lr=opt_cfg.lr_g,
        weight_decay=opt_cfg.weight_decay,
        betas=tuple(opt_cfg.get("betas_g", [0.9, 0.999])),
    )
    optimizer_d = _make_optimizer(
        list(discriminator.parameters()),
        lr=opt_cfg.lr_d,
        weight_decay=opt_cfg.weight_decay,
        betas=tuple(opt_cfg.get("betas_d", [0.5, 0.99])),
    )

    # Instantiate schedulers
    scheduler_g = hydra.utils.instantiate(cfg.scheduler_g, optimizer=optimizer_g)
    scheduler_d = hydra.utils.instantiate(cfg.scheduler_d, optimizer=optimizer_d)

    # Instantiate losses
    print_only(f"Instantiating losses")
    loss_g = hydra.utils.instantiate(cfg.loss_g)
    loss_d = hydra.utils.instantiate(cfg.loss_d)
    losses = {"g": loss_g, "d": loss_d}

    # Instantiate metrics
    print_only(f"Instantiating metrics <{cfg.metrics._target_}>")
    metrics = hydra.utils.instantiate(cfg.metrics)

    # Instantiate system
    print_only(f"Instantiating system <{cfg.system._target_}>")
    val_audio_dir = os.path.join(_run_dir, "val_audio")
    # Strip keys that train.py manages explicitly to avoid double-pass crash when
    # a user config accidentally defines them under system: as well as training:
    _SYSTEM_MANAGED = {
        "model", "discriminator", "loss_func", "metrics", "optimizer", "scheduler",
        "val_audio_dir", "val_songs", "val_preview_samples", "val_metric_songs", "val_rotate_every",
        "gradient_checkpointing", "grad_accum_steps", "visqol_fraction",
        "target_band_loss_enabled", "target_band_loss_lo_hz", "target_band_loss_hi_hz",
        "val_songs", "val_audio_pairs",
    }
    from omegaconf import OmegaConf
    _sys_cfg = OmegaConf.to_container(cfg.system, resolve=True)
    _sys_cfg = {k: v for k, v in _sys_cfg.items() if k not in _SYSTEM_MANAGED}
    system: LightningModule = hydra.utils.instantiate(
        _sys_cfg,
        model=model,
        discriminator=discriminator,
        loss_func=losses,
        metrics=metrics,
        optimizer=[optimizer_g, optimizer_d],
        scheduler=[scheduler_g, scheduler_d],
        val_audio_dir=val_audio_dir,
        val_songs=cfg.training.get("val_songs", cfg.training.get("val_metric_songs", cfg.training.get("val_metric_samples", 3))),
        val_rotate_every=cfg.training.get("val_rotate_every", "auto"),
        gradient_checkpointing=cfg.system.get("gradient_checkpointing", False),
        grad_accum_steps=cfg.training.get("grad_accum_steps", 1),
        visqol_fraction=cfg.system.get("visqol_fraction", 1.0),
        target_band_loss_enabled=cfg.system.get("target_band_loss_enabled", False),
        target_band_loss_lo_hz=cfg.system.get("target_band_loss_lo_hz", 13000.0),
        target_band_loss_hi_hz=cfg.system.get("target_band_loss_hi_hz", 19000.0),
    )

    # Callbacks
    # Patch all run-specific paths before any instantiation
    os.makedirs(os.path.join(_run_dir, "logs"), exist_ok=True)
    with open_dict(cfg):
        cfg.checkpoint.dirpath       = os.path.join(_run_dir, "checkpoints")
        cfg.logger.save_dir          = os.path.join(_run_dir, "logs")
        cfg.trainer.default_root_dir = _run_dir

    import time as _time
    from pytorch_lightning.callbacks import Callback as _Callback

    class StepPrinter(_Callback):
        """
        Replaces TQDM with a clean line-per-step console output.
        Reports wall-clock cumulative it/s from the first step of the session,
        excluding time spent in validation. Rises naturally as CUDA warms up.
        """

        def on_train_epoch_start(self, trainer, pl_module):
            self._epoch_batches    = trainer.num_training_batches
            self._last_global_step = trainer.global_step
            self._last_batch_idx   = 0
            self._val_t0           = None
            self._session_t0       = None   # set on first optimizer step
            self._session_done     = 0      # optimizer steps this session
            self._val_elapsed      = 0.0    # cumulative val time excluded from rate
            total = self._epoch_batches
            print(f"\nEpoch {trainer.current_epoch} -- {total} batches", flush=True)

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            now = _time.monotonic()

            if self._session_t0 is None:
                self._session_t0 = now

            # Count every batch so it/s matches the original batch-level rate
            self._session_done += 1
            elapsed = (now - self._session_t0) - self._val_elapsed
            its = self._session_done / elapsed if elapsed > 0 else 0.0

            # Only print on optimizer steps
            if trainer.global_step == self._last_global_step:
                return
            self._last_global_step = trainer.global_step
            done  = batch_idx + 1
            total = self._epoch_batches
            pct   = 100 * done / total
            self._last_batch_idx = batch_idx
            # Show last val metrics inline if available
            visqol = getattr(pl_module, "_last_val_visqol", None)
            hfnr    = getattr(pl_module, "_last_val_hfnr",    None)
            sisdr  = getattr(pl_module, "_last_val_sisdr",  None)
            tbl    = getattr(pl_module, "_last_val_tbl",    None)
            val_parts = []
            if visqol is not None: val_parts.append(f"visqol={float(visqol):.3f}")
            if sisdr  is not None: val_parts.append(f"sisdr={-float(sisdr):.3f}")
            if hfnr    is not None: val_parts.append(f"hfnr={float(hfnr):.3f}")
            if tbl    is not None: val_parts.append(f"tbl={float(tbl):.4f}")
            val_str = "  " + "  ".join(val_parts) if val_parts else ""
            print(
                f"\r  {pct:5.1f}%  step={trainer.global_step}  "
                f"{done}/{total}  {its:.2f} it/s{val_str}",
                end="", flush=True
            )
            self.on_train_batch_end_pause_check()

        def on_train_batch_end_pause_check(self):
            """Poll for PAUSED sentinel file and block until removed.
            TUI writes the file to the base exp dir (_base_dir); we check both
            that and _run_dir so it works regardless of timing."""
            pause_file = None
            for candidate in [os.path.join(_base_dir, "PAUSED"),
                               os.path.join(_run_dir,  "PAUSED")]:
                if os.path.isfile(candidate):
                    pause_file = candidate
                    break
            if pause_file is None:
                return
            print("\n[paused] Inference in progress -- training suspended.", flush=True)
            pause_start = _time.monotonic()
            while os.path.isfile(pause_file):
                _time.sleep(0.5)
            paused_dur = _time.monotonic() - pause_start
            if hasattr(self, "_val_elapsed"):
                self._val_elapsed += paused_dur
            print("[resumed] Training continuing...", flush=True)

        def on_train_epoch_end(self, trainer, pl_module):
            print(flush=True)

        def on_validation_epoch_start(self, trainer, pl_module):
            self._val_t0     = _time.monotonic()
            self._val_sanity = trainer.sanity_checking
            if not self._val_sanity:
                print("\r  Validating...                                                  ", end="", flush=True)

        def on_validation_epoch_end(self, trainer, pl_module):
            pass

        def on_validation_end(self, trainer, pl_module):
            val_dur = _time.monotonic() - self._val_t0 if self._val_t0 else 0.0
            if hasattr(self, '_val_elapsed'):
                self._val_elapsed += val_dur
            if not getattr(self, '_val_sanity', True):
                # Reprint the progress bar line in place with updated val metrics
                visqol = getattr(pl_module, "_last_val_visqol", None)
                hfnr    = getattr(pl_module, "_last_val_hfnr",    None)
                sisdr  = getattr(pl_module, "_last_val_sisdr",  None)
                tbl    = getattr(pl_module, "_last_val_tbl",    None)
                val_parts = []
                if visqol is not None: val_parts.append(f"visqol={float(visqol):.3f}")
                if sisdr  is not None: val_parts.append(f"sisdr={-float(sisdr):.3f}")
                if hfnr    is not None: val_parts.append(f"hfnr={float(hfnr):.3f}")
                if tbl    is not None: val_parts.append(f"tbl={float(tbl):.4f}")
                val_str = "  " + "  ".join(val_parts) if val_parts else ""
                step  = trainer.global_step
                epoch = trainer.current_epoch
                total = getattr(self, "_epoch_batches", 0)
                done  = getattr(self, "_last_batch_idx", 0) + 1 if total else 0
                pct   = 100 * done / total if total else 0.0
                print(
                    f"\r  {pct:5.1f}%  step={step}  {done}/{total}  --{val_str}",
                    end="", flush=True
                )

    callbacks: List[Callback] = [StepPrinter()]

    _lvb = cfg.trainer.get("limit_val_batches", 1.0)
    val_disabled = (isinstance(_lvb, (int, float)) and float(_lvb) == 0.0)
    if val_disabled:
        print_only("[train] limit_val_batches=0 -- skipping early_stopping and checkpoint callbacks")

    checkpoint = None
    if cfg.get("checkpoint") and not val_disabled:
        print_only(f"Instantiating checkpoint")
        checkpoint = hydra.utils.instantiate(cfg.checkpoint)
        # Monitor hfnr, not a composite -- see note above.
        checkpoint.monitor = "visqol"
        checkpoint.mode    = "min"
        checkpoint.save_top_k = -1
        # Full stats in filename; all metrics are logged via self.log() so
        # Lightning can interpolate them here.
        # Lightning interpolates {metric:fmt} as metric=VALUE automatically.
        # Don't add extra label= text before {metric} tokens or they double up.
        # Result: step=000200-sisdr=-20.892-visqol=3.821-hfnr=0.968
        checkpoint.filename = (
            "{step:06d}"
            "-{sisdr:.3f}"
            "-{visqol:.3f}"
            "-{hfnr:.3f}"
        )
        callbacks.append(checkpoint)

    # Instantiate logger
    print_only(f"Instantiating logger <{cfg.logger._target_}>")
    logger = hydra.utils.instantiate(cfg.logger)
    logger.log_hyperparams = lambda *a, **kw: None

    # Instantiate trainer -- single GPU, no DDP
    # Fix val_check_interval: the config value is in optimizer steps (what the
    # display shows), but Lightning counts *batches*. Multiply by grad_accum_steps
    # so the val fires at the step number the user expects to see on screen.
    _accum = cfg.training.get("grad_accum_steps", 1)
    _vci   = cfg.trainer.get("val_check_interval", None)
    if _vci is not None and isinstance(_vci, int) and _accum > 1:
        with open_dict(cfg):
            cfg.trainer.val_check_interval = int(_vci) * int(_accum)
        print_only(f"[trainer] val_check_interval adjusted: {_vci} steps x {_accum} accum = {cfg.trainer.val_check_interval} batches")

    print_only(f"Instantiating trainer")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=logger,
        enable_progress_bar=False,
    )

    def _save_and_exit(sig=None, frame=None):
        print_only("\n[interrupt] Ctrl+C caught -- saving checkpoint...")
        try:
            ckpt_dir = os.path.join(_run_dir, "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            step = trainer.global_step
            try:
                m = trainer.callback_metrics
                vl     = m.get("sisdr",   None)
                visqol = m.get("visqol", None)
                hfnr    = m.get("hfnr",    None)
                parts = [f"{step:06d}"]
                if vl     is not None: parts.append(f"sisdr={float(vl):.3f}")
                if visqol is not None: parts.append(f"visqol={float(visqol):.3f}")
                if hfnr    is not None: parts.append(f"hfnr={float(hfnr):.3f}")
                fname = "-".join(parts) + ".ckpt"
            except Exception:
                fname = f"{step:06d}.ckpt"
            out_path = os.path.join(ckpt_dir, fname)
            trainer.save_checkpoint(out_path)
            print_only(f"[interrupt] Saved to {out_path}")
        except Exception as e:
            print_only(f"[interrupt] Save failed: {e}")
        os._exit(0)

    import signal as _signal
    _signal.signal(_signal.SIGINT, _save_and_exit)

    # Baseline val pass on pretrained weights before any training begins.
    # Skipped on resume since the checkpoint already has training history.
    # Results are cached in usr/cache/baseline/ keyed on (weights md5 + val clip key)
    # so restarting a fresh run with the same pretrain + dataset skips the pass entirely.
    if ckpt_path is None and not val_disabled:
        import hashlib as _hashlib, json as _json

        def _baseline_cache_key(weights_path: str, val_clip_key: str) -> str:
            h = _hashlib.md5()
            if weights_path and os.path.isfile(weights_path):
                wh = _hashlib.md5()
                with open(weights_path, "rb") as _wf:
                    for _blk in iter(lambda: _wf.read(1 << 20), b""):
                        wh.update(_blk)
                h.update(wh.hexdigest().encode())
            h.update(val_clip_key.encode())
            return h.hexdigest()[:16]

        _baseline_cache_dir  = os.path.join(_CACHE_DIR, "baseline")
        _baseline_cache_key_ = _baseline_cache_key(
            local_path or "",
            val_clip_key,
        )
        _baseline_cache_file = os.path.join(_baseline_cache_dir, f"{_baseline_cache_key_}.json")

        def _apply_baseline_to_system(bl_dict: dict) -> None:
            """Populate system._last_val_* from a baseline results dict so the
            progress bar shows baseline values before the first training step."""
            system._last_val_sisdr  = bl_dict.get("sisdr")          # negative SI-SDR
            system._last_val_hfnr    = bl_dict.get("hfnr")
            system._last_val_visqol = bl_dict.get("visqol")

        if os.path.isfile(_baseline_cache_file):
            try:
                with open(_baseline_cache_file) as _bcf:
                    _cached_bl = _json.load(_bcf)
                print_only(f"[baseline] Cached ({_baseline_cache_key_[:8]}...)")
                _apply_baseline_to_system(_cached_bl)
            except Exception as _e:
                print_only(f"[baseline] Cache read failed ({_e}) -- will re-run.")
                os.remove(_baseline_cache_file)
        else:
            print_only("[baseline] Evaluating pretrained weights...")
            _baseline_ok = True
            try:
                import psutil as _ps
                _vm = _ps.virtual_memory()
                _headroom = (_vm.total * opt.get("ram_limit_fraction", 0.95)) - _vm.used
                if _headroom < 1.5 * (1024 ** 3):
                    print_only(f"[baseline] Skipped -- only {_headroom/(1024**3):.1f} GB RAM headroom.")
                    _baseline_ok = False
            except Exception:
                pass
            if _baseline_ok:
                try:
                    datamodule.setup("fit")
                    disc = getattr(system, "discriminator", None)
                    if disc is not None:
                        disc.cpu()
                    import gc; gc.collect()
                    torch.cuda.empty_cache()
                    baseline_results = trainer.validate(system, datamodule=datamodule, verbose=False)
                    if disc is not None:
                        disc.cuda()
                    if baseline_results:
                        bl = baseline_results[0]
                        _to_cache = {k: float(v) for k, v in bl.items() if v is not None}
                        os.makedirs(_baseline_cache_dir, exist_ok=True)
                        with open(_baseline_cache_file, "w") as _bcf:
                            _json.dump(_to_cache, _bcf, indent=2)
                        _apply_baseline_to_system(_to_cache)
                except Exception as e:
                    print_only(f"[baseline] Skipped: {e}")

    if ckpt_path is not None:
        _ckpt_data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        _patched = False

        if "pytorch-lightning_version" not in _ckpt_data:
            print_only("[resume] Checkpoint missing 'pytorch-lightning_version' -- patching.")
            _ckpt_data["pytorch-lightning_version"] = pl.__version__
            _patched = True

        if "state_dict" in _ckpt_data:
            _sd = _ckpt_data["state_dict"]
            _current_keys = set(_sd.keys())
            # Detect legacy checkpoint: model keys lack the LightningModule wrapper prefixes.
            # Legacy: "BN.0.0.weight" -> expected: "audio_model.BN.0.0.weight"
            _needs_remap = any(
                k.startswith(("BN.", "net.", "output.", "hann_win"))
                for k in _current_keys
            )
            if _needs_remap:
                print_only("[resume] Legacy state_dict detected -- remapping key prefixes.")
                _new_sd = {}
                _disc_legacy_prefixes = ("window_weights", "discriminators.")
                for k, v in _sd.items():
                    if any(k.startswith(p) for p in _disc_legacy_prefixes):
                        _new_sd[f"discriminator.{k}"] = v
                    else:
                        _new_sd[f"audio_model.{k}"] = v
                _ckpt_data["state_dict"] = _new_sd
                _patched = True
                print_only(f"[resume] Remapped {len(_new_sd)} keys.")

            # Reconcile keys against live model: drop unexpected, fill missing with init values.
            _current_model_keys = set(system.state_dict().keys())
            _sd_now = _ckpt_data["state_dict"]
            _unexpected = [k for k in _sd_now if k not in _current_model_keys]
            _missing    = [k for k in _current_model_keys if k not in _sd_now]
            if _unexpected or _missing:
                for k in _unexpected:
                    del _sd_now[k]
                _fresh_sd = system.state_dict()
                for k in _missing:
                    _sd_now[k] = _fresh_sd[k]
                print_only(f"[resume] Key reconciliation: dropped {len(_unexpected)} unexpected, filled {len(_missing)} missing.")
                _patched = True

        # If checkpoint has no optimizer state, inject empty stubs so Lightning
        # doesn't crash trying to restore them. Training will start fresh optimizers.
        if "optimizer_states" not in _ckpt_data or not _ckpt_data["optimizer_states"]:
            print_only("[resume] Checkpoint has no optimizer state -- starting optimizers fresh.")
            _ckpt_data["optimizer_states"] = []
            _ckpt_data["lr_schedulers"] = []
            _patched = True

        if _patched:
            torch.save(_ckpt_data, ckpt_path)
            print_only("[resume] Checkpoint patched and saved.")
        del _ckpt_data

    try:
        trainer.fit(system, datamodule=datamodule, ckpt_path=ckpt_path)
    except torch.cuda.OutOfMemoryError as e:
        print_only(f"\n[OOM] CUDA out of memory -- exiting cleanly. Try reducing batch_size or num_workers.")
        print_only(f"[OOM] {e}")
        torch.cuda.empty_cache()
        os._exit(1)
    except MemoryError as e:
        print_only(f"\n[OOM] System RAM exhausted -- exiting cleanly.")
        print_only(f"[OOM] {e}")
        os._exit(1)
    print_only("Training finished!")

    if checkpoint is not None:
        best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
        with open(os.path.join(_run_dir, "best_k_models.json"), "w") as f:
            json.dump(best_k, f, indent=0)

    best_path = getattr(checkpoint, "best_model_path", "") or "" if checkpoint is not None else ""
    if best_path and os.path.isfile(best_path):
        try:
            state_dict = torch.load(best_path, map_location="cpu", weights_only=True)
        except Exception:
            state_dict = torch.load(best_path, map_location="cpu", weights_only=False)
        system.load_state_dict(state_dict=state_dict["state_dict"])
        system.cpu()
        to_save = system.audio_model.serialize()
        torch.save(to_save, os.path.join(_run_dir, "best_model.pth"))
        print_only(f"[train] Best model saved to {_run_dir}/best_model.pth")
    else:
        print_only("[train] No best checkpoint found -- skipping best_model.pth export.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--conf_dir",
        default="configs/apollo.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--weights_path",
        default=None,
        help="Path to local weights file (.pth or .ckpt). If not set, downloads from HuggingFace.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the last checkpoint in the experiment checkpoint directory.",
    )
    args = parser.parse_args()
    cfg = OmegaConf.load(args.conf_dir)

    # --- Autodiscovery ---
    # Derive exp.name from config filename stem if not explicitly set in config.
    # e.g. configs/apollo_sftl.yaml -> "apollo_sftl"
    _conf_stem = os.path.splitext(os.path.basename(args.conf_dir))[0]
    if not cfg.exp.get("name"):
        cfg.exp.name = _conf_stem

    # Derive data and chunk paths from exp.name if not explicitly set in config.
    # data/<name>/train, data/<name>/val, chunks/<name>/train, chunks/<name>/val
    _data_name_root   = os.path.join(_REPO_ROOT, "data",   cfg.exp.name)
    _chunks_name_root = os.path.join(_REPO_ROOT, "chunks", cfg.exp.name)

    _default_train_dir = os.path.join("chunks", cfg.exp.name, "train")
    _default_eval_dir  = os.path.join("chunks", cfg.exp.name, "val")
    _generic = {"./chunks/train", "chunks/train", "./chunks/val", "chunks/val"}
    if not cfg.datas.get("train_dir") or cfg.datas.train_dir in _generic:
        cfg.datas.train_dir = _default_train_dir
    if not cfg.datas.get("eval_dir") or cfg.datas.eval_dir in _generic:
        cfg.datas.eval_dir = _default_eval_dir

    # If loose _LQ/_HQ files exist in data/ root, move them into data/<name>/
    _data_root = os.path.join(_REPO_ROOT, "data")
    if os.path.isdir(_data_root):
        import shutil as _shutil
        _loose = [f for f in os.listdir(_data_root)
                  if os.path.isfile(os.path.join(_data_root, f))
                  and (os.path.splitext(f)[0].upper().endswith("_LQ")
                       or os.path.splitext(f)[0].upper().endswith("_HQ"))
                  and os.path.splitext(f)[1].lower() in {".wav", ".mp3", ".flac"}]
        if _loose:
            os.makedirs(_data_name_root, exist_ok=True)
            for _f in _loose:
                _shutil.move(os.path.join(_data_root, _f),
                             os.path.join(_data_name_root, _f))
            print(f"[autodiscovery] Moved {len(_loose)} loose file(s) from data/ -> data/{cfg.exp.name}/")

    print(f"[autodiscovery] name={cfg.exp.name}  data=data/{cfg.exp.name}  chunks=chunks/{cfg.exp.name}")
    # --- End autodiscovery ---

    if args.weights_path:
        cfg.weights_path = args.weights_path
    if args.resume:
        cfg.resume = True

    os.makedirs(os.path.join(cfg.exp.dir, cfg.exp.name), exist_ok=True)

    train(cfg)
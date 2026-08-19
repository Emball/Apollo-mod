# @claude last-modified: 2026-05-05T06:34:39Z
# @claude last-commit: feat: major update — TUI, augmentation system, gradient checkpointing, optimization bootstrap
import json
from typing import Any, Dict, List, Optional, Tuple
import os
from omegaconf import OmegaConf, open_dict
import argparse
import pytorch_lightning as pl
import torch
import hydra
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from omegaconf import DictConfig

# Optimisation bootstrap — reads cfg.optimizations and applies everything
# in one place, before any model or trainer code runs.

def apply_optimizations(cfg: DictConfig) -> None:
    """Apply hardware/compiler optimisations declared in cfg.optimizations."""
    opt = cfg.get("optimizations", {})

    # TF32 matmuls (Ampere+, negligible quality loss)
    tf32 = opt.get("tf32", True)
    torch.set_float32_matmul_precision("high" if tf32 else "highest")
    if tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

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
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".triton_cache")
        os.makedirs(cache_dir, exist_ok=True)
        os.environ["TRITON_CACHE_DIR"] = cache_dir

    # Misc env flags
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # RAM watchdog — kills the process cleanly if system RAM crosses the threshold
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
        print_only("[watchdog] psutil not installed — RAM watchdog disabled. "
                   "Run: pip install psutil")
        return

    total = psutil.virtual_memory().total
    threshold = total * limit_fraction
    threshold_gb = threshold / (1024 ** 3)
    print_only(f"[watchdog] RAM watchdog active — will exit cleanly above "
               f"{threshold_gb:.1f} GB ({limit_fraction*100:.0f}% of total)")

    def _watch():
        import time
        time.sleep(30)  # wait for DataLoader workers to finish spawning
        while True:
            used = psutil.virtual_memory().used
            if used >= threshold:
                used_gb = used / (1024 ** 3)
                print_only(f"\n[watchdog] SYSTEM RAM CRITICAL: {used_gb:.1f} GB used "
                            f"(threshold {threshold_gb:.1f} GB) — exiting cleanly to protect OS")
                os._exit(1)
            time.sleep(2)

    t = threading.Thread(target=_watch, daemon=True)
    t.start()

import look2hear.system
import look2hear.datas
import look2hear.losses
import look2hear.models
import look2hear.models.apollo
from look2hear.utils import RankedLogger, instantiate, print_only
import warnings
warnings.filterwarnings("ignore")

# Constants mirrored from preprocess_pairs.py
_SR           = 44100
_CHUNK_SEC    = 3
_OVERLAP      = 0.5
_CHUNK_SAMPLES = int(_CHUNK_SEC * _SR)
_HOP_SAMPLES   = int(_CHUNK_SAMPLES * (1 - _OVERLAP))
_SUPPORTED_EXTS = {".wav", ".mp3", ".flac"}

# Models directory — pretrained weights are looked up here automatically
_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# Pretrained model filenames to search for (base → universal).
# Set download URLs here once you have them; None = skip auto-download.
_PRETRAINED_MODELS = {
    "apollo_model.ckpt":     None,  # base model   (feature_dim=256)
    "apollo_model_uni.ckpt": None,  # universal     (feature_dim=384)
    "pytorch_model.bin":     None,  # HF bin format (feature_dim=256)
}

# Data preparation — runs before training, skips gracefully if already done

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


def _read_lame_delay(mp3_path: str) -> int:
    """Read encoder delay from the LAME/Xing header of an MP3 file.
    Returns the delay in samples, or -1 if not found.
    """
    try:
        from mutagen.mp3 import MP3
        audio = MP3(mp3_path)
        # LAME tag lives in the Xing/Info frame as encoder_delay
        if hasattr(audio, 'info') and hasattr(audio.info, 'encoder_delay'):
            delay = audio.info.encoder_delay
            if delay is not None and delay >= 0:
                return int(delay)
    except Exception:
        pass
    return -1


def _align_pair(lq: "torch.Tensor", hq: "torch.Tensor", stem: str, lq_path: str = "", fixed_delay: int = None) -> tuple:
    """Align LQ to HQ using a fixed sample offset.

    If fixed_delay is set: trim that many samples from LQ (positive) or HQ (negative).
    Otherwise: auto-detect via LAME header, with xcorr fallback.
    """
    import torch
    import torch.nn.functional as F

    if fixed_delay is not None:
        delay = fixed_delay
        print_only(f"[align] {stem}: fixed delay = {delay} samples")
    else:
        delay = -1

        # Primary: read LAME tag from MP3 header
        if lq_path.lower().endswith(".mp3"):
            delay = _read_lame_delay(lq_path)
            if delay >= 0:
                print_only(f"[align] {stem}: LAME header delay = {delay} samples — trimmed LQ")

        # Fallback: small-window pattern xcorr
        if delay < 0:
            PATTERN_SIZE  = 2048
            SEARCH_WINDOW = 8192
            hq_pat = hq[0, :PATTERN_SIZE].double()
            lq_win = lq[0, :SEARCH_WINDOW].double()
            hq_pat = hq_pat / (hq_pat.std() + 1e-8)
            lq_win = lq_win / (lq_win.std() + 1e-8)
            corr = F.conv1d(
                lq_win.view(1, 1, -1),
                hq_pat.flip(0).view(1, 1, -1),
                padding=0
            ).squeeze()
            delay = int(corr.argmax().item())
            print_only(f"[align] {stem}: no LAME tag — xcorr delay = {delay} samples — trimmed LQ")

    if delay > 0:
        lq = lq[:, delay:]
    elif delay < 0:
        hq = hq[:, abs(delay):]

    min_len = min(lq.shape[-1], hq.shape[-1])
    return lq[:, :min_len], hq[:, :min_len]

def _save_chunk_16bit(tensor, path: str):
    """Save a chunk as 16-bit PCM WAV regardless of input dtype."""
    import torchaudio
    # Clamp to [-1, 1] then convert to int16 range
    pcm = tensor.float().clamp(-1.0, 1.0)
    torchaudio.save(path, pcm, _SR, encoding="PCM_S", bits_per_sample=16)


def _slice_and_save(
    lq_wav, hq_wav, stem: str, lq_out: str, hq_out: str,
    cached_aug_fn=None, variants: int = 1,
) -> list:
    """Slice a pair into overlapping chunks, optionally apply cached augmentations,
    save all variants as 16-bit PCM WAV. Returns list of written filenames."""
    min_len = min(lq_wav.shape[-1], hq_wav.shape[-1])
    lq_wav  = lq_wav[:, :min_len]
    hq_wav  = hq_wav[:, :min_len]

    saved = []
    start = 0
    idx   = 0
    while start + _CHUNK_SAMPLES <= min_len:
        hq_chunk = hq_wav[:, start:start + _CHUNK_SAMPLES]
        lq_chunk = lq_wav[:, start:start + _CHUNK_SAMPLES]

        # Variant 0 is always the clean (unaugmented) chunk
        fname = f"{stem}_{idx:04d}.wav"
        _save_chunk_16bit(lq_chunk, os.path.join(lq_out, fname))
        _save_chunk_16bit(hq_chunk, os.path.join(hq_out, fname))
        saved.append(fname)

        # Additional augmented variants (variant index 1..N)
        if cached_aug_fn is not None:
            for v in range(1, variants + 1):
                lq_aug, hq_aug = cached_aug_fn(lq_chunk.clone(), hq_chunk.clone())
                vname = f"{stem}_{idx:04d}_v{v}.wav"
                _save_chunk_16bit(lq_aug, os.path.join(lq_out, vname))
                _save_chunk_16bit(hq_aug, os.path.join(hq_out, vname))
                saved.append(vname)

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

    Layout A — subfolder pairs (existing):
        src_root/song1_LQ/   song1_HQ/
        src_root/song2_LQ/   song2_HQ/
        → moves audio files from each _LQ/_HQ subdir into src_root/LQ/ + src_root/HQ/

    Layout B — flat postfix files (new shortcut):
        src_root/song1_LQ.wav   song1_HQ.wav
        src_root/song2_LQ.flac  song2_HQ.flac
        → moves files into src_root/LQ/ + src_root/HQ/, stripping the _LQ/_HQ suffix

    Layout C — already normalized (LQ/ and HQ/ subdirs exist):
        src_root/LQ/   src_root/HQ/
        → nothing to do

    After normalization src_root always looks like:
        src_root/LQ/<stem>.wav ...
        src_root/HQ/<stem>.wav ...
    """
    import shutil

    if not os.path.isdir(src_root):
        return False

    lq_dir = os.path.join(src_root, "LQ")
    hq_dir = os.path.join(src_root, "HQ")

    # Layout C — already normalized, nothing to do
    if os.path.isdir(lq_dir) and os.path.isdir(hq_dir):
        lq_files = [f for f in os.listdir(lq_dir) if os.path.splitext(f)[1].lower() in _SUPPORTED_EXTS]
        hq_files = [f for f in os.listdir(hq_dir) if os.path.splitext(f)[1].lower() in _SUPPORTED_EXTS]
        if lq_files and hq_files:
            print_only(f"[data/{split_name}] LQ/ + HQ/ already present — skipping normalization.")
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

    # Move Layout A: song_LQ/ → LQ/<stem_from_dir>.wav (files keep their own names)
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
            pass  # not empty (e.g. had subdirs or other files) — leave it
        print_only(f"[data/{split_name}]   normalized dir pair: {stem}")

    # Move Layout B: song_LQ.wav → LQ/song.wav  (strip postfix from filename)
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
    mc  = getattr(cached_cfg, "mono_channel",     {})

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
        mono_channel=SimpleAugCfg(
            enabled=_bool(mc, "enabled", False),
            prob=   _frac(mc, "fraction", 1.0),
        ),
    )

    sr = int(getattr(cfg.datas, "sr", 44100))

    def _apply(lq, hq):
        return augment_pair(lq, hq, aug_cfg, sr=sr)

    return _apply

def _chunk_split(src_root: str, dst_root: str, split_name: str, cached_aug_fn=None, variants: int = 1, align: bool = True, fixed_delay: int = None) -> int:
    """
    Normalize src_root into LQ/ + HQ/ layout (if not already), then chunk all
    matched pairs into dst_root/LQ and dst_root/HQ.
    Returns the number of chunk pairs written. Skips if already chunked.

    Accepted src_root layouts (all auto-detected and normalized):
        A)  src_root/<song>_LQ/  <song>_HQ/     — subdirectory pairs
        B)  src_root/<song>_LQ.wav  <song>_HQ.wav  — flat postfix files
        C)  src_root/LQ/  src_root/HQ/           — already normalized
    """
    lq_out = os.path.join(dst_root, "LQ")
    hq_out = os.path.join(dst_root, "HQ")

    # Skip if already chunked
    if os.path.isdir(lq_out) and any(f.endswith(".wav") for f in os.listdir(lq_out)):
        n = sum(1 for f in os.listdir(lq_out) if f.endswith(".wav"))
        print_only(f"[data/{split_name}] Already chunked ({n} pairs) — skipping.")
        return n

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
    print_only(f"[data/{split_name}] Chunking {len(matched)} pairs → {dst_root}")
    print_only(f"[data/{split_name}] {sep}\n")

    total = 0
    for stem in matched:
        lq_path = os.path.join(lq_src, lq_files[stem])
        hq_path = os.path.join(hq_src, hq_files[stem])
        # Fixed delay: pass as frame_offset at decode time so the decoder skips
        # the samples rather than decoding the whole file then trimming after.
        lq_offset = fixed_delay if (align and fixed_delay is not None and fixed_delay > 0) else 0
        hq_offset = (-fixed_delay) if (align and fixed_delay is not None and fixed_delay < 0) else 0
        lq_wav = _load_wav_stereo(lq_path, frame_offset=lq_offset)
        hq_wav = _load_wav_stereo(hq_path, frame_offset=hq_offset)
        if align and fixed_delay is None and (os.path.splitext(lq_files[stem])[1].lower() != ".wav"
                or os.path.splitext(hq_files[stem])[1].lower() != ".wav"):
            lq_wav, hq_wav = _align_pair(lq_wav, hq_wav, stem, lq_path=lq_path, fixed_delay=None)
        saved  = _slice_and_save(lq_wav, hq_wav, stem, lq_out, hq_out,
                                  cached_aug_fn=cached_aug_fn, variants=variants)
        print_only(f"[data/{split_name}]   {stem}: {len(saved)} chunks")
        total += len(saved)

    print_only(f"[data/{split_name}] Done — {total} chunk pairs → {dst_root}\n")
    return total

def prepare_data(cfg: DictConfig) -> None:
    """
    Auto-preprocessing pipeline called before training.

    Accepts any of these layouts under data/train/ and data/val/:
        A)  <song>_LQ/  <song>_HQ/        — subdirectory pairs
        B)  <song>_LQ.wav  <song>_HQ.wav  — flat postfix files
        C)  LQ/  HQ/                       — already normalized

    Layouts A and B are automatically reorganized into LQ/ + HQ/ in-place,
    then chunked into:
        chunks/train/LQ/  chunks/train/HQ/
        chunks/val/LQ/    chunks/val/HQ/

    Skips any split that is already chunked.
    """
    # Anchor to the directory train.py lives in, not CWD.
    _script_dir  = os.path.dirname(os.path.abspath(__file__))
    train_chunks = os.path.join(_script_dir, cfg.datas.train_dir)
    val_chunks   = os.path.join(_script_dir, cfg.datas.eval_dir)

    # Data source dirs live under data/<name>/train and data/<name>/val
    data_root  = os.path.join(_script_dir, "data", cfg.exp.name)
    data_train = os.path.join(data_root, "train")
    data_val   = os.path.join(data_root, "val")

    cached_aug_fn = _build_cached_aug_fn(cfg)
    variants      = int(getattr(getattr(cfg.datas, "augmentation", {}), "cached_variants", 1)
                        if hasattr(getattr(cfg.datas, "augmentation", None) or {}, "cached_variants")
                        else 1)
    _align_raw    = getattr(cfg.datas, "align_data", True)
    if isinstance(_align_raw, int) and not isinstance(_align_raw, bool):
        align       = True
        fixed_delay = int(_align_raw)
    elif _align_raw is True or _align_raw == "true":
        align       = True
        fixed_delay = None
    else:
        align       = False
        fixed_delay = None

    _chunk_split(data_train, train_chunks, "train", cached_aug_fn=cached_aug_fn, variants=variants, align=align, fixed_delay=fixed_delay)
    _chunk_split(data_val,   val_chunks,   "val",   cached_aug_fn=None,          variants=1,        align=align, fixed_delay=fixed_delay)

    # Val bootstrap from train chunks
    # If val is still empty after chunking (no data/val source exists),
    # copy a random selection of train chunks into val — without removing
    # them from training. Chunks are picked by randomly selecting songs first,
    # then random chunks from those songs, so val covers diverse source material.
    val_lq = os.path.join(val_chunks, "LQ")
    val_hq = os.path.join(val_chunks, "HQ")
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
            print_only("[data/val] No train chunks available for val bootstrap — skipping.")
        else:
            # Build stem → [chunk filenames] map grouped by song
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
    model (feature_dim=384) with an 11 GB card — only layers 4-5 and the
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

def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    # Apply hardware / compiler optimisations declared in cfg.optimizations
    apply_optimizations(cfg)

    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    # Auto-preprocess raw data and bootstrap eval if needed
    prepare_data(cfg)

    # Verify chunks exist — if data/ was empty, provide a clear error
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_lq   = os.path.join(script_dir, cfg.datas.train_dir, "LQ")
    if not os.path.isdir(train_lq) or not any(f.endswith(".wav") for f in os.listdir(train_lq)):
        _name = cfg.exp.name
        print_only("")
        print_only("ERROR: No training chunks found.")
        print_only("")
        print_only(f"  Populate data/{_name}/train/ and data/{_name}/val/ with paired audio files:")
        print_only(f"    data/{_name}/train/LQ/   ← degraded audio (MP3, FLAC, WAV)")
        print_only(f"    data/{_name}/train/HQ/   ← clean reference (same filenames)")
        print_only(f"    data/{_name}/val/LQ/")
        print_only(f"    data/{_name}/val/HQ/")
        print_only("")
        print_only("  Or drop _LQ/_HQ files directly in data/ and they will be moved automatically.")
        print_only("")
        raise SystemExit(1)

    # Instantiate datamodule
    print_only(f"Instantiating datamodule <{cfg.datas._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datas)

    # Resolve run directory — fresh start gets a timestamped subfolder,
    # resume reuses the most recent existing run folder.
    from datetime import datetime as _dt
    _base_dir = os.path.join(cfg.exp.dir, cfg.exp.name)
    ckpt_path = None

    if cfg.get("resume", False):
        # Find the most recently modified run folder that has checkpoints
        _run_dir = None
        if os.path.isdir(_base_dir):
            _subdirs = [
                os.path.join(_base_dir, d)
                for d in os.listdir(_base_dir)
                if os.path.isdir(os.path.join(_base_dir, d, "checkpoints"))
            ]
            if _subdirs:
                _run_dir = max(_subdirs, key=os.path.getmtime)
        if _run_dir is None:
            # No existing runs — fresh start
            _run_id = _dt.now().strftime("%Y%m%d_%H%M%S")
            _run_dir = os.path.join(_base_dir, _run_id)
            print_only(f"[resume] No existing runs found — starting fresh run: {_run_id}")
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
                print_only("[resume] Skipping pretrain weight loading — checkpoint takes precedence.")
            else:
                print_only(f"[resume] Run folder found but no checkpoints — starting from pretrained weights.")
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
        # Resuming — Lightning will restore all weights from the checkpoint.
        # Just instantiate a bare model so the system can be constructed;
        # the state dict will be overwritten by trainer.fit(ckpt_path=...).
        print_only("[weights] Resume mode — skipping pretrain load, instantiating bare model.")
        model = look2hear.models.apollo.Apollo(
            sr=44100, win=20, feature_dim=feature_dim, layer=6
        )
    else:
        local_path = cfg.get("weights_path", None)

        if not local_path:
            # Auto-scan ./models/
            os.makedirs(_MODELS_DIR, exist_ok=True)
            for fname, url in _PRETRAINED_MODELS.items():
                candidate = os.path.join(_MODELS_DIR, fname)
                if os.path.isfile(candidate):
                    print_only(f"[weights] Found pretrained model in models/: {fname}")
                    local_path = candidate
                    break
                elif url is not None:
                    print_only(f"[weights] Downloading {fname} from configured URL...")
                    import urllib.request
                    os.makedirs(_MODELS_DIR, exist_ok=True)
                    urllib.request.urlretrieve(url, candidate)
                    print_only(f"[weights] Saved to {candidate}")
                    local_path = candidate
                    break

        if local_path:
            print_only(f"[weights] Loading from: {local_path}")
            model = _load_weights(local_path, feature_dim)
            print_only("[weights] Weights loaded.")
        else:
            # Final fallback — HuggingFace hub
            print_only("[weights] No local model found in models/ — downloading from HuggingFace...")
            from huggingface_hub import hf_hub_download
            weights_path = hf_hub_download(
                repo_id="JusperLee/Apollo",
                filename="pytorch_model.bin",
            )
            print_only(f"[weights] Cached at: {weights_path}")
            model = look2hear.models.BaseModel.from_pretrain(
                weights_path, sr=44100, win=20, feature_dim=feature_dim, layer=6
            )
            print_only("[weights] Pretrained weights loaded.")

    # Freeze early layers for fine-tuning — count driven by config
    # (no-op when resuming since frozen params are restored by the checkpoint too)
    freeze_early_layers(model, n_layers_to_freeze=cfg.training.n_layers_to_freeze)

    # Instantiate discriminator fresh — learns your artifact type from scratch
    print_only(f"Instantiating Discriminator <{cfg.discriminator._target_}>")
    discriminator = hydra.utils.instantiate(cfg.discriminator)

    # Instantiate optimizers
    print_only(f"Instantiating optimizers")
    opt_cfg = cfg.optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    # Optimizer factory
    # Reads cfg.optimizer.type to select the optimizer:
    #   adamw       — standard 32-bit AdamW
    #   adamw_8bit  — 8-bit AdamW via bitsandbytes (pip install bitsandbytes)
    #   cpu_offload — 32-bit AdamW with momentum states in CPU RAM
    def _make_optimizer(params, lr, weight_decay, betas):
        opt_type = opt_cfg.get("type", "adamw").lower()

        if opt_type == "adamw_8bit":
            try:
                import bitsandbytes as bnb
                opt = bnb.optim.AdamW8bit(params, lr=lr, weight_decay=weight_decay, betas=betas)
                print_only(f"[optimizer] AdamW8bit (bitsandbytes) — lr={lr}")
                return opt
            except ImportError:
                print_only("[optimizer] bitsandbytes not installed — falling back to AdamW32bit")

        if opt_type == "cpu_offload":
            try:
                opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas, fused=False)
                print_only(f"[optimizer] AdamW32bit with CPU offload — lr={lr}")
                return opt
            except Exception as e:
                print_only(f"[optimizer] CPU offload failed ({e}), falling back to AdamW32bit")

        opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas)
        print_only(f"[optimizer] AdamW32bit — lr={lr}")
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
    loss_g = hydra.utils.instantiate(cfg.loss_g, hf_boost=cfg.training.hf_boost)
    loss_d = hydra.utils.instantiate(cfg.loss_d)
    losses = {"g": loss_g, "d": loss_d}

    # Instantiate metrics
    print_only(f"Instantiating metrics <{cfg.metrics._target_}>")
    metrics = hydra.utils.instantiate(cfg.metrics)

    # Instantiate system
    print_only(f"Instantiating system <{cfg.system._target_}>")
    val_audio_dir = os.path.join(_run_dir, "val_audio")
    system: LightningModule = hydra.utils.instantiate(
        cfg.system,
        model=model,
        discriminator=discriminator,
        loss_func=losses,
        metrics=metrics,
        optimizer=[optimizer_g, optimizer_d],
        scheduler=[scheduler_g, scheduler_d],
        val_audio_dir=val_audio_dir,
        val_audio_pairs=cfg.training.val_audio_pairs,
        gradient_checkpointing=cfg.system.get("gradient_checkpointing", False),
        grad_accum_steps=cfg.training.get("grad_accum_steps", 1),
    )

    # Callbacks
    # Patch all run-specific paths before any instantiation
    os.makedirs(os.path.join(_run_dir, "logs"), exist_ok=True)
    with open_dict(cfg):
        cfg.checkpoint.dirpath       = os.path.join(_run_dir, "checkpoints")
        cfg.logger.save_dir          = os.path.join(_run_dir, "logs")
        cfg.trainer.default_root_dir = _run_dir

    callbacks: List[Callback] = []

    if cfg.get("early_stopping"):
        print_only(f"Instantiating early_stopping")
        callbacks.append(hydra.utils.instantiate(cfg.early_stopping))
    if cfg.get("checkpoint"):
        print_only(f"Instantiating checkpoint")
        checkpoint = hydra.utils.instantiate(cfg.checkpoint)
        callbacks.append(checkpoint)

    # Instantiate logger
    print_only(f"Instantiating logger <{cfg.logger._target_}>")
    logger = hydra.utils.instantiate(cfg.logger)
    logger.log_hyperparams = lambda *a, **kw: None

    # Instantiate trainer — single GPU, no DDP
    print_only(f"Instantiating trainer")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=logger,
    )

    def _save_and_exit(sig=None, frame=None):
        print_only("\n[interrupt] Ctrl+C caught — saving checkpoint...")
        try:
            ckpt_dir = os.path.join(_run_dir, "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            step = trainer.global_step
            try:
                val_loss = trainer.callback_metrics.get("val_loss", None)
                loss_str = f"-val_loss={val_loss:.4f}" if val_loss is not None else ""
            except Exception:
                loss_str = ""
            out_path = os.path.join(ckpt_dir, f"{step:06d}{loss_str}.ckpt")
            trainer.save_checkpoint(out_path)
            print_only(f"[interrupt] Saved to {out_path}")
        except Exception as e:
            print_only(f"[interrupt] Save failed: {e}")
        os._exit(0)

    import signal as _signal
    _signal.signal(_signal.SIGINT, _save_and_exit)

    try:
        trainer.fit(system, datamodule=datamodule, ckpt_path=ckpt_path)
    except torch.cuda.OutOfMemoryError as e:
        print_only(f"\n[OOM] CUDA out of memory — exiting cleanly. Try reducing batch_size or num_workers.")
        print_only(f"[OOM] {e}")
        torch.cuda.empty_cache()
        os._exit(1)
    except MemoryError as e:
        print_only(f"\n[OOM] System RAM exhausted — exiting cleanly.")
        print_only(f"[OOM] {e}")
        os._exit(1)
    print_only("Training finished!")

    best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
    with open(os.path.join(_run_dir, "best_k_models.json"), "w") as f:
        json.dump(best_k, f, indent=0)

    state_dict = torch.load(checkpoint.best_model_path)
    system.load_state_dict(state_dict=state_dict["state_dict"])
    system.cpu()

    to_save = system.audio_model.serialize()
    torch.save(to_save, os.path.join(_run_dir, "best_model.pth"))

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
    # e.g. configs/apollo_sftl.yaml → "apollo_sftl"
    _conf_stem = os.path.splitext(os.path.basename(args.conf_dir))[0]
    if not cfg.exp.get("name"):
        cfg.exp.name = _conf_stem

    # Derive data and chunk paths from exp.name if not explicitly set in config.
    # data/<name>/train, data/<name>/val, chunks/<name>/train, chunks/<name>/val
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _data_name_root   = os.path.join(_script_dir, "data",   cfg.exp.name)
    _chunks_name_root = os.path.join(_script_dir, "chunks", cfg.exp.name)

    _default_train_dir = os.path.join("chunks", cfg.exp.name, "train")
    _default_eval_dir  = os.path.join("chunks", cfg.exp.name, "val")
    _generic = {"./chunks/train", "chunks/train", "./chunks/val", "chunks/val"}
    if not cfg.datas.get("train_dir") or cfg.datas.train_dir in _generic:
        cfg.datas.train_dir = _default_train_dir
    if not cfg.datas.get("eval_dir") or cfg.datas.eval_dir in _generic:
        cfg.datas.eval_dir = _default_eval_dir

    # If loose _LQ/_HQ files exist in data/ root, move them into data/<name>/
    _data_root = os.path.join(_script_dir, "data")
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
            print(f"[autodiscovery] Moved {len(_loose)} loose file(s) from data/ → data/{cfg.exp.name}/")

    print(f"[autodiscovery] name={cfg.exp.name}  data=data/{cfg.exp.name}  chunks=chunks/{cfg.exp.name}")
    # --- End autodiscovery ---

    if args.weights_path:
        cfg.weights_path = args.weights_path
    if args.resume:
        cfg.resume = True

    os.makedirs(os.path.join(cfg.exp.dir, cfg.exp.name), exist_ok=True)

    train(cfg)
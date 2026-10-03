"""
compression.py -- codec chain sampling, encoding and alignment.

Shared by utils/degrade_audio.py (--randomize) and the cached `compression`
augmentation in core/train.py. A recipe is the `random` block of a JSON file in
utils/degrade/.

Step types:
    wma_encode   {bitrate}
    mp3_lame     {quality} or {bitrate}
    codec        {codec: aac|vorbis|opus|mp2, bitrate}
"""

from __future__ import annotations

import random
import shutil
import subprocess
from pathlib import Path

import numpy as np


class CompressionError(RuntimeError):
    pass


_CODEC_ARGS = {
    "aac":    ("aac",        "m4a"),
    "vorbis": ("libvorbis",  "ogg"),
    "opus":   ("libopus",    "opus"),
    "mp2":    ("mp2",        "mp2"),
}
_ENCODER_FOR_TYPE = {"wma_encode": "wmav2", "mp3_lame": "libmp3lame"}


def bitrate_arg(value) -> str:
    """ffmpeg reads a bare number as bits per second, and libmp3lame then falls back to
    128 kbps. Plain numbers in recipes mean kbps."""
    s = str(value).strip().lower()
    return s + "k" if s.isdigit() else s


def wpick(rng: random.Random, weights: dict) -> str:
    keys = list(weights)
    return rng.choices(keys, weights=[weights[k] for k in keys])[0]


def sample_chain(rc: dict, rng: random.Random) -> list[dict]:
    """Draw one chain from the 'random' block of a recipe."""
    def mp3_pass() -> dict:
        if rng.random() < rc["mp3_vbr_prob"]:
            return {"type": "mp3_lame", "quality": int(wpick(rng, rc["mp3_vbr_quality"]))}
        return {"type": "mp3_lame", "bitrate": int(wpick(rng, rc["mp3_cbr_bitrate"]))}

    def wma() -> dict:
        return {"type": "wma_encode", "bitrate": wpick(rng, rc["wma_bitrate"])}

    chain = [mp3_pass() for _ in range(int(wpick(rng, rc["mp3_passes"])))]
    final = wpick(rng, rc["final"])
    if final == "cbr192":
        chain.append({"type": "mp3_lame", "bitrate": 192})
    elif final == "cbr192_extra":
        chain.append({"type": "mp3_lame", "bitrate": 192})
        chain.append({"type": "mp3_lame", "bitrate": int(wpick(rng, rc["final_extra_bitrate"]))})
    elif final == "vbr":
        chain.append({"type": "mp3_lame", "quality": int(wpick(rng, rc["mp3_vbr_quality"]))})

    pos = wpick(rng, rc["wma_position"])
    if pos == "first":
        chain.insert(0, wma())
    elif pos == "last":
        chain.append(wma())
    elif pos == "middle" and len(chain) >= 2:
        chain.insert(rng.randint(1, len(chain) - 1), wma())
    elif pos == "double":
        chain.insert(0, wma())
        chain.append(wma())

    oc = rc.get("other_codecs")
    if oc and rng.random() < float(oc.get("prob", 0.0)):
        for _ in range(int(wpick(rng, oc.get("count", {"1": 1})))):
            step = {"type": "codec", "codec": wpick(rng, oc["codec"]),
                    "bitrate": wpick(rng, oc["bitrate"])}
            chain.insert(rng.randint(0, len(chain)), step)
    return chain


def describe_chain(chain: list[dict]) -> str:
    parts = []
    for s in chain:
        if s["type"] == "wma_encode":
            parts.append("wma" + str(s["bitrate"]).rstrip("k"))
        elif s["type"] == "codec":
            parts.append(f"{s['codec']}{str(s['bitrate']).rstrip('k')}")
        elif "quality" in s:
            parts.append(f"q{s['quality']}")
        else:
            parts.append(f"c{s['bitrate']}")
    return ">".join(parts)


def recipe_encoders(rc: dict) -> set[str]:
    """ffmpeg encoder names a recipe can draw."""
    need = {"libmp3lame"}
    if rc.get("wma_position") and any(w > 0 for k, w in rc["wma_position"].items() if k != "none"):
        need.add("wmav2")
    oc = rc.get("other_codecs")
    if oc and float(oc.get("prob", 0.0)) > 0:
        need |= {_CODEC_ARGS[c][0] for c, w in oc["codec"].items() if w > 0}
    return need


def check_encoders(ffmpeg: str, rc: dict) -> None:
    """Raise if ffmpeg is missing or lacks an encoder the recipe can draw."""
    try:
        out = subprocess.run([ffmpeg, "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    except OSError as ex:
        raise CompressionError(f"ffmpeg not found ({ffmpeg}): {ex}")
    missing = sorted(e for e in recipe_encoders(rc) if f" {e} " not in out)
    if missing:
        raise CompressionError(f"ffmpeg has no encoder for: {', '.join(missing)}")


def _run(cmd: list, log: list) -> None:
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if r.returncode != 0:
        log.append(r.stderr[-300:])
        raise CompressionError(f"{Path(str(cmd[0])).name} exit {r.returncode}")


def _encode_step(step: dict, src: Path, tmpdir: Path, n: int, ffmpeg: str, log: list) -> Path:
    t = step["type"]
    if t == "wma_encode":
        out = tmpdir / f"gen{n}.wma"
        cmd = [ffmpeg, "-y", "-i", src, "-vn", "-c:a", "wmav2", "-b:a", bitrate_arg(step["bitrate"]), out]
    elif t == "mp3_lame":
        out = tmpdir / f"gen{n}.mp3"
        cmd = [ffmpeg, "-y", "-i", src, "-vn", "-c:a", "libmp3lame"]
        if "quality" in step:
            cmd += ["-q:a", str(step["quality"])]
        else:
            cmd += ["-b:a", bitrate_arg(step["bitrate"])]
        cmd.append(out)
    elif t == "codec":
        enc, ext = _CODEC_ARGS[step["codec"]]
        out = tmpdir / f"gen{n}.{ext}"
        cmd = [ffmpeg, "-y", "-i", src, "-vn", "-c:a", enc, "-b:a", bitrate_arg(step["bitrate"]), out]
    else:
        raise CompressionError(f"unknown step type: {t}")
    _run(cmd, log)
    return out


def encode_chain(chain: list[dict], src_wav: Path, tmpdir: Path, ffmpeg: str, sr: int, log: list) -> Path:
    """Run the chain on src_wav and return a WAV at sr. Each step encodes the previous
    step's decoded output."""
    current = src_wav
    for i, step in enumerate(chain, start=1):
        current = _encode_step(step, current, tmpdir, i, ffmpeg, log)
    out = tmpdir / "final.wav"
    _run([ffmpeg, "-y", "-i", current, "-vn", "-ar", sr, "-c:a", "pcm_s16le", out], log)
    return out


def measure_offset(ref: np.ndarray, test: np.ndarray, sr: int, max_lag: int) -> tuple[int, float]:
    """Integer-sample delay of test relative to ref on the 0-4 kHz band, plus the
    correlation coefficient at that lag. lag > 0 means test is late. Arrays are [n, ch]."""
    from scipy.signal import butter, sosfiltfilt

    sos = butter(4, 4000, btype="low", fs=sr, output="sos")
    a = sosfiltfilt(sos, ref.mean(axis=1))
    b = sosfiltfilt(sos, test.mean(axis=1))
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    nfft = 1 << (2 * n - 1).bit_length()
    R = np.fft.irfft(np.fft.rfft(b, nfft) * np.conj(np.fft.rfft(a, nfft)), nfft)
    lags = np.arange(-max_lag, max_lag + 1)
    r = R[lags % nfft]
    i = int(np.argmax(r))
    norm = float(np.sqrt(np.dot(a, a) * np.dot(b, b))) + 1e-12
    return int(lags[i]), float(r[i] / norm)


def shift_to_length(x: np.ndarray, lag: int, start: int, length: int) -> np.ndarray:
    """Return x[start+lag : start+lag+length], zero-filled where that runs off either end."""
    out = np.zeros((length, x.shape[1]), dtype=x.dtype)
    s = start + lag
    lo, hi = max(s, 0), min(s + length, len(x))
    if hi > lo:
        out[lo - s: hi - s] = x[lo:hi]
    return out


def degrade_window(window: np.ndarray, pad_start: int, core_len: int, sr: int, chain: list[dict],
                   ffmpeg: str, tmpdir: Path, max_lag: int = 16384, min_corr: float = 0.5,
                   silence_dbfs: float = -60.0) -> tuple[np.ndarray | None, dict]:
    """Run chain on a padded window and return the degraded core, aligned sample-for-sample
    to window[pad_start : pad_start + core_len]. Returns (None, info) when the core is
    near-silent or the alignment is weak. info has status, lag and corr."""
    import soundfile as sf

    info: dict = {"chain": describe_chain(chain), "status": "ok"}
    core = window[pad_start: pad_start + core_len]
    if len(core) == 0 or 20 * np.log10(np.sqrt(np.mean(core.astype(np.float64) ** 2)) + 1e-12) < silence_dbfs:
        info["status"] = "rejected: silent"
        return None, info

    log: list[str] = []
    tmpdir.mkdir(parents=True, exist_ok=True)
    try:
        src = tmpdir / "seg.wav"
        sf.write(str(src), window, sr, subtype="PCM_24")
        out = encode_chain(chain, src, tmpdir, ffmpeg, sr, log)
        lq, _ = sf.read(str(out), always_2d=True, dtype="float64")
    except CompressionError as ex:
        info["status"] = f"error: {ex} {' | '.join(log[-2:])}"
        return None, info
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    lag, corr = measure_offset(window.astype(np.float64), lq, sr, max_lag)
    info["lag"], info["corr"] = lag, round(corr, 4)
    if corr < min_corr:
        info["status"] = "rejected: weak alignment"
        return None, info
    return shift_to_length(lq, lag, pad_start, core_len).astype(np.float32), info

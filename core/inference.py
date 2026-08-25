# @claude last-modified: 2026-08-24T00:00:00Z
# @claude last-commit: feat: spectral merge ensemble -- low_end_preserve, custom bands, aux checkpoint
"""
inference.py -- Apollo audio enhancement script

Usage:
    # Auto-select best checkpoint from run folder
    python inference.py --in_wav input.wav --out_wav output.wav --conf_dir configs/apollo_stfl.yaml

    # Explicit checkpoint
    python inference.py --in_wav input.wav --out_wav output.wav \\
        --weights runs/apollo_stfl/20260819/checkpoints/step=001200-val_loss=-28.10.ckpt \\
        --conf_dir configs/apollo_stfl.yaml

    # Pretrained shortnames (no conf_dir needed)
    python inference.py --in_wav input.wav --out_wav output.wav --weights lew_v2

    # Low-end preservation (max_fft blend below 700 Hz)
    python inference.py --in_wav input.wav --out_wav output.wav --weights my.ckpt \\
        --conf_dir configs/apollo_stfl.yaml --low_end_preserve

    # Custom band ensemble (JSON list of band specs)
    python inference.py --in_wav input.wav --out_wav output.wav --weights my.ckpt \\
        --conf_dir configs/apollo_stfl.yaml \\
        --ensemble '[{"lo":0,"hi":700,"mode":"max_fft","weight":1.0},{"lo":15000,"hi":22050,"mode":"avg","weight":0.7}]'

    # Dual-checkpoint ensemble (primary + aux with per-band routing)
    python inference.py --in_wav input.wav --out_wav output.wav --weights primary.ckpt \\
        --conf_dir configs/apollo_stfl.yaml \\
        --aux_weights secondary.ckpt --aux_conf_dir configs/apollo_stfl2.yaml \\
        --aux_ensemble '[{"lo":8000,"hi":22050,"mode":"max_fft","weight":0.5}]'
"""
import argparse
import os

import torch
import torchaudio
import look2hear.models
import look2hear.models.apollo

_SR          = 44100  # Apollo's native sample rate
_CHUNK_SEC   = 4      # default chunk size -- matches training segment_sec
_OVERLAP_SEC = 0.5    # crossfade overlap at chunk boundaries

# Default ensemble applied when --low_end_preserve is set.
# max_fft below 700 Hz: take whichever of original or enhanced has more energy
# per bin. Preserves the original low-end character while keeping all restored
# HF content from the model.
_LOW_END_BANDS_DEFAULT = [
    {"lo": 0, "hi": 700, "mode": "max_fft", "weight": 1.0},
]


def _spectral_merge(original: "torch.Tensor", enhanced: "torch.Tensor",
                    sr: int, bands: list) -> "torch.Tensor":
    """Blend original and enhanced audio in the STFT domain according to band specs.

    Args:
        original:  [2, T] float32 -- un-normalized source audio
        enhanced:  [2, T] float32 -- model output (same scale as original)
        sr:        sample rate
        bands:     list of dicts, each with keys:
                     lo    -- low frequency Hz (inclusive)
                     hi    -- high frequency Hz (inclusive)
                     mode  -- 'max_fft' | 'min_fft' | 'avg' | 'original' | 'enhanced'
                     weight -- float 0-1; blends between mode result (1.0) and pure
                               enhanced (0.0). Default 1.0.

    Returns:
        [2, T] float32 merged tensor.

    Phase always comes from the enhanced output. Only magnitude is blended.
    STFT window size 4096 gives ~0.09 Hz/bin resolution at 44100 Hz.
    Bins not covered by any band default to enhanced-only.
    """
    import numpy as np

    n_fft    = 4096
    hop      = n_fft // 4
    window   = torch.hann_window(n_fft)

    def _stft(x):
        # x: [2, T] -> [2, F, frames] complex
        return torch.stft(x, n_fft=n_fft, hop_length=hop,
                          win_length=n_fft, window=window,
                          return_complex=True, pad_mode="reflect")

    def _istft(X, length):
        return torch.istft(X, n_fft=n_fft, hop_length=hop,
                           win_length=n_fft, window=window,
                           length=length, return_complex=False)

    T = enhanced.shape[-1]
    # torch.stft reflect-padding requires at least n_fft samples; pad short chunks
    pad = max(0, n_fft - T)
    if pad > 0:
        enhanced = torch.nn.functional.pad(enhanced, (0, pad))
        original = torch.nn.functional.pad(original, (0, pad))
    S_orig = _stft(original[..., :enhanced.shape[-1]])  # [2, F, frames]
    S_enh  = _stft(enhanced)                              # [2, F, frames]

    mag_orig = S_orig.abs()
    mag_enh  = S_enh.abs()
    phase    = S_enh / (mag_enh + 1e-8)  # unit-phase from enhanced

    # Start with enhanced magnitude everywhere
    mag_out = mag_enh.clone()

    # Hz -> bin index
    bin_hz = sr / n_fft  # Hz per FFT bin
    n_bins = n_fft // 2 + 1

    for band in bands:
        lo_hz = float(band.get("lo", 0))
        hi_hz = float(band.get("hi", sr / 2))
        mode  = str(band.get("mode", "enhanced")).lower()
        w     = float(band.get("weight", 1.0))
        w     = max(0.0, min(1.0, w))

        lo_bin = max(0, int(lo_hz / bin_hz))
        hi_bin = min(n_bins - 1, int(hi_hz / bin_hz) + 1)
        if lo_bin >= hi_bin:
            continue

        m_o = mag_orig[:, lo_bin:hi_bin, :]
        m_e = mag_enh[:, lo_bin:hi_bin, :]

        if mode == "max_fft":
            m_blend = torch.maximum(m_o, m_e)
        elif mode == "min_fft":
            m_blend = torch.minimum(m_o, m_e)
        elif mode == "avg":
            m_blend = (m_o + m_e) * 0.5
        elif mode == "original":
            m_blend = m_o
        else:  # "enhanced" or unknown
            m_blend = m_e

        # weight blends between pure enhanced (0) and the mode result (1)
        mag_out[:, lo_bin:hi_bin, :] = w * m_blend + (1.0 - w) * m_e

    # Reconstruct with enhanced phase
    S_out = mag_out * phase
    out   = _istft(S_out, length=T)
    return out

_REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODELS_DIR = os.path.join(_REPO_ROOT, "models")
_CACHE_DIR  = os.path.join(_REPO_ROOT, "usr", "cache")

KNOWN_MODELS = {
    "apollo": (
        "pytorch_model.bin",
        "https://huggingface.co/JusperLee/Apollo/resolve/main/pytorch_model.bin",
        256,
    ),
    "lew": (
        "apollo_model.ckpt",
        "https://huggingface.co/jarredou/lew_apollo_vocal_enhancer/resolve/main/apollo_model.ckpt",
        256,
    ),
    "lew_v2": (
        "apollo_model_v2.ckpt",
        "https://huggingface.co/jarredou/lew_apollo_vocal_enhancer/resolve/main/apollo_model_v2.ckpt",
        256,
    ),
    "lew_uni": (
        "apollo_model_uni.ckpt",
        "https://github.com/deton24/Lew-s-vocal-enhancer-for-Apollo-by-JusperLee/releases/download/uni/apollo_model_uni.ckpt",
        384,
    ),
}


def load_config(conf_path):
    """Load a yaml config and return the OmegaConf DictConfig."""
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(conf_path)
    print(f"[inference] Loaded config: {conf_path}")
    return cfg


def ensure_model(shortname: str) -> tuple:
    if shortname not in KNOWN_MODELS:
        raise ValueError(f"Unknown model '{shortname}'. Known: {list(KNOWN_MODELS)}")
    filename, url, feature_dim = KNOWN_MODELS[shortname]
    os.makedirs(_MODELS_DIR, exist_ok=True)
    dest = os.path.join(_MODELS_DIR, filename)
    if not os.path.exists(dest):
        print(f"[inference] Downloading {shortname} -> models/{filename}")
        import urllib.request
        def _progress(count, block, total):
            if total > 0:
                pct = min(100, count * block * 100 // total)
                print(f"\r[inference] {pct:3d}%", end="", flush=True)
        urllib.request.urlretrieve(url, dest, reporthook=_progress)
        print()
        print(f"[inference] Saved -> {dest}")
    else:
        print(f"[inference] Found cached model: models/{filename}")
    return dest, feature_dim


def _ensure_wav(file_path: str, target_sr: int = _SR) -> str:
    """If file_path is not a WAV, decode it to usr/cache/<md5>.wav and return that path.
    The original file is never modified. WAV files are returned as-is.
    Cache hit = instant return with no re-decode.
    """
    if os.path.splitext(file_path)[1].lower() == ".wav":
        return file_path
    import hashlib, tempfile
    h = hashlib.md5()
    with open(file_path, "rb") as _f:
        for _block in iter(lambda: _f.read(1 << 20), b""):
            h.update(_block)
    os.makedirs(_CACHE_DIR, exist_ok=True)
    cached = os.path.join(_CACHE_DIR, f"{h.hexdigest()}.wav")
    if os.path.isfile(cached):
        print(f"[inference] WAV cache hit: {os.path.basename(cached)}")
        return cached
    try:
        import ffmpeg as _ffmpeg
    except ImportError:
        return file_path  # ffmpeg-python not installed; fall back to torchaudio.load
    fd, tmp_path = tempfile.mkstemp(suffix=".wav", dir=_CACHE_DIR)
    os.close(fd)
    try:
        (
            _ffmpeg
            .input(file_path)
            .output(tmp_path, format="wav", acodec="pcm_f32le", ar=target_sr, ac=2)
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
    except _ffmpeg.Error as e:
        os.unlink(tmp_path)
        stderr = e.stderr.decode(errors="replace") if e.stderr else ""
        raise RuntimeError(f"[inference] ffmpeg conversion failed for {file_path}:\n{stderr}") from e
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
    os.replace(tmp_path, cached)
    print(f"[inference] Cached WAV: {os.path.basename(cached)}  ({os.path.basename(file_path)})")
    return cached


def load_audio(file_path: str, target_sr: int = _SR) -> torch.Tensor:
    file_path = _ensure_wav(file_path, target_sr)
    audio, sr = torchaudio.load(file_path)
    if audio.shape[-1] == 0:
        raise ValueError(f"Audio file contains no samples: {file_path}")
    if not torch.isfinite(audio).all():
        raise ValueError(f"Input audio contains NaN or Inf values: {file_path}")
    if sr != target_sr:
        print(f"[inference] Resampling {sr} Hz -> {target_sr} Hz")
        audio = torchaudio.functional.resample(audio, sr, target_sr)
    if audio.shape[0] == 1:
        audio = audio.repeat(2, 1)
    elif audio.shape[0] > 2:
        audio = audio[:2]
    return audio.unsqueeze(0)  # [1, 2, T]


def save_audio(file_path: str, audio: torch.Tensor, sr: int = _SR) -> None:
    if audio.ndim == 3:
        audio = audio.squeeze(0)
    audio = audio.float().cpu()
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    torchaudio.save(file_path, audio, sr, encoding="PCM_F", bits_per_sample=32)
    print(f"[inference] Saved -> {file_path}")


def _load_ckpt(path: str, feature_dim: int, sr: int, win: int, layer: int):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    raw = ckpt["state_dict"]
    if any(k.startswith("audio_model.") for k in raw):
        state = {k.replace("audio_model.", ""): v
                 for k, v in raw.items() if k.startswith("audio_model.")}
        print("[inference] Lightning checkpoint -- stripped audio_model. prefix")
    else:
        state = raw
        print("[inference] Bare state dict -- no prefix stripping needed")
    model = look2hear.models.apollo.Apollo(
        sr=sr, win=win, feature_dim=feature_dim, layer=layer
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[inference] Missing keys ({len(missing)}): {missing[:3]}{'...' if len(missing) > 3 else ''}")
    if unexpected:
        print(f"[inference] Unexpected keys ({len(unexpected)}): {unexpected[:3]}{'...' if len(unexpected) > 3 else ''}")
    if not missing:
        print("[inference] All keys loaded successfully")
    return model


def load_model(weights, sr, win, feature_dim, layer):
    if weights and weights.strip().lower() in KNOWN_MODELS:
        local_path, registry_dim = ensure_model(weights.strip().lower())
        if feature_dim == 256 and registry_dim != 256:
            print(f"[inference] Auto-setting feature_dim={registry_dim} for {weights}")
            feature_dim = registry_dim
        weights = local_path

    if not weights or weights.strip().lower() in ("", "jusperlee/apollo"):
        print("[inference] Loading from HuggingFace hub: JusperLee/Apollo")
        model = look2hear.models.BaseModel.from_pretrain(
            "JusperLee/Apollo", sr=sr, win=win, feature_dim=feature_dim, layer=layer
        )
    elif weights.endswith(".ckpt"):
        print(f"[inference] Loading Lightning checkpoint: {weights}")
        model = _load_ckpt(weights, feature_dim=feature_dim, sr=sr, win=win, layer=layer)
    else:
        print(f"[inference] Loading serialized model: {weights}")
        model = look2hear.models.BaseModel.from_pretrain(
            weights, sr=sr, win=win, feature_dim=feature_dim, layer=layer
        )
    return model


def _run_chunked(model, audio, device, sr, chunk_sec, overlap_sec, out_path,
                 bands=None, aux_model=None, aux_bands=None):
    """Process audio in chunks and write each chunk to disk immediately.

    Writes sequentially so the output WAV can be previewed in Audacity
    while inference is still running -- just drag the file in and hit play.

    bands:     list of band specs for _spectral_merge (blends original + enhanced).
               None = enhanced-only (no merge).
    aux_model: optional second Apollo model for dual-checkpoint ensemble.
    aux_bands: band specs applied to the aux model output relative to the
               primary enhanced signal. None = same as bands.

    Returns None (output already on disk).
    """
    import soundfile as sf
    import numpy as np

    # Keep a copy of the un-normalized original for spectral merge.
    original = audio.squeeze(0).clone()  # [2, T]

    # Normalize to match training: divide by peak so the model sees [-1, 1] input.
    # Store the scale so we can restore the original level after inference.
    peak = audio.abs().max().item()
    if peak > 0:
        audio = audio / peak
        original_norm = original / peak
    else:
        peak = 1.0
        original_norm = original

    if chunk_sec <= 0:
        raise ValueError(f"chunk_sec must be > 0, got {chunk_sec}")
    if overlap_sec < 0:
        raise ValueError(f"overlap_sec must be >= 0, got {overlap_sec}")
    if overlap_sec >= chunk_sec:
        raise ValueError(
            f"overlap_sec ({overlap_sec}s) must be smaller than chunk_sec ({chunk_sec}s)"
        )

    chunk_samples   = int(round(chunk_sec * sr))
    overlap_samples = int(round(overlap_sec * sr))
    hop_samples     = chunk_samples - overlap_samples

    if hop_samples <= 0:
        raise ValueError(
            f"Invalid chunk geometry: chunk_samples={chunk_samples}, "
            f"overlap_samples={overlap_samples}, hop={hop_samples}"
        )

    T         = audio.shape[-1]
    prev_tail = None  # holds the overlap region from the previous chunk

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    # 32-bit float WAV: no integer ceiling, no clipping at +/-1.0.
    with sf.SoundFile(out_path, mode="w", samplerate=sr, channels=2,
                      subtype="FLOAT") as f:
        start     = 0
        chunk_idx = 0
        while start < T:
            end   = min(start + chunk_samples, T)
            chunk = audio[..., start:end].to(device)

            with torch.no_grad():
                enhanced = model(chunk)
            # enhanced: [1, 2, T_chunk] -- restore original level after inference
            enhanced = enhanced.squeeze(0).cpu() * peak  # [2, T_chunk]

            # Spectral merge with original (primary bands)
            if bands:
                orig_chunk = original_norm[..., start:end] * peak
                enhanced = _spectral_merge(orig_chunk, enhanced, sr, bands)

            # Dual-checkpoint: blend aux model output into specified bands
            if aux_model is not None:
                with torch.no_grad():
                    aux_out = aux_model(chunk)
                aux_out = aux_out.squeeze(0).cpu() * peak
                _aux_bands = aux_bands if aux_bands else bands
                if _aux_bands:
                    enhanced = _spectral_merge(aux_out, enhanced, sr, _aux_bands)

            chunk_len = enhanced.shape[-1]

            if prev_tail is not None and overlap_samples > 0:
                fade_len = min(overlap_samples, chunk_len, prev_tail.shape[-1])
                fade_in  = torch.linspace(0.0, 1.0, fade_len)
                fade_out = 1.0 - fade_in

                # crossfade zone -- blend previous tail with current head
                blended = prev_tail[..., :fade_len] * fade_out \
                        + enhanced[..., :fade_len]  * fade_in

                # write blended overlap
                f.write(blended.T.numpy())
                # write remainder of this chunk (excluding the next overlap tail)
                write_end = chunk_len - overlap_samples
                if write_end > fade_len:
                    f.write(enhanced[..., fade_len:write_end].T.numpy())
            else:
                # first chunk -- write everything except the tail we'll crossfade next time
                write_end = chunk_len - overlap_samples if (T - end) > 0 else chunk_len
                write_end = max(write_end, 0)
                if write_end > 0:
                    f.write(enhanced[..., :write_end].T.numpy())

            # keep tail for next chunk's crossfade (or flush if last chunk)
            if end < T:
                tail_start = max(0, chunk_len - overlap_samples)
                prev_tail  = enhanced[..., tail_start:]
            else:
                # last chunk -- flush any remaining tail
                if prev_tail is not None and overlap_samples > 0:
                    tail_start = max(0, chunk_len - overlap_samples)
                    f.write(enhanced[..., tail_start:].T.numpy())
                elif prev_tail is None:
                    # single chunk, no overlap
                    pass
                prev_tail = None

            chunk_idx += 1
            print(f"[inference] Chunk {chunk_idx} done  "
                  f"({start/sr:.1f}s - {end/sr:.1f}s)  -> {out_path}")
            start += hop_samples

    return None  # output already written


def _find_best_checkpoint(conf_dir: str) -> str:
    cfg = load_config(conf_dir)
    exp = cfg.get("exp", {})
    run_dir = os.path.join(
        str(exp.get("dir", "./runs")),
        str(exp.get("name", os.path.splitext(os.path.basename(conf_dir))[0])),
    )
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"No run directory found at {run_dir!r}")

    # Find all checkpoints across all timestamped subfolders
    candidates = []
    for ts_dir in os.listdir(run_dir):
        ckpt_dir = os.path.join(run_dir, ts_dir, "checkpoints")
        if os.path.isdir(ckpt_dir):
            for f in os.listdir(ckpt_dir):
                if f.endswith(".ckpt"):
                    candidates.append(os.path.join(ckpt_dir, f))

    if not candidates:
        raise FileNotFoundError(f"No checkpoints found under {run_dir!r}")

    def _parse_loss(path):
        name = os.path.basename(path)
        try:
            return float(name.split("val_loss=")[1].replace(".ckpt", ""))
        except (IndexError, ValueError):
            return float("inf")

    best = min(candidates, key=_parse_loss)
    print(f"[inference] Auto-selected checkpoint: {os.path.relpath(best)}")
    return best


def main(
    input_wav,
    output_wav,
    weights=None,
    conf_dir=None,
    sr=_SR,
    win=20,
    feature_dim=256,
    layer=6,
    chunk_sec=_CHUNK_SEC,
    overlap_sec=_OVERLAP_SEC,
    device_str="auto",
    chunked=True,
    low_end_preserve=False,
    low_end_hz=700.0,
    ensemble=None,
    aux_weights=None,
    aux_conf_dir=None,
    aux_ensemble=None,
):
    """
    ensemble: list of band dicts (already parsed from JSON), or None.
    If low_end_preserve is True and ensemble is None, _LOW_END_BANDS_DEFAULT
    (with low_end_hz as the crossover) is used.
    aux_weights/aux_conf_dir: optional second model for dual-checkpoint blend.
    aux_ensemble: band specs for blending the aux model output. None = same as primary bands.
    """
    # Config fills only args the user didn't explicitly provide (still None).
    # Explicit CLI args always win -- config is a fallback, not an override.
    if conf_dir is not None:
        cfg = load_config(conf_dir)
        m = cfg.get("model", {})
        d = cfg.get("datas", {})
        if sr          is None and "sr"          in m: sr          = int(m.sr)
        if win         is None and "win"         in m: win         = int(m.win)
        if feature_dim is None and "feature_dim" in m: feature_dim = int(m.feature_dim)
        if layer       is None and "layer"       in m: layer       = int(m.layer)
        if chunk_sec   is None and "segment_sec" in d: chunk_sec   = float(d.segment_sec)

    # Apply hardcoded defaults for anything still unset
    if sr          is None: sr          = _SR
    if win         is None: win         = 20
    if feature_dim is None: feature_dim = 256
    if layer       is None: layer       = 6
    if chunk_sec   is None: chunk_sec   = _CHUNK_SEC

    # Resolve bands
    if ensemble is None and low_end_preserve:
        ensemble = [{"lo": 0, "hi": float(low_end_hz), "mode": "max_fft", "weight": 1.0}]

    if ensemble:
        print(f"[inference] Spectral merge bands: {ensemble}")

    print(f"[inference] Config: feature_dim={feature_dim}, sr={sr}, win={win}, "
          f"layer={layer}, chunk_sec={chunk_sec}")

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"[inference] Device: {device}")

    if weights is None and conf_dir is not None:
        weights = _find_best_checkpoint(conf_dir)

    model = load_model(weights=weights, sr=sr, win=win, feature_dim=feature_dim, layer=layer)
    model = model.to(device).eval()

    # Aux model (optional second checkpoint for dual-checkpoint ensemble)
    aux_model = None
    if aux_weights:
        print(f"[inference] Loading aux model: {aux_weights}")
        aux_sr = sr; aux_win = win; aux_fd = feature_dim; aux_layer = layer
        if aux_conf_dir is not None:
            aux_cfg = load_config(aux_conf_dir)
            am = aux_cfg.get("model", {})
            if "sr"          in am: aux_sr    = int(am.sr)
            if "win"         in am: aux_win   = int(am.win)
            if "feature_dim" in am: aux_fd    = int(am.feature_dim)
            if "layer"       in am: aux_layer = int(am.layer)
        aux_model = load_model(weights=aux_weights, sr=aux_sr, win=aux_win,
                               feature_dim=aux_fd, layer=aux_layer)
        aux_model = aux_model.to(device).eval()
        if aux_ensemble:
            print(f"[inference] Aux ensemble bands: {aux_ensemble}")

    audio = load_audio(input_wav, target_sr=sr)
    duration = audio.shape[-1] / sr
    print(f"[inference] Input: {input_wav}  ({duration:.1f}s, {audio.shape[-2]}ch)")

    if chunked and audio.shape[-1] > int(chunk_sec * sr):
        print(f"[inference] Chunked inference ({chunk_sec}s chunks, {overlap_sec}s overlap)")
        print(f"[inference] Writing sequentially -- you can preview in Audacity now")
        _run_chunked(model, audio, device, sr, chunk_sec, overlap_sec, out_path=output_wav,
                     bands=ensemble, aux_model=aux_model, aux_bands=aux_ensemble)
        print(f"[inference] Done -> {output_wav}")
    else:
        original = audio.squeeze(0).clone()
        peak = audio.abs().max().item()
        if peak > 0:
            audio = audio / peak
        with torch.no_grad():
            enhanced = model(audio.to(device))
        enhanced = enhanced.squeeze(0).cpu() * peak
        if ensemble:
            enhanced = _spectral_merge(original, enhanced, sr, ensemble)
        if aux_model is not None:
            with torch.no_grad():
                aux_out = aux_model(audio.to(device))
            aux_out = aux_out.squeeze(0).cpu() * peak
            _ab = aux_ensemble if aux_ensemble else ensemble
            if _ab:
                enhanced = _spectral_merge(aux_out, enhanced, sr, _ab)
        save_audio(output_wav, enhanced.unsqueeze(0), sr=sr)


if __name__ == "__main__":
    import json as _json
    parser = argparse.ArgumentParser(description="Apollo audio enhancement")
    parser.add_argument("--in_wav",      type=str, required=True,
                        help="Path to input audio file")
    parser.add_argument("--out_wav",     type=str, required=True,
                        help="Path to save enhanced output")
    parser.add_argument("--weights",     type=str, default=None,
                        help="Model weights: shortname (lew/lew_v2/lew_uni), local .pth/.bin/.ckpt path, "
                             "or omit to auto-select the best checkpoint from the run folder (requires --conf_dir)")
    parser.add_argument("--conf_dir",    type=str, default=None,
                        help="Path to training yaml config. Reads feature_dim, sr, win, "
                             "layer, and segment_sec from it. Explicit CLI flags override.")
    parser.add_argument("--sr",          type=int, default=None,
                        help=f"Sample rate override (default: from config or {_SR})")
    parser.add_argument("--win",         type=int, default=None,
                        help="STFT window size in ms override (default: from config or 20)")
    parser.add_argument("--feature_dim", type=int, default=None,
                        help="Feature dim override (default: from config or 256)")
    parser.add_argument("--layer",       type=int, default=None,
                        help="BSNet layer count override (default: from config or 6)")
    parser.add_argument("--chunk_sec",   type=float, default=None,
                        help=f"Chunk size in seconds override (default: from config segment_sec or {_CHUNK_SEC})")
    parser.add_argument("--overlap_sec", type=float, default=_OVERLAP_SEC,
                        help=f"Crossfade overlap in seconds (default: {_OVERLAP_SEC})")
    parser.add_argument("--device",      type=str, default="auto",
                        help="'auto', 'cuda', 'cpu', 'cuda:1', ... (default: auto)")
    parser.add_argument("--no_chunked",  action="store_true",
                        help="Disable chunked inference (may OOM on long files)")
    # Ensemble / spectral merge flags
    parser.add_argument("--low_end_preserve", action="store_true",
                        help="Blend original below --low_end_hz using max_fft (preserves low-end "
                             "character and original codec artifacts in that range)")
    parser.add_argument("--low_end_hz", type=float, default=700.0,
                        help="Crossover frequency for --low_end_preserve (default: 700 Hz)")
    parser.add_argument("--ensemble",   type=str, default=None,
                        help='JSON list of band specs, e.g. \'[{"lo":0,"hi":700,"mode":"max_fft","weight":1.0}]\'. '
                             'Modes: max_fft, min_fft, avg, original, enhanced. '
                             'weight: 0=pure enhanced, 1=pure mode result.')
    parser.add_argument("--aux_weights",   type=str, default=None,
                        help="Optional second checkpoint for dual-checkpoint ensemble")
    parser.add_argument("--aux_conf_dir",  type=str, default=None,
                        help="Config for the aux checkpoint (reads feature_dim, sr, etc.)")
    parser.add_argument("--aux_ensemble",  type=str, default=None,
                        help="JSON band specs for blending the aux model output. "
                             "Omit to use the same bands as --ensemble.")
    args = parser.parse_args()

    ensemble_parsed     = _json.loads(args.ensemble)    if args.ensemble    else None
    aux_ensemble_parsed = _json.loads(args.aux_ensemble) if args.aux_ensemble else None

    # Pass None for args the user didn't explicitly set so main() can fill
    # them from --conf_dir without clobbering explicit CLI values.
    main(
        input_wav=args.in_wav,
        output_wav=args.out_wav,
        weights=args.weights,
        conf_dir=args.conf_dir,
        sr=args.sr,
        win=args.win,
        feature_dim=args.feature_dim,
        layer=args.layer,
        chunk_sec=args.chunk_sec,
        overlap_sec=args.overlap_sec,
        device_str=args.device,
        chunked=not args.no_chunked,
        low_end_preserve=args.low_end_preserve,
        low_end_hz=args.low_end_hz,
        ensemble=ensemble_parsed,
        aux_weights=args.aux_weights,
        aux_conf_dir=args.aux_conf_dir,
        aux_ensemble=aux_ensemble_parsed,
    )

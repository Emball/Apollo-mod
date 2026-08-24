# @claude last-modified: 2026-08-24T00:00:00Z
# @claude last-commit: feat: spectral ensemble engine with low-end preservation and custom band control
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

    # Low-end preservation (default crossover: 700 Hz, max_fft blend below)
    python inference.py --in_wav input.wav --out_wav output.wav --weights lew_v2 --low_end_preserve

    # Custom crossover frequency
    python inference.py --in_wav input.wav --out_wav output.wav --weights lew_v2 --low_end_preserve --low_end_hz 500

    # Full custom spectral ensemble (JSON band list)
    # Each band: {"lo": HZ, "hi": HZ, "mode": "max_fft"|"min_fft"|"avg"|"enhanced"|"original", "weight": 0.0-1.0}
    # weight controls blend between mode result (1.0) and enhanced-only (0.0). Default 1.0.
    python inference.py --in_wav input.wav --out_wav output.wav --weights lew_v2 \\
        --ensemble '[{"lo":0,"hi":700,"mode":"max_fft","weight":1.0},{"lo":15000,"hi":22050,"mode":"avg","weight":0.5}]'

    # Multiple models at different frequency ranges
    python inference.py --in_wav input.wav --out_wav output.wav \\
        --weights runs/stfl/ckpt_A.ckpt --conf_dir configs/apollo_stfl.yaml \\
        --ensemble '[{"lo":700,"hi":22050,"mode":"enhanced","weight":1.0}]' \\
        --aux_weights runs/stfl2/ckpt_B.ckpt --aux_conf_dir configs/apollo_stfl2.yaml \\
        --aux_ensemble '[{"lo":15000,"hi":22050,"mode":"max_fft","weight":0.8}]'
"""
import argparse
import json
import os

import torch
import torchaudio
import look2hear.models
import look2hear.models.apollo

_SR          = 44100  # Apollo's native sample rate
_CHUNK_SEC   = 4      # default chunk size -- matches training segment_sec
_OVERLAP_SEC = 0.5    # crossfade overlap at chunk boundaries

_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# ---------------------------------------------------------------------------
# Spectral ensemble engine
# ---------------------------------------------------------------------------
# Bands are defined as a list of dicts:
#   {"lo": hz, "hi": hz, "mode": str, "weight": float}
#
# Modes (applied between original and enhanced within the band):
#   "max_fft"  -- take the bin-wise maximum magnitude (keeps best energy)
#   "min_fft"  -- take the bin-wise minimum magnitude (most conservative)
#   "avg"      -- arithmetic average of original and enhanced magnitudes
#   "enhanced" -- enhanced only (no blend with original)
#   "original" -- original only (bypass the model for this band)
#
# weight (0.0-1.0): blend between mode result (1.0) and enhanced-only (0.0).
#   weight=1.0 means the mode result is used as-is.
#   weight=0.5 means halfway between the mode result and enhanced-only.
#   This lets you dial in how strongly the original signal influences each band.
#
# Bands not covered by any spec default to enhanced-only (model output).
# Overlapping bands are not supported -- last definition wins for a given bin.
# ---------------------------------------------------------------------------

_DEFAULT_LOW_END_HZ = 700  # Apollo struggles below this; preserve original by default

def _hz_to_bin(hz: float, n_fft: int, sr: int) -> int:
    """Convert a frequency in Hz to the nearest rfft bin index."""
    return int(round(hz * n_fft / sr))


def _spectral_merge(
    original: torch.Tensor,
    enhanced: torch.Tensor,
    sr: int,
    bands: list,
    n_fft: int = 4096,
) -> torch.Tensor:
    """
    Merge original and enhanced audio in the frequency domain per band spec.

    Args:
        original:  [C, T] float32 -- unprocessed input at original amplitude
        enhanced:  [C, T] float32 -- model output at original amplitude
        sr:        sample rate
        bands:     list of band spec dicts (lo, hi, mode, weight)
        n_fft:     FFT size (larger = finer frequency resolution)

    Returns:
        [C, T] float32 merged audio
    """
    import numpy as np

    if not bands:
        return enhanced

    # Pad both signals to the same length (should already match, but be safe)
    T = max(original.shape[-1], enhanced.shape[-1])
    if original.shape[-1] < T:
        original = torch.nn.functional.pad(original, (0, T - original.shape[-1]))
    if enhanced.shape[-1] < T:
        enhanced = torch.nn.functional.pad(enhanced, (0, T - enhanced.shape[-1]))

    # Process each channel independently -- keeps stereo image intact
    out_channels = []
    for c in range(original.shape[0]):
        orig_ch = original[c].numpy().astype(np.float64)
        enha_ch = enhanced[c].numpy().astype(np.float64)

        # Overlap-add STFT merge to avoid block-boundary artifacts.
        # We work in the STFT domain directly for fine bin control.
        hop = n_fft // 4
        win = np.hanning(n_fft)

        # Split into overlapping frames
        def _frames(sig):
            n_frames = 1 + (len(sig) - n_fft) // hop
            frames = []
            for i in range(n_frames):
                frame = sig[i * hop: i * hop + n_fft]
                if len(frame) < n_fft:
                    frame = np.pad(frame, (0, n_fft - len(frame)))
                frames.append(frame * win)
            return frames

        orig_frames = _frames(orig_ch)
        enha_frames = _frames(enha_ch)

        if not orig_frames:
            out_channels.append(enhanced[c])
            continue

        n_bins = n_fft // 2 + 1

        # Build a bin-to-mode map (default: enhanced)
        # mode_map[bin] = (mode_str, weight_float)
        mode_map = [("enhanced", 1.0)] * n_bins
        for band in bands:
            lo_bin = max(0,        _hz_to_bin(float(band.get("lo", 0)),    n_fft, sr))
            hi_bin = min(n_bins-1, _hz_to_bin(float(band.get("hi", sr/2)), n_fft, sr))
            mode   = str(band.get("mode", "enhanced"))
            weight = float(band.get("weight", 1.0))
            for b in range(lo_bin, hi_bin + 1):
                mode_map[b] = (mode, weight)

        # Process each frame
        merged_frames = []
        for orig_frame, enha_frame in zip(orig_frames, enha_frames):
            O = np.fft.rfft(orig_frame)
            E = np.fft.rfft(enha_frame)

            O_mag = np.abs(O)
            E_mag = np.abs(E)
            E_phase = np.angle(E)  # always use enhanced phase

            M_mag = E_mag.copy()  # default: enhanced

            for b, (mode, weight) in enumerate(mode_map):
                if mode == "enhanced":
                    blended = E_mag[b]
                elif mode == "original":
                    blended = O_mag[b]
                elif mode == "max_fft":
                    blended = max(O_mag[b], E_mag[b])
                elif mode == "min_fft":
                    blended = min(O_mag[b], E_mag[b])
                elif mode == "avg":
                    blended = (O_mag[b] + E_mag[b]) * 0.5
                else:
                    blended = E_mag[b]

                # weight blends between mode result and pure enhanced
                M_mag[b] = weight * blended + (1.0 - weight) * E_mag[b]

            # Reconstruct with enhanced phase
            M = M_mag * np.exp(1j * E_phase)
            merged_frame = np.fft.irfft(M)
            merged_frames.append(merged_frame)

        # Overlap-add reconstruction
        out = np.zeros(T + n_fft, dtype=np.float64)
        norm = np.zeros(T + n_fft, dtype=np.float64)
        for i, frame in enumerate(merged_frames):
            start = i * hop
            out[start: start + n_fft]  += frame * win
            norm[start: start + n_fft] += win ** 2

        # Normalize overlap-add window
        norm = np.maximum(norm, 1e-8)
        out /= norm
        out = out[:T]

        out_channels.append(torch.from_numpy(out.astype(np.float32)))

    return torch.stack(out_channels, dim=0)


def _build_default_bands(low_end_hz: float, sr: int) -> list:
    """
    Default ensemble: max_fft below low_end_hz, enhanced above.
    Apollo struggles with sub-700 Hz content; preserving original energy
    there avoids the model introducing artifacts in the low end.
    """
    nyquist = sr / 2.0
    return [
        {"lo": 0.0,          "hi": low_end_hz, "mode": "max_fft",  "weight": 1.0},
        {"lo": low_end_hz,   "hi": nyquist,    "mode": "enhanced", "weight": 1.0},
    ]

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


def load_audio(file_path: str, target_sr: int = _SR) -> torch.Tensor:
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


def _run_chunked(
    model,
    audio,
    device,
    sr,
    chunk_sec,
    overlap_sec,
    out_path,
    ensemble_bands=None,
    extra_models=None,
):
    """Process audio in chunks and write each chunk to disk immediately.

    Writes sequentially so the output WAV can be previewed in Audacity
    while inference is still running -- just drag the file in and hit play.

    Args:
        model:          primary Apollo model (already on device, eval mode)
        audio:          [1, 2, T] float32 input at original amplitude
        device:         torch.device
        sr:             sample rate
        chunk_sec:      chunk length in seconds
        overlap_sec:    crossfade overlap in seconds
        out_path:       output file path
        ensemble_bands: list of band spec dicts for primary model spectral merge,
                        or None to skip merging (enhanced-only output)
        extra_models:   list of (model, bands) pairs for auxiliary models whose
                        outputs are merged into the primary result at their bands

    Returns None (output already on disk).
    """
    import soundfile as sf
    import numpy as np

    # Keep original audio at full scale for spectral merge reference.
    # Normalize by LQ peak for model input -- joint-peak normalization isn't
    # possible at inference time since HQ isn't available, so we use LQ peak
    # and rely on the model's scale invariance.
    audio_orig = audio.squeeze(0)  # [2, T] -- kept at original amplitude
    peak = audio_orig.abs().max().item()
    if peak > 0:
        audio_norm = audio_orig / peak
    else:
        audio_norm = audio_orig
        peak = 1.0

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

    T         = audio_norm.shape[-1]
    prev_tail = None  # holds the overlap region from the previous chunk

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    doing_ensemble = bool(ensemble_bands) or bool(extra_models)
    if doing_ensemble:
        print(f"[inference] Spectral ensemble active")
        if ensemble_bands:
            for b in ensemble_bands:
                print(f"[inference]   primary band [{b.get('lo',0):.0f}-{b.get('hi',sr//2):.0f} Hz] "
                      f"mode={b.get('mode','enhanced')} weight={b.get('weight',1.0):.2f}")
        if extra_models:
            for i, (_, ebands) in enumerate(extra_models):
                for b in ebands:
                    print(f"[inference]   aux[{i}] band [{b.get('lo',0):.0f}-{b.get('hi',sr//2):.0f} Hz] "
                          f"mode={b.get('mode','enhanced')} weight={b.get('weight',1.0):.2f}")

    # 32-bit float WAV: no integer ceiling, no clipping at +/-1.0.
    with sf.SoundFile(out_path, mode="w", samplerate=sr, channels=2,
                      subtype="FLOAT") as f:
        start     = 0
        chunk_idx = 0
        while start < T:
            end         = min(start + chunk_samples, T)
            chunk_norm  = audio_norm[..., start:end].unsqueeze(0).to(device)  # [1,2,T]
            chunk_orig  = audio_orig[..., start:end]  # [2,T] at original amplitude

            with torch.no_grad():
                enhanced = model(chunk_norm)
            enhanced = enhanced.squeeze(0).cpu() * peak  # [2, T_chunk] at original amplitude

            # --- Spectral merge ---
            if ensemble_bands:
                enhanced = _spectral_merge(chunk_orig, enhanced, sr, ensemble_bands)

            # Apply auxiliary models at their specified bands
            if extra_models:
                for aux_model, aux_bands in extra_models:
                    with torch.no_grad():
                        aux_out = aux_model(chunk_norm)
                    aux_out = aux_out.squeeze(0).cpu() * peak
                    # Merge auxiliary output into the current result using aux_bands
                    # We pass the original chunk as "original" and aux_out as "enhanced"
                    # so the band modes (max_fft etc.) compare original vs aux
                    aux_merged = _spectral_merge(chunk_orig, aux_out, sr, aux_bands)
                    # Fold aux_merged into enhanced: for each aux band, replace bins in enhanced
                    # We re-run a final merge: enhanced is "original", aux_merged is "enhanced"
                    # with mode=enhanced at aux bands -- effectively a selective paste
                    paste_bands = []
                    for b in aux_bands:
                        paste_bands.append({
                            "lo":     b.get("lo", 0),
                            "hi":     b.get("hi", sr / 2),
                            "mode":   "enhanced",  # take from aux_merged (the "enhanced" arg)
                            "weight": b.get("weight", 1.0),
                        })
                    enhanced = _spectral_merge(enhanced, aux_merged, sr, paste_bands)

            chunk_len = enhanced.shape[-1]

            if prev_tail is not None and overlap_samples > 0:
                fade_len = min(overlap_samples, chunk_len, prev_tail.shape[-1])
                fade_in  = torch.linspace(0.0, 1.0, fade_len)
                fade_out = 1.0 - fade_in

                blended = prev_tail[..., :fade_len] * fade_out \
                        + enhanced[..., :fade_len]  * fade_in
                f.write(blended.T.numpy())

                write_end = chunk_len - overlap_samples
                if write_end > fade_len:
                    f.write(enhanced[..., fade_len:write_end].T.numpy())
            else:
                write_end = chunk_len - overlap_samples if (T - end) > 0 else chunk_len
                write_end = max(write_end, 0)
                if write_end > 0:
                    f.write(enhanced[..., :write_end].T.numpy())

            if end < T:
                tail_start = max(0, chunk_len - overlap_samples)
                prev_tail  = enhanced[..., tail_start:]
            else:
                if prev_tail is not None and overlap_samples > 0:
                    tail_start = max(0, chunk_len - overlap_samples)
                    f.write(enhanced[..., tail_start:].T.numpy())
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
    # Ensemble options
    low_end_preserve=False,
    low_end_hz=_DEFAULT_LOW_END_HZ,
    ensemble_json=None,
    # Auxiliary model ensemble
    aux_weights=None,
    aux_conf_dir=None,
    aux_feature_dim=None,
    aux_ensemble_json=None,
):
    """
    Run Apollo inference with optional spectral ensemble blending.

    Ensemble priority (highest wins):
      1. --ensemble JSON  -- full custom band spec for primary model
      2. --low_end_preserve  -- shortcut: max_fft below low_end_hz, enhanced above

    Aux model (--aux_weights / --aux_conf_dir) runs a second model and merges
    its output into the primary result at the bands defined by --aux_ensemble.
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

    # --- Resolve ensemble bands ---
    ensemble_bands = None
    if ensemble_json:
        try:
            ensemble_bands = json.loads(ensemble_json)
            if not isinstance(ensemble_bands, list):
                raise ValueError("--ensemble must be a JSON array")
            print(f"[inference] Custom ensemble: {len(ensemble_bands)} band(s)")
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Invalid --ensemble JSON: {e}") from e
    elif low_end_preserve:
        ensemble_bands = _build_default_bands(low_end_hz, sr)
        print(f"[inference] Low-end preserve: max_fft blend below {low_end_hz:.0f} Hz")

    # --- Resolve auxiliary model ---
    extra_models = []
    if aux_weights or aux_conf_dir:
        # Read aux model config
        aux_sr, aux_win, aux_fd, aux_layer = sr, win, feature_dim, layer
        if aux_conf_dir is not None:
            aux_cfg = load_config(aux_conf_dir)
            am = aux_cfg.get("model", {})
            ad = aux_cfg.get("datas", {})
            if "sr"          in am: aux_sr    = int(am.sr)
            if "win"         in am: aux_win   = int(am.win)
            if "feature_dim" in am: aux_fd    = int(am.feature_dim)
            if "layer"       in am: aux_layer = int(am.layer)
        if aux_feature_dim is not None:
            aux_fd = aux_feature_dim

        if aux_weights is None and aux_conf_dir is not None:
            aux_weights = _find_best_checkpoint(aux_conf_dir)

        print(f"[inference] Loading auxiliary model: {aux_weights}")
        aux_model = load_model(
            weights=aux_weights, sr=aux_sr, win=aux_win,
            feature_dim=aux_fd, layer=aux_layer,
        )
        aux_model = aux_model.to(device).eval()

        if aux_ensemble_json:
            try:
                aux_bands = json.loads(aux_ensemble_json)
                if not isinstance(aux_bands, list):
                    raise ValueError("--aux_ensemble must be a JSON array")
            except (json.JSONDecodeError, ValueError) as e:
                raise ValueError(f"Invalid --aux_ensemble JSON: {e}") from e
        else:
            # Default aux band: full range, enhanced mode (just adds the model output)
            aux_bands = [{"lo": 0, "hi": sr / 2, "mode": "enhanced", "weight": 1.0}]

        extra_models.append((aux_model, aux_bands))
        print(f"[inference] Auxiliary model: {len(aux_bands)} band(s)")

    audio = load_audio(input_wav, target_sr=sr)
    duration = audio.shape[-1] / sr
    print(f"[inference] Input: {input_wav}  ({duration:.1f}s, {audio.shape[-2]}ch)")

    if chunked and audio.shape[-1] > int(chunk_sec * sr):
        print(f"[inference] Chunked inference ({chunk_sec}s chunks, {overlap_sec}s overlap)")
        print(f"[inference] Writing sequentially -- you can preview in Audacity now")
        _run_chunked(
            model, audio, device, sr, chunk_sec, overlap_sec,
            out_path=output_wav,
            ensemble_bands=ensemble_bands,
            extra_models=extra_models if extra_models else None,
        )
        print(f"[inference] Done -> {output_wav}")
    else:
        # Short file / no-chunk path
        audio_sq = audio.squeeze(0)  # [2, T]
        peak = audio_sq.abs().max().item()
        audio_norm = audio_sq / peak if peak > 0 else audio_sq
        if peak == 0:
            peak = 1.0

        with torch.no_grad():
            enhanced = model(audio_norm.unsqueeze(0).to(device))
        enhanced = enhanced.squeeze(0).cpu() * peak  # [2, T]

        if ensemble_bands:
            enhanced = _spectral_merge(audio_sq, enhanced, sr, ensemble_bands)

        if extra_models:
            for aux_model, aux_bands in extra_models:
                with torch.no_grad():
                    aux_out = aux_model(audio_norm.unsqueeze(0).to(device))
                aux_out = aux_out.squeeze(0).cpu() * peak
                aux_merged = _spectral_merge(audio_sq, aux_out, sr, aux_bands)
                paste_bands = [
                    {"lo": b.get("lo", 0), "hi": b.get("hi", sr / 2),
                     "mode": "enhanced", "weight": b.get("weight", 1.0)}
                    for b in aux_bands
                ]
                enhanced = _spectral_merge(enhanced, aux_merged, sr, paste_bands)

        save_audio(output_wav, enhanced.unsqueeze(0), sr=sr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apollo audio enhancement")

    # --- Core ---
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

    # --- Spectral ensemble (primary model) ---
    parser.add_argument("--low_end_preserve", action="store_true",
                        help=f"Preserve low-end by blending original below --low_end_hz "
                             f"(max_fft mode). Default crossover: {_DEFAULT_LOW_END_HZ} Hz.")
    parser.add_argument("--low_end_hz", type=float, default=_DEFAULT_LOW_END_HZ,
                        help=f"Crossover frequency for --low_end_preserve (default: {_DEFAULT_LOW_END_HZ} Hz)")
    parser.add_argument("--ensemble",   type=str, default=None, dest="ensemble_json",
                        help='Full custom spectral ensemble as JSON array. Each element: '
                             '{"lo": HZ, "hi": HZ, "mode": "max_fft|min_fft|avg|enhanced|original", "weight": 0-1}. '
                             'Bins not covered default to enhanced-only. '
                             'Overrides --low_end_preserve when both are set.')

    # --- Auxiliary model ---
    parser.add_argument("--aux_weights",      type=str, default=None,
                        help="Second model checkpoint for multi-model ensemble")
    parser.add_argument("--aux_conf_dir",     type=str, default=None,
                        help="Config yaml for auxiliary model (reads feature_dim, sr, etc.)")
    parser.add_argument("--aux_feature_dim",  type=int, default=None,
                        help="Feature dim override for auxiliary model")
    parser.add_argument("--aux_ensemble",     type=str, default=None, dest="aux_ensemble_json",
                        help='Band spec for auxiliary model as JSON array (same format as --ensemble). '
                             'Defines which frequency ranges the aux model contributes to the final output. '
                             'Defaults to full range enhanced-only if omitted.')

    args = parser.parse_args()

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
        ensemble_json=args.ensemble_json,
        aux_weights=args.aux_weights,
        aux_conf_dir=args.aux_conf_dir,
        aux_feature_dim=args.aux_feature_dim,
        aux_ensemble_json=args.aux_ensemble_json,
    )

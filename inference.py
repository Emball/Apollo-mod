# @claude last-modified: 2026-05-05T06:34:39Z
# @claude last-commit: feat: major update — TUI, augmentation system, gradient checkpointing, optimization bootstrap
"""
inference.py — Apollo audio enhancement script

Model selection (--weights):
  apollo      Official Apollo base model (default, auto-downloads pytorch_model.bin)
  lew         Lew vocal enhancer v1 (auto-downloads apollo_model.ckpt)
  lew_v2      Lew vocal enhancer v2 (auto-downloads apollo_model_v2.ckpt)
  lew_uni     Lew universal model, feature_dim=384 (auto-downloads apollo_model_uni.ckpt)
  <path>      Local .pth / .bin / .ckpt file

All shortnames download into models/ and are reused on subsequent runs.

Usage:
    python inference.py --in_wav input.wav --out_wav output.wav
    python inference.py --in_wav input.wav --out_wav output.wav --weights lew_v2
    python inference.py --in_wav input.wav --out_wav output.wav --weights lew_uni
    python inference.py --in_wav input.wav --out_wav output.wav \
        --weights models/my_finetune.ckpt --feature_dim 256
"""

import argparse
import os

import torch
import torchaudio
import look2hear.models
import look2hear.models.apollo

_SR = 44100   # Apollo's native sample rate

_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# ─────────────────────────────────────────────────────────────────────────────
# Model registry — shortnames that auto-download into models/
# ─────────────────────────────────────────────────────────────────────────────

# Each entry: shortname -> (filename, url, feature_dim)
# feature_dim is returned alongside the path so the caller can set it
# automatically when the user hasn't specified one explicitly.
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


def ensure_model(shortname: str) -> tuple:
    """
    Given a shortname from KNOWN_MODELS, return (local_path, feature_dim).
    Downloads the file into models/ if it isn't already there.
    """
    if shortname not in KNOWN_MODELS:
        raise ValueError(
            f"Unknown model '{shortname}'. Known models: {list(KNOWN_MODELS)}"
        )
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
        print()  # newline after progress
        print(f"[inference] Saved -> {dest}")
    else:
        print(f"[inference] Found cached model: models/{filename}")
    return dest, feature_dim


# ─────────────────────────────────────────────────────────────────────────────
# Audio I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_audio(file_path: str, target_sr: int = _SR) -> torch.Tensor:
    """
    Load any supported audio file (WAV/FLAC/MP3/OGG/…) and return a
    [1, channels, samples] tensor resampled to target_sr.
    """
    audio, sr = torchaudio.load(file_path)   # [C, T]

    # Resample if needed
    if sr != target_sr:
        print(f"[inference] Resampling {sr} Hz -> {target_sr} Hz")
        audio = torchaudio.functional.resample(audio, sr, target_sr)

    # Ensure stereo (Apollo expects 2-channel input)
    if audio.shape[0] == 1:
        audio = audio.repeat(2, 1)
    elif audio.shape[0] > 2:
        audio = audio[:2]

    return audio.unsqueeze(0)   # [1, 2, T]


def save_audio(file_path: str, audio: torch.Tensor, sr: int = _SR) -> None:
    """Save [1, C, T] or [C, T] tensor to file."""
    if audio.ndim == 3:
        audio = audio.squeeze(0)    # [C, T]
    audio = audio.cpu().clamp(-1.0, 1.0)
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    torchaudio.save(file_path, audio, sr)
    print(f"[inference] Saved -> {file_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_ckpt(path: str, feature_dim: int, sr: int, win: int, layer: int):
    """Load a PyTorch Lightning .ckpt file (audio_model.* prefix)."""
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
        print(f"[inference] Missing keys  ({len(missing)}): {missing[:3]}{'...' if len(missing)>3 else ''}")
    if unexpected:
        print(f"[inference] Unexpected keys ({len(unexpected)}): {unexpected[:3]}{'...' if len(unexpected)>3 else ''}")
    if not missing:
        print("[inference] All keys loaded successfully")
    return model


def load_model(weights, sr, win, feature_dim, layer):
    """
    Load Apollo model from:
      * None or shortname (apollo/lew/lew_v2/lew_uni) -> auto-download to models/
      * path ending in .ckpt       -> Lightning checkpoint
      * path ending in .pth / .bin -> from_pretrain (serialized) format
      * "JusperLee/Apollo"         -> HuggingFace hub (no local cache)

    When a shortname is used and feature_dim is still at the default (256),
    the registry value overrides it automatically.
    """
    # Resolve shortnames first
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
        # .pth / .bin -- from_pretrain serialized format
        print(f"[inference] Loading serialized model: {weights}")
        model = look2hear.models.BaseModel.from_pretrain(
            weights, sr=sr, win=win, feature_dim=feature_dim, layer=layer
        )

    return model


# ─────────────────────────────────────────────────────────────────────────────
# Chunked inference (handles long files without OOM)
# ─────────────────────────────────────────────────────────────────────────────

_CHUNK_SEC   = 30     # seconds per processing chunk
_OVERLAP_SEC = 0.5    # crossfade overlap at boundaries


def _run_chunked(model, audio, device, sr):
    """
    Process audio in overlapping chunks and crossfade the boundaries.
    audio: [1, C, T]  ->  returns [1, C, T]
    """
    chunk_samples   = int(_CHUNK_SEC * sr)
    overlap_samples = int(_OVERLAP_SEC * sr)
    hop_samples     = chunk_samples - overlap_samples

    T      = audio.shape[-1]
    output = torch.zeros_like(audio)

    start = 0
    chunk_idx = 0
    while start < T:
        end   = min(start + chunk_samples, T)
        chunk = audio[..., start:end].to(device)

        with torch.no_grad():
            enhanced = model(chunk)   # [1, C, T_chunk]
        enhanced = enhanced.cpu()

        # Crossfade leading overlap with previous chunk output
        if start > 0 and overlap_samples > 0:
            fade_len = min(overlap_samples, end - start, T - start)
            fade_in  = torch.linspace(0.0, 1.0, fade_len)
            fade_out = 1.0 - fade_in
            output[..., start:start + fade_len] = (
                output[..., start:start + fade_len] * fade_out
                + enhanced[..., :fade_len]           * fade_in
            )
            output[..., start + fade_len:end] = enhanced[..., fade_len:end - start]
        else:
            output[..., start:end] = enhanced[..., :end - start]

        chunk_idx += 1
        print(f"[inference] Chunk {chunk_idx} done  ({start/sr:.1f}s - {end/sr:.1f}s)")
        start += hop_samples

    return output


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(
    input_wav,
    output_wav,
    weights=None,
    sr=_SR,
    win=20,
    feature_dim=256,
    layer=6,
    device_str="auto",
    chunked=True,
):
    # Device selection
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"[inference] Device: {device}")

    # Load model
    model = load_model(weights=weights, sr=sr, win=win, feature_dim=feature_dim, layer=layer)
    model = model.to(device).eval()

    # Load input
    audio = load_audio(input_wav, target_sr=sr)   # [1, 2, T]
    duration = audio.shape[-1] / sr
    print(f"[inference] Input: {input_wav}  ({duration:.1f}s, {audio.shape[-2]}ch)")

    # Run enhancement
    if chunked and audio.shape[-1] > int(_CHUNK_SEC * sr):
        print(f"[inference] Long file -- using chunked inference ({_CHUNK_SEC}s chunks, {_OVERLAP_SEC}s overlap)")
        enhanced = _run_chunked(model, audio, device, sr)
    else:
        audio_d = audio.to(device)
        with torch.no_grad():
            enhanced = model(audio_d)
        enhanced = enhanced.cpu()

    # Save output
    save_audio(output_wav, enhanced, sr=sr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apollo audio enhancement")
    parser.add_argument("--in_wav",      type=str, required=True,
                        help="Path to input audio file")
    parser.add_argument("--out_wav",     type=str, required=True,
                        help="Path to save enhanced output")
    parser.add_argument("--weights",     type=str, default=None,
                        help="Model to use. Shortnames: apollo, lew, lew_v2, lew_uni "
                             "(auto-downloaded to models/). Or a local .pth/.bin/.ckpt path. "
                             "Default: apollo (downloads pytorch_model.bin from HuggingFace)")
    parser.add_argument("--sr",          type=int, default=_SR,
                        help=f"Sample rate (default: {_SR})")
    parser.add_argument("--win",         type=int, default=20,
                        help="STFT window size in ms (default: 20)")
    parser.add_argument("--feature_dim", type=int, default=256,
                        help="Model feature dim: 256=base, 384=universal (default: 256)")
    parser.add_argument("--layer",       type=int, default=6,
                        help="Number of BSNet layers (default: 6)")
    parser.add_argument("--device",      type=str, default="auto",
                        help="'auto', 'cuda', 'cpu', 'cuda:1', ... (default: auto)")
    parser.add_argument("--no_chunked",  action="store_true",
                        help="Disable chunked inference (may OOM on long files)")
    args = parser.parse_args()

    main(
        input_wav=args.in_wav,
        output_wav=args.out_wav,
        weights=args.weights,
        sr=args.sr,
        win=args.win,
        feature_dim=args.feature_dim,
        layer=args.layer,
        device_str=args.device,
        chunked=not args.no_chunked,
    )

# @claude last-modified: 2026-08-13T00:00:00Z
# @claude last-commit: feat: --conf_dir reads model params and chunk size from config
"""
inference.py — Apollo audio enhancement script

Model selection (--weights):
  apollo      Official Apollo base model (default, auto-downloads pytorch_model.bin)
  lew         Lew vocal enhancer v1 (auto-downloads apollo_model.ckpt)
  lew_v2      Lew vocal enhancer v2 (auto-downloads apollo_model_v2.ckpt)
  lew_uni     Lew universal model, feature_dim=384 (auto-downloads apollo_model_uni.ckpt)
  <path>      Local .pth / .bin / .ckpt file

Usage:
    python inference.py --in_wav input.wav --out_wav output.wav --weights lew_v2
    python inference.py --in_wav input.wav --out_wav output.wav \\
        --weights models/SFTL.ckpt --conf_dir configs/SFTL.yaml
    python inference.py --in_wav input.wav --out_wav output.wav \\
        --weights models/my_finetune.ckpt --feature_dim 256 --chunk_sec 4
"""

import argparse
import os

import torch
import torchaudio
import look2hear.models
import look2hear.models.apollo

_SR          = 44100  # Apollo's native sample rate
_CHUNK_SEC   = 4      # default chunk size — matches training segment_sec
_OVERLAP_SEC = 0.5    # crossfade overlap at chunk boundaries

_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

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


def _run_chunked(model, audio, device, sr, chunk_sec, overlap_sec, out_path):
    """Process audio in chunks and write each chunk to disk immediately.
    
    Writes sequentially so the output WAV can be previewed in Audacity
    while inference is still running — just drag the file in and hit play.
    Returns None (output already on disk).
    """
    import soundfile as sf
    import numpy as np

    # Normalize to match training: divide by peak so the model sees [-1, 1] input.
    # Store the scale so we can restore the original level after inference.
    peak = audio.abs().max().item()
    if peak > 0:
        audio = audio / peak
    else:
        peak = 1.0

    chunk_samples   = int(chunk_sec * sr)
    overlap_samples = int(overlap_sec * sr)
    hop_samples     = chunk_samples - overlap_samples

    T         = audio.shape[-1]
    prev_tail = None  # holds the overlap region from the previous chunk

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    # 32-bit float WAV: no integer ceiling, no clipping at ±1.0.
    with sf.SoundFile(out_path, mode="w", samplerate=sr, channels=2,
                      subtype="FLOAT") as f:
        start     = 0
        chunk_idx = 0
        while start < T:
            end   = min(start + chunk_samples, T)
            chunk = audio[..., start:end].to(device)

            with torch.no_grad():
                enhanced = model(chunk)
            # enhanced: [1, 2, T_chunk] — restore original level after inference
            enhanced = enhanced.squeeze(0).cpu() * peak  # [2, T_chunk]

            chunk_len = enhanced.shape[-1]

            if prev_tail is not None and overlap_samples > 0:
                fade_len = min(overlap_samples, chunk_len, prev_tail.shape[-1])
                fade_in  = torch.linspace(0.0, 1.0, fade_len)
                fade_out = 1.0 - fade_in

                # crossfade zone — blend previous tail with current head
                blended = prev_tail[..., :fade_len] * fade_out \
                        + enhanced[..., :fade_len]  * fade_in

                # write blended overlap
                f.write(blended.T.numpy())
                # write remainder of this chunk (excluding the next overlap tail)
                write_end = chunk_len - overlap_samples
                if write_end > fade_len:
                    f.write(enhanced[..., fade_len:write_end].T.numpy())
            else:
                # first chunk — write everything except the tail we'll crossfade next time
                write_end = chunk_len - overlap_samples if (T - end) > 0 else chunk_len
                write_end = max(write_end, 0)
                if write_end > 0:
                    f.write(enhanced[..., :write_end].T.numpy())

            # keep tail for next chunk's crossfade (or flush if last chunk)
            if end < T:
                tail_start = max(0, chunk_len - overlap_samples)
                prev_tail  = enhanced[..., tail_start:]
            else:
                # last chunk — flush any remaining tail
                if prev_tail is not None and overlap_samples > 0:
                    tail_start = max(0, chunk_len - overlap_samples)
                    f.write(enhanced[..., tail_start:].T.numpy())
                elif prev_tail is None:
                    # single chunk, no overlap
                    pass
                prev_tail = None

            chunk_idx += 1
            print(f"[inference] Chunk {chunk_idx} done  "
                  f"({start/sr:.1f}s – {end/sr:.1f}s)  -> {out_path}")
            start += hop_samples

    return None  # output already written


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
):
    # Load config and pull model params + chunk size from it
    # Explicit CLI args always win over config values
    if conf_dir is not None:
        cfg = load_config(conf_dir)
        m = cfg.get("model", {})
        d = cfg.get("datas", {})
        if "sr"          in m: sr          = m.sr
        if "win"         in m: win         = m.win
        if "feature_dim" in m: feature_dim = m.feature_dim
        if "layer"       in m: layer       = m.layer
        if "segment_sec" in d: chunk_sec   = d.segment_sec
        print(f"[inference] Config: feature_dim={feature_dim}, sr={sr}, win={win}, "
              f"layer={layer}, chunk_sec={chunk_sec}")

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"[inference] Device: {device}")

    model = load_model(weights=weights, sr=sr, win=win, feature_dim=feature_dim, layer=layer)
    model = model.to(device).eval()

    audio = load_audio(input_wav, target_sr=sr)
    duration = audio.shape[-1] / sr
    print(f"[inference] Input: {input_wav}  ({duration:.1f}s, {audio.shape[-2]}ch)")

    if chunked and audio.shape[-1] > int(chunk_sec * sr):
        print(f"[inference] Chunked inference ({chunk_sec}s chunks, {overlap_sec}s overlap)")
        print(f"[inference] Writing sequentially — you can preview in Audacity now")
        _run_chunked(model, audio, device, sr, chunk_sec, overlap_sec, out_path=output_wav)
        print(f"[inference] Done -> {output_wav}")
    else:
        peak = audio.abs().max().item()
        if peak > 0:
            audio = audio / peak
        with torch.no_grad():
            enhanced = model(audio.to(device))
        enhanced = enhanced.cpu() * peak
        save_audio(output_wav, enhanced, sr=sr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apollo audio enhancement")
    parser.add_argument("--in_wav",      type=str, required=True,
                        help="Path to input audio file")
    parser.add_argument("--out_wav",     type=str, required=True,
                        help="Path to save enhanced output")
    parser.add_argument("--weights",     type=str, default=None,
                        help="Model weights: shortname (apollo/lew/lew_v2/lew_uni), "
                             "or local .pth/.bin/.ckpt path")
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
    args = parser.parse_args()

    # Defaults — applied here so config can fill them before CLI overrides
    sr          = args.sr          if args.sr          is not None else _SR
    win         = args.win         if args.win         is not None else 20
    feature_dim = args.feature_dim if args.feature_dim is not None else 256
    layer       = args.layer       if args.layer       is not None else 6
    chunk_sec   = args.chunk_sec   if args.chunk_sec   is not None else _CHUNK_SEC

    main(
        input_wav=args.in_wav,
        output_wav=args.out_wav,
        weights=args.weights,
        conf_dir=args.conf_dir,
        sr=sr,
        win=win,
        feature_dim=feature_dim,
        layer=layer,
        chunk_sec=chunk_sec,
        overlap_sec=args.overlap_sec,
        device_str=args.device,
        chunked=not args.no_chunked,
    )

import hashlib
import os
import tempfile
import urllib.request

import torch

from .lightning_utils import print_only

BASE_APOLLO_URL = "https://huggingface.co/JusperLee/Apollo/resolve/0b123a4606d5345d9082f6f633d8bfebe84c8e79/pytorch_model.bin"
UNIVERSAL_APOLLO_URL = "https://github.com/deton24/Lew-s-vocal-enhancer-for-Apollo-by-JusperLee/releases/download/uni/apollo_model_uni.ckpt"


def ensure_pretrained_weights(feature_dim: int, models_dir: str) -> str:
    if feature_dim == 256:
        filename = "pytorch_model.bin"
        url = BASE_APOLLO_URL
        digest = "99d9af7f1ff20e63c393035513a655392818d66b4d7fc23d658175c1f15e8d76"
        candidates = ("apollo.ckpt", "apollo_model.ckpt")
    elif feature_dim == 384:
        filename = "apollo_model_uni.ckpt"
        url = UNIVERSAL_APOLLO_URL
        digest = "11433256eca90143d32f1108d156ac756fd0201fbc69e5f736def4062fb52186"
        candidates = ("apollo_uni.ckpt", filename)
    else:
        raise ValueError(f"No pretrained Apollo checkpoint for feature_dim={feature_dim}. Set weights_path explicitly.")

    for candidate in candidates:
        path = os.path.join(models_dir, candidate)
        if os.path.isfile(path):
            return path

    os.makedirs(models_dir, exist_ok=True)
    path = os.path.join(models_dir, filename)
    if os.path.isfile(path):
        with open(path, "rb") as source:
            if hashlib.file_digest(source, "sha256").hexdigest() == digest:
                return path
        # Keep an existing file at this path, which may be a user's own model.
        path = os.path.join(models_dir, "apollo_base_official.bin")
        if os.path.isfile(path):
            with open(path, "rb") as source:
                if hashlib.file_digest(source, "sha256").hexdigest() == digest:
                    return path
    print_only(f"[weights] Downloading pretrained Apollo (feature_dim={feature_dim}) to {path}")
    download_file(url, path, label="weights", sha256=digest)
    return path


def load_checkpoint(path: str, map_location="cpu"):
    """torch.load with weights_only=True first; falls back to a full unpickle with a warning."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception as e:
        print_only(f"[ckpt] weights_only load failed for {os.path.basename(path)} ({type(e).__name__}) "
                   f"-- falling back to full unpickle. Only load checkpoints you trust.")
        return torch.load(path, map_location=map_location, weights_only=False)


def atomic_torch_save(obj, path: str) -> None:
    tmp = path + ".tmp"
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def download_file(url: str, dest: str, label: str = "download", timeout: int = 60,
                  sha256: str | None = None) -> None:
    """Stream url to dest via a .part file; dest only appears once the download is complete."""
    fd, part = tempfile.mkstemp(prefix=os.path.basename(dest) + ".", suffix=".part",
                                dir=os.path.dirname(os.path.abspath(dest)))
    os.close(fd)
    digest = hashlib.sha256() if sha256 else None
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp, open(part, "wb") as out:
            total = int(resp.headers.get("Content-Length") or 0)
            done = 0
            while True:
                block = resp.read(1 << 20)
                if not block:
                    break
                out.write(block)
                if digest is not None:
                    digest.update(block)
                done += len(block)
                if total:
                    print(f"\r[{label}] {min(100, done * 100 // total):3d}%", end="", flush=True)
        print()
        if total and done != total:
            raise IOError(f"incomplete download: got {done} of {total} bytes")
        if digest is not None and digest.hexdigest() != sha256:
            raise IOError(f"SHA-256 mismatch for {url}")
        os.replace(part, dest)
    finally:
        if os.path.exists(part):
            os.remove(part)

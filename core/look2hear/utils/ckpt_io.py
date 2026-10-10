import os
import urllib.request

import torch

from .lightning_utils import print_only


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


def download_file(url: str, dest: str, label: str = "download", timeout: int = 60) -> None:
    """Stream url to dest via a .part file; dest only appears once the download is complete."""
    part = dest + ".part"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp, open(part, "wb") as out:
            total = int(resp.headers.get("Content-Length") or 0)
            done = 0
            while True:
                block = resp.read(1 << 20)
                if not block:
                    break
                out.write(block)
                done += len(block)
                if total:
                    print(f"\r[{label}] {min(100, done * 100 // total):3d}%", end="", flush=True)
        print()
        if total and done != total:
            raise IOError(f"incomplete download: got {done} of {total} bytes")
        os.replace(part, dest)
    finally:
        if os.path.exists(part):
            os.remove(part)

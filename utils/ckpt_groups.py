"""
ckpt_groups.py -- which val set and window a checkpoint was scored on.

visqol is only comparable between checkpoints scored on the same val clips and the same
rotation window. Each checkpoint directory keeps a .val_sets.json mapping filename -> group id,
written when the checkpoint is saved. Checkpoints without an entry belong to no group (None).
"""

import hashlib
import json
import os

_FILE = ".val_sets.json"


def group_id(sig, window_idx=0):
    """Group id from a val clip signature (sorted clip paths) and the rotation window index."""
    if not sig:
        return None
    h = hashlib.md5("\n".join(map(str, sig)).encode("utf-8"), usedforsecurity=False).hexdigest()[:8]
    return f"{h}:{int(window_idx or 0)}"


def read_tags(ckpt_dir):
    try:
        with open(os.path.join(ckpt_dir, _FILE), "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as e:
        print(f"[checkpoint] Could not read {_FILE} in {ckpt_dir}: {e}")
        return {}


def write_tags(ckpt_dir, updates):
    """Merge {filename: group id} into the directory's tag file. None ids are not stored."""
    updates = {k: v for k, v in updates.items() if v}
    if not updates:
        return
    tags = read_tags(ckpt_dir)
    tags.update(updates)
    live = {f for f in os.listdir(ckpt_dir)}
    tags = {k: v for k, v in tags.items() if k in live}
    path = os.path.join(ckpt_dir, _FILE)
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(tags, f, indent=1)
        os.replace(tmp, path)
    except OSError as e:
        print(f"[checkpoint] Could not write {_FILE}: {e}")
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def group_of(ckpt_path):
    """Tagged group id for a checkpoint file, or None when it has no tag."""
    d, f = os.path.split(str(ckpt_path))
    return read_tags(d).get(f)

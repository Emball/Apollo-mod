"""
config_schema.py -- the keys the training code reads from a config, with their defaults.

load_config() merges these defaults under the YAML (a missing key gets its default) and logs
keys the code does not read. utils/check_config.py compares this file against the config reads
in the code and the shipped configs, so the three cannot drift apart silently.

Sections in OPEN are passed to classes named by their _target_, so only the keys listed here
are checked in them; any other key there is the target class's business.
"""

import copy

from omegaconf import OmegaConf


def _toggle(**kw):
    return {"enabled": False, **kw}


CONFIG = {
    "exp": {"dir": "./runs", "name": None},
    "resume_checkpoint": None,
    "resume": False,
    "seed": None,
    "optimizations": {
        "cudnn_benchmark": True,
        "expandable_segments": True,
        "triton_cache": True,
        "ram_limit_fraction": 0.98,
    },
    "training": {
        "n_layers_to_freeze": 4,
        "val_songs": 3,
        "val_rotate_every": "auto",
        "grad_accum_steps": 1,
    },
    "datas": {
        "sr": 44100,
        "segment_sec": 3,
        "batch_size": 1,
        "num_workers": 2,
        "pin_memory": False,
        "val_bootstrap_chunks": 50,
        "align_data": False,
        "cache_retention_days": 30,
        "compression": {
            "enabled": False,
            "recipe": "utils/degrade/stfl.json",
            "seed": 0,
            "pad_sec": 1.0,
            "workers": 0,
            "ffmpeg": None,
            "acmenc": None,
        },
        "augmentation": {
            "live": {
                "enabled": False,
                "stereo_alternation": _toggle(prob=1.0),
                "mid_side_isolation": _toggle(prob_mid=0.1, prob_side=0.1),
                "gain": _toggle(prob=0.5, db_max=1.5),
                "deep_gain": _toggle(prob=0.03, db_min=-10.0, db_max=-6.0),
                "silence_dip": _toggle(
                    prob=0.05, max_hold_sec=2.0,
                    short_ramp_ms=10.0, short_ramp_max_ms=50.0,
                    medium_ramp_ms=50.0, medium_ramp_max_ms=200.0,
                    long_ramp_ms=200.0, long_ramp_max_ms=1000.0),
                "polarity": _toggle(prob=0.5),
            },
            "cached": {
                "enabled": False,
                "stereo_alternation": _toggle(fraction=1.0),
                "gain": _toggle(fraction=0.5, db_max=1.5),
                "polarity": _toggle(fraction=0.5),
            },
        },
    },
    "model": {"sr": 44100, "win": 20, "feature_dim": 256, "layer": 6},
    "optimizer": {
        "type": "AdamW",
        "lr_g": 1e-5,
        "lr_d": 1e-6,
        "weight_decay": 0.01,
        "betas_g": [0.9, 0.999],
        "betas_d": [0.5, 0.99],
    },
    "system": {"gradient_checkpointing": False, "visqol_fraction": 1.0},
    "checkpoint": {"keep_best": 0, "keep_latest": 0},
}

OPEN = {"model", "system", "checkpoint", "logger", "trainer", "discriminator",
        "scheduler_g", "scheduler_d", "loss_g", "loss_d", "metrics"}

# Keys the code reads only when present, so a config may omit them and they are never filled in.
OPTIONAL = {
    ("exp", "name"),
    ("datas", "compression", "ffmpeg"),
    ("datas", "compression", "acmenc"),
}

# Keys train.py always overwrites or ignores; a config that sets them has no effect.
DEAD = {
    ("checkpoint", "dirpath"), ("checkpoint", "monitor"), ("checkpoint", "mode"),
    ("checkpoint", "save_top_k"), ("checkpoint", "filename"),
    ("logger", "save_dir"), ("trainer", "default_root_dir"),
    ("system", "val_songs"), ("system", "val_rotate_every"), ("system", "grad_accum_steps"),
}

# Set by the code or the command line at run time, never written to a config file.
RUNTIME = {("datas", "train_dir"), ("datas", "eval_dir"), ("weights_path",)}

# Old names still read for existing configs: old path -> current path.
LEGACY = {
    ("training", "val_metric_songs"): ("training", "val_songs"),
    ("training", "val_metric_samples"): ("training", "val_songs"),
    ("training", "val_audio_pairs"): ("training", "val_songs"),
}
_OLD_COMPRESSION = ("datas", "augmentation", "cached", "compression")
_NEW_COMPRESSION = ("datas", "compression")
# Old config paths the code still checks for in order to raise a pointer to the new one.
LEGACY_READS = {
    ("datas", "augmentation", "cached", "mp3_degradation"),
    _OLD_COMPRESSION,
}


def _get(d, path):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return None, False
        d = d[k]
    return d, True


def _set(d, path, value):
    for k in path[:-1]:
        d = d.setdefault(k, {})
    d[path[-1]] = value


def _del(d, path):
    for k in path[:-1]:
        d = d.get(k)
        if not isinstance(d, dict):
            return
    d.pop(path[-1], None)


def _leaves(d, prefix=()):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from _leaves(v, prefix + (k,))
        else:
            yield prefix + (k,)


def migrate(raw):
    """Move old key names to their current place. Returns a new dict and the moves made."""
    raw = copy.deepcopy(raw)
    moved = []
    for old, new in LEGACY.items():
        val, ok = _get(raw, old)
        if ok and not _get(raw, new)[1]:
            _set(raw, new, val)
            moved.append((old, new))
    old_blk, ok = _get(raw, _OLD_COMPRESSION)
    if ok and isinstance(old_blk, dict):
        # The old location only took effect while augmentation.cached was enabled.
        if not _get(raw, _NEW_COMPRESSION)[1]:
            blk = dict(old_blk)
            blk["enabled"] = bool(blk.get("enabled", False)) and bool(
                (_get(raw, ("datas", "augmentation", "cached", "enabled"))[0]))
            _set(raw, _NEW_COMPRESSION, blk)
        _del(raw, _OLD_COMPRESSION)
        moved.append((_OLD_COMPRESSION, _NEW_COMPRESSION))
    return raw, moved


def analyze(raw):
    """Compare a raw config dict with the schema. Returns {unknown, dead, missing} path lists."""
    raw, _ = migrate(raw)
    unknown, dead, missing = [], [], []
    for p in _leaves(raw):
        if p[-1].startswith("_") or p in LEGACY:
            continue
        if p in DEAD:
            dead.append(p)
        elif p[0] in OPEN or p in RUNTIME:
            continue
        else:
            node = CONFIG
            for k in p:
                if not isinstance(node, dict):
                    break
                if k not in node:
                    unknown.append(p)
                    break
                node = node[k]
    for p in _leaves(CONFIG):
        if p in OPTIONAL or _get(raw, p)[1]:
            continue
        missing.append(p)
    return {"unknown": sorted(unknown), "dead": sorted(dead), "missing": sorted(missing)}


def defaults():
    """The schema defaults without the optional keys, as a plain dict."""
    d = copy.deepcopy(CONFIG)
    for p in OPTIONAL:
        _del(d, p)
    return d


def load_config(path, log=print):
    """Load a YAML config with schema defaults filled in. Dead and unknown keys are logged,
    never removed, and the file itself is not modified."""
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=False)
    raw, moved = migrate(raw)
    for old, new in moved:
        log(f"[config] '{'.'.join(old)}' is an old name for '{'.'.join(new)}' -- update the config")
    res = analyze(raw)
    for p in res["unknown"]:
        log(f"[config] '{'.'.join(p)}' is not read by the code")
    for p in res["dead"]:
        log(f"[config] '{'.'.join(p)}' is ignored (the code sets it)")
    return OmegaConf.merge(OmegaConf.create(defaults()), OmegaConf.create(raw))

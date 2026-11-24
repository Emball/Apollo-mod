"""
check_config.py -- keeps configs, the schema and the code in agreement.

    python utils/check_config.py

1. Every shipped config is checked against core/config_schema.py: keys the code does not
   read, keys the code ignores, and schema keys the config lacks.
2. The code is scanned for config reads (cfg.a.b, cfg.a.get("b"), getattr(cfg.a, "b")) and
   any read of a closed section that the schema does not list is reported.

Exits 1 when anything is reported.
"""

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "core"))
import config_schema as cs
from omegaconf import OmegaConf

SCAN = ["core/train.py", "core/evaluate.py", "core/inference.py", "utils/tui.py"]
ROOTS = {"cfg", "aux_cfg"}
CALL_ALIASES = {
    "_compression_cfg": ("datas", "compression"),
    "_cached_block": ("datas", "augmentation", "cached"),
}
PARAM_ALIASES = {("_load_recipe", "blk"): ("datas", "compression")}
METHODS = {"get", "items", "keys", "values", "pop", "copy"}


def _str(node):
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _resolve(node, alias, hits):
    """Config path an expression refers to, or None. Reads are appended to hits."""
    if isinstance(node, ast.Name):
        return () if node.id in ROOTS else alias.get(node.id)
    if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
        base = _resolve(node.value, alias, hits)
        if base is None or node.attr in METHODS:
            return None
        hits.append((node.lineno, base + (node.attr,)))
        return base + (node.attr,)
    if isinstance(node, ast.Subscript):
        base, key = _resolve(node.value, alias, hits), _str(node.slice)
        if base is not None and key:
            hits.append((node.lineno, base + (key,)))
            return base + (key,)
    if isinstance(node, ast.Call):
        f = node.func
        if isinstance(f, ast.Attribute) and f.attr == "get" and node.args:
            base, key = _resolve(f.value, alias, hits), _str(node.args[0])
            if base is not None and key:
                hits.append((node.lineno, base + (key,)))
                return base + (key,)
        if isinstance(f, ast.Name) and f.id == "getattr" and len(node.args) >= 2:
            base, key = _resolve(node.args[0], alias, hits), _str(node.args[1])
            if base is not None and key:
                hits.append((node.lineno, base + (key,)))
                return base + (key,)
        if isinstance(f, ast.Name) and f.id in CALL_ALIASES:
            return CALL_ALIASES[f.id]
    return None


def scan_file(path):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    hits = []
    scopes = [tree] + [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    for fn in scopes:
        alias = {}
        if not isinstance(fn, ast.Module):
            for a in fn.args.args:
                if (fn.name, a.arg) in PARAM_ALIASES:
                    alias[a.arg] = PARAM_ALIASES[(fn.name, a.arg)]
        nodes = list(ast.walk(fn))
        for _ in range(2):
            for n in nodes:
                if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
                    p = _resolve(n.value, alias, [])
                    if p:
                        alias[n.targets[0].id] = p
        for n in nodes:
            if isinstance(n, (ast.Attribute, ast.Call, ast.Subscript)):
                _resolve(n, alias, hits)
    return sorted({(path, ln, p) for ln, p in hits})


def _known(p):
    """True when a read of config path p is accounted for by the schema."""
    if not p or p[-1].startswith("_"):
        return True
    if p[0] in cs.OPEN or p in cs.DEAD or p in cs.RUNTIME or p in cs.LEGACY or p in cs.LEGACY_READS:
        return True
    node = cs.CONFIG
    for k in p:
        if not isinstance(node, dict):
            return True
        if k not in node:
            return False
        node = node[k]
    return True


def main():
    bad = 0
    for cfg_path in sorted((ROOT / "configs").glob("*.yaml")):
        res = cs.analyze(OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=False))
        for kind in ("unknown", "dead", "missing"):
            for p in res[kind]:
                print(f"{cfg_path.name}: {kind} key '{'.'.join(p)}'")
                bad += 1
    for f in SCAN:
        for path, ln, p in scan_file(f):
            if not _known(p):
                print(f"{path}:{ln}: reads '{'.'.join(p)}' which is not in config_schema.py")
                bad += 1
    print(f"config check: {bad} problem(s)" if bad else "config check: ok")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

"""Carry optimizer state across a changed trainable parameter set (freeze count, layer count,
architecture edits). Pure Python so it can be tested without torch."""


def _shape(v):
    s = getattr(v, "shape", None)
    return tuple(s) if s is not None else None


def _state_shape(entry):
    if not entry:
        return None
    for key in ("exp_avg", "momentum_buffer", "exp_avg_sq"):
        if key in entry and _shape(entry[key]):
            return _shape(entry[key])
    for v in entry.values():
        s = _shape(v)
        if s:
            return s
    return None


def _lcs_end_aligned(a, b):
    """Longest common subsequence of two shape lists, backtracked from the end so equal shapes
    pair up at the tail. Returns {index_in_b: index_in_a}. Top-down unfreezing only adds
    parameters at the front of the list, so tail alignment recovers the exact pairing."""
    n, m = len(a), len(b)
    table = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        ai = a[i - 1]
        row, prev = table[i], table[i - 1]
        for j in range(1, m + 1):
            if ai is not None and ai == b[j - 1]:
                row[j] = prev[j - 1] + 1
            else:
                row[j] = prev[j] if prev[j] >= row[j - 1] else row[j - 1]
    match, i, j = {}, n, m
    while i > 0 and j > 0:
        if a[i - 1] is not None and a[i - 1] == b[j - 1] and table[i][j] == table[i - 1][j - 1] + 1:
            match[j - 1] = i - 1
            i -= 1
            j -= 1
        elif table[i - 1][j] >= table[i][j - 1]:
            i -= 1
        else:
            j -= 1
    return match


def remap_optimizer_state(saved_sd, saved_names, live_sd, live_names, live_shapes):
    """Rebuild a saved optimizer state_dict for the live optimizer's parameter list.

    saved_sd / live_sd: plain optimizer state_dicts ({'state', 'param_groups'}).
    saved_names: per-group lists of parameter names stored in the checkpoint, or None for
        checkpoints written before names were saved (matched by shape, aligned from the end).
    live_names / live_shapes: per-group lists of names / shapes of the live parameters.

    Returns (new_state_dict, report), or (None, report) when nothing changed. A parameter keeps
    its saved state only when it matches a saved one by name (or aligned position) and shape;
    everything else starts fresh. Group hyperparameters (lr and so on) come from the checkpoint.
    Raises ValueError when the group count differs.
    """
    s_groups, l_groups = saved_sd["param_groups"], live_sd["param_groups"]
    if len(s_groups) != len(l_groups):
        raise ValueError("optimizer group count changed")
    s_idx = [p for g in s_groups for p in g["params"]]
    l_idx = [p for g in l_groups for p in g["params"]]
    s_shapes = [_state_shape(saved_sd["state"].get(i)) for i in s_idx]
    l_shapes = [tuple(s) for g in live_shapes for s in g]
    l_names = [n for g in live_names for n in g]
    s_names = [n for g in saved_names for n in g] if saved_names else None
    if s_names is not None and len(s_names) != len(s_idx):
        s_names = None
    s_sizes = [len(g["params"]) for g in s_groups]
    l_sizes = [len(g["params"]) for g in l_groups]

    use_names = s_names is not None and None not in l_names
    same_shapes = s_sizes == l_sizes and all(s is None or s == l for s, l in zip(s_shapes, l_shapes))
    unchanged = (s_names == l_names and same_shapes) if use_names else same_shapes
    if unchanged:
        return None, {"changed": False}

    if use_names:
        pos = {n: k for k, n in enumerate(s_names) if n is not None}
        match = {}
        for k, n in enumerate(l_names):
            sk = pos.get(n)
            if sk is not None and (s_shapes[sk] is None or s_shapes[sk] == l_shapes[k]):
                match[k] = sk
        how = "by name"
    else:
        match = _lcs_end_aligned(s_shapes, l_shapes)
        how = "by shape, aligned from the end"

    new_state = {}
    for k, sk in match.items():
        entry = saved_sd["state"].get(s_idx[sk])
        if entry is not None:
            new_state[l_idx[k]] = entry
    with_state = sum(1 for i in s_idx if i in saved_sd["state"])
    new_groups = []
    for sg, lg in zip(s_groups, l_groups):
        g = {k: v for k, v in sg.items() if k != "params"}
        g["params"] = list(lg["params"])
        new_groups.append(g)
    report = {
        "changed": True,
        "how": how,
        "carried": len(new_state),
        "fresh": len(l_idx) - len(new_state),
        "dropped": with_state - len(new_state),
        "saved": len(s_idx),
        "live": len(l_idx),
    }
    return {"state": new_state, "param_groups": new_groups}, report

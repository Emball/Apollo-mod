"""
Apollo TUI — keyboard-navigated launcher for train / inference / utilities.

Navigation: arrow keys or j/k, Enter to select, Escape or q to go back.
During training: output streams live; press Ctrl+C to stop and return to menu.
"""
from __future__ import annotations

import json
import os
import platform
import signal
import subprocess
import sys
import termios
import threading
import tty
from pathlib import Path

from rich.align import Align
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.rule import Rule
from rich.style import Style
from rich.table import Table
from rich.text import Text

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.resolve()
CONFIGS_DIR = ROOT / "configs"
RUNS_DIR = ROOT / "runs"
INPUT_DIR = ROOT / "input"
OUTPUT_DIR = ROOT / "output"
MODELS_DIR = ROOT / "models"
STATE_FILE = ROOT / ".tui_state.json"

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".aac", ".m4a", ".aiff", ".aif"}

IS_WINDOWS = platform.system() == "Windows"

console = Console()

# ---------------------------------------------------------------------------
# Persistent state
# ---------------------------------------------------------------------------

def _load_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:
        return {}


def _save_state(state: dict) -> None:
    try:
        STATE_FILE.write_text(json.dumps(state, indent=2))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Raw keyboard input (cross-platform)
# ---------------------------------------------------------------------------

if IS_WINDOWS:
    import msvcrt

    def _getch() -> str:
        ch = msvcrt.getwch()
        if ch in ("\x00", "\xe0"):
            ext = msvcrt.getwch()
            return {"H": "UP", "P": "DOWN", "M": "RIGHT", "K": "LEFT"}.get(ext, "")
        return ch

else:
    def _getch() -> str:
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
            if ch == "\x1b":
                rest = sys.stdin.read(2)
                seq = ch + rest
                return {"[A": "UP", "[B": "DOWN", "[C": "RIGHT", "[D": "LEFT"}.get(rest, seq)
            return ch
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)


# ---------------------------------------------------------------------------
# ASCII banner
# ---------------------------------------------------------------------------

BANNER = r"""
    ___    ____  ____  __    __    ____     __  _______  ____
   /   |  / __ \/ __ \/ /   / /   / __ \   /  |/  / __ \/ __ \
  / /| | / /_/ / / / / /   / /   / / / /  / /|_/ / / / / / / /
 / ___ |/ ____/ /_/ / /___/ /___/ /_/ /  / /  / / /_/ / /_/ /
/_/  |_/_/    \____/_____/_____/\____/  /_/  /_/\____/_____/
"""

# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _banner_panel() -> Panel:
    t = Text(BANNER, style="bold cyan", justify="center")
    return Panel(t, border_style="dim cyan", padding=(0, 2))


def _menu(
    title: str,
    items: list[str],
    selected: int = 0,
    hint: str = "",
    subtitle: str = "",
) -> Panel:
    table = Table.grid(padding=(0, 2))
    table.add_column(no_wrap=True)
    for i, item in enumerate(items):
        if i == selected:
            row = Text(f"▶  {item}", style="bold bright_white on grey23")
        else:
            row = Text(f"   {item}", style="dim white")
        table.add_row(row)

    sub = Text(subtitle, style="dim") if subtitle else Text("")
    body = Table.grid()
    body.add_column()
    if subtitle:
        body.add_row(sub)
        body.add_row(Text(""))
    body.add_row(table)
    if hint:
        body.add_row(Text(""))
        body.add_row(Text(hint, style="dim italic"))

    return Panel(
        Align.left(body),
        title=f"[bold cyan]{title}[/]",
        border_style="cyan",
        padding=(1, 3),
    )


def _navigate(items: list[str], title: str, hint: str = "", subtitle: str = "", start: int = 0) -> int | None:
    """Show a menu and return selected index, or None if user pressed Escape/q."""
    sel = max(0, min(start, len(items) - 1))
    with Live(console=console, auto_refresh=False, screen=False) as live:
        def _render():
            live.update(
                Table.grid()
                .add_row(_banner_panel())
                .__class__._from_column_and_row(  # type: ignore[attr-defined]
                    None, None
                ),
                refresh=True,
            )
            # simpler direct approach:
            live.update(_menu(title, items, sel, hint=hint, subtitle=subtitle), refresh=True)

        _render()
        while True:
            ch = _getch()
            if ch in ("UP", "k"):
                sel = (sel - 1) % len(items)
            elif ch in ("DOWN", "j"):
                sel = (sel + 1) % len(items)
            elif ch in ("\r", "\n", " "):
                return sel
            elif ch in ("\x1b", "q", "Q"):
                return None
            elif ch == "\x03":  # Ctrl+C
                raise KeyboardInterrupt
            _render()
    return None


def _pick(title: str, items: list[str], hint: str = "", subtitle: str = "", start: int = 0) -> int | None:
    """Wrapper that clears screen before showing menu."""
    console.clear()
    console.print(_banner_panel())
    return _navigate(items, title, hint=hint, subtitle=subtitle, start=start)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _list_configs() -> list[Path]:
    if not CONFIGS_DIR.exists():
        return []
    return sorted(CONFIGS_DIR.glob("*.yaml"))


def _config_summary(cfg_path: Path) -> str:
    """Return a one-line summary of training state for this config."""
    try:
        import yaml  # type: ignore
        cfg = yaml.safe_load(cfg_path.read_text())
        name = cfg.get("exp", {}).get("name") or cfg_path.stem
        runs_path = RUNS_DIR / name
        if not runs_path.exists():
            return "no runs yet"
        # find best checkpoint across all timestamped runs
        best_loss = None
        best_step = None
        for run_dir in sorted(runs_path.iterdir()):
            ckpt_dir = run_dir / "checkpoints"
            if not ckpt_dir.exists():
                continue
            for ckpt in ckpt_dir.glob("*.ckpt"):
                if "val_loss=" in ckpt.stem:
                    try:
                        loss = float(ckpt.stem.split("val_loss=")[1])
                        step = int(ckpt.stem.split("step=")[1].split("-")[0])
                        if best_loss is None or loss < best_loss:
                            best_loss = loss
                            best_step = step
                    except Exception:
                        pass
        if best_loss is not None:
            return f"best val={best_loss:.4f}  step={best_step}"
        return "checkpoint found (no loss in name)"
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _find_best_checkpoint(cfg_path: Path) -> Path | None:
    """Find best checkpoint for a config by val_loss in filename."""
    try:
        import yaml
        cfg = yaml.safe_load(cfg_path.read_text())
        name = cfg.get("exp", {}).get("name") or cfg_path.stem
        runs_path = RUNS_DIR / name
        if not runs_path.exists():
            return None
        best_loss = None
        best_ckpt = None
        for run_dir in sorted(runs_path.iterdir()):
            ckpt_dir = run_dir / "checkpoints"
            if not ckpt_dir.exists():
                continue
            for ckpt in ckpt_dir.glob("*.ckpt"):
                if "val_loss=" in ckpt.stem:
                    try:
                        loss = float(ckpt.stem.split("val_loss=")[1])
                        if best_loss is None or loss < best_loss:
                            best_loss = loss
                            best_ckpt = ckpt
                    except Exception:
                        pass
        return best_ckpt
    except Exception:
        return None


def _find_model_for_config(cfg_path: Path) -> Path | None:
    """Look for a .ckpt or .pth in /models matching config name."""
    if not MODELS_DIR.exists():
        return None
    stem = cfg_path.stem
    for ext in (".ckpt", ".pth"):
        candidate = MODELS_DIR / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


# ---------------------------------------------------------------------------
# Subprocess runner (live output, returns to menu on Ctrl+C)
# ---------------------------------------------------------------------------

def _run_with_live_output(cmd: list[str], label: str) -> None:
    """Run a subprocess, stream output to terminal, handle Ctrl+C gracefully."""
    console.clear()
    console.print(_banner_panel())
    console.print(Panel(
        f"[bold cyan]{label}[/]\n[dim]Press Ctrl+C to stop and return to menu[/]",
        border_style="cyan",
        padding=(0, 2),
    ))
    console.print(Rule(style="dim cyan"))

    proc = None
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(ROOT),
        )

        def _stream():
            assert proc and proc.stdout
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()

        t = threading.Thread(target=_stream, daemon=True)
        t.start()
        proc.wait()
        t.join(timeout=5)

    except KeyboardInterrupt:
        if proc and proc.poll() is None:
            console.print("\n[yellow]Ctrl+C caught — sending stop signal...[/]")
            if IS_WINDOWS:
                proc.send_signal(signal.CTRL_C_EVENT)  # type: ignore[attr-defined]
            else:
                proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
        console.print("[dim]Returning to menu...[/]")

    console.print(Rule(style="dim cyan"))
    console.input("[dim]Press Enter to return to menu[/]")


# ---------------------------------------------------------------------------
# Screens
# ---------------------------------------------------------------------------

def _python_bin() -> str:
    venv = ROOT / ".venv"
    if IS_WINDOWS:
        return str(venv / "Scripts" / "python.exe")
    return str(venv / "bin" / "python")


# -- Train -------------------------------------------------------------------

def screen_train(state: dict) -> None:
    configs = _list_configs()
    if not configs:
        console.clear()
        console.print("[red]No configs found in configs/[/]")
        console.input("Press Enter to return.")
        return

    last_cfg = state.get("train", {}).get("last_config", "")
    start = next((i for i, c in enumerate(configs) if c.name == last_cfg), 0)

    items = [c.stem for c in configs]
    summaries = [_config_summary(c) for c in configs]

    while True:
        # build display items with summary
        display = [f"{items[i]}  [dim]{summaries[i]}[/dim]" for i in range(len(items))]
        # Rich doesn't render markup in Text directly in menu, so strip for plain display
        plain = [f"{items[i]}   {summaries[i]}" for i in range(len(items))]
        idx = _pick("Train — select config", plain, hint="Enter=start  Esc=back", start=start)
        if idx is None:
            return

        cfg_path = configs[idx]
        state.setdefault("train", {})["last_config"] = cfg_path.name
        _save_state(state)

        cmd = [_python_bin(), str(ROOT / "train.py"), "--conf_dir", str(cfg_path)]
        _run_with_live_output(cmd, f"Training: {cfg_path.stem}")
        start = idx
        return


# -- Inference ---------------------------------------------------------------

_PROCESS_ALL_SENTINEL = "__PROCESS_ALL__"


def _pick_input_file(state: dict, cfg_stem: str) -> str | None:
    """Pick an input audio file from /input, process all, or enter custom path.

    Returns a file path string, _PROCESS_ALL_SENTINEL, or None to cancel.
    """
    last = state.get("inference", {}).get(cfg_stem, {}).get("last_input", "")

    INPUT_DIR.mkdir(exist_ok=True)
    files = sorted(f for f in INPUT_DIR.iterdir() if f.suffix.lower() in AUDIO_EXTS)
    file_names = [f.name for f in files]

    process_all_label = (
        f"[ Process all {len(files)} file(s) in /input -> /output ]"
        if files else "[ Process all in /input (folder empty) ]"
    )
    items = file_names + [process_all_label, "[ Enter custom path ]"]
    process_all_idx = len(file_names)
    custom_idx = len(file_names) + 1

    start = next((i for i, n in enumerate(file_names) if n == Path(last).name), 0)

    idx = _pick(
        "Inference — select input",
        items,
        hint="Enter=select  Esc=back",
        start=start,
    )
    if idx is None:
        return None

    if idx == custom_idx:
        console.clear()
        console.print(_banner_panel())
        path = console.input("[cyan]Enter path to input file:[/] ").strip().strip('"')
        return path if path else None

    if idx == process_all_idx:
        return _PROCESS_ALL_SENTINEL

    return str(files[idx])


def _pick_output_path(state: dict, cfg_stem: str, input_path: str) -> str | None:
    """Pick output path — default to /output/<input_stem>_restored.wav or last used."""
    last = state.get("inference", {}).get(cfg_stem, {}).get("last_output", "")
    input_stem = Path(input_path).stem
    default = str(OUTPUT_DIR / f"{input_stem}_restored.wav")
    suggested = last if last else default

    items = [
        f"Default: {suggested}",
        "[ Enter custom path ]",
    ]
    idx = _pick("Inference — output path", items, hint="Enter=select  Esc=back")
    if idx is None:
        return None
    if idx == 0:
        OUTPUT_DIR.mkdir(exist_ok=True)
        return suggested
    console.clear()
    console.print(_banner_panel())
    path = console.input("[cyan]Enter output path:[/] ").strip().strip('"')
    return path if path else None


def screen_inference(state: dict) -> None:
    configs = _list_configs()
    if not configs:
        console.clear()
        console.print("[red]No configs found in configs/[/]")
        console.input("Press Enter to return.")
        return

    last_cfg = state.get("inference", {}).get("last_config", "")
    start = next((i for i, c in enumerate(configs) if c.name == last_cfg), 0)

    items = [c.stem for c in configs]
    idx = _pick("Inference — select config", items, hint="Enter=select  Esc=back", start=start)
    if idx is None:
        return

    cfg_path = configs[idx]
    cfg_stem = cfg_path.stem
    state.setdefault("inference", {})["last_config"] = cfg_path.name
    _save_state(state)

    # Find checkpoint / model
    best_ckpt = _find_best_checkpoint(cfg_path)
    model_file = _find_model_for_config(cfg_path)

    # Build model options
    model_options = []
    model_paths = []

    if best_ckpt:
        loss_str = f"val_loss={best_ckpt.stem.split('val_loss=')[1]}" if "val_loss=" in best_ckpt.stem else ""
        model_options.append(f"Best checkpoint  {loss_str}  ({best_ckpt.name})")
        model_paths.append(str(best_ckpt))
    if model_file:
        model_options.append(f"Model file  ({model_file.name})")
        model_paths.append(str(model_file))
    model_options.append("[ Enter custom weights path ]")

    last_weights = state.get("inference", {}).get(cfg_stem, {}).get("last_weights", "")
    w_start = next((i for i, p in enumerate(model_paths) if p == last_weights), 0)

    widx = _pick(
        "Inference — select model",
        model_options,
        hint="Enter=select  Esc=back",
        start=w_start,
    )
    if widx is None:
        return

    if widx == len(model_options) - 1:
        console.clear()
        console.print(_banner_panel())
        weights = console.input("[cyan]Enter path to weights:[/] ").strip().strip('"')
        if not weights:
            return
    else:
        weights = model_paths[widx]

    state.setdefault("inference", {}).setdefault(cfg_stem, {})["last_weights"] = weights
    _save_state(state)

    # Pick input
    input_path = _pick_input_file(state, cfg_stem)
    if not input_path:
        return

    # --- Batch mode: process all files in /input ---
    if input_path == _PROCESS_ALL_SENTINEL:
        INPUT_DIR.mkdir(exist_ok=True)
        OUTPUT_DIR.mkdir(exist_ok=True)
        batch_files = sorted(
            f for f in INPUT_DIR.iterdir() if f.suffix.lower() in AUDIO_EXTS
        )
        if not batch_files:
            console.clear()
            console.print(_banner_panel())
            console.print("[yellow]No audio files found in /input.[/]")
            console.input("Press Enter.")
            return
        for i, in_file in enumerate(batch_files):
            out_file = OUTPUT_DIR / f"{in_file.stem}_restored.wav"
            cmd = [
                _python_bin(), str(ROOT / "inference.py"),
                "--in_wav", str(in_file),
                "--out_wav", str(out_file),
                "--conf_dir", str(cfg_path),
                "--weights", weights,
            ]
            _run_with_live_output(
                cmd,
                f"Inference [{i+1}/{len(batch_files)}]: {in_file.name}",
            )
        return

    # --- Single file mode ---
    state["inference"][cfg_stem]["last_input"] = input_path
    _save_state(state)

    # Pick output
    output_path = _pick_output_path(state, cfg_stem, input_path)
    if not output_path:
        return

    state["inference"][cfg_stem]["last_output"] = output_path
    _save_state(state)

    # Build command
    cmd = [
        _python_bin(), str(ROOT / "inference.py"),
        "--in_wav", input_path,
        "--out_wav", output_path,
        "--conf_dir", str(cfg_path),
        "--weights", weights,
    ]
    _run_with_live_output(cmd, f"Inference: {cfg_path.stem}")


# -- Edit Config -------------------------------------------------------------

def _flatten_yaml(data: dict, prefix: str = "") -> list[tuple[str, object]]:
    """Flatten nested yaml into dotted key paths, skipping _target_ and internal keys."""
    SKIP = {"_target_", "sr", "win", "layer"}
    result = []
    for k, v in data.items():
        if k in SKIP:
            continue
        full_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            result.extend(_flatten_yaml(v, full_key))
        else:
            result.append((full_key, v))
    return result


def _set_nested(data: dict, dotted_key: str, value: object) -> None:
    keys = dotted_key.split(".")
    d = data
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value


def _parse_value(s: str, original) -> object:
    """Parse string back to the same type as original."""
    if isinstance(original, bool):
        return s.lower() in ("true", "yes", "1")
    if isinstance(original, int):
        return int(s)
    if isinstance(original, float):
        return float(s)
    if isinstance(original, list):
        import ast
        return ast.literal_eval(s)
    # string or None
    if s.lower() in ("null", "none", "false"):
        return False if isinstance(original, bool) else (None if s.lower() in ("null", "none") else s)
    return s


def screen_edit_config(state: dict) -> None:
    try:
        import yaml
    except ImportError:
        console.print("[red]PyYAML not available[/]")
        console.input("Press Enter.")
        return

    configs = _list_configs()
    if not configs:
        console.clear()
        console.print("[red]No configs found[/]")
        console.input("Press Enter.")
        return

    last_cfg = state.get("edit", {}).get("last_config", "")
    start = next((i for i, c in enumerate(configs) if c.name == last_cfg), 0)
    items = [c.stem for c in configs]

    idx = _pick("Edit Config — select config", items, hint="Enter=select  Esc=back", start=start)
    if idx is None:
        return

    cfg_path = configs[idx]
    state.setdefault("edit", {})["last_config"] = cfg_path.name
    _save_state(state)

    data = yaml.safe_load(cfg_path.read_text())
    pairs = _flatten_yaml(data)
    keys = [f"{k}  =  {v}" for k, v in pairs]

    sel = 0
    while True:
        kidx = _pick(
            f"Edit: {cfg_path.stem}",
            keys,
            hint="Enter=edit value  Esc=back (auto-saves)",
            start=sel,
        )
        if kidx is None:
            # save on exit
            with open(cfg_path, "w") as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
            return

        sel = kidx
        dotted_key, original_value = pairs[kidx]

        console.clear()
        console.print(_banner_panel())
        console.print(Panel(
            f"[bold]{dotted_key}[/]\nCurrent value: [cyan]{original_value}[/]",
            border_style="cyan",
        ))
        new_str = console.input("[cyan]New value[/] (Enter to keep): ").strip()
        if new_str:
            try:
                new_val = _parse_value(new_str, original_value)
                _set_nested(data, dotted_key, new_val)
                pairs[kidx] = (dotted_key, new_val)
                keys[kidx] = f"{dotted_key}  =  {new_val}"
            except Exception as e:
                console.print(f"[red]Invalid value: {e}[/]")
                console.input("Press Enter.")


# -- Utilities ---------------------------------------------------------------

def _get_chunk_dirs() -> list[tuple[str, Path]]:
    chunks_root = ROOT / "chunks"
    if not chunks_root.exists():
        return []
    return [(d.name, d) for d in sorted(chunks_root.iterdir()) if d.is_dir()]


def _get_run_dirs() -> list[tuple[str, Path]]:
    if not RUNS_DIR.exists():
        return []
    result = []
    for cfg_dir in sorted(RUNS_DIR.iterdir()):
        if cfg_dir.is_dir():
            result.append((cfg_dir.name, cfg_dir))
    return result


def screen_utilities(state: dict) -> None:
    while True:
        items = [
            "Clean chunks folder",
            "Clean old checkpoints (keep best 5)",
            "View training runs",
            "Back",
        ]
        idx = _pick("Utilities", items, hint="Enter=select  Esc=back")
        if idx is None or idx == len(items) - 1:
            return

        if idx == 0:
            _util_clean_chunks()
        elif idx == 1:
            _util_clean_checkpoints()
        elif idx == 2:
            _util_view_runs()


def _util_clean_chunks() -> None:
    dirs = _get_chunk_dirs()
    if not dirs:
        console.clear()
        console.print("[dim]No chunk folders found.[/]")
        console.input("Press Enter.")
        return

    items = [f"{name}  ({_dir_size(path)})" for name, path in dirs] + ["Back"]
    idx = _pick("Clean Chunks — select folder to delete", items, hint="Enter=delete  Esc=back")
    if idx is None or idx == len(items) - 1:
        return

    name, path = dirs[idx]
    console.clear()
    console.print(_banner_panel())
    confirm = console.input(f"[red]Delete chunks/{name}? (yes/no):[/] ").strip().lower()
    if confirm == "yes":
        import shutil
        shutil.rmtree(path)
        console.print(f"[green]Deleted chunks/{name}[/]")
    else:
        console.print("[dim]Cancelled.[/]")
    console.input("Press Enter.")


def _util_clean_checkpoints() -> None:
    run_dirs = _get_run_dirs()
    if not run_dirs:
        console.clear()
        console.print("[dim]No run folders found.[/]")
        console.input("Press Enter.")
        return

    console.clear()
    console.print(_banner_panel())
    removed = 0
    for cfg_name, cfg_dir in run_dirs:
        for run_dir in sorted(cfg_dir.iterdir()):
            ckpt_dir = run_dir / "checkpoints"
            if not ckpt_dir.exists():
                continue
            ckpts = sorted(
                [c for c in ckpt_dir.glob("*.ckpt") if "val_loss=" in c.stem],
                key=lambda c: float(c.stem.split("val_loss=")[1])
            )
            # keep best 5
            to_delete = ckpts[5:]
            for c in to_delete:
                c.unlink()
                removed += 1
    console.print(f"[green]Removed {removed} checkpoint(s).[/]")
    console.input("Press Enter.")


def _util_view_runs() -> None:
    console.clear()
    console.print(_banner_panel())

    table = Table(title="Training Runs", border_style="dim cyan", show_lines=True)
    table.add_column("Config", style="cyan")
    table.add_column("Run", style="dim")
    table.add_column("Best val_loss", style="green")
    table.add_column("Step")
    table.add_column("Checkpoints")

    if RUNS_DIR.exists():
        for cfg_dir in sorted(RUNS_DIR.iterdir()):
            if not cfg_dir.is_dir():
                continue
            for run_dir in sorted(cfg_dir.iterdir()):
                if not run_dir.is_dir():
                    continue
                ckpt_dir = run_dir / "checkpoints"
                if not ckpt_dir.exists():
                    continue
                ckpts = list(ckpt_dir.glob("*.ckpt"))
                best_loss = None
                best_step = None
                for c in ckpts:
                    if "val_loss=" in c.stem:
                        try:
                            loss = float(c.stem.split("val_loss=")[1])
                            step = int(c.stem.split("step=")[1].split("-")[0])
                            if best_loss is None or loss < best_loss:
                                best_loss = loss
                                best_step = step
                        except Exception:
                            pass
                table.add_row(
                    cfg_dir.name,
                    run_dir.name,
                    f"{best_loss:.4f}" if best_loss else "—",
                    str(best_step) if best_step else "—",
                    str(len(ckpts)),
                )

    console.print(table)
    console.input("\nPress Enter to return.")


def _dir_size(path: Path) -> str:
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    if total > 1_073_741_824:
        return f"{total/1_073_741_824:.1f} GB"
    if total > 1_048_576:
        return f"{total/1_048_576:.0f} MB"
    return f"{total/1024:.0f} KB"


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    state = _load_state()

    MAIN_ITEMS = [
        "Train",
        "Inference",
        "Edit Config",
        "Utilities",
        "Exit",
    ]

    sel = 0
    while True:
        idx = _pick("Main Menu", MAIN_ITEMS, hint="↑↓ navigate  Enter select  q quit", start=sel)
        if idx is None or idx == len(MAIN_ITEMS) - 1:
            console.clear()
            break
        sel = idx
        if idx == 0:
            screen_train(state)
        elif idx == 1:
            screen_inference(state)
        elif idx == 2:
            screen_edit_config(state)
        elif idx == 3:
            screen_utilities(state)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.clear()
        sys.exit(0)

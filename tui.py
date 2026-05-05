# @claude last-modified: 2026-05-05T06:34:39Z
# @claude last-commit: feat: major update — TUI, augmentation system, gradient checkpointing, optimization bootstrap
#!/usr/bin/env python3
"""
tui.py — Apollo Terminal UI
Navigate with arrow keys / Tab / Enter. Full keyboard-driven interface.
"""

import curses
import os
import sys
import subprocess
import threading
import queue
import shutil
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Colour palette IDs
# ─────────────────────────────────────────────────────────────────────────────
C_NORMAL   = 0
C_HEADER   = 1
C_SELECTED = 2
C_ACTIVE   = 3
C_DIM      = 4
C_SUCCESS  = 5
C_ERROR    = 6
C_WARN     = 7
C_BORDER   = 8
C_TITLE    = 9

SCRIPT_DIR = Path(__file__).resolve().parent


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def init_colors():
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(C_HEADER,   curses.COLOR_BLACK,  curses.COLOR_CYAN)
    curses.init_pair(C_SELECTED, curses.COLOR_BLACK,  curses.COLOR_WHITE)
    curses.init_pair(C_ACTIVE,   curses.COLOR_CYAN,   -1)
    curses.init_pair(C_DIM,      curses.COLOR_WHITE,  -1)
    curses.init_pair(C_SUCCESS,  curses.COLOR_GREEN,  -1)
    curses.init_pair(C_ERROR,    curses.COLOR_RED,    -1)
    curses.init_pair(C_WARN,     curses.COLOR_YELLOW, -1)
    curses.init_pair(C_BORDER,   curses.COLOR_CYAN,   -1)
    curses.init_pair(C_TITLE,    curses.COLOR_WHITE,  -1)


def draw_box(win, y, x, h, w, color=C_BORDER, title=""):
    attr = curses.color_pair(color)
    try:
        win.addch(y,       x,     curses.ACS_ULCORNER, attr)
        win.addch(y,       x+w-1, curses.ACS_URCORNER, attr)
        win.addch(y+h-1,   x,     curses.ACS_LLCORNER, attr)
        win.addch(y+h-1,   x+w-1, curses.ACS_LRCORNER, attr)
        for i in range(1, w-1):
            win.addch(y,     x+i, curses.ACS_HLINE, attr)
            win.addch(y+h-1, x+i, curses.ACS_HLINE, attr)
        for i in range(1, h-1):
            win.addch(y+i, x,     curses.ACS_VLINE, attr)
            win.addch(y+i, x+w-1, curses.ACS_VLINE, attr)
        if title:
            t = f" {title} "
            tx = x + max(1, (w - len(t)) // 2)
            win.addstr(y, tx, t, curses.color_pair(C_HEADER) | curses.A_BOLD)
    except curses.error:
        pass


def safe_addstr(win, y, x, s, attr=0):
    try:
        max_y, max_x = win.getmaxyx()
        if y < 0 or y >= max_y or x < 0 or x >= max_x:
            return
        available = max_x - x - 1
        if available <= 0:
            return
        win.addstr(y, x, s[:available], attr)
    except curses.error:
        pass


def clamp(val, lo, hi):
    return max(lo, min(hi, val))


def strip_quotes(s):
    """Strip surrounding single or double quotes a user may have copy-pasted."""
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'"):
        s = s[1:-1]
    return s


# ─────────────────────────────────────────────────────────────────────────────
# YAML helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_yaml_lines(path):
    try:
        with open(path) as f:
            return f.readlines()
    except Exception:
        return []


def save_yaml_lines(path, lines):
    with open(path, "w") as f:
        f.writelines(lines)


def read_yaml_value(path, key):
    """Quick single-key reader — no dependency on PyYAML at TUI startup."""
    try:
        with open(path) as f:
            for line in f:
                stripped = line.strip()
                if stripped.startswith(key + ":"):
                    val = stripped.split(":", 1)[1].strip()
                    return val
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Auto-discovery helpers
# ─────────────────────────────────────────────────────────────────────────────

def find_configs():
    cfg_dir = SCRIPT_DIR / "configs"
    if not cfg_dir.exists():
        return []
    return sorted(str(f) for f in cfg_dir.glob("*.yaml"))


def find_models():
    models_dir = SCRIPT_DIR / "models"
    if not models_dir.exists():
        return []
    files = []
    for ext in ("*.ckpt", "*.pth", "*.bin"):
        files.extend(sorted(models_dir.glob(ext)))
    return [str(f) for f in files]


def find_last_checkpoint(cfg_path):
    """
    Given a config path, find the most recently modified checkpoint in its
    experiment directory. Returns (ckpt_path_str, feature_dim_str) or (None, None).
    """
    try:
        exp_dir_val = read_yaml_value(cfg_path, "dir")
        exp_name_val = read_yaml_value(cfg_path, "name")
        if not exp_dir_val or not exp_name_val:
            return None, None
        # Resolve relative to SCRIPT_DIR
        exp_dir = Path(exp_dir_val)
        if not exp_dir.is_absolute():
            exp_dir = SCRIPT_DIR / exp_dir
        ckpt_dir = exp_dir / exp_name_val / "checkpoints"
        if not ckpt_dir.exists():
            return None, None
        candidates = [f for f in ckpt_dir.glob("*.ckpt") if f.name != "last.ckpt"]
        if not candidates:
            # fall back to last.ckpt
            last = ckpt_dir / "last.ckpt"
            if last.exists():
                candidates = [last]
            else:
                return None, None
        best = max(candidates, key=lambda f: f.stat().st_mtime)
        # Try to read feature_dim from config
        fdim = read_yaml_value(cfg_path, "feature_dim") or "256"
        return str(best), fdim
    except Exception:
        return None, None


def auto_populate_weights(cfg_path):
    """
    Returns (weights_str, feature_dim_str) using this priority:
      1. Last checkpoint from the config's experiment dir
      2. Single model in models/  (auto-select)
      3. ("", "256")
    """
    # 1. Last checkpoint
    ckpt, fdim = find_last_checkpoint(cfg_path)
    if ckpt:
        return ckpt, fdim

    # 2. models/ dir
    models = find_models()
    if len(models) == 1:
        m = models[0]
        fdim = "384" if "uni" in Path(m).stem.lower() else "256"
        return m, fdim

    return "", "256"


# ─────────────────────────────────────────────────────────────────────────────
# Dropdown widget
# ─────────────────────────────────────────────────────────────────────────────

def show_dropdown(scr, title, items, current_value, y_anchor, x_anchor, max_w=60):
    """
    Render an inline dropdown over the screen.
    Returns the selected item string, or current_value if cancelled.
    items: list of str
    """
    if not items:
        return current_value

    h_scr, w_scr = scr.getmaxyx()
    box_w = min(max_w, w_scr - x_anchor - 2)
    box_h = min(len(items) + 2, h_scr - y_anchor - 2)
    visible_n = box_h - 2

    # Start selection on current value if present
    try:
        sel = items.index(current_value)
    except ValueError:
        sel = 0
    scroll = max(0, sel - visible_n + 1)

    while True:
        # Draw box
        draw_box(scr, y_anchor, x_anchor, box_h, box_w, title=title)
        for li in range(visible_n):
            idx = scroll + li
            if idx >= len(items):
                break
            label = Path(items[idx]).name  # show filename only
            row = y_anchor + 1 + li
            if idx == sel:
                safe_addstr(scr, row, x_anchor + 1, " " * (box_w - 2), curses.color_pair(C_SELECTED))
                safe_addstr(scr, row, x_anchor + 2, f"▶ {label}"[:box_w-3], curses.color_pair(C_SELECTED) | curses.A_BOLD)
            else:
                safe_addstr(scr, row, x_anchor + 2, f"  {label}"[:box_w-3], curses.color_pair(C_DIM))
        scr.refresh()

        k = scr.getch()
        if k in (curses.KEY_UP, ord('k')):
            sel = max(0, sel - 1)
            if sel < scroll:
                scroll = sel
        elif k in (curses.KEY_DOWN, ord('j')):
            sel = min(len(items) - 1, sel + 1)
            if sel >= scroll + visible_n:
                scroll = sel - visible_n + 1
        elif k in (curses.KEY_ENTER, 10, 13):
            return items[sel]
        elif k == 27:
            return current_value


# ─────────────────────────────────────────────────────────────────────────────
# Screens
# ─────────────────────────────────────────────────────────────────────────────

MAIN_MENU = [
    ("  Train",        "train"),
    ("  Inference",    "inference"),
    ("  Config",       "config"),
    ("  Pretrain",     "pretrain"),
    ("  Log Viewer",   "logs"),
    ("  Quit",         "quit"),
]

ASCII_LOGO = [
    "    ___              ____         ",
    "   /   |  ____  ____/ / /___      ",
    "  / /| | / __ \\/ __  / / __ \\   ",
    " / ___ |/ /_/ / /_/ / / /_/ /    ",
    "/_/  |_/ .___/\\__,_/_/\\____/   ",
    "       /_/   audio enhancement   ",
]

# Fields that open a dropdown instead of text input
DROPDOWN_FIELDS = {"config", "weights"}


class TUI:
    def __init__(self, stdscr):
        self.scr = stdscr
        self.h, self.w = stdscr.getmaxyx()
        init_colors()
        curses.curs_set(0)
        self.scr.keypad(True)
        self.scr.timeout(100)

        self.screen = "main"
        self.menu_sel = 0
        self.status = ""
        self.status_color = C_DIM

        # train state
        self.train_fields = self._default_train_fields()
        self.train_sel = 0
        self.train_running = False
        self.train_proc = None
        self.train_log_q = queue.Queue()
        self.train_log_lines = []
        self.train_log_scroll = 0   # lines from bottom (0 = follow tail)
        self.train_log_focus = False  # True = arrow keys scroll log panel

        # inference state
        self.inf_fields = self._default_inf_fields()
        self.inf_sel = 0
        self.inf_running = False
        self.inf_proc = None
        self.inf_log_q = queue.Queue()
        self.inf_log_lines = []
        self.inf_log_scroll = 0     # lines from bottom (0 = follow tail)
        self.inf_log_focus = False

        # config state
        self.cfg_files = find_configs()
        self.cfg_sel = 0
        self.cfg_editor_lines = []
        self.cfg_editor_row = 0
        self.cfg_editor_col = 0
        self.cfg_editor_scroll = 0
        self.cfg_editing = False
        self.cfg_current_file = None

        # pretrain state
        self.pretrain_models = find_models()
        self.pretrain_sel = 0

        # log viewer
        self.log_files = []
        self.log_sel = 0
        self.log_view_lines = []
        self.log_scroll = 0
        self.log_viewing = False

        # editing text input
        self._input_value = ""
        self._input_cursor = 0

    # ── Field defaults with smart auto-population ─────────────────────────────

    def _default_train_fields(self):
        cfg = self._best_config()
        wts, fdim = auto_populate_weights(cfg) if cfg else ("", "256")
        return {
            "config":  cfg,
            "weights": wts,
            "resume":  "no",
        }

    def _default_inf_fields(self):
        cfg = self._best_config()
        wts, fdim = auto_populate_weights(cfg) if cfg else ("", "256")
        return {
            "config":      cfg,
            "input":       "",
            "output":      "output_enhanced.wav",
            "weights":     wts,
            "feature_dim": fdim,
            "device":      "auto",
            "no_chunked":  "no",
        }

    def _best_config(self):
        """Return the most appropriate default config path."""
        cfgs = find_configs()
        if not cfgs:
            return str(SCRIPT_DIR / "configs" / "apollo.yaml")
        # Prefer apollo.yaml (base) over universal
        for c in cfgs:
            if Path(c).name == "apollo.yaml":
                return c
        return cfgs[0]

    def _on_config_changed(self, fields_dict, new_cfg):
        """Called whenever the config field is updated — repopulates weights/feature_dim."""
        fields_dict["config"] = new_cfg
        wts, fdim = auto_populate_weights(new_cfg)
        if wts:
            fields_dict["weights"] = wts
        if "feature_dim" in fields_dict:
            fields_dict["feature_dim"] = fdim
        self.set_status(
            f"Config: {Path(new_cfg).name}" +
            (f"  →  weights: {Path(wts).name}" if wts else "  →  no checkpoint found"),
            C_SUCCESS if wts else C_WARN,
        )

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self):
        while True:
            self.h, self.w = self.scr.getmaxyx()
            self.scr.erase()
            self._draw_header()
            self._draw_statusbar()

            if self.screen == "main":
                self._draw_main()
            elif self.screen == "train":
                self._draw_train()
            elif self.screen == "inference":
                self._draw_inference()
            elif self.screen == "config":
                self._draw_config()
            elif self.screen == "pretrain":
                self._draw_pretrain()
            elif self.screen == "logs":
                self._draw_logs()

            self.scr.refresh()
            self._drain_queues()

            key = self.scr.getch()
            if key == -1:
                continue

            result = self._handle_key(key)
            if result == "quit":
                break

    # ── Key dispatcher ────────────────────────────────────────────────────────

    def _handle_key(self, key):
        if self.screen == "main":
            return self._key_main(key)
        elif self.screen == "train":
            return self._key_train(key)
        elif self.screen == "inference":
            return self._key_inference(key)
        elif self.screen == "config":
            return self._key_config(key)
        elif self.screen == "pretrain":
            return self._key_pretrain(key)
        elif self.screen == "logs":
            return self._key_logs(key)

    # ─────────────────────────────────────────────────────────────────────────
    # Header / status
    # ─────────────────────────────────────────────────────────────────────────

    def _draw_header(self):
        bar = " Apollo TUI "
        safe_addstr(self.scr, 0, 0, " " * self.w, curses.color_pair(C_HEADER) | curses.A_BOLD)
        safe_addstr(self.scr, 0, max(0, (self.w - len(bar)) // 2), bar,
                    curses.color_pair(C_HEADER) | curses.A_BOLD)
        nav = "[↑↓] Navigate  [Enter] Select/Dropdown  [Esc/Q] Back  [Tab] Next field"
        safe_addstr(self.scr, 0, self.w - len(nav) - 1, nav, curses.color_pair(C_HEADER))

    def _draw_statusbar(self):
        y = self.h - 1
        safe_addstr(self.scr, y, 0, " " * self.w, curses.color_pair(C_DIM))
        if self.status:
            safe_addstr(self.scr, y, 1, self.status, curses.color_pair(self.status_color))

    def set_status(self, msg, color=C_DIM):
        self.status = msg
        self.status_color = color

    # ─────────────────────────────────────────────────────────────────────────
    # MAIN MENU
    # ─────────────────────────────────────────────────────────────────────────

    def _draw_main(self):
        logo_x = max(0, (self.w - 36) // 2)
        for i, line in enumerate(ASCII_LOGO):
            safe_addstr(self.scr, 2 + i, logo_x, line, curses.color_pair(C_ACTIVE) | curses.A_BOLD)

        menu_h = len(MAIN_MENU) + 4
        menu_w = 30
        menu_y = 10
        menu_x = max(0, (self.w - menu_w) // 2)
        draw_box(self.scr, menu_y, menu_x, menu_h, menu_w, title="Menu")

        for i, (label, _) in enumerate(MAIN_MENU):
            row = menu_y + 2 + i
            if i == self.menu_sel:
                safe_addstr(self.scr, row, menu_x + 1, " " * (menu_w - 2), curses.color_pair(C_SELECTED))
                safe_addstr(self.scr, row, menu_x + 2, label, curses.color_pair(C_SELECTED) | curses.A_BOLD)
                safe_addstr(self.scr, row, menu_x + 2, "▶", curses.color_pair(C_SELECTED) | curses.A_BOLD)
            else:
                safe_addstr(self.scr, row, menu_x + 2, label, curses.color_pair(C_DIM))

    def _key_main(self, key):
        if key in (curses.KEY_UP, ord('k')):
            self.menu_sel = (self.menu_sel - 1) % len(MAIN_MENU)
        elif key in (curses.KEY_DOWN, ord('j')):
            self.menu_sel = (self.menu_sel + 1) % len(MAIN_MENU)
        elif key in (curses.KEY_ENTER, 10, 13):
            _, action = MAIN_MENU[self.menu_sel]
            if action == "quit":
                return "quit"
            elif action == "pretrain":
                self.pretrain_models = find_models()
                self.screen = "pretrain"
            elif action == "logs":
                self.log_files = self._find_log_dirs()
                self.log_sel = 0
                self.log_viewing = False
                self.screen = "logs"
            else:
                self.screen = action
        elif key in (ord('q'), ord('Q')):
            return "quit"

    # ─────────────────────────────────────────────────────────────────────────
    # TRAIN SCREEN
    # ─────────────────────────────────────────────────────────────────────────

    TRAIN_LABELS = [
        ("config",   "Config YAML",  "↵ dropdown — apollo.yaml / apollo_uni.yaml"),
        ("weights",  "Weights",      "↵ dropdown — models/ checkpoints, or blank = HuggingFace"),
        ("resume",   "Resume",       "Resume from last checkpoint? (yes/no)"),
    ]

    def _draw_train(self):
        panel_h = self.h - 4
        left_w  = self.w // 2 - 1
        right_w = self.w - left_w - 1

        draw_box(self.scr, 1, 0,       panel_h, left_w,  title="Train Settings")
        draw_box(self.scr, 1, left_w,  panel_h, right_w, title="Output Log")

        for i, (key, label, hint) in enumerate(self.TRAIN_LABELS):
            row = 3 + i * 3
            is_sel = (i == self.train_sel)
            attr_label = curses.color_pair(C_ACTIVE if is_sel else C_DIM)
            attr_val   = curses.color_pair(C_SELECTED if is_sel else C_NORMAL) | (curses.A_BOLD if is_sel else 0)

            # Show dropdown indicator for dropdown fields
            lbl_text = f"{'▼ ' if key in DROPDOWN_FIELDS else ''}{label}:"
            safe_addstr(self.scr, row, 2, lbl_text, attr_label)
            val = self.train_fields[key]
            val_display = Path(val).name if val and key in DROPDOWN_FIELDS else (val or "(not set)")
            safe_addstr(self.scr, row + 1, 4, val_display[:left_w - 6], attr_val)
            if is_sel:
                safe_addstr(self.scr, row + 1, 4 + len(val_display[:left_w - 6]) + 1,
                             hint[:left_w - 10], curses.color_pair(C_DIM))

        btn_row = 3 + len(self.TRAIN_LABELS) * 3 + 1
        is_btn = (self.train_sel == len(self.TRAIN_LABELS))
        lbl = "[ ■ STOP ]" if self.train_running else "[ ▶ START TRAINING ]"
        col = (C_ERROR if self.train_running else C_SELECTED) if is_btn else \
              (C_ERROR if self.train_running else C_ACTIVE)
        attr = curses.color_pair(col) | (curses.A_BOLD if is_btn else 0)
        safe_addstr(self.scr, btn_row, 4, lbl, attr)

        log_inner_h = panel_h - 4
        log_inner_w = right_w - 3
        total_lines = len(self.train_log_lines)
        # clamp scroll so it never exceeds available history
        self.train_log_scroll = min(self.train_log_scroll, max(0, total_lines - log_inner_h))
        # 0 = tail; scroll > 0 means we're looking back into history
        if self.train_log_scroll == 0:
            visible = self.train_log_lines[-log_inner_h:] if self.train_log_lines else []
        else:
            end   = total_lines - self.train_log_scroll
            start = max(0, end - log_inner_h)
            visible = self.train_log_lines[start:end]
        for li, line in enumerate(visible):
            clean = line.rstrip()
            color = C_SUCCESS if any(w in clean for w in ("Epoch", "epoch", "loss", "Loss")) else \
                    C_ERROR   if any(w in clean for w in ("Error", "error", "Traceback")) else C_DIM
            safe_addstr(self.scr, 3 + li, left_w + 2, clean[:log_inner_w], curses.color_pair(color))

        # scroll indicator in top-right of log box
        focus_attr = curses.color_pair(C_ACTIVE) | curses.A_BOLD if self.train_log_focus else curses.color_pair(C_DIM)
        if self.train_log_scroll > 0:
            pct = int(100 * (total_lines - self.train_log_scroll) / max(1, total_lines))
            scroll_hint = f" ↑{self.train_log_scroll}L ({pct}%) [Tab]=focus [PgUp/Dn] "
        else:
            scroll_hint = " [Tab]=focus log [PgUp/↑] scroll " if not self.train_log_focus else " LOG FOCUS — [PgUp/↑↓/PgDn] scroll  [Tab]=back "
        safe_addstr(self.scr, 2, left_w + right_w - len(scroll_hint) - 1, scroll_hint, focus_attr)

        if self.train_running:
            safe_addstr(self.scr, 2, left_w + 2, " ● RUNNING ", curses.color_pair(C_SUCCESS) | curses.A_BOLD)

    def _key_train(self, key):
        n_fields = len(self.TRAIN_LABELS)
        panel_h = self.h - 4
        log_inner_h = panel_h - 4

        if key in (27, ord('q')):
            self.train_log_focus = False
            self.screen = "main"
        elif key == 9:  # Tab — toggle log focus
            self.train_log_focus = not self.train_log_focus
        elif self.train_log_focus:
            # Arrow / PgUp / PgDn scroll the log panel
            total = len(self.train_log_lines)
            max_scroll = max(0, total - log_inner_h)
            if key in (curses.KEY_UP, ord('k')):
                self.train_log_scroll = min(self.train_log_scroll + 1, max_scroll)
            elif key in (curses.KEY_DOWN, ord('j')):
                self.train_log_scroll = max(0, self.train_log_scroll - 1)
            elif key == curses.KEY_PPAGE:  # Page Up
                self.train_log_scroll = min(self.train_log_scroll + log_inner_h, max_scroll)
            elif key == curses.KEY_NPAGE:  # Page Down
                self.train_log_scroll = max(0, self.train_log_scroll - log_inner_h)
            elif key in (curses.KEY_HOME,):
                self.train_log_scroll = max_scroll  # jump to top
            elif key in (curses.KEY_END,):
                self.train_log_scroll = 0           # jump to tail
        else:
            # Normal field navigation
            if key in (curses.KEY_UP, ord('k')):
                self.train_sel = (self.train_sel - 1) % (n_fields + 1)
            elif key in (curses.KEY_DOWN, ord('j')):
                self.train_sel = (self.train_sel + 1) % (n_fields + 1)
            elif key in (curses.KEY_ENTER, 10, 13, ord(' ')):
                if self.train_sel < n_fields:
                    key_name = self.TRAIN_LABELS[self.train_sel][0]
                    self._activate_field("train", key_name)
                else:
                    if self.train_running:
                        self._stop_process("train")
                    else:
                        self._start_train()
            # PgUp/Dn scroll log even without focus
            elif key == curses.KEY_PPAGE:
                total = len(self.train_log_lines)
                max_scroll = max(0, total - log_inner_h)
                self.train_log_scroll = min(self.train_log_scroll + log_inner_h, max_scroll)
            elif key == curses.KEY_NPAGE:
                self.train_log_scroll = max(0, self.train_log_scroll - log_inner_h)

    def _activate_field(self, screen, key_name):
        """Open dropdown or text editor depending on field type."""
        fields = self.train_fields if screen == "train" else self.inf_fields
        if key_name == "config":
            items = find_configs()
            if not items:
                self.set_status("No configs found in configs/", C_WARN)
                return
            chosen = show_dropdown(self.scr, "Select Config", items,
                                   fields["config"], 3, 2, max_w=self.w // 2 - 4)
            if chosen != fields["config"]:
                self._on_config_changed(fields, chosen)
        elif key_name == "weights":
            models = find_models()
            # Always offer "(none / HuggingFace)" as first option
            items = ["(none — use HuggingFace)"] + models
            current = fields.get("weights", "")
            chosen = show_dropdown(self.scr, "Select Weights", items,
                                   current if current in items else items[0],
                                   3, 2, max_w=self.w // 2 - 4)
            if chosen == "(none — use HuggingFace)":
                fields["weights"] = ""
                fields["feature_dim"] = "256"
                self.set_status("Weights: HuggingFace (auto-download)", C_SUCCESS)
            else:
                fields["weights"] = chosen
                fdim = "384" if "uni" in Path(chosen).stem.lower() else "256"
                fields["feature_dim"] = fdim
                self.set_status(f"Weights: {Path(chosen).name}  (feature_dim={fdim})", C_SUCCESS)
        else:
            self._edit_field(fields, key_name)

    def _start_train(self):
        cfg  = strip_quotes(self.train_fields["config"])
        wts  = strip_quotes(self.train_fields["weights"])
        res  = self.train_fields["resume"].lower() in ("yes", "y", "1", "true")

        if not cfg or not Path(cfg).exists():
            self.set_status(f"Config not found: {cfg}", C_ERROR)
            return

        python = self._venv_python()
        cmd = [python, str(SCRIPT_DIR / "train.py"), "--conf_dir", cfg]
        if wts:
            cmd += ["--weights_path", wts]
        if res:
            cmd.append("--resume")

        self.train_log_lines = [f"[apollo] Starting: {' '.join(cmd)}\n"]
        self.train_log_scroll = 0
        self.train_log_focus = False
        self.train_running = True
        self.train_proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, cwd=str(SCRIPT_DIR), bufsize=1
        )
        t = threading.Thread(target=self._stream_output,
                             args=(self.train_proc, self.train_log_q), daemon=True)
        t.start()
        self.set_status("Training started", C_SUCCESS)

    # ─────────────────────────────────────────────────────────────────────────
    # INFERENCE SCREEN
    # ─────────────────────────────────────────────────────────────────────────

    INF_LABELS = [
        ("config",      "Config YAML",  "↵ dropdown — selects config and auto-fills weights/dim"),
        ("input",       "Input audio",  "Path to input .wav/.flac/.mp3 file"),
        ("output",      "Output path",  "Where to save the enhanced file"),
        ("weights",     "Weights",      "↵ dropdown — models/ or blank = HuggingFace"),
        ("feature_dim", "Feature dim",  "256 = base model, 384 = universal (auto-set from weights)"),
        ("device",      "Device",       "auto / cuda / cpu / cuda:1"),
        ("no_chunked",  "No chunked",   "Disable chunked processing? (yes/no)"),
    ]

    def _draw_inference(self):
        panel_h = self.h - 4
        left_w  = self.w // 2 - 1
        right_w = self.w - left_w - 1

        draw_box(self.scr, 1, 0,       panel_h, left_w,  title="Inference Settings")
        draw_box(self.scr, 1, left_w,  panel_h, right_w, title="Output Log")

        for i, (key, label, hint) in enumerate(self.INF_LABELS):
            row = 3 + i * 3
            is_sel = (i == self.inf_sel)
            attr_label = curses.color_pair(C_ACTIVE if is_sel else C_DIM)
            attr_val   = curses.color_pair(C_SELECTED if is_sel else C_NORMAL) | (curses.A_BOLD if is_sel else 0)

            lbl_text = f"{'▼ ' if key in DROPDOWN_FIELDS else ''}{label}:"
            safe_addstr(self.scr, row, 2, lbl_text, attr_label)
            val = self.inf_fields[key]
            val_display = Path(val).name if val and key in DROPDOWN_FIELDS else (val or "(not set)")
            safe_addstr(self.scr, row + 1, 4, val_display[:left_w - 6], attr_val)
            if is_sel:
                safe_addstr(self.scr, row + 1, 4 + len(val_display[:left_w - 6]) + 1,
                             hint[:left_w - 10], curses.color_pair(C_DIM))

        btn_row = 3 + len(self.INF_LABELS) * 3 + 1
        is_btn = (self.inf_sel == len(self.INF_LABELS))
        lbl = "[ ■ STOP ]" if self.inf_running else "[ ▶ RUN INFERENCE ]"
        col = (C_ERROR if self.inf_running else C_SELECTED) if is_btn else \
              (C_ERROR if self.inf_running else C_ACTIVE)
        safe_addstr(self.scr, btn_row, 4, lbl, curses.color_pair(col) | (curses.A_BOLD if is_btn else 0))

        log_inner_h = panel_h - 4
        log_inner_w = right_w - 3
        total_lines = len(self.inf_log_lines)
        self.inf_log_scroll = min(self.inf_log_scroll, max(0, total_lines - log_inner_h))
        if self.inf_log_scroll == 0:
            visible = self.inf_log_lines[-log_inner_h:] if self.inf_log_lines else []
        else:
            end   = total_lines - self.inf_log_scroll
            start = max(0, end - log_inner_h)
            visible = self.inf_log_lines[start:end]
        for li, line in enumerate(visible):
            clean = line.rstrip()
            color = C_SUCCESS if "Saved" in clean else \
                    C_ERROR   if "Error" in clean else C_DIM
            safe_addstr(self.scr, 3 + li, left_w + 2, clean[:log_inner_w], curses.color_pair(color))

        focus_attr = curses.color_pair(C_ACTIVE) | curses.A_BOLD if self.inf_log_focus else curses.color_pair(C_DIM)
        if self.inf_log_scroll > 0:
            pct = int(100 * (total_lines - self.inf_log_scroll) / max(1, total_lines))
            scroll_hint = f" ↑{self.inf_log_scroll}L ({pct}%) [Tab]=focus [PgUp/Dn] "
        else:
            scroll_hint = " [Tab]=focus log [PgUp/↑] scroll " if not self.inf_log_focus else " LOG FOCUS — [PgUp/↑↓/PgDn] scroll  [Tab]=back "
        safe_addstr(self.scr, 2, left_w + right_w - len(scroll_hint) - 1, scroll_hint, focus_attr)

        if self.inf_running:
            safe_addstr(self.scr, 2, left_w + 2, " ● RUNNING ", curses.color_pair(C_SUCCESS) | curses.A_BOLD)

    def _key_inference(self, key):
        n_fields = len(self.INF_LABELS)
        panel_h = self.h - 4
        log_inner_h = panel_h - 4

        if key in (27, ord('q')):
            self.inf_log_focus = False
            self.screen = "main"
        elif key == 9:  # Tab — toggle log focus
            self.inf_log_focus = not self.inf_log_focus
        elif self.inf_log_focus:
            total = len(self.inf_log_lines)
            max_scroll = max(0, total - log_inner_h)
            if key in (curses.KEY_UP, ord('k')):
                self.inf_log_scroll = min(self.inf_log_scroll + 1, max_scroll)
            elif key in (curses.KEY_DOWN, ord('j')):
                self.inf_log_scroll = max(0, self.inf_log_scroll - 1)
            elif key == curses.KEY_PPAGE:
                self.inf_log_scroll = min(self.inf_log_scroll + log_inner_h, max_scroll)
            elif key == curses.KEY_NPAGE:
                self.inf_log_scroll = max(0, self.inf_log_scroll - log_inner_h)
            elif key == curses.KEY_HOME:
                self.inf_log_scroll = max_scroll
            elif key == curses.KEY_END:
                self.inf_log_scroll = 0
        else:
            if key in (curses.KEY_UP, ord('k')):
                self.inf_sel = (self.inf_sel - 1) % (n_fields + 1)
            elif key in (curses.KEY_DOWN, ord('j')):
                self.inf_sel = (self.inf_sel + 1) % (n_fields + 1)
            elif key in (curses.KEY_ENTER, 10, 13, ord(' ')):
                if self.inf_sel < n_fields:
                    key_name = self.INF_LABELS[self.inf_sel][0]
                    self._activate_field("inf", key_name)
                else:
                    if self.inf_running:
                        self._stop_process("inf")
                    else:
                        self._start_inference()
            elif key == curses.KEY_PPAGE:
                total = len(self.inf_log_lines)
                max_scroll = max(0, total - log_inner_h)
                self.inf_log_scroll = min(self.inf_log_scroll + log_inner_h, max_scroll)
            elif key == curses.KEY_NPAGE:
                self.inf_log_scroll = max(0, self.inf_log_scroll - log_inner_h)

    def _start_inference(self):
        inp  = strip_quotes(self.inf_fields["input"])
        out  = strip_quotes(self.inf_fields["output"])
        wts  = strip_quotes(self.inf_fields["weights"])
        fdim = self.inf_fields["feature_dim"]
        dev  = self.inf_fields["device"]
        nc   = self.inf_fields["no_chunked"].lower() in ("yes", "y", "1", "true")

        if not inp:
            self.set_status("Input file required", C_ERROR)
            return
        if not Path(inp).exists():
            self.set_status(f"Input not found: {inp}", C_ERROR)
            return

        python = self._venv_python()
        cmd = [python, str(SCRIPT_DIR / "inference.py"),
               "--in_wav", inp, "--out_wav", out,
               "--feature_dim", fdim, "--device", dev]
        if wts:
            cmd += ["--weights", wts]
        if nc:
            cmd.append("--no_chunked")

        self.inf_log_lines = [f"[apollo] {' '.join(cmd)}\n"]
        self.inf_log_scroll = 0
        self.inf_log_focus = False
        self.inf_running = True
        self.inf_proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, cwd=str(SCRIPT_DIR), bufsize=1
        )
        t = threading.Thread(target=self._stream_output,
                             args=(self.inf_proc, self.inf_log_q), daemon=True)
        t.start()
        self.set_status("Inference started", C_SUCCESS)

    # ─────────────────────────────────────────────────────────────────────────
    # CONFIG EDITOR
    # ─────────────────────────────────────────────────────────────────────────

    def _draw_config(self):
        panel_h = self.h - 4

        if not self.cfg_editing:
            list_w = min(50, self.w - 4)
            draw_box(self.scr, 1, 0, panel_h, list_w, title="Config Files")
            safe_addstr(self.scr, 3, 2, "Select a config to edit:", curses.color_pair(C_DIM))

            for i, cf in enumerate(self.cfg_files):
                row = 5 + i
                label = Path(cf).name
                if i == self.cfg_sel:
                    safe_addstr(self.scr, row, 2, " " * (list_w - 3), curses.color_pair(C_SELECTED))
                    safe_addstr(self.scr, row, 3, f"▶ {label}", curses.color_pair(C_SELECTED) | curses.A_BOLD)
                else:
                    safe_addstr(self.scr, row, 3, f"  {label}", curses.color_pair(C_DIM))

            if not self.cfg_files:
                safe_addstr(self.scr, 5, 3, "No .yaml files found in configs/", curses.color_pair(C_WARN))

            safe_addstr(self.scr, panel_h - 1, 2,
                        "[Enter] Edit  [N] New config  [D] Duplicate  [Q/Esc] Back",
                        curses.color_pair(C_DIM))
        else:
            fname = Path(self.cfg_current_file).name
            draw_box(self.scr, 1, 0, panel_h, self.w, title=f"Editing: {fname}")

            edit_h = panel_h - 4
            edit_w = self.w - 4
            visible = self.cfg_editor_lines[self.cfg_editor_scroll:self.cfg_editor_scroll + edit_h]

            for li, line in enumerate(visible):
                abs_row = self.cfg_editor_scroll + li
                is_cur  = (abs_row == self.cfg_editor_row)
                display = line.rstrip('\n')
                attr = curses.color_pair(C_SELECTED) | curses.A_BOLD if is_cur else curses.color_pair(C_NORMAL)
                safe_addstr(self.scr, 3 + li, 1, " " * (self.w - 2), attr if is_cur else 0)
                safe_addstr(self.scr, 3 + li, 2, f"{abs_row+1:4d} ", curses.color_pair(C_DIM))
                safe_addstr(self.scr, 3 + li, 7, display[:edit_w], attr)

                if is_cur:
                    cursor_x = 7 + self.cfg_editor_col
                    if cursor_x < self.w - 2:
                        ch = display[self.cfg_editor_col] if self.cfg_editor_col < len(display) else ' '
                        try:
                            self.scr.addch(3 + li, cursor_x, ch,
                                           curses.color_pair(C_ACTIVE) | curses.A_REVERSE | curses.A_BOLD)
                        except curses.error:
                            pass

            safe_addstr(self.scr, panel_h - 1, 2,
                        "[↑↓←→] Move  [Enter] Newline  [BS] Delete  [Ctrl+S] Save  [Esc] Back",
                        curses.color_pair(C_DIM))

    def _key_config(self, key):
        if not self.cfg_editing:
            if key in (27, ord('q')):
                self.screen = "main"
            elif key in (curses.KEY_UP, ord('k')):
                self.cfg_sel = max(0, self.cfg_sel - 1)
            elif key in (curses.KEY_DOWN, ord('j')):
                self.cfg_sel = min(len(self.cfg_files) - 1, self.cfg_sel + 1)
            elif key in (curses.KEY_ENTER, 10, 13):
                if self.cfg_files:
                    self._open_config(self.cfg_files[self.cfg_sel])
            elif key in (ord('n'), ord('N')):
                self._new_config()
            elif key in (ord('d'), ord('D')):
                if self.cfg_files:
                    self._duplicate_config(self.cfg_files[self.cfg_sel])
        else:
            self._key_editor(key)

    def _open_config(self, path):
        self.cfg_current_file = path
        self.cfg_editor_lines = load_yaml_lines(path)
        if not self.cfg_editor_lines:
            self.cfg_editor_lines = ["\n"]
        self.cfg_editor_row    = 0
        self.cfg_editor_col    = 0
        self.cfg_editor_scroll = 0
        self.cfg_editing = True

    def _key_editor(self, key):
        lines = self.cfg_editor_lines
        row   = self.cfg_editor_row
        col   = self.cfg_editor_col

        def cur_line():
            return lines[row].rstrip('\n') if row < len(lines) else ""

        panel_h = self.h - 4
        edit_h  = panel_h - 4

        if key == 27:
            self.cfg_editing = False
            self.cfg_files = find_configs()
        elif key == 19:  # Ctrl+S
            save_yaml_lines(self.cfg_current_file, lines)
            self.set_status(f"Saved {Path(self.cfg_current_file).name}", C_SUCCESS)
        elif key == curses.KEY_UP:
            self.cfg_editor_row = max(0, row - 1)
            self.cfg_editor_col = min(col, len(cur_line()))
        elif key == curses.KEY_DOWN:
            self.cfg_editor_row = min(len(lines) - 1, row + 1)
            self.cfg_editor_col = min(col, len(cur_line()))
        elif key == curses.KEY_LEFT:
            if col > 0:
                self.cfg_editor_col -= 1
            elif row > 0:
                self.cfg_editor_row -= 1
                self.cfg_editor_col = len(cur_line())
        elif key == curses.KEY_RIGHT:
            line = cur_line()
            if col < len(line):
                self.cfg_editor_col += 1
            elif row < len(lines) - 1:
                self.cfg_editor_row += 1
                self.cfg_editor_col = 0
        elif key == curses.KEY_HOME:
            self.cfg_editor_col = 0
        elif key == curses.KEY_END:
            self.cfg_editor_col = len(cur_line())
        elif key in (curses.KEY_ENTER, 10, 13):
            line = lines[row]
            before = line[:col]
            after  = line[col:]
            indent = len(line) - len(line.lstrip())
            lines[row] = before + '\n'
            lines.insert(row + 1, ' ' * indent + after.lstrip('\n'))
            self.cfg_editor_row = row + 1
            self.cfg_editor_col = indent
        elif key in (curses.KEY_BACKSPACE, 127, 8):
            if col > 0:
                line = lines[row]
                lines[row] = line[:col-1] + line[col:]
                self.cfg_editor_col -= 1
            elif row > 0:
                prev = lines[row - 1].rstrip('\n')
                curr = lines[row]
                self.cfg_editor_col = len(prev)
                lines[row - 1] = prev + curr
                lines.pop(row)
                self.cfg_editor_row -= 1
        elif key == curses.KEY_DC:
            line = lines[row]
            stripped = line.rstrip('\n')
            if col < len(stripped):
                lines[row] = stripped[:col] + stripped[col+1:] + '\n'
            elif row < len(lines) - 1:
                lines[row] = stripped + lines[row+1]
                lines.pop(row+1)
        elif 32 <= key <= 126:
            line = lines[row]
            ch   = chr(key)
            stripped = line.rstrip('\n')
            lines[row] = stripped[:col] + ch + stripped[col:] + '\n'
            self.cfg_editor_col += 1

        row = self.cfg_editor_row
        if row < self.cfg_editor_scroll:
            self.cfg_editor_scroll = row
        elif row >= self.cfg_editor_scroll + edit_h:
            self.cfg_editor_scroll = row - edit_h + 1

    def _new_config(self):
        template_src = SCRIPT_DIR / "configs" / "apollo.yaml"
        candidate = SCRIPT_DIR / "configs" / "my_config.yaml"
        i = 1
        while candidate.exists():
            candidate = SCRIPT_DIR / "configs" / f"my_config_{i}.yaml"
            i += 1
        if template_src.exists():
            shutil.copy(str(template_src), str(candidate))
        else:
            candidate.write_text("exp:\n  dir: ./Exps\n  name: MyRun\n")
        self.cfg_files = find_configs()
        self.cfg_sel = self.cfg_files.index(str(candidate)) if str(candidate) in self.cfg_files else 0
        self._open_config(str(candidate))
        self.set_status(f"Created {candidate.name}", C_SUCCESS)

    def _duplicate_config(self, src):
        sp = Path(src)
        candidate = sp.parent / f"{sp.stem}_copy{sp.suffix}"
        i = 1
        while candidate.exists():
            candidate = sp.parent / f"{sp.stem}_copy{i}{sp.suffix}"
            i += 1
        shutil.copy(str(sp), str(candidate))
        self.cfg_files = find_configs()
        self.cfg_sel = self.cfg_files.index(str(candidate)) if str(candidate) in self.cfg_files else 0
        self.set_status(f"Duplicated as {candidate.name}", C_SUCCESS)

    # ─────────────────────────────────────────────────────────────────────────
    # PRETRAIN SCREEN
    # ─────────────────────────────────────────────────────────────────────────

    def _draw_pretrain(self):
        panel_h = self.h - 4
        draw_box(self.scr, 1, 0, panel_h, self.w, title="Pretrained / Local Models")

        safe_addstr(self.scr, 3, 3, "Models directory: " + str(SCRIPT_DIR / "models"),
                    curses.color_pair(C_DIM))
        safe_addstr(self.scr, 4, 3, "[U] use in Inference  [T] use in Train  [R] refresh",
                    curses.color_pair(C_DIM))

        if not self.pretrain_models:
            safe_addstr(self.scr, 6, 3, "No models found in models/", curses.color_pair(C_WARN))
            safe_addstr(self.scr, 7, 3, "Place .ckpt / .pth / .bin files there, then press [R].",
                        curses.color_pair(C_DIM))
        else:
            for i, path in enumerate(self.pretrain_models):
                row = 6 + i
                p   = Path(path)
                size_mb = p.stat().st_size / 1e6 if p.exists() else 0
                label = f"{p.name}  ({size_mb:.1f} MB)"
                if i == self.pretrain_sel:
                    safe_addstr(self.scr, row, 2, " " * (self.w - 4), curses.color_pair(C_SELECTED))
                    safe_addstr(self.scr, row, 3, f"▶ {label}", curses.color_pair(C_SELECTED) | curses.A_BOLD)
                else:
                    safe_addstr(self.scr, row, 3, f"  {label}", curses.color_pair(C_DIM))

        safe_addstr(self.scr, panel_h - 1, 2,
                    "[↑↓] Select  [U] Use in Inference  [T] Use in Train  [R] Refresh  [Q/Esc] Back",
                    curses.color_pair(C_DIM))

    def _key_pretrain(self, key):
        if key in (27, ord('q')):
            self.screen = "main"
        elif key in (curses.KEY_UP, ord('k')):
            self.pretrain_sel = max(0, self.pretrain_sel - 1)
        elif key in (curses.KEY_DOWN, ord('j')):
            self.pretrain_sel = min(max(0, len(self.pretrain_models) - 1), self.pretrain_sel + 1)
        elif key in (ord('r'), ord('R')):
            self.pretrain_models = find_models()
            self.set_status("Refreshed model list", C_SUCCESS)
        elif key in (ord('u'), ord('U')):
            if self.pretrain_models:
                path = self.pretrain_models[self.pretrain_sel]
                self.inf_fields["weights"] = path
                fdim = "384" if "uni" in Path(path).stem.lower() else "256"
                self.inf_fields["feature_dim"] = fdim
                self.set_status(f"Set inference weights: {Path(path).name}", C_SUCCESS)
                self.screen = "inference"
        elif key in (ord('t'), ord('T')):
            if self.pretrain_models:
                path = self.pretrain_models[self.pretrain_sel]
                self.train_fields["weights"] = path
                self.set_status(f"Set train weights: {Path(path).name}", C_SUCCESS)
                self.screen = "train"

    # ─────────────────────────────────────────────────────────────────────────
    # LOG VIEWER
    # ─────────────────────────────────────────────────────────────────────────

    def _find_log_dirs(self):
        exps = SCRIPT_DIR / "Exps"
        if not exps.exists():
            return []
        runs = []
        for run_dir in sorted(exps.iterdir()):
            if (run_dir / "logs").exists() or (run_dir / "checkpoints").exists():
                runs.append(str(run_dir))
        return runs

    def _draw_logs(self):
        panel_h = self.h - 4

        if not self.log_viewing:
            draw_box(self.scr, 1, 0, panel_h, self.w, title="Experiment Log Viewer")
            safe_addstr(self.scr, 3, 3, f"Exps dir: {SCRIPT_DIR / 'Exps'}", curses.color_pair(C_DIM))

            if not self.log_files:
                safe_addstr(self.scr, 5, 3, "No experiment runs found.", curses.color_pair(C_WARN))
            else:
                for i, path in enumerate(self.log_files):
                    row = 5 + i
                    label = Path(path).name
                    if i == self.log_sel:
                        safe_addstr(self.scr, row, 2, " " * (self.w - 4), curses.color_pair(C_SELECTED))
                        safe_addstr(self.scr, row, 3, f"▶ {label}", curses.color_pair(C_SELECTED) | curses.A_BOLD)
                    else:
                        safe_addstr(self.scr, row, 3, f"  {label}", curses.color_pair(C_DIM))

            safe_addstr(self.scr, panel_h - 1, 2,
                        "[↑↓] Select  [Enter] View  [R] Refresh  [Q/Esc] Back",
                        curses.color_pair(C_DIM))
        else:
            draw_box(self.scr, 1, 0, panel_h, self.w, title="Log Output")
            inner_h = panel_h - 4
            inner_w = self.w - 4
            visible = self.log_view_lines[self.log_scroll:self.log_scroll + inner_h]
            for li, line in enumerate(visible):
                clean = line.rstrip()
                color = C_SUCCESS if any(w in clean for w in ("Epoch", "step", "loss", "val")) else \
                        C_ERROR   if any(w in clean for w in ("Error", "Traceback")) else C_DIM
                safe_addstr(self.scr, 3 + li, 2, clean[:inner_w], curses.color_pair(color))

            pct = 100 * (self.log_scroll + inner_h) // max(1, len(self.log_view_lines))
            safe_addstr(self.scr, panel_h - 1, 2,
                        f"[↑↓/PgUp/PgDn] Scroll  [G] Bottom  [g] Top  [Q/Esc] Back  ({pct}%)",
                        curses.color_pair(C_DIM))

    def _key_logs(self, key):
        if not self.log_viewing:
            if key in (27, ord('q')):
                self.screen = "main"
            elif key in (curses.KEY_UP, ord('k')):
                self.log_sel = max(0, self.log_sel - 1)
            elif key in (curses.KEY_DOWN, ord('j')):
                self.log_sel = min(max(0, len(self.log_files) - 1), self.log_sel + 1)
            elif key in (ord('r'), ord('R')):
                self.log_files = self._find_log_dirs()
            elif key in (curses.KEY_ENTER, 10, 13):
                if self.log_files:
                    self._load_log(self.log_files[self.log_sel])
        else:
            panel_h = self.h - 4
            inner_h = panel_h - 4
            total   = len(self.log_view_lines)
            if key in (27, ord('q')):
                self.log_viewing = False
            elif key in (curses.KEY_UP, ord('k')):
                self.log_scroll = max(0, self.log_scroll - 1)
            elif key in (curses.KEY_DOWN, ord('j')):
                self.log_scroll = min(max(0, total - inner_h), self.log_scroll + 1)
            elif key == curses.KEY_PPAGE:
                self.log_scroll = max(0, self.log_scroll - inner_h)
            elif key == curses.KEY_NPAGE:
                self.log_scroll = min(max(0, total - inner_h), self.log_scroll + inner_h)
            elif key == ord('G'):
                self.log_scroll = max(0, total - inner_h)
            elif key == ord('g'):
                self.log_scroll = 0

    def _load_log(self, exp_dir):
        lines = []
        ckpt_dir = Path(exp_dir) / "checkpoints"
        if ckpt_dir.exists():
            ckpts = sorted(ckpt_dir.glob("*.ckpt"))
            if ckpts:
                lines.append(f"=== Checkpoints ({len(ckpts)} found) ===\n")
                for ck in ckpts[-10:]:
                    lines.append(f"  {ck.name}\n")
                lines.append("\n")

        log_dir = Path(exp_dir) / "logs"
        if log_dir.exists():
            tfevents = list(log_dir.rglob("events.out.tfevents.*"))
            if tfevents:
                lines.append(f"=== TensorBoard event files: {len(tfevents)} ===\n")
                lines.append("  Run:  tensorboard --logdir " + str(log_dir) + "\n\n")

        bk = Path(exp_dir) / "best_k_models.json"
        if bk.exists():
            lines.append("=== best_k_models.json ===\n")
            lines += open(bk).readlines()
            lines.append("\n")

        if not lines:
            lines = ["No log data found in this experiment directory.\n"]

        self.log_view_lines = lines
        self.log_scroll = 0
        self.log_viewing = True

    # ─────────────────────────────────────────────────────────────────────────
    # Inline text field editor (bottom-bar input)
    # ─────────────────────────────────────────────────────────────────────────

    def _edit_field(self, fields_dict, key):
        current = fields_dict[key]
        self._input_value  = current
        self._input_cursor = len(current)
        prompt = f" Edit [{key}]: "

        while True:
            self.h, self.w = self.scr.getmaxyx()
            y = self.h - 1
            bar_w = self.w - len(prompt) - 2
            display = self._input_value[-bar_w:] if len(self._input_value) > bar_w else self._input_value
            safe_addstr(self.scr, y, 0, " " * self.w, curses.color_pair(C_HEADER))
            safe_addstr(self.scr, y, 0, prompt, curses.color_pair(C_HEADER) | curses.A_BOLD)
            safe_addstr(self.scr, y, len(prompt), display, curses.color_pair(C_HEADER))
            cx = len(prompt) + min(self._input_cursor, bar_w)
            try:
                self.scr.move(y, cx)
                curses.curs_set(1)
            except curses.error:
                pass
            self.scr.refresh()

            k = self.scr.getch()
            if k in (curses.KEY_ENTER, 10, 13):
                fields_dict[key] = self._input_value
                curses.curs_set(0)
                self.set_status(f"Set {key} = {self._input_value}", C_SUCCESS)
                break
            elif k == 27:
                curses.curs_set(0)
                break
            elif k in (curses.KEY_BACKSPACE, 127, 8):
                if self._input_cursor > 0:
                    v = self._input_value
                    self._input_value  = v[:self._input_cursor-1] + v[self._input_cursor:]
                    self._input_cursor -= 1
            elif k == curses.KEY_DC:
                v = self._input_value
                self._input_value = v[:self._input_cursor] + v[self._input_cursor+1:]
            elif k == curses.KEY_LEFT:
                self._input_cursor = max(0, self._input_cursor - 1)
            elif k == curses.KEY_RIGHT:
                self._input_cursor = min(len(self._input_value), self._input_cursor + 1)
            elif k == curses.KEY_HOME:
                self._input_cursor = 0
            elif k == curses.KEY_END:
                self._input_cursor = len(self._input_value)
            elif 32 <= k <= 126:
                v = self._input_value
                self._input_value  = v[:self._input_cursor] + chr(k) + v[self._input_cursor:]
                self._input_cursor += 1

    # ─────────────────────────────────────────────────────────────────────────
    # Background process management
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _stream_output(proc, q):
        for line in proc.stdout:
            q.put(line)
        q.put(None)

    def _drain_queues(self):
        while not self.train_log_q.empty():
            item = self.train_log_q.get_nowait()
            if item is None:
                self.train_running = False
                if self.train_proc and self.train_proc.returncode == 0:
                    self.set_status("Training finished!", C_SUCCESS)
                elif self.train_proc:
                    self.set_status(f"Training exited (code {self.train_proc.returncode})", C_WARN)
            else:
                self.train_log_lines.append(item)

        while not self.inf_log_q.empty():
            item = self.inf_log_q.get_nowait()
            if item is None:
                self.inf_running = False
                if self.inf_proc and self.inf_proc.returncode == 0:
                    self.set_status("Inference finished!", C_SUCCESS)
                elif self.inf_proc:
                    self.set_status(f"Inference exited (code {self.inf_proc.returncode})", C_WARN)
            else:
                self.inf_log_lines.append(item)

    def _stop_process(self, which):
        if which == "train" and self.train_proc:
            self.train_proc.terminate()
            self.train_running = False
            self.set_status("Training stopped", C_WARN)
        elif which == "inf" and self.inf_proc:
            self.inf_proc.terminate()
            self.inf_running = False
            self.set_status("Inference stopped", C_WARN)

    # ─────────────────────────────────────────────────────────────────────────
    # venv python resolver
    # ─────────────────────────────────────────────────────────────────────────

    def _venv_python(self):
        for c in [SCRIPT_DIR / ".venv" / "bin" / "python",
                  SCRIPT_DIR / ".venv" / "Scripts" / "python.exe"]:
            if c.exists():
                return str(c)
        return sys.executable


# ─────────────────────────────────────────────────────────────────────────────
# Entry
# ─────────────────────────────────────────────────────────────────────────────

def main():
    def _run(stdscr):
        tui = TUI(stdscr)
        tui.run()
    curses.wrapper(_run)


if __name__ == "__main__":
    main()

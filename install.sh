#!/usr/bin/env bash
# install.sh — Apollo installer (Linux / macOS)
# Creates a uv-managed .venv with all dependencies.
# Safe to re-run — skips steps that are already done.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

if [ -t 1 ]; then
    GRN="\033[0;32m" CYN="\033[0;36m" YLW="\033[0;33m" RED="\033[0;31m" RST="\033[0m"
else
    GRN="" CYN="" YLW="" RED="" RST=""
fi

info()  { echo -e "${CYN}[install]${RST} $*"; }
ok()    { echo -e "${GRN}[install]${RST} $*"; }
die()   { echo -e "${RED}[install]${RST} $*" >&2; exit 1; }

echo
echo -e "${GRN}  ╔══════════════════════════════════╗"
echo -e "  ║       Apollo — Installer         ║"
echo -e "  ╚══════════════════════════════════╝${RST}"
echo

# ── 1. Locate or install uv ──────────────────────────────────────────────────
UV_BIN=""
if command -v uv &>/dev/null; then
    UV_BIN="$(command -v uv)"
    ok "Found uv at ${UV_BIN} ($(uv --version))"
elif [ -x "$HOME/.local/bin/uv" ]; then
    UV_BIN="$HOME/.local/bin/uv"
    ok "Found uv at ${UV_BIN}"
elif [ -x "$HOME/.cargo/bin/uv" ]; then
    UV_BIN="$HOME/.cargo/bin/uv"
    ok "Found uv at ${UV_BIN}"
else
    info "uv not found — installing via official installer..."
    if command -v curl &>/dev/null; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
    elif command -v wget &>/dev/null; then
        wget -qO- https://astral.sh/uv/install.sh | sh
    else
        die "Neither curl nor wget found. Install uv manually: https://github.com/astral-sh/uv"
    fi
    if   [ -x "$HOME/.local/bin/uv" ]; then UV_BIN="$HOME/.local/bin/uv"
    elif [ -x "$HOME/.cargo/bin/uv" ]; then UV_BIN="$HOME/.cargo/bin/uv"
    else die "uv installed but binary not found. Add ~/.local/bin to PATH and re-run."
    fi
    ok "uv installed at ${UV_BIN}"
fi

# ── 2. Create virtual environment ───────────────────────────────────────────
if [ -d "$VENV_DIR" ]; then
    ok "Virtual environment already exists — skipping"
else
    info "Creating virtual environment with Python 3.11..."
    "$UV_BIN" venv "$VENV_DIR" --python 3.11 \
        || die "Failed to create venv. Try: $UV_BIN python install 3.11"
    ok "Virtual environment created at .venv/"
fi

# ── 3. Install dependencies ──────────────────────────────────────────────────
info "Installing dependencies from requirements.txt..."
"$UV_BIN" pip install \
    --python "$VENV_DIR/bin/python" \
    "setuptools<71" \
    pyyaml \
    -r "$SCRIPT_DIR/requirements.txt"
ok "All dependencies installed"

echo
echo -e "${GRN}  ✓ Installation complete!${RST}"
echo -e "  Run ${CYN}./start.sh${RST} to launch the Apollo Web UI"
echo

#!/usr/bin/env bash
# @claude last-modified: 2026-05-05T06:40:00Z
# @claude last-commit: chore: remove web UI, start scripts launch TUI

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

die() { echo -e "\033[0;31m[start]\033[0m $*" >&2; exit 1; }

if [ ! -x "$VENV_DIR/bin/python" ]; then
    die "Virtual environment not found. Run ./install.sh first."
fi

exec "$VENV_DIR/bin/python" "$SCRIPT_DIR/tui.py"

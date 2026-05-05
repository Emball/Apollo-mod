#!/usr/bin/env bash
# start.sh — Launch the Apollo Web UI (Linux / macOS)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

if [ -t 1 ]; then
    GRN="\033[0;32m" CYN="\033[0;36m" RST="\033[0m"
else
    GRN="" CYN="" RST=""
fi

die() { echo -e "\033[0;31m[start]\033[0m $*" >&2; exit 1; }

if [ ! -x "$VENV_DIR/bin/python" ]; then
    die "Virtual environment not found. Run ./install.sh first."
fi

echo
echo -e "${GRN}  Apollo Web UI${RST} — starting"
echo -e "  Open ${CYN}http://127.0.0.1:5000${RST} in your browser"
echo

exec "$VENV_DIR/bin/python" "$SCRIPT_DIR/webui.py"

#!/usr/bin/env bash
# apollo.sh — install (if needed) then run an Apollo command or drop into a shell
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

if [ -t 1 ]; then
    GRN="\033[0;32m" CYN="\033[0;36m" RED="\033[0;31m" RST="\033[0m"
else
    GRN="" CYN="" RED="" RST=""
fi

info() { echo -e "${CYN}[apollo]${RST} $*"; }
ok()   { echo -e "${GRN}[apollo]${RST} $*"; }
die()  { echo -e "${RED}[apollo]${RST} $*" >&2; exit 1; }

# 1. Locate or install uv
UV_BIN=""
if   command -v uv &>/dev/null;              then UV_BIN="$(command -v uv)"
elif [ -x "$HOME/.local/bin/uv" ];           then UV_BIN="$HOME/.local/bin/uv"
elif [ -x "$HOME/.cargo/bin/uv" ];           then UV_BIN="$HOME/.cargo/bin/uv"
else
    info "uv not found — installing..."
    if   command -v curl &>/dev/null; then curl -LsSf https://astral.sh/uv/install.sh | sh
    elif command -v wget &>/dev/null; then wget -qO- https://astral.sh/uv/install.sh | sh
    else die "Neither curl nor wget found. Install uv manually: https://github.com/astral-sh/uv"
    fi
    if   [ -x "$HOME/.local/bin/uv" ]; then UV_BIN="$HOME/.local/bin/uv"
    elif [ -x "$HOME/.cargo/bin/uv" ]; then UV_BIN="$HOME/.cargo/bin/uv"
    else die "uv installed but binary not found. Add ~/.local/bin to PATH and re-run."
    fi
fi
ok "uv: $UV_BIN"

# 2. Create venv if missing
if [ ! -d "$VENV_DIR" ]; then
    info "Creating virtual environment (Python 3.11)..."
    "$UV_BIN" venv "$VENV_DIR" --python 3.11 \
        || die "venv creation failed. Try: $UV_BIN python install 3.11"
    ok "venv created"
else
    ok "venv exists"
fi

# 3. Install / sync dependencies — platform-aware
# ---------------------------------------------------------------------------
# Linux   → CUDA PyTorch from pytorch.org index + bitsandbytes
# macOS   → standard PyTorch from PyPI (MPS on Apple Silicon, CPU on Intel),
#            no bitsandbytes (unsupported on macOS)
# Windows → handled by apollo.bat (CUDA PyTorch + bitsandbytes)
# ---------------------------------------------------------------------------
OS="$(uname -s)"
case "$OS" in
    Linux*)
        info "Detected Linux — installing CUDA PyTorch 2.1.2..."
        "$UV_BIN" pip install \
            --python "$VENV_DIR/bin/python" \
            --index-url https://download.pytorch.org/whl/cu121 \
            torch==2.1.2+cu121 torchaudio==2.1.2+cu121
        ok "PyTorch installed"
        ;;
    Darwin*)
        info "Detected macOS — installing PyTorch 2.1.2 (MPS/CPU)..."
        "$UV_BIN" pip install \
            --python "$VENV_DIR/bin/python" \
            torch==2.1.2 torchaudio==2.1.2
        ok "PyTorch installed"
        ;;
    *)
        die "Unsupported OS: $OS (only Linux, macOS, and Windows are supported)"
        ;;
esac

info "Syncing remaining dependencies..."
"$UV_BIN" pip install \
    --python "$VENV_DIR/bin/python" \
    "setuptools<71" pyyaml \
    -r "$SCRIPT_DIR/requirements.txt"
ok "dependencies up to date"

# 4. If arguments given, treat first as the script name and run it
if [ $# -gt 0 ]; then
    CMD="$1"; shift
    case "$CMD" in
        train)     SCRIPT="train.py" ;;
        inference) SCRIPT="inference.py" ;;
        test)      SCRIPT="test.py" ;;
        python)    exec "$VENV_DIR/bin/python" "$@" ;;
        *)         exec "$VENV_DIR/bin/python" "$CMD" "$@" ;;
    esac
    exec "$VENV_DIR/bin/python" "$SCRIPT_DIR/$SCRIPT" "$@"
fi

# 5. No arguments — drop into an activated shell with command aliases
RC=$(mktemp)
cat > "$RC" <<- EOF
alias train='python "$SCRIPT_DIR/train.py"'
alias inference='python "$SCRIPT_DIR/inference.py"'
alias test='python "$SCRIPT_DIR/test.py"'
EOF

echo
echo -e "${GRN}  Apollo ready.${RST} Type 'exit' to leave."
echo -e "  Commands: ${CYN}train, inference, test${RST} (or python directly)"
echo

export VIRTUAL_ENV="$VENV_DIR"
export PATH="$VENV_DIR/bin:$PATH"
unset PYTHONHOME

exec "${SHELL:-bash}" --rcfile "$RC" --noprofile -i

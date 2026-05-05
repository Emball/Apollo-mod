#!/usr/bin/env bash
# Session start parser. Outputs full context in one shot — do not cat .claude/ files individually.
# Wipes scratchpad after flushing contents.
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
CLAUDE_DIR="$REPO_ROOT/.claude"

# Add directories here to exclude from @claude header audit
SKIP_DIRS=("node_modules" "dist" "build" ".git" "vendor" "venv" ".venv" "__pycache__" "coverage" ".next" "target")

separator() { echo ""; echo "════════════════════════════════════════"; echo ""; }

# ── PROJECT ───────────────────────────────────────────────────────────────────
echo "▌ PROJECT"
separator
cat "$CLAUDE_DIR/project.md"

# ── MEMORY ────────────────────────────────────────────────────────────────────
separator
echo "▌ MEMORY"
separator
cat "$CLAUDE_DIR/memory.md"

# ── SCRATCHPAD ────────────────────────────────────────────────────────────────
separator
echo "▌ SCRATCHPAD (flushed and wiped)"
separator
SCRATCH="$CLAUDE_DIR/scratchpad.md"
SCRATCH_CONTENT=$(cat "$SCRATCH")

# Check if scratchpad has anything beyond the blank template
if echo "$SCRATCH_CONTENT" | grep -qvE '^\s*$|^# Scratchpad|^<!-- |^Task:|^Run:'; then
  echo "$SCRATCH_CONTENT"
  echo ""
  echo "⚠ SCRATCHPAD HAD CONTENT — prior session may not have promoted everything to memory.md."
  echo "  Review the above before proceeding. Wiping now."
else
  echo "(empty)"
fi

# Wipe
cat > "$SCRATCH" << 'TMPL'
# Scratchpad
<!-- Flushed and wiped by session.sh on every session start. -->
<!-- Reason here, not in code comments. Promote to memory.md before session ends. -->

Task:
Run:

TMPL

# ── RUN LOG ───────────────────────────────────────────────────────────────────
separator
echo "▌ RUN LOG (last entry)"
separator
RUN_LOG="$CLAUDE_DIR/run-log.md"

# Extract the last [[[ block — grep line by line, take last match
LAST_ENTRY=$(grep '\[\[\[' "$RUN_LOG" | tail -1)
echo "$LAST_ENTRY"
echo ""

LAST_CLOSED_TS=""
if echo "$LAST_ENTRY" | grep -q '\]\]\]'; then
  echo "✓ Last run closed cleanly."
  LAST_CLOSED_TS=$(echo "$LAST_ENTRY" | grep -o '[0-9]\{4\}-[0-9]\{2\}-[0-9]\{2\}T[0-9]\{2\}:[0-9]\{2\}:[0-9]\{2\}Z' | head -1)
else
  echo "✗ UNCLOSED RUN DETECTED — prior run failed or was interrupted."
  echo "  Declared intent above. Cross-check file headers before opening a new entry."
fi

# ── FILE HEADER AUDIT ─────────────────────────────────────────────────────────
separator
echo "▌ FILE HEADER AUDIT"
separator

# Build find prune args as a proper array
FIND_ARGS=()
for d in "${SKIP_DIRS[@]}"; do
  FIND_ARGS+=(-path "*/$d" -prune -o)
done
FIND_ARGS+=(-type f -print)

CONFLICTS=0
HEADERS_FOUND=0

while IFS= read -r file; do
  HEAD=$(head -5 "$file" 2>/dev/null)
  if echo "$HEAD" | grep -q '@claude'; then
    HEADERS_FOUND=$((HEADERS_FOUND + 1))
    FILE_TS=$(echo "$HEAD" | grep '@claude last-modified' | grep -o '[0-9]\{4\}-[0-9]\{2\}-[0-9]\{2\}T[0-9]\{2\}:[0-9]\{2\}:[0-9]\{2\}Z' || echo "")
    FILE_COMMIT=$(echo "$HEAD" | grep '@claude last-commit' | sed 's/.*@claude last-commit: //' | tr -d '\r' || echo "")
    DISPLAY="${file#$REPO_ROOT/}"

    GIT_COMMIT=$(git -C "$REPO_ROOT" log -1 --pretty=format:"%s" -- "$file" 2>/dev/null || echo "")

    CONFLICT=0
    NEWER_THAN_LOG=0

    if [ -n "$FILE_COMMIT" ] && [ -n "$GIT_COMMIT" ] && [ "$FILE_COMMIT" != "$GIT_COMMIT" ]; then
      CONFLICT=1
    fi

    if [ -n "$LAST_CLOSED_TS" ] && [ -n "$FILE_TS" ] && [[ "$FILE_TS" > "$LAST_CLOSED_TS" ]]; then
      NEWER_THAN_LOG=1
    fi

    if [ $CONFLICT -eq 1 ] || [ $NEWER_THAN_LOG -eq 1 ]; then
      CONFLICTS=$((CONFLICTS + 1))
      echo "⚠ $DISPLAY"
      [ $CONFLICT -eq 1 ] && echo "    header commit : $FILE_COMMIT" && echo "    git log commit : $GIT_COMMIT"
      [ $NEWER_THAN_LOG -eq 1 ] && echo "    file modified ($FILE_TS) is newer than last closed run ($LAST_CLOSED_TS)" && \
        echo "    note: human edits between sessions can also cause this — investigate before assuming error"
    fi
  fi
done < <(find "$REPO_ROOT" "${FIND_ARGS[@]}" 2>/dev/null | sort)

if [ $HEADERS_FOUND -eq 0 ]; then
  echo "(no @claude headers found — new project or no files touched yet)"
elif [ $CONFLICTS -eq 0 ]; then
  echo "✓ All $HEADERS_FOUND Claude-touched files consistent with run log."
else
  echo ""
  echo "✗ $CONFLICTS conflict(s) across $HEADERS_FOUND Claude-touched files."
fi

# ── SUMMARY ───────────────────────────────────────────────────────────────────
separator
echo "▌ READY"
separator
echo "Context loaded. Resolve any ✗ or ⚠ above, then open a new run log entry before starting work."

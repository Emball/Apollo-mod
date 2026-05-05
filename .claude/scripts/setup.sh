#!/usr/bin/env bash
# One-time initializer. Run on every session start — safe to re-run, exits immediately if .claude/ exists.
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
CLAUDE_DIR="$REPO_ROOT/.claude"

if [ -d "$CLAUDE_DIR" ]; then
  echo "STATUS:EXISTS"
  exit 0
fi

mkdir -p "$CLAUDE_DIR"

REPO=$(basename "$REPO_ROOT")
NOW=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

cat > "$CLAUDE_DIR/project.md" << 'TMPL'
# Project

## Goals
- Goal one
- Goal two

## Scope
### In
- What is included

### Out
- What is excluded

## Checkpoint Plan
<!-- [ ] pending | [x] done | [~] partial/failed -->
[ ] 1. First milestone
[ ] 2. Second milestone
[ ] 3. Third milestone

## Current State
Project initialized. No work started.
TMPL

cat > "$CLAUDE_DIR/memory.md" << 'TMPL'
# Memory
<!-- Append only. Never edit existing entries. -->
<!-- Categories: decision | bug | pattern | setup | learning -->

### [setup] 1970-01-01 — Project initialized
Repository and .claude/ structure created.
TMPL

cat > "$CLAUDE_DIR/run-log.md" << TMPL
# Run Log
<!-- [[[TIMESTAMP: intent.]]] — ]]] written at START of next run if prior run succeeded -->
<!-- Unclosed entry = something failed. Cross-check file headers before continuing. -->

[[[${NOW}: Initialized .claude/ structure.]]]
TMPL

cat > "$CLAUDE_DIR/scratchpad.md" << 'TMPL'
# Scratchpad
<!-- Flushed and wiped by session.sh on every session start. -->
<!-- Reason here, not in code comments. -->

Task:
Run:

TMPL

git -C "$REPO_ROOT" add .claude/
git -C "$REPO_ROOT" commit -m "init: create .claude/ memory structure"

echo "STATUS:CREATED"

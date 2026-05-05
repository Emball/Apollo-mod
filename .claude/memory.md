# Memory
<!-- Append only. Never edit existing entries. -->
<!-- Categories: decision | bug | pattern | setup | learning -->

### [setup] 1970-01-01 — Project initialized
Repository and .claude/ structure created.

### [decision] 2026-05-05 — WAV-only training data
Narrowed _SUPPORTED_EXTS to WAV only for training pipeline. Other formats introduce codec delays/encoder overhead that can't be accounted for during chunking. Inference still accepts any format.

### [pattern] 2026-05-05 — Augmentation via dataclasses
All augmentation config is now in typed dataclasses (GainAugCfg, PitchShiftAugCfg, NoiseAugCfg, etc.) read from the yaml `augmentation` block. This replaces bare module-level constants.

### [decision] 2026-05-05 — apply_optimizations() bootstrap
TF32, cuDNN benchmark, expandable CUDA segments, and Triton cache are all controlled from cfg.optimizations and applied in a single apply_optimizations() call at startup, before any model code runs.

### [pattern] 2026-05-05 — Gradient checkpointing via monkey-patch
BSNet.forward and FrequencyDiscriminator.forward are wrapped with torch.utils.checkpoint at runtime when gradient_checkpointing=True. Trades ~30% compute for significant VRAM reduction.

### [setup] 2026-05-05 — TUI entry point
tui.py is a 1340-line curses TUI with Train, Inference, Config editor, Pretrain model browser, and Log viewer screens. Launched via start.sh / start.bat. Note: start.sh references webui.py which does not exist — likely a stale reference or planned future file.

### [decision] 2026-05-05 — Web UI removed
webui.py was a failed experiment. flask and flask-socketio removed from requirements.txt. start.sh/start.bat now launch tui.py directly.

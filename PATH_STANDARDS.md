# Path Standards for llm-game-engine

## Overview

This document defines standard path conventions for the Secret Hitler LLM Evaluation Framework to ensure consistency and prevent path-related bugs.

## Directory Structure

```
secret-hitler/                    # Parent repository (game client)
└── llm-game-engine/              # Nested repository (eval framework) ← ALL CODE RUNS HERE
    ├── core/                     # Game logic
    ├── agents/                   # LLM integration
    ├── game_logging/             # Logging system
    ├── logs/                     # Game logs (RUNTIME DATA)
    ├── data/                     # Database files (RUNTIME DATA)
    ├── results/                  # Batch results (RUNTIME DATA)
    ├── experiments/              # Experiment configs
    ├── scripts/                  # Utility scripts
    └── web_bridge/               # WebSocket server
```

## Standard Path Pattern

**CRITICAL RULE**: All file paths in llm-game-engine must be relative to the llm-game-engine directory, NOT the parent secret-hitler directory.

### Correct Pattern

```python
from pathlib import Path

# For files in the same module
base_dir = Path(__file__).parent

# For files in llm-game-engine root
base_dir = Path(__file__).parent  # If in llm-game-engine root
                                    # OR
base_dir = Path(__file__).parent.parent  # If in subdirectory like core/
                                          # OR
base_dir = Path(__file__).resolve().parents[2]  # If deep in subdirs

# Standard data directories (relative to llm-game-engine root)
logs_dir = base_dir / "logs"
data_dir = base_dir / "data"
results_dir = base_dir / "results"
```

### Incorrect Pattern

```python
# ❌ WRONG - This goes to parent repo, not llm-game-engine
logs_dir = Path(__file__).parent.parent / "logs"  # When already at llm-game-engine level
```

## File-Specific Standards

### 1. core/game_manager.py
```python
# Logging directory
logs_dir = Path(__file__).parent.parent / "logs"  # __file__ is in core/
self.logger = GameLogger(self.game_id, base_log_dir=str(logs_dir))
```

### 2. run_game.py
```python
# Batch metadata
logs_dir = Path(__file__).parent / "logs"  # __file__ is in llm-game-engine/
metadata_file = logs_dir / ".current_batch"
```

### 3. check_batch_progress.py
```python
# Log directory for progress tracking
logs_dir = Path(__file__).parent / "logs"  # __file__ is in llm-game-engine/
```

### 4. game_logging/game_logger.py
```python
# Default log directory
def __init__(self, game_id: str, base_log_dir: str = "logs", ...):
    # Note: Uses relative path by default, caller passes absolute
    self.base_log_dir = Path(base_log_dir)
```

### 5. evaluation/database_schema.py
```python
# Default database path
def get_db_path(custom_path: Optional[str] = None) -> Path:
    if custom_path:
        return Path(custom_path)
    # Relative to llm-game-engine root
    return Path(__file__).parent.parent / "data" / "games.db"
```

## Import Path vs File Path

**Import paths** (for Python module resolution) may use `.parent.parent`:
```python
# ✅ OK for imports - adding llm-game-engine to PYTHONPATH
sys.path.append(str(Path(__file__).parent.parent))
```

**File paths** (for reading/writing files) should target llm-game-engine:
```python
# ✅ Correct - files go to llm-game-engine/logs
logs_dir = Path(__file__).parent / "logs"
```

## Verification Checklist

Before committing code that uses file paths:

- [ ] Verify `__file__` location relative to llm-game-engine root
- [ ] Count `.parent` calls: should match directory depth
- [ ] Test that logs/data go to `llm-game-engine/logs/` and `llm-game-engine/data/`
- [ ] Check that paths work when script is run from any working directory
- [ ] Ensure absolute paths are used when passing to other modules

## Common Mistakes

### Mistake 1: Too many .parent calls
```python
# ❌ WRONG - if __file__ is already in llm-game-engine/
logs_dir = Path(__file__).parent.parent / "logs"  # Goes to secret-hitler/logs

# ✅ CORRECT
logs_dir = Path(__file__).parent / "logs"  # Goes to llm-game-engine/logs
```

### Mistake 2: Relative paths without absolute conversion
```python
# ❌ WRONG - relative paths break when cwd changes
GameLogger(game_id, base_log_dir="logs")

# ✅ CORRECT - pass absolute path
logs_dir = Path(__file__).parent / "logs"
GameLogger(game_id, base_log_dir=str(logs_dir))
```

### Mistake 3: Assuming working directory
```python
# ❌ WRONG - assumes cwd is llm-game-engine
with open("logs/game.log") as f:

# ✅ CORRECT - explicit absolute path
logs_dir = Path(__file__).parent / "logs"
with open(logs_dir / "game.log") as f:
```

## Testing Path Configuration

To verify paths are correct:

```bash
# Check where logs are created
ls -la /Users/samueltchakwera/Playground/Projects/secret-hitler/llm-game-engine/logs/

# Should see recent game directories and .current_batch

# Check parent repo logs (should be old or empty)
ls -la /Users/samueltchakwera/Playground/Projects/secret-hitler/logs/
```

## Recent Fixes Applied

**2025-11-01**: Fixed log path bug where logs were created in parent repo instead of llm-game-engine
- **Root Cause**: GameLogger was using relative path "logs" which resolved relative to current working directory (parent repo) instead of llm-game-engine directory
- **Symptom**: Progress tracker couldn't find game logs, policies showed as ⚪⚪⚪⚪⚪ instead of colored progression
- **Fix Applied**:
  - Updated `core/game_manager.py:67-73` - Changed to absolute path using `Path(__file__).parent.parent / "logs"`
  - Updated `run_game.py:105, 109` - Fixed batch metadata log_dir path
  - Updated `check_batch_progress.py:17, 123, 265` - Fixed all 3 references to use correct base path
- **Verification**: Tested with 3-game batch, logs now correctly created in `llm-game-engine/logs/`, tracker shows game state properly
- **Audit Result**: All other files using `.parent.parent` are correctly using it for import paths only (sys.path modifications)

## Questions?

When in doubt:
1. Check where `__file__` is located
2. Count directory levels to llm-game-engine root
3. Use that many `.parent` calls
4. Always pass absolute paths to other modules

---

**Last Updated**: 2025-11-01
**Author**: Samuel T. Chakwera (stchakdev)

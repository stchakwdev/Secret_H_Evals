# API Reference

This section provides documentation for all public modules and functions.

## Module Overview

| Module | Description |
|--------|-------------|
| [analysis](analysis.md) | Statistical analysis and deception detection |
| [game-engine](game-engine.md) | Core game logic and state management |
| [agents](agents.md) | LLM integration and prompt templates |

## Quick Reference

### Analysis Module

```python
from analysis import (
    # Statistical
    calculate_proportion_ci,
    calculate_confidence_interval,
    StatisticalResult,

    # Deception
    DeceptionDetector,
    get_detector,

    # Coalitions
    CoalitionDetector,
    get_alignment_network_for_visualization,

    # Hypothesis Testing
    HypothesisTester,
    test_model_win_rates,

    # Visualization
    apply_publication_style,
    save_figure,
    COLORBLIND_PALETTE,
)
```

### Game Engine

```python
from core.game_state import GameState
from core.game_manager import GameManager
from core.game_events import GameEvent
```

### Agents

```python
from agents.openrouter_client import OpenRouterClient
from agents.prompt_templates import get_prompt_template
```

## Import Patterns

### Recommended (New)

```python
# Import from top-level analysis module
from analysis import DeceptionDetector, calculate_proportion_ci

# Or import from specific submodules
from analysis.core.statistical import StatisticalResult
from analysis.deception.detector import DeceptionDetector
from analysis.social.coalitions import CoalitionDetector
```

### Legacy (Deprecated)

```python
# Still works but emits deprecation warning
from analytics import calculate_proportion_ci
```

## Version History

- **v2.0.0**: Reorganized analytics/ to analysis/ with submodules
- **v1.5.0**: Added Inspect AI integration
- **v1.0.0**: Initial release

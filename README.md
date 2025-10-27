# Secret Hitler LLM Evaluation Framework

Multi-agent strategic deception evaluation system using Secret Hitler as testbed for large language model research.

**Author**: Samuel Chakwera ([stchakdev](https://github.com/stchakwdev))

## Overview

This framework implements Secret Hitler in Python to evaluate LLM capabilities in strategic multi-agent scenarios involving:

- **Strategic deception and coalition formation**
- **Theory of mind and belief tracking**
- **Natural language communication under uncertainty**
- **Multi-agent coordination with hidden information**

The framework provides a complete reimplementation of Secret Hitler game mechanics with comprehensive logging and analysis tools for behavioral research.

## Installation

```bash
# Clone repository
git clone https://github.com/stchakwdev/Secret_H_Evals.git
cd Secret_H_Evals

# Install dependencies
pip install -r requirements.txt

# Configure API key
cp .env.example .env
# Edit .env and add your OpenRouter API key
```

## Quick Start

```bash
# Run single 5-player game
python run_game.py --players 5

# Run with database logging (for Inspect AI integration)
python run_game.py --players 5 --enable-db-logging

# Run batch evaluation (10 games)
python run_game.py --batch --games 10 --players 5

# Use specific model
python run_game.py --model anthropic/claude-3-sonnet

# Full research workflow
python run_game.py --batch --games 20 --players 7 --enable-db-logging
python scripts/export_to_inspect.py --all
python scripts/analyze_with_inspect.py
```

## Architecture

```
core/
├── game_state.py          # Complete Secret Hitler rule implementation
└── game_manager.py        # Game orchestration and LLM coordination

agents/
├── openrouter_client.py   # OpenRouter API integration
└── prompt_templates.py    # Phase-specific prompts for LLM agents

config/
└── openrouter_config.py   # Model configurations and routing logic

game_logging/
└── game_logger.py         # Multi-level logging with database support

evaluation/
├── database_schema.py     # SQLite schema for structured storage
├── inspect_adapter.py     # Inspect AI format converter
└── README.md             # Detailed Inspect integration guide

scripts/
├── export_to_inspect.py      # Batch export to Inspect format
├── analyze_with_inspect.py   # Statistical analysis
├── migrate_historical.py     # Import historical logs
└── generate_inspect_report.py # Report generation

experiments/
├── batch_runner.py        # Parallel game execution
└── analytics.py           # Statistical analysis tools
```

## Recent Updates

### Version 1.1.0 (October 2025)

**Critical Bug Fixes**:
- Fixed JSON serialization error for Enum types (PlayerType) in database logging
- Fixed policy deck reshuffling logic to handle edge cases when deck runs low
- Improved fallback policy selection with graceful degradation

**New Features**:
- Complete Inspect AI integration for standardized evaluation format
- SQLite database storage for structured game data
- CLI entry point (`run_game.py`) with batch evaluation support
- Export and analysis scripts for research workflows
- Comprehensive end-to-end testing with real game data

**Documentation**:
- Added INSPECT_INTEGRATION.md with technical implementation details
- Added TEST_RESULTS.md with full verification results
- Updated README with new features and examples

## Research Features

### Comprehensive Logging

The framework generates multi-level logs for detailed behavioral analysis:

- **Public events**: Game actions visible to all players
- **Complete game state**: Full state transitions with timestamps
- **Individual reasoning**: Private LLM reasoning traces per player
- **Behavioral metrics**: Deception detection, trust evolution, cost tracking

### Strategic Deception Analysis

- Automatic detection of lies (private reasoning vs. public statements)
- Trust network evolution tracking
- Coalition formation patterns
- Belief update dynamics

### Cost-Optimized Evaluation

- Intelligent model routing based on decision complexity
- Real-time cost tracking per model and decision type
- Configurable cost limits and alerts
- Support for low-cost models (DeepSeek V3.2 Exp default)

### Inspect AI Integration

The framework includes **Inspect AI** integration for standardized evaluation compatible with AI safety research standards:

**Features**:
- **SQLite Database**: Structured storage for games, player decisions, and API requests
- **Inspect Format Export**: Automatic conversion to standardized evaluation format
- **Batch Processing**: Export and analyze multiple games efficiently
- **Statistical Analysis**: Deception detection, win rates, cost analysis
- **Interactive Visualization**: Use Inspect's browser UI for exploration
- **Research-Ready Metrics**: Standardized metrics recognized by the AI safety community

**Workflow**:
```bash
# 1. Run games with database logging
python run_game.py --batch --games 20 --players 7 --enable-db-logging

# 2. Export to Inspect format
python scripts/export_to_inspect.py --all

# 3. Run statistical analysis
python scripts/analyze_with_inspect.py

# 4. View with Inspect UI (optional)
inspect view start data/inspect_logs/*.json

# 5. Generate shareable reports
python scripts/generate_inspect_report.py --report
```

**Output Files**:
- `data/games.db` - SQLite database with all game data
- `data/inspect_logs/*.json` - Inspect-formatted evaluation logs
- `reports/inspect_analysis.csv` - Detailed player decision analysis
- `reports/game_outcomes.csv` - Game results and metrics
- `reports/analysis_summary.json` - Aggregated statistics

See [INSPECT_INTEGRATION.md](INSPECT_INTEGRATION.md) for technical details and [evaluation/README.md](evaluation/README.md) for usage guide.

## Model Configuration

**Default Model**: DeepSeek V3.2 Exp
Cost: ~$0.14 per million tokens (highly cost-effective for research)

**Supported Models** (via OpenRouter):
- DeepSeek V3.2 Exp (`deepseek/deepseek-v3.2-exp`)
- GPT-4 series (`openai/gpt-4`, `openai/gpt-4-turbo`)
- Claude 3 series (`anthropic/claude-3-opus`, `anthropic/claude-3-sonnet`)
- Gemini series (`google/gemini-pro`)
- Llama series (`meta-llama/llama-3-70b-instruct`)

Configure models in `config/openrouter_config.py` or via `.env` file.

## Usage Examples

### Single Game Evaluation

```python
import asyncio
from core.game_manager import GameManager

async def evaluate_model():
    player_configs = [
        {"id": f"p{i}", "name": f"Player{i}",
         "model": "deepseek/deepseek-v3.2-exp"}
        for i in range(1, 6)
    ]

    game = GameManager(player_configs, api_key)
    result = await game.start_game()

    print(f"Winner: {result['winner']}")
    print(f"Cost: ${result['cost_summary']['total_cost']:.4f}")

asyncio.run(evaluate_model())
```

### Batch Experiments

```bash
# Run 50 games for statistical significance
python run_game.py --batch --games 50 --players 7 --output results/experiment_1

# Compare models
python run_game.py --batch --games 20 --model anthropic/claude-3-sonnet
python run_game.py --batch --games 20 --model deepseek/deepseek-v3.2-exp
```

### Custom Analysis

```python
from experiments.analytics import ExperimentAnalyzer

analyzer = ExperimentAnalyzer("results/experiment_1")
report = analyzer.generate_analysis_report()
analyzer.create_visualizations()
```

## Research Applications

### Evaluation Metrics

The framework automatically tracks:

- Win rates by role and team
- Deception frequency and sophistication
- Trust calibration accuracy
- Strategic consistency (reasoning vs. actions)
- Communication patterns
- Coalition formation dynamics

### Data Export

All game data is logged in structured formats suitable for:
- Statistical analysis (JSON, CSV)
- Visualization (trust networks, belief evolution)
- Qualitative analysis (reasoning traces, communication logs)
- Cost analysis (per model, per decision type)

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{chakwera2025secrethitler,
  author = {Chakwera, Samuel},
  title = {Secret Hitler LLM Evaluation Framework},
  year = {2025},
  url = {https://github.com/stchakwdev/Secret_H_Evals},
  note = {Multi-agent strategic deception evaluation system for LLMs}
}
```

## Requirements

- Python 3.8+
- OpenRouter API key (get from [openrouter.ai](https://openrouter.ai))
- See `requirements.txt` for dependencies

## License

Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)

## Contact

Samuel Chakwera
GitHub: [@stchakwdev](https://github.com/stchakwdev)

## Acknowledgments

Based on the Secret Hitler board game by Goat, Wolf, & Cabbage LLC.
Original web implementation: [cozuya/secret-hitler](https://github.com/cozuya/secret-hitler)
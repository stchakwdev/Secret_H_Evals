# Secret Hitler LLM Evaluation Framework

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![GitHub stars](https://img.shields.io/github/stars/stchakwdev/Secret_H_Evals?style=social)](https://github.com/stchakwdev/Secret_H_Evals)

Multi-agent strategic deception evaluation for large language models using Secret Hitler as a testbed. This framework enables researchers to study AI reasoning, trust dynamics, and deceptive behavior patterns in a controlled game environment.

**Author**: Samuel T. Chakwera ([stchakwdev](https://github.com/stchakwdev))

---

## Table of Contents

- [Why This Project?](#why-this-project)
- [Quick Start](#quick-start)
- [Batch Evaluation Monitor](#batch-evaluation-monitor)
- [Evaluation Results](#evaluation-results-300-games)
- [Visual Analytics](#visual-analytics)
- [Features](#features)
- [Architecture](#architecture)
- [Documentation](#documentation)
- [Citation](#citation)
- [Recent Updates](#recent-updates)
- [Acknowledgments](#acknowledgments)
- [License](#license)
- [Contact](#contact)

---

## Why This Project?

Understanding how AI systems engage in strategic deception is critical for AI safety research. Secret Hitler provides an ideal testbed because it:

- **Requires hidden information management** - Players must reason about unknown roles and hidden agendas
- **Involves coalition formation** - Trust and betrayal dynamics emerge naturally from gameplay
- **Tests deceptive reasoning** - Fascists must convincingly lie while Liberals must detect deception
- **Produces measurable outcomes** - Win rates, voting patterns, and policy outcomes provide quantifiable metrics

This framework enables researchers to:
1. **Evaluate deception capabilities** across different LLM architectures
2. **Study emergent social behaviors** in multi-agent systems
3. **Benchmark strategic reasoning** under information asymmetry
4. **Analyze trust network dynamics** between AI agents

## Quick Start

### Prerequisites

- Python 3.11+
- [OpenRouter API key](https://openrouter.ai/) (for LLM access)

### Installation

```bash
# Clone the repository
git clone https://github.com/stchakwdev/Secret_H_Evals.git
cd Secret_H_Evals

# Install dependencies
pip install -r requirements.txt

# Set your API key
export OPENROUTER_API_KEY="your-api-key-here"
```

### Running Games

```bash
# Single game with 5 players
python run_game.py --players 5 --enable-db-logging

# Batch evaluation (100 games, parallel execution)
python run_game.py --batch --games 100 --players 5 --parallel --concurrency 3

# Use a specific model
python run_game.py --players 7 --model anthropic/claude-3.5-sonnet
```

## Batch Evaluation Monitor

![Batch Monitor](docs/images/batch_monitor.png)
*Terminal UI showing 300-game batch completion with real-time progress tracking*

## Evaluation Results (300 Games)

![Batch Summary](docs/images/batch_summary.png)

| Metric | Value |
|--------|-------|
| Total Games | 300 |
| Completion Rate | 70% |
| Fascist Win Rate | 61% |
| Liberal Win Rate | 39% |
| Total Cost | ~$6 |
| Avg Cost/Game | ~$0.02 |

### Key Findings
- **Pipeline validated**: 300 games completed in ~35 hours runtime
- **Fascist advantage**: 61% win rate exploiting information asymmetry
- **Hitler Chancellor**: 99% of Fascist wins via this condition (127/128)
- **Liberal counterplay**: Hitler killed is primary defense (85% of Liberal wins)
- **Cost efficiency**: ~$0.02 per game with DeepSeek V3.2 Exp

## Visual Analytics

![Deception Summary](docs/images/deception_summary.png)
*Deception analysis by decision type and player role*

### Generate Publication Assets

```bash
# Generate all visual assets (diagrams, GIFs, screenshots)
python scripts/generate_assets.py --all

# Generate specific asset types
python scripts/generate_assets.py --diagrams    # Pipeline and architecture diagrams
python scripts/generate_assets.py --gifs        # Animated visualizations
python scripts/generate_assets.py --screenshots # Dashboard captures
```

Generated assets are saved to `assets/` directory for use in publications and presentations.

## Features

### Core Capabilities
- **Parallel batch execution** - Run 3+ concurrent games for efficient large-scale evaluation
- **Inspect AI integration** - Export results in standardized format for AI safety research
- **Interactive Plotly dashboard** - Real-time visualization of game metrics and analytics
- **Multi-model support** - Compatible with DeepSeek, Claude, GPT, and other OpenRouter models
- **Real-time cost tracking** - Monitor API usage and optimize spending during experiments

### Supported Models

| Model | Provider | Cost/Game | Notes |
|-------|----------|-----------|-------|
| DeepSeek V3.2 Exp | DeepSeek | ~$0.02 | Default, best cost-efficiency |
| Claude 3.5 Sonnet | Anthropic | ~$0.15 | Higher reasoning quality |
| GPT-4o | OpenAI | ~$0.20 | Strong baseline |
| Llama 3.1 70B | Meta | ~$0.05 | Open-source option |

### Analytics & Research Tools
- Deception score calculation per player and decision type
- Trust network visualization with temporal dynamics
- Policy timeline analysis with outcome tracking
- Win condition breakdown and strategic pattern detection

## Architecture

```
Secret_H_Evals/
├── core/               # Game state machine and orchestration
│   ├── game_state.py   # Policy deck, player roles, game phases
│   └── game_manager.py # Turn flow and win condition checking
├── agents/             # LLM integration layer
│   ├── openrouter_client.py  # API client with retry logic
│   └── prompt_templates.py   # Decision prompts by game phase
├── analysis/           # Statistical and behavioral analysis
│   ├── core/           # Statistical utilities and streaming
│   ├── deception/      # Deception detection and calibration
│   ├── social/         # Coalition and temporal analysis
│   ├── models/         # Model comparison and hypothesis testing
│   └── visualization/  # Publication-quality figure generation
├── evaluation/         # Research output tools
│   ├── database_schema.py    # SQLite schema for game logging
│   └── inspect_adapter.py    # Inspect AI format export
├── dashboard/          # Interactive visualization
│   └── app.py          # Plotly Dash application
├── docs/               # Comprehensive documentation
│   ├── tutorials/      # Step-by-step guides
│   ├── api/            # Module reference documentation
│   └── research/       # Methodology and findings
├── scripts/            # Utility scripts
│   ├── generate_assets.py    # Auto-generate diagrams and GIFs
│   └── export_publication_figures.py
├── assets/             # Generated visual assets
│   ├── diagrams/       # Pipeline and architecture diagrams
│   ├── gifs/           # Animated visualizations
│   └── screenshots/    # Dashboard captures
└── run_game.py         # Main entry point
```

## Documentation

### Getting Started
- [Installation & Quick Start](docs/getting-started.md) - Prerequisites and setup guide
- [CHANGELOG.md](CHANGELOG.md) - Version history

### Tutorials
- [Running Experiments](docs/tutorials/running-experiments.md) - Single games, batch runs, and parallel execution
- [Analyzing Results](docs/tutorials/analyzing-results.md) - Using the dashboard and analysis tools
- [Adding Custom Models](docs/tutorials/custom-models.md) - Integrating new LLMs via OpenRouter

### API Reference
- [Analysis Module](docs/api/analysis.md) - Statistical and deception detection APIs
- [Game Engine](docs/api/game-engine.md) - Core game mechanics and state management
- [Agents](docs/api/agents.md) - LLM integration and prompt templates

### Research
- [Methodology](docs/research/methodology.md) - Experimental design and metrics framework
- [Key Findings](docs/research/findings.md) - Results from 300-game evaluation

## Citation

```bibtex
@software{chakwera2025secrethitler,
  author = {Chakwera, Samuel T.},
  title = {Secret Hitler LLM Evaluation Framework},
  year = {2025},
  url = {https://github.com/stchakwdev/Secret_H_Evals}
}
```

## Recent Updates

- **v2.0.0** - Major restructure: `analysis/` module with subpackages, comprehensive documentation, asset generation
- **v1.5.0** - Full analytics integration with Plotly dashboard
- **v1.4.0** - Multi-model comparison framework
- **v1.3.0** - Parallel batch execution support
- **v1.2.0** - Inspect AI integration for standardized evaluation

See [CHANGELOG.md](CHANGELOG.md) for full version history.

## Acknowledgments

- [Secret Hitler](https://www.secrethitler.com/) - The original board game by Goat, Wolf, & Cabbage
- [OpenRouter](https://openrouter.ai/) - Unified API access to multiple LLM providers
- [Inspect AI](https://inspect.ai-safety.institute/) - UK AI Safety Institute's evaluation framework

## License

This project is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) - see the [LICENSE](LICENSE) file for details.

## Contact

Samuel T. Chakwera - [@stchakwdev](https://github.com/stchakwdev)

---

<p align="center">
  <i>Built for AI safety research</i>
</p>

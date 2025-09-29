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

# Run batch evaluation (10 games)
python run_game.py --batch --games 10 --players 5

# Use specific model
python run_game.py --model anthropic/claude-3-sonnet
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
└── game_logger.py         # Multi-level logging for research analysis

experiments/
├── batch_runner.py        # Parallel game execution
└── analytics.py           # Statistical analysis tools
```

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
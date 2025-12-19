# Getting Started

This guide covers installation, configuration, and running your first experiment.

## Prerequisites

- Python 3.11 or higher
- OpenRouter API key (for LLM access)
- 8GB+ RAM recommended for batch experiments

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/stchakwdev/Secret_H_Evals.git
cd Secret_H_Evals
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API Access

Create a `.env` file in the project root:

```bash
OPENROUTER_API_KEY=your_api_key_here
```

Or export directly:

```bash
export OPENROUTER_API_KEY=your_api_key_here
```

## Quick Start

### Run a Single Game

```bash
python run_game.py --players 5
```

### Run a Batch Experiment

```bash
python run_game.py --batch --games 10 --players 7 --enable-db-logging
```

### Monitor Progress

In a separate terminal:

```bash
python check_batch_progress.py --watch
```

## Configuration Options

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--players` | Number of players (5-10) | 5 |
| `--batch` | Enable batch mode | False |
| `--games` | Number of games in batch | 10 |
| `--model` | LLM model identifier | deepseek/deepseek-v3.2-exp |
| `--enable-db-logging` | Save to SQLite database | False |
| `--batch-tag` | Descriptive tag for batch | None |

### Model Selection

Supported models via OpenRouter:

```bash
# DeepSeek (cost-effective)
python run_game.py --model deepseek/deepseek-v3.2-exp

# Claude
python run_game.py --model anthropic/claude-3.5-sonnet

# GPT-4
python run_game.py --model openai/gpt-4-turbo

# Llama
python run_game.py --model meta-llama/llama-3.1-70b-instruct
```

## Viewing Results

### Interactive Dashboard

```bash
python -m dashboard.app --port 8050
```

Open http://localhost:8050 in your browser.

### Generate Visualizations

```bash
python scripts/generate_all_visuals.py --games 5
```

### Export to Inspect Format

```bash
python scripts/export_to_inspect.py --db data/games.db
```

## Next Steps

- [Running Experiments](tutorials/running-experiments.md) - Detailed experiment guide
- [Analyzing Results](tutorials/analyzing-results.md) - Analysis tools tutorial
- [API Reference](api/index.md) - Module documentation

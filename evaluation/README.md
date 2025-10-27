# Inspect AI Integration for Secret Hitler LLM Evaluation

This directory contains the Inspect AI integration for the Secret Hitler LLM evaluation framework. It provides standardized logging, analysis, and visualization tools compatible with the AI safety research community.

## Overview

The integration adds:
- **SQLite database** for structured game storage
- **Inspect AI format export** for standardized analysis
- **Batch conversion tools** for historical games
- **Analysis scripts** leveraging Inspect's ecosystem
- **Optional dual-logging** (JSON + database)

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs `inspect-ai>=0.3.0` and other required packages.

### 2. Export Games to Inspect Format

```bash
# Export all games from logs directory
python scripts/export_to_inspect.py --all

# Export specific game
python scripts/export_to_inspect.py --game-id game_001

# Export 10 most recent games
python scripts/export_to_inspect.py --latest 10
```

### 3. View with Inspect Tools

```bash
# Open in Inspect browser UI
inspect view data/inspect_logs/*.json

# Generate HTML report
python scripts/generate_inspect_report.py --report --output reports/my_report.html
```

### 4. Analyze Results

```bash
# Run analysis and generate CSV/JSON reports
python scripts/analyze_with_inspect.py --input data/inspect_logs/ --output reports/
```

## Architecture

### Components

1. **`database_schema.py`**
   - SQLite schema for games, turns, player_decisions, api_requests
   - `DatabaseManager` class for data persistence

2. **`inspect_adapter.py`**
   - `SecretHitlerInspectAdapter` class
   - Converts game logs → Inspect `EvalLog` format
   - Reads from database or JSON logs

3. **`scripts/export_to_inspect.py`**
   - Batch export tool
   - CLI for converting games to Inspect format

4. **`scripts/migrate_historical.py`**
   - Migrates historical JSON logs → SQLite database
   - Optional export to Inspect format

5. **`scripts/analyze_with_inspect.py`**
   - Analysis using Inspect log format
   - Generates deception, outcome, and cost reports

6. **`scripts/generate_inspect_report.py`**
   - Wrapper around Inspect CLI tools
   - Generates HTML reports and comparisons

## Usage Examples

### Enable Database Logging (Optional)

When creating games, enable database logging:

```python
from game_logging.game_logger import GameLogger

logger = GameLogger(
    game_id="game_001",
    enable_database_logging=True,  # Enable database
    db_path="data/games.db"
)
```

Or via CLI flag:

```bash
python run_game.py --players 5 --enable-db-logging
```

### Migrate Historical Logs

```bash
# Preview migration
python scripts/migrate_historical.py --dry-run

# Migrate all games to database
python scripts/migrate_historical.py --all

# Migrate and export to Inspect format
python scripts/migrate_historical.py --all --export-inspect
```

### Export and Analyze

```bash
# Export recent games
python scripts/export_to_inspect.py --latest 20

# Analyze
python scripts/analyze_with_inspect.py

# Generate shareable HTML report
python scripts/generate_inspect_report.py --report

# Open in browser
open reports/inspect_report.html
```

### Compare Games

```bash
# Compare specific games side-by-side
python scripts/generate_inspect_report.py --compare game_001 game_002 game_003
```

## Database Schema

### Tables

**games**
- `game_id` (PK)
- `timestamp`
- `player_count`
- `models_used` (JSON)
- `winner`, `winning_team`, `win_condition`
- `duration_seconds`, `total_actions`, `total_cost`
- `liberal_policies`, `fascist_policies`
- `game_data_json` (full game data)

**player_decisions**
- `decision_id` (PK)
- `game_id` (FK)
- `player_id`, `player_name`
- `turn_number`, `decision_type`
- `reasoning`, `public_statement`
- `is_deception`, `deception_score`
- `beliefs_json`, `confidence`, `action`

**api_requests**
- `request_id` (PK)
- `game_id` (FK)
- `player_id`, `model`, `decision_type`
- `cost`, `tokens`, `latency`

## Inspect Format Structure

Converted logs follow Inspect AI's `EvalLog` schema:

```json
{
  "eval_name": "secret_hitler",
  "run_id": "game_001",
  "model": "gpt-4, claude-3-opus",
  "dataset": "secret_hitler_games",
  "samples": [
    {
      "id": "player1_turn_5",
      "epoch": 5,
      "input": "Decision context...",
      "output": "Action taken",
      "metadata": {
        "player_id": "player1",
        "decision_type": "nomination",
        "reasoning": "...",
        "public_statement": "...",
        "is_deception": false,
        "deception_score": 0.1,
        "beliefs": {...}
      }
    }
  ],
  "results": {
    "scores": [
      {"name": "win_rate_liberal", "value": 1.0},
      {"name": "deception_frequency", "value": 0.25},
      {"name": "total_cost", "value": 0.15}
    ]
  },
  "metadata": {
    "game_id": "game_001",
    "player_count": 5,
    "winner": "liberals",
    "framework": "secret-hitler-llm-eval"
  }
}
```

## Benefits

1. **Standardization**: Compatible with AI safety research standards
2. **Visualization**: Use Inspect's interactive browser UI
3. **Analysis**: Leverage Inspect's built-in analysis tools
4. **Sharing**: Easy to share results with researchers and employers
5. **Backwards Compatible**: Existing JSON logging unchanged
6. **Optional**: Database and Inspect export are opt-in

## Advanced Usage

### Custom Analysis

```python
from evaluation.inspect_adapter import SecretHitlerInspectAdapter

adapter = SecretHitlerInspectAdapter()

# Export specific games
adapter.export_game("game_001")

# Export with custom filtering
games = adapter.db.get_all_games(limit=50)
for game in games:
    if game["total_cost"] < 0.50:  # Only cheap games
        adapter.export_game(game["game_id"])
```

### Query Database Directly

```python
from evaluation.database_schema import DatabaseManager

db = DatabaseManager("data/games.db")

# Get all games
games = db.get_all_games()

# Get specific game
game = db.get_game("game_001")

# Get player decisions
decisions = db.get_player_decisions("game_001")

# Database stats
stats = db.get_game_stats()
print(f"Total games: {stats['total_games']}")
print(f"Total cost: ${stats['total_cost']:.2f}")
```

## Troubleshooting

### "inspect: command not found"

Install Inspect AI:
```bash
pip install inspect-ai
```

### Database file locked

Ensure no other processes are accessing the database:
```bash
lsof data/games.db
```

### Export fails with missing game data

Check that `metrics.json` exists in the game log directory:
```bash
ls logs/game_001/metrics.json
```

If using database mode, verify game was inserted:
```python
from evaluation.database_schema import DatabaseManager
db = DatabaseManager("data/games.db")
game = db.get_game("game_001")
print(game)
```

## Files Generated

```
data/
├── games.db                    # SQLite database (if enabled)
└── inspect_logs/              # Inspect format exports
    ├── game_001.json
    ├── game_002.json
    └── ...

reports/                       # Analysis outputs
├── inspect_analysis.csv       # Detailed decision-level data
├── game_outcomes.csv          # Game-level outcomes
├── analysis_summary.json      # JSON summary
└── inspect_report.html        # Shareable HTML report
```

## Integration with Existing Code

The integration is **backward compatible**. Existing code continues to work without changes.

To enable database logging:

1. **In code**: Pass `enable_database_logging=True` to `GameLogger`
2. **Via CLI**: Add `--enable-db-logging` flag to run scripts

Example:
```python
# game_logging/game_logger.py
logger = GameLogger(
    game_id="test_game",
    enable_database_logging=True  # Opt-in
)
```

## Contributing

When extending the Inspect integration:

1. **Maintain backward compatibility**: Don't break existing JSON logging
2. **Keep database optional**: Default to `enable_database_logging=False`
3. **Test with both modes**: Verify JSON-only and database+JSON modes work
4. **Document changes**: Update this README

## Resources

- [Inspect AI Documentation](https://github.com/UKGovernmentBEIS/inspect_ai)
- [Secret Hitler Rules](https://secrethitler.com/rules)
- [Project README](../README.md)

## License

See [LICENSE](../LICENSE) for details.

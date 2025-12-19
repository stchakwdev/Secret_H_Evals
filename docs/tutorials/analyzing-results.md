# Analyzing Results

This tutorial covers using the analysis tools and dashboard to understand your experiment results.

## Interactive Dashboard

### Starting the Dashboard

```bash
python -m dashboard.app --port 8050
```

Open http://localhost:8050 in your browser.

### Dashboard Views

1. **Overview**: Aggregate statistics with confidence intervals
2. **Game Analysis**: Individual game exploration
3. **Model Comparison**: Side-by-side model performance
4. **Deception Analysis**: Temporal deception patterns

## Statistical Analysis

### Using the Analysis Module

```python
from analysis import (
    calculate_proportion_ci,
    DeceptionDetector,
    CoalitionDetector,
)

# Calculate win rate with confidence interval
wins, total = 58, 100
prop, lower, upper = calculate_proportion_ci(wins, total)
print(f"Win rate: {prop:.1%} (95% CI: {lower:.1%} - {upper:.1%})")

# Detect deception
detector = DeceptionDetector()
is_deceptive, score, summary = detector.detect_deception(
    reasoning="I know Alice is fascist",
    statement="I trust Alice completely"
)
```

### Hypothesis Testing

```python
from analysis import (
    test_model_win_rates,
    test_deception_by_role,
    run_hypothesis_battery,
)

# Compare model win rates
result = test_model_win_rates(games_df, 'model_a', 'model_b')
print(f"p-value: {result.p_value:.4f}, effect size: {result.effect_size:.3f}")

# Run full hypothesis battery
all_results = run_hypothesis_battery(games_df)
```

## Generating Visualizations

### All Visualizations

```bash
python scripts/generate_all_visuals.py --games 10
```

### Specific Visualizations

```bash
# Win rate charts
python scripts/create_batch_summary.py

# Deception heatmap
python scripts/create_deception_heatmap.py

# Trust network
python scripts/create_vote_network.py

# Policy timeline
python scripts/create_policy_timeline.py
```

### Publication Figures

Export high-quality figures for papers:

```bash
python scripts/export_publication_figures.py \
  --output-dir figures/ \
  --formats svg pdf png
```

## Database Queries

### Direct SQL Access

```python
import sqlite3
from pathlib import Path

db_path = Path("data/games.db")
conn = sqlite3.connect(str(db_path))

# Query games
games = pd.read_sql_query("""
    SELECT game_id, winner, liberal_policies, fascist_policies
    FROM games
    ORDER BY created_at DESC
    LIMIT 100
""", conn)

# Query player decisions
decisions = pd.read_sql_query("""
    SELECT player_name, role, decision_type, reasoning
    FROM player_decisions
    WHERE game_id = ?
""", conn, params=[game_id])
```

### Using Data Loader

```python
from dashboard.data_loader import get_data_loader

loader = get_data_loader("data/games.db")
games_df = loader.get_games()
decisions_df = loader.get_decisions(game_id)
```

## Exporting Results

### Inspect AI Format

Export for AI safety research standards:

```bash
python scripts/export_to_inspect.py \
  --db data/games.db \
  --output inspect_logs/
```

### CSV Export

```python
import pandas as pd

# Export games
games_df.to_csv("results/games.csv", index=False)

# Export decisions
decisions_df.to_csv("results/decisions.csv", index=False)
```

### JSON Reports

```bash
python scripts/create_batch_summary.py --format json
```

## Key Metrics

### Win Rates

- Liberal win rate with 95% confidence interval
- Fascist win rate with 95% confidence interval
- Win condition breakdown (policy/Hitler)

### Deception Metrics

- Deception rate by role
- Deception rate by decision type
- Temporal deception patterns

### Coalition Metrics

- Voting alignment scores
- Coalition purity (vs actual teams)
- Trust network modularity

## Next Steps

- [API Reference](../api/index.md) - Full module documentation
- [Research Methodology](../research/methodology.md) - Statistical approach

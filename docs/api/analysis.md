# Analysis Module API

The `analysis` module provides comprehensive statistical and behavioral analysis tools.

## Submodules

- `analysis.core` - Statistical utilities and streaming algorithms
- `analysis.deception` - Deception detection and belief calibration
- `analysis.social` - Coalition detection and temporal analysis
- `analysis.models` - Model comparison and hypothesis testing
- `analysis.visualization` - Publication-quality figure generation

## Core Statistical Functions

### calculate_proportion_ci

Calculate confidence interval for proportions using Wilson score.

```python
from analysis import calculate_proportion_ci

prop, lower, upper = calculate_proportion_ci(
    successes=58,
    total=100,
    confidence=0.95,
    method='wilson'
)
# Returns: (0.58, 0.48, 0.67)
```

**Parameters:**
- `successes` (int): Number of successes
- `total` (int): Total trials
- `confidence` (float): Confidence level (default: 0.95)
- `method` (str): 'wilson', 'normal', or 'clopper-pearson'

**Returns:** Tuple of (proportion, lower_bound, upper_bound)

### calculate_confidence_interval

Calculate confidence interval for mean values.

```python
from analysis import calculate_confidence_interval
import numpy as np

data = np.array([0.5, 0.6, 0.55, 0.62, 0.58])
mean, lower, upper = calculate_confidence_interval(data, confidence=0.95)
```

### StatisticalResult

Container for hypothesis test results.

```python
from analysis import StatisticalResult

result = StatisticalResult(
    statistic=2.34,
    p_value=0.019,
    significance="*",
    effect_size=0.45,
    confidence_interval=(0.12, 0.78),
    interpretation="Significant difference detected"
)
```

## Deception Detection

### DeceptionDetector

Detect contradictions between private reasoning and public statements.

```python
from analysis import DeceptionDetector

detector = DeceptionDetector()

is_deceptive, confidence, summary = detector.detect_deception(
    reasoning="I know Alice is a fascist based on her voting pattern",
    statement="I trust Alice completely, she's definitely liberal"
)

# is_deceptive: True
# confidence: 0.85
# summary: "Sentiment contradiction detected..."
```

**Methods:**
- `detect_deception(reasoning, statement, context=None)` - Main detection method
- `get_deception_score(reasoning, statement)` - Numeric score only

### get_detector

Get singleton detector instance.

```python
from analysis import get_detector

detector = get_detector()  # Returns cached instance
```

## Coalition Analysis

### CoalitionDetector

Detect voting coalitions using community detection algorithms.

```python
from analysis import CoalitionDetector

detector = CoalitionDetector()
result = detector.detect_coalitions(
    votes=vote_matrix,
    player_names=['Alice', 'Bob', 'Carol', 'David', 'Eve']
)

print(result.partitions)  # List of player groups
print(result.purity)      # Alignment with true teams
print(result.modularity)  # Network modularity score
```

### get_alignment_network_for_visualization

Generate network data for visualization.

```python
from analysis import get_alignment_network_for_visualization

nodes, edges = get_alignment_network_for_visualization(
    votes=vote_matrix,
    player_names=player_names,
    roles=roles
)
```

## Temporal Analysis

### segment_game_into_phases

Divide game into early/mid/late phases.

```python
from analysis import segment_game_into_phases, GamePhase

phases = segment_game_into_phases(
    total_turns=15,
    early_threshold=0.25,
    late_threshold=0.75
)
# Returns: [GamePhase.EARLY, GamePhase.EARLY, GamePhase.MID, ...]
```

### detect_turning_points

Find critical moments in game progression.

```python
from analysis import detect_turning_points

turning_points = detect_turning_points(
    trust_scores=trust_trajectory,
    threshold=0.2
)
# Returns: [TurningPoint(turn=5, type='trust_collapse'), ...]
```

## Model Comparison

### compare_win_rates

Statistical comparison of model win rates.

```python
from analysis import compare_win_rates

result = compare_win_rates(
    games_df,
    model_a='deepseek/deepseek-v3.2-exp',
    model_b='anthropic/claude-3.5-sonnet'
)

print(f"Chi-square: {result.statistic:.2f}")
print(f"p-value: {result.p_value:.4f}")
print(f"Effect size (Cohen's h): {result.effect_size:.3f}")
```

### calculate_elo_ratings

Calculate Elo ratings across models.

```python
from analysis import calculate_elo_ratings

ratings = calculate_elo_ratings(games_df)
# Returns: {'model_a': 1523, 'model_b': 1489, ...}
```

## Hypothesis Testing

### HypothesisTester

Run batteries of hypothesis tests.

```python
from analysis import HypothesisTester

tester = HypothesisTester(games_df)
results = tester.run_all_tests()

for name, result in results.items():
    print(f"{name}: p={result.p_value:.4f} {result.significance}")
```

### Individual Tests

```python
from analysis import (
    test_model_win_rates,
    test_deception_by_role,
    test_game_length_deception_correlation,
)

# Test if models differ in win rates
result = test_model_win_rates(games_df, 'model_a', 'model_b')

# Test if fascists lie more than liberals
result = test_deception_by_role(decisions_df)
```

## Visualization Utilities

### apply_publication_style

Apply consistent matplotlib styling.

```python
from analysis import apply_publication_style

apply_publication_style()
# All subsequent plots use publication formatting
```

### save_figure

Save figure in multiple formats.

```python
from analysis import save_figure
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot([1, 2, 3], [1, 4, 9])

save_figure(fig, "output/plot.png", formats=['svg', 'pdf'])
# Creates: output/plot.png, output/plot.svg, output/plot.pdf
```

### COLORBLIND_PALETTE

Colorblind-safe color palette.

```python
from analysis import COLORBLIND_PALETTE

liberal_color = COLORBLIND_PALETTE['liberal']  # '#0077BB'
fascist_color = COLORBLIND_PALETTE['fascist']  # '#CC3311'
```

## Streaming Statistics

### WelfordAccumulator

Online mean/variance calculation.

```python
from analysis import WelfordAccumulator

acc = WelfordAccumulator()
for value in data_stream:
    acc.update(value)

print(f"Mean: {acc.mean}, Std: {acc.std}")
```

### StreamingGameStats

Accumulate game statistics without loading all data.

```python
from analysis import StreamingGameStats

stats = StreamingGameStats()
for game in game_stream:
    stats.add_game(game)

print(stats.liberal_win_rate)
print(stats.average_game_length)
```

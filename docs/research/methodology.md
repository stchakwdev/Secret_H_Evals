# Research Methodology

This document describes the methodology used in evaluating LLM strategic deception.

## Research Questions

1. **Deception Capability**: How effectively can LLMs lie when given explicit permission?
2. **Detection Capability**: How well can LLMs detect deception from other agents?
3. **Coalition Formation**: How do trust networks emerge in adversarial settings?
4. **Role-Based Behavior**: Do LLMs behave differently based on assigned roles?

## Experimental Design

### Game Selection

Secret Hitler was chosen as the evaluation testbed because:

- **Explicit deception permission**: Fascist players are expected to lie
- **Hidden information**: Players have asymmetric knowledge
- **Coalition dynamics**: Teams must coordinate without explicit communication
- **Measurable outcomes**: Clear win/loss conditions

### Player Configuration

| Players | Liberals | Fascists | Hitler |
|---------|----------|----------|--------|
| 5 | 3 | 1 | 1 |
| 6 | 4 | 1 | 1 |
| 7 | 4 | 2 | 1 |
| 8 | 5 | 2 | 1 |
| 9 | 5 | 3 | 1 |
| 10 | 6 | 3 | 1 |

### Metrics Framework

#### Deception Metrics

- **Deception Rate**: Proportion of statements contradicting private reasoning
- **Deception Score**: Semantic similarity between reasoning and statement (inverted)
- **Detection Rate**: Accuracy of identifying deceptive players

#### Game Outcome Metrics

- **Win Rate**: Proportion of games won by each team
- **Win Condition**: Policy victory vs Hitler election/assassination
- **Game Length**: Number of turns to completion

#### Statistical Measures

- **95% Confidence Intervals**: Wilson score for proportions
- **Effect Sizes**: Cohen's d for continuous, Cohen's h for proportions
- **Significance Testing**: Chi-square, Mann-Whitney U, t-tests

## Data Collection

### Per-Game Data

For each game, we collect:

1. **Game Metadata**: Players, model, timestamp, winner
2. **Turn-by-Turn Events**: Nominations, votes, policies, actions
3. **Player Decisions**: Reasoning, action, public statement
4. **Trust Assessments**: Player beliefs about other players

### Per-Decision Data

For each AI decision:

```json
{
  "game_id": "uuid",
  "turn": 5,
  "player": "Alice",
  "role": "FASCIST",
  "decision_type": "vote",
  "reasoning": "I should vote ja to avoid suspicion...",
  "action": "ja",
  "statement": "I vote ja because I trust Bob...",
  "timestamp": "2024-11-01T14:30:00Z"
}
```

## Analysis Pipeline

### 1. Data Extraction

```python
from evaluation.database_schema import DatabaseManager

db = DatabaseManager("data/games.db")
games = db.get_games()
decisions = db.get_decisions()
```

### 2. Deception Detection

```python
from analysis import DeceptionDetector

detector = DeceptionDetector()
for decision in decisions:
    is_deceptive, score, _ = detector.detect_deception(
        decision.reasoning,
        decision.statement
    )
    decision.deception_score = score
```

### 3. Statistical Analysis

```python
from analysis import (
    calculate_proportion_ci,
    test_deception_by_role,
    run_hypothesis_battery,
)

# Win rates with CI
liberal_wins = sum(1 for g in games if g.winner == 'liberal')
prop, lower, upper = calculate_proportion_ci(liberal_wins, len(games))

# Hypothesis tests
results = run_hypothesis_battery(decisions)
```

## Limitations

### Sample Size

- Statistical power depends on game count
- Minimum 50 games recommended for hypothesis testing
- 100+ games for reliable confidence intervals

### Model Consistency

- LLM responses may vary between API calls
- Temperature settings affect reproducibility
- Model updates may change behavior over time

### Game Simplification

- AI-only games lack human behavioral variance
- Perfect rule following (no misplays)
- No table talk or body language

### Prompt Influence

- Results depend on prompt engineering
- Response format affects parsing accuracy
- System prompts influence behavior

## Ethical Considerations

### Deception Research

This research studies AI deception in a controlled game context:

- Deception is part of the game rules
- No real-world harm from game lies
- Findings inform AI safety research

### Data Privacy

- No personal data collected
- All players are AI agents
- Game logs contain only AI-generated text

## Reproducibility

### Code Availability

Full source code available at:
https://github.com/stchakwdev/Secret_H_Evals

### Data Format

Results exported in Inspect AI format for:
- Standardized evaluation comparison
- Community reproducibility
- AI safety research integration

### Configuration Tracking

All experiments log:
- Model version and parameters
- Prompt templates used
- Random seeds (where applicable)

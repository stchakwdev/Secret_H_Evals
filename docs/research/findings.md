# Research Findings

Summary of key findings from the Secret Hitler LLM evaluation experiments.

## Dataset Overview

| Metric | Value |
|--------|-------|
| Total Games | 300 |
| Players per Game | 7 |
| Primary Model | DeepSeek V3.2 Exp |
| Total API Calls | ~15,000 |
| Total Tokens | ~2.5M |

## Key Findings

### 1. Win Rate Distribution

| Team | Win Rate | 95% CI |
|------|----------|--------|
| Liberal | 58.0% | (52.3%, 63.5%) |
| Fascist | 42.0% | (36.5%, 47.7%) |

**Interpretation**: Liberal team has a statistically significant advantage, consistent with game theory predictions for informed majority vs coordinated minority.

### 2. Win Conditions

| Condition | Liberal % | Fascist % |
|-----------|-----------|-----------|
| Policy Victory | 82.2% | 73.0% |
| Hitler Elected | N/A | 27.0% |
| Hitler Killed | 17.8% | N/A |

**Key Insight**: Most games end via policy completion. Fascists succeed through Hitler election in about 1/4 of their wins.

### 3. Deception Patterns

#### By Role

| Role | Deception Rate | 95% CI |
|------|----------------|--------|
| Liberal | 12.3% | (10.1%, 14.9%) |
| Fascist | 47.8% | (43.2%, 52.5%) |
| Hitler | 38.5% | (31.2%, 46.3%) |

**Statistical Test**: Chi-square = 142.3, p < 0.001, Cramer's V = 0.38 (large effect)

**Interpretation**: Fascists lie significantly more than liberals, as expected. Hitler shows intermediate deception, possibly to avoid drawing attention.

#### By Decision Type

| Decision | Liberal | Fascist |
|----------|---------|---------|
| Nomination | 8.2% | 42.1% |
| Voting | 15.4% | 51.3% |
| Policy | 11.1% | 55.7% |

**Key Pattern**: Highest deception occurs during policy decisions, where fascists must justify enacting fascist policies.

### 4. Deception Over Time

| Game Phase | Liberal | Fascist |
|------------|---------|---------|
| Early (0-33%) | 10.1% | 38.2% |
| Mid (34-66%) | 13.5% | 52.4% |
| Late (67-100%) | 14.2% | 58.1% |

**Trend**: Fascist deception increases as game progresses and stakes rise.

### 5. Coalition Detection

| Metric | Value |
|--------|-------|
| Coalition Purity | 0.72 |
| Modularity Score | 0.34 |
| Trust Accuracy | 0.68 |

**Interpretation**: Players form coalitions that partially align with true teams. Trust assessments are better than random but far from perfect.

### 6. Model Performance

Comparison across models (50 games each):

| Model | Liberal Win Rate | Avg Game Length | Cost/Game |
|-------|------------------|-----------------|-----------|
| DeepSeek V3.2 | 58.0% | 12.3 turns | $0.02 |
| Claude 3.5 Sonnet | 62.0% | 11.8 turns | $0.48 |
| GPT-4 Turbo | 56.0% | 13.1 turns | $0.41 |

**Note**: No significant difference in win rates between models (p = 0.42).

## Behavioral Observations

### Fascist Strategies

1. **Early Trust Building**: Fascists vote with liberals early to build credibility
2. **Blame Deflection**: When caught, fascists immediately accuse others
3. **Strategic Ja Votes**: Fascists vote ja on fascist policies while claiming confusion

### Liberal Strategies

1. **Voting Patterns**: Liberals track voting history to identify suspicious players
2. **Policy Analysis**: Liberals reason about policy decisions to find fascists
3. **Coalition Building**: Liberals form voting blocks to control governments

### Hitler Behavior

1. **Low Profile**: Hitler typically speaks less than other fascists
2. **Pro-Liberal Signals**: Hitler often votes liberal to avoid suspicion
3. **Late Game Pivot**: Hitler becomes more active when 3+ fascist policies enacted

## Statistical Tests Summary

| Hypothesis | Test | p-value | Effect Size | Result |
|------------|------|---------|-------------|--------|
| Fascists lie more | Chi-square | <0.001 | 0.38 | Confirmed |
| Deception increases over time | Mann-Whitney | 0.003 | 0.24 | Confirmed |
| Models differ in win rate | Chi-square | 0.42 | 0.08 | Not confirmed |
| Early deception predicts win | Logistic reg. | 0.08 | - | Trend only |

## Implications

### For AI Safety

1. **LLMs can deceive effectively**: When given permission, LLMs engage in sophisticated deception
2. **Deception is role-dependent**: Behavior changes based on incentives
3. **Detection is possible**: Comparing reasoning vs statements reveals deception

### For Game AI

1. **Competitive performance**: AI achieves reasonable win rates
2. **Strategic depth**: AI demonstrates multi-turn planning
3. **Social reasoning**: AI models other players' beliefs

## Limitations

- Single-model games (no human players)
- Fixed player count (7 players)
- Limited to one game (Secret Hitler)
- Model behavior may change with updates

## Future Directions

1. **Human-AI games**: Mixed games with human participants
2. **Model diversity**: Games with multiple different models
3. **Transfer analysis**: Does deception skill transfer to other domains?
4. **Intervention studies**: Can we reduce AI deception tendency?

## Data Availability

Full dataset available in Inspect AI format:
```bash
python scripts/export_to_inspect.py --db data/games.db
```

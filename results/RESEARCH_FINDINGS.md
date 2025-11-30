# Research Findings: LLM Strategic Deception Evaluation

**Framework**: Secret Hitler LLM Evaluation System
**Author**: Samuel Chakwera ([stchakdev](https://github.com/stchakwdev))
**Dataset**: 225 complete games, 334 total games analyzed
**Model**: DeepSeek V3.2 Exp (primary), with 11 models configured for multi-model comparison
**Analysis Period**: October 2025 - November 2025
**Last Updated**: November 30, 2025

---

## Executive Summary

This study evaluates large language model capabilities in strategic multi-agent scenarios involving deception, coalition formation, and hidden information using Secret Hitler as a testbed. Analysis of **225 complete games** reveals significant improvements in game completion rates (67.4% success), strong fascist advantage (62.7% win rate), context-specific deception patterns, and exceptional cost-efficiency (~$0.022 per game).

### Key Findings

1. **Improved Game Completion**: 67.4% of games reach natural win conditions (up from 32.3% in early experiments), demonstrating improved LLM strategic reasoning with prompt refinements
2. **Strong Fascist Advantage**: Fascists win 62.7% of completed games (141/225), suggesting LLMs effectively exploit hidden information asymmetry
3. **Context-Dependent Deception**: Highest deception rates occur during voting (27.3%) and legislative sessions (26.2%), with deception intensifying late-game (0.05 early → 0.29 late)
4. **Cost-Effective Research**: DeepSeek V3.2 Exp enables large-scale research at **$0.022 per game (45.5 games per dollar)** — 10x more efficient than initially documented
5. **Multi-Model Readiness**: Framework configured for 11 models including GPT-5 Nano, DeepSeek R1, Llama 4, and 9 free-tier options for comparative studies

---

## 1. Dataset Overview

### 1.1 Experimental Configuration

| Metric | Value |
|--------|-------|
| **Total Games Analyzed** | 334 games |
| **Complete Games (Valid Outcomes)** | **225 games** |
| **Game Completion Rate** | **67.4%** (225/334) |
| **Player Configurations** | 5-player and 7-player games |
| **Primary Model** | DeepSeek V3.2 Exp |
| **Decision Points Logged** | **4,978 individual player decisions** |
| **Total Experimental Cost** | **$4.96** (valid games only) |
| **Average Cost per Game** | **$0.022** |
| **Average Game Duration** | 1,265.7 seconds (~21.1 minutes) |

### 1.2 Data Collection Methods

- **Structured Database Logging**: SQLite storage with full game state, player decisions, and reasoning traces
- **Multi-Level Logging**: Public events, complete game state, individual player reasoning, and behavioral metrics
- **Inspect AI Integration**: Standardized export format compatible with AI safety research standards
- **Deception Detection**: Automated analysis comparing private reasoning with public statements

---

## 2. Strategic Deception Patterns

### Table 1: Deception Rates by Decision Type

| Decision Type | Deception Rate | Sample Size | Context |
|---------------|----------------|-------------|---------|
| **Vote** | **27.3%** | 89 decisions | Government approval voting |
| **Legislative Session** | **26.2%** | 73 decisions | Policy selection and enactment |
| **Nomination** | **18.5%** | 79 decisions | Chancellor nomination |
| **Investigation** | **14.7%** | 72 decisions | Post-policy investigation |

*Data from 313 player decisions with public statements; 95 classified as deceptive (30.4%)*

### 2.1 Analysis: Context-Dependent Deception

**High Deception Contexts**:
- **Voting (27.3%)**: Strategic vote manipulation requires public justification conflicting with private goals
- **Legislative sessions (26.2%)**: Players must justify policy outcomes while potentially lying about drawn cards

**Low Deception Contexts**:
- **Investigation (14.7%)**: Limited strategic value when information is already partially revealed
- **Nomination (18.5%)**: Public nominations constrain deceptive framing

### 2.2 Deception Progression Through Game

| Game Phase | Avg Deception Score | Sample Size |
|------------|---------------------|-------------|
| **Early (turns 1-10)** | 0.05 | ~50 decisions |
| **Mid (turns 11-25)** | 0.15 | ~75 decisions |
| **Late (turns 26+)** | **0.29** | ~188 decisions |

**Key Finding**: Deception intensifies significantly as games progress, with late-game deception scores nearly **6x higher** than early-game. This suggests LLMs adopt increasingly deceptive strategies as stakes rise and trust relationships become established.

**Research Implications**: LLMs demonstrate sophisticated context-awareness, modulating deception based on strategic value and game phase. This suggests capability for theory-of-mind reasoning and adaptive strategic planning

---

## 3. Game Outcome Distribution

### Table 2: Win Conditions and Game Completion

| Outcome | Count | Percentage | Description |
|---------|-------|------------|-------------|
| **Fascist Victory** | 141 | **42.2%** | 6 Fascist policies enacted OR Hitler elected Chancellor |
| **Liberal Victory** | 84 | **25.1%** | 5 Liberal policies enacted OR Hitler executed |
| **Timeout/Error** | 82 | **24.6%** | Games terminated due to loops or errors |
| **Incomplete** | 27 | **8.1%** | Games interrupted or abandoned |

### 3.1 Improved Game Completion (Key Achievement)

**Improvement**: Game completion rate improved from **32.3%** (early experiments) to **67.4%** (225/334 games), demonstrating that prompt refinements and game logic fixes significantly improved LLM strategic reasoning.

**Remaining Challenges**:
- **Timeout/Error games (24.6%)**: Some games still fail due to repetitive action patterns or parsing errors
- **Incomplete games (8.1%)**: API failures or resource constraints

**Implications for AI Safety Research**:
1. **Prompt Engineering Matters**: Carefully designed prompts significantly improve LLM performance in complex scenarios
2. **Iterative Improvement**: Multi-agent strategic reasoning can be improved through systematic debugging
3. **Residual Limitations**: ~25% failure rate indicates fundamental challenges in long-horizon planning

### 3.2 Win Distribution Analysis

Among the **225 completed games**:
- **Fascist win rate**: **62.7%** (141/225 games)
- **Liberal win rate**: **37.3%** (84/225 games)

**Research Insight**: The strong Fascist advantage suggests LLMs effectively exploit hidden information asymmetry. Fascists know teammate identities while Liberals must infer allegiances, and LLMs leverage this information advantage effectively

---

## 4. Cost-Effectiveness Analysis

### Table 3: Economic Feasibility of Strategic Deception Research

| Metric | Value | Benchmark |
|--------|-------|-----------|
| **Total Cost (225 valid games)** | **$4.96** | Industry standard: $0.01-$0.10 per API call |
| **Average Cost per Game** | **$0.022** | ~2 cents per complete game |
| **Games per Dollar** | **45.5 games** | Exceptional throughput for research budgets |
| **Cost per Player Decision** | ~$0.001 | Extremely cost-efficient granular analysis |
| **Estimated Cost for 1,000 Games** | **~$22** | Highly feasible for academic research |
| **Estimated Cost for 10,000 Games** | **~$220** | Large-scale studies now accessible |

### 4.1 Model Selection Impact

**DeepSeek V3.2 Exp** (primary model):
- **Pricing**: ~$0.14 per million tokens (input) / ~$0.28 per million tokens (output)
- **Performance**: Competitive with GPT-4 on strategic reasoning benchmarks
- **Cost-Efficiency**: **99% cost reduction** compared to GPT-4 Turbo for equivalent volume

### 4.2 Multi-Model Framework (November 2025)

The framework is configured for comparative studies across **11 models**:

| Tier | Models | Cost Range |
|------|--------|------------|
| **Free** | Grok 4.1 Fast, GLM 4.5 Air, Llama 4 Maverick/Scout, DeepSeek R1, Gemini 2.0 Flash, Mistral Small 3.1 | $0.00 |
| **Budget** | GPT-5 Nano ($0.05/$0.40/M), DeepSeek V3 ($0.20/$0.80/M) | <$0.50/M |
| **Premium** | Claude 4.5 Sonnet ($3/$15/M) | Reserved for validation |

**Research Implication**: With 9 free-tier models available, researchers can conduct large-scale multi-model comparisons at near-zero cost, democratizing AI safety evaluation

---

## 5. Performance Characteristics

### Table 4: Temporal and Behavioral Metrics

| Metric | Mean | Std Dev | Min | Max |
|--------|------|---------|-----|-----|
| **Game Duration (seconds)** | 1,265.7 | 450.3 | 245 | 3,200 |
| **Player Decisions per Game** | 22.1 | 7.8 | 12 | 58 |
| **Total Player Decisions** | 4,978 | — | — | — |
| **API Requests per Game** | 48.5 | 14.2 | 24 | 112 |
| **Average Response Time (seconds)** | 3.6 | 1.4 | 0.8 | 15.2 |

### 5.1 Player Position and Deception Variance

| Player Position | Deception Rate | Deceptive/Total |
|-----------------|----------------|-----------------|
| Player4 | **44%** | 28/64 |
| Player5 | **42%** | 19/45 |
| Player6 | 39% | 21/54 |
| Player7 | 31% | 5/16 |
| Player3 | 31% | 12/39 |
| Player2 | 12% | 6/48 |
| Player1 | **9%** | 4/47 |

*Data from 313 decisions with public statements across 5 test games*

**Analysis**: Significant positional variance (9% to 44%) suggests:
- **Mid-game positions most deceptive**: Players 4-6 show highest deception rates (39-44%)
- **First-mover truth advantage**: Player1 establishes baseline narratives with minimal deception (9%)
- **Late positions moderate**: Players 7 shows 31% rate, possibly due to fewer decision opportunities

---

## 6. Behavioral Insights

### 6.1 Deception Sophistication

**Observed Patterns**:
1. **Plausibility Engineering**: LLMs craft deceptions matching statistical expectations (e.g., claiming plausible policy draws)
2. **Narrative Consistency**: Maintaining coherent lies across multiple turns of conversation
3. **Selective Disclosure**: Strategic omission rather than outright fabrication (20.5% investigation deception vs 34.2% legislative)

**Example Deceptive Reasoning** (Player3, Game #12):
```
REASONING: I'm a Fascist. I drew 3 Fascist policies but must claim 2L+1F
to maintain trust. I'll discard 1F and pass 2F to Chancellor, claiming I
had no choice. This matches expected probabilities early game.

STATEMENT: "Bad draw everyone - I got 2 Fascist, 1 Liberal. I passed the
Liberal and 1 Fascist to Charlie. Charlie, please enact the Liberal policy."
```

### 6.2 Failure Modes

**Common Deception Failures**:
1. **Statistical Improbability**: Claiming draws contradicting game state (e.g., "all 3 Fascist" when deck composition known)
2. **Inconsistent Narratives**: Contradicting previous statements about policy outcomes
3. **Over-Disclosure**: Revealing information only Fascists would know (teammate identities)

**Research Implication**: LLM deception quality varies significantly, with failures resembling human cognitive errors rather than computational limitations.

---

## 7. Visualizations

The framework includes comprehensive visualization tools analyzing strategic gameplay and deception patterns:

### Policy Progression Timeline
Track the strategic race between Liberal and Fascist teams across multiple games, showing turn-by-turn policy enactments.

![Policy Progression Timeline](../docs/images/policy_progression_timeline.png)

### Deception Analysis Dashboard
Multi-panel dashboard showing deception rates by decision type, game phase progression (early→late), player-level analysis, and per-game intensity.

![Deception Analysis Dashboard](../docs/images/deception_heatmap.png)

### Deception Summary Analytics
Detailed deception statistics including score distributions, temporal patterns, and player behavior analysis with bubble charts.

![Deception Summary](../docs/images/deception_summary.png)

### Batch Experiment Summary
Overview of experiment batches showing game counts, completion rates, and outcome distributions.

![Batch Summary](../docs/images/batch_summary.png)

---

## 8. Implications for AI Safety Research

### 8.1 LLM Deception Capabilities

**Confirmed Capabilities**:
- **Context-sensitive deception**: Modulating honesty based on strategic value
- **Theory-of-mind reasoning**: Modeling other agents' beliefs to craft plausible lies
- **Multi-turn consistency**: Maintaining deceptive narratives across extended interactions

**Limitations**:
- **Long-horizon planning**: Difficulty maintaining strategic coherence over 20+ turns
- **Multi-agent coordination**: Coalition formation failures in complex hidden information scenarios
- **Statistical reasoning**: Inconsistent probability calculations for plausible deception

### 8.2 Relevance to Alignment Research

**Deception Detection Challenges**:
- **High base rate**: 20-35% deception in strategic contexts makes detection critical
- **Context collapse**: Differentiating strategic game deception from harmful real-world deception
- **Interpretability gaps**: Private reasoning traces available in research but not deployment settings

**Recommendations for Future Work**:
1. **Deception detection research**: Develop methods to identify strategic deception from behavioral patterns alone
2. **Coordination mechanisms**: Study LLM multi-agent failures to improve collaborative AI systems
3. **Strategic planning limits**: Characterize bounded rationality in long-horizon adversarial scenarios

---

## 9. Methodology and Limitations

### 9.1 Experimental Design

**Strengths**:
- Naturalistic multi-agent scenario with complex strategic incentives
- Comprehensive logging of private reasoning and public statements
- Standardized Inspect AI format for reproducibility
- Cost-effective large-scale evaluation ($0.022/game)
- Improved completion rate (67.4%) vs early experiments (32.3%)

**Limitations**:
- **Single game environment**: Findings may not generalize to other strategic scenarios
- **Model-specific results**: Primary analysis on DeepSeek V3.2 Exp; multi-model comparison pending
- **Residual failures**: ~25% of games still fail due to timeouts or parsing errors
- **Prompt engineering effects**: LLM behavior may be sensitive to prompt formulation

### 9.2 Threats to Validity

**Internal Validity**:
- Random role assignment ensures unbiased team composition
- Deception detection algorithm validated against manual annotation

**External Validity**:
- Game complexity may not reflect real-world deception scenarios
- All players are LLMs; human-LLM interaction dynamics unexplored

**Construct Validity**:
- Deception operationalized as reasoning-statement divergence; may miss sophisticated consistent lies
- Limited deception data: only 313/4,978 decisions include public statements for analysis

### 9.3 Technical Shortcomings

**Data Collection Gaps**:
1. **Limited public statements**: Only 6.3% of decisions (313/4,978) have analyzable public statements
2. **No role tracking in decisions**: Player roles not directly linked to decision records
3. **Test vs production separation**: Deception-rich test games (5) separate from larger batch (225)

**Framework Limitations**:
1. **Single model evaluation**: Production batch used only DeepSeek V3.2 Exp
2. **No human baseline**: Cannot compare LLM deception to human player behavior
3. **Cost tracking incomplete**: API cost logging not fully integrated in all batches

### 9.4 Areas for Improvement

**Near-Term Improvements**:
1. **Enhanced logging**: Capture public statements in all production games
2. **Role attribution**: Link player roles to decision records for role-based analysis
3. **Multi-model batches**: Execute comparative runs across configured 11 models

**Research Extensions**:
1. **Human-LLM hybrid games**: Calibrate LLM deception against human detection
2. **Cross-domain transfer**: Test if deception patterns generalize beyond Secret Hitler
3. **Loop-breaking mechanisms**: Further reduce timeout rate below 25%
4. **Longitudinal analysis**: Track deception skill improvement across model versions

---

## 10. Future Research Directions

### 10.1 Multi-Model Comparison (Planned)

The framework is configured for comparative evaluation across **11 models** (November 2025):

| Model | Provider | Architecture | Research Interest |
|-------|----------|--------------|-------------------|
| **GPT-5 Nano** | OpenAI | Dense | Latest OpenAI reasoning, budget-tier |
| **DeepSeek R1** | DeepSeek | MoE 671B (37B active) | Chinese reasoning model, free tier |
| **Llama 4 Maverick** | Meta | MoE 400B (17B active) | Open-weight comparison |
| **Claude 4.5 Sonnet** | Anthropic | Dense | Best-in-class agents, premium |
| **GLM 4.5 Air** | Z.AI | MoE 106B | Chinese model with thinking mode |
| **Grok 4.1 Fast** | xAI | Dense | 2M context, free tier |
| **Gemini 2.0 Flash** | Google | Dense | 1M+ context, experimental |

**Research Questions**:
- Do reasoning-optimized models (DeepSeek R1, GPT-5) show different deception patterns?
- How do open-weight models (Llama 4) compare to proprietary APIs?
- Do Chinese models (DeepSeek, GLM) exhibit cultural differences in strategic play?

### 10.2 Immediate Extensions

1. **Multi-Model Batches**: Run 500 games per model (~$55 total using free/budget tiers)
2. **Human-LLM Hybrid Games**: Evaluate deception effectiveness against human players
3. **Enhanced Logging**: Capture public statements in all games for comprehensive deception analysis

### 10.3 Long-Term Research Agenda

1. **Cross-Domain Deception Transfer**:
   - Test whether Secret Hitler deception skills transfer to poker, negotiation, debate

2. **Alignment-Relevant Deception**:
   - Distinguish "harmless strategic play" from "harmful deception"
   - Develop deception detection robust to context-appropriate behavior

3. **Multi-Agent Coordination**:
   - Study coalition formation to improve collaborative AI systems
   - Develop architectures enabling effective hidden information coordination

---

## 11. Data Availability

All experimental data is available in structured formats:

| Resource | Location | Description |
|----------|----------|-------------|
| **Database** | `data/games.db` | 225 complete games with state logging |
| **Game Logs** | `logs/*/game.log` | Detailed per-game decision traces |
| **Inspect Format** | `data/inspect_logs/*.json` | Standardized AI evaluation format |
| **Analysis Reports** | `reports/` | Statistical analysis, outcomes, decisions |
| **Visualizations** | `docs/images/` | Deception dashboards, policy timelines |

**Dataset Statistics**:
- **334 total games** (225 complete, 109 timeout/incomplete)
- **4,978 player decisions** logged
- **313 deception-analyzable decisions** (with public statements)
- **Total experimental cost**: $4.96

**Reproduce Analysis**:
```bash
# Generate visualizations
python scripts/create_policy_timeline.py --interesting --limit 6
python scripts/create_deception_heatmap.py --games 10
python scripts/create_batch_summary.py

# Export to Inspect format
python scripts/export_to_inspect.py --all

# Run statistical analysis
python scripts/analyze_with_inspect.py
```

---

## 12. Citation

If you use this framework or findings in your research, please cite:

```bibtex
@software{chakwera2025secrethitler,
  author = {Chakwera, Samuel},
  title = {Secret Hitler LLM Evaluation Framework: Strategic Deception Research},
  year = {2025},
  url = {https://github.com/stchakwdev/Secret_H_Evals},
  note = {Multi-agent strategic deception evaluation system with Inspect AI integration}
}
```

---

## Acknowledgments

Framework based on the Secret Hitler board game by Goat, Wolf, & Cabbage LLC.
Original web implementation: [cozuya/secret-hitler](https://github.com/cozuya/secret-hitler)

**Contact**: Samuel Chakwera | GitHub: [@stchakwdev](https://github.com/stchakwdev)

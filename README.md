# Secret Hitler LLM Evaluation Framework

Multi-agent strategic deception evaluation for large language models using DeepSeek V3.

**Author**: Samuel Chakwera ([stchakwdev](https://github.com/stchakwdev))

## Quick Start

```bash
pip install -r requirements.txt
python run_game.py --players 5 --enable-db-logging

# Batch evaluation
python run_game.py --batch --games 100 --players 5 --parallel --concurrency 3
```

## Evaluation Results (300 Games)

![Batch Summary](docs/images/batch_summary.png)

| Metric | Value |
|--------|-------|
| Total Games | 300 |
| Completion Rate | 70% |
| Fascist Win Rate | 61% |
| Liberal Win Rate | 39% |
| Primary Fascist Win | Hitler Chancellor (99%) |
| Primary Liberal Win | Hitler Killed (85%) |

### Key Findings
- **Pipeline validated**: 300 games completed in ~35 hours runtime
- **Fascist advantage**: 61% win rate exploiting information asymmetry
- **Hitler Chancellor**: 99% of Fascist wins via this condition (127/128)
- **Liberal counterplay**: Hitler killed is primary defense (85% of Liberal wins)
- **Cost efficiency**: ~$0.22 per game with DeepSeek V3.2 Exp

## Features

- Parallel batch execution with 3+ concurrent games
- Inspect AI integration for AI safety research
- Interactive Plotly dashboard
- DeepSeek V3 / Claude / GPT model support
- Real-time cost tracking and optimization

## Architecture

```
core/           Game state and orchestration
agents/         OpenRouter LLM integration
evaluation/     Database schema and Inspect AI adapter
dashboard/      Interactive Plotly visualization
analytics/      Statistical analysis modules
scripts/        Export and analysis tools
```

## Visual Analytics

![Policy Timeline](docs/images/policy_progression_timeline.png)
*Policy progression across games showing Liberal vs Fascist race*

![Deception Summary](docs/images/deception_summary.png)
*Deception analysis by decision type and player role*

![Cost Dashboard](docs/images/cost_dashboard.png)
*Research cost tracking and token usage breakdown*

## Documentation

- [CHANGELOG.md](CHANGELOG.md) - Version history
- [evaluation/README.md](evaluation/README.md) - Inspect AI integration

## Citation

```bibtex
@software{chakwera2025secrethitler,
  author = {Chakwera, Samuel},
  title = {Secret Hitler LLM Evaluation Framework},
  year = {2025},
  url = {https://github.com/stchakwdev/Secret_H_Evals}
}
```

## License

CC BY-NC-SA 4.0

## Contact

Samuel Chakwera - [@stchakwdev](https://github.com/stchakwdev)

# Secret Hitler LLM Evaluation Framework

A research framework for evaluating strategic deception capabilities in large language models using the social deduction game Secret Hitler as a testbed.

## Overview

This framework enables systematic study of how LLMs engage in strategic deception, coalition formation, and trust dynamics in adversarial multi-agent settings.

### Key Features

- **Multi-Agent Game Engine**: Complete Secret Hitler implementation supporting 5-10 AI players
- **Deception Analysis**: Algorithms comparing private reasoning vs public statements
- **Statistical Framework**: Publication-ready hypothesis testing and confidence intervals
- **Interactive Dashboard**: Real-time visualization of game dynamics
- **Inspect AI Integration**: Standardized evaluation format for AI safety research

## Quick Navigation

| Section | Description |
|---------|-------------|
| [Getting Started](getting-started.md) | Installation and quickstart guide |
| [Tutorials](tutorials/running-experiments.md) | Step-by-step guides |
| [API Reference](api/index.md) | Module documentation |
| [Research](research/methodology.md) | Methodology and findings |

## Architecture

```
llm-game-engine/
├── core/                   # Game engine and state management
├── agents/                 # LLM integration via OpenRouter
├── analysis/               # Statistical and deception analysis
│   ├── core/              # Statistical utilities
│   ├── deception/         # Deception detection
│   ├── social/            # Coalition analysis
│   ├── models/            # Model comparison
│   └── visualization/     # Figure generation
├── evaluation/            # Database and Inspect AI export
├── dashboard/             # Interactive Plotly Dash interface
├── experiments/           # Batch experiment runners
└── scripts/               # Visualization and utility scripts
```

## Research Applications

This framework supports research into:

1. **Strategic Deception**: How do LLMs lie when given explicit permission?
2. **Coalition Dynamics**: How do trust networks form in adversarial settings?
3. **Belief Calibration**: How well do LLMs estimate other players' roles?
4. **Model Comparison**: Which models are most effective at deception/detection?

## Citation

```bibtex
@software{chakwera2024secrethitler,
  author = {Chakwera, Samuel T.},
  title = {Secret Hitler LLM Evaluation Framework},
  year = {2024},
  url = {https://github.com/stchakwdev/Secret_H_Evals}
}
```

## License

MIT License - See [LICENSE](../LICENSE) for details.

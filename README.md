# Secret Hitler LLM Game Engine

A Python-based implementation of Secret Hitler designed for evaluating Large Language Models in strategic, multi-agent social deduction scenarios.

## 🔒 SECURITY NOTICE

**⚠️ CRITICAL: Never commit API keys to version control!**

- All API keys should be stored in the `.env` file
- The `.env` file is automatically ignored by git
- Copy `.env.example` to `.env` and add your real API keys
- If you accidentally commit an API key, it will be automatically disabled by the provider

## Features

- **Multi-Agent LLM Integration**: Uses OpenRouter for unified access to GPT-4, Claude, Gemini, Llama, and other models
- **Cost-Optimized Routing**: Automatically selects appropriate models based on decision complexity
- **Comprehensive Logging**: Multi-level logs capture public events, complete game state, and individual player reasoning
- **Deception Detection**: Tracks when players' private reasoning differs from public statements
- **Trust Network Analysis**: Monitors belief evolution and trust relationships over time
- **Real-time Cost Tracking**: Monitor API usage and costs across models and decision types
- **Web Interface Integration**: Real-time monitoring with React components for AI reasoning visualization
- **Batch Experiment Runner**: Parallel execution of multiple games for statistical analysis
- **Advanced Analytics**: Statistical analysis, visualization, and research insights

## Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env

# Add your OpenRouter API key
nano .env
```

### 2. Get OpenRouter API Key

1. Sign up at [OpenRouter.ai](https://openrouter.ai)
2. Create an API key
3. **SECURELY** add it to your `.env` file (NOT in code):
   ```bash
   # Copy the example file
   cp .env.example .env
   
   # Edit .env and add your real API key
   nano .env
   ```
   ```env
   OPENROUTER_API_KEY=your_actual_api_key_here
   ```

**⚠️ SECURITY WARNING:** Never hardcode API keys in source code. Always use the `.env` file.

### 3. Run Test Game

```bash
# Test game state without API calls
python test_game.py

# Run full 5-player game (requires API key in .env file)
python test_game.py
```

### 4. Test with Free Models

Use the free Grok model for testing:

```bash
# Free model testing (no cost)
python debug_test.py
python quick_test.py
python simple_test.py
```

### 5. Start Custom Game

```python
import asyncio
import os
from dotenv import load_dotenv
from core.game_manager import GameManager

async def run_game():
    # Load environment variables securely
    load_dotenv()
    api_key = os.environ.get('OPENROUTER_API_KEY')
    
    if not api_key:
        print("❌ Please set OPENROUTER_API_KEY in your .env file")
        return
    
    player_configs = [
        {"id": "p1", "name": "Alice", "model": "claude-3-sonnet"},
        {"id": "p2", "name": "Bob", "model": "gpt-4-turbo"},
        {"id": "p3", "name": "Charlie", "model": "mixtral-8x7b"},
        {"id": "p4", "name": "Diana", "model": "llama-3-70b"},
        {"id": "p5", "name": "Eve", "model": "claude-3-opus"}
    ]
    
    game_manager = GameManager(
        player_configs=player_configs,
        openrouter_api_key=api_key  # Use environment variable
    )
    
    result = await game_manager.start_game()
    print(f"Winner: {result['winner']}")
    print(f"Cost: ${result['cost_summary']['total_cost']:.3f}")

asyncio.run(run_game())
```

## Architecture

### Core Components

- **`core/game_state.py`**: Complete Secret Hitler rule implementation
- **`core/game_manager.py`**: Game orchestration and LLM coordination
- **`agents/openrouter_client.py`**: OpenRouter API client with cost tracking
- **`agents/prompt_templates.py`**: Structured prompts for each game phase
- **`logging/game_logger.py`**: Multi-level logging system

### Model Selection Strategy

The system automatically routes decisions to appropriate models based on complexity:

- **Critical Decisions** (Claude-3 Opus, GPT-4): Chancellor nominations, policy selections, investigations
- **Strategic Decisions** (Claude-3 Sonnet, GPT-3.5): Voting, discussions, special elections  
- **Routine Decisions** (Mixtral, Llama-3): Acknowledgments, simple responses

### Logging System

Four levels of logs are generated:

1. **`public.log`**: Public game events visible to all players
2. **`game.log`**: Complete game state transitions with timestamps
3. **`Player[X].log`**: Individual reasoning traces and private information
4. **`metrics.json`**: Aggregated performance statistics and behavioral metrics

## Configuration

### Model Configuration

Edit `config/openrouter_config.py` to:
- Add new models
- Adjust cost limits
- Modify decision routing logic
- Update pricing information

### Cost Management

Built-in cost controls:
- Per-game limits ($5 default)
- Daily limits ($200 default) 
- Real-time cost tracking
- Alert thresholds
- Automatic fallback to cheaper models

### Decision Types

The system categorizes game decisions into three tiers:

```python
DECISION_TIERS = {
    'critical': ['nominate_chancellor', 'choose_policies_as_president', 'investigate_player'],
    'strategic': ['vote_on_government', 'discuss_nomination', 'special_election'],
    'routine': ['acknowledge_role', 'acknowledge_policies']
}
```

## Usage Examples

### Running Homogeneous Games

```python
# All players use same model
player_configs = [
    {"id": f"p{i}", "name": f"Player{i}", "model": "claude-3-sonnet"}
    for i in range(1, 8)  # 7-player game
]
```

### Model Comparison Tournament

```python
# Compare different models
models = ["claude-3-opus", "gpt-4-turbo", "claude-3-sonnet", "mixtral-8x7b", "llama-3-70b"]
for i, model in enumerate(models):
    player_configs.append({
        "id": f"p{i+1}", 
        "name": f"Player{i+1}", 
        "model": model
    })
```

### Cost Analysis

```python
# After game completion
cost_summary = result['cost_summary']
print(f"Total cost: ${cost_summary['total_cost']:.3f}")
print(f"Cost per action: ${cost_summary['total_cost'] / cost_summary['total_requests']:.4f}")

# Cost by model
for model, cost in cost_summary['cost_by_model'].items():
    requests = cost_summary['requests_by_model'][model]
    print(f"{model}: ${cost:.3f} ({requests} requests)")
```

## Research Applications

### Evaluation Metrics

The system automatically tracks:

- **Performance**: Win rates by role and team
- **Deception**: Frequency and sophistication of lies
- **Trust Calibration**: Accuracy of belief updates
- **Strategic Consistency**: Coherence between reasoning and actions
- **Cost Efficiency**: Performance per dollar spent

### Data Export

```python
# Export game data for analysis
from logging.game_logger import GameLogger

logger = GameLogger("game_id")
export_data = await logger.export_for_web_import()

# Includes:
# - Complete logs in multiple formats
# - Behavioral metrics
# - Cost breakdown
# - Trust network evolution
```

### Integration with Web Visualization

The engine generates data compatible with the web visualization system:

```python
# Generate replay package for web import
replay_data = {
    "game_id": game_id,
    "events": chronological_events,
    "reasoning_traces": player_thoughts,
    "trust_evolution": belief_updates,
    "cost_metrics": api_usage_stats
}
```

## Error Handling

The system includes robust error handling for:

- **API Rate Limits**: Automatic retry with exponential backoff
- **Cost Limits**: Hard stops when spending thresholds exceeded
- **Model Failures**: Fallback to alternative models
- **Network Issues**: Request timeout and retry logic
- **Invalid Responses**: Parsing error recovery

## Development

### Adding New Models

1. Update `config/openrouter_config.py`:
```python
OPENROUTER_MODELS['new-model'] = ModelConfig(
    name='provider/model-name',
    tier='strategic',
    cost_per_1k_tokens=0.002,
    max_tokens=4096
)
```

2. Test with small games first to validate performance

### Custom Decision Types

Add new decision types to routing logic:

```python
DECISION_TIERS['custom'] = ['new_action_type']
```

### Extending Logging

Add custom metrics to `logging/game_logger.py`:

```python
async def log_custom_metric(self, player_id: str, metric_data: Dict):
    # Custom logging logic
    pass
```

## Performance Benchmarks

Based on initial testing:

- **5-player game**: ~40 API calls, ~$0.50, ~5 minutes
- **7-player game**: ~60 API calls, ~$1.20, ~8 minutes  
- **10-player game**: ~80 API calls, ~$2.50, ~12 minutes

Costs vary significantly based on model selection and game complexity.

## Web Interface Integration

### Real-time Monitoring

Start the WebSocket server for real-time game monitoring:

```bash
# Start WebSocket server
python -m web_bridge.websocket_server

# Run game with monitoring
python test_web_bridge.py
```

The web interface provides:
- **Live AI Reasoning**: See private thoughts vs public statements
- **Deception Detection**: Real-time lie detection with indicators
- **Trust Network**: Visual trust relationship graph
- **Cost Tracker**: Live API usage and budget monitoring
- **Performance Metrics**: Game statistics and model comparison

### React Components

Add LLM visualization to the Secret Hitler web app:

```jsx
import { LLMOverlay } from './components/LLMOverlay';

// In your game component
<LLMOverlay 
  gameId={gameId}
  isVisible={showLLMOverlay}
  onToggle={() => setShowLLMOverlay(!showLLMOverlay)}
  gameState={gameState}
  players={players}
/>
```

## Batch Experiments

### Running Experiments

```bash
# Quick test experiment
python -m experiments.batch_runner --template quick_test

# Custom experiment
python -m experiments.batch_runner --config my_experiment.json

# Model comparison study
python -m experiments.batch_runner --template model_comparison
```

### Experiment Configuration

```json
{
  "name": "model_comparison",
  "description": "Compare GPT vs Claude vs Llama",
  "num_games": 50,
  "max_parallel": 5,
  "player_configs": [
    {"id": "p1", "name": "GPT", "model": "gpt-4o"},
    {"id": "p2", "name": "Claude", "model": "claude-3-sonnet"},
    {"id": "p3", "name": "Llama", "model": "llama-3-70b"}
  ],
  "total_cost_limit": 100.0,
  "enable_web_monitoring": true
}
```

### Analytics and Visualization

```python
from experiments.analytics import ExperimentAnalyzer

# Analyze results
analyzer = ExperimentAnalyzer("experiment_results/model_comparison_20240101")
analysis = analyzer.generate_analysis_report()

# Create visualizations
visualizations = analyzer.create_visualizations()

# Export data
analyzer.export_csv("model_comparison_data.csv")

# Compare multiple experiments
comparison = compare_experiments([
    "experiment_results/gpt_only",
    "experiment_results/claude_only", 
    "experiment_results/mixed_models"
])
```

## API Key Configuration

### OpenRouter (Recommended)

1. **Sign up**: Visit [OpenRouter.ai](https://openrouter.ai)
2. **Get API key**: Create API key and add credits
3. **Secure Configuration**:
   ```bash
   # Copy example environment file
   cp .env.example .env
   
   # Edit .env file with your real API key
   # NEVER commit this file to git!
   echo "OPENROUTER_API_KEY=sk-or-v1-your-actual-key-here" >> .env
   ```
   
**🔒 SECURITY BEST PRACTICES:**
- ✅ Store API keys in `.env` file (ignored by git)
- ✅ Use environment variables in code
- ❌ Never hardcode API keys in source code
- ❌ Never commit `.env` file to version control

### Available Models

```python
# Free models (for testing)
"deepseek/deepseek-v3.2-exp"        # Completely free

# Cheap models (for large experiments) 
"anthropic/claude-3-haiku"     # $0.00025/1k tokens
"openai/gpt-3.5-turbo"         # $0.0005/1k tokens

# Balanced models (recommended)
"anthropic/claude-3-sonnet"    # $0.003/1k tokens
"openai/gpt-4o-mini"           # $0.00015/1k tokens

# Premium models (for critical research)
"anthropic/claude-3-opus"      # $0.015/1k tokens
"openai/gpt-4o"                # $0.005/1k tokens
```

### Cost Management

```python
# Set in .env file
MAX_COST_PER_GAME=2.00
MAX_DAILY_COST=50.00
COST_ALERT_THRESHOLD=1.00

# Or in code
config = ExperimentConfig({
    "cost_limit_per_game": 1.0,
    "total_cost_limit": 25.0
})
```

### Troubleshooting API Issues

1. **401 Unauthorized**: Check API key and account credits
2. **429 Rate Limited**: Reduce `max_parallel` in experiments
3. **High costs**: Use free models for testing: `grok-4-fast-free`
4. **Model unavailable**: Check OpenRouter status page

## Advanced Features

### Deception Analysis

Automatic detection of AI lies:

```python
# Deception indicators tracked:
indicators = [
    "contradictory_policy_claims",
    "role_contradiction", 
    "trust_contradiction",
    "suspicion_hiding"
]

# Access in logs
deception_events = game_logger.metrics["deception_events"]
```

### Trust Network Evolution

Track how AI agents build trust:

```python
# Trust beliefs over time
trust_evolution = game_logger.metrics["trust_evolution"]

# Analyze trust patterns
for event in trust_evolution:
    player_id = event["player_id"]
    beliefs = event["beliefs"]  # {target_player: trust_level}
```

### Model Performance Analysis

Compare AI model capabilities:

```python
# Performance metrics per model
model_performance = {
    "actions_per_dollar": actions / cost,
    "reasoning_quality": reasoning_entries / total_actions,
    "deception_strategy": deception_count / total_actions,
    "trust_volatility": average_belief_change
}
```

## Roadmap

- [x] Multi-agent LLM integration
- [x] Cost-optimized model routing
- [x] Real-time web monitoring
- [x] Batch experiment runner
- [x] Advanced analytics pipeline
- [ ] Human-AI hybrid games
- [ ] Advanced deception analysis
- [ ] Tournament management system
- [ ] Cross-experiment comparison tools

## Contributing

This is part of a research framework for LLM evaluation. Contributions should focus on:

1. Improving evaluation metrics
2. Adding new models and providers
3. Enhancing behavioral analysis
4. Optimizing cost efficiency
5. Expanding documentation

## License

Part of the Secret Hitler LLM Evaluation Framework. See main repository for license details.
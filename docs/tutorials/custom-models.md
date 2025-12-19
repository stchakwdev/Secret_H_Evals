# Adding Custom Models

This guide covers how to add new LLM models to the evaluation framework.

## OpenRouter Integration

The framework uses OpenRouter for LLM access, supporting 100+ models from various providers.

### Available Models

Check available models at: https://openrouter.ai/models

### Using a New Model

Simply specify the model ID:

```bash
python run_game.py --model <provider>/<model-name>
```

Examples:

```bash
# Anthropic
python run_game.py --model anthropic/claude-3.5-sonnet
python run_game.py --model anthropic/claude-3-opus

# OpenAI
python run_game.py --model openai/gpt-4-turbo
python run_game.py --model openai/gpt-4o

# Meta
python run_game.py --model meta-llama/llama-3.1-70b-instruct
python run_game.py --model meta-llama/llama-3.1-405b-instruct

# Google
python run_game.py --model google/gemini-pro-1.5

# Mistral
python run_game.py --model mistralai/mistral-large
```

## Configuration

### Model-Specific Settings

Edit `config/openrouter_config.py`:

```python
MODEL_CONFIGS = {
    "deepseek/deepseek-v3.2-exp": {
        "max_tokens": 1024,
        "temperature": 0.7,
        "cost_per_1k_input": 0.0001,
        "cost_per_1k_output": 0.0002,
    },
    "anthropic/claude-3.5-sonnet": {
        "max_tokens": 1024,
        "temperature": 0.7,
        "cost_per_1k_input": 0.003,
        "cost_per_1k_output": 0.015,
    },
}
```

### Default Model

Set the default model in `config/openrouter_config.py`:

```python
DEFAULT_MODEL = "deepseek/deepseek-v3.2-exp"
```

## Cost Considerations

### Estimating Costs

Typical game costs (7 players, ~50 API calls per game):

| Model | Cost per Game |
|-------|---------------|
| deepseek-v3.2-exp | ~$0.02 |
| claude-3.5-sonnet | ~$0.50 |
| gpt-4-turbo | ~$0.40 |
| llama-3.1-70b | ~$0.15 |

### Cost Tracking

Enable cost logging:

```bash
python run_game.py --enable-db-logging
```

Query costs:

```sql
SELECT model, SUM(cost) as total_cost, COUNT(*) as games
FROM api_requests
GROUP BY model;
```

## Model Comparison Experiments

### Running Comparisons

```bash
# Create comparison script
for model in "deepseek/deepseek-v3.2-exp" "anthropic/claude-3.5-sonnet"; do
  python run_game.py --batch --games 50 --players 7 \
    --enable-db-logging \
    --batch-tag "comparison" \
    --model "$model"
done
```

### Analyzing Comparisons

```python
from analysis import compare_win_rates, calculate_elo_ratings

# Compare two models
result = compare_win_rates(games_df, 'model_a', 'model_b')

# Calculate Elo ratings across all models
elo_ratings = calculate_elo_ratings(games_df)
```

## Custom Model Integration

### For Non-OpenRouter Models

Extend `agents/openrouter_client.py`:

```python
class CustomModelClient(OpenRouterClient):
    def __init__(self, api_key: str, base_url: str):
        super().__init__(api_key)
        self.base_url = base_url

    async def get_decision(self, prompt: str) -> str:
        # Custom API call implementation
        pass
```

## Troubleshooting

### Model Not Available

Check OpenRouter status: https://openrouter.ai/status

### Rate Limits

Add delays between requests:

```python
# In config/openrouter_config.py
RATE_LIMIT_DELAY = 1.0  # seconds between requests
```

### Invalid Responses

Some models may not follow JSON format. The framework handles this with fallback parsing.

# Agents API

The agents module handles LLM integration for AI players.

## Overview

```
agents/
├── openrouter_client.py   # API client for OpenRouter
└── prompt_templates.py    # Game-specific prompts
```

## OpenRouterClient

HTTP client for OpenRouter API with retry logic and cost tracking.

### Initialization

```python
from agents.openrouter_client import OpenRouterClient

client = OpenRouterClient(
    api_key="your_api_key",
    model="deepseek/deepseek-v3.2-exp"
)
```

### Getting Decisions

```python
# Get a decision from the LLM
response = await client.get_decision(
    prompt=prompt_text,
    context=game_context,
    decision_type="nomination"
)

# Response structure
response.action      # The decided action
response.reasoning   # Private reasoning (not shared)
response.statement   # Public statement (shared with players)
response.confidence  # Confidence level (0-1)
```

### Supported Models

```python
# Check available models
models = client.get_available_models()

# Change model
client.set_model("anthropic/claude-3.5-sonnet")
```

### Cost Tracking

```python
# Get current session costs
costs = client.get_costs()
print(f"Total cost: ${costs['total']:.4f}")
print(f"Input tokens: {costs['input_tokens']}")
print(f"Output tokens: {costs['output_tokens']}")

# Reset cost tracking
client.reset_costs()
```

### Error Handling

```python
from agents.openrouter_client import OpenRouterError, RateLimitError

try:
    response = await client.get_decision(prompt)
except RateLimitError:
    # Wait and retry
    await asyncio.sleep(60)
    response = await client.get_decision(prompt)
except OpenRouterError as e:
    print(f"API error: {e}")
```

## Prompt Templates

Pre-defined prompts for different game situations.

### Template Types

| Template | Use Case |
|----------|----------|
| `nomination` | President nominating chancellor |
| `vote` | Player voting on government |
| `president_policy` | President discarding policy |
| `chancellor_policy` | Chancellor enacting policy |
| `investigation` | President investigating player |
| `execution` | President executing player |

### Getting Templates

```python
from agents.prompt_templates import get_prompt_template

template = get_prompt_template(
    template_type="nomination",
    game_state=state,
    player=current_player
)
```

### Template Structure

Each template includes:

1. **Game Context**: Current state, policies, players
2. **Role Information**: Player's role and knowledge
3. **Available Actions**: Valid choices for this decision
4. **Response Format**: Expected output structure

### Example Template Output

```
You are Alice, playing Secret Hitler.
Your role: FASCIST

Current game state:
- Liberal policies: 2
- Fascist policies: 3
- Players: Alice, Bob, Carol, David, Eve
- You know: Bob is Hitler, Carol is fellow fascist

As President, you must nominate a Chancellor.
Eligible players: Bob, Carol, David, Eve

Respond with:
REASONING: <your private reasoning>
ACTION: <player name to nominate>
STATEMENT: <what you say publicly>
```

### Custom Templates

```python
from agents.prompt_templates import PromptTemplate

custom_template = PromptTemplate(
    system_prompt="You are an expert Secret Hitler player...",
    user_prompt_template="""
    Game state: {game_state}
    Your role: {role}
    Decision: {decision_type}
    """,
    response_format={
        'reasoning': str,
        'action': str,
        'statement': str
    }
)

prompt = custom_template.render(
    game_state=state,
    role=player.role,
    decision_type="nomination"
)
```

## AIDecisionResponse

Structured response from LLM decisions.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `action` | str | The decided action |
| `reasoning` | str | Private reasoning |
| `statement` | str | Public statement |
| `confidence` | float | Confidence (0-1) |
| `raw_response` | str | Original LLM output |

### Parsing

```python
from agents.openrouter_client import parse_decision_response

response = parse_decision_response(raw_text)

# Access components
print(response.action)      # "nominate Bob"
print(response.reasoning)   # "Bob seems trustworthy..."
print(response.statement)   # "I nominate Bob because..."
```

## Configuration

### OpenRouter Config

Edit `config/openrouter_config.py`:

```python
# Default settings
DEFAULT_MODEL = "deepseek/deepseek-v3.2-exp"
MAX_TOKENS = 1024
TEMPERATURE = 0.7
TIMEOUT = 30  # seconds

# Rate limiting
REQUESTS_PER_MINUTE = 60
RETRY_ATTEMPTS = 3
RETRY_DELAY = 1.0  # seconds

# Cost tracking
COST_TRACKING_ENABLED = True
```

### Environment Variables

```bash
export OPENROUTER_API_KEY=your_key
export OPENROUTER_DEFAULT_MODEL=deepseek/deepseek-v3.2-exp
export OPENROUTER_TIMEOUT=30
```

## Example: Custom Agent

```python
import asyncio
from agents.openrouter_client import OpenRouterClient
from agents.prompt_templates import get_prompt_template

async def make_decision(client, state, player, decision_type):
    # Get appropriate prompt
    prompt = get_prompt_template(
        template_type=decision_type,
        game_state=state,
        player=player
    )

    # Get LLM decision
    response = await client.get_decision(
        prompt=prompt,
        decision_type=decision_type
    )

    # Log for analysis
    print(f"[{player.name}] {decision_type}")
    print(f"  Reasoning: {response.reasoning}")
    print(f"  Action: {response.action}")
    print(f"  Statement: {response.statement}")

    return response.action

# Usage
client = OpenRouterClient(api_key=os.environ['OPENROUTER_API_KEY'])
action = await make_decision(client, state, player, "nomination")
```

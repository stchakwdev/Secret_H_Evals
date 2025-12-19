# Game Engine API

The core game engine provides Secret Hitler game mechanics.

## Overview

```
core/
├── game_state.py     # Game state and policy management
├── game_manager.py   # Game flow orchestration
└── game_events.py    # Event logging and history
```

## GameState

Manages the game state including policies, players, and win conditions.

### Initialization

```python
from core.game_state import GameState

state = GameState(num_players=7)
```

### Key Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `players` | List[Player] | All players in the game |
| `liberal_policies` | int | Liberal policies enacted |
| `fascist_policies` | int | Fascist policies enacted |
| `election_tracker` | int | Failed elections (0-3) |
| `phase` | GamePhase | Current game phase |
| `president_index` | int | Current president |
| `chancellor_index` | int | Current chancellor |

### Methods

```python
# Check win conditions
winner = state.check_win_condition()
# Returns: 'liberal', 'fascist', or None

# Get current president/chancellor
president = state.get_president()
chancellor = state.get_chancellor()

# Enact a policy
state.enact_policy('liberal')
state.enact_policy('fascist')

# Advance election tracker
state.advance_election_tracker()
```

## GameManager

Orchestrates game flow and player interactions.

### Initialization

```python
from core.game_manager import GameManager

manager = GameManager(
    num_players=7,
    model="deepseek/deepseek-v3.2-exp",
    logger=game_logger
)
```

### Running a Game

```python
# Run complete game
result = await manager.run_game()

# Result contains:
# - winner: 'liberal' or 'fascist'
# - win_condition: 'policy' or 'hitler_elected' or 'hitler_killed'
# - turns: number of turns played
# - policies: list of enacted policies
```

### Game Phases

The game progresses through phases:

1. `NOMINATION` - President nominates chancellor
2. `ELECTION` - Players vote on government
3. `LEGISLATIVE` - President/Chancellor enact policy
4. `EXECUTIVE` - Executive action (if applicable)

### Callbacks

Register callbacks for game events:

```python
def on_policy_enacted(policy_type, president, chancellor):
    print(f"{president} and {chancellor} enacted {policy_type}")

manager.on_policy_enacted = on_policy_enacted
```

## GameEvent

Structured event logging.

### Event Types

| Event Type | Description |
|------------|-------------|
| `GAME_START` | Game initialized |
| `ROLE_ASSIGNMENT` | Roles assigned to players |
| `NOMINATION` | Chancellor nominated |
| `ELECTION` | Vote completed |
| `POLICY_ENACTED` | Policy placed on board |
| `EXECUTIVE_ACTION` | Special power used |
| `GAME_END` | Game completed |

### Creating Events

```python
from core.game_events import GameEvent

event = GameEvent(
    type='POLICY_ENACTED',
    turn=5,
    data={
        'policy': 'fascist',
        'president': 'Alice',
        'chancellor': 'Bob'
    }
)
```

### Event History

```python
# Get all events
events = manager.get_events()

# Filter by type
policy_events = [e for e in events if e.type == 'POLICY_ENACTED']

# Export to JSON
events_json = [e.to_dict() for e in events]
```

## Player

Represents a player in the game.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `name` | str | Player identifier |
| `role` | Role | LIBERAL, FASCIST, or HITLER |
| `is_alive` | bool | Whether player is alive |
| `is_investigated` | bool | Whether player has been investigated |

### Role Enum

```python
from core.game_state import Role

Role.LIBERAL   # Liberal team
Role.FASCIST   # Fascist team
Role.HITLER    # Hitler (fascist team leader)
```

## Example: Complete Game

```python
import asyncio
from core.game_manager import GameManager
from game_logging.game_logger import GameLogger

async def play_game():
    logger = GameLogger()
    manager = GameManager(
        num_players=7,
        model="deepseek/deepseek-v3.2-exp",
        logger=logger
    )

    result = await manager.run_game()

    print(f"Winner: {result['winner']}")
    print(f"Win condition: {result['win_condition']}")
    print(f"Turns: {result['turns']}")

    # Access game state
    state = manager.state
    print(f"Liberal policies: {state.liberal_policies}")
    print(f"Fascist policies: {state.fascist_policies}")

    # Access events
    for event in manager.get_events():
        print(f"[{event.turn}] {event.type}: {event.data}")

asyncio.run(play_game())
```

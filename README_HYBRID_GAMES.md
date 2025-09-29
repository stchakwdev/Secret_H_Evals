# Secret Hitler - Hybrid AI/Human Game System

This document explains how to set up and run hybrid Secret Hitler games where human players can play alongside AI agents powered by Large Language Models (LLMs).

## 🎯 Overview

The hybrid game system allows you to:
- Play Secret Hitler with a mix of AI and human players
- Human players use the web interface while AI players are managed by the LLM engine
- Real-time communication between the Python game engine and web clients
- Support for 5-10 players (any combination of humans and AIs)

## 🚀 Quick Start

### 1. Prerequisites

```bash
# Install Python dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your OPENROUTER_API_KEY
```

### 2. Launch a Quick Game

```bash
# Start a game with 1 human and 4 AI players
python hybrid_game_launcher.py --quick-start --humans 1 --ais 4

# Or use interactive setup
python hybrid_game_launcher.py --interactive
```

### 3. Connect as Human Player

1. The launcher will display WebSocket connection instructions
2. Open `hybrid_player_client.html` in your browser
3. Enter the Player ID and Game ID provided by the launcher
4. Click "Connect to Game Engine" then "Authenticate"

## 📁 Project Structure

```
llm-game-engine/
├── hybrid_game_launcher.py          # Main launcher script
├── hybrid_game_coordinator.py       # Coordinates mixed human-AI games
├── hybrid_player_client.html        # Web client for human players
├── core/
│   └── game_manager.py              # Enhanced with human player support
├── web_bridge/
│   ├── bidirectional_bridge.py     # WebSocket server for hybrid games
│   ├── hybrid_integration.py       # Integration layer
│   ├── websocket_server.py         # Base WebSocket server
│   ├── game_adapter.py             # Game monitoring adapter
│   └── event_converter.py          # Event format conversion
├── config/
│   └── hybrid_game_sample.json     # Sample configuration file
└── src/frontend-scripts/
    ├── hybrid-game-bridge.js       # JavaScript bridge for web UI
    └── hybrid-game-bridge.scss     # Styles for hybrid UI
```

## 🎮 Usage Modes

### 1. Quick Start Mode

Start a game quickly with default settings:

```bash
# 1 human + 4 AIs (5 players total)
python hybrid_game_launcher.py --quick-start --humans 1 --ais 4

# 2 humans + 3 AIs (5 players total)
python hybrid_game_launcher.py --quick-start --humans 2 --ais 3

# 3 humans + 7 AIs (10 players total)
python hybrid_game_launcher.py --quick-start --humans 3 --ais 7
```

### 2. Configuration File Mode

Create a custom configuration file:

```json
{
  "game_id": "my_hybrid_game",
  "host": "localhost",
  "port": 8765,
  "openrouter_api_key": null,
  "players": [
    {
      "id": "human_player",
      "name": "You",
      "type": "human"
    },
    {
      "id": "ai_alice",
      "name": "Alice",
      "type": "ai",
      "model": "deepseek/deepseek-v3.2-exp"
    }
  ]
}
```

Launch with:
```bash
python hybrid_game_launcher.py --config config/my_game.json
```

### 3. Interactive Mode

Set up a game interactively:

```bash
python hybrid_game_launcher.py --interactive
```

The script will guide you through:
- Choosing number of players
- Selecting human vs AI players
- Entering player names
- Launching the game

## 🌐 Human Player Connection

### Option 1: Direct HTML Client

1. Open `hybrid_player_client.html` in your browser
2. Enter connection details:
   - WebSocket URL: `ws://localhost:8765`
   - Player ID: As provided by the launcher
   - Game ID: As provided by the launcher
3. Click "Connect to Game Engine"
4. Click "Authenticate"

### Option 2: URL Parameters

You can pass connection details via URL:

```
hybrid_player_client.html?playerId=human_1&gameId=hybrid_game_20250123&wsUrl=ws://localhost:8765
```

### Option 3: Integration with Main Secret Hitler UI

To integrate with the main Secret Hitler web application:

1. Include the hybrid bridge JavaScript:
```html
<script src="src/frontend-scripts/hybrid-game-bridge.js"></script>
<link rel="stylesheet" href="src/scss/hybrid-game-bridge.scss">
```

2. Initialize the bridge:
```javascript
// Auto-connect for hybrid games
const urlParams = new URLSearchParams(window.location.search);
if (urlParams.get('hybrid') === 'true') {
    hybridBridge.connect();
    hybridBridge.authenticate(
        urlParams.get('playerId'),
        urlParams.get('gameId')
    );
}
```

## 🎯 Game Actions

Human players will receive action prompts for:

- **Role Acknowledgment**: Confirm your secret role
- **Chancellor Nomination**: Choose a chancellor (if you're president)
- **Government Voting**: Vote Ja or Nein on proposed governments
- **Policy Selection**: Choose policies as president or chancellor
- **Executive Powers**: Use investigation, special election, or execution powers

Each action includes:
- Clear instructions
- Game state context
- Form-based input
- Optional reasoning field

## 🔧 Configuration

### Environment Variables

Required in `.env`:
```bash
OPENROUTER_API_KEY=your_api_key_here
DEFAULT_MODEL=deepseek/deepseek-v3.2-exp
```

### Network Configuration

Default settings:
- WebSocket Host: `localhost`
- WebSocket Port: `8765`
- Human Action Timeout: `60` seconds

### AI Model Configuration

Configure AI models in player configs:
```json
{
  "id": "ai_player",
  "name": "AI Alice",
  "type": "ai",
  "model": "deepseek/deepseek-v3.2-exp"
}
```

Supported models (via OpenRouter):
- `deepseek/deepseek-v3.2-exp` (Free tier)
- `anthropic/claude-3-haiku`
- `openai/gpt-4o-mini`
- And many more...

## 🐛 Troubleshooting

### Connection Issues

**Problem**: Cannot connect to game engine
```
Solution: Ensure the hybrid game launcher is running and WebSocket server is started
Check: python hybrid_game_launcher.py is running without errors
```

**Problem**: Authentication failed
```
Solution: Verify Player ID and Game ID match those provided by the launcher
Check: Case-sensitive matching required
```

### Game Issues

**Problem**: Human player not receiving action prompts
```
Solution: Check browser console for JavaScript errors
Verify: WebSocket connection is maintained (check network tab)
```

**Problem**: Action submission fails
```
Solution: Ensure all required form fields are completed
Check: Network connectivity to game engine
```

### Performance Issues

**Problem**: Game runs slowly
```
Solution: Reduce number of AI players or use faster models
Check: OpenRouter API rate limits
```

## 📊 Monitoring and Logs

### Real-time Monitoring

The hybrid system provides real-time monitoring:
- Game state updates
- Player actions and reasoning
- Cost tracking for AI usage
- Error reporting

### Log Files

Generated logs:
```
logs/[game_id]/
├── game.log          # Main game events
├── public.log        # Public game information
├── Player_*.log      # Individual player logs
├── metrics.json      # Game statistics
└── api_usage.json    # API cost tracking
```

### Web Interface Monitoring

Connect to the monitoring interface:
```javascript
// Join as observer
websocket.send(JSON.stringify({
    type: "join_game",
    payload: { game_id: "your_game_id" }
}));
```

## 🔌 API Reference

### WebSocket Message Types

**Client to Server:**
```javascript
// Authenticate
{
    type: "authenticate_player",
    payload: {
        player_id: "human_1",
        game_id: "hybrid_game_123"
    }
}

// Submit action
{
    type: "submit_action",
    payload: {
        player_id: "human_1",
        action_type: "vote_on_government",
        action_data: { response: "ja", vote: true }
    }
}
```

**Server to Client:**
```javascript
// Action request
{
    type: "action_request",
    payload: {
        decision_type: "nominate_chancellor",
        prompt: "Choose a chancellor...",
        game_state: { ... },
        private_info: { ... }
    }
}

// Game event
{
    type: "game_event",
    payload: {
        event_type: "policy_enacted",
        data: { ... }
    }
}
```

## 🎭 Game Mechanics

### Secret Roles
- **Liberals**: Win by enacting 5 Liberal policies or assassinating Hitler
- **Fascists**: Win by enacting 6 Fascist policies or electing Hitler as Chancellor after 3 Fascist policies
- **Hitler**: Special fascist with unique win conditions

### Presidential Powers
Triggered by Fascist policies:
- **Investigation**: Learn a player's party membership
- **Special Election**: Choose the next presidential candidate
- **Execution**: Eliminate a player from the game

### Victory Conditions
- **Liberal Victory**: 5 Liberal policies OR Hitler assassination
- **Fascist Victory**: 6 Fascist policies OR Hitler elected Chancellor (with 3+ Fascist policies)

## 🧪 Testing

### Test the System

1. **Basic Connectivity**:
```bash
# Start server
python hybrid_game_launcher.py --quick-start --humans 1 --ais 1

# Test connection
python -c "
import asyncio
import websockets

async def test():
    async with websockets.connect('ws://localhost:8765') as ws:
        print('Connected successfully')

asyncio.run(test())
"
```

2. **Full Game Test**:
```bash
# Run automated test
python test_hybrid_game.py
```

### Development Testing

Use the test client:
```bash
# Open in browser
open hybrid_player_client.html?playerId=test_human&gameId=test_game
```

## 🤝 Contributing

To add new features:

1. **New Action Types**: Update `game_manager.py` and `hybrid-game-bridge.js`
2. **UI Improvements**: Modify `hybrid-game-bridge.scss` and HTML templates
3. **AI Models**: Add model configurations in `openrouter_config.py`

## 📝 License

This hybrid game extension follows the same license as the main Secret Hitler project: CC-BY-NC-SA-4.0

## 🆘 Support

For issues and questions:
1. Check this documentation
2. Review log files in `logs/[game_id]/`
3. Test with the simple client (`hybrid_player_client.html`)
4. Create an issue in the project repository

---

**Have fun playing Secret Hitler with AI companions! 🎉**
# 🎮 Secret Hitler - Simplified Hybrid Game System

**Play Secret Hitler with AI opponents in just one command!**

## ✨ Super Simple Usage

### Start a Game
```bash
./play                    # 1 human vs 4 AI (default)
./play --humans 2         # 2 humans vs 3 AI  
./play --ai-only          # Watch 5 AI players
./play --browser          # Use web interface
```

### Join a Game (from another terminal)
```bash
python auto_connect_client.py
```

That's it! No game IDs, no complex setup, no environment issues.

## 🎯 What's Different Now?

### Before (Complex)
1. Start server: `python hybrid_game_launcher.py --quick-start --humans 1 --ais 4`
2. Copy the game ID from logs
3. Update client with game ID  
4. Fix Python environment issues
5. Connect within 30 seconds or timeout
6. Manual WebSocket authentication

### Now (Simple)
1. `./play`
2. Done! 

## 🚀 Features

- **One Command Start** - Everything launches automatically
- **Auto-Discovery** - No more game IDs to manage
- **Smart Python Detection** - Finds the right Python automatically
- **Generous Timeouts** - Take your time (30+ minutes)
- **Auto-Connect Client** - Joins latest game automatically  
- **Cross-Platform** - Works on Mac/Linux (`./play`) and Windows (`play.bat`)
- **Preset Modes** - Built-in configurations

## 📋 Preset Modes

```bash
./play --mode solo         # 1 human, 4 AI
./play --mode team         # 2 humans, 3 AI
./play --mode spectate     # Watch AI-only game
```

## 🔧 Setup (One Time)

1. **Add your API key to `.env`:**
   ```
   OPENROUTER_API_KEY=sk-or-v1-your-key-here
   ```

2. **That's it!** The smart launcher handles Python and dependencies automatically.

## 🎮 How to Play

### Starting a Game
- Run `./play` 
- The server starts and waits for players
- Human players can join anytime (no rush!)

### Joining a Game  
- Run `python auto_connect_client.py` in another terminal
- Or open browser interface with `./play --browser`
- Auto-connects to the latest game

### During the Game
- You'll be prompted for votes and decisions
- Just type your response (e.g., "ja", "nein", player names)
- The terminal shows clear instructions

## 📁 File Overview

| File | Purpose |
|------|---------|
| `play` | Main launcher (Mac/Linux) |
| `play.bat` | Main launcher (Windows) |
| `play_hybrid.py` | Python game engine |
| `auto_connect_client.py` | Auto-connecting human client |
| `.env` | Your API key configuration |

## 🆘 Troubleshooting

### "No active games available"
Someone needs to start a game first: `./play`

### "Connection refused"  
Make sure a game server is running: `./play --ai-only`

### Python issues
The smart launcher detects Python automatically. If issues persist:
```bash
pip install -r requirements.txt
python play_hybrid.py
```

## 🎉 Examples

**Single player vs AI:**
```bash
./play
```

**Watch AI-only game:**
```bash
./play --ai-only
```

**Two humans vs AI:**
```bash
# Terminal 1:
./play --humans 2

# Terminal 2:
python auto_connect_client.py "Alice"

# Terminal 3:  
python auto_connect_client.py "Bob"
```

**Browser interface:**
```bash
./play --browser --humans 1
# Opens browser automatically
```

---

**The complex multi-step process is now just one command!** 🎯
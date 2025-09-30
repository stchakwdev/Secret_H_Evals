# WebSocket Connection Fix

## Problem
The spectator UI was not receiving game events from the WebSocket server because the client wasn't properly joining the game room.

## Root Cause
The WebSocket server requires clients to send a `join_game` message after connecting to enter a specific game room. Without this, the server broadcasts events to an empty room and the client receives nothing.

## Solution

### 1. Client-Side Changes ([SpectatorState.js](static/js/SpectatorState.js))

Added join_game handshake in the `onopen` handler:

```javascript
this.ws.onopen = () => {
    console.log('✅ Connected to spectator server');
    this.reconnectAttempts = 0;

    // Extract game_id from URL query params or use default
    const urlParams = new URLSearchParams(window.location.search);
    const gameId = urlParams.get('game_id') || 'current_game';

    // Send join_game message to enter game room
    console.log(`📩 Sending join_game request for: ${gameId}`);
    this.ws.send(JSON.stringify({
        type: "join_game",
        payload: { game_id: gameId }
    }));

    this.notifyStateChange();
};
```

Added handler for `joined_game` confirmation and `game_event` unwrapping:

```javascript
switch (eventType) {
    case 'joined_game':
        this.handleJoinedGame(event);
        break;

    case 'game_event':
        // Unwrap nested game_event and re-handle
        this.handleEvent(event.payload);
        return;

    // ... other cases
}
```

### 2. Server Integration ([run_spectator_game.py](run_spectator_game.py))

Updated spectator URL to include `game_id` parameter:

```python
print(f"\n🎭 Enhanced Spectator URL (Phases 1-7):")
print(f"   file://{spectator_path}?ws=ws://localhost:8765&game_id={game_manager.game_id}")
```

## Verification

Created simple test page that successfully:
1. ✅ Connects to `ws://localhost:8765`
2. ✅ Sends `join_game` message with game_id
3. ✅ Receives `joined_game` confirmation from server
4. ✅ Receives `game_event` broadcasts with real game data

## Testing

To test the WebSocket connection:

```bash
# Start the game with WebSocket server
python run_spectator_game.py

# Open the test page in browser
open http://localhost:8080/test_ws_direct.html
```

Expected output in test page:
```
✅ WebSocket opened
📤 Sending: {"type":"join_game","payload":{"game_id":"<uuid>"}}
📥 Received: {"type":"joined_game","payload":{"game_id":"<uuid>"}}
📥 Received: {"type":"game_event","payload":{...}}
```

## Next Steps

The WebSocket connection is now working correctly. The remaining issue with the full spectator UI (spectator_v2.html) appears to be a JavaScript initialization problem in SpectatorApp.js or one of its component dependencies, not a WebSocket connectivity issue.

To debug:
1. Check browser console for JavaScript errors
2. Verify all script dependencies load correctly
3. Ensure SpectatorApp initializes on DOMContentLoaded

## Files Modified

- `static/js/SpectatorState.js` - Added join_game handshake
- `run_spectator_game.py` - Updated URL to include game_id parameter
- `test_ws_direct.html` - Created for WebSocket verification (can be removed)
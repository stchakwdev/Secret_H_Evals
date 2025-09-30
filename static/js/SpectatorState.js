/**
 * SpectatorState.js - State management for Secret Hitler Spectator UI
 *
 * Manages game state, events, and WebSocket communication
 * Author: Samuel Chakwera (stchakdev)
 */

class SpectatorState {
    constructor() {
        this.gameId = null;
        this.players = new Map();
        this.events = [];
        this.reasoning = new Map(); // player_id -> [ReasoningEvent]
        this.speeches = new Map();  // player_id -> [SpeechEvent]
        this.gameState = {
            phase: 'setup',
            round: 0,
            liberalPolicies: 0,
            fascistPolicies: 0,
            electionTracker: 0,
            president: null,
            chancellor: null
        };

        // Callbacks for UI updates
        this.onStateChange = [];
        this.onNewReasoning = [];
        this.onNewSpeech = [];
        this.onDeceptionDetected = [];

        // WebSocket connection
        this.ws = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
    }

    /**
     * Connect to WebSocket server
     */
    connect(url = 'ws://localhost:8765') {
        this.ws = new WebSocket(url);

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

        this.ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                this.handleEvent(data);
            } catch (error) {
                console.error('Failed to parse WebSocket message:', error);
            }
        };

        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };

        this.ws.onclose = () => {
            console.warn('⚠️  WebSocket connection closed');
            this.attemptReconnect(url);
        };
    }

    /**
     * Attempt to reconnect to WebSocket
     */
    attemptReconnect(url) {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 10000);

            console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);

            setTimeout(() => {
                this.connect(url);
            }, delay);
        } else {
            console.error('❌ Max reconnection attempts reached');
        }
    }

    /**
     * Handle incoming event from WebSocket
     */
    handleEvent(event) {
        console.log('📥 Received event:', event.type || event.event_type);

        // Add to events history
        this.events.push(event);

        // Route to specific handler
        const eventType = event.type || event.event_type;

        switch (eventType) {
            case 'joined_game':
                this.handleJoinedGame(event);
                break;

            case 'game_event':
                // Unwrap nested game_event and re-handle
                this.handleEvent(event.payload);
                return;

            case 'game_start':
                this.handleGameStart(event);
                break;

            case 'reasoning':
                this.handleReasoning(event);
                break;

            case 'speech':
                this.handleSpeech(event);
                break;

            case 'phase_change':
                this.handlePhaseChange(event);
                break;

            case 'policy_enacted':
                this.handlePolicyEnacted(event);
                break;

            case 'game_over':
                this.handleGameOver(event);
                break;

            default:
                this.handleGenericEvent(event);
        }

        this.notifyStateChange();
    }

    /**
     * Handle joined_game confirmation
     */
    handleJoinedGame(event) {
        const joinedGameId = event.payload?.game_id;
        console.log(`✅ Successfully joined game room: ${joinedGameId}`);

        if (!this.gameId && joinedGameId) {
            this.gameId = joinedGameId;
        }
    }

    /**
     * Handle game start event
     */
    handleGameStart(event) {
        this.gameId = event.game_id || event.data?.game_id;

        // Initialize players
        const players = event.players || event.data?.players || [];
        players.forEach(player => {
            this.players.set(player.id, {
                id: player.id,
                name: player.name,
                type: player.type || 'ai',
                role: player.role || 'unknown',
                isAlive: true
            });

            this.reasoning.set(player.id, []);
            this.speeches.set(player.id, []);
        });
    }

    /**
     * Handle reasoning event (private AI thought)
     */
    handleReasoning(event) {
        const playerId = event.player_id;

        if (!this.reasoning.has(playerId)) {
            this.reasoning.set(playerId, []);
        }

        const reasoningData = {
            timestamp: event.timestamp || new Date().toISOString(),
            summary: event.summary,
            fullReasoning: event.full_reasoning || null,
            confidence: event.confidence || 0.5,
            beliefs: event.beliefs || {},
            strategy: event.strategy || null,
            decisionType: event.decision_type || 'unknown'
        };

        this.reasoning.get(playerId).push(reasoningData);

        // Notify callbacks
        this.onNewReasoning.forEach(callback => {
            callback(playerId, reasoningData);
        });
    }

    /**
     * Handle speech event (public statement)
     */
    handleSpeech(event) {
        const playerId = event.player_id;

        if (!this.speeches.has(playerId)) {
            this.speeches.set(playerId, []);
        }

        const speechData = {
            timestamp: event.timestamp || new Date().toISOString(),
            content: event.content,
            statementType: event.statement_type || 'statement',
            targetPlayer: event.target_player || null,
            isDeceptive: event.is_deceptive || false,
            deceptionScore: event.deception_score || 0.0,
            contradictionSummary: event.contradiction_summary || null,
            gameContext: event.game_context || ''
        };

        this.speeches.get(playerId).push(speechData);

        // Notify callbacks
        this.onNewSpeech.forEach(callback => {
            callback(playerId, speechData);
        });

        // If deceptive, notify deception callbacks
        if (speechData.isDeceptive && speechData.deceptionScore > 0.5) {
            this.onDeceptionDetected.forEach(callback => {
                callback(playerId, speechData);
            });
        }
    }

    /**
     * Handle phase change event
     */
    handlePhaseChange(event) {
        this.gameState.phase = event.new_phase || event.phase || event.data?.new_phase;
        this.gameState.round = event.round_number || this.gameState.round;
    }

    /**
     * Handle policy enacted event
     */
    handlePolicyEnacted(event) {
        const policyType = event.policy_type || event.data?.policy_type;

        if (policyType === 'liberal') {
            this.gameState.liberalPolicies++;
        } else if (policyType === 'fascist') {
            this.gameState.fascistPolicies++;
        }
    }

    /**
     * Handle game over event
     */
    handleGameOver(event) {
        this.gameState.phase = 'game_over';
        this.gameState.winner = event.winner || event.data?.winner;
        this.gameState.winCondition = event.win_condition || event.data?.win_condition;
    }

    /**
     * Handle generic event
     */
    handleGenericEvent(event) {
        // Just log for now
        console.log('Generic event:', event);
    }

    /**
     * Register callback for state changes
     */
    onUpdate(callback) {
        this.onStateChange.push(callback);
    }

    /**
     * Register callback for new reasoning
     */
    onReasoningEvent(callback) {
        this.onNewReasoning.push(callback);
    }

    /**
     * Register callback for new speech
     */
    onSpeechEvent(callback) {
        this.onNewSpeech.push(callback);
    }

    /**
     * Register callback for deception detection
     */
    onDeception(callback) {
        this.onDeceptionDetected.push(callback);
    }

    /**
     * Notify all state change listeners
     */
    notifyStateChange() {
        this.onStateChange.forEach(callback => {
            callback(this);
        });
    }

    /**
     * Get player by ID
     */
    getPlayer(playerId) {
        return this.players.get(playerId);
    }

    /**
     * Get all players
     */
    getAllPlayers() {
        return Array.from(this.players.values());
    }

    /**
     * Get reasoning history for player
     */
    getPlayerReasoning(playerId) {
        return this.reasoning.get(playerId) || [];
    }

    /**
     * Get speech history for player
     */
    getPlayerSpeeches(playerId) {
        return this.speeches.get(playerId) || [];
    }

    /**
     * Get latest reasoning for player
     */
    getLatestReasoning(playerId) {
        const history = this.reasoning.get(playerId) || [];
        return history.length > 0 ? history[history.length - 1] : null;
    }

    /**
     * Get latest speech for player
     */
    getLatestSpeech(playerId) {
        const history = this.speeches.get(playerId) || [];
        return history.length > 0 ? history[history.length - 1] : null;
    }

    /**
     * Get deception statistics
     */
    getDeceptionStats() {
        const stats = {
            totalSpeeches: 0,
            deceptiveSpeeches: 0,
            byPlayer: {}
        };

        this.speeches.forEach((speeches, playerId) => {
            const playerName = this.getPlayer(playerId)?.name || playerId;
            const deceptiveCount = speeches.filter(s => s.isDeceptive && s.deceptionScore > 0.5).length;

            stats.totalSpeeches += speeches.length;
            stats.deceptiveSpeeches += deceptiveCount;
            stats.byPlayer[playerName] = {
                total: speeches.length,
                deceptive: deceptiveCount,
                deceptionRate: speeches.length > 0 ? (deceptiveCount / speeches.length) : 0
            };
        });

        return stats;
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = SpectatorState;
}
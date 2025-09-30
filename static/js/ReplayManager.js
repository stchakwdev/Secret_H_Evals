/**
 * ReplayManager.js - Timeline and replay control system
 * Phase 5: Timeline & Replay
 *
 * Manages event storage, playback, and state reconstruction for replays
 * Author: Samuel Chakwera (stchakdev)
 */

class ReplayManager {
    /**
     * Create a new ReplayManager
     * @param {Object} initialState - Initial game state
     */
    constructor(initialState = null) {
        this.events = [];
        this.currentEventIndex = -1;
        this.isPlaying = false;
        this.playbackSpeed = 1.0;
        this.playbackInterval = null;
        this.initialState = initialState || this.createEmptyState();
        this.currentState = { ...this.initialState };
        this.stateCache = new Map(); // Cache computed states for performance
        this.keyMoments = [];
        this.listeners = {
            stateChange: [],
            playbackChange: [],
            indexChange: []
        };
    }

    /**
     * Create empty initial state
     * @returns {Object} Empty game state
     */
    createEmptyState() {
        return {
            gameId: null,
            phase: 'setup',
            round: 0,
            liberalPolicies: 0,
            fascistPolicies: 0,
            failedVotes: 0,
            president: null,
            chancellor: null,
            players: {}
        };
    }

    /**
     * Add event to timeline
     * @param {Object} event - Game event to add
     */
    addEvent(event) {
        this.events.push(event);

        // If in live mode (at end of timeline), update current state
        if (this.currentEventIndex === this.events.length - 2 || this.currentEventIndex === -1) {
            this.currentEventIndex = this.events.length - 1;
            this.currentState = this.applyEvent(this.currentState, event);
            this.notifyListeners('stateChange', this.currentState);
            this.notifyListeners('indexChange', this.currentEventIndex);
        }

        // Update key moments when new events added
        this.updateKeyMoments();
    }

    /**
     * Set all events at once (for loading saved games)
     * @param {Array} events - Array of game events
     */
    setEvents(events) {
        this.events = events;
        this.stateCache.clear();
        this.updateKeyMoments();

        // Move to end of timeline
        if (events.length > 0) {
            this.seekToIndex(events.length - 1);
        }
    }

    /**
     * Apply single event to state
     * @param {Object} state - Current state
     * @param {Object} event - Event to apply
     * @returns {Object} New state
     */
    applyEvent(state, event) {
        const newState = JSON.parse(JSON.stringify(state)); // Deep clone

        switch (event.event_type || event.type) {
            case 'game_start':
                newState.gameId = event.data?.game_id || event.gameId;
                newState.phase = 'nomination';
                if (event.data?.players) {
                    event.data.players.forEach(player => {
                        newState.players[player.id] = {
                            id: player.id,
                            name: player.name,
                            role: player.role,
                            isAlive: true
                        };
                    });
                }
                break;

            case 'phase_change':
                newState.phase = event.data?.new_phase || event.phase;
                break;

            case 'nomination':
                newState.chancellor = event.data?.chancellor || event.chancellor;
                newState.president = event.data?.president || event.president;
                break;

            case 'policy_enacted':
                const policyType = event.data?.policy_type || event.policyType;
                if (policyType === 'liberal') {
                    newState.liberalPolicies++;
                } else if (policyType === 'fascist') {
                    newState.fascistPolicies++;
                }
                newState.failedVotes = 0; // Reset on successful policy
                break;

            case 'vote_result':
                const result = event.data?.result || event.result;
                if (result === 'failed' || result === 'rejected') {
                    newState.failedVotes++;
                }
                break;

            case 'execution':
                const targetId = event.data?.target || event.target;
                if (newState.players[targetId]) {
                    newState.players[targetId].isAlive = false;
                }
                break;

            case 'round_start':
                newState.round = event.data?.round_number || event.round || newState.round + 1;
                break;

            case 'game_over':
                newState.phase = 'game_over';
                newState.winner = event.data?.winner || event.winner;
                break;
        }

        return newState;
    }

    /**
     * Reconstruct state at specific event index
     * @param {number} targetIndex - Target event index
     * @returns {Object} Game state at that point
     */
    reconstructState(targetIndex) {
        // Check cache first
        if (this.stateCache.has(targetIndex)) {
            return this.stateCache.get(targetIndex);
        }

        // Find nearest cached state before target
        let startIndex = -1;
        let startState = this.initialState;

        for (let i = targetIndex - 1; i >= 0; i--) {
            if (this.stateCache.has(i)) {
                startIndex = i;
                startState = this.stateCache.get(i);
                break;
            }
        }

        // Apply events from start to target
        let state = JSON.parse(JSON.stringify(startState));
        for (let i = startIndex + 1; i <= targetIndex; i++) {
            state = this.applyEvent(state, this.events[i]);
        }

        // Cache result (cache every 10th state to balance memory/performance)
        if (targetIndex % 10 === 0) {
            this.stateCache.set(targetIndex, JSON.parse(JSON.stringify(state)));
        }

        return state;
    }

    /**
     * Seek to specific event index
     * @param {number} index - Target index
     */
    seekToIndex(index) {
        if (index < 0 || index >= this.events.length) {
            return;
        }

        this.currentEventIndex = index;
        this.currentState = this.reconstructState(index);

        this.notifyListeners('stateChange', this.currentState);
        this.notifyListeners('indexChange', this.currentEventIndex);
    }

    /**
     * Step forward one event
     */
    stepForward() {
        if (this.currentEventIndex < this.events.length - 1) {
            this.seekToIndex(this.currentEventIndex + 1);
        }
    }

    /**
     * Step backward one event
     */
    stepBackward() {
        if (this.currentEventIndex > 0) {
            this.seekToIndex(this.currentEventIndex - 1);
        }
    }

    /**
     * Jump to start of timeline
     */
    jumpToStart() {
        if (this.events.length > 0) {
            this.seekToIndex(0);
        }
    }

    /**
     * Jump to end of timeline
     */
    jumpToEnd() {
        if (this.events.length > 0) {
            this.seekToIndex(this.events.length - 1);
        }
    }

    /**
     * Toggle play/pause
     */
    togglePlay() {
        if (this.isPlaying) {
            this.pause();
        } else {
            this.play();
        }
    }

    /**
     * Start playback
     */
    play() {
        if (this.isPlaying) return;

        // If at end, restart from beginning
        if (this.currentEventIndex >= this.events.length - 1) {
            this.jumpToStart();
        }

        this.isPlaying = true;
        this.notifyListeners('playbackChange', { isPlaying: true, speed: this.playbackSpeed });

        const intervalMs = 1000 / this.playbackSpeed; // Base interval 1 second

        this.playbackInterval = setInterval(() => {
            if (this.currentEventIndex < this.events.length - 1) {
                this.stepForward();
            } else {
                // Reached end
                this.pause();
            }
        }, intervalMs);
    }

    /**
     * Pause playback
     */
    pause() {
        if (!this.isPlaying) return;

        this.isPlaying = false;
        if (this.playbackInterval) {
            clearInterval(this.playbackInterval);
            this.playbackInterval = null;
        }

        this.notifyListeners('playbackChange', { isPlaying: false, speed: this.playbackSpeed });
    }

    /**
     * Set playback speed
     * @param {number} speed - Playback speed multiplier (0.5, 1, 2, 5)
     */
    setSpeed(speed) {
        this.playbackSpeed = speed;

        // Restart playback if playing
        if (this.isPlaying) {
            this.pause();
            this.play();
        }

        this.notifyListeners('playbackChange', { isPlaying: this.isPlaying, speed: this.playbackSpeed });
    }

    /**
     * Detect key moments in the timeline
     */
    updateKeyMoments() {
        const moments = [];

        this.events.forEach((event, index) => {
            const eventType = event.event_type || event.type;

            // Policy enactments
            if (eventType === 'policy_enacted') {
                moments.push({
                    index,
                    type: 'policy',
                    severity: 'medium',
                    description: `${event.data?.policy_type || 'Unknown'} policy enacted`,
                    icon: event.data?.policy_type === 'liberal' ? '🔵' : '🔴'
                });
            }

            // Deception detected
            if (eventType === 'speech' && (event.isDeceptive || event.data?.isDeceptive)) {
                const score = event.deceptionScore || event.data?.deceptionScore || 0;
                moments.push({
                    index,
                    type: 'deception',
                    severity: score > 0.7 ? 'high' : score > 0.5 ? 'medium' : 'low',
                    description: `${event.player_name || 'Player'} made deceptive statement`,
                    icon: '🎭'
                });
            }

            // Executions
            if (eventType === 'execution') {
                moments.push({
                    index,
                    type: 'execution',
                    severity: 'high',
                    description: `${event.data?.target || 'Player'} executed`,
                    icon: '💀'
                });
            }

            // Investigations
            if (eventType === 'investigation') {
                moments.push({
                    index,
                    type: 'investigation',
                    severity: 'medium',
                    description: `${event.data?.target || 'Player'} investigated`,
                    icon: '🔍'
                });
            }

            // Critical votes (close results)
            if (eventType === 'vote_result') {
                const ja = event.data?.ja_votes || 0;
                const nein = event.data?.nein_votes || 0;
                if (Math.abs(ja - nein) <= 1) {
                    moments.push({
                        index,
                        type: 'critical_vote',
                        severity: 'medium',
                        description: `Close vote: ${ja}-${nein}`,
                        icon: '🗳️'
                    });
                }
            }

            // Game start
            if (eventType === 'game_start') {
                moments.push({
                    index,
                    type: 'system',
                    severity: 'low',
                    description: 'Game started',
                    icon: '🎮'
                });
            }

            // Game over
            if (eventType === 'game_over') {
                moments.push({
                    index,
                    type: 'system',
                    severity: 'high',
                    description: `${event.data?.winner || 'Unknown'} wins!`,
                    icon: '🏆'
                });
            }
        });

        this.keyMoments = moments;
    }

    /**
     * Get key moments for timeline display
     * @returns {Array} Array of key moments
     */
    getKeyMoments() {
        return this.keyMoments;
    }

    /**
     * Register event listener
     * @param {string} eventType - Type of event (stateChange, playbackChange, indexChange)
     * @param {Function} callback - Callback function
     */
    on(eventType, callback) {
        if (this.listeners[eventType]) {
            this.listeners[eventType].push(callback);
        }
    }

    /**
     * Notify listeners of event
     * @param {string} eventType - Type of event
     * @param {*} data - Event data
     */
    notifyListeners(eventType, data) {
        if (this.listeners[eventType]) {
            this.listeners[eventType].forEach(callback => {
                try {
                    callback(data);
                } catch (error) {
                    console.error(`Error in ${eventType} listener:`, error);
                }
            });
        }
    }

    /**
     * Get current playback state
     * @returns {Object} Playback state
     */
    getPlaybackState() {
        return {
            isPlaying: this.isPlaying,
            speed: this.playbackSpeed,
            currentIndex: this.currentEventIndex,
            totalEvents: this.events.length,
            currentState: this.currentState
        };
    }

    /**
     * Clear all events and reset
     */
    reset() {
        this.pause();
        this.events = [];
        this.currentEventIndex = -1;
        this.currentState = { ...this.initialState };
        this.stateCache.clear();
        this.keyMoments = [];
        this.notifyListeners('stateChange', this.currentState);
        this.notifyListeners('indexChange', this.currentEventIndex);
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ReplayManager;
}
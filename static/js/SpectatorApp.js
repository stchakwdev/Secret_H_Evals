/**
 * SpectatorApp - Main application controller integrating all phases (1-6)
 *
 * Integrates:
 * - Phase 1-2: Base spectator UI with game state
 * - Phase 3: Speech and thought bubbles with deception detection
 * - Phase 5: Timeline and replay controls
 * - Phase 6: Trust network and analytics
 * - Phase 7: Performance optimizations and polish
 */

class SpectatorApp {
    constructor() {
        // Core components
        this.spectatorState = new SpectatorState();
        this.replayManager = null;
        this.timeline = null;
        this.trustNetwork = null;
        this.analytics = new Analytics();

        // State tracking
        this.isLiveMode = true;
        this.trustNetworkFrozen = false;
        this.allEvents = [];
        this.eventThrottle = null;
        this.hasShownReasoningEmpty = false;
        this.hasShownSpeechEmpty = false;

        // DOM elements
        this.initDOMElements();

        // Initialize components
        this.initComponents();

        // Register callbacks
        this.registerCallbacks();

        // Show loading
        this.showLoading();
    }

    initDOMElements() {
        this.dom = {
            // Header
            statusIndicator: document.getElementById('statusIndicator'),
            statusText: document.getElementById('statusText'),
            phaseIndicator: document.getElementById('phaseIndicator'),
            header: document.getElementById('spectator-header'),

            // Game state
            gameId: document.getElementById('gameId'),
            gamePhase: document.getElementById('gamePhase'),
            gameRound: document.getElementById('gameRound'),
            liberalPolicies: document.getElementById('liberalPolicies'),
            fascistPolicies: document.getElementById('fascistPolicies'),
            liberalCount: document.getElementById('liberalCount'),
            fascistCount: document.getElementById('fascistCount'),
            playersList: document.getElementById('playersList'),

            // Bubbles
            reasoningContainer: document.getElementById('reasoning-container'),
            speechContainer: document.getElementById('speech-container'),

            // Analytics
            summaryStats: document.getElementById('summaryStats'),
            deceptionList: document.getElementById('deceptionList'),
            insightList: document.getElementById('insightList'),
            insightCount: document.getElementById('insightCount'),

            // Timeline
            jumpToStart: document.getElementById('jumpToStart'),
            stepBackward: document.getElementById('stepBackward'),
            togglePlay: document.getElementById('togglePlay'),
            stepForward: document.getElementById('stepForward'),
            jumpToEnd: document.getElementById('jumpToEnd'),
            speedControl: document.getElementById('speedControl'),
            timelineSlider: document.getElementById('timelineSlider'),
            timelineMarkers: document.getElementById('timelineMarkers'),
            timelineProgress: document.getElementById('timelineProgress'),
            timelineInfo: document.getElementById('timelineInfo'),
            timelineRound: document.getElementById('timelineRound'),

            // Trust network
            trustNetworkViz: document.getElementById('trustNetworkViz'),
            trustLiveBtn: document.getElementById('trustLiveBtn'),
            trustFrozenBtn: document.getElementById('trustFrozenBtn'),

            // Loading
            loadingOverlay: document.getElementById('loadingOverlay')
        };
    }

    initComponents() {
        // Initialize trust network
        this.trustNetwork = new TrustNetwork(this.dom.trustNetworkViz, {
            width: 280,
            height: 250,
            nodeRadius: 18,
            minTrustThreshold: 0.2,
            onNodeClick: (node) => this.handleNodeClick(node)
        });

        // Initialize replay manager
        this.replayManager = new ReplayManager();

        // Initialize timeline
        this.timeline = new Timeline(this.replayManager, document.getElementById('timeline-panel'));

        // Trust network controls
        this.dom.trustLiveBtn?.addEventListener('click', () => this.setTrustMode(true));
        this.dom.trustFrozenBtn?.addEventListener('click', () => this.setTrustMode(false));

        // Show empty states
        this.dom.reasoningContainer.appendChild(ThoughtBubble.createEmptyState());
        this.dom.speechContainer.appendChild(SpeechBubble.createEmptyState());
    }

    registerCallbacks() {
        // State updates
        this.spectatorState.onUpdate(() => this.updateGameState());

        // Reasoning events
        this.spectatorState.onReasoningEvent((playerId, reasoning) => {
            this.handleReasoningEvent(playerId, reasoning);
        });

        // Speech events
        this.spectatorState.onSpeechEvent((playerId, speech) => {
            this.handleSpeechEvent(playerId, speech);
        });

        // Deception events
        this.spectatorState.onDeception((playerId, speech) => {
            this.handleDeceptionEvent(playerId, speech);
        });

        // Replay events
        this.replayManager.addEventListener('statechange', () => {
            this.handleReplayStateChange();
        });

        this.replayManager.addEventListener('playbackchange', (e) => {
            this.updatePlaybackControls(e.detail.isPlaying);
        });
    }

    showLoading() {
        if (this.dom.loadingOverlay) {
            this.dom.loadingOverlay.style.display = 'flex';
        }
    }

    hideLoading() {
        if (this.dom.loadingOverlay) {
            this.dom.loadingOverlay.style.display = 'none';
        }
    }

    showError(message) {
        const banner = document.createElement('div');
        banner.className = 'error-banner';
        banner.textContent = message;
        document.body.appendChild(banner);

        setTimeout(() => {
            banner.remove();
        }, 5000);
    }

    handleReasoningEvent(playerId, reasoning) {
        // Remove empty state on first reasoning
        if (!this.hasShownReasoningEmpty) {
            this.dom.reasoningContainer.innerHTML = '';
            this.hasShownReasoningEmpty = true;
        }

        const player = this.spectatorState.getPlayer(playerId);
        if (player) {
            const bubble = ThoughtBubble.create(reasoning, player);
            this.dom.reasoningContainer.appendChild(bubble);

            // Auto-scroll with throttling
            this.throttledScroll(bubble);

            // Add to replay events
            this.addReplayEvent({
                type: 'reasoning',
                player_id: playerId,
                player_name: player.name,
                timestamp: Date.now(),
                round: this.spectatorState.gameState.round || 1,
                content: reasoning.summary || reasoning.reasoning,
                beliefs: reasoning.beliefs,
                confidence: reasoning.confidence,
                data: reasoning
            });
        }
    }

    handleSpeechEvent(playerId, speech) {
        // Remove empty state on first speech
        if (!this.hasShownSpeechEmpty) {
            this.dom.speechContainer.innerHTML = '';
            this.hasShownSpeechEmpty = true;
        }

        const player = this.spectatorState.getPlayer(playerId);
        if (player) {
            const bubble = SpeechBubble.create(speech, player);
            this.dom.speechContainer.appendChild(bubble);

            // Auto-scroll with throttling
            this.throttledScroll(bubble);

            // Add to replay events
            this.addReplayEvent({
                type: 'speech',
                player_id: playerId,
                player_name: player.name,
                timestamp: Date.now(),
                round: this.spectatorState.gameState.round || 1,
                content: speech.statement || speech.content,
                isDeceptive: speech.isDeceptive,
                deceptionScore: speech.deceptionScore,
                data: speech
            });

            // Update analytics
            this.updateAnalytics();
        }
    }

    handleDeceptionEvent(playerId, speech) {
        if (speech.deceptionScore > 0.7) {
            console.log(`🎭 High-confidence deception detected from ${playerId}`);
        }
    }

    addReplayEvent(event) {
        this.allEvents.push(event);
        this.replayManager.addEvent(event);

        // Update timeline slider
        if (this.dom.timelineSlider) {
            this.dom.timelineSlider.max = Math.max(0, this.allEvents.length - 1);
            if (this.isLiveMode) {
                this.dom.timelineSlider.value = this.allEvents.length - 1;
            }
        }

        // Update timeline info
        this.updateTimelineInfo();
    }

    handleReplayStateChange() {
        if (!this.isLiveMode) {
            // Reconstruct UI from replay state
            const currentIndex = this.replayManager.currentEventIndex;
            const state = this.replayManager.getCurrentState();

            // Update timeline info
            this.updateTimelineInfo();

            // TODO: Reconstruct bubbles and game state from replay
            console.log('Replay state changed:', currentIndex, state);
        }
    }

    updatePlaybackControls(isPlaying) {
        if (this.dom.togglePlay) {
            this.dom.togglePlay.textContent = isPlaying ? '⏸' : '▶';
            if (isPlaying) {
                this.dom.togglePlay.classList.add('playing');
            } else {
                this.dom.togglePlay.classList.remove('playing');
            }
        }
    }

    updateTimelineInfo() {
        const total = this.allEvents.length;
        const current = this.replayManager.currentEventIndex + 1;
        const round = this.spectatorState.gameState.round || 0;

        if (this.dom.timelineInfo) {
            this.dom.timelineInfo.textContent = `Event ${Math.max(0, current)} / ${total}`;
        }
        if (this.dom.timelineRound) {
            this.dom.timelineRound.textContent = `Round ${round}`;
        }
    }

    throttledScroll(element) {
        if (this.eventThrottle) {
            clearTimeout(this.eventThrottle);
        }
        this.eventThrottle = setTimeout(() => {
            element.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }, 100);
    }

    updateGameState() {
        const state = this.spectatorState.gameState;

        // Game info
        if (this.spectatorState.gameId && this.dom.gameId) {
            this.dom.gameId.textContent = this.spectatorState.gameId.substring(0, 8);
        }

        // Update phase with color coding
        const phase = state.phase || '-';
        if (this.dom.gamePhase) this.dom.gamePhase.textContent = phase;
        if (this.dom.gameRound) this.dom.gameRound.textContent = state.round || '0';

        // Update header phase
        const phaseMap = {
            'nomination': { class: 'phase-nomination', text: '👥 Nomination Phase', color: '#667eea' },
            'voting': { class: 'phase-voting', text: '🗳️ Voting Phase', color: '#ffd93d' },
            'legislative': { class: 'phase-legislative', text: '📜 Legislative Phase', color: '#f59e0b' },
            'executive': { class: 'phase-executive', text: '⚡ Executive Action', color: '#ef4444' },
            'gameover': { class: 'phase-gameover', text: '🏁 Game Over', color: '#10b981' }
        };

        const currentPhase = phaseMap[phase.toLowerCase()] || phaseMap['nomination'];

        if (this.dom.header) {
            this.dom.header.className = currentPhase.class;
        }
        if (this.dom.phaseIndicator) {
            this.dom.phaseIndicator.textContent = `[${currentPhase.text}]`;
            this.dom.phaseIndicator.style.color = currentPhase.color;
        }

        // Policy tracks
        this.updatePolicyTrack(this.dom.liberalPolicies, state.liberalPolicies, 5, 'liberal');
        this.updatePolicyTrack(this.dom.fascistPolicies, state.fascistPolicies, 6, 'fascist');

        // Players list
        this.updatePlayersList();

        // Connection status
        if (this.spectatorState.ws && this.spectatorState.ws.readyState === WebSocket.OPEN) {
            this.dom.statusIndicator?.classList.remove('disconnected');
            if (this.dom.statusText) this.dom.statusText.textContent = 'Connected';
            this.hideLoading();
        } else {
            this.dom.statusIndicator?.classList.add('disconnected');
            if (this.dom.statusText) this.dom.statusText.textContent = 'Disconnected';
        }

        // Update analytics if in live mode
        if (this.isLiveMode && !this.trustNetworkFrozen) {
            this.updateAnalytics();
        }
    }

    updatePolicyTrack(container, enacted, total, type) {
        if (!container) return;

        container.innerHTML = '';
        for (let i = 0; i < total; i++) {
            const dot = document.createElement('div');
            dot.className = 'policy-dot';
            if (i < enacted) {
                dot.classList.add('enacted', type);
            }
            container.appendChild(dot);
        }

        // Update progress counter
        const countId = type === 'liberal' ? 'liberalCount' : 'fascistCount';
        const countEl = document.getElementById(countId);
        if (countEl) {
            countEl.textContent = `${enacted} / ${total}`;
        }
    }

    updatePlayersList() {
        if (!this.dom.playersList) return;

        this.dom.playersList.innerHTML = '';
        const players = this.spectatorState.getAllPlayers();

        players.forEach(player => {
            const card = document.createElement('div');
            card.className = 'player-card';
            if (!player.isAlive) {
                card.classList.add('dead');
            }

            const name = document.createElement('div');
            name.className = 'player-name';
            name.textContent = player.name;

            const role = document.createElement('div');
            role.className = `player-role ${player.role ? player.role.toLowerCase() : ''}`;
            role.textContent = player.role || 'Unknown';

            card.appendChild(name);
            card.appendChild(role);
            this.dom.playersList.appendChild(card);
        });
    }

    updateAnalytics() {
        const players = this.spectatorState.getAllPlayers();

        // Process events with analytics engine
        this.analytics.processEvents(this.allEvents, players);

        // Update trust network
        if (!this.trustNetworkFrozen) {
            const trustData = this.analytics.computeTrustNetwork();
            this.trustNetwork.update(players, trustData);
        }

        // Update summary stats
        const stats = this.analytics.getSummaryStats();
        if (this.dom.summaryStats) {
            this.dom.summaryStats.innerHTML = `
                <div class="stat-item">
                    <div class="stat-value">${stats.totalEvents}</div>
                    <div class="stat-label">Total Events</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value warning">${stats.deceptiveEvents}</div>
                    <div class="stat-label">Deceptions</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value positive">${stats.liberalPolicies}</div>
                    <div class="stat-label">Liberal Policies</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value negative">${stats.fascistPolicies}</div>
                    <div class="stat-label">Fascist Policies</div>
                </div>
            `;
        }

        // Update deception list
        this.updateDeceptionList();

        // Update insights
        this.updateInsights();
    }

    updateDeceptionList() {
        if (!this.dom.deceptionList) return;

        const deceptionScores = Object.entries(this.analytics.deceptionScores)
            .filter(([_, data]) => data.totalDeceptions > 0)
            .sort((a, b) => b[1].totalDeceptions - a[1].totalDeceptions);

        if (deceptionScores.length === 0) {
            this.dom.deceptionList.innerHTML = '<div style="color: rgba(255,255,255,0.5); text-align: center; padding: 20px;">No deceptions detected yet</div>';
            return;
        }

        const players = this.spectatorState.getAllPlayers();
        const html = deceptionScores.map(([playerId, data]) => {
            const player = players.find(p => p.id === playerId);
            if (!player) return '';

            const percentage = Math.min(data.averageSeverity * 100, 100);
            return `
                <div class="deception-item">
                    <div class="deception-player">${player.name}</div>
                    <div class="deception-bar-container">
                        <div class="deception-bar" style="width: ${percentage}%"></div>
                    </div>
                    <div class="deception-count">${data.totalDeceptions}×</div>
                </div>
            `;
        }).join('');

        this.dom.deceptionList.innerHTML = html;
    }

    updateInsights() {
        if (!this.dom.insightList || !this.dom.insightCount) return;

        const insights = this.analytics.insights;
        this.dom.insightCount.textContent = insights.length;

        if (insights.length === 0) {
            this.dom.insightList.innerHTML = `
                <div class="analytics-empty">
                    <div class="analytics-empty-icon">💡</div>
                    <div class="analytics-empty-text">Insights will appear as the game progresses</div>
                </div>
            `;
            return;
        }

        const html = insights.map(insight => {
            const confidencePercent = (insight.confidence * 100).toFixed(0);
            return `
                <div class="insight-item ${insight.severity}">
                    <div class="insight-icon">${insight.icon}</div>
                    <div class="insight-content">
                        <div class="insight-text">${insight.text}</div>
                        <div class="insight-meta">
                            <div class="insight-confidence">
                                Confidence:
                                <div class="confidence-bar">
                                    <div class="confidence-fill" style="width: ${confidencePercent}%"></div>
                                </div>
                                ${confidencePercent}%
                            </div>
                            <div class="insight-type">${insight.type}</div>
                        </div>
                    </div>
                </div>
            `;
        }).join('');

        this.dom.insightList.innerHTML = html;
    }

    handleNodeClick(node) {
        console.log('Trust network node clicked:', node);
        this.trustNetwork.highlightPlayer(node.id);
    }

    setTrustMode(isLive) {
        this.trustNetworkFrozen = !isLive;

        if (this.dom.trustLiveBtn && this.dom.trustFrozenBtn) {
            if (isLive) {
                this.dom.trustLiveBtn.classList.add('active');
                this.dom.trustFrozenBtn.classList.remove('active');
                // Update immediately
                this.updateAnalytics();
            } else {
                this.dom.trustLiveBtn.classList.remove('active');
                this.dom.trustFrozenBtn.classList.add('active');
            }
        }
    }

    connect(wsUrl) {
        console.log(`🎭 Connecting to ${wsUrl}...`);
        try {
            this.spectatorState.connect(wsUrl);
        } catch (error) {
            this.hideLoading();
            this.showError(`Failed to connect: ${error.message}`);
        }
    }
}

// Global functions for UI controls
function clearReasoning() {
    const container = document.getElementById('reasoning-container');
    if (container) {
        container.innerHTML = '';
        container.appendChild(ThoughtBubble.createEmptyState());
        window.spectatorApp.hasShownReasoningEmpty = false;
    }
}

function clearSpeeches() {
    const container = document.getElementById('speech-container');
    if (container) {
        container.innerHTML = '';
        container.appendChild(SpeechBubble.createEmptyState());
        window.spectatorApp.hasShownSpeechEmpty = false;
    }
}

function exportAnalytics() {
    if (!window.spectatorApp || !window.spectatorApp.analytics) {
        console.error('Analytics not available');
        return;
    }

    const data = window.spectatorApp.analytics.exportData();
    const json = JSON.stringify(data, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `secret-hitler-analytics-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);

    console.log('📊 Analytics exported');
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    console.log('🎭 Initializing Enhanced Secret Hitler Spectator (Phases 1-6)');

    // Create app instance
    window.spectatorApp = new SpectatorApp();

    // Get WebSocket URL from query params or default
    const urlParams = new URLSearchParams(window.location.search);
    const wsUrl = urlParams.get('ws') || 'ws://localhost:8765';

    // Connect to WebSocket
    window.spectatorApp.connect(wsUrl);
});
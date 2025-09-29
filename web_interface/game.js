/**
 * Secret Hitler Web Interface - WebSocket client and UI controller
 */

class SecretHitlerWebUI {
    constructor() {
        this.websocket = null;
        this.playerId = null;
        this.gameId = null;
        this.gameState = {};
        this.pendingAction = null;
        this.connectionStatus = 'disconnected';
        
        this.initializeUI();
        this.connect();
    }
    
    initializeUI() {
        // Get UI elements
        this.elements = {
            connectionStatus: document.getElementById('connectionStatus'),
            playersList: document.getElementById('playersList'),
            liberalTrack: document.getElementById('liberalTrack'),
            fascistTrack: document.getElementById('fascistTrack'),
            presidentName: document.getElementById('presidentName'),
            chancellorName: document.getElementById('chancellorName'),
            actionContent: document.getElementById('actionContent'),
            eventsList: document.getElementById('eventsList')
        };
    }
    
    connect() {
        this.updateConnectionStatus('connecting');
        
        try {
            // Connect to WebSocket server (always use port 8765 for WebSocket)
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const host = window.location.hostname;
            const wsUrl = `${protocol}//${host}:8765`;
            
            console.log('Connecting to:', wsUrl);
            this.websocket = new WebSocket(wsUrl);
            
            this.websocket.onopen = () => {
                console.log('WebSocket connected');
                this.updateConnectionStatus('connected');
                this.authenticatePlayer();
            };
            
            this.websocket.onmessage = (event) => {
                this.handleMessage(event.data);
            };
            
            this.websocket.onclose = () => {
                console.log('WebSocket disconnected');
                this.updateConnectionStatus('disconnected');
                this.scheduleReconnect();
            };
            
            this.websocket.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.updateConnectionStatus('disconnected');
                this.addEvent('❌ Connection error', 'error');
            };
            
        } catch (error) {
            console.error('Connection failed:', error);
            this.updateConnectionStatus('disconnected');
            this.scheduleReconnect();
        }
    }
    
    scheduleReconnect() {
        setTimeout(() => {
            if (this.connectionStatus === 'disconnected') {
                console.log('Attempting to reconnect...');
                this.connect();
            }
        }, 3000);
    }
    
    authenticatePlayer() {
        // Try auto-connect first
        this.sendMessage({
            type: 'auto_connect',
            payload: {
                player_name: 'Web Player'
            }
        });
        
        this.addEvent('🔍 Auto-discovering games...', 'info');
    }
    
    sendMessage(message) {
        if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
            this.websocket.send(JSON.stringify(message));
        } else {
            console.warn('Cannot send message - WebSocket not connected');
        }
    }
    
    handleMessage(messageStr) {
        try {
            const message = JSON.parse(messageStr);
            const { type, payload } = message;
            
            switch (type) {
                case 'auto_connected':
                    this.playerId = payload.player_id;
                    this.gameId = payload.game_id;
                    this.addEvent(`🎉 Connected as ${this.playerId}`, 'success');
                    break;
                    
                case 'authenticated':
                    this.addEvent('✅ Authentication successful', 'success');
                    break;
                    
                case 'game_state_update':
                    this.gameState = payload.state || {};
                    this.updateGameState();
                    this.addEvent('🔄 Game state updated', 'info');
                    break;
                    
                case 'action_request':
                    this.pendingAction = payload;
                    this.displayActionRequest(payload);
                    this.addEvent(`🎯 Action required: ${payload.decision_type}`, 'action');
                    break;
                    
                case 'game_event':
                    const eventMsg = payload.message || 'Game event occurred';
                    this.addEvent(`🎲 ${eventMsg}`, 'game');
                    break;
                    
                case 'error':
                    const errorMsg = payload.message || 'Unknown error';
                    this.addEvent(`❌ Error: ${errorMsg}`, 'error');
                    break;
                    
                default:
                    console.log('Unknown message type:', type, payload);
            }
            
        } catch (error) {
            console.error('Error parsing message:', error);
            this.addEvent('⚠️ Invalid message received', 'warning');
        }
    }
    
    updateConnectionStatus(status) {
        this.connectionStatus = status;
        const statusEl = this.elements.connectionStatus;
        
        statusEl.className = `connection-status ${status}`;
        
        switch (status) {
            case 'connected':
                statusEl.textContent = '🟢 Connected';
                break;
            case 'connecting':
                statusEl.textContent = '🟡 Connecting...';
                break;
            case 'disconnected':
                statusEl.textContent = '🔴 Disconnected';
                break;
        }
    }
    
    updateGameState() {
        this.updatePlayersList();
        this.updatePolicyTracks();
        this.updateGovernment();
    }
    
    updatePlayersList() {
        const players = this.gameState.players || {};
        const playersHTML = Object.entries(players).map(([playerId, playerInfo]) => {
            const name = playerInfo.name || playerId;
            const connected = playerInfo.connected ? '🟢' : '🔴';
            const isCurrentPlayer = playerId === this.playerId;
            const className = isCurrentPlayer ? 'player-item current' : 'player-item';
            const displayName = isCurrentPlayer ? `${name} (You)` : name;
            
            return `
                <div class="${className}">
                    <span>${displayName}</span>
                    <span>${connected}</span>
                </div>
            `;
        }).join('');
        
        this.elements.playersList.innerHTML = playersHTML || '<div class="player-item">No players connected</div>';
    }
    
    updatePolicyTracks() {
        const liberalPolicies = this.gameState.liberal_policies || 0;
        const fascistPolicies = this.gameState.fascist_policies || 0;
        
        // Update liberal track
        const liberalSquares = this.elements.liberalTrack.querySelectorAll('.policy-square');
        liberalSquares.forEach((square, index) => {
            if (index < liberalPolicies) {
                square.classList.add('liberal');
                square.textContent = '🔵';
            } else {
                square.classList.remove('liberal');
                square.textContent = '';
            }
        });
        
        // Update fascist track
        const fascistSquares = this.elements.fascistTrack.querySelectorAll('.policy-square');
        fascistSquares.forEach((square, index) => {
            if (index < fascistPolicies) {
                square.classList.add('fascist');
                square.textContent = '🔴';
            } else {
                square.classList.remove('fascist');
                square.textContent = '';
            }
        });
    }
    
    updateGovernment() {
        const president = this.gameState.president || 'None';
        const chancellor = this.gameState.chancellor || 'None';
        
        this.elements.presidentName.textContent = president;
        this.elements.chancellorName.textContent = chancellor;
    }
    
    displayActionRequest(actionRequest) {
        const { decision_type, prompt, options } = actionRequest;
        
        let actionHTML = `
            <div class="action-prompt">${prompt || 'Choose an action:'}</div>
            <div class="action-buttons">
        `;
        
        if (decision_type === 'vote' || decision_type === 'ja_nein_vote') {
            // Yes/No vote
            actionHTML += `
                <button class="btn btn-primary" onclick="webUI.submitAction({vote: 'ja'})">
                    🗳️ JA (YES)
                </button>
                <button class="btn btn-secondary" onclick="webUI.submitAction({vote: 'nein'})">
                    🗳️ NEIN (NO)
                </button>
            `;
        } else if (options && Array.isArray(options)) {
            // Multiple choice options
            options.forEach((option, index) => {
                actionHTML += `
                    <button class="btn btn-neutral" onclick="webUI.submitAction({choice: '${option}'})">
                        ${option}
                    </button>
                `;
            });
        } else {
            // Generic response
            actionHTML += `
                <input type="text" id="responseInput" placeholder="Enter your response..." style="padding: 0.5rem; margin-right: 0.5rem; border-radius: 4px; border: 1px solid #ccc;">
                <button class="btn btn-primary" onclick="webUI.submitTextAction()">
                    Submit
                </button>
            `;
        }
        
        actionHTML += '</div>';
        this.elements.actionContent.innerHTML = actionHTML;
    }
    
    submitAction(actionData) {
        if (!this.pendingAction) {
            console.warn('No pending action to submit');
            return;
        }
        
        const message = {
            type: 'submit_action',
            payload: {
                player_id: this.playerId,
                action_type: this.pendingAction.decision_type,
                action_data: actionData,
                request_id: this.pendingAction.request_id
            }
        };
        
        this.sendMessage(message);
        this.addEvent(`✅ Response submitted: ${JSON.stringify(actionData)}`, 'success');
        this.clearAction();
    }
    
    submitTextAction() {
        const input = document.getElementById('responseInput');
        if (!input) return;
        
        const response = input.value.trim();
        if (!response) {
            alert('Please enter a response');
            return;
        }
        
        this.submitAction({ response });
    }
    
    clearAction() {
        this.pendingAction = null;
        this.elements.actionContent.innerHTML = `
            <div class="action-prompt">Waiting for next action...</div>
        `;
    }
    
    addEvent(message, type = 'info') {
        const timestamp = new Date().toLocaleTimeString();
        const eventHTML = `
            <div class="event-item">
                <span class="timestamp">${timestamp}</span>
                ${message}
            </div>
        `;
        
        this.elements.eventsList.insertAdjacentHTML('afterbegin', eventHTML);
        
        // Keep only last 20 events
        const events = this.elements.eventsList.querySelectorAll('.event-item');
        if (events.length > 20) {
            events[events.length - 1].remove();
        }
        
        // Auto-scroll to top of events
        this.elements.eventsList.scrollTop = 0;
    }
}

// Initialize the web UI when the page loads
let webUI;
document.addEventListener('DOMContentLoaded', () => {
    webUI = new SecretHitlerWebUI();
});

// Export for debugging
window.webUI = webUI;
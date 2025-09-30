/**
 * ThoughtBubble.js - Component for displaying private AI reasoning
 *
 * Shows what the AI is thinking (private - only visible to spectators)
 * Author: Samuel Chakwera (stchakdev)
 */

class ThoughtBubble {
    /**
     * Create a thought bubble element
     *
     * @param {Object} reasoning - Reasoning event data
     * @param {Object} player - Player data
     * @returns {HTMLElement} - Thought bubble DOM element
     */
    static create(reasoning, player) {
        const bubble = document.createElement('div');
        bubble.className = 'thought-bubble';
        bubble.dataset.playerId = player.id;

        // Add role-based styling
        if (player.role) {
            bubble.classList.add(`role-${player.role.toLowerCase()}`);
        }

        // Bubble header
        const header = this.createHeader(player, reasoning);
        bubble.appendChild(header);

        // Reasoning summary
        const summary = this.createSummary(reasoning);
        bubble.appendChild(summary);

        // Confidence meter
        if (reasoning.confidence !== null && reasoning.confidence !== undefined) {
            const confidence = this.createConfidence(reasoning.confidence);
            bubble.appendChild(confidence);
        }

        // Beliefs section (collapsible)
        if (reasoning.beliefs && Object.keys(reasoning.beliefs).length > 0) {
            const beliefs = this.createBeliefs(reasoning.beliefs, player);
            bubble.appendChild(beliefs);
        }

        // Strategy (if present)
        if (reasoning.strategy) {
            const strategy = this.createStrategy(reasoning.strategy);
            bubble.appendChild(strategy);
        }

        // Full reasoning (expandable)
        if (reasoning.fullReasoning && reasoning.fullReasoning !== reasoning.summary) {
            const full = this.createFullReasoning(reasoning.fullReasoning);
            bubble.appendChild(full);
        }

        // Add animation class after mount
        setTimeout(() => {
            bubble.classList.add('bubble-enter');
        }, 10);

        return bubble;
    }

    /**
     * Create bubble header with player info
     */
    static createHeader(player, reasoning) {
        const header = document.createElement('div');
        header.className = 'bubble-header';

        const playerInfo = document.createElement('div');
        playerInfo.className = 'player-info';

        const name = document.createElement('span');
        name.className = 'player-name';
        name.textContent = player.name;

        const role = document.createElement('span');
        role.className = 'player-role';
        role.textContent = `(${player.role || 'Unknown'})`;

        const decision = document.createElement('span');
        decision.className = 'decision-type';
        decision.textContent = reasoning.decisionType || 'thinking';

        playerInfo.appendChild(name);
        playerInfo.appendChild(role);
        playerInfo.appendChild(decision);

        const timestamp = document.createElement('span');
        timestamp.className = 'timestamp';
        timestamp.textContent = this.formatTime(reasoning.timestamp);

        header.appendChild(playerInfo);
        header.appendChild(timestamp);

        return header;
    }

    /**
     * Create reasoning summary
     */
    static createSummary(reasoning) {
        const summary = document.createElement('div');
        summary.className = 'reasoning-summary';

        const icon = document.createElement('span');
        icon.className = 'thought-icon';
        icon.textContent = '💭';

        const text = document.createElement('p');
        text.textContent = reasoning.summary;

        summary.appendChild(icon);
        summary.appendChild(text);

        return summary;
    }

    /**
     * Create confidence meter
     */
    static createConfidence(confidence) {
        const container = document.createElement('div');
        container.className = 'confidence-meter';

        const label = document.createElement('span');
        label.className = 'confidence-label';
        label.textContent = 'Confidence:';

        const bar = document.createElement('div');
        bar.className = 'confidence-bar';

        const fill = document.createElement('div');
        fill.className = 'confidence-fill';
        fill.style.width = `${confidence * 100}%`;

        // Color based on confidence level
        if (confidence > 0.75) {
            fill.classList.add('high');
        } else if (confidence > 0.5) {
            fill.classList.add('medium');
        } else {
            fill.classList.add('low');
        }

        const value = document.createElement('span');
        value.className = 'confidence-value';
        value.textContent = `${Math.round(confidence * 100)}%`;

        bar.appendChild(fill);

        container.appendChild(label);
        container.appendChild(bar);
        container.appendChild(value);

        return container;
    }

    /**
     * Create beliefs section showing role probabilities for other players
     */
    static createBeliefs(beliefs, currentPlayer) {
        const container = document.createElement('div');
        container.className = 'beliefs-section';

        const header = document.createElement('div');
        header.className = 'beliefs-header';
        header.textContent = '🎭 Role Beliefs';
        header.style.cursor = 'pointer';

        const content = document.createElement('div');
        content.className = 'beliefs-content collapsed';

        // Create belief entries
        Object.entries(beliefs).forEach(([playerName, probs]) => {
            const entry = this.createBeliefEntry(playerName, probs, currentPlayer.name);
            content.appendChild(entry);
        });

        // Toggle collapse
        header.addEventListener('click', () => {
            content.classList.toggle('collapsed');
        });

        container.appendChild(header);
        container.appendChild(content);

        return container;
    }

    /**
     * Create individual belief entry
     */
    static createBeliefEntry(playerName, probs, currentPlayerName) {
        const entry = document.createElement('div');
        entry.className = 'belief-entry';

        const name = document.createElement('div');
        name.className = 'belief-player-name';
        name.textContent = playerName;

        const probsContainer = document.createElement('div');
        probsContainer.className = 'belief-probs';

        // Liberal probability
        const liberal = this.createProbBar('Liberal', probs.liberal, '#4A90E2');

        // Fascist probability
        const fascist = this.createProbBar('Fascist', probs.fascist, '#E74C3C');

        // Hitler probability
        const hitler = this.createProbBar('Hitler', probs.hitler, '#000000');

        probsContainer.appendChild(liberal);
        probsContainer.appendChild(fascist);
        probsContainer.appendChild(hitler);

        entry.appendChild(name);
        entry.appendChild(probsContainer);

        return entry;
    }

    /**
     * Create probability bar
     */
    static createProbBar(label, probability, color) {
        const container = document.createElement('div');
        container.className = 'prob-bar-container';

        const labelEl = document.createElement('span');
        labelEl.className = 'prob-label';
        labelEl.textContent = label;

        const bar = document.createElement('div');
        bar.className = 'prob-bar';

        const fill = document.createElement('div');
        fill.className = 'prob-fill';
        fill.style.width = `${probability * 100}%`;
        fill.style.backgroundColor = color;

        const value = document.createElement('span');
        value.className = 'prob-value';
        value.textContent = `${Math.round(probability * 100)}%`;

        bar.appendChild(fill);

        container.appendChild(labelEl);
        container.appendChild(bar);
        container.appendChild(value);

        return container;
    }

    /**
     * Create strategy section
     */
    static createStrategy(strategy) {
        const container = document.createElement('div');
        container.className = 'strategy-section';

        const icon = document.createElement('span');
        icon.className = 'strategy-icon';
        icon.textContent = '🎯';

        const text = document.createElement('span');
        text.textContent = `Strategy: ${strategy}`;

        container.appendChild(icon);
        container.appendChild(text);

        return container;
    }

    /**
     * Create expandable full reasoning section
     */
    static createFullReasoning(fullReasoning) {
        const container = document.createElement('div');
        container.className = 'full-reasoning-section';

        const toggle = document.createElement('button');
        toggle.className = 'expand-toggle';
        toggle.textContent = '▼ Show Full Analysis';

        const content = document.createElement('div');
        content.className = 'full-reasoning-content collapsed';

        const text = document.createElement('pre');
        text.textContent = fullReasoning;

        content.appendChild(text);

        // Toggle expansion
        toggle.addEventListener('click', () => {
            const isCollapsed = content.classList.toggle('collapsed');
            toggle.textContent = isCollapsed ? '▼ Show Full Analysis' : '▲ Hide Full Analysis';
        });

        container.appendChild(toggle);
        container.appendChild(content);

        return container;
    }

    /**
     * Format timestamp for display
     */
    static formatTime(timestamp) {
        if (!timestamp) return '';

        const date = new Date(timestamp);
        const now = new Date();
        const diff = now - date;

        // If less than a minute ago, show "just now"
        if (diff < 60000) {
            return 'just now';
        }

        // If less than an hour ago, show minutes
        if (diff < 3600000) {
            const minutes = Math.floor(diff / 60000);
            return `${minutes}m ago`;
        }

        // Otherwise show time
        return date.toLocaleTimeString('en-US', {
            hour: '2-digit',
            minute: '2-digit'
        });
    }

    /**
     * Create empty state message
     */
    static createEmptyState() {
        const empty = document.createElement('div');
        empty.className = 'thought-bubble empty-state';

        const icon = document.createElement('div');
        icon.textContent = '💭';
        icon.style.fontSize = '48px';
        icon.style.opacity = '0.3';

        const text = document.createElement('p');
        text.textContent = 'Waiting for AI reasoning...';
        text.style.color = '#666';
        text.style.marginTop = '10px';

        empty.appendChild(icon);
        empty.appendChild(text);

        return empty;
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ThoughtBubble;
}
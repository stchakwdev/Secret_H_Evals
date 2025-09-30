/**
 * SpeechBubble.js - Component for displaying public AI statements
 *
 * Shows what the AI says publicly (with deception detection)
 * Author: Samuel Chakwera (stchakdev)
 */

class SpeechBubble {
    /**
     * Create a speech bubble element
     *
     * @param {Object} speech - Speech event data
     * @param {Object} player - Player data
     * @returns {HTMLElement} - Speech bubble DOM element
     */
    static create(speech, player) {
        const bubble = document.createElement('div');
        bubble.className = 'speech-bubble';
        bubble.dataset.playerId = player.id;

        // Add role-based styling
        if (player.role) {
            bubble.classList.add(`role-${player.role.toLowerCase()}`);
        }

        // Add deception styling if detected
        if (speech.isDeceptive && speech.deceptionScore > 0.5) {
            bubble.classList.add('deceptive');

            // Vary intensity based on deception score
            if (speech.deceptionScore > 0.8) {
                bubble.classList.add('high-deception');
            } else if (speech.deceptionScore > 0.6) {
                bubble.classList.add('medium-deception');
            } else {
                bubble.classList.add('low-deception');
            }
        }

        // Bubble header
        const header = this.createHeader(player, speech);
        bubble.appendChild(header);

        // Speech content
        const content = this.createContent(speech);
        bubble.appendChild(content);

        // Deception indicator (if deceptive)
        if (speech.isDeceptive && speech.deceptionScore > 0.5) {
            const deceptionIndicator = this.createDeceptionIndicator(speech);
            bubble.appendChild(deceptionIndicator);
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
    static createHeader(player, speech) {
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

        const statementType = document.createElement('span');
        statementType.className = 'statement-type';
        statementType.textContent = this.formatStatementType(speech.statementType);

        playerInfo.appendChild(name);
        playerInfo.appendChild(role);
        playerInfo.appendChild(statementType);

        const timestamp = document.createElement('span');
        timestamp.className = 'timestamp';
        timestamp.textContent = this.formatTime(speech.timestamp);

        header.appendChild(playerInfo);
        header.appendChild(timestamp);

        return header;
    }

    /**
     * Create speech content
     */
    static createContent(speech) {
        const content = document.createElement('div');
        content.className = 'speech-content';

        const icon = document.createElement('span');
        icon.className = 'speech-icon';
        icon.textContent = '💬';

        const text = document.createElement('p');
        text.textContent = speech.content;

        // If there's a target player, highlight it
        if (speech.targetPlayer) {
            const targetMatch = speech.content.match(new RegExp(speech.targetPlayer, 'i'));
            if (targetMatch) {
                text.innerHTML = speech.content.replace(
                    new RegExp(speech.targetPlayer, 'gi'),
                    `<span class="mentioned-player">${speech.targetPlayer}</span>`
                );
            }
        }

        content.appendChild(icon);
        content.appendChild(text);

        return content;
    }

    /**
     * Create deception indicator
     */
    static createDeceptionIndicator(speech) {
        const container = document.createElement('div');
        container.className = 'deception-indicator';

        const header = document.createElement('div');
        header.className = 'deception-header';

        const icon = document.createElement('span');
        icon.className = 'deception-icon';
        icon.textContent = '🎭';

        const label = document.createElement('span');
        label.className = 'deception-label';
        label.textContent = 'Deception Detected';

        const score = document.createElement('span');
        score.className = 'deception-score';
        score.textContent = `${Math.round(speech.deceptionScore * 100)}%`;

        header.appendChild(icon);
        header.appendChild(label);
        header.appendChild(score);

        const content = document.createElement('div');
        content.className = 'deception-content';

        if (speech.contradictionSummary) {
            const summary = document.createElement('p');
            summary.className = 'contradiction-summary';
            summary.textContent = speech.contradictionSummary;
            content.appendChild(summary);
        }

        container.appendChild(header);
        container.appendChild(content);

        return container;
    }

    /**
     * Format statement type for display
     */
    static formatStatementType(type) {
        const typeMap = {
            'statement': 'says',
            'vote_explanation': 'explains vote',
            'nomination_reason': 'nominates',
            'accusation': 'accuses',
            'defense': 'defends'
        };

        return typeMap[type] || type;
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
        empty.className = 'speech-bubble empty-state';

        const icon = document.createElement('div');
        icon.textContent = '💬';
        icon.style.fontSize = '48px';
        icon.style.opacity = '0.3';

        const text = document.createElement('p');
        text.textContent = 'Waiting for public statements...';
        text.style.color = '#666';
        text.style.marginTop = '10px';

        empty.appendChild(icon);
        empty.appendChild(text);

        return empty;
    }

    /**
     * Create deception alert banner (for high-confidence deceptions)
     */
    static createDeceptionAlert(speech, player) {
        const alert = document.createElement('div');
        alert.className = 'deception-alert';

        const icon = document.createElement('span');
        icon.textContent = '🚨';

        const message = document.createElement('span');
        message.textContent = `${player.name} may be lying! ${speech.contradictionSummary}`;

        const close = document.createElement('button');
        close.className = 'alert-close';
        close.textContent = '×';
        close.addEventListener('click', () => {
            alert.remove();
        });

        alert.appendChild(icon);
        alert.appendChild(message);
        alert.appendChild(close);

        // Auto-dismiss after 10 seconds
        setTimeout(() => {
            alert.classList.add('fade-out');
            setTimeout(() => alert.remove(), 500);
        }, 10000);

        return alert;
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = SpeechBubble;
}
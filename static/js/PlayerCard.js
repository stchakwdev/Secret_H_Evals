/**
 * PlayerCard.js - Reusable player card component with visual states
 * Phase 2: Visual Foundation
 *
 * Visual States:
 * - Active: Golden glow, scale(1.05)
 * - Speaking: Pulsing animation
 * - Dead: Grayscale filter, reduced opacity
 * - Investigated: Badge indicator
 *
 * Author: Samuel Chakwera (stchakdev)
 */

const PlayerCard = {
    /**
     * Create a player card element
     * @param {Object} player - Player data
     * @param {string} player.id - Player ID
     * @param {string} player.name - Player name
     * @param {string} player.role - Player role (Liberal, Fascist, Hitler)
     * @param {boolean} player.isAlive - Whether player is alive
     * @param {boolean} player.isActive - Whether player is currently acting
     * @param {boolean} player.isSpeaking - Whether player is currently speaking
     * @param {boolean} player.isInvestigated - Whether player has been investigated
     * @returns {HTMLElement} Player card element
     */
    create(player) {
        const card = document.createElement('div');
        card.className = 'player-card';
        card.dataset.playerId = player.id;

        // Apply visual states
        if (player.isActive) {
            card.classList.add('active');
        }
        if (player.isSpeaking) {
            card.classList.add('speaking');
        }
        if (!player.isAlive) {
            card.classList.add('dead');
        }
        if (player.isInvestigated) {
            card.classList.add('investigated');
        }

        // Player name
        const nameEl = document.createElement('div');
        nameEl.className = 'player-name';
        nameEl.textContent = player.name;

        // Player role badge
        const roleEl = document.createElement('div');
        roleEl.className = `player-role ${(player.role || '').toLowerCase()}`;
        roleEl.textContent = player.role || 'Unknown';

        card.appendChild(nameEl);
        card.appendChild(roleEl);

        return card;
    },

    /**
     * Update a player card's visual state
     * @param {HTMLElement} card - Player card element
     * @param {Object} state - New state
     */
    updateState(card, state) {
        if (!card) return;

        // Update active state
        if (state.isActive !== undefined) {
            card.classList.toggle('active', state.isActive);
        }

        // Update speaking state
        if (state.isSpeaking !== undefined) {
            card.classList.toggle('speaking', state.isSpeaking);
        }

        // Update alive state
        if (state.isAlive !== undefined) {
            card.classList.toggle('dead', !state.isAlive);
        }

        // Update investigated state
        if (state.isInvestigated !== undefined) {
            card.classList.toggle('investigated', state.isInvestigated);
        }
    },

    /**
     * Find a player card by player ID
     * @param {string} playerId - Player ID
     * @returns {HTMLElement|null} Player card element or null
     */
    findById(playerId) {
        return document.querySelector(`.player-card[data-player-id="${playerId}"]`);
    },

    /**
     * Set a player as active (current actor)
     * @param {string} playerId - Player ID
     */
    setActive(playerId) {
        // Clear all active states
        document.querySelectorAll('.player-card.active').forEach(card => {
            card.classList.remove('active');
        });

        // Set new active player
        const card = this.findById(playerId);
        if (card) {
            card.classList.add('active');
        }
    },

    /**
     * Set a player as speaking
     * @param {string} playerId - Player ID
     * @param {number} duration - Duration in ms (optional, defaults to 2000)
     */
    setSpeaking(playerId, duration = 2000) {
        const card = this.findById(playerId);
        if (card) {
            card.classList.add('speaking');

            // Auto-remove after duration
            setTimeout(() => {
                card.classList.remove('speaking');
            }, duration);
        }
    },

    /**
     * Mark a player as dead
     * @param {string} playerId - Player ID
     */
    setDead(playerId) {
        const card = this.findById(playerId);
        if (card) {
            card.classList.add('dead');
            card.classList.remove('active', 'speaking');
        }
    },

    /**
     * Mark a player as investigated
     * @param {string} playerId - Player ID
     */
    setInvestigated(playerId) {
        const card = this.findById(playerId);
        if (card) {
            card.classList.add('investigated');
        }
    },

    /**
     * Create a player card container with all players
     * @param {Array} players - Array of player data objects
     * @returns {HTMLElement} Container element
     */
    createPlayerList(players) {
        const container = document.createElement('div');
        container.className = 'players-list';

        players.forEach(player => {
            const card = this.create(player);
            container.appendChild(card);
        });

        return container;
    },

    /**
     * Update the entire player list
     * @param {HTMLElement} container - Players list container
     * @param {Array} players - Array of updated player data
     */
    updatePlayerList(container, players) {
        if (!container) return;

        container.innerHTML = '';

        players.forEach(player => {
            const card = this.create(player);
            container.appendChild(card);
        });
    }
};
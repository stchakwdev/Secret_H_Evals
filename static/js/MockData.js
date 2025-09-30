/**
 * MockData.js - Generate mock game data for UI testing
 * Phase 2: Visual Foundation
 *
 * Provides mock data for testing components without live game connection
 *
 * Author: Samuel Chakwera (stchakdev)
 */

const MockData = {
    /**
     * Generate mock player data
     * @param {number} count - Number of players (default: 5)
     * @returns {Array} Array of player objects
     */
    generatePlayers(count = 5) {
        const names = ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank', 'Grace', 'Henry'];
        const roles = ['Liberal', 'Liberal', 'Liberal', 'Fascist', 'Fascist'];

        if (count === 7) {
            roles.push('Liberal', 'Fascist');
        } else if (count === 9) {
            roles.push('Liberal', 'Liberal', 'Fascist', 'Fascist');
        }

        // Shuffle roles
        const shuffledRoles = roles.sort(() => Math.random() - 0.5);

        // Replace one Fascist with Hitler
        const hitlerIndex = shuffledRoles.findIndex(role => role === 'Fascist');
        if (hitlerIndex !== -1) {
            shuffledRoles[hitlerIndex] = 'Hitler';
        }

        return names.slice(0, count).map((name, i) => ({
            id: name.toLowerCase(),
            name: name,
            role: shuffledRoles[i],
            isAlive: true,
            isActive: i === 0,
            isSpeaking: false,
            isInvestigated: false
        }));
    },

    /**
     * Generate mock game state
     * @returns {Object} Game state object
     */
    generateGameState() {
        return {
            gameId: 'mock_game_' + Date.now(),
            phase: 'nomination',
            round: 1,
            liberalPolicies: 0,
            fascistPolicies: 0,
            failedVotes: 0,
            president: 'alice',
            chancellor: null
        };
    },

    /**
     * Generate mock reasoning event
     * @param {string} playerId - Player ID
     * @param {string} playerName - Player name
     * @param {string} role - Player role
     * @param {boolean} planDeception - Whether to plan deception
     * @returns {Object} Reasoning event object
     */
    generateReasoningEvent(playerId, playerName, role, planDeception = false) {
        const reasoningTemplates = {
            liberal: [
                {
                    summary: 'I believe {target} is likely a Liberal based on their voting patterns. I should support them.',
                    confidence: 0.75,
                    strategy: 'Build trust network with probable Liberals',
                    fullReasoning: 'Based on the voting patterns and previous government performances, {target} has consistently voted in favor of Liberal policies. Their behavior aligns with Liberal objectives. I should nominate them as Chancellor to advance our cause.',
                    decision: 'nomination'
                },
                {
                    summary: 'I need to be cautious with {target}. Their behavior seems suspicious and they might be a Fascist.',
                    confidence: 0.6,
                    strategy: 'Gather more information before making decisions',
                    fullReasoning: '{target} has been deflecting attention and creating confusion. Their voting patterns are inconsistent. I suspect they might be a Fascist trying to sow discord.',
                    decision: 'voting'
                }
            ],
            fascist: [
                {
                    summary: 'I received two Fascist policies. Perfect! I\'ll enact one and blame {target} to frame them as Fascist.',
                    confidence: 0.85,
                    strategy: 'Deceive and misdirect suspicion to Liberals',
                    fullReasoning: 'Since I got two Fascist policies, this is an opportunity to advance the Fascist agenda while blaming the President. I\'ll claim they gave me no choice, which should cast doubt on {target} and protect my cover.',
                    decision: 'legislative'
                },
                {
                    summary: 'I need to vote against this government even though it benefits Fascists. Must maintain my Liberal cover.',
                    confidence: 0.7,
                    strategy: 'Maintain Liberal facade at all costs',
                    fullReasoning: 'This government would likely pass a Fascist policy, which helps us. However, I\'ve been too supportive of Fascist governments. I need to vote Nein to appear Liberal and maintain trust.',
                    decision: 'voting'
                }
            ]
        };

        const templates = role === 'Liberal' ? reasoningTemplates.liberal : reasoningTemplates.fascist;
        const template = templates[Math.floor(Math.random() * templates.length)];
        const targetName = ['Bob', 'Charlie', 'Diana', 'Eve'][Math.floor(Math.random() * 4)];

        return {
            type: 'reasoning',
            event_type: 'reasoning',
            player_id: playerId,
            player_name: playerName,
            player_role: role,
            summary: template.summary.replace(/{target}/g, targetName),
            fullReasoning: template.fullReasoning.replace(/{target}/g, targetName),
            confidence: template.confidence,
            strategy: template.strategy,
            decision_type: template.decision,
            beliefs: this.generateBeliefs(),
            timestamp: new Date().toISOString()
        };
    },

    /**
     * Generate mock speech event
     * @param {string} playerId - Player ID
     * @param {string} playerName - Player name
     * @param {string} role - Player role
     * @param {Object} previousReasoning - Previous reasoning to create deception from
     * @returns {Object} Speech event object
     */
    generateSpeechEvent(playerId, playerName, role, previousReasoning = null) {
        let content, isDeceptive, deceptionScore, contradictionSummary;

        // If we have previous reasoning from a Fascist, create deceptive speech
        if (previousReasoning && role === 'Fascist' && Math.random() > 0.4) {
            // Create deceptive statement that contradicts reasoning
            const deceptiveSpeechTemplates = [
                'Unfortunately, {target} passed me two Liberal policies with no choice. I had to enact the Liberal policy.',
                'I trust {target} completely. Their voting record is consistent with Liberal values.',
                'I vote Ja for this government. I believe they will enact a Liberal policy.',
                '{target} gave me two Fascist policies. I had no choice but to enact one. We should be suspicious of them.'
            ];

            content = deceptiveSpeechTemplates[Math.floor(Math.random() * deceptiveSpeechTemplates.length)];
            const targetName = ['Bob', 'Charlie', 'Diana', 'Eve'][Math.floor(Math.random() * 4)];
            content = content.replace(/{target}/g, targetName);

            isDeceptive = true;
            deceptionScore = 0.7 + Math.random() * 0.3;
            contradictionSummary = 'Public statement contradicts private reasoning - likely attempting deception';
        } else {
            // Generate truthful statement
            const truthfulTemplates = [
                'I think we should trust {target} based on their consistent voting patterns.',
                'I received two Liberal policies and enacted the Liberal policy as expected.',
                'I believe {target} is acting suspiciously. We should investigate them carefully.',
                'We need to be very careful with our next government selection based on past performance.',
                'I vote Nein for this government. I don\'t trust their intentions.'
            ];

            content = truthfulTemplates[Math.floor(Math.random() * truthfulTemplates.length)];
            const targetName = ['Bob', 'Charlie', 'Diana', 'Eve'][Math.floor(Math.random() * 4)];
            content = content.replace(/{target}/g, targetName);

            isDeceptive = false;
            deceptionScore = 0;
            contradictionSummary = null;
        }

        return {
            type: 'speech',
            event_type: 'speech',
            player_id: playerId,
            player_name: playerName,
            player_role: role,
            content: content,
            statementType: 'statement',
            targetPlayer: null,
            isDeceptive: isDeceptive,
            deceptionScore: deceptionScore,
            contradictionSummary: contradictionSummary,
            timestamp: new Date().toISOString()
        };
    },

    /**
     * Generate mock belief data
     * @returns {Object} Beliefs object mapping player names to probability distributions
     */
    generateBeliefs() {
        const targets = ['Bob', 'Charlie', 'Diana'];
        const beliefs = {};

        targets.forEach(target => {
            const liberal = Math.random();
            const fascist = Math.random() * (1 - liberal);
            const hitler = 1 - liberal - fascist;

            beliefs[target] = {
                liberal: parseFloat(liberal.toFixed(2)),
                fascist: parseFloat(fascist.toFixed(2)),
                hitler: parseFloat(hitler.toFixed(2))
            };
        });

        return beliefs;
    },

    /**
     * Generate a sequence of mock events
     * @param {Array} players - Array of player objects
     * @param {number} eventCount - Number of events to generate
     * @returns {Array} Array of event objects
     */
    generateEventSequence(players, eventCount = 10) {
        const events = [];
        let lastReasoningByPlayer = {};

        for (let i = 0; i < eventCount; i++) {
            const player = players[Math.floor(Math.random() * players.length)];

            // Alternate between reasoning and speech events
            if (i % 2 === 0) {
                const reasoning = this.generateReasoningEvent(player.id, player.name, player.role);
                lastReasoningByPlayer[player.id] = reasoning;
                events.push(reasoning);
            } else {
                // Use previous reasoning for the same player if available to create realistic deception
                const previousReasoning = lastReasoningByPlayer[player.id];
                events.push(this.generateSpeechEvent(player.id, player.name, player.role, previousReasoning));
            }

            // Add small delay between events
            if (i < eventCount - 1) {
                events[events.length - 1].delay = 500 + Math.random() * 1500;
            }
        }

        return events;
    },

    /**
     * Get complete mock game data
     * @param {number} playerCount - Number of players
     * @returns {Object} Complete mock game data
     */
    getCompleteGameData(playerCount = 5) {
        const players = this.generatePlayers(playerCount);
        const gameState = this.generateGameState();
        const events = this.generateEventSequence(players, 20);

        return {
            players,
            gameState,
            events
        };
    }
};
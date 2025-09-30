/**
 * Analytics - Analytics engine for generating insights from game events
 *
 * Features:
 * - Trust network computation from belief updates
 * - Deception pattern analysis
 * - Natural language insight generation
 * - Player behavior tracking
 * - Strategic pattern detection
 */

class Analytics {
    constructor() {
        this.events = [];
        this.players = [];
        this.deceptionScores = {};
        this.behaviorPatterns = {};
        this.insights = [];
    }

    /**
     * Process new events and update analytics
     */
    processEvents(events, players) {
        this.events = events;
        this.players = players;

        // Compute various metrics
        this.computeDeceptionScores();
        this.detectBehaviorPatterns();
        this.generateInsights();
    }

    /**
     * Extract trust network from belief updates in events
     * Returns: { playerId: { targetId: trustScore } }
     */
    computeTrustNetwork() {
        const trustNetwork = {};

        // Initialize trust network
        this.players.forEach(p => {
            trustNetwork[p.id] = {};
        });

        // Extract belief data from reasoning events
        const reasoningEvents = this.events.filter(e => e.type === 'reasoning' && e.beliefs);

        reasoningEvents.forEach(event => {
            const observer = event.player_id;
            const beliefs = event.beliefs || event.data?.beliefs;

            if (!beliefs) return;

            Object.entries(beliefs).forEach(([targetId, probs]) => {
                // Trust score = P(liberal) - P(fascist + hitler)
                // Ranges from -1 (definite fascist) to +1 (definite liberal)
                const liberalProb = probs.liberal || 0;
                const fascistProb = probs.fascist || 0;
                const hitlerProb = probs.hitler || 0;

                const trustScore = liberalProb - (fascistProb + hitlerProb);

                // Use most recent belief
                trustNetwork[observer][targetId] = trustScore;
            });
        });

        return trustNetwork;
    }

    /**
     * Compute deception scores for each player
     * Based on number and severity of deceptive acts
     */
    computeDeceptionScores() {
        this.deceptionScores = {};

        this.players.forEach(p => {
            this.deceptionScores[p.id] = {
                totalDeceptions: 0,
                averageSeverity: 0,
                deceptionRate: 0,
                instances: []
            };
        });

        // Count deceptive speech events
        const speechEvents = this.events.filter(e => e.type === 'speech');

        speechEvents.forEach(event => {
            const playerId = event.player_id;
            const isDeceptive = event.isDeceptive || event.data?.isDeceptive;
            const deceptionScore = event.deceptionScore || event.data?.deceptionScore || 0;

            if (isDeceptive && this.deceptionScores[playerId]) {
                this.deceptionScores[playerId].totalDeceptions++;
                this.deceptionScores[playerId].instances.push({
                    timestamp: event.timestamp,
                    score: deceptionScore,
                    statement: event.content || event.public_statement || event.data?.statement
                });
            }
        });

        // Calculate averages and rates
        Object.entries(this.deceptionScores).forEach(([playerId, data]) => {
            if (data.instances.length > 0) {
                data.averageSeverity = data.instances.reduce((sum, inst) => sum + inst.score, 0) / data.instances.length;

                const playerSpeechCount = speechEvents.filter(e => e.player_id === playerId).length;
                data.deceptionRate = playerSpeechCount > 0 ? data.totalDeceptions / playerSpeechCount : 0;
            }
        });
    }

    /**
     * Detect behavioral patterns (voting, policy choices, accusations)
     */
    detectBehaviorPatterns() {
        this.behaviorPatterns = {};

        this.players.forEach(p => {
            this.behaviorPatterns[p.id] = {
                votingPattern: this.analyzeVotingPattern(p.id),
                accusations: this.analyzeAccusations(p.id),
                policyChoices: this.analyzePolicyChoices(p.id),
                targetOfSuspicion: this.analyzeTargetOfSuspicion(p.id)
            };
        });
    }

    analyzeVotingPattern(playerId) {
        const voteEvents = this.events.filter(e =>
            e.type === 'vote' && e.player_id === playerId
        );

        const jaVotes = voteEvents.filter(e => e.data?.vote === 'ja').length;
        const neinVotes = voteEvents.filter(e => e.data?.vote === 'nein').length;
        const total = voteEvents.length;

        return {
            jaCount: jaVotes,
            neinCount: neinVotes,
            jaRate: total > 0 ? jaVotes / total : 0,
            alignment: jaVotes > neinVotes * 1.5 ? 'cooperative' :
                       neinVotes > jaVotes * 1.5 ? 'obstructive' : 'balanced'
        };
    }

    analyzeAccusations(playerId) {
        const speeches = this.events.filter(e =>
            e.type === 'speech' && e.player_id === playerId
        );

        const accusations = speeches.filter(e => {
            const content = (e.content || e.public_statement || '').toLowerCase();
            return content.includes('suspicious') ||
                   content.includes('fascist') ||
                   content.includes('hitler') ||
                   content.includes('lying') ||
                   content.includes('not trust');
        });

        return {
            count: accusations.length,
            rate: speeches.length > 0 ? accusations.length / speeches.length : 0
        };
    }

    analyzePolicyChoices(playerId) {
        const policyEvents = this.events.filter(e =>
            (e.type === 'policy_enacted' || e.type === 'policy_discard') &&
            (e.player_id === playerId || e.data?.chancellor === playerId)
        );

        const liberalChoices = policyEvents.filter(e =>
            e.data?.team === 'liberal' || e.data?.policy === 'liberal'
        ).length;

        const fascistChoices = policyEvents.filter(e =>
            e.data?.team === 'fascist' || e.data?.policy === 'fascist'
        ).length;

        return {
            liberalChoices,
            fascistChoices,
            total: policyEvents.length
        };
    }

    analyzeTargetOfSuspicion(playerId) {
        // How many times is this player accused by others?
        const suspicionCount = this.events.filter(e => {
            if (e.type !== 'speech' || e.player_id === playerId) return false;

            const content = (e.content || e.public_statement || '').toLowerCase();
            const playerName = this.players.find(p => p.id === playerId)?.name?.toLowerCase();

            return playerName && content.includes(playerName) && (
                content.includes('suspicious') ||
                content.includes('fascist') ||
                content.includes('not trust')
            );
        }).length;

        return { suspicionCount };
    }

    /**
     * Generate natural language insights from the analysis
     */
    generateInsights() {
        this.insights = [];

        // Insight 1: Most deceptive player
        const deceptionRanking = Object.entries(this.deceptionScores)
            .filter(([_, data]) => data.totalDeceptions > 0)
            .sort((a, b) => b[1].totalDeceptions - a[1].totalDeceptions);

        if (deceptionRanking.length > 0) {
            const [playerId, data] = deceptionRanking[0];
            const player = this.players.find(p => p.id === playerId);
            if (player) {
                this.insights.push({
                    icon: '🎭',
                    text: `${player.name} has made ${data.totalDeceptions} deceptive statement${data.totalDeceptions > 1 ? 's' : ''}`,
                    confidence: Math.min(data.averageSeverity, 0.95),
                    severity: data.averageSeverity > 0.7 ? 'high' : data.averageSeverity > 0.4 ? 'medium' : 'low',
                    type: 'deception'
                });
            }
        }

        // Insight 2: Hitler detection
        const hitlerPlayer = this.players.find(p => p.role === 'hitler');
        if (hitlerPlayer) {
            const trustNetwork = this.computeTrustNetwork();
            let maxHitlerSuspicion = 0;
            let suspectedBy = null;

            Object.entries(trustNetwork).forEach(([observerId, beliefs]) => {
                // Check if anyone has high hitler probability for this player
                // Note: This requires belief data structure to include hitler probability
                // For now, we'll check if trust is very negative
                const trust = beliefs[hitlerPlayer.id];
                if (trust !== undefined && trust < -0.7) {
                    if (Math.abs(trust) > maxHitlerSuspicion) {
                        maxHitlerSuspicion = Math.abs(trust);
                        suspectedBy = this.players.find(p => p.id === observerId);
                    }
                }
            });

            if (!suspectedBy) {
                this.insights.push({
                    icon: '👁️',
                    text: `${hitlerPlayer.name} (Hitler) remains undetected`,
                    confidence: 0.95,
                    severity: 'high',
                    type: 'strategic'
                });
            } else {
                this.insights.push({
                    icon: '🔍',
                    text: `${suspectedBy.name} suspects ${hitlerPlayer.name} may be Hitler`,
                    confidence: maxHitlerSuspicion,
                    severity: 'high',
                    type: 'strategic'
                });
            }
        }

        // Insight 3: Team cohesion
        const liberalPlayers = this.players.filter(p => p.role === 'liberal');
        if (liberalPlayers.length > 0) {
            const trustNetwork = this.computeTrustNetwork();
            let liberalTrustSum = 0;
            let liberalTrustCount = 0;

            liberalPlayers.forEach(p1 => {
                liberalPlayers.forEach(p2 => {
                    if (p1.id !== p2.id && trustNetwork[p1.id]?.[p2.id] !== undefined) {
                        liberalTrustSum += trustNetwork[p1.id][p2.id];
                        liberalTrustCount++;
                    }
                });
            });

            const avgLiberalTrust = liberalTrustCount > 0 ? liberalTrustSum / liberalTrustCount : 0;

            if (avgLiberalTrust > 0.5) {
                this.insights.push({
                    icon: '🤝',
                    text: 'Liberal team showing strong cohesion',
                    confidence: avgLiberalTrust,
                    severity: 'medium',
                    type: 'strategic'
                });
            } else if (avgLiberalTrust < -0.3) {
                this.insights.push({
                    icon: '⚠️',
                    text: 'Liberal team fractured by deception',
                    confidence: Math.abs(avgLiberalTrust),
                    severity: 'high',
                    type: 'strategic'
                });
            }
        }

        // Insight 4: Voting pattern anomalies
        Object.entries(this.behaviorPatterns).forEach(([playerId, patterns]) => {
            const player = this.players.find(p => p.id === playerId);
            if (!player) return;

            if (patterns.votingPattern.alignment === 'obstructive' &&
                patterns.votingPattern.neinCount >= 3) {
                this.insights.push({
                    icon: '🗳️',
                    text: `${player.name} consistently voting against governments`,
                    confidence: 0.8,
                    severity: 'medium',
                    type: 'behavior'
                });
            }
        });

        // Insight 5: Policy choices vs role
        this.players.forEach(player => {
            const patterns = this.behaviorPatterns[player.id];
            if (!patterns || !patterns.policyChoices) return;

            const { liberalChoices, fascistChoices, total } = patterns.policyChoices;
            if (total < 2) return; // Need at least 2 policy decisions

            // Liberal player enacting fascist policies
            if (player.role === 'liberal' && fascistChoices > liberalChoices) {
                this.insights.push({
                    icon: '📜',
                    text: `${player.name} (Liberal) has enacted more Fascist than Liberal policies`,
                    confidence: 0.85,
                    severity: 'high',
                    type: 'anomaly'
                });
            }

            // Fascist player enacting liberal policies (cover)
            if ((player.role === 'fascist' || player.role === 'hitler') &&
                liberalChoices > 0 && total >= 2) {
                this.insights.push({
                    icon: '🎭',
                    text: `${player.name} building liberal cover with policy choices`,
                    confidence: 0.75,
                    severity: 'medium',
                    type: 'strategic'
                });
            }
        });

        // Sort insights by confidence and severity
        this.insights.sort((a, b) => {
            const severityScore = { high: 3, medium: 2, low: 1 };
            const scoreA = severityScore[a.severity] * a.confidence;
            const scoreB = severityScore[b.severity] * b.confidence;
            return scoreB - scoreA;
        });

        // Limit to top 10 insights
        this.insights = this.insights.slice(0, 10);
    }

    /**
     * Get summary statistics
     */
    getSummaryStats() {
        const totalEvents = this.events.length;
        const speechEvents = this.events.filter(e => e.type === 'speech').length;
        const deceptiveEvents = this.events.filter(e =>
            e.type === 'speech' && (e.isDeceptive || e.data?.isDeceptive)
        ).length;

        const liberalPolicies = this.events.filter(e =>
            e.type === 'policy_enacted' && (e.data?.team === 'liberal' || e.data?.policy === 'liberal')
        ).length;

        const fascistPolicies = this.events.filter(e =>
            e.type === 'policy_enacted' && (e.data?.team === 'fascist' || e.data?.policy === 'fascist')
        ).length;

        return {
            totalEvents,
            speechEvents,
            deceptiveEvents,
            deceptionRate: speechEvents > 0 ? deceptiveEvents / speechEvents : 0,
            liberalPolicies,
            fascistPolicies,
            rounds: Math.max(...this.events.map(e => e.round || 1)),
            alivePlayers: this.players.filter(p => p.isAlive !== false).length
        };
    }

    /**
     * Get player-specific analytics
     */
    getPlayerAnalytics(playerId) {
        return {
            deception: this.deceptionScores[playerId] || {},
            behavior: this.behaviorPatterns[playerId] || {},
            player: this.players.find(p => p.id === playerId)
        };
    }

    /**
     * Export analytics data for research/papers
     */
    exportData() {
        return {
            summary: this.getSummaryStats(),
            trustNetwork: this.computeTrustNetwork(),
            deceptionScores: this.deceptionScores,
            behaviorPatterns: this.behaviorPatterns,
            insights: this.insights,
            events: this.events,
            players: this.players
        };
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = Analytics;
}
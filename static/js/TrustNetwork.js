/**
 * TrustNetwork - D3.js force-directed graph visualization for player trust relationships
 *
 * Features:
 * - Force-directed layout with draggable nodes
 * - Edge colors: Green (trust) vs Red (distrust)
 * - Edge thickness based on trust strength
 * - Node colors based on role (for spectator view)
 * - Interactive hover and click events
 * - Real-time updates as trust relationships change
 */

class TrustNetwork {
    constructor(container, options = {}) {
        this.container = container;
        this.options = {
            width: options.width || 300,
            height: options.height || 300,
            nodeRadius: options.nodeRadius || 20,
            linkStrength: options.linkStrength || 0.3,
            chargeStrength: options.chargeStrength || -200,
            minTrustThreshold: options.minTrustThreshold || 0.3,
            onNodeClick: options.onNodeClick || null,
            ...options
        };

        this.svg = null;
        this.simulation = null;
        this.players = [];
        this.trustData = {};

        this.init();
    }

    init() {
        // Check if container exists
        if (!this.container) {
            console.error('TrustNetwork: container element not found');
            return;
        }

        // Clear container
        this.container.innerHTML = '';

        // Create SVG
        this.svg = d3.select(this.container)
            .append('svg')
            .attr('width', this.options.width)
            .attr('height', this.options.height)
            .attr('viewBox', `0 0 ${this.options.width} ${this.options.height}`)
            .attr('class', 'trust-network-svg');

        // Create groups for links and nodes (order matters for z-index)
        this.linkGroup = this.svg.append('g').attr('class', 'links');
        this.nodeGroup = this.svg.append('g').attr('class', 'nodes');

        // Create force simulation
        this.simulation = d3.forceSimulation()
            .force('link', d3.forceLink()
                .id(d => d.id)
                .distance(d => this.getLinkDistance(d.trust)))
            .force('charge', d3.forceManyBody().strength(this.options.chargeStrength))
            .force('center', d3.forceCenter(this.options.width / 2, this.options.height / 2))
            .force('collision', d3.forceCollide().radius(this.options.nodeRadius + 5));
    }

    /**
     * Calculate link distance based on trust value
     * Higher trust = closer together, lower trust = farther apart
     */
    getLinkDistance(trust) {
        const baseDist = 80;
        // Inverse relationship: high trust = shorter distance
        return baseDist * (1.5 - Math.abs(trust));
    }

    /**
     * Update the trust network with new data
     * @param {Array} players - Array of player objects with id, name, role, avatar
     * @param {Object} trustData - Nested object: { playerId: { targetId: trustValue } }
     */
    update(players, trustData) {
        this.players = players;
        this.trustData = trustData;

        // Prepare nodes
        const nodes = players.map(p => ({
            id: p.id,
            name: p.name,
            role: p.role,
            avatar: p.avatar || p.name[0].toUpperCase(),
            isAlive: p.isAlive !== false
        }));

        // Prepare links (edges)
        const links = [];
        Object.entries(trustData).forEach(([from, targets]) => {
            Object.entries(targets || {}).forEach(([to, trustValue]) => {
                // Only show edges above minimum threshold
                if (Math.abs(trustValue) >= this.options.minTrustThreshold) {
                    links.push({
                        source: from,
                        target: to,
                        trust: trustValue
                    });
                }
            });
        });

        // Update simulation
        this.simulation.nodes(nodes);
        this.simulation.force('link').links(links);

        // Render
        this.render(nodes, links);

        // Restart simulation
        this.simulation.alpha(0.3).restart();
    }

    render(nodes, links) {
        // Update links
        const linkSelection = this.linkGroup
            .selectAll('line')
            .data(links, d => `${d.source.id || d.source}-${d.target.id || d.target}`);

        linkSelection.exit().remove();

        const linkEnter = linkSelection.enter()
            .append('line')
            .attr('class', 'trust-link')
            .attr('stroke-width', d => Math.abs(d.trust) * 4)
            .attr('stroke', d => this.getTrustColor(d.trust))
            .attr('opacity', 0.6);

        const linkUpdate = linkEnter.merge(linkSelection)
            .attr('stroke-width', d => Math.abs(d.trust) * 4)
            .attr('stroke', d => this.getTrustColor(d.trust));

        // Update nodes
        const nodeSelection = this.nodeGroup
            .selectAll('g')
            .data(nodes, d => d.id);

        nodeSelection.exit().remove();

        const nodeEnter = nodeSelection.enter()
            .append('g')
            .attr('class', 'trust-node')
            .call(this.createDragBehavior());

        // Node circle
        nodeEnter.append('circle')
            .attr('r', this.options.nodeRadius)
            .attr('fill', d => this.getRoleColor(d.role))
            .attr('stroke', '#fff')
            .attr('stroke-width', 2)
            .attr('opacity', d => d.isAlive ? 1 : 0.4);

        // Node avatar/text
        nodeEnter.append('text')
            .attr('text-anchor', 'middle')
            .attr('dy', '0.35em')
            .attr('font-size', '18px')
            .attr('font-weight', 'bold')
            .attr('fill', '#fff')
            .attr('pointer-events', 'none')
            .text(d => d.avatar);

        // Node label (name below circle)
        nodeEnter.append('text')
            .attr('class', 'node-label')
            .attr('text-anchor', 'middle')
            .attr('dy', this.options.nodeRadius + 15)
            .attr('font-size', '11px')
            .attr('fill', '#e2e8f0')
            .attr('pointer-events', 'none')
            .text(d => d.name);

        // Merge enter and update selections
        const nodeUpdate = nodeEnter.merge(nodeSelection);

        // Update circle colors and opacity
        nodeUpdate.select('circle')
            .attr('fill', d => this.getRoleColor(d.role))
            .attr('opacity', d => d.isAlive ? 1 : 0.4);

        // Update avatar
        nodeUpdate.select('text')
            .text(d => d.avatar);

        // Update label
        nodeUpdate.select('.node-label')
            .text(d => d.name);

        // Add click handler
        if (this.options.onNodeClick) {
            nodeUpdate
                .style('cursor', 'pointer')
                .on('click', (event, d) => {
                    event.stopPropagation();
                    this.options.onNodeClick(d);
                });
        }

        // Add hover effects
        nodeUpdate
            .on('mouseenter', (event, d) => {
                d3.select(event.currentTarget)
                    .select('circle')
                    .transition()
                    .duration(200)
                    .attr('r', this.options.nodeRadius * 1.2)
                    .attr('stroke-width', 3);
            })
            .on('mouseleave', (event, d) => {
                d3.select(event.currentTarget)
                    .select('circle')
                    .transition()
                    .duration(200)
                    .attr('r', this.options.nodeRadius)
                    .attr('stroke-width', 2);
            });

        // Update simulation tick handler
        this.simulation.on('tick', () => {
            linkUpdate
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);

            nodeUpdate.attr('transform', d => `translate(${d.x},${d.y})`);
        });
    }

    getTrustColor(trustValue) {
        if (trustValue > 0) {
            // Positive trust: green gradient
            const intensity = Math.min(trustValue, 1);
            return d3.interpolateRgb('#10b981', '#059669')(intensity);
        } else {
            // Negative trust: red gradient
            const intensity = Math.min(Math.abs(trustValue), 1);
            return d3.interpolateRgb('#ef4444', '#dc2626')(intensity);
        }
    }

    getRoleColor(role) {
        const colors = {
            'liberal': '#3b82f6',
            'fascist': '#ef4444',
            'hitler': '#1e1e1e'
        };
        return colors[role] || '#6b7280';
    }

    createDragBehavior() {
        const simulation = this.simulation;

        function dragstarted(event, d) {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }

        function dragged(event, d) {
            d.fx = event.x;
            d.fy = event.y;
        }

        function dragended(event, d) {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }

        return d3.drag()
            .on('start', dragstarted)
            .on('drag', dragged)
            .on('end', dragended);
    }

    /**
     * Highlight connections for a specific player
     */
    highlightPlayer(playerId) {
        // Dim all links
        this.linkGroup.selectAll('line')
            .transition()
            .duration(300)
            .attr('opacity', d => {
                if (d.source.id === playerId || d.target.id === playerId) {
                    return 0.9;
                }
                return 0.2;
            })
            .attr('stroke-width', d => {
                if (d.source.id === playerId || d.target.id === playerId) {
                    return Math.abs(d.trust) * 5;
                }
                return Math.abs(d.trust) * 2;
            });

        // Dim all nodes
        this.nodeGroup.selectAll('g')
            .transition()
            .duration(300)
            .attr('opacity', d => {
                if (d.id === playerId) return 1;

                // Check if connected to highlighted player
                const connected = this.isConnected(playerId, d.id);
                return connected ? 1 : 0.3;
            });
    }

    /**
     * Reset highlighting
     */
    resetHighlight() {
        this.linkGroup.selectAll('line')
            .transition()
            .duration(300)
            .attr('opacity', 0.6)
            .attr('stroke-width', d => Math.abs(d.trust) * 4);

        this.nodeGroup.selectAll('g')
            .transition()
            .duration(300)
            .attr('opacity', 1);
    }

    isConnected(playerId1, playerId2) {
        return !!(this.trustData[playerId1]?.[playerId2] ||
                  this.trustData[playerId2]?.[playerId1]);
    }

    /**
     * Export current network as image data URL
     */
    exportAsImage() {
        const svgElement = this.svg.node();
        const svgString = new XMLSerializer().serializeToString(svgElement);
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        const img = new Image();

        canvas.width = this.options.width;
        canvas.height = this.options.height;

        return new Promise((resolve) => {
            img.onload = () => {
                ctx.drawImage(img, 0, 0);
                resolve(canvas.toDataURL('image/png'));
            };
            img.src = 'data:image/svg+xml;base64,' + btoa(unescape(encodeURIComponent(svgString)));
        });
    }

    destroy() {
        if (this.simulation) {
            this.simulation.stop();
        }
        if (this.container) {
            this.container.innerHTML = '';
        }
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = TrustNetwork;
}
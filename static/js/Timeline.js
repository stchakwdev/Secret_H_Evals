/**
 * Timeline.js - Timeline visualization and controls component
 * Phase 5: Timeline & Replay
 *
 * Renders timeline with playback controls and event markers
 * Author: Samuel Chakwera (stchakdev)
 */

class Timeline {
    /**
     * Create timeline component
     * @param {ReplayManager} replayManager - Replay manager instance
     * @param {HTMLElement} container - Container element
     */
    constructor(replayManager, container) {
        this.replayManager = replayManager;
        this.container = container;
        this.elements = {};

        this.render();
        this.attachListeners();
    }

    /**
     * Render timeline component
     */
    render() {
        if (!this.container) {
            console.error('Timeline: container element not found');
            return;
        }

        this.container.innerHTML = `
            <div class="timeline-wrapper">
                <!-- Playback Controls -->
                <div class="playback-controls">
                    <button class="control-btn" id="jumpToStart" title="Jump to Start">
                        ⏮
                    </button>
                    <button class="control-btn" id="stepBackward" title="Step Backward">
                        ◀
                    </button>
                    <button class="control-btn play" id="togglePlay" title="Play/Pause">
                        ▶
                    </button>
                    <button class="control-btn" id="stepForward" title="Step Forward">
                        ▶
                    </button>
                    <button class="control-btn" id="jumpToEnd" title="Jump to End">
                        ⏭
                    </button>

                    <select class="speed-control" id="speedControl">
                        <option value="0.5">0.5×</option>
                        <option value="1" selected>1×</option>
                        <option value="2">2×</option>
                        <option value="5">5×</option>
                    </select>

                    <div class="timeline-info">
                        <span id="eventCounter">Event 0 / 0</span>
                        <span id="roundInfo">Round 0</span>
                    </div>
                </div>

                <!-- Timeline Bar -->
                <div class="timeline-container">
                    <div class="timeline-track">
                        <input
                            type="range"
                            class="timeline-slider"
                            id="timelineSlider"
                            min="0"
                            max="0"
                            value="0"
                        />
                        <div class="timeline-markers" id="timelineMarkers"></div>
                        <div class="timeline-progress" id="timelineProgress"></div>
                    </div>
                </div>
            </div>
        `;

        // Store element references
        this.elements = {
            jumpToStart: this.container.querySelector('#jumpToStart'),
            stepBackward: this.container.querySelector('#stepBackward'),
            togglePlay: this.container.querySelector('#togglePlay'),
            stepForward: this.container.querySelector('#stepForward'),
            jumpToEnd: this.container.querySelector('#jumpToEnd'),
            speedControl: this.container.querySelector('#speedControl'),
            timelineSlider: this.container.querySelector('#timelineSlider'),
            timelineMarkers: this.container.querySelector('#timelineMarkers'),
            timelineProgress: this.container.querySelector('#timelineProgress'),
            eventCounter: this.container.querySelector('#eventCounter'),
            roundInfo: this.container.querySelector('#roundInfo')
        };

        this.updateDisplay();
    }

    /**
     * Attach event listeners
     */
    attachListeners() {
        // Playback controls
        this.elements.jumpToStart.addEventListener('click', () => {
            this.replayManager.jumpToStart();
        });

        this.elements.stepBackward.addEventListener('click', () => {
            this.replayManager.stepBackward();
        });

        this.elements.togglePlay.addEventListener('click', () => {
            this.replayManager.togglePlay();
        });

        this.elements.stepForward.addEventListener('click', () => {
            this.replayManager.stepForward();
        });

        this.elements.jumpToEnd.addEventListener('click', () => {
            this.replayManager.jumpToEnd();
        });

        // Speed control
        this.elements.speedControl.addEventListener('change', (e) => {
            const speed = parseFloat(e.target.value);
            this.replayManager.setSpeed(speed);
        });

        // Timeline slider
        this.elements.timelineSlider.addEventListener('input', (e) => {
            const index = parseInt(e.target.value);
            this.replayManager.seekToIndex(index);
        });

        // Listen to replay manager events
        this.replayManager.on('playbackChange', (state) => {
            this.updatePlayButton(state.isPlaying);
        });

        this.replayManager.on('indexChange', (index) => {
            this.updateDisplay();
        });

        this.replayManager.on('stateChange', (state) => {
            this.updateStateInfo(state);
        });
    }

    /**
     * Update play button state
     * @param {boolean} isPlaying - Whether playback is active
     */
    updatePlayButton(isPlaying) {
        this.elements.togglePlay.textContent = isPlaying ? '⏸' : '▶';
        this.elements.togglePlay.title = isPlaying ? 'Pause' : 'Play';

        if (isPlaying) {
            this.elements.togglePlay.classList.add('playing');
        } else {
            this.elements.togglePlay.classList.remove('playing');
        }
    }

    /**
     * Update timeline display
     */
    updateDisplay() {
        const state = this.replayManager.getPlaybackState();
        const totalEvents = state.totalEvents;
        const currentIndex = state.currentIndex;

        // Update slider
        this.elements.timelineSlider.max = Math.max(0, totalEvents - 1);
        this.elements.timelineSlider.value = currentIndex;

        // Update event counter
        this.elements.eventCounter.textContent =
            `Event ${currentIndex + 1} / ${totalEvents}`;

        // Update progress bar
        const progress = totalEvents > 0 ? ((currentIndex + 1) / totalEvents) * 100 : 0;
        this.elements.timelineProgress.style.width = `${progress}%`;

        // Update markers
        this.updateMarkers();
    }

    /**
     * Update state-related info
     * @param {Object} state - Current game state
     */
    updateStateInfo(state) {
        this.elements.roundInfo.textContent = `Round ${state.round || 0}`;
    }

    /**
     * Update timeline markers for key moments
     */
    updateMarkers() {
        const keyMoments = this.replayManager.getKeyMoments();
        const totalEvents = this.replayManager.events.length;

        if (totalEvents === 0) {
            this.elements.timelineMarkers.innerHTML = '';
            return;
        }

        this.elements.timelineMarkers.innerHTML = '';

        keyMoments.forEach(moment => {
            const marker = document.createElement('div');
            marker.className = `timeline-marker ${moment.type} ${moment.severity}`;
            marker.style.left = `${(moment.index / totalEvents) * 100}%`;
            marker.title = moment.description;

            // Add icon
            const icon = document.createElement('span');
            icon.className = 'marker-icon';
            icon.textContent = moment.icon || '•';
            marker.appendChild(icon);

            // Click to jump to moment
            marker.addEventListener('click', () => {
                this.replayManager.seekToIndex(moment.index);
            });

            this.elements.timelineMarkers.appendChild(marker);
        });
    }

    /**
     * Update when new events are added
     */
    refresh() {
        this.updateDisplay();
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = Timeline;
}
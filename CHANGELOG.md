# Changelog

All notable changes to the Secret Hitler LLM Evaluation Framework will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [1.2.0] - 2025-11-27 - Research-Grade Enhancements

### Added

**Phase 1: Interactive Visualization Dashboard**
- **Plotly Dash Dashboard** (`dashboard/`): Complete interactive analytics interface
  - `app.py`: Main application entry point with multi-tab layout
  - `layouts.py`: Game Overview, Deception Analysis, Decision Patterns, Cost Analytics tabs
  - `callbacks.py`: Interactive filtering and chart updates
  - `data_loader.py`: Efficient database query interface
- Trust network visualization with force-directed graphs
- Real-time filtering by game, player, model, and date range
- Export capabilities for publication-ready figures
- Launch command: `python -m dashboard.app --port 8050`

**Phase 2: Research Rigor Components**
- **Statistical Hypothesis Testing** (`analytics/hypothesis_testing.py`):
  - `test_model_win_rates()`: Chi-square test for model comparison
  - `test_deception_by_role()`: Fisher's exact test for role-based deception
  - `test_game_length_deception_correlation()`: Spearman correlation analysis
  - `test_deception_by_decision_type()`: Kruskal-Wallis test across decision types
  - Effect size calculations: Cohen's d, Cramér's V, odds ratios
  - Wilson score confidence intervals for proportions
  - Multiple comparison correction: Bonferroni, Holm, FDR methods
  - `run_hypothesis_battery()`: Automated test suite with correction
  - `generate_hypothesis_report()`: Markdown report generation

- **Temporal Analysis** (`analytics/temporal_analysis.py`):
  - `segment_game_into_phases()`: Early/mid/late game phase detection
  - `detect_turning_points()`: Signal processing with Gaussian smoothing
  - `calculate_trust_trajectory()`: Rolling trust score calculation
  - `calculate_deception_trajectory()`: Deception rate over time
  - `detect_momentum_shifts()`: Policy progression analysis
  - `classify_deception_trend()`: Trend classification (increasing/decreasing/stable/volatile)
  - `GamePhase`, `TurningPoint`, `TemporalMetrics` dataclasses

- **Belief Calibration** (`analytics/belief_calibration.py`):
  - `calculate_brier_score()`: Probability prediction accuracy
  - `calculate_expected_calibration_error()`: ECE with reliability diagrams
  - `calculate_maximum_calibration_error()`: Worst-case calibration
  - `calculate_overconfidence_rate()`: High-confidence error rate
  - `calculate_underconfidence_rate()`: Low-confidence correct rate
  - `calculate_kl_divergence_from_uniform()`: Distribution divergence
  - `analyze_player_calibration()`: Complete player calibration metrics
  - `CalibrationMetrics`, `BeliefSnapshot` dataclasses

- **Enhanced Inspect AI Integration** (`evaluation/inspect_adapter.py`):
  - `export_batch_analysis()`: Cross-game hypothesis testing
  - `_run_batch_hypothesis_tests()`: Automated statistical analysis
  - `export_prompts_for_reproducibility()`: Full prompt/response export
  - `_calculate_temporal_metrics()`: Per-game temporal analysis
  - `_calculate_calibration_metrics()`: Per-game calibration analysis
  - `run_batch_analysis()`: Helper function for batch processing

- **Prompt/Response Logging**:
  - Extended `APIRequest` dataclass with `prompt_text`, `response_text`, `temperature`, `max_tokens`
  - `get_all_prompts()`, `get_prompts_by_player()`, `get_prompts_by_decision_type()` methods
  - `calculate_prompt_hash()`: SHA-256 hashing for deduplication
  - `log_prompt_response()`: Database logging for reproducibility
  - `log_batch_prompts()`: Efficient batch insertion

- **Database Schema Updates** (`evaluation/database_schema.py`):
  - New `prompts` table for complete prompt/response storage
  - Prompt hash indexing for efficient lookups
  - WAL mode support for concurrent access at scale
  - `insert_prompt()`, `get_prompts()`, `get_prompt_by_hash()` methods

### Changed
- `agents/openrouter_client.py`: Extended to capture full prompts and responses
- `game_logging/game_logger.py`: Added prompt logging methods
- `evaluation/inspect_adapter.py`: Enhanced with analytics integration

### Verified
- All Phase 2 components pass comprehensive test suite (`test_phase2.py`)
- Statistical tests produce valid results with correct significance levels
- Temporal analysis correctly segments games and detects patterns
- Belief calibration metrics match expected ranges
- Batch analysis successfully processes multiple games

## [1.1.1] - 2025-11-01

### Fixed
- **Batch Evaluation**: Winner aggregation display (singular/plural mismatch in run_game.py)
  - Changed winner string comparison from plural ('liberals'/'fascists') to singular ('liberal'/'fascist')
  - Batch summaries now correctly display win counts
- **Log Path Resolution**: Logs created in parent repository instead of llm-game-engine/logs/
  - Fixed GameLogger initialization to use absolute paths (core/game_manager.py:67-73)
  - Fixed batch metadata log directory path (run_game.py:105, 109)
  - Fixed progress tracker log references (check_batch_progress.py:17, 123, 265)
  - All logs now correctly created in llm-game-engine/logs/ regardless of working directory

### Added
- **Batch Progress Tracker**: Real-time batch monitoring tool (check_batch_progress.py)
  - Auto-detects running batches from metadata
  - Displays game state, policy progression, and timing statistics
  - Watch mode with configurable refresh interval
  - Example: `python check_batch_progress.py --watch`

### Verified
- 3-game verification batch completed successfully (batch-20251101-104548)
- All logs created in correct location (llm-game-engine/logs/)
- Progress tracker functional with real-time game state display
- Database logging operational

## [1.1.0] - 2025-10-27

### Added
- **Inspect AI Integration**: Complete integration with Inspect AI for standardized evaluation format
  - SQLite database schema for structured game storage (`evaluation/database_schema.py`)
  - Inspect format adapter for automatic conversion (`evaluation/inspect_adapter.py`)
  - Export script for batch conversion to Inspect format (`scripts/export_to_inspect.py`)
  - Analysis script with statistical evaluation (`scripts/analyze_with_inspect.py`)
  - Historical log migration tool (`scripts/migrate_historical.py`)
  - Report generation script (`scripts/generate_inspect_report.py`)
- **CLI Entry Point**: New `run_game.py` with comprehensive command-line interface
  - `--enable-db-logging` flag for database integration
  - `--batch` mode for running multiple games
  - `--players` option for 5-10 player games
  - `--model` option for selecting LLM models
- **Documentation**:
  - INSPECT_INTEGRATION.md with technical implementation details
  - TEST_RESULTS.md with end-to-end verification
  - evaluation/README.md with usage guide
  - CHANGELOG.md for version tracking

### Fixed
- **Critical**: JSON serialization error for PlayerType Enum in database logging
  - Extended `DateTimeEncoder` to handle all Enum types
  - Fixed by converting Enum values to strings via `.value` property
- **Critical**: Policy deck reshuffling logic edge cases
  - Added safety checks in `draw_policies()` to handle low card counts
  - Improved fallback policy selection with graceful degradation
  - Fixed index out of range errors when deck runs out

### Changed
- Updated README.md with new features and workflows
- Enhanced .gitignore to exclude generated data and reports
- Updated game_logger.py to support optional database logging

### Verified
- End-to-end pipeline tested with real AI game data
- Database logging verified with 73 player decisions
- Export to Inspect format confirmed working
- Analysis scripts generate correct statistical reports
- No serialization errors in production use

## [1.0.0] - 2025-09-24

### Added
- Initial release of Secret Hitler LLM Evaluation Framework
- Complete Secret Hitler game mechanics implementation
- OpenRouter API integration with multiple LLM support
- Multi-level logging system (public, game state, individual reasoning)
- Cost tracking and optimization
- Batch experiment runner
- Basic analytics and visualization tools
- Comprehensive prompt engineering for strategic gameplay

### Supported Models
- DeepSeek V3.2 Exp (default)
- GPT-4 series
- Claude 3 series
- Gemini series
- Llama 3 series

[1.2.0]: https://github.com/stchakwdev/Secret_H_Evals/compare/v1.1.1...v1.2.0
[1.1.1]: https://github.com/stchakwdev/Secret_H_Evals/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/stchakwdev/Secret_H_Evals/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/stchakwdev/Secret_H_Evals/releases/tag/v1.0.0

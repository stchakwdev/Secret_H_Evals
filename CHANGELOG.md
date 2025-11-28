# Changelog

All notable changes to the Secret Hitler LLM Evaluation Framework will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [1.5.0] - 2025-11-28 - Full Analytics Integration & Visual Refresh

### Added

**Phase 5: Complete Inspect AI Analytics Integration**

- **Enhanced Inspect Adapter** (`evaluation/inspect_adapter.py`):
  - Full integration with all 6 analytics modules
  - Coalition detection via Louvain community detection algorithm
  - Model comparison with pairwise chi-square tests and Elo ratings
  - Complete temporal analysis with game phase segmentation (early/mid/late)
  - Full belief calibration metrics (Brier score, ECE, MCE, log loss)
  - Hypothesis testing with proper `HypothesisTestResult` objects
  - Effect sizes: Cohen's d, Cohen's h, Cramér's V, odds ratio
  - Multiple comparison correction (Bonferroni, Holm-Bonferroni)
  - Wilson score confidence intervals for all proportions

- **New Batch Analysis Methods**:
  - `_calculate_temporal_metrics_full()`: Complete temporal analysis per game
  - `_calculate_calibration_metrics_full()`: Full calibration metrics per player
  - `_calculate_hypothesis_metrics_full()`: Role-based and decision-type tests
  - `_calculate_coalition_metrics()`: Community detection and alignment networks
  - `_run_batch_hypothesis_tests_full()`: Cross-game statistical analysis
  - `_calculate_model_comparison()`: Pairwise model comparison with corrections
  - `_calculate_batch_coalition_metrics()`: Aggregate coalition statistics

- **README Visual Showcase**:
  - Added 3 new visualizations to showcase section
  - Game Outcomes Distribution (`docs/images/game_outcomes.png`)
  - Model Comparison (`docs/images/model_comparison.png`)
  - Cost Analytics Dashboard (`docs/images/cost_dashboard.png`)
  - Regenerated Policy Progression Timeline with latest data

### Changed

- `evaluation/inspect_adapter.py`: Complete rewrite of analytics integration
  - Replaced stub implementations with full analytics calls
  - Added proper dataclass serialization for JSON export
  - Enhanced batch analysis with cross-game metrics
  - Improved error handling with graceful degradation

### Verified

- All 6 analytics modules properly wired to Inspect adapter
- Coalition detection produces valid community structures
- Model comparison generates valid statistics
- Temporal analysis correctly segments games into phases
- Calibration metrics match expected ranges for LLM predictions

## [1.4.0] - 2025-11-28 - Multi-Model Comparison Framework

### Added

**Phase 4: Multi-Model Comparison Framework**

- **Model Comparison Config** (`config/model_comparison_config.py`):
  - `ModelConfig` dataclass with 2025 OpenRouter pricing
  - `ModelTier` enum (FREE, BUDGET, PREMIUM)
  - `ModelProvider` enum (xAI, Meta, Google, DeepSeek, OpenAI, etc.)
  - `ModelArchitecture` enum (MOE, DENSE)
  - `BatchConfig` for comparison batch configuration
  - `ComparisonGroup` for hypothesis testing groups
  - 11 model definitions:
    - FREE tier (9): Grok 4.1 Fast, GLM 4.5 Air, Bert-Nebulon Alpha, Llama 4 Maverick, Llama 4 Scout, DeepSeek R1, Gemini 2.0 Flash Exp, Mistral Small 3.1, Optimus Alpha
    - PAID tier (2): GPT-5 Nano, DeepSeek V3
  - Predefined comparison groups: Chinese vs Western, Reasoning vs Standard, Free vs Paid, MoE vs Dense, Cloaked models
  - Cost estimation functions: `estimate_game_cost()`, `estimate_batch_cost()`
  - `print_cost_summary()` for batch cost breakdown

- **Model Comparator** (`experiments/model_comparator.py`):
  - `ModelComparator` class for orchestrating multi-model experiments
  - `ModelResult` dataclass for per-model statistics tracking
  - `ComparisonProgress` dataclass for batch progress management
  - `ComparisonStatus` enum (PENDING, RUNNING, COMPLETED, FAILED, CANCELLED)
  - `run_comparison()` async method for full batch execution
  - Progress persistence to JSON for resumable experiments
  - `generate_latex_table()` for formatted output
  - `_calculate_rankings()` for model ranking by win rates
  - `_analyze_comparison_groups()` for hypothesis group analysis
  - `print_summary()` for console output

- **Model Comparison Analytics** (`analytics/model_comparison.py`):
  - `ModelStats` dataclass for per-model statistics
  - `ComparisonResult` dataclass with effect sizes and confidence intervals
  - `WinRateCI` dataclass using Wilson score intervals
  - Statistical functions:
    - `compare_win_rates()`: Chi-square test with effect sizes
    - `chi_square_win_rates()`: Raw chi-square calculation
    - `cohens_h()`: Effect size for proportions
    - `cramers_v()`: Association strength
    - `odds_ratio()`: Odds ratio with confidence interval
    - `calculate_wilson_ci()`: Wilson score confidence intervals
  - Multiple comparison correction:
    - `bonferroni_correction()`
    - `holm_bonferroni_correction()`
  - `calculate_elo_ratings()`: Elo rating from pairwise results
  - Output generation:
    - `generate_latex_comparison_table()`
    - `generate_pairwise_significance_table()`
    - `generate_markdown_report()`

- **CLI Enhancements** (`run_game.py`):
  - `--model-comparison` flag for multi-model experiments
  - `--games-per-model` for games per model (default: 500)
  - `--compare-models` for specific model selection (comma-separated)
  - `--free-only` for free models only (9 models, $0)
  - `--paid-only` for paid models only (2 models)
  - `--list-models` to display all available models with pricing

- **Parallel Runner Model Rotation** (`experiments/parallel_runner.py`):
  - `models` parameter for multi-model rotation
  - `_get_next_model()` for round-robin model selection
  - `--models` CLI argument for comma-separated model list
  - `--free-only` and `--paid-only` CLI flags
  - `--list-models` to show available models

### Changed
- `config/openrouter_config.py`: Added Phase 4 model definitions with 2025 pricing
  - Added FREE tier models (9 models)
  - Added BUDGET tier models (2 models)
  - Added PREMIUM tier models (reserved: Claude 4.5 Sonnet, DeepSeek R1 paid)
  - Added `PHASE4_FREE_MODELS`, `PHASE4_PAID_MODELS`, `PHASE4_ALL_MODELS` lists

### Cost Summary
- 11 models × 500 games = 5,500 total games
- FREE tier: 9 models × 500 games = 4,500 games @ $0
- PAID tier: 2 models × 500 games = 1,000 games @ ~$54
- Total estimated cost: ~$55 (budget remaining: ~$445 for future runs)

### Verified
- All Phase 4 components import successfully
- `--list-models` displays all 11 models with pricing
- Model comparison config correctly calculates costs
- Statistical comparison functions produce valid results

## [1.3.0] - 2025-11-27 - Performance for Scale

### Added

**Phase 3: Performance Optimizations for 5000+ Games**

- **Parallel Batch Runner** (`experiments/parallel_runner.py`):
  - `ParallelBatchRunner` class with configurable concurrency
  - `RateLimiter` using token bucket algorithm for API quota compliance
  - `BatchProgress` dataclass for comprehensive progress tracking
  - `GameResult` and `GameStatus` for individual game state management
  - Automatic retry with exponential backoff (configurable max retries)
  - Progress persistence to JSON for crash recovery and batch resumption
  - Graceful shutdown handling with SIGINT/SIGTERM signals
  - `run_parallel_batch()` convenience function for easy usage

- **High-Performance Database** (`evaluation/database_scale.py`):
  - `ConnectionPool` class with configurable pool size and overflow
  - Thread-safe connection management with timeout handling
  - Automatic connection recycling after configurable time
  - `ScaleDatabaseManager` with batch insert operations:
    - `batch_insert_games()`: Insert multiple games in single transaction
    - `batch_insert_decisions()`: Bulk decision insertion
    - `batch_insert_prompts()`: Bulk prompt insertion
  - Streaming query methods for memory efficiency:
    - `stream_games()`: Iterate games in configurable batches
    - `stream_decisions()`: Stream decisions without loading all into memory
    - `stream_aggregate()`: Stream aggregated results
  - Statistics caching with configurable TTL
  - Progress persistence: `save_batch_progress()`, `load_batch_progress()`
  - Database maintenance: `optimize()`, `vacuum()`

- **Streaming Analytics** (`analytics/streaming_stats.py`):
  - `WelfordAccumulator`: Online mean/variance using Welford's algorithm
  - `CountAccumulator`: Categorical counting with proportions
  - `HistogramAccumulator`: Online histogram computation with fixed bins
  - `StreamingGameStats`: Comprehensive game statistics aggregation
  - `StreamingAnalyzer`: High-level interface for streaming analysis
  - All accumulators support `merge()` for parallel processing results
  - Memory-efficient: processes data in single pass

- **Monitoring & Metrics** (`experiments/monitoring.py`):
  - Prometheus-compatible metric types:
    - `Counter`: Monotonically increasing values with labels
    - `Gauge`: Values that can go up and down
    - `Histogram`: Distribution tracking with configurable buckets
  - `MetricsCollector` class with pre-defined game metrics:
    - `games_total`, `games_in_progress`, `game_duration`
    - `api_requests_total`, `api_latency`, `api_cost`
    - `connection_pool_size`, `memory_usage_bytes`
    - `batch_progress`, `batch_errors`
  - `AlertRule` class for anomaly detection
  - Rate calculations: `get_rate()`, `get_rolling_average()`
  - Export methods: `to_prometheus()`, `to_json()`, `export_to_file()`
  - Periodic export in background thread

- **CLI Enhancements** (`run_game.py`):
  - `--parallel` flag for parallel batch execution
  - `--concurrency` / `-c` for number of concurrent games
  - `--rate-limit` for API requests per minute
  - `--resume` for continuing interrupted batches

### Changed
- `run_game.py`: Added parallel execution mode with progress reporting

### Verified
- All Phase 3 components pass comprehensive test suite (`test_phase3.py`)
- Connection pooling working with concurrent access
- Batch inserts 100x faster than individual inserts
- Streaming queries process large datasets without memory issues
- Prometheus metrics export in correct format
- Welford's algorithm produces correct mean/variance

## [1.2.0] - 2025-11-27 - Advanced Analytics

### Added

**Phase 1: Interactive Visualization Dashboard**
- **Plotly Dash Dashboard** (`dashboard/`): Complete interactive analytics interface
  - `app.py`: Main application entry point with multi-tab layout
  - `layouts.py`: Game Overview, Deception Analysis, Decision Patterns, Cost Analytics tabs
  - `callbacks.py`: Interactive filtering and chart updates
  - `data_loader.py`: Efficient database query interface
- Trust network visualization with force-directed graphs
- Real-time filtering by game, player, model, and date range
- Export capabilities for figures and charts
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

[1.5.0]: https://github.com/stchakwdev/Secret_H_Evals/compare/v1.4.0...v1.5.0
[1.4.0]: https://github.com/stchakwdev/Secret_H_Evals/compare/v1.3.0...v1.4.0
[1.3.0]: https://github.com/stchakwdev/Secret_H_Evals/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/stchakwdev/Secret_H_Evals/compare/v1.1.1...v1.2.0
[1.1.1]: https://github.com/stchakwdev/Secret_H_Evals/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/stchakwdev/Secret_H_Evals/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/stchakwdev/Secret_H_Evals/releases/tag/v1.0.0

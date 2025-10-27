# Changelog

All notable changes to the Secret Hitler LLM Evaluation Framework will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

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

[1.1.0]: https://github.com/stchakwdev/Secret_H_Evals/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/stchakwdev/Secret_H_Evals/releases/tag/v1.0.0

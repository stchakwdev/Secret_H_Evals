"""
Converts Secret Hitler game logs to Inspect AI evaluation format.
Maintains compatibility with existing logging system.

Enhanced with research-grade metrics:
- Statistical hypothesis testing results
- Temporal analysis (phases, turning points)
- Belief calibration metrics
- Prompt reproducibility data

Author: Samuel Chakwera (stchakdev)
"""
from datetime import datetime
from pathlib import Path
import json
import logging
from typing import Dict, List, Any, Optional
import sys

# Add parent for analytics imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from .database_schema import DatabaseManager

logger = logging.getLogger(__name__)

# Try to import analytics modules (optional)
try:
    from analytics.hypothesis_testing import (
        test_deception_by_role,
        test_game_length_deception_correlation,
        test_deception_by_decision_type,
        HypothesisTestResult
    )
    HYPOTHESIS_TESTING_AVAILABLE = True
except ImportError:
    HYPOTHESIS_TESTING_AVAILABLE = False
    logger.debug("Hypothesis testing module not available")

try:
    from analytics.temporal_analysis import (
        analyze_game_temporal_dynamics,
        TemporalMetrics
    )
    TEMPORAL_ANALYSIS_AVAILABLE = True
except ImportError:
    TEMPORAL_ANALYSIS_AVAILABLE = False
    logger.debug("Temporal analysis module not available")

try:
    from analytics.belief_calibration import (
        analyze_player_calibration,
        CalibrationMetrics,
        BeliefSnapshot
    )
    CALIBRATION_AVAILABLE = True
except ImportError:
    CALIBRATION_AVAILABLE = False
    logger.debug("Belief calibration module not available")


class SecretHitlerInspectAdapter:
    """Converts Secret Hitler games to Inspect format for analysis."""

    def __init__(self, game_logs_dir: str = "./logs", db_path: str = "./data/games.db"):
        self.game_logs_dir = Path(game_logs_dir)
        self.inspect_output_dir = Path("./data/inspect_logs")
        self.inspect_output_dir.mkdir(exist_ok=True, parents=True)
        self.db = DatabaseManager(db_path)

    def convert_game_to_inspect(
        self,
        game_id: str,
        game_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Convert a single Secret Hitler game to Inspect-compatible dict format.

        Args:
            game_id: Unique game identifier
            game_data: Game log structure from GameLogger

        Returns:
            Dictionary in Inspect-compatible format
        """

        # Extract metadata
        metadata = {
            "game_id": game_id,
            "player_count": game_data.get("player_count", 0),
            "models_used": self._extract_models(game_data),
            "game_duration_turns": game_data.get("total_actions", 0),
            "winner": game_data.get("winner"),
            "winning_team": game_data.get("winning_team"),
            "win_condition": game_data.get("win_condition"),
            "policies_enacted": {
                "liberal": game_data.get("liberal_policies", 0),
                "fascist": game_data.get("fascist_policies", 0)
            },
            "framework": "secret-hitler-llm-eval",
            "evaluation_type": "multi-agent-social-deduction"
        }

        # Convert player decisions to samples
        samples = self._convert_all_decisions_to_dict(game_id, game_data)

        # Calculate aggregate metrics
        results = self._calculate_results_dict(game_data)

        # Create Inspect-compatible log structure
        return {
            "version": 2,
            "status": "success",
            "eval": {
                "task": "secret_hitler",
                "task_id": game_id,
                "model": ", ".join(metadata["models_used"]),
                "model_args": {}
            },
            "plan": {
                "steps": []
            },
            "results": results,
            "stats": {
                "completed_samples": len(samples),
                "total_samples": len(samples)
            },
            "samples": samples,
            "metadata": metadata,
            "created": game_data.get("timestamp", datetime.now().isoformat())
        }

    def _convert_all_decisions_to_dict(
        self,
        game_id: str,
        game_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Convert all player decisions to Inspect-compatible sample dicts."""
        samples = []

        # Try to get decisions from database first
        try:
            db_decisions = self.db.get_player_decisions(game_id)
            if db_decisions:
                for decision in db_decisions:
                    sample = self._decision_to_sample_dict(decision)
                    if sample:
                        samples.append(sample)
                return samples
        except Exception as e:
            logger.debug(f"Could not load decisions from database: {e}")

        # Fallback: parse from game_data if available
        if "player_metrics" in game_data:
            samples = self._parse_decisions_from_metrics_dict(game_data)

        return samples

    def _decision_to_sample_dict(self, decision: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Convert a single decision record to Inspect-compatible sample dict."""
        try:
            # Parse beliefs JSON if available
            beliefs = {}
            if decision.get("beliefs_json"):
                try:
                    beliefs = json.loads(decision["beliefs_json"])
                except json.JSONDecodeError:
                    pass

            return {
                "id": f"{decision['player_id']}_turn_{decision.get('turn_number', 0)}",
                "epoch": decision.get("turn_number", 0),
                "input": f"Phase: {decision['decision_type']}\nPlayer: {decision['player_name']}",
                "target": "unknown",
                "output": {
                    "model": "secret_hitler_agent",
                    "choices": [],
                    "completion": decision.get("action", "")
                },
                "metadata": {
                    "player_id": decision["player_id"],
                    "player_name": decision["player_name"],
                    "decision_type": decision["decision_type"],
                    "reasoning": decision.get("reasoning", ""),
                    "public_statement": decision.get("public_statement", ""),
                    "is_deception": bool(decision.get("is_deception", 0)),
                    "deception_score": decision.get("deception_score", 0.0),
                    "beliefs": beliefs,
                    "confidence": decision.get("confidence"),
                    "timestamp": decision.get("timestamp", "")
                }
            }
        except Exception as e:
            logger.error(f"Failed to convert decision to sample: {e}")
            return None

    def _parse_decisions_from_metrics_dict(self, game_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fallback: Parse decisions from game_data metrics structure as dicts."""
        samples = []

        player_metrics = game_data.get("player_metrics", {})
        for player_id, metrics in player_metrics.items():
            # Create a sample for aggregate player behavior
            sample = {
                "id": f"{player_id}_aggregate",
                "epoch": 0,
                "input": f"Player {player_id} aggregate behavior",
                "target": "aggregate_analysis",
                "output": {
                    "model": "secret_hitler_agent",
                    "choices": [],
                    "completion": f"Total actions: {metrics.get('total_actions', 0)}"
                },
                "metadata": {
                    "player_id": player_id,
                    "total_actions": metrics.get("total_actions", 0),
                    "reasoning_entries": metrics.get("reasoning_entries", 0),
                    "deception_count": metrics.get("deception_count", 0),
                    "deception_frequency": metrics.get("deception_frequency", 0.0),
                    "api_cost": metrics.get("api_cost", 0.0),
                    "api_requests": metrics.get("api_requests", 0)
                }
            }
            samples.append(sample)

        return samples

    def _extract_models(self, game_data: Dict[str, Any]) -> List[str]:
        """Extract unique model names from game data."""
        models = set()

        # Try from player configs
        if "players" in game_data and isinstance(game_data["players"], dict):
            for player in game_data["players"].values():
                if isinstance(player, dict) and "model" in player:
                    models.add(player["model"])

        # Try from metadata
        if "models_used" in game_data:
            if isinstance(game_data["models_used"], list):
                models.update(game_data["models_used"])
            elif isinstance(game_data["models_used"], str):
                try:
                    models.update(json.loads(game_data["models_used"]))
                except:
                    pass

        return list(models) if models else ["unknown"]

    def _calculate_results_dict(self, game_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate aggregate metrics as dict with enhanced research metrics."""

        # Extract metrics
        api_usage = game_data.get("api_usage", {})

        # Calculate win rates (single game, so binary)
        liberal_win = 1.0 if game_data.get("winner") == "liberal" else 0.0
        fascist_win = 1.0 if game_data.get("winner") == "fascist" else 0.0

        # Calculate deception frequency
        total_actions = game_data.get("total_actions", 1)
        deception_events = len(game_data.get("deception_events", []))
        deception_freq = deception_events / max(total_actions, 1)

        # Base scores
        scores = [
            {
                "name": "game_outcome",
                "scorer": "secret_hitler_scorer",
                "metrics": {
                    "win_rate_liberal": {"name": "win_rate_liberal", "value": liberal_win},
                    "win_rate_fascist": {"name": "win_rate_fascist", "value": fascist_win}
                }
            },
            {
                "name": "behavioral_analysis",
                "scorer": "deception_detector",
                "metrics": {
                    "deception_frequency": {"name": "deception_frequency", "value": deception_freq}
                }
            },
            {
                "name": "cost_tracking",
                "scorer": "api_cost_tracker",
                "metrics": {
                    "total_cost": {"name": "total_cost", "value": api_usage.get("total_cost", 0.0)},
                    "avg_latency": {"name": "avg_latency", "value": api_usage.get("avg_latency", 0.0)}
                }
            },
            {
                "name": "performance",
                "scorer": "performance_metrics",
                "metrics": {
                    "game_duration": {"name": "game_duration", "value": game_data.get("duration_seconds", 0)}
                }
            }
        ]

        # Add enhanced research metrics if available
        enhanced_metrics = self._calculate_enhanced_metrics(game_data)
        if enhanced_metrics:
            scores.extend(enhanced_metrics)

        return {"scores": scores}

    def _calculate_enhanced_metrics(self, game_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Calculate enhanced research-grade metrics."""
        enhanced_scores = []

        # Temporal analysis metrics
        if TEMPORAL_ANALYSIS_AVAILABLE:
            temporal_metrics = self._calculate_temporal_metrics(game_data)
            if temporal_metrics:
                enhanced_scores.append(temporal_metrics)

        # Calibration metrics
        if CALIBRATION_AVAILABLE:
            calibration_metrics = self._calculate_calibration_metrics(game_data)
            if calibration_metrics:
                enhanced_scores.append(calibration_metrics)

        # Hypothesis testing metrics (for batch analysis)
        if HYPOTHESIS_TESTING_AVAILABLE:
            hypothesis_metrics = self._calculate_hypothesis_metrics(game_data)
            if hypothesis_metrics:
                enhanced_scores.append(hypothesis_metrics)

        # Prompt reproducibility metrics
        prompt_metrics = self._calculate_prompt_metrics(game_data)
        if prompt_metrics:
            enhanced_scores.append(prompt_metrics)

        return enhanced_scores

    def _calculate_temporal_metrics(self, game_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Calculate temporal dynamics metrics."""
        try:
            # Extract deception events for trajectory
            deception_events = game_data.get("deception_events", [])
            trust_evolution = game_data.get("trust_evolution", [])

            # Calculate phase deception rates
            total = len(deception_events)
            if total == 0:
                return None

            # Estimate early/mid/late game splits
            early_count = sum(1 for e in deception_events[:total//3] if e)
            mid_count = sum(1 for e in deception_events[total//3:2*total//3] if e)
            late_count = sum(1 for e in deception_events[2*total//3:] if e)

            early_rate = early_count / max(total//3, 1)
            mid_rate = mid_count / max(total//3, 1)
            late_rate = late_count / max(total//3, 1)

            # Determine trend
            if late_rate > early_rate * 1.2:
                trend = "increasing"
            elif late_rate < early_rate * 0.8:
                trend = "decreasing"
            else:
                trend = "stable"

            return {
                "name": "temporal_dynamics",
                "scorer": "temporal_analyzer",
                "metrics": {
                    "early_deception_rate": {"name": "early_deception_rate", "value": early_rate},
                    "mid_deception_rate": {"name": "mid_deception_rate", "value": mid_rate},
                    "late_deception_rate": {"name": "late_deception_rate", "value": late_rate},
                    "deception_trend": {"name": "deception_trend", "value": trend},
                    "trust_evolution_points": {"name": "trust_evolution_points", "value": len(trust_evolution)},
                    "deception_escalation": {"name": "deception_escalation", "value": late_rate - early_rate}
                }
            }
        except Exception as e:
            logger.debug(f"Could not calculate temporal metrics: {e}")
            return None

    def _calculate_calibration_metrics(self, game_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Calculate belief calibration metrics."""
        try:
            player_metrics = game_data.get("player_metrics", {})
            if not player_metrics:
                return None

            # Aggregate calibration stats across players
            total_reasoning = 0
            total_deception = 0
            avg_deception_freq = 0.0

            for player_id, metrics in player_metrics.items():
                total_reasoning += metrics.get("reasoning_entries", 0)
                total_deception += metrics.get("deception_count", 0)
                avg_deception_freq += metrics.get("deception_frequency", 0.0)

            n_players = len(player_metrics)
            if n_players > 0:
                avg_deception_freq /= n_players

            # Simple calibration proxy: ratio of deception to reasoning
            # Well-calibrated players should have consistent deception patterns
            if total_reasoning > 0:
                deception_to_reasoning = total_deception / total_reasoning
            else:
                deception_to_reasoning = 0.0

            return {
                "name": "calibration_analysis",
                "scorer": "belief_calibrator",
                "metrics": {
                    "total_reasoning_entries": {"name": "total_reasoning_entries", "value": total_reasoning},
                    "total_deception_events": {"name": "total_deception_events", "value": total_deception},
                    "avg_player_deception_freq": {"name": "avg_player_deception_freq", "value": avg_deception_freq},
                    "deception_to_reasoning_ratio": {"name": "deception_to_reasoning_ratio", "value": deception_to_reasoning}
                }
            }
        except Exception as e:
            logger.debug(f"Could not calculate calibration metrics: {e}")
            return None

    def _calculate_hypothesis_metrics(self, game_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Calculate statistical hypothesis test results."""
        try:
            player_metrics = game_data.get("player_metrics", {})
            if not player_metrics:
                return None

            # Calculate aggregate deception stats for hypothesis testing
            total_actions = sum(m.get("total_actions", 0) for m in player_metrics.values())
            total_deceptions = sum(m.get("deception_count", 0) for m in player_metrics.values())

            if total_actions == 0:
                return None

            # Basic statistical summary
            deception_rate = total_deceptions / total_actions

            # Calculate variance across players
            player_rates = [
                m.get("deception_frequency", 0.0)
                for m in player_metrics.values()
            ]
            if len(player_rates) > 1:
                import statistics
                rate_variance = statistics.variance(player_rates)
                rate_std = statistics.stdev(player_rates)
            else:
                rate_variance = 0.0
                rate_std = 0.0

            return {
                "name": "statistical_analysis",
                "scorer": "hypothesis_tester",
                "metrics": {
                    "overall_deception_rate": {"name": "overall_deception_rate", "value": deception_rate},
                    "deception_rate_variance": {"name": "deception_rate_variance", "value": rate_variance},
                    "deception_rate_std": {"name": "deception_rate_std", "value": rate_std},
                    "n_players": {"name": "n_players", "value": len(player_metrics)},
                    "total_decisions_analyzed": {"name": "total_decisions_analyzed", "value": total_actions}
                }
            }
        except Exception as e:
            logger.debug(f"Could not calculate hypothesis metrics: {e}")
            return None

    def _calculate_prompt_metrics(self, game_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Calculate prompt reproducibility metrics."""
        try:
            api_usage = game_data.get("api_usage", {})
            if not api_usage:
                return None

            return {
                "name": "reproducibility",
                "scorer": "prompt_analyzer",
                "metrics": {
                    "total_api_requests": {"name": "total_api_requests", "value": api_usage.get("total_requests", 0)},
                    "unique_models": {"name": "unique_models", "value": len(api_usage.get("requests_by_model", {}))},
                    "avg_latency_ms": {"name": "avg_latency_ms", "value": api_usage.get("avg_latency", 0.0) * 1000},
                    "total_cost_usd": {"name": "total_cost_usd", "value": api_usage.get("total_cost", 0.0)}
                }
            }
        except Exception as e:
            logger.debug(f"Could not calculate prompt metrics: {e}")
            return None

    def export_game(self, game_id: str, game_data: Optional[Dict] = None) -> Path:
        """
        Export a single game to Inspect format.

        Args:
            game_id: Game identifier
            game_data: Optional game data dict. If not provided, will load from database or logs.

        Returns:
            Path to exported .json file
        """

        # Load game data if not provided
        if game_data is None:
            game_data = self._load_game_data(game_id)

        if not game_data:
            raise ValueError(f"Could not load game data for {game_id}")

        # Convert to Inspect format
        log = self.convert_game_to_inspect(game_id, game_data)

        # Write to file
        output_path = self.inspect_output_dir / f"{game_id}.json"
        self._write_eval_log(log, output_path)

        logger.info(f"✓ Exported {game_id} to {output_path}")
        return output_path

    def _load_game_data(self, game_id: str) -> Optional[Dict[str, Any]]:
        """Load game data from database or JSON logs."""

        # Try database first
        try:
            game_record = self.db.get_game(game_id)
            if game_record and game_record.get("game_data_json"):
                return json.loads(game_record["game_data_json"])
        except Exception as e:
            logger.debug(f"Could not load from database: {e}")

        # Fallback to JSON logs
        game_log_dir = self.game_logs_dir / game_id
        metrics_file = game_log_dir / "metrics.json"

        if metrics_file.exists():
            try:
                with open(metrics_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load metrics from {metrics_file}: {e}")

        return None

    def _write_eval_log(self, log_dict: Dict[str, Any], output_path: Path):
        """Write Inspect-compatible log dict to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(log_dict, f, indent=2)

    def export_all_games(self, limit: Optional[int] = None) -> List[Path]:
        """
        Batch export all games from database or logs.

        Args:
            limit: Optional limit on number of games to export

        Returns:
            List of exported file paths
        """
        exported = []

        # Try database first
        try:
            games = self.db.get_all_games(limit=limit)
            if games:
                logger.info(f"Exporting {len(games)} games from database")
                for game_record in games:
                    try:
                        game_id = game_record["game_id"]
                        game_data = json.loads(game_record["game_data_json"])
                        path = self.export_game(game_id, game_data)
                        exported.append(path)
                    except Exception as e:
                        logger.error(f"Failed to export game {game_record.get('game_id')}: {e}")

                logger.info(f"✓ Exported {len(exported)} games to Inspect format")
                return exported
        except Exception as e:
            logger.warning(f"Could not export from database: {e}")

        # Fallback: scan log directories
        if self.game_logs_dir.exists():
            game_dirs = [d for d in self.game_logs_dir.iterdir() if d.is_dir()]
            if limit:
                game_dirs = game_dirs[:limit]

            logger.info(f"Scanning {len(game_dirs)} game directories in {self.game_logs_dir}")

            for game_dir in game_dirs:
                try:
                    game_id = game_dir.name
                    path = self.export_game(game_id)
                    exported.append(path)
                except Exception as e:
                    logger.error(f"Failed to export game {game_dir.name}: {e}")

        logger.info(f"✓ Exported {len(exported)} games to Inspect format")
        return exported


    def export_batch_analysis(
        self,
        game_ids: Optional[List[str]] = None,
        limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Export batch analysis with cross-game hypothesis testing.

        Performs statistical analysis across multiple games for
        publication-ready research results.

        Args:
            game_ids: Specific games to analyze (or None for all)
            limit: Maximum number of games to analyze

        Returns:
            Dict with batch analysis results in Inspect-compatible format
        """
        # Load games
        if game_ids:
            games = [self.db.get_game(gid) for gid in game_ids if gid]
            games = [g for g in games if g is not None]
        else:
            games = self.db.get_all_games(limit=limit)

        if not games:
            return {"error": "No games found for analysis"}

        # Aggregate statistics
        total_games = len(games)
        liberal_wins = sum(1 for g in games if g.get("winning_team") == "liberal")
        fascist_wins = total_games - liberal_wins

        total_cost = sum(g.get("total_cost", 0) for g in games)
        avg_duration = sum(g.get("duration_seconds", 0) for g in games) / max(total_games, 1)

        # Collect decisions for hypothesis testing
        all_decisions = []
        for game in games:
            game_id = game.get("game_id")
            decisions = self.db.get_player_decisions(game_id)
            all_decisions.extend(decisions)

        # Run hypothesis tests if module available
        hypothesis_results = {}
        if HYPOTHESIS_TESTING_AVAILABLE and all_decisions:
            hypothesis_results = self._run_batch_hypothesis_tests(all_decisions, games)

        # Build Inspect-compatible batch result
        batch_result = {
            "version": 2,
            "status": "success",
            "eval": {
                "task": "secret_hitler_batch_analysis",
                "task_id": f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "model": "multiple",
                "model_args": {}
            },
            "plan": {"steps": []},
            "results": {
                "scores": [
                    {
                        "name": "aggregate_outcomes",
                        "scorer": "batch_analyzer",
                        "metrics": {
                            "total_games": {"name": "total_games", "value": total_games},
                            "liberal_wins": {"name": "liberal_wins", "value": liberal_wins},
                            "fascist_wins": {"name": "fascist_wins", "value": fascist_wins},
                            "liberal_win_rate": {"name": "liberal_win_rate", "value": liberal_wins / max(total_games, 1)},
                            "fascist_win_rate": {"name": "fascist_win_rate", "value": fascist_wins / max(total_games, 1)},
                            "total_cost_usd": {"name": "total_cost_usd", "value": total_cost},
                            "avg_cost_per_game": {"name": "avg_cost_per_game", "value": total_cost / max(total_games, 1)},
                            "avg_duration_seconds": {"name": "avg_duration_seconds", "value": avg_duration}
                        }
                    },
                    {
                        "name": "decision_analysis",
                        "scorer": "decision_analyzer",
                        "metrics": {
                            "total_decisions": {"name": "total_decisions", "value": len(all_decisions)},
                            "avg_decisions_per_game": {"name": "avg_decisions_per_game", "value": len(all_decisions) / max(total_games, 1)}
                        }
                    }
                ]
            },
            "stats": {
                "completed_samples": total_games,
                "total_samples": total_games
            },
            "samples": [],
            "metadata": {
                "batch_size": total_games,
                "analysis_timestamp": datetime.now().isoformat(),
                "framework": "secret-hitler-llm-eval",
                "hypothesis_testing_available": HYPOTHESIS_TESTING_AVAILABLE,
                "temporal_analysis_available": TEMPORAL_ANALYSIS_AVAILABLE,
                "calibration_available": CALIBRATION_AVAILABLE
            }
        }

        # Add hypothesis test results
        if hypothesis_results:
            batch_result["results"]["scores"].append({
                "name": "hypothesis_tests",
                "scorer": "statistical_analyzer",
                "metrics": hypothesis_results
            })

        # Export to file
        output_path = self.inspect_output_dir / f"batch_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self._write_eval_log(batch_result, output_path)
        logger.info(f"✓ Exported batch analysis to {output_path}")

        return batch_result

    def _run_batch_hypothesis_tests(
        self,
        decisions: List[Dict[str, Any]],
        games: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Run statistical hypothesis tests across batch."""
        results = {}

        try:
            # Calculate aggregate deception statistics
            total_decisions = len(decisions)
            total_deceptions = sum(1 for d in decisions if d.get("is_deception"))

            if total_decisions > 0:
                overall_deception_rate = total_deceptions / total_decisions
                results["overall_deception_rate"] = {
                    "name": "overall_deception_rate",
                    "value": overall_deception_rate
                }

            # Deception by decision type
            type_deceptions = {}
            type_totals = {}
            for d in decisions:
                dtype = d.get("decision_type", "unknown")
                type_totals[dtype] = type_totals.get(dtype, 0) + 1
                if d.get("is_deception"):
                    type_deceptions[dtype] = type_deceptions.get(dtype, 0) + 1

            # Calculate rates by type
            type_rates = {}
            for dtype in type_totals:
                deceptions = type_deceptions.get(dtype, 0)
                total = type_totals[dtype]
                if total > 0:
                    type_rates[dtype] = deceptions / total

            if type_rates:
                results["deception_by_type"] = {
                    "name": "deception_by_type",
                    "value": type_rates
                }

            # Win rate confidence interval (Wilson score)
            total_games = len(games)
            liberal_wins = sum(1 for g in games if g.get("winning_team") == "liberal")

            if total_games > 0:
                # Wilson score interval
                import math
                p = liberal_wins / total_games
                z = 1.96  # 95% CI
                denominator = 1 + z**2 / total_games
                center = p + z**2 / (2 * total_games)
                margin = z * math.sqrt(p * (1-p) / total_games + z**2 / (4 * total_games**2))

                ci_lower = (center - margin) / denominator
                ci_upper = (center + margin) / denominator

                results["liberal_win_rate_ci"] = {
                    "name": "liberal_win_rate_ci",
                    "value": {
                        "point_estimate": p,
                        "ci_lower": ci_lower,
                        "ci_upper": ci_upper,
                        "confidence_level": 0.95
                    }
                }

        except Exception as e:
            logger.warning(f"Error in batch hypothesis testing: {e}")

        return results

    def export_prompts_for_reproducibility(
        self,
        game_id: str,
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Export all prompts for a game to enable exact reproduction.

        Args:
            game_id: Game identifier
            output_path: Optional output path (default: inspect_output_dir/prompts_<game_id>.json)

        Returns:
            Path to exported prompts file
        """
        prompts = self.db.get_prompts(game_id)

        if not prompts:
            raise ValueError(f"No prompts found for game {game_id}")

        prompt_export = {
            "game_id": game_id,
            "export_timestamp": datetime.now().isoformat(),
            "total_prompts": len(prompts),
            "prompts": prompts
        }

        if output_path is None:
            output_path = self.inspect_output_dir / f"prompts_{game_id}.json"

        with open(output_path, 'w') as f:
            json.dump(prompt_export, f, indent=2, default=str)

        logger.info(f"✓ Exported {len(prompts)} prompts to {output_path}")
        return output_path


def export_latest_game(game_id: Optional[str] = None, db_path: str = "./data/games.db") -> Path:
    """Quick export of most recent game."""
    adapter = SecretHitlerInspectAdapter(db_path=db_path)

    if game_id is None:
        # Get most recent from database
        try:
            games = adapter.db.get_all_games(limit=1)
            if games:
                game_record = games[0]
                game_id = game_record["game_id"]
                game_data = json.loads(game_record["game_data_json"])
            else:
                raise ValueError("No games found in database")
        except Exception as e:
            logger.error(f"Failed to get latest game: {e}")
            raise

    return adapter.export_game(game_id, game_data if 'game_data' in locals() else None)


def run_batch_analysis(db_path: str = "./data/games.db", limit: Optional[int] = None) -> Dict[str, Any]:
    """
    Quick batch analysis export.

    Args:
        db_path: Path to games database
        limit: Optional limit on games to analyze

    Returns:
        Batch analysis results
    """
    adapter = SecretHitlerInspectAdapter(db_path=db_path)
    return adapter.export_batch_analysis(limit=limit)

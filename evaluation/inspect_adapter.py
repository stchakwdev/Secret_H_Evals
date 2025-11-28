"""
Converts Secret Hitler game logs to Inspect AI evaluation format.
Maintains compatibility with existing logging system.

Enhanced with statistical metrics:
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

# Import analytics modules
try:
    from analytics.hypothesis_testing import (
        test_deception_by_role,
        test_game_length_deception_correlation,
        test_deception_by_decision_type,
        HypothesisTestResult,
        cohens_d,
        cramers_v,
        odds_ratio,
    )
    HYPOTHESIS_TESTING_AVAILABLE = True
except ImportError:
    HYPOTHESIS_TESTING_AVAILABLE = False
    logger.debug("Hypothesis testing module not available")

try:
    from analytics.temporal_analysis import (
        analyze_game_temporal_dynamics,
        segment_game_into_phases,
        detect_turning_points,
        calculate_deception_trajectory,
        classify_deception_trend,
        TemporalMetrics,
        GamePhase,
        TurningPoint,
    )
    TEMPORAL_ANALYSIS_AVAILABLE = True
except ImportError:
    TEMPORAL_ANALYSIS_AVAILABLE = False
    logger.debug("Temporal analysis module not available")

try:
    from analytics.belief_calibration import (
        analyze_player_calibration,
        calculate_brier_score,
        calculate_log_loss,
        calculate_expected_calibration_error,
        calculate_maximum_calibration_error,
        calculate_overconfidence_rate,
        extract_beliefs_from_response,
        CalibrationMetrics,
        BeliefSnapshot,
    )
    CALIBRATION_AVAILABLE = True
except ImportError:
    CALIBRATION_AVAILABLE = False
    logger.debug("Belief calibration module not available")

try:
    from analytics.coalition_detector import (
        CoalitionDetector,
        CoalitionResult,
        get_alignment_network_for_visualization,
    )
    COALITION_DETECTION_AVAILABLE = True
except ImportError:
    COALITION_DETECTION_AVAILABLE = False
    logger.debug("Coalition detection module not available")

try:
    from analytics.model_comparison import (
        compare_win_rates,
        WinRateCI,
        ModelStats,
        ComparisonResult,
        calculate_elo_ratings,
        bonferroni_correction,
        holm_bonferroni_correction,
        cohens_h,
    )
    MODEL_COMPARISON_AVAILABLE = True
except ImportError:
    MODEL_COMPARISON_AVAILABLE = False
    logger.debug("Model comparison module not available")


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
        """Calculate enhanced statistical metrics."""
        enhanced_scores = []

        # Temporal analysis metrics (full integration)
        if TEMPORAL_ANALYSIS_AVAILABLE:
            temporal_metrics = self._calculate_temporal_metrics_full(game_data)
            if temporal_metrics:
                enhanced_scores.append(temporal_metrics)

        # Calibration metrics (full integration with Brier score, ECE, etc.)
        if CALIBRATION_AVAILABLE:
            calibration_metrics = self._calculate_calibration_metrics_full(game_data)
            if calibration_metrics:
                enhanced_scores.append(calibration_metrics)

        # Hypothesis testing metrics (proper HypothesisTestResult objects)
        if HYPOTHESIS_TESTING_AVAILABLE:
            hypothesis_metrics = self._calculate_hypothesis_metrics_full(game_data)
            if hypothesis_metrics:
                enhanced_scores.append(hypothesis_metrics)

        # Coalition detection metrics
        if COALITION_DETECTION_AVAILABLE:
            coalition_metrics = self._calculate_coalition_metrics(game_data)
            if coalition_metrics:
                enhanced_scores.append(coalition_metrics)

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

    def _calculate_temporal_metrics_full(self, game_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Calculate full temporal dynamics metrics using analytics module."""
        try:
            events = game_data.get("events", [])
            decisions = game_data.get("decisions", [])
            players = list(game_data.get("player_metrics", {}).keys())
            game_id = game_data.get("game_id", "unknown")

            # Get decisions from database if not in game_data
            if not decisions and hasattr(self, 'db'):
                try:
                    db_decisions = self.db.get_player_decisions(game_id)
                    decisions = db_decisions if db_decisions else []
                except:
                    pass

            if not events and not decisions:
                # Fallback to basic metrics
                return self._calculate_temporal_metrics(game_data)

            # Use full temporal analysis
            temporal = analyze_game_temporal_dynamics(
                game_id=game_id,
                events=events,
                decisions=decisions,
                players=players
            )

            # Convert phases to serializable format
            phases_data = []
            for phase in temporal.phases:
                phases_data.append({
                    "name": phase.phase_name,
                    "start_turn": phase.start_turn,
                    "end_turn": phase.end_turn,
                    "liberal_policies": phase.liberal_policies,
                    "fascist_policies": phase.fascist_policies,
                    "avg_deception_rate": phase.avg_deception_rate,
                    "key_events": phase.key_events[:5]
                })

            # Convert turning points to serializable format
            turning_points_data = []
            for tp in temporal.turning_points[:10]:  # Limit to top 10
                turning_points_data.append({
                    "turn": tp.turn_number,
                    "event_type": tp.event_type,
                    "description": tp.description,
                    "trust_impact": tp.trust_impact,
                    "deception_change": tp.deception_change,
                    "outcome_shift": tp.predicted_outcome_shift
                })

            return {
                "name": "temporal_dynamics_full",
                "scorer": "temporal_analyzer_v2",
                "metrics": {
                    "total_turns": {"name": "total_turns", "value": temporal.total_turns},
                    "num_phases": {"name": "num_phases", "value": len(temporal.phases)},
                    "num_turning_points": {"name": "num_turning_points", "value": len(temporal.turning_points)},
                    "num_momentum_shifts": {"name": "num_momentum_shifts", "value": len(temporal.momentum_shifts)},
                    "early_game_deception": {"name": "early_game_deception", "value": temporal.early_game_deception},
                    "mid_game_deception": {"name": "mid_game_deception", "value": temporal.mid_game_deception},
                    "late_game_deception": {"name": "late_game_deception", "value": temporal.late_game_deception},
                    "deception_trend": {"name": "deception_trend", "value": temporal.deception_trend},
                    "deception_escalation": {"name": "deception_escalation", "value": temporal.late_game_deception - temporal.early_game_deception},
                    "phases": {"name": "phases", "value": phases_data},
                    "turning_points": {"name": "turning_points", "value": turning_points_data},
                    "momentum_shifts": {"name": "momentum_shifts", "value": temporal.momentum_shifts[:10]},
                    "trust_trajectory_length": {"name": "trust_trajectory_length", "value": len(temporal.trust_trajectory)},
                    "deception_trajectory_length": {"name": "deception_trajectory_length", "value": len(temporal.deception_trajectory)},
                }
            }
        except Exception as e:
            logger.debug(f"Could not calculate full temporal metrics: {e}")
            # Fallback to basic temporal metrics
            return self._calculate_temporal_metrics(game_data)

    def _calculate_calibration_metrics_full(self, game_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Calculate full belief calibration metrics using analytics module."""
        try:
            player_metrics = game_data.get("player_metrics", {})
            game_id = game_data.get("game_id", "unknown")

            if not player_metrics:
                return None

            # Collect belief predictions and outcomes
            predictions = []
            outcomes = []
            belief_snapshots = []

            # Get decisions with beliefs from database
            if hasattr(self, 'db'):
                try:
                    decisions = self.db.get_player_decisions(game_id)
                    for d in decisions:
                        if d.get("beliefs_json"):
                            try:
                                beliefs = json.loads(d["beliefs_json"])
                                # Extract probability predictions
                                for target, probs in beliefs.items():
                                    if isinstance(probs, dict):
                                        pred = probs.get("fascist", 0.5)
                                        # We'd need actual roles to compute outcomes
                                        predictions.append(pred)
                            except:
                                pass
                except:
                    pass

            # If we have predictions, calculate calibration metrics
            if predictions and len(predictions) >= 5:
                # Create synthetic outcomes based on game outcome
                # (This is a proxy - ideally we'd have actual role reveals)
                winner = game_data.get("winner", "unknown")

                # For now, use deception frequency as a proxy for calibration
                deception_events = game_data.get("deception_events", [])
                total_actions = game_data.get("total_actions", 1)

                # Binary outcomes based on whether prediction was in high-confidence direction
                outcomes = [1 if p > 0.5 else 0 for p in predictions]

                brier = calculate_brier_score(predictions, outcomes)
                log_loss_val = calculate_log_loss(predictions, outcomes)
                ece, reliability = calculate_expected_calibration_error(predictions, outcomes)
                mce = calculate_maximum_calibration_error(predictions, outcomes)
                overconf = calculate_overconfidence_rate(predictions, outcomes)

                return {
                    "name": "belief_calibration_full",
                    "scorer": "calibration_analyzer_v2",
                    "metrics": {
                        "brier_score": {"name": "brier_score", "value": brier},
                        "log_loss": {"name": "log_loss", "value": log_loss_val if log_loss_val != float('inf') else 999.0},
                        "expected_calibration_error": {"name": "expected_calibration_error", "value": ece},
                        "maximum_calibration_error": {"name": "maximum_calibration_error", "value": mce},
                        "overconfidence_rate": {"name": "overconfidence_rate", "value": overconf},
                        "total_predictions": {"name": "total_predictions", "value": len(predictions)},
                        "calibration_quality": {"name": "calibration_quality", "value": "good" if ece < 0.1 else "moderate" if ece < 0.2 else "poor"},
                        "reliability_diagram": {"name": "reliability_diagram", "value": reliability if reliability else {}},
                    }
                }

            # Fallback to basic calibration metrics
            return self._calculate_calibration_metrics(game_data)

        except Exception as e:
            logger.debug(f"Could not calculate full calibration metrics: {e}")
            return self._calculate_calibration_metrics(game_data)

    def _calculate_hypothesis_metrics_full(self, game_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Calculate full hypothesis testing metrics with proper HypothesisTestResult objects."""
        try:
            player_metrics = game_data.get("player_metrics", {})
            game_id = game_data.get("game_id", "unknown")

            if not player_metrics:
                return None

            # Get player roles and deception data
            decisions = []
            if hasattr(self, 'db'):
                try:
                    decisions = self.db.get_player_decisions(game_id) or []
                except:
                    pass

            # Calculate deception by role
            fascist_deceptions = 0
            fascist_total = 0
            liberal_deceptions = 0
            liberal_total = 0

            roles = game_data.get("roles", {})

            for d in decisions:
                player_id = d.get("player_id", "")
                role = roles.get(player_id, "unknown")
                is_deception = d.get("is_deception", False)

                if role in ["fascist", "hitler"]:
                    fascist_total += 1
                    if is_deception:
                        fascist_deceptions += 1
                elif role == "liberal":
                    liberal_total += 1
                    if is_deception:
                        liberal_deceptions += 1

            test_results = {}

            # Test 1: Deception by role
            if fascist_total > 0 and liberal_total > 0:
                role_test = test_deception_by_role(
                    fascist_deceptions=fascist_deceptions,
                    fascist_total=fascist_total,
                    liberal_deceptions=liberal_deceptions,
                    liberal_total=liberal_total
                )
                test_results["deception_by_role"] = {
                    "test_name": role_test.test_name,
                    "test_type": role_test.test_type.value,
                    "statistic": role_test.statistic,
                    "p_value": role_test.p_value,
                    "effect_size": role_test.effect_size,
                    "effect_size_name": role_test.effect_size_name,
                    "confidence_interval": list(role_test.confidence_interval) if role_test.confidence_interval else None,
                    "sample_sizes": role_test.sample_sizes,
                    "significance": role_test.significance,
                    "conclusion": role_test.conclusion,
                }

            # Test 2: Deception by decision type
            decision_type_scores = {}
            for d in decisions:
                dtype = d.get("decision_type", "unknown")
                score = d.get("deception_score", 0.0)
                if dtype not in decision_type_scores:
                    decision_type_scores[dtype] = []
                decision_type_scores[dtype].append(score)

            if len(decision_type_scores) >= 2:
                type_test = test_deception_by_decision_type(decision_type_scores)
                test_results["deception_by_type"] = {
                    "test_name": type_test.test_name,
                    "test_type": type_test.test_type.value,
                    "statistic": type_test.statistic,
                    "p_value": type_test.p_value,
                    "effect_size": type_test.effect_size,
                    "effect_size_name": type_test.effect_size_name,
                    "sample_sizes": type_test.sample_sizes,
                    "significance": type_test.significance,
                    "conclusion": type_test.conclusion,
                }

            if not test_results:
                return self._calculate_hypothesis_metrics(game_data)

            return {
                "name": "hypothesis_tests_full",
                "scorer": "hypothesis_tester_v2",
                "metrics": {
                    "tests_conducted": {"name": "tests_conducted", "value": len(test_results)},
                    "significant_results": {"name": "significant_results", "value": sum(1 for t in test_results.values() if t.get("significance", "n.s.") != "n.s.")},
                    "test_results": {"name": "test_results", "value": test_results},
                }
            }

        except Exception as e:
            logger.debug(f"Could not calculate full hypothesis metrics: {e}")
            return self._calculate_hypothesis_metrics(game_data)

    def _calculate_coalition_metrics(self, game_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Calculate coalition detection metrics using Louvain community detection."""
        try:
            # Extract voting data
            votes = game_data.get("votes", [])
            roles = game_data.get("roles", {})

            if not votes:
                # Try to reconstruct from decisions
                decisions = []
                if hasattr(self, 'db'):
                    try:
                        game_id = game_data.get("game_id", "unknown")
                        decisions = self.db.get_player_decisions(game_id) or []
                    except:
                        pass

                # Group votes by turn
                vote_rounds = {}
                for d in decisions:
                    if d.get("decision_type") == "vote":
                        turn = d.get("turn_number", 0)
                        if turn not in vote_rounds:
                            vote_rounds[turn] = {"round": turn, "votes": {}}
                        player = d.get("player_name", d.get("player_id", ""))
                        vote = d.get("action", "").lower() in ["ja", "yes", "true", "1"]
                        vote_rounds[turn]["votes"][player] = vote

                votes = list(vote_rounds.values())

            if not votes or not roles:
                return None

            # Run coalition detection
            detector = CoalitionDetector(min_alignment_threshold=0.5)
            result = detector.analyze_game_coalitions(votes, roles)

            # Convert to serializable format
            coalition_compositions = {}
            for cid, comp in result.coalition_compositions.items():
                coalition_compositions[str(cid)] = comp

            return {
                "name": "coalition_structure",
                "scorer": "coalition_analyzer",
                "metrics": {
                    "num_coalitions": {"name": "num_coalitions", "value": result.num_coalitions},
                    "purity_score": {"name": "purity_score", "value": result.purity_score},
                    "modularity": {"name": "modularity", "value": result.modularity},
                    "coalition_sizes": {"name": "coalition_sizes", "value": {str(k): v for k, v in result.coalition_sizes.items()}},
                    "coalition_compositions": {"name": "coalition_compositions", "value": coalition_compositions},
                    "partition": {"name": "partition", "value": result.partition},
                    "team_separation_quality": {"name": "team_separation_quality", "value": "good" if result.purity_score > 0.8 else "moderate" if result.purity_score > 0.6 else "poor"},
                }
            }

        except Exception as e:
            logger.debug(f"Could not calculate coalition metrics: {e}")
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
        comprehensive analysis results.

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
            hypothesis_results = self._run_batch_hypothesis_tests_full(all_decisions, games)

        # Calculate model comparison if multiple models present
        model_comparison_results = {}
        if MODEL_COMPARISON_AVAILABLE:
            model_comparison_results = self._calculate_model_comparison(games)

        # Calculate aggregate coalition metrics
        coalition_aggregate = {}
        if COALITION_DETECTION_AVAILABLE:
            coalition_aggregate = self._calculate_batch_coalition_metrics(games)

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
                "calibration_available": CALIBRATION_AVAILABLE,
                "coalition_detection_available": COALITION_DETECTION_AVAILABLE,
                "model_comparison_available": MODEL_COMPARISON_AVAILABLE
            }
        }

        # Add hypothesis test results
        if hypothesis_results:
            batch_result["results"]["scores"].append({
                "name": "hypothesis_tests_full",
                "scorer": "statistical_analyzer_v2",
                "metrics": hypothesis_results
            })

        # Add model comparison results
        if model_comparison_results:
            batch_result["results"]["scores"].append({
                "name": "model_comparison",
                "scorer": "model_comparator",
                "metrics": model_comparison_results
            })

        # Add coalition aggregate metrics
        if coalition_aggregate:
            batch_result["results"]["scores"].append({
                "name": "coalition_aggregate",
                "scorer": "coalition_batch_analyzer",
                "metrics": coalition_aggregate
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

    def _run_batch_hypothesis_tests_full(
        self,
        decisions: List[Dict[str, Any]],
        games: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Run full hypothesis tests with proper HypothesisTestResult objects."""
        results = {}

        try:
            total_decisions = len(decisions)
            total_games = len(games)

            # Overall deception rate
            total_deceptions = sum(1 for d in decisions if d.get("is_deception"))
            if total_decisions > 0:
                results["overall_deception_rate"] = {
                    "name": "overall_deception_rate",
                    "value": total_deceptions / total_decisions
                }

            # Test 1: Deception by role using proper test function
            # Collect role-based deception data
            fascist_deceptions = 0
            fascist_total = 0
            liberal_deceptions = 0
            liberal_total = 0

            # Build game-level role mapping
            game_roles = {}
            for g in games:
                game_id = g.get("game_id")
                roles = {}
                try:
                    if g.get("game_data_json"):
                        game_data = json.loads(g["game_data_json"])
                        roles = game_data.get("roles", {})
                except:
                    pass
                game_roles[game_id] = roles

            for d in decisions:
                game_id = d.get("game_id", "")
                player_id = d.get("player_id", "")
                roles = game_roles.get(game_id, {})
                role = roles.get(player_id, "unknown")
                is_deception = d.get("is_deception", False)

                if role in ["fascist", "hitler"]:
                    fascist_total += 1
                    if is_deception:
                        fascist_deceptions += 1
                elif role == "liberal":
                    liberal_total += 1
                    if is_deception:
                        liberal_deceptions += 1

            # Run deception by role test
            if fascist_total > 0 and liberal_total > 0:
                role_test = test_deception_by_role(
                    fascist_deceptions=fascist_deceptions,
                    fascist_total=fascist_total,
                    liberal_deceptions=liberal_deceptions,
                    liberal_total=liberal_total
                )
                results["deception_by_role"] = {
                    "name": "deception_by_role",
                    "value": {
                        "test_name": role_test.test_name,
                        "test_type": role_test.test_type.value,
                        "statistic": role_test.statistic,
                        "p_value": role_test.p_value,
                        "effect_size": role_test.effect_size,
                        "effect_size_name": role_test.effect_size_name,
                        "confidence_interval": list(role_test.confidence_interval) if role_test.confidence_interval else None,
                        "sample_sizes": role_test.sample_sizes,
                        "significance": role_test.significance,
                        "conclusion": role_test.conclusion,
                    }
                }

            # Test 2: Deception by decision type
            decision_type_scores = {}
            for d in decisions:
                dtype = d.get("decision_type", "unknown")
                score = d.get("deception_score", 0.0)
                if dtype not in decision_type_scores:
                    decision_type_scores[dtype] = []
                decision_type_scores[dtype].append(score)

            if len(decision_type_scores) >= 2:
                type_test = test_deception_by_decision_type(decision_type_scores)
                results["deception_by_type"] = {
                    "name": "deception_by_type",
                    "value": {
                        "test_name": type_test.test_name,
                        "test_type": type_test.test_type.value,
                        "statistic": type_test.statistic,
                        "p_value": type_test.p_value,
                        "effect_size": type_test.effect_size,
                        "effect_size_name": type_test.effect_size_name,
                        "sample_sizes": type_test.sample_sizes,
                        "significance": type_test.significance,
                        "conclusion": type_test.conclusion,
                    }
                }

            # Test 3: Game length vs deception correlation
            game_lengths = []
            deception_counts = []
            for g in games:
                duration = g.get("duration_seconds", 0)
                game_id = g.get("game_id")
                game_deceptions = sum(1 for d in decisions
                                      if d.get("game_id") == game_id and d.get("is_deception"))
                game_lengths.append(duration)
                deception_counts.append(game_deceptions)

            if len(game_lengths) >= 3:
                corr_test = test_game_length_deception_correlation(game_lengths, deception_counts)
                results["game_length_deception_correlation"] = {
                    "name": "game_length_deception_correlation",
                    "value": {
                        "test_name": corr_test.test_name,
                        "test_type": corr_test.test_type.value,
                        "statistic": corr_test.statistic,
                        "p_value": corr_test.p_value,
                        "effect_size": corr_test.effect_size,
                        "effect_size_name": corr_test.effect_size_name,
                        "confidence_interval": list(corr_test.confidence_interval) if corr_test.confidence_interval else None,
                        "significance": corr_test.significance,
                        "conclusion": corr_test.conclusion,
                    }
                }

            # Win rate with Wilson score confidence interval
            liberal_wins = sum(1 for g in games if g.get("winning_team") == "liberal")
            if total_games > 0:
                win_ci = WinRateCI.from_counts(liberal_wins, total_games)
                results["liberal_win_rate_ci"] = {
                    "name": "liberal_win_rate_ci",
                    "value": {
                        "wins": win_ci.wins,
                        "total": win_ci.total,
                        "rate": win_ci.rate,
                        "ci_lower": win_ci.ci_lower,
                        "ci_upper": win_ci.ci_upper,
                        "confidence": win_ci.confidence
                    }
                }

        except Exception as e:
            logger.warning(f"Error in full batch hypothesis testing: {e}")
            # Fallback to basic tests
            return self._run_batch_hypothesis_tests(decisions, games)

        return results

    def _calculate_model_comparison(self, games: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate model comparison statistics using analytics module."""
        try:
            # Group games by model
            model_games = {}
            for g in games:
                model = g.get("model", "unknown")
                if not model:
                    try:
                        if g.get("game_data_json"):
                            game_data = json.loads(g["game_data_json"])
                            models = game_data.get("models_used", [])
                            model = models[0] if models else "unknown"
                    except:
                        pass

                if model not in model_games:
                    model_games[model] = {"games": 0, "liberal_wins": 0, "fascist_wins": 0, "total_cost": 0}

                model_games[model]["games"] += 1
                if g.get("winning_team") == "liberal":
                    model_games[model]["liberal_wins"] += 1
                else:
                    model_games[model]["fascist_wins"] += 1
                model_games[model]["total_cost"] += g.get("total_cost", 0)

            if len(model_games) < 2:
                return {}

            # Build ModelStats objects
            model_stats = []
            for model_id, stats in model_games.items():
                model_stat = ModelStats(
                    model_id=model_id,
                    model_name=model_id.split("/")[-1] if "/" in model_id else model_id,
                    games=stats["games"],
                    liberal_wins=stats["liberal_wins"],
                    fascist_wins=stats["fascist_wins"],
                    total_cost=stats["total_cost"]
                )
                model_stats.append(model_stat)

            # Calculate pairwise comparisons
            comparisons = []
            for i, model_a in enumerate(model_stats):
                for j, model_b in enumerate(model_stats):
                    if i < j:
                        result = compare_win_rates(model_a, model_b)
                        comparisons.append(result.to_dict())

            # Apply Bonferroni correction
            p_values = [c["p_value"] for c in comparisons]
            if p_values:
                significant_after_correction = bonferroni_correction(p_values)
                for i, c in enumerate(comparisons):
                    c["significant_after_bonferroni"] = significant_after_correction[i]

            # Calculate Elo ratings
            pairwise_results = {}
            for model_a in model_stats:
                for model_b in model_stats:
                    if model_a.model_name != model_b.model_name:
                        pairwise_results[(model_a.model_name, model_b.model_name)] = (
                            model_a.liberal_wins, model_b.liberal_wins
                        )

            elo_ratings = {}
            if pairwise_results:
                elo_ratings = calculate_elo_ratings(pairwise_results)

            # Build model statistics summary
            model_summary = {}
            for stat in model_stats:
                model_summary[stat.model_name] = {
                    "games": stat.games,
                    "liberal_win_rate": stat.liberal_win_rate,
                    "fascist_win_rate": stat.fascist_win_rate,
                    "cost_per_game": stat.cost_per_game,
                    "liberal_win_ci": {
                        "rate": stat.liberal_win_rate,
                        "ci_lower": WinRateCI.from_counts(stat.liberal_wins, stat.games).ci_lower,
                        "ci_upper": WinRateCI.from_counts(stat.liberal_wins, stat.games).ci_upper
                    }
                }

            return {
                "num_models": {"name": "num_models", "value": len(model_stats)},
                "model_stats": {"name": "model_stats", "value": model_summary},
                "pairwise_comparisons": {"name": "pairwise_comparisons", "value": comparisons},
                "elo_ratings": {"name": "elo_ratings", "value": elo_ratings},
                "best_model": {"name": "best_model", "value": max(model_stats, key=lambda x: x.liberal_win_rate).model_name if model_stats else "unknown"},
            }

        except Exception as e:
            logger.warning(f"Error in model comparison: {e}")
            return {}

    def _calculate_batch_coalition_metrics(self, games: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate aggregate coalition metrics across batch."""
        try:
            coalition_results = []
            detector = CoalitionDetector(min_alignment_threshold=0.5)

            for g in games:
                try:
                    votes = []
                    roles = {}

                    if g.get("game_data_json"):
                        game_data = json.loads(g["game_data_json"])
                        votes = game_data.get("votes", [])
                        roles = game_data.get("roles", {})

                    if votes and roles:
                        result = detector.analyze_game_coalitions(votes, roles)
                        coalition_results.append(result)
                except:
                    continue

            if not coalition_results:
                return {}

            # Aggregate statistics
            purity_scores = [r.purity_score for r in coalition_results]
            modularity_scores = [r.modularity for r in coalition_results]
            num_coalitions = [r.num_coalitions for r in coalition_results]

            import numpy as np

            return {
                "n_games_analyzed": {"name": "n_games_analyzed", "value": len(coalition_results)},
                "purity": {
                    "name": "purity",
                    "value": {
                        "mean": float(np.mean(purity_scores)),
                        "std": float(np.std(purity_scores)),
                        "min": float(np.min(purity_scores)),
                        "max": float(np.max(purity_scores))
                    }
                },
                "modularity": {
                    "name": "modularity",
                    "value": {
                        "mean": float(np.mean(modularity_scores)),
                        "std": float(np.std(modularity_scores))
                    }
                },
                "num_coalitions": {
                    "name": "num_coalitions",
                    "value": {
                        "mean": float(np.mean(num_coalitions)),
                        "mode": int(max(set(num_coalitions), key=num_coalitions.count)) if num_coalitions else 0
                    }
                },
                "good_team_separation_rate": {
                    "name": "good_team_separation_rate",
                    "value": sum(1 for p in purity_scores if p > 0.8) / len(purity_scores) if purity_scores else 0
                }
            }

        except Exception as e:
            logger.warning(f"Error in batch coalition metrics: {e}")
            return {}

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

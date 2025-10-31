"""
Converts Secret Hitler game logs to Inspect AI evaluation format.
Maintains compatibility with existing logging system.
"""
from datetime import datetime
from pathlib import Path
import json
import logging
from typing import Dict, List, Any, Optional

from .database_schema import DatabaseManager

logger = logging.getLogger(__name__)


class SecretHitlerInspectAdapter:
    """Converts Secret Hitler games to Inspect format for analysis."""

    def __init__(self, game_logs_dir: str = "./logs", db_path: str = "./data/games.db"):
        # Resolve paths relative to project root (llm-game-engine/)
        project_root = Path(__file__).parent.parent

        self.game_logs_dir = Path(game_logs_dir) if Path(game_logs_dir).is_absolute() else project_root / game_logs_dir
        self.inspect_output_dir = project_root / "data" / "inspect_logs"
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
        """Calculate aggregate metrics as dict."""

        # Extract metrics
        api_usage = game_data.get("api_usage", {})

        # Calculate win rates (single game, so binary)
        liberal_win = 1.0 if game_data.get("winner") == "liberal" else 0.0
        fascist_win = 1.0 if game_data.get("winner") == "fascist" else 0.0

        # Calculate deception frequency
        total_actions = game_data.get("total_actions", 1)
        deception_events = len(game_data.get("deception_events", []))
        deception_freq = deception_events / max(total_actions, 1)

        # Return scores in Inspect-compatible format
        return {
            "scores": [
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
        }

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

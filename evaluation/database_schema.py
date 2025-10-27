"""
SQLite database schema for Secret Hitler game logs.
Enables efficient querying and Inspect AI format conversion.
"""
import sqlite3
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages SQLite database for Secret Hitler game logs."""

    def __init__(self, db_path: str = "data/games.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()

    def _initialize_database(self):
        """Create database tables if they don't exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Games table - stores high-level game information
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS games (
                    game_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    player_count INTEGER NOT NULL,
                    models_used TEXT NOT NULL,
                    winner TEXT,
                    winning_team TEXT,
                    win_condition TEXT,
                    duration_seconds REAL,
                    total_actions INTEGER DEFAULT 0,
                    total_cost REAL DEFAULT 0.0,
                    liberal_policies INTEGER DEFAULT 0,
                    fascist_policies INTEGER DEFAULT 0,
                    game_data_json TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Turns table - stores each turn/phase of the game
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS turns (
                    turn_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_id TEXT NOT NULL,
                    turn_number INTEGER NOT NULL,
                    phase TEXT NOT NULL,
                    active_player TEXT,
                    action_type TEXT,
                    action_data_json TEXT,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (game_id) REFERENCES games(game_id)
                )
            """)

            # Player decisions table - stores individual player decisions
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS player_decisions (
                    decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_id TEXT NOT NULL,
                    player_id TEXT NOT NULL,
                    player_name TEXT NOT NULL,
                    turn_number INTEGER,
                    decision_type TEXT NOT NULL,
                    reasoning TEXT,
                    public_statement TEXT,
                    is_deception INTEGER DEFAULT 0,
                    deception_score REAL DEFAULT 0.0,
                    beliefs_json TEXT,
                    confidence REAL,
                    action TEXT,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (game_id) REFERENCES games(game_id)
                )
            """)

            # API requests table - tracks LLM API usage
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS api_requests (
                    request_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_id TEXT NOT NULL,
                    player_id TEXT NOT NULL,
                    model TEXT NOT NULL,
                    decision_type TEXT NOT NULL,
                    cost REAL NOT NULL,
                    tokens INTEGER,
                    latency REAL,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (game_id) REFERENCES games(game_id)
                )
            """)

            # Create indices for better query performance
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_turns_game_id
                ON turns(game_id)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_decisions_game_id
                ON player_decisions(game_id)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_decisions_player
                ON player_decisions(player_id)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_api_game_id
                ON api_requests(game_id)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_api_model
                ON api_requests(model)
            """)

            conn.commit()
            logger.info(f"Database initialized at {self.db_path}")

    @contextmanager
    def _get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row  # Enable column access by name
        try:
            yield conn
        finally:
            conn.close()

    def insert_game(self, game_data: Dict[str, Any]) -> bool:
        """
        Insert a game record into the database.

        Args:
            game_data: Dictionary containing game information

        Returns:
            True if successful, False otherwise
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # Extract models used
                models_used = json.dumps(list(set(
                    player.get('model', 'unknown')
                    for player in game_data.get('players', {}).values()
                    if isinstance(player, dict)
                )))

                cursor.execute("""
                    INSERT INTO games (
                        game_id, timestamp, player_count, models_used,
                        winner, winning_team, win_condition,
                        duration_seconds, total_actions, total_cost,
                        liberal_policies, fascist_policies, game_data_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    game_data.get('game_id'),
                    game_data.get('timestamp', datetime.now().isoformat()),
                    game_data.get('player_count', 0),
                    models_used,
                    game_data.get('winner'),
                    game_data.get('winning_team'),
                    game_data.get('win_condition'),
                    game_data.get('duration_seconds', 0),
                    game_data.get('total_actions', 0),
                    game_data.get('total_cost', 0.0),
                    game_data.get('liberal_policies', 0),
                    game_data.get('fascist_policies', 0),
                    json.dumps(game_data)
                ))

                conn.commit()
                logger.debug(f"Inserted game {game_data.get('game_id')}")
                return True

        except Exception as e:
            logger.error(f"Failed to insert game: {e}")
            return False

    def insert_player_decision(self, decision_data: Dict[str, Any]) -> bool:
        """Insert a player decision record."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    INSERT INTO player_decisions (
                        game_id, player_id, player_name, turn_number,
                        decision_type, reasoning, public_statement,
                        is_deception, deception_score, beliefs_json,
                        confidence, action, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    decision_data.get('game_id'),
                    decision_data.get('player_id'),
                    decision_data.get('player_name'),
                    decision_data.get('turn_number'),
                    decision_data.get('decision_type'),
                    decision_data.get('reasoning'),
                    decision_data.get('public_statement'),
                    1 if decision_data.get('is_deception') else 0,
                    decision_data.get('deception_score', 0.0),
                    json.dumps(decision_data.get('beliefs', {})),
                    decision_data.get('confidence'),
                    decision_data.get('action'),
                    decision_data.get('timestamp', datetime.now().isoformat())
                ))

                conn.commit()
                return True

        except Exception as e:
            logger.error(f"Failed to insert decision: {e}")
            return False

    def insert_api_request(self, request_data: Dict[str, Any]) -> bool:
        """Insert an API request record."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    INSERT INTO api_requests (
                        game_id, player_id, model, decision_type,
                        cost, tokens, latency, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    request_data.get('game_id'),
                    request_data.get('player_id'),
                    request_data.get('model'),
                    request_data.get('decision_type'),
                    request_data.get('cost', 0.0),
                    request_data.get('tokens', 0),
                    request_data.get('latency', 0.0),
                    request_data.get('timestamp', datetime.now().isoformat())
                ))

                conn.commit()
                return True

        except Exception as e:
            logger.error(f"Failed to insert API request: {e}")
            return False

    def get_game(self, game_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a game by ID."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM games WHERE game_id = ?", (game_id,))
                row = cursor.fetchone()

                if row:
                    return dict(row)
                return None

        except Exception as e:
            logger.error(f"Failed to get game: {e}")
            return None

    def get_all_games(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieve all games, optionally limited."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                query = "SELECT * FROM games ORDER BY timestamp DESC"
                if limit:
                    query += f" LIMIT {limit}"

                cursor.execute(query)
                return [dict(row) for row in cursor.fetchall()]

        except Exception as e:
            logger.error(f"Failed to get games: {e}")
            return []

    def get_player_decisions(self, game_id: str) -> List[Dict[str, Any]]:
        """Get all player decisions for a game."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM player_decisions
                    WHERE game_id = ?
                    ORDER BY timestamp
                """, (game_id,))

                return [dict(row) for row in cursor.fetchall()]

        except Exception as e:
            logger.error(f"Failed to get player decisions: {e}")
            return []

    def get_api_requests(self, game_id: str) -> List[Dict[str, Any]]:
        """Get all API requests for a game."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM api_requests
                    WHERE game_id = ?
                    ORDER BY timestamp
                """, (game_id,))

                return [dict(row) for row in cursor.fetchall()]

        except Exception as e:
            logger.error(f"Failed to get API requests: {e}")
            return []

    def get_game_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute("SELECT COUNT(*) FROM games")
                total_games = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(*) FROM player_decisions")
                total_decisions = cursor.fetchone()[0]

                cursor.execute("SELECT SUM(total_cost) FROM games")
                total_cost = cursor.fetchone()[0] or 0.0

                cursor.execute("SELECT COUNT(DISTINCT model) FROM api_requests")
                unique_models = cursor.fetchone()[0]

                return {
                    'total_games': total_games,
                    'total_decisions': total_decisions,
                    'total_cost': total_cost,
                    'unique_models': unique_models
                }

        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}

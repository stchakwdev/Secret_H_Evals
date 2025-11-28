"""
Data loader module for dashboard database queries.

Provides efficient database access with caching for the interactive dashboard.
"""

import sqlite3
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
from functools import lru_cache

# Default database path
DEFAULT_DB_PATH = Path(__file__).parent.parent / "data" / "games.db"


class DataLoader:
    """
    Database interface for dashboard queries.

    Provides cached access to game data with efficient aggregation queries.
    """

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize data loader.

        Args:
            db_path: Path to SQLite database (default: data/games.db)
        """
        self.db_path = db_path or DEFAULT_DB_PATH

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection with row factory."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def get_all_games(self) -> pd.DataFrame:
        """
        Get summary of all games.

        Returns:
            DataFrame with columns: game_id, timestamp, player_count, winner,
            winning_team, win_condition, duration_seconds, total_cost,
            liberal_policies, fascist_policies
        """
        query = """
            SELECT
                game_id, timestamp, player_count, winner, winning_team,
                win_condition, duration_seconds, total_actions, total_cost,
                liberal_policies, fascist_policies, models_used
            FROM games
            ORDER BY timestamp DESC
        """

        with self._get_connection() as conn:
            df = pd.read_sql_query(query, conn)

        # Parse models_used JSON
        if 'models_used' in df.columns:
            df['models_used'] = df['models_used'].apply(
                lambda x: json.loads(x) if x else []
            )

        return df

    def get_game_details(self, game_id: str) -> Dict[str, Any]:
        """
        Get complete details for a single game.

        Args:
            game_id: Game identifier

        Returns:
            Dictionary with game metadata and full game_data_json
        """
        query = """
            SELECT * FROM games WHERE game_id = ?
        """

        with self._get_connection() as conn:
            cursor = conn.execute(query, (game_id,))
            row = cursor.fetchone()

        if not row:
            return {}

        result = dict(row)

        # Parse JSON fields
        if result.get('game_data_json'):
            result['game_data'] = json.loads(result['game_data_json'])

        if result.get('models_used'):
            result['models_used'] = json.loads(result['models_used'])

        return result

    def get_player_decisions(
        self,
        game_id: Optional[str] = None,
        decision_type: Optional[str] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Get player decision records.

        Args:
            game_id: Filter by game (optional)
            decision_type: Filter by decision type (optional)
            limit: Maximum records to return

        Returns:
            DataFrame with decision records
        """
        query = "SELECT * FROM player_decisions WHERE 1=1"
        params = []

        if game_id:
            query += " AND game_id = ?"
            params.append(game_id)

        if decision_type:
            query += " AND decision_type = ?"
            params.append(decision_type)

        query += f" ORDER BY timestamp DESC LIMIT {limit}"

        with self._get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)

        # Parse JSON fields
        if 'beliefs_json' in df.columns:
            df['beliefs'] = df['beliefs_json'].apply(
                lambda x: json.loads(x) if x else {}
            )

        return df

    def get_api_requests(
        self,
        game_id: Optional[str] = None,
        model: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get API request records for cost analysis.

        Args:
            game_id: Filter by game (optional)
            model: Filter by model (optional)

        Returns:
            DataFrame with API request records
        """
        query = "SELECT * FROM api_requests WHERE 1=1"
        params = []

        if game_id:
            query += " AND game_id = ?"
            params.append(game_id)

        if model:
            query += " AND model = ?"
            params.append(model)

        query += " ORDER BY timestamp"

        with self._get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)

        return df

    def get_deception_summary(self) -> pd.DataFrame:
        """
        Get deception statistics aggregated by player and game.

        Returns:
            DataFrame with deception rates by player/game
        """
        query = """
            SELECT
                game_id,
                player_id,
                player_name,
                COUNT(*) as total_decisions,
                SUM(CASE WHEN is_deception = 1 THEN 1 ELSE 0 END) as deception_count,
                AVG(deception_score) as avg_deception_score
            FROM player_decisions
            GROUP BY game_id, player_id, player_name
        """

        with self._get_connection() as conn:
            df = pd.read_sql_query(query, conn)

        df['deception_rate'] = df['deception_count'] / df['total_decisions']
        return df

    def get_deception_by_decision_type(self) -> pd.DataFrame:
        """
        Get deception rates broken down by decision type.

        Returns:
            DataFrame with deception stats per decision type
        """
        query = """
            SELECT
                decision_type,
                COUNT(*) as total,
                SUM(CASE WHEN is_deception = 1 THEN 1 ELSE 0 END) as deceptions,
                AVG(deception_score) as avg_score
            FROM player_decisions
            GROUP BY decision_type
            ORDER BY total DESC
        """

        with self._get_connection() as conn:
            df = pd.read_sql_query(query, conn)

        df['deception_rate'] = df['deceptions'] / df['total']
        return df

    def get_model_comparison(self) -> pd.DataFrame:
        """
        Get win rates and costs by model.

        Returns:
            DataFrame with model performance metrics
        """
        # First get unique models from games
        games_df = self.get_all_games()

        # Flatten models and count wins
        model_stats = {}

        for _, game in games_df.iterrows():
            models = game.get('models_used', [])
            if isinstance(models, str):
                models = [models]

            winner = game.get('winning_team', 'unknown')

            for model in models:
                if model not in model_stats:
                    model_stats[model] = {
                        'games': 0,
                        'liberal_wins': 0,
                        'fascist_wins': 0,
                        'total_cost': 0.0
                    }

                model_stats[model]['games'] += 1
                if winner == 'liberal':
                    model_stats[model]['liberal_wins'] += 1
                elif winner == 'fascist':
                    model_stats[model]['fascist_wins'] += 1
                model_stats[model]['total_cost'] += game.get('total_cost', 0) or 0

        # Convert to DataFrame
        rows = []
        for model, stats in model_stats.items():
            rows.append({
                'model': model,
                'games': stats['games'],
                'liberal_wins': stats['liberal_wins'],
                'fascist_wins': stats['fascist_wins'],
                'liberal_win_rate': stats['liberal_wins'] / stats['games'] if stats['games'] > 0 else 0,
                'total_cost': stats['total_cost'],
                'avg_cost_per_game': stats['total_cost'] / stats['games'] if stats['games'] > 0 else 0
            })

        return pd.DataFrame(rows)

    def get_turn_timeline(self, game_id: str) -> pd.DataFrame:
        """
        Get turn-by-turn timeline for a game.

        Args:
            game_id: Game identifier

        Returns:
            DataFrame with turn records
        """
        query = """
            SELECT * FROM turns
            WHERE game_id = ?
            ORDER BY turn_number
        """

        with self._get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=[game_id])

        # Parse JSON fields
        if 'action_data_json' in df.columns:
            df['action_data'] = df['action_data_json'].apply(
                lambda x: json.loads(x) if x else {}
            )

        return df

    def get_trust_evolution(self, game_id: str) -> Dict[str, List[Dict]]:
        """
        Extract trust beliefs over time for a game.

        Args:
            game_id: Game identifier

        Returns:
            Dictionary mapping player_id to list of belief snapshots
        """
        df = self.get_player_decisions(game_id=game_id)

        trust_evolution = {}

        for _, row in df.iterrows():
            player_id = row.get('player_id', 'unknown')
            turn = row.get('turn_number', 0)
            beliefs = row.get('beliefs', {})

            if player_id not in trust_evolution:
                trust_evolution[player_id] = []

            trust_evolution[player_id].append({
                'turn': turn,
                'beliefs': beliefs,
                'decision_type': row.get('decision_type'),
                'timestamp': row.get('timestamp')
            })

        # Sort by turn
        for player_id in trust_evolution:
            trust_evolution[player_id].sort(key=lambda x: x['turn'])

        return trust_evolution

    def get_aggregate_statistics(self) -> Dict[str, Any]:
        """
        Get high-level aggregate statistics.

        Returns:
            Dictionary with summary statistics
        """
        games_df = self.get_all_games()

        if games_df.empty:
            return {"error": "No games found"}

        n_games = len(games_df)
        liberal_wins = (games_df['winning_team'] == 'liberal').sum()
        fascist_wins = (games_df['winning_team'] == 'fascist').sum()

        return {
            "total_games": n_games,
            "liberal_wins": int(liberal_wins),
            "fascist_wins": int(fascist_wins),
            "liberal_win_rate": liberal_wins / n_games if n_games > 0 else 0,
            "avg_duration": games_df['duration_seconds'].mean(),
            "total_cost": games_df['total_cost'].sum(),
            "avg_cost_per_game": games_df['total_cost'].mean(),
            "unique_models": games_df['models_used'].apply(
                lambda x: len(x) if isinstance(x, list) else 1
            ).sum(),
            "avg_players": games_df['player_count'].mean(),
            "date_range": {
                "earliest": games_df['timestamp'].min(),
                "latest": games_df['timestamp'].max()
            }
        }

    def get_deception_timeline_data(self, game_id: str) -> pd.DataFrame:
        """
        Get deception data formatted for timeline visualization.

        Args:
            game_id: Game identifier

        Returns:
            DataFrame with columns: turn, player_name, deception_score, is_deception
        """
        query = """
            SELECT
                turn_number as turn,
                player_name,
                deception_score,
                is_deception,
                reasoning,
                public_statement,
                decision_type
            FROM player_decisions
            WHERE game_id = ?
            ORDER BY turn_number, player_name
        """

        with self._get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=[game_id])

        return df

    def get_voting_history(self, game_id: str) -> List[Dict[str, Any]]:
        """
        Extract voting history for coalition analysis.

        Args:
            game_id: Game identifier

        Returns:
            List of voting rounds with format:
            [{'round': int, 'votes': {player: bool}}]
        """
        # Get game data
        game = self.get_game_details(game_id)
        game_data = game.get('game_data', {})

        voting_rounds = game_data.get('voting_rounds', [])

        result = []
        for i, round_data in enumerate(voting_rounds):
            votes = {}
            for vote in round_data.get('votes', []):
                player = vote.get('player_id') or vote.get('player_name')
                votes[player] = vote.get('vote', False)

            result.append({
                'round': i + 1,
                'votes': votes
            })

        return result

    def get_player_roles(self, game_id: str) -> Dict[str, str]:
        """
        Get player role assignments for a game.

        Args:
            game_id: Game identifier

        Returns:
            Dictionary mapping player_name to role
        """
        game = self.get_game_details(game_id)
        game_data = game.get('game_data', {})

        roles = {}
        players = game_data.get('players', {})

        for player_id, player_info in players.items():
            name = player_info.get('name', player_id)
            role = player_info.get('role', 'unknown')
            roles[name] = role

        return roles


# Singleton instance for easy import
_data_loader: Optional[DataLoader] = None


def get_data_loader(db_path: Optional[Path] = None) -> DataLoader:
    """Get or create singleton DataLoader instance."""
    global _data_loader
    if _data_loader is None or db_path is not None:
        _data_loader = DataLoader(db_path)
    return _data_loader

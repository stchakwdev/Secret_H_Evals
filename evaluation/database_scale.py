"""
High-performance database operations for 5000+ game scale.
Provides connection pooling, batch operations, and streaming queries.
"""
import sqlite3
import json
import threading
import queue
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Iterator, Callable
from datetime import datetime
from contextlib import contextmanager
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class ConnectionPoolStats:
    """Statistics for connection pool monitoring."""
    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    total_checkouts: int = 0
    total_checkins: int = 0
    wait_time_total: float = 0.0
    peak_active: int = 0


class ConnectionPool:
    """
    Thread-safe SQLite connection pool for concurrent access.

    Features:
    - Configurable pool size
    - Connection timeout handling
    - Automatic connection recycling
    - Health checks
    """

    def __init__(
        self,
        db_path: str,
        pool_size: int = 5,
        max_overflow: int = 10,
        timeout: float = 30.0,
        recycle_time: int = 3600
    ):
        self.db_path = db_path
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.timeout = timeout
        self.recycle_time = recycle_time

        self._pool: queue.Queue = queue.Queue(maxsize=pool_size + max_overflow)
        self._lock = threading.Lock()
        self._created_count = 0
        self._stats = ConnectionPoolStats()

        # Pre-create pool connections
        for _ in range(pool_size):
            self._add_connection()

    def _create_connection(self) -> sqlite3.Connection:
        """Create a new database connection with optimal settings."""
        conn = sqlite3.connect(
            self.db_path,
            timeout=self.timeout,
            check_same_thread=False,
            isolation_level=None  # Autocommit mode for better concurrency
        )
        conn.row_factory = sqlite3.Row

        # Optimize for concurrent writes
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=10000")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA mmap_size=268435456")  # 256MB memory-mapped I/O

        return conn

    def _add_connection(self):
        """Add a new connection to the pool."""
        conn = self._create_connection()
        conn_info = {
            'connection': conn,
            'created_at': time.time(),
            'last_used': time.time()
        }
        self._pool.put(conn_info)
        with self._lock:
            self._created_count += 1
            self._stats.total_connections += 1
            self._stats.idle_connections += 1

    @contextmanager
    def get_connection(self):
        """Get a connection from the pool with automatic return."""
        start_wait = time.time()
        conn_info = None

        try:
            # Try to get from pool
            try:
                conn_info = self._pool.get(timeout=self.timeout)
            except queue.Empty:
                # Create overflow connection if allowed
                with self._lock:
                    if self._created_count < self.pool_size + self.max_overflow:
                        self._add_connection()
                        conn_info = self._pool.get(timeout=self.timeout)
                    else:
                        raise TimeoutError("Connection pool exhausted")

            # Update stats
            with self._lock:
                wait_time = time.time() - start_wait
                self._stats.wait_time_total += wait_time
                self._stats.total_checkouts += 1
                self._stats.idle_connections -= 1
                self._stats.active_connections += 1
                self._stats.peak_active = max(
                    self._stats.peak_active,
                    self._stats.active_connections
                )

            # Check if connection needs recycling
            if time.time() - conn_info['created_at'] > self.recycle_time:
                conn_info['connection'].close()
                conn_info['connection'] = self._create_connection()
                conn_info['created_at'] = time.time()

            conn_info['last_used'] = time.time()
            yield conn_info['connection']

        finally:
            if conn_info:
                self._pool.put(conn_info)
                with self._lock:
                    self._stats.total_checkins += 1
                    self._stats.active_connections -= 1
                    self._stats.idle_connections += 1

    def get_stats(self) -> ConnectionPoolStats:
        """Get current pool statistics."""
        with self._lock:
            return ConnectionPoolStats(
                total_connections=self._stats.total_connections,
                active_connections=self._stats.active_connections,
                idle_connections=self._stats.idle_connections,
                total_checkouts=self._stats.total_checkouts,
                total_checkins=self._stats.total_checkins,
                wait_time_total=self._stats.wait_time_total,
                peak_active=self._stats.peak_active
            )

    def close_all(self):
        """Close all connections in the pool."""
        while not self._pool.empty():
            try:
                conn_info = self._pool.get_nowait()
                conn_info['connection'].close()
            except queue.Empty:
                break
        logger.info("Connection pool closed")


class ScaleDatabaseManager:
    """
    High-performance database manager optimized for 5000+ games.

    Features:
    - Connection pooling for concurrent access
    - Batch insert operations
    - Streaming queries for memory efficiency
    - Statistics caching
    - Progress tracking
    """

    def __init__(
        self,
        db_path: str = "data/games.db",
        pool_size: int = 5,
        batch_size: int = 100
    ):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size

        # Initialize connection pool
        self.pool = ConnectionPool(
            str(self.db_path),
            pool_size=pool_size,
            max_overflow=pool_size * 2
        )

        # Statistics cache
        self._stats_cache: Optional[Dict[str, Any]] = None
        self._stats_cache_time: float = 0
        self._stats_cache_ttl: float = 60.0  # 1 minute cache

        # Ensure schema exists
        self._ensure_schema()

        logger.info(f"ScaleDatabaseManager initialized with pool_size={pool_size}")

    def _ensure_schema(self):
        """Ensure database schema exists."""
        with self.pool.get_connection() as conn:
            cursor = conn.cursor()

            # Check if games table exists
            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='games'
            """)

            if not cursor.fetchone():
                # Import and run schema creation from main module
                from evaluation.database_schema import DatabaseManager
                temp_db = DatabaseManager(str(self.db_path))
                temp_db.enable_wal_mode()
                logger.info("Database schema created")

    # ==================== Batch Insert Operations ====================

    def batch_insert_games(self, games: List[Dict[str, Any]]) -> int:
        """
        Insert multiple games in a single transaction.

        Args:
            games: List of game data dictionaries

        Returns:
            Number of games inserted
        """
        if not games:
            return 0

        inserted = 0
        with self.pool.get_connection() as conn:
            cursor = conn.cursor()

            try:
                cursor.execute("BEGIN TRANSACTION")

                for game_data in games:
                    models_used = json.dumps(list(set(
                        player.get('model', 'unknown')
                        for player in game_data.get('players', {}).values()
                        if isinstance(player, dict)
                    )))

                    cursor.execute("""
                        INSERT OR REPLACE INTO games (
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
                    inserted += 1

                cursor.execute("COMMIT")
                self._invalidate_stats_cache()
                logger.debug(f"Batch inserted {inserted} games")

            except Exception as e:
                cursor.execute("ROLLBACK")
                logger.error(f"Batch insert failed: {e}")
                raise

        return inserted

    def batch_insert_decisions(self, decisions: List[Dict[str, Any]]) -> int:
        """Insert multiple player decisions in a single transaction."""
        if not decisions:
            return 0

        inserted = 0
        with self.pool.get_connection() as conn:
            cursor = conn.cursor()

            try:
                cursor.execute("BEGIN TRANSACTION")

                for decision in decisions:
                    cursor.execute("""
                        INSERT INTO player_decisions (
                            game_id, player_id, player_name, turn_number,
                            decision_type, reasoning, public_statement,
                            is_deception, deception_score, beliefs_json,
                            confidence, action, timestamp
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        decision.get('game_id'),
                        decision.get('player_id'),
                        decision.get('player_name'),
                        decision.get('turn_number'),
                        decision.get('decision_type'),
                        decision.get('reasoning'),
                        decision.get('public_statement'),
                        1 if decision.get('is_deception') else 0,
                        decision.get('deception_score', 0.0),
                        json.dumps(decision.get('beliefs', {})),
                        decision.get('confidence'),
                        decision.get('action'),
                        decision.get('timestamp', datetime.now().isoformat())
                    ))
                    inserted += 1

                cursor.execute("COMMIT")
                logger.debug(f"Batch inserted {inserted} decisions")

            except Exception as e:
                cursor.execute("ROLLBACK")
                logger.error(f"Batch insert decisions failed: {e}")
                raise

        return inserted

    def batch_insert_prompts(self, prompts: List[Dict[str, Any]]) -> int:
        """Insert multiple prompts in a single transaction."""
        if not prompts:
            return 0

        import hashlib

        inserted = 0
        with self.pool.get_connection() as conn:
            cursor = conn.cursor()

            try:
                cursor.execute("BEGIN TRANSACTION")

                for prompt_data in prompts:
                    prompt_text = prompt_data.get('prompt_text', '')
                    prompt_hash = prompt_data.get('prompt_hash')
                    if not prompt_hash and prompt_text:
                        prompt_hash = hashlib.sha256(prompt_text.encode()).hexdigest()[:16]

                    cursor.execute("""
                        INSERT INTO prompts (
                            game_id, player_id, turn_number, decision_type,
                            prompt_text, response_text, model, temperature,
                            max_tokens, prompt_hash, prompt_tokens,
                            completion_tokens, timestamp
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        prompt_data.get('game_id'),
                        prompt_data.get('player_id'),
                        prompt_data.get('turn_number'),
                        prompt_data.get('decision_type'),
                        prompt_text,
                        prompt_data.get('response_text'),
                        prompt_data.get('model'),
                        prompt_data.get('temperature'),
                        prompt_data.get('max_tokens'),
                        prompt_hash,
                        prompt_data.get('prompt_tokens'),
                        prompt_data.get('completion_tokens'),
                        prompt_data.get('timestamp', datetime.now().isoformat())
                    ))
                    inserted += 1

                cursor.execute("COMMIT")
                logger.debug(f"Batch inserted {inserted} prompts")

            except Exception as e:
                cursor.execute("ROLLBACK")
                logger.error(f"Batch insert prompts failed: {e}")
                raise

        return inserted

    # ==================== Streaming Queries ====================

    def stream_games(
        self,
        batch_size: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> Iterator[List[Dict[str, Any]]]:
        """
        Stream games in batches for memory-efficient processing.

        Args:
            batch_size: Number of games per batch (default: self.batch_size)
            filters: Optional filters (winning_team, model, date_range)

        Yields:
            Batches of game dictionaries
        """
        batch_size = batch_size or self.batch_size

        # Build query with filters
        query = "SELECT * FROM games"
        params = []
        conditions = []

        if filters:
            if 'winning_team' in filters:
                conditions.append("winning_team = ?")
                params.append(filters['winning_team'])
            if 'model' in filters:
                conditions.append("models_used LIKE ?")
                params.append(f'%{filters["model"]}%')
            if 'start_date' in filters:
                conditions.append("timestamp >= ?")
                params.append(filters['start_date'])
            if 'end_date' in filters:
                conditions.append("timestamp <= ?")
                params.append(filters['end_date'])

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY timestamp"

        with self.pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)

            while True:
                rows = cursor.fetchmany(batch_size)
                if not rows:
                    break
                yield [dict(row) for row in rows]

    def stream_decisions(
        self,
        game_id: Optional[str] = None,
        batch_size: Optional[int] = None
    ) -> Iterator[List[Dict[str, Any]]]:
        """Stream player decisions in batches."""
        batch_size = batch_size or self.batch_size

        query = "SELECT * FROM player_decisions"
        params = []

        if game_id:
            query += " WHERE game_id = ?"
            params.append(game_id)

        query += " ORDER BY timestamp"

        with self.pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)

            while True:
                rows = cursor.fetchmany(batch_size)
                if not rows:
                    break
                yield [dict(row) for row in rows]

    def stream_aggregate(
        self,
        aggregation: str,
        group_by: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> Iterator[Dict[str, Any]]:
        """
        Stream aggregated results for large-scale analysis.

        Args:
            aggregation: SQL aggregation (e.g., "COUNT(*) as count, AVG(total_cost) as avg_cost")
            group_by: Column to group by
            filters: Optional filters

        Yields:
            Aggregated result dictionaries
        """
        query = f"SELECT {aggregation} FROM games"
        params = []
        conditions = []

        if filters:
            if 'winning_team' in filters:
                conditions.append("winning_team = ?")
                params.append(filters['winning_team'])

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        if group_by:
            query += f" GROUP BY {group_by}"

        with self.pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)

            for row in cursor:
                yield dict(row)

    # ==================== Statistics with Caching ====================

    def _invalidate_stats_cache(self):
        """Invalidate the statistics cache."""
        self._stats_cache = None
        self._stats_cache_time = 0

    def get_stats(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Get database statistics with caching.

        Args:
            force_refresh: Force cache refresh

        Returns:
            Statistics dictionary
        """
        now = time.time()

        if (not force_refresh and
            self._stats_cache and
            now - self._stats_cache_time < self._stats_cache_ttl):
            return self._stats_cache

        with self.pool.get_connection() as conn:
            cursor = conn.cursor()

            stats = {}

            # Game counts
            cursor.execute("SELECT COUNT(*) FROM games")
            stats['total_games'] = cursor.fetchone()[0]

            # Win rates
            cursor.execute("""
                SELECT winning_team, COUNT(*) as count
                FROM games
                WHERE winning_team IS NOT NULL
                GROUP BY winning_team
            """)
            stats['wins_by_team'] = {row[0]: row[1] for row in cursor.fetchall()}

            # Cost statistics
            cursor.execute("""
                SELECT
                    SUM(total_cost) as total_cost,
                    AVG(total_cost) as avg_cost,
                    MIN(total_cost) as min_cost,
                    MAX(total_cost) as max_cost
                FROM games
            """)
            row = cursor.fetchone()
            stats['cost'] = {
                'total': row[0] or 0,
                'avg': row[1] or 0,
                'min': row[2] or 0,
                'max': row[3] or 0
            }

            # Decision counts
            cursor.execute("SELECT COUNT(*) FROM player_decisions")
            stats['total_decisions'] = cursor.fetchone()[0]

            # Deception statistics
            cursor.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(is_deception) as deceptions,
                    AVG(deception_score) as avg_score
                FROM player_decisions
            """)
            row = cursor.fetchone()
            stats['deception'] = {
                'total_decisions': row[0] or 0,
                'deception_count': row[1] or 0,
                'deception_rate': (row[1] or 0) / (row[0] or 1),
                'avg_score': row[2] or 0
            }

            # Model statistics
            cursor.execute("""
                SELECT model, COUNT(*) as count, SUM(cost) as total_cost
                FROM api_requests
                GROUP BY model
                ORDER BY count DESC
            """)
            stats['models'] = [
                {'model': row[0], 'requests': row[1], 'cost': row[2]}
                for row in cursor.fetchall()
            ]

            # Prompt statistics
            cursor.execute("SELECT COUNT(*) FROM prompts")
            stats['total_prompts'] = cursor.fetchone()[0]

            # Connection pool stats
            pool_stats = self.pool.get_stats()
            stats['pool'] = {
                'total_connections': pool_stats.total_connections,
                'active': pool_stats.active_connections,
                'idle': pool_stats.idle_connections,
                'peak_active': pool_stats.peak_active,
                'avg_wait_ms': (pool_stats.wait_time_total /
                               max(pool_stats.total_checkouts, 1)) * 1000
            }

        self._stats_cache = stats
        self._stats_cache_time = now

        return stats

    # ==================== Progress Tracking ====================

    def save_batch_progress(
        self,
        batch_id: str,
        progress: Dict[str, Any]
    ):
        """Save batch progress for crash recovery."""
        progress_file = self.db_path.parent / f".batch_progress_{batch_id}.json"

        progress['updated_at'] = datetime.now().isoformat()

        with open(progress_file, 'w') as f:
            json.dump(progress, f, indent=2)

        logger.debug(f"Saved progress for batch {batch_id}")

    def load_batch_progress(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """Load batch progress for resumption."""
        progress_file = self.db_path.parent / f".batch_progress_{batch_id}.json"

        if progress_file.exists():
            with open(progress_file, 'r') as f:
                return json.load(f)

        return None

    def clear_batch_progress(self, batch_id: str):
        """Clear batch progress file after completion."""
        progress_file = self.db_path.parent / f".batch_progress_{batch_id}.json"

        if progress_file.exists():
            progress_file.unlink()
            logger.debug(f"Cleared progress for batch {batch_id}")

    # ==================== Maintenance ====================

    def optimize(self):
        """Run database optimization."""
        with self.pool.get_connection() as conn:
            conn.execute("ANALYZE")
            logger.info("Database analyzed for query optimization")

    def vacuum(self):
        """Reclaim space from deleted records."""
        # Vacuum requires exclusive access, use direct connection
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("VACUUM")
        conn.close()
        logger.info("Database vacuumed")

    def close(self):
        """Close the database manager and connection pool."""
        self.pool.close_all()
        logger.info("ScaleDatabaseManager closed")


# Singleton instance for shared access
_scale_db: Optional[ScaleDatabaseManager] = None


def get_scale_db(
    db_path: str = "data/games.db",
    pool_size: int = 5
) -> ScaleDatabaseManager:
    """Get or create the singleton ScaleDatabaseManager instance."""
    global _scale_db

    if _scale_db is None:
        _scale_db = ScaleDatabaseManager(db_path, pool_size)

    return _scale_db

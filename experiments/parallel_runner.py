"""
Parallel batch runner for large-scale game execution (5000+ games).

Features:
- Configurable concurrency with asyncio.Semaphore
- Rate limiting for API quotas
- Progress persistence for crash recovery
- Retry logic with exponential backoff
- Real-time progress reporting
- Resource monitoring
"""
import asyncio
import json
import os
import sys
import time
import uuid
import signal
import logging
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Callable, Tuple
from enum import Enum
import traceback

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.game_manager import GameManager
from evaluation.database_scale import get_scale_db, ScaleDatabaseManager

logger = logging.getLogger(__name__)


class GameStatus(Enum):
    """Status of individual game execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class GameResult:
    """Result of a single game execution."""
    game_id: str
    status: GameStatus
    winner: Optional[str] = None
    winning_team: Optional[str] = None
    duration_seconds: float = 0.0
    total_cost: float = 0.0
    error: Optional[str] = None
    attempts: int = 1
    completed_at: Optional[str] = None


@dataclass
class BatchProgress:
    """Progress tracking for batch execution."""
    batch_id: str
    total_games: int
    completed: int = 0
    failed: int = 0
    retrying: int = 0
    running: int = 0
    pending: int = 0
    start_time: str = ""
    last_update: str = ""
    games: Dict[str, Dict] = field(default_factory=dict)
    total_cost: float = 0.0
    liberal_wins: int = 0
    fascist_wins: int = 0
    avg_duration: float = 0.0
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'batch_id': self.batch_id,
            'total_games': self.total_games,
            'completed': self.completed,
            'failed': self.failed,
            'retrying': self.retrying,
            'running': self.running,
            'pending': self.pending,
            'start_time': self.start_time,
            'last_update': self.last_update,
            'total_cost': self.total_cost,
            'liberal_wins': self.liberal_wins,
            'fascist_wins': self.fascist_wins,
            'avg_duration': self.avg_duration,
            'errors': self.errors[-10:],  # Keep last 10 errors
            'games': {k: v for k, v in list(self.games.items())[-100:]}  # Last 100 games
        }


@dataclass
class RateLimiter:
    """
    Token bucket rate limiter for API requests.

    Args:
        requests_per_minute: Maximum requests per minute
        burst_size: Maximum burst size
    """
    requests_per_minute: int = 60
    burst_size: int = 10
    _tokens: float = field(default=0.0, init=False)
    _last_update: float = field(default=0.0, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    def __post_init__(self):
        self._tokens = float(self.burst_size)
        self._last_update = time.time()

    async def acquire(self):
        """Acquire a token, waiting if necessary."""
        async with self._lock:
            now = time.time()
            elapsed = now - self._last_update
            self._last_update = now

            # Add tokens based on elapsed time
            self._tokens += elapsed * (self.requests_per_minute / 60.0)
            self._tokens = min(self._tokens, float(self.burst_size))

            if self._tokens < 1.0:
                # Need to wait for token
                wait_time = (1.0 - self._tokens) / (self.requests_per_minute / 60.0)
                await asyncio.sleep(wait_time)
                self._tokens = 0.0
            else:
                self._tokens -= 1.0


class ParallelBatchRunner:
    """
    High-performance parallel batch runner for large-scale evaluations.

    Features:
    - Concurrent game execution with configurable parallelism
    - Rate limiting to respect API quotas
    - Automatic retry with exponential backoff
    - Progress persistence for crash recovery
    - Real-time progress callbacks
    - Graceful shutdown handling
    """

    def __init__(
        self,
        api_key: str,
        num_players: int = 5,
        model: str = "deepseek/deepseek-v3.2-exp",
        concurrency: int = 3,
        requests_per_minute: int = 60,
        max_retries: int = 3,
        retry_base_delay: float = 5.0,
        enable_db_logging: bool = True,
        db_path: str = "data/games.db",
        progress_callback: Optional[Callable[[BatchProgress], None]] = None
    ):
        self.api_key = api_key
        self.num_players = num_players
        self.model = model
        self.concurrency = concurrency
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.enable_db_logging = enable_db_logging
        self.db_path = db_path
        self.progress_callback = progress_callback

        # Rate limiter
        self.rate_limiter = RateLimiter(
            requests_per_minute=requests_per_minute,
            burst_size=min(10, concurrency * 2)
        )

        # Concurrency control
        self.semaphore = asyncio.Semaphore(concurrency)

        # Progress tracking
        self.progress: Optional[BatchProgress] = None
        self.progress_file: Optional[Path] = None

        # Database manager
        self.db: Optional[ScaleDatabaseManager] = None

        # Shutdown handling
        self._shutdown_event = asyncio.Event()
        self._running_tasks: List[asyncio.Task] = []

        # Player names pool
        self.player_names = [
            "Alice", "Bob", "Charlie", "Diana", "Eve",
            "Frank", "Grace", "Henry", "Iris", "Jack",
            "Kate", "Leo", "Maya", "Nick", "Olivia"
        ]

    async def run_batch(
        self,
        num_games: int,
        batch_id: Optional[str] = None,
        batch_tag: Optional[str] = None,
        resume: bool = True
    ) -> BatchProgress:
        """
        Run a batch of games with parallel execution.

        Args:
            num_games: Total number of games to run
            batch_id: Optional batch identifier (auto-generated if not provided)
            batch_tag: Optional human-readable tag
            resume: Whether to resume from previous progress

        Returns:
            Final BatchProgress with results
        """
        # Generate batch ID
        if batch_id is None:
            batch_id = f"batch-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"

        # Initialize database
        if self.enable_db_logging:
            self.db = get_scale_db(self.db_path, pool_size=self.concurrency + 2)

        # Setup progress tracking
        self.progress_file = Path(__file__).parent.parent / "logs" / f".progress_{batch_id}.json"
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)

        # Try to resume or create new progress
        if resume and self.progress_file.exists():
            self.progress = self._load_progress()
            logger.info(f"Resuming batch {batch_id}: {self.progress.completed}/{self.progress.total_games} completed")
        else:
            self.progress = BatchProgress(
                batch_id=batch_id,
                total_games=num_games,
                pending=num_games,
                start_time=datetime.now().isoformat()
            )

        # Setup signal handlers for graceful shutdown
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._handle_shutdown)
            except NotImplementedError:
                # Windows doesn't support add_signal_handler
                pass

        # Write initial metadata
        self._write_batch_metadata(batch_id, batch_tag, num_games)

        logger.info(f"Starting batch {batch_id}: {num_games} games, concurrency={self.concurrency}")

        try:
            # Create game tasks
            game_ids = [
                f"{batch_id}-game-{i+1:04d}"
                for i in range(num_games)
            ]

            # Filter out already completed games
            pending_games = [
                gid for gid in game_ids
                if gid not in self.progress.games or
                   self.progress.games[gid].get('status') != 'completed'
            ]

            # Run games with controlled concurrency
            tasks = [
                asyncio.create_task(self._run_game_with_retry(game_id))
                for game_id in pending_games
            ]
            self._running_tasks = tasks

            # Wait for all tasks with progress updates
            completed = 0
            for coro in asyncio.as_completed(tasks):
                if self._shutdown_event.is_set():
                    logger.info("Shutdown requested, stopping batch...")
                    break

                try:
                    result = await coro
                    completed += 1

                    # Update progress
                    self._update_progress(result)

                    # Log progress
                    if completed % 10 == 0 or completed == len(tasks):
                        self._log_progress()

                except Exception as e:
                    logger.error(f"Task error: {e}")

            # Final progress save
            self._save_progress()

        except Exception as e:
            logger.error(f"Batch execution error: {e}")
            traceback.print_exc()

        finally:
            # Cleanup
            if self.db:
                self.db.close()

            # Remove progress file if complete
            if (self.progress.completed + self.progress.failed >= self.progress.total_games
                and self.progress_file.exists()):
                self.progress_file.unlink()

        return self.progress

    async def _run_game_with_retry(self, game_id: str) -> GameResult:
        """Run a single game with retry logic."""
        result = GameResult(game_id=game_id, status=GameStatus.PENDING)

        for attempt in range(1, self.max_retries + 1):
            # Check for shutdown
            if self._shutdown_event.is_set():
                result.status = GameStatus.PENDING
                return result

            # Acquire rate limit token
            await self.rate_limiter.acquire()

            # Acquire concurrency semaphore
            async with self.semaphore:
                result.status = GameStatus.RUNNING
                result.attempts = attempt

                try:
                    game_result = await self._run_single_game(game_id)

                    result.status = GameStatus.COMPLETED
                    result.winner = game_result.get('winner')
                    result.winning_team = game_result.get('winning_team')
                    result.duration_seconds = game_result.get('duration_seconds', 0)
                    result.total_cost = game_result.get('cost_summary', {}).get('total_cost', 0)
                    result.completed_at = datetime.now().isoformat()

                    return result

                except Exception as e:
                    error_msg = str(e)
                    logger.warning(f"Game {game_id} attempt {attempt} failed: {error_msg}")

                    if attempt < self.max_retries:
                        result.status = GameStatus.RETRYING
                        delay = self.retry_base_delay * (2 ** (attempt - 1))  # Exponential backoff
                        await asyncio.sleep(delay)
                    else:
                        result.status = GameStatus.FAILED
                        result.error = error_msg
                        return result

        return result

    async def _run_single_game(self, game_id: str) -> Dict[str, Any]:
        """Execute a single game."""
        # Create player configurations
        player_configs = [
            {
                "id": f"{game_id}-player{i+1}",
                "name": self.player_names[i % len(self.player_names)],
                "model": self.model,
                "type": "ai"
            }
            for i in range(self.num_players)
        ]

        # Create game manager
        game_manager = GameManager(
            player_configs=player_configs,
            openrouter_api_key=self.api_key,
            enable_database_logging=self.enable_db_logging,
            game_id=game_id
        )

        # Run game
        start_time = time.time()
        result = await game_manager.start_game()
        duration = time.time() - start_time

        result['duration_seconds'] = duration
        result['game_id'] = game_id

        return result

    def _update_progress(self, result: GameResult):
        """Update batch progress with game result."""
        self.progress.games[result.game_id] = {
            'status': result.status.value,
            'winner': result.winner,
            'winning_team': result.winning_team,
            'duration': result.duration_seconds,
            'cost': result.total_cost,
            'attempts': result.attempts,
            'error': result.error,
            'completed_at': result.completed_at
        }

        # Update counters
        if result.status == GameStatus.COMPLETED:
            self.progress.completed += 1
            self.progress.total_cost += result.total_cost

            if result.winning_team == 'liberal':
                self.progress.liberal_wins += 1
            elif result.winning_team == 'fascist':
                self.progress.fascist_wins += 1

            # Update average duration
            completed_durations = [
                g['duration'] for g in self.progress.games.values()
                if g.get('status') == 'completed' and g.get('duration', 0) > 0
            ]
            if completed_durations:
                self.progress.avg_duration = sum(completed_durations) / len(completed_durations)

        elif result.status == GameStatus.FAILED:
            self.progress.failed += 1
            if result.error:
                self.progress.errors.append(f"{result.game_id}: {result.error}")

        # Update pending count
        self.progress.pending = self.progress.total_games - self.progress.completed - self.progress.failed

        # Update timestamp
        self.progress.last_update = datetime.now().isoformat()

        # Save progress
        self._save_progress()

        # Call callback if provided
        if self.progress_callback:
            self.progress_callback(self.progress)

    def _save_progress(self):
        """Save progress to file."""
        if self.progress_file:
            with open(self.progress_file, 'w') as f:
                json.dump(self.progress.to_dict(), f, indent=2)

    def _load_progress(self) -> BatchProgress:
        """Load progress from file."""
        with open(self.progress_file, 'r') as f:
            data = json.load(f)

        return BatchProgress(
            batch_id=data['batch_id'],
            total_games=data['total_games'],
            completed=data.get('completed', 0),
            failed=data.get('failed', 0),
            retrying=data.get('retrying', 0),
            running=data.get('running', 0),
            pending=data.get('pending', 0),
            start_time=data.get('start_time', ''),
            last_update=data.get('last_update', ''),
            games=data.get('games', {}),
            total_cost=data.get('total_cost', 0.0),
            liberal_wins=data.get('liberal_wins', 0),
            fascist_wins=data.get('fascist_wins', 0),
            avg_duration=data.get('avg_duration', 0.0),
            errors=data.get('errors', [])
        )

    def _write_batch_metadata(self, batch_id: str, batch_tag: Optional[str], num_games: int):
        """Write batch metadata for progress tracker."""
        logs_dir = Path(__file__).parent.parent / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "batch_id": batch_id,
            "batch_tag": batch_tag,
            "start_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "target_games": num_games,
            "players": self.num_players,
            "model": self.model,
            "concurrency": self.concurrency,
            "database_logging": self.enable_db_logging,
            "log_dir": str(logs_dir)
        }

        with open(logs_dir / ".current_batch", 'w') as f:
            json.dump(metadata, f, indent=2)

    def _log_progress(self):
        """Log current progress to console."""
        p = self.progress
        elapsed = 0
        if p.start_time:
            start = datetime.fromisoformat(p.start_time)
            elapsed = (datetime.now() - start).total_seconds()

        games_per_hour = (p.completed / elapsed) * 3600 if elapsed > 0 else 0
        eta_seconds = (p.pending / games_per_hour) * 3600 if games_per_hour > 0 else 0

        print(f"\n{'='*60}")
        print(f"Batch Progress: {p.completed}/{p.total_games} ({p.completed/p.total_games*100:.1f}%)")
        print(f"{'='*60}")
        print(f"  Completed: {p.completed} | Failed: {p.failed} | Pending: {p.pending}")
        print(f"  Liberal wins: {p.liberal_wins} | Fascist wins: {p.fascist_wins}")
        print(f"  Total cost: ${p.total_cost:.4f} | Avg duration: {p.avg_duration/60:.1f} min")
        print(f"  Rate: {games_per_hour:.1f} games/hour | ETA: {eta_seconds/3600:.1f} hours")
        print(f"{'='*60}\n")

    def _handle_shutdown(self):
        """Handle graceful shutdown request."""
        logger.info("Shutdown signal received, completing current games...")
        self._shutdown_event.set()

        # Cancel pending tasks
        for task in self._running_tasks:
            if not task.done():
                task.cancel()


async def run_parallel_batch(
    num_games: int,
    num_players: int = 5,
    model: str = "deepseek/deepseek-v3.2-exp",
    concurrency: int = 3,
    batch_id: Optional[str] = None,
    batch_tag: Optional[str] = None,
    resume: bool = True,
    api_key: Optional[str] = None
) -> BatchProgress:
    """
    Convenience function to run a parallel batch.

    Args:
        num_games: Number of games to run
        num_players: Players per game (5-10)
        model: OpenRouter model ID
        concurrency: Number of concurrent games
        batch_id: Optional batch identifier
        batch_tag: Optional batch tag
        resume: Whether to resume from previous progress
        api_key: OpenRouter API key (uses env var if not provided)

    Returns:
        BatchProgress with results
    """
    from dotenv import load_dotenv
    load_dotenv()

    api_key = api_key or os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set")

    runner = ParallelBatchRunner(
        api_key=api_key,
        num_players=num_players,
        model=model,
        concurrency=concurrency,
        enable_db_logging=True
    )

    return await runner.run_batch(
        num_games=num_games,
        batch_id=batch_id,
        batch_tag=batch_tag,
        resume=resume
    )


def main():
    """CLI entry point for parallel batch runner."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Parallel batch runner for large-scale evaluations"
    )
    parser.add_argument('--games', '-g', type=int, default=100, help='Number of games')
    parser.add_argument('--players', '-p', type=int, default=5, help='Players per game')
    parser.add_argument('--model', '-m', default='deepseek/deepseek-v3.2-exp', help='Model ID')
    parser.add_argument('--concurrency', '-c', type=int, default=3, help='Concurrent games')
    parser.add_argument('--batch-id', type=str, help='Batch identifier')
    parser.add_argument('--batch-tag', type=str, help='Batch tag')
    parser.add_argument('--no-resume', action='store_true', help='Do not resume from previous progress')

    args = parser.parse_args()

    progress = asyncio.run(run_parallel_batch(
        num_games=args.games,
        num_players=args.players,
        model=args.model,
        concurrency=args.concurrency,
        batch_id=args.batch_id,
        batch_tag=args.batch_tag,
        resume=not args.no_resume
    ))

    print(f"\n{'='*60}")
    print("BATCH COMPLETE")
    print(f"{'='*60}")
    print(f"Total games: {progress.total_games}")
    print(f"Completed: {progress.completed}")
    print(f"Failed: {progress.failed}")
    print(f"Liberal wins: {progress.liberal_wins}")
    print(f"Fascist wins: {progress.fascist_wins}")
    print(f"Total cost: ${progress.total_cost:.4f}")
    print(f"Average duration: {progress.avg_duration/60:.1f} minutes")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

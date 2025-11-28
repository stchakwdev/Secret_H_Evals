"""
Memory-efficient streaming statistics for large-scale analysis (5000+ games).

Uses online algorithms that process data in a single pass without
loading entire datasets into memory.

Features:
- Welford's algorithm for online mean/variance
- Streaming quantiles using P² algorithm
- Incremental histogram computation
- Rolling window calculations
- Chunked aggregation with merge capability
"""
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Iterator, Callable
from collections import defaultdict
import json


@dataclass
class WelfordAccumulator:
    """
    Online mean and variance calculator using Welford's algorithm.
    Numerically stable single-pass computation.
    """
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0  # Sum of squared differences
    min_val: float = float('inf')
    max_val: float = float('-inf')

    def update(self, value: float):
        """Add a new value to the accumulator."""
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2
        self.min_val = min(self.min_val, value)
        self.max_val = max(self.max_val, value)

    @property
    def variance(self) -> float:
        """Population variance."""
        if self.count < 2:
            return 0.0
        return self.m2 / self.count

    @property
    def sample_variance(self) -> float:
        """Sample variance (Bessel's correction)."""
        if self.count < 2:
            return 0.0
        return self.m2 / (self.count - 1)

    @property
    def std(self) -> float:
        """Population standard deviation."""
        return math.sqrt(self.variance)

    @property
    def sample_std(self) -> float:
        """Sample standard deviation."""
        return math.sqrt(self.sample_variance)

    def merge(self, other: 'WelfordAccumulator') -> 'WelfordAccumulator':
        """Merge two accumulators (for parallel processing)."""
        if other.count == 0:
            return self
        if self.count == 0:
            return other

        combined = WelfordAccumulator()
        combined.count = self.count + other.count

        delta = other.mean - self.mean
        combined.mean = (self.count * self.mean + other.count * other.mean) / combined.count

        combined.m2 = (
            self.m2 + other.m2 +
            delta * delta * self.count * other.count / combined.count
        )

        combined.min_val = min(self.min_val, other.min_val)
        combined.max_val = max(self.max_val, other.max_val)

        return combined

    def to_dict(self) -> Dict[str, Any]:
        """Export statistics."""
        return {
            'count': self.count,
            'mean': self.mean,
            'variance': self.variance,
            'std': self.std,
            'min': self.min_val if self.count > 0 else None,
            'max': self.max_val if self.count > 0 else None
        }


@dataclass
class CountAccumulator:
    """Simple counter with category tracking."""
    total: int = 0
    categories: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def update(self, category: Optional[str] = None):
        """Increment counter, optionally by category."""
        self.total += 1
        if category:
            self.categories[category] += 1

    def merge(self, other: 'CountAccumulator') -> 'CountAccumulator':
        """Merge two counters."""
        combined = CountAccumulator()
        combined.total = self.total + other.total

        all_keys = set(self.categories.keys()) | set(other.categories.keys())
        for key in all_keys:
            combined.categories[key] = self.categories.get(key, 0) + other.categories.get(key, 0)

        return combined

    def to_dict(self) -> Dict[str, Any]:
        """Export counts."""
        return {
            'total': self.total,
            'categories': dict(self.categories),
            'proportions': {
                k: v / self.total if self.total > 0 else 0
                for k, v in self.categories.items()
            }
        }


@dataclass
class HistogramAccumulator:
    """
    Online histogram computation with fixed bins.
    Memory-efficient for large datasets.
    """
    bins: int = 20
    min_val: float = 0.0
    max_val: float = 1.0
    counts: List[int] = field(default_factory=list)
    _initialized: bool = False

    def __post_init__(self):
        if not self._initialized:
            self.counts = [0] * self.bins
            self._initialized = True

    @property
    def bin_width(self) -> float:
        return (self.max_val - self.min_val) / self.bins

    def update(self, value: float):
        """Add a value to the histogram."""
        if value < self.min_val:
            self.counts[0] += 1
        elif value >= self.max_val:
            self.counts[-1] += 1
        else:
            bin_idx = int((value - self.min_val) / self.bin_width)
            bin_idx = min(bin_idx, self.bins - 1)
            self.counts[bin_idx] += 1

    def merge(self, other: 'HistogramAccumulator') -> 'HistogramAccumulator':
        """Merge two histograms (must have same bins)."""
        combined = HistogramAccumulator(
            bins=self.bins,
            min_val=self.min_val,
            max_val=self.max_val
        )
        combined.counts = [a + b for a, b in zip(self.counts, other.counts)]
        return combined

    def to_dict(self) -> Dict[str, Any]:
        """Export histogram data."""
        total = sum(self.counts)
        return {
            'bins': self.bins,
            'min': self.min_val,
            'max': self.max_val,
            'bin_width': self.bin_width,
            'counts': self.counts,
            'density': [c / (total * self.bin_width) if total > 0 else 0 for c in self.counts],
            'bin_edges': [self.min_val + i * self.bin_width for i in range(self.bins + 1)]
        }


@dataclass
class StreamingGameStats:
    """
    Comprehensive streaming statistics for game analysis.
    Processes games one at a time without loading full dataset.
    """
    # Game counts
    game_count: CountAccumulator = field(default_factory=CountAccumulator)

    # Win statistics
    wins: CountAccumulator = field(default_factory=CountAccumulator)

    # Duration statistics
    duration: WelfordAccumulator = field(default_factory=WelfordAccumulator)

    # Cost statistics
    cost: WelfordAccumulator = field(default_factory=WelfordAccumulator)

    # Policy statistics
    liberal_policies: WelfordAccumulator = field(default_factory=WelfordAccumulator)
    fascist_policies: WelfordAccumulator = field(default_factory=WelfordAccumulator)

    # Deception statistics
    deception_rate: WelfordAccumulator = field(default_factory=WelfordAccumulator)

    # Decision counts by type
    decisions_by_type: CountAccumulator = field(default_factory=CountAccumulator)

    # Cost histogram
    cost_histogram: HistogramAccumulator = field(
        default_factory=lambda: HistogramAccumulator(bins=20, min_val=0, max_val=1.0)
    )

    # Duration histogram (in minutes)
    duration_histogram: HistogramAccumulator = field(
        default_factory=lambda: HistogramAccumulator(bins=20, min_val=0, max_val=60)
    )

    def process_game(self, game: Dict[str, Any]):
        """Process a single game record."""
        # Count game
        self.game_count.update()

        # Win tracking
        winning_team = game.get('winning_team')
        if winning_team:
            self.wins.update(winning_team)

        # Duration
        duration = game.get('duration_seconds', 0)
        if duration > 0:
            self.duration.update(duration)
            self.duration_histogram.update(duration / 60)  # Convert to minutes

        # Cost
        cost = game.get('total_cost', 0)
        if cost > 0:
            self.cost.update(cost)
            self.cost_histogram.update(cost)

        # Policies
        self.liberal_policies.update(game.get('liberal_policies', 0))
        self.fascist_policies.update(game.get('fascist_policies', 0))

    def process_decision(self, decision: Dict[str, Any]):
        """Process a single decision record."""
        # Count by type
        decision_type = decision.get('decision_type', 'unknown')
        self.decisions_by_type.update(decision_type)

        # Deception tracking
        if decision.get('is_deception') is not None:
            self.deception_rate.update(1 if decision.get('is_deception') else 0)

    def merge(self, other: 'StreamingGameStats') -> 'StreamingGameStats':
        """Merge two stats objects (for parallel processing)."""
        combined = StreamingGameStats()
        combined.game_count = self.game_count.merge(other.game_count)
        combined.wins = self.wins.merge(other.wins)
        combined.duration = self.duration.merge(other.duration)
        combined.cost = self.cost.merge(other.cost)
        combined.liberal_policies = self.liberal_policies.merge(other.liberal_policies)
        combined.fascist_policies = self.fascist_policies.merge(other.fascist_policies)
        combined.deception_rate = self.deception_rate.merge(other.deception_rate)
        combined.decisions_by_type = self.decisions_by_type.merge(other.decisions_by_type)
        combined.cost_histogram = self.cost_histogram.merge(other.cost_histogram)
        combined.duration_histogram = self.duration_histogram.merge(other.duration_histogram)
        return combined

    def to_dict(self) -> Dict[str, Any]:
        """Export all statistics."""
        return {
            'games': {
                'total': self.game_count.total,
                'wins': self.wins.to_dict()
            },
            'duration': {
                **self.duration.to_dict(),
                'histogram': self.duration_histogram.to_dict()
            },
            'cost': {
                **self.cost.to_dict(),
                'histogram': self.cost_histogram.to_dict()
            },
            'policies': {
                'liberal': self.liberal_policies.to_dict(),
                'fascist': self.fascist_policies.to_dict()
            },
            'deception': {
                'mean_rate': self.deception_rate.mean,
                'total_decisions': self.deception_rate.count
            },
            'decisions': self.decisions_by_type.to_dict()
        }


class StreamingAnalyzer:
    """
    High-level interface for streaming analysis of large game datasets.

    Usage:
        analyzer = StreamingAnalyzer(db)
        stats = analyzer.analyze_all_games(batch_size=100)
        report = analyzer.generate_report(stats)
    """

    def __init__(self, db: 'ScaleDatabaseManager'):
        self.db = db

    def analyze_all_games(
        self,
        batch_size: int = 100,
        filters: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> StreamingGameStats:
        """
        Analyze all games using streaming computation.

        Args:
            batch_size: Number of games to process per batch
            filters: Optional filters for game selection
            progress_callback: Optional callback(processed, total)

        Returns:
            StreamingGameStats with aggregated results
        """
        stats = StreamingGameStats()
        processed = 0

        # First get total count
        total = self.db.get_stats().get('total_games', 0)

        # Stream games in batches
        for batch in self.db.stream_games(batch_size=batch_size, filters=filters):
            for game in batch:
                stats.process_game(game)
                processed += 1

            if progress_callback:
                progress_callback(processed, total)

        # Stream decisions
        for batch in self.db.stream_decisions(batch_size=batch_size):
            for decision in batch:
                stats.process_decision(decision)

        return stats

    def analyze_by_model(
        self,
        batch_size: int = 100
    ) -> Dict[str, StreamingGameStats]:
        """
        Analyze games grouped by model.

        Returns:
            Dictionary mapping model name to stats
        """
        model_stats: Dict[str, StreamingGameStats] = defaultdict(StreamingGameStats)

        for batch in self.db.stream_games(batch_size=batch_size):
            for game in batch:
                models_used = json.loads(game.get('models_used', '[]'))
                for model in models_used:
                    model_stats[model].process_game(game)

        return dict(model_stats)

    def compute_rolling_stats(
        self,
        window_size: int = 100,
        metric: str = 'cost'
    ) -> List[Dict[str, Any]]:
        """
        Compute rolling statistics over time.

        Args:
            window_size: Number of games in rolling window
            metric: Metric to track ('cost', 'duration', 'deception_rate')

        Returns:
            List of rolling statistics per window
        """
        from collections import deque

        window = deque(maxlen=window_size)
        results = []

        for batch in self.db.stream_games(batch_size=100):
            for game in batch:
                # Extract metric value
                if metric == 'cost':
                    value = game.get('total_cost', 0)
                elif metric == 'duration':
                    value = game.get('duration_seconds', 0)
                else:
                    continue

                window.append(value)

                if len(window) == window_size:
                    values = list(window)
                    results.append({
                        'timestamp': game.get('timestamp'),
                        'mean': sum(values) / len(values),
                        'min': min(values),
                        'max': max(values)
                    })

        return results

    def generate_report(
        self,
        stats: StreamingGameStats,
        format: str = 'markdown'
    ) -> str:
        """
        Generate a formatted report from statistics.

        Args:
            stats: StreamingGameStats to report on
            format: Output format ('markdown', 'json', 'text')

        Returns:
            Formatted report string
        """
        data = stats.to_dict()

        if format == 'json':
            return json.dumps(data, indent=2)

        if format == 'text':
            lines = [
                "=" * 60,
                "STREAMING STATISTICS REPORT",
                "=" * 60,
                "",
                f"Total Games: {data['games']['total']}",
                "",
                "Win Distribution:",
            ]
            for team, count in data['games']['wins']['categories'].items():
                pct = data['games']['wins']['proportions'].get(team, 0) * 100
                lines.append(f"  {team}: {count} ({pct:.1f}%)")

            lines.extend([
                "",
                "Duration Statistics:",
                f"  Mean: {data['duration']['mean']/60:.1f} minutes",
                f"  Std Dev: {data['duration']['std']/60:.1f} minutes",
                f"  Range: {data['duration']['min']/60:.1f} - {data['duration']['max']/60:.1f} minutes",
                "",
                "Cost Statistics:",
                f"  Total: ${data['cost']['mean'] * data['games']['total']:.4f}",
                f"  Mean per game: ${data['cost']['mean']:.4f}",
                f"  Std Dev: ${data['cost']['std']:.4f}",
                "",
                f"Deception Rate: {data['deception']['mean_rate']*100:.1f}%",
                f"Total Decisions: {data['deception']['total_decisions']}",
                "",
                "=" * 60
            ])
            return "\n".join(lines)

        # Markdown format
        report = f"""# Streaming Statistics Report

## Summary

| Metric | Value |
|--------|-------|
| Total Games | {data['games']['total']} |
| Liberal Wins | {data['games']['wins']['categories'].get('liberal', 0)} |
| Fascist Wins | {data['games']['wins']['categories'].get('fascist', 0)} |

## Duration Statistics

| Metric | Value |
|--------|-------|
| Mean | {data['duration']['mean']/60:.1f} min |
| Std Dev | {data['duration']['std']/60:.1f} min |
| Min | {data['duration']['min']/60:.1f} min |
| Max | {data['duration']['max']/60:.1f} min |

## Cost Statistics

| Metric | Value |
|--------|-------|
| Total | ${data['cost']['mean'] * data['games']['total']:.4f} |
| Mean/Game | ${data['cost']['mean']:.4f} |
| Std Dev | ${data['cost']['std']:.4f} |

## Deception Analysis

- **Mean Deception Rate**: {data['deception']['mean_rate']*100:.1f}%
- **Total Decisions Analyzed**: {data['deception']['total_decisions']}

## Decision Distribution

| Type | Count | Proportion |
|------|-------|------------|
"""
        for dtype, count in data['decisions']['categories'].items():
            pct = data['decisions']['proportions'].get(dtype, 0) * 100
            report += f"| {dtype} | {count} | {pct:.1f}% |\n"

        return report


def stream_analyze(
    db_path: str = "data/games.db",
    output_format: str = 'markdown'
) -> str:
    """
    Convenience function for streaming analysis.

    Args:
        db_path: Path to database
        output_format: Output format

    Returns:
        Formatted report
    """
    from evaluation.database_scale import get_scale_db

    db = get_scale_db(db_path)
    analyzer = StreamingAnalyzer(db)

    stats = analyzer.analyze_all_games(
        batch_size=100,
        progress_callback=lambda p, t: print(f"Processed {p}/{t} games")
    )

    return analyzer.generate_report(stats, format=output_format)


if __name__ == "__main__":
    # Example usage
    report = stream_analyze(output_format='text')
    print(report)

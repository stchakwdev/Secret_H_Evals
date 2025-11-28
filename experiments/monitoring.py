"""
Monitoring and metrics collection for large-scale evaluations.

Features:
- Prometheus-compatible metrics export
- Real-time performance tracking
- Alert rules for anomalies
- Resource monitoring
- Cost tracking and alerts
"""
import time
import threading
import json
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from collections import defaultdict, deque
from pathlib import Path
import os

logger = logging.getLogger(__name__)


@dataclass
class MetricValue:
    """Single metric observation with timestamp."""
    value: float
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)


class Counter:
    """Prometheus-style counter metric."""

    def __init__(self, name: str, description: str, labels: List[str] = None):
        self.name = name
        self.description = description
        self.label_names = labels or []
        self._values: Dict[tuple, float] = defaultdict(float)
        self._lock = threading.Lock()

    def inc(self, amount: float = 1.0, **labels):
        """Increment the counter."""
        key = tuple(labels.get(l, "") for l in self.label_names)
        with self._lock:
            self._values[key] += amount

    def get(self, **labels) -> float:
        """Get current counter value."""
        key = tuple(labels.get(l, "") for l in self.label_names)
        return self._values.get(key, 0.0)

    def to_prometheus(self) -> str:
        """Export in Prometheus format."""
        lines = [
            f"# HELP {self.name} {self.description}",
            f"# TYPE {self.name} counter"
        ]
        for key, value in self._values.items():
            if key and any(key):
                label_str = ",".join(
                    f'{name}="{val}"'
                    for name, val in zip(self.label_names, key) if val
                )
                lines.append(f"{self.name}{{{label_str}}} {value}")
            else:
                lines.append(f"{self.name} {value}")
        return "\n".join(lines)


class Gauge:
    """Prometheus-style gauge metric."""

    def __init__(self, name: str, description: str, labels: List[str] = None):
        self.name = name
        self.description = description
        self.label_names = labels or []
        self._values: Dict[tuple, float] = defaultdict(float)
        self._lock = threading.Lock()

    def set(self, value: float, **labels):
        """Set the gauge value."""
        key = tuple(labels.get(l, "") for l in self.label_names)
        with self._lock:
            self._values[key] = value

    def inc(self, amount: float = 1.0, **labels):
        """Increment the gauge."""
        key = tuple(labels.get(l, "") for l in self.label_names)
        with self._lock:
            self._values[key] += amount

    def dec(self, amount: float = 1.0, **labels):
        """Decrement the gauge."""
        key = tuple(labels.get(l, "") for l in self.label_names)
        with self._lock:
            self._values[key] -= amount

    def get(self, **labels) -> float:
        """Get current gauge value."""
        key = tuple(labels.get(l, "") for l in self.label_names)
        return self._values.get(key, 0.0)

    def to_prometheus(self) -> str:
        """Export in Prometheus format."""
        lines = [
            f"# HELP {self.name} {self.description}",
            f"# TYPE {self.name} gauge"
        ]
        for key, value in self._values.items():
            if key and any(key):
                label_str = ",".join(
                    f'{name}="{val}"'
                    for name, val in zip(self.label_names, key) if val
                )
                lines.append(f"{self.name}{{{label_str}}} {value}")
            else:
                lines.append(f"{self.name} {value}")
        return "\n".join(lines)


class Histogram:
    """Prometheus-style histogram metric."""

    DEFAULT_BUCKETS = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, float('inf')]

    def __init__(
        self,
        name: str,
        description: str,
        labels: List[str] = None,
        buckets: List[float] = None
    ):
        self.name = name
        self.description = description
        self.label_names = labels or []
        self.buckets = buckets or self.DEFAULT_BUCKETS
        self._counts: Dict[tuple, List[int]] = defaultdict(lambda: [0] * len(self.buckets))
        self._sums: Dict[tuple, float] = defaultdict(float)
        self._totals: Dict[tuple, int] = defaultdict(int)
        self._lock = threading.Lock()

    def observe(self, value: float, **labels):
        """Record an observation."""
        key = tuple(labels.get(l, "") for l in self.label_names)
        with self._lock:
            self._sums[key] += value
            self._totals[key] += 1
            for i, bucket in enumerate(self.buckets):
                if value <= bucket:
                    self._counts[key][i] += 1

    def get_percentile(self, p: float, **labels) -> float:
        """Estimate percentile from histogram."""
        key = tuple(labels.get(l, "") for l in self.label_names)
        counts = self._counts.get(key, [0] * len(self.buckets))
        total = self._totals.get(key, 0)

        if total == 0:
            return 0.0

        target = p * total
        cumulative = 0
        prev_bucket = 0

        for i, (bucket, count) in enumerate(zip(self.buckets, counts)):
            cumulative = count
            if cumulative >= target:
                # Linear interpolation within bucket
                if i == 0:
                    return bucket * (target / count) if count > 0 else bucket
                prev_count = counts[i - 1] if i > 0 else 0
                bucket_count = count - prev_count
                if bucket_count > 0:
                    ratio = (target - prev_count) / bucket_count
                    return prev_bucket + ratio * (bucket - prev_bucket)
                return bucket
            prev_bucket = bucket

        return self.buckets[-2] if len(self.buckets) > 1 else 0.0

    def to_prometheus(self) -> str:
        """Export in Prometheus format."""
        lines = [
            f"# HELP {self.name} {self.description}",
            f"# TYPE {self.name} histogram"
        ]

        for key in self._counts.keys():
            label_base = ",".join(
                f'{name}="{val}"'
                for name, val in zip(self.label_names, key) if val
            )

            # Bucket counts
            cumulative = 0
            for bucket, count in zip(self.buckets, self._counts[key]):
                cumulative = count
                le = "+Inf" if bucket == float('inf') else str(bucket)
                if label_base:
                    lines.append(f'{self.name}_bucket{{{label_base},le="{le}"}} {cumulative}')
                else:
                    lines.append(f'{self.name}_bucket{{le="{le}"}} {cumulative}')

            # Sum and count
            if label_base:
                lines.append(f"{self.name}_sum{{{label_base}}} {self._sums[key]}")
                lines.append(f"{self.name}_count{{{label_base}}} {self._totals[key]}")
            else:
                lines.append(f"{self.name}_sum {self._sums[key]}")
                lines.append(f"{self.name}_count {self._totals[key]}")

        return "\n".join(lines)


@dataclass
class AlertRule:
    """Definition of an alert rule."""
    name: str
    condition: Callable[['MetricsCollector'], bool]
    message: str
    severity: str = "warning"  # "info", "warning", "critical"
    cooldown_seconds: int = 300  # Minimum time between alerts

    _last_triggered: float = 0.0

    def check(self, metrics: 'MetricsCollector') -> Optional[str]:
        """Check if alert should fire."""
        now = time.time()
        if now - self._last_triggered < self.cooldown_seconds:
            return None

        if self.condition(metrics):
            self._last_triggered = now
            return f"[{self.severity.upper()}] {self.name}: {self.message}"

        return None


class MetricsCollector:
    """
    Central metrics collector for monitoring evaluations.

    Usage:
        metrics = MetricsCollector()
        metrics.game_started()
        metrics.api_request_completed(model="deepseek", latency=0.5, cost=0.001)
        metrics.game_completed(winner="liberal", duration=120)

        # Export for monitoring
        print(metrics.to_prometheus())
    """

    def __init__(self, export_path: Optional[Path] = None):
        self.export_path = export_path or Path("data/metrics")
        self.export_path.mkdir(parents=True, exist_ok=True)

        # Initialize metrics
        self._init_metrics()

        # Alert rules
        self.alert_rules: List[AlertRule] = []
        self.alert_callbacks: List[Callable[[str], None]] = []

        # Time series storage for rolling calculations
        self._time_series: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

        # Periodic export thread
        self._export_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def _init_metrics(self):
        """Initialize all metrics."""
        # Game metrics
        self.games_total = Counter(
            "secrethitler_games_total",
            "Total number of games played",
            labels=["status", "winner"]
        )

        self.games_in_progress = Gauge(
            "secrethitler_games_in_progress",
            "Number of games currently running"
        )

        self.game_duration = Histogram(
            "secrethitler_game_duration_seconds",
            "Game duration in seconds",
            buckets=[60, 120, 180, 300, 600, 900, 1200, 1800, 3600]
        )

        # API metrics
        self.api_requests_total = Counter(
            "secrethitler_api_requests_total",
            "Total API requests",
            labels=["model", "decision_type", "status"]
        )

        self.api_latency = Histogram(
            "secrethitler_api_latency_seconds",
            "API request latency",
            labels=["model"],
            buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0]
        )

        self.api_cost = Counter(
            "secrethitler_api_cost_dollars",
            "Total API cost in dollars",
            labels=["model"]
        )

        # Resource metrics
        self.connection_pool_size = Gauge(
            "secrethitler_connection_pool_size",
            "Database connection pool size",
            labels=["state"]
        )

        self.memory_usage_bytes = Gauge(
            "secrethitler_memory_usage_bytes",
            "Process memory usage"
        )

        # Batch metrics
        self.batch_progress = Gauge(
            "secrethitler_batch_progress",
            "Batch completion progress (0-1)",
            labels=["batch_id"]
        )

        self.batch_errors = Counter(
            "secrethitler_batch_errors_total",
            "Total batch errors",
            labels=["batch_id", "error_type"]
        )

    # ==================== Recording Methods ====================

    def game_started(self):
        """Record game start."""
        self.games_in_progress.inc()
        self._record_time_series('games_started', 1)

    def game_completed(
        self,
        winner: str,
        duration_seconds: float,
        status: str = "completed"
    ):
        """Record game completion."""
        self.games_in_progress.dec()
        self.games_total.inc(status=status, winner=winner)
        self.game_duration.observe(duration_seconds)
        self._record_time_series('games_completed', 1)
        self._record_time_series('game_duration', duration_seconds)

        # Check alerts
        self._check_alerts()

    def api_request_completed(
        self,
        model: str,
        decision_type: str,
        latency_seconds: float,
        cost: float,
        status: str = "success"
    ):
        """Record API request."""
        self.api_requests_total.inc(model=model, decision_type=decision_type, status=status)
        self.api_latency.observe(latency_seconds, model=model)
        self.api_cost.inc(cost, model=model)
        self._record_time_series('api_latency', latency_seconds)
        self._record_time_series('api_cost', cost)

    def update_connection_pool(self, active: int, idle: int):
        """Update connection pool metrics."""
        self.connection_pool_size.set(active, state="active")
        self.connection_pool_size.set(idle, state="idle")

    def update_memory_usage(self):
        """Update memory usage metric."""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            self.memory_usage_bytes.set(process.memory_info().rss)
        except ImportError:
            pass

    def update_batch_progress(self, batch_id: str, progress: float):
        """Update batch progress."""
        self.batch_progress.set(progress, batch_id=batch_id)

    def record_batch_error(self, batch_id: str, error_type: str):
        """Record batch error."""
        self.batch_errors.inc(batch_id=batch_id, error_type=error_type)

    def _record_time_series(self, name: str, value: float):
        """Record value in time series for rolling calculations."""
        self._time_series[name].append((time.time(), value))

    # ==================== Alert System ====================

    def add_alert_rule(self, rule: AlertRule):
        """Add an alert rule."""
        self.alert_rules.append(rule)

    def add_alert_callback(self, callback: Callable[[str], None]):
        """Add callback for alert notifications."""
        self.alert_callbacks.append(callback)

    def _check_alerts(self):
        """Check all alert rules."""
        for rule in self.alert_rules:
            message = rule.check(self)
            if message:
                logger.warning(message)
                for callback in self.alert_callbacks:
                    try:
                        callback(message)
                    except Exception as e:
                        logger.error(f"Alert callback error: {e}")

    # ==================== Analysis Methods ====================

    def get_rate(self, metric_name: str, window_seconds: int = 60) -> float:
        """Calculate rate of change over time window."""
        series = self._time_series.get(metric_name, deque())
        if not series:
            return 0.0

        now = time.time()
        cutoff = now - window_seconds

        window_values = [(t, v) for t, v in series if t >= cutoff]
        if len(window_values) < 2:
            return 0.0

        total = sum(v for _, v in window_values)
        elapsed = window_values[-1][0] - window_values[0][0]

        return total / elapsed if elapsed > 0 else 0.0

    def get_rolling_average(self, metric_name: str, window_seconds: int = 300) -> float:
        """Calculate rolling average over time window."""
        series = self._time_series.get(metric_name, deque())
        if not series:
            return 0.0

        now = time.time()
        cutoff = now - window_seconds

        window_values = [v for t, v in series if t >= cutoff]
        return sum(window_values) / len(window_values) if window_values else 0.0

    # ==================== Export Methods ====================

    def to_prometheus(self) -> str:
        """Export all metrics in Prometheus format."""
        sections = [
            self.games_total.to_prometheus(),
            self.games_in_progress.to_prometheus(),
            self.game_duration.to_prometheus(),
            self.api_requests_total.to_prometheus(),
            self.api_latency.to_prometheus(),
            self.api_cost.to_prometheus(),
            self.connection_pool_size.to_prometheus(),
            self.memory_usage_bytes.to_prometheus(),
            self.batch_progress.to_prometheus(),
            self.batch_errors.to_prometheus()
        ]
        return "\n\n".join(sections)

    def to_json(self) -> Dict[str, Any]:
        """Export metrics as JSON."""
        return {
            'timestamp': datetime.now().isoformat(),
            'games': {
                'total': self.games_total.get(status="completed", winner=""),
                'in_progress': self.games_in_progress.get(),
                'completed_rate': self.get_rate('games_completed', 300),
                'avg_duration': self.get_rolling_average('game_duration', 300)
            },
            'api': {
                'total_requests': self.api_requests_total.get(model="", decision_type="", status="success"),
                'request_rate': self.get_rate('api_latency', 60),
                'avg_latency': self.get_rolling_average('api_latency', 300),
                'p95_latency': self.api_latency.get_percentile(0.95),
                'total_cost': self.api_cost.get(model="")
            },
            'resources': {
                'active_connections': self.connection_pool_size.get(state="active"),
                'idle_connections': self.connection_pool_size.get(state="idle"),
                'memory_bytes': self.memory_usage_bytes.get()
            }
        }

    def export_to_file(self, format: str = 'prometheus'):
        """Export metrics to file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        if format == 'prometheus':
            path = self.export_path / f"metrics_{timestamp}.prom"
            with open(path, 'w') as f:
                f.write(self.to_prometheus())
        else:
            path = self.export_path / f"metrics_{timestamp}.json"
            with open(path, 'w') as f:
                json.dump(self.to_json(), f, indent=2)

        logger.debug(f"Exported metrics to {path}")

    def start_periodic_export(self, interval_seconds: int = 60, format: str = 'json'):
        """Start periodic metrics export in background."""
        def export_loop():
            while not self._stop_event.is_set():
                self.update_memory_usage()
                self.export_to_file(format=format)
                self._stop_event.wait(interval_seconds)

        self._export_thread = threading.Thread(target=export_loop, daemon=True)
        self._export_thread.start()
        logger.info(f"Started periodic metrics export every {interval_seconds}s")

    def stop_periodic_export(self):
        """Stop periodic export."""
        if self._export_thread:
            self._stop_event.set()
            self._export_thread.join(timeout=5)
            logger.info("Stopped periodic metrics export")


# Default alert rules
def create_default_alerts(metrics: MetricsCollector):
    """Create default alert rules."""

    # High error rate
    metrics.add_alert_rule(AlertRule(
        name="HighErrorRate",
        condition=lambda m: m.get_rate('batch_errors', 300) > 0.1,
        message="Error rate exceeds 10% in last 5 minutes",
        severity="warning"
    ))

    # High API latency
    metrics.add_alert_rule(AlertRule(
        name="HighAPILatency",
        condition=lambda m: m.api_latency.get_percentile(0.95) > 30,
        message="95th percentile API latency exceeds 30 seconds",
        severity="warning"
    ))

    # Cost alert
    metrics.add_alert_rule(AlertRule(
        name="HighCostRate",
        condition=lambda m: m.get_rate('api_cost', 3600) > 1.0,
        message="Cost rate exceeds $1/hour",
        severity="info"
    ))


# Global metrics instance
_metrics: Optional[MetricsCollector] = None


def get_metrics() -> MetricsCollector:
    """Get or create global metrics collector."""
    global _metrics
    if _metrics is None:
        _metrics = MetricsCollector()
        create_default_alerts(_metrics)
    return _metrics

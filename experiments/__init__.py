"""
Experiments module for Secret Hitler LLM evaluation.
Provides batch running and analysis capabilities for research.
"""

from .batch_runner import BatchRunner, ExperimentConfig, create_experiment_config
from .analytics import ExperimentAnalyzer, compare_experiments

__all__ = [
    'BatchRunner',
    'ExperimentConfig', 
    'create_experiment_config',
    'ExperimentAnalyzer',
    'compare_experiments'
]
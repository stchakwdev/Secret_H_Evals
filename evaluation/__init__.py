"""
Inspect AI integration for Secret Hitler LLM evaluation.

This module provides tools to convert Secret Hitler game logs to Inspect AI format
for standardized logging, analysis, and visualization.
"""

__version__ = "1.0.0"

from .inspect_adapter import SecretHitlerInspectAdapter
from .database_schema import DatabaseManager

__all__ = [
    "SecretHitlerInspectAdapter",
    "DatabaseManager",
]

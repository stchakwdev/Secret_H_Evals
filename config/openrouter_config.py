"""
OpenRouter configuration for Secret Hitler LLM evaluation.
"""
import os
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    name: str
    tier: str  # 'critical', 'strategic', 'routine'
    cost_per_1k_tokens: float
    max_tokens: int
    temperature: float = 0.7

# OpenRouter model configurations with current pricing (November 2025)
OPENROUTER_MODELS = {
    # ==========================================================================
    # TIER 1: FREE MODELS (Phase 4 Multi-Model Comparison)
    # ==========================================================================
    'grok-4.1-fast': ModelConfig(
        name='x-ai/grok-4.1-fast:free',
        tier='free',
        cost_per_1k_tokens=0.0,  # FREE
        max_tokens=8192,
        temperature=0.7
    ),
    'glm-4.5-air': ModelConfig(
        name='z-ai/glm-4.5-air:free',
        tier='free',
        cost_per_1k_tokens=0.0,  # FREE
        max_tokens=8192,
        temperature=0.7
    ),
    'bert-nebulon-alpha': ModelConfig(
        name='openrouter/bert-nebulon-alpha',
        tier='free',
        cost_per_1k_tokens=0.0,  # FREE
        max_tokens=8192,
        temperature=0.7
    ),
    'llama-4-maverick': ModelConfig(
        name='meta-llama/llama-4-maverick:free',
        tier='free',
        cost_per_1k_tokens=0.0,  # FREE
        max_tokens=8192,
        temperature=0.7
    ),
    'llama-4-scout': ModelConfig(
        name='meta-llama/llama-4-scout:free',
        tier='free',
        cost_per_1k_tokens=0.0,  # FREE
        max_tokens=8192,
        temperature=0.7
    ),
    'deepseek-r1-free': ModelConfig(
        name='deepseek/deepseek-r1:free',
        tier='free',
        cost_per_1k_tokens=0.0,  # FREE
        max_tokens=8192,
        temperature=0.7
    ),
    'gemini-2.0-flash-exp': ModelConfig(
        name='google/gemini-2.0-flash-exp:free',
        tier='free',
        cost_per_1k_tokens=0.0,  # FREE
        max_tokens=8192,
        temperature=0.7
    ),
    'mistral-small-3.1': ModelConfig(
        name='mistralai/mistral-small-3.1-24b-instruct:free',
        tier='free',
        cost_per_1k_tokens=0.0,  # FREE
        max_tokens=8192,
        temperature=0.7
    ),
    'optimus-alpha': ModelConfig(
        name='openrouter/optimus-alpha',
        tier='free',
        cost_per_1k_tokens=0.0,  # FREE
        max_tokens=8192,
        temperature=0.7
    ),

    # ==========================================================================
    # TIER 2: BUDGET MODELS (Low cost, high volume)
    # ==========================================================================
    'gpt-5-nano': ModelConfig(
        name='openai/gpt-5-nano',
        tier='budget',
        cost_per_1k_tokens=0.000225,  # avg ($0.05 input + $0.40 output) / 2 / 1000
        max_tokens=8192,
        temperature=0.7
    ),
    'deepseek-v3': ModelConfig(
        name='deepseek/deepseek-chat',
        tier='budget',
        cost_per_1k_tokens=0.0005,  # avg ($0.20 input + $0.80 output) / 2 / 1000
        max_tokens=8192,
        temperature=0.7
    ),
    'gemini-2.5-flash-lite': ModelConfig(
        name='google/gemini-2.5-flash-lite',
        tier='budget',
        cost_per_1k_tokens=0.00025,  # avg ($0.10 input + $0.40 output) / 2 / 1000
        max_tokens=8192,
        temperature=0.7
    ),

    # ==========================================================================
    # TIER 3: PREMIUM MODELS (Calibration baseline - reserved for future)
    # ==========================================================================
    'claude-4.5-sonnet': ModelConfig(
        name='anthropic/claude-sonnet-4.5',
        tier='premium',
        cost_per_1k_tokens=0.009,  # avg ($3.00 input + $15.00 output) / 2 / 1000
        max_tokens=8192,
        temperature=0.5
    ),
    'deepseek-r1-paid': ModelConfig(
        name='deepseek/deepseek-reasoner',
        tier='premium',
        cost_per_1k_tokens=0.00235,  # avg ($0.20 input + $4.50 output) / 2 / 1000
        max_tokens=8192,
        temperature=0.5
    ),

    # ==========================================================================
    # LEGACY MODELS (kept for backwards compatibility)
    # ==========================================================================
    'claude-3-opus': ModelConfig(
        name='anthropic/claude-3-opus',
        tier='critical',
        cost_per_1k_tokens=0.015,  # $15/1M input tokens
        max_tokens=4096,
        temperature=0.3
    ),
    'gpt-4-turbo': ModelConfig(
        name='openai/gpt-4-turbo',
        tier='critical',
        cost_per_1k_tokens=0.010,  # $10/1M input tokens
        max_tokens=4096,
        temperature=0.3
    ),
    'claude-3-sonnet': ModelConfig(
        name='anthropic/claude-3-sonnet',
        tier='strategic',
        cost_per_1k_tokens=0.003,  # $3/1M input tokens
        max_tokens=4096,
        temperature=0.5
    ),
    'gpt-3.5-turbo': ModelConfig(
        name='openai/gpt-3.5-turbo',
        tier='strategic',
        cost_per_1k_tokens=0.0015,  # $1.5/1M input tokens
        max_tokens=4096,
        temperature=0.5
    ),
    'deepseek-chat': ModelConfig(
        name='deepseek/deepseek-v3.2-exp',
        tier='routine',
        cost_per_1k_tokens=0.00014,  # $0.14/1M input tokens (very low cost)
        max_tokens=8192,
        temperature=0.7
    ),
    'mixtral-8x7b': ModelConfig(
        name='mistralai/mixtral-8x7b-instruct',
        tier='routine',
        cost_per_1k_tokens=0.0005,  # $0.5/1M input tokens
        max_tokens=4096,
        temperature=0.7
    ),
    'llama-3-70b': ModelConfig(
        name='meta-llama/llama-3-70b-instruct',
        tier='routine',
        cost_per_1k_tokens=0.0008,  # $0.8/1M input tokens
        max_tokens=4096,
        temperature=0.7
    )
}

# Phase 4 model IDs for easy access
PHASE4_FREE_MODELS = [
    'x-ai/grok-4.1-fast:free',
    'z-ai/glm-4.5-air:free',
    'openrouter/bert-nebulon-alpha',
    'meta-llama/llama-4-maverick:free',
    'meta-llama/llama-4-scout:free',
    'deepseek/deepseek-r1:free',
    'google/gemini-2.0-flash-exp:free',
    'mistralai/mistral-small-3.1-24b-instruct:free',
    'openrouter/optimus-alpha',
]

PHASE4_PAID_MODELS = [
    'openai/gpt-5-nano',
    'deepseek/deepseek-chat',
]

PHASE4_ALL_MODELS = PHASE4_FREE_MODELS + PHASE4_PAID_MODELS

# Decision complexity mapping
DECISION_TIERS = {
    'critical': [
        'nominate_chancellor',  # Key trust decisions
        'choose_policies_as_president',  # Policy selection with deception
        'choose_policies_as_chancellor',  # Policy selection with veto
        'investigate_player',  # Information gathering
        'execute_player',  # Elimination decisions
    ],
    'strategic': [
        'vote_on_government',  # Government approval
        'discuss_nomination',  # Discussion and persuasion
        'discuss_investigation',  # Information sharing
        'special_election',  # Presidential selection
    ],
    'routine': [
        'acknowledge_role',  # Simple confirmations
        'acknowledge_policies',  # Policy acknowledgments
        'acknowledge_investigation',  # Investigation results
    ]
}

# OpenRouter API configuration
OPENROUTER_CONFIG = {
    'base_url': 'https://openrouter.ai/api/v1',
    'timeout': 60,
    'max_retries': 3,
    'rate_limit_strategy': 'adaptive',
    'cost_tracking': True,
    'model_fallbacks': True,
    'headers': {
        'HTTP-Referer': 'https://secret-hitler-llm.example.com',
        'X-Title': 'Secret Hitler LLM Evaluation'
    }
}

# Cost management
COST_LIMITS = {
    'per_game': 5.00,  # Maximum cost per game
    'per_tournament': 100.00,  # Maximum cost per tournament 
    'daily_limit': 200.00,  # Daily spending limit
    'alert_thresholds': [0.5, 1.0, 2.0, 5.0]  # Alert at these costs
}

def get_model_for_decision(decision_type: str) -> str:
    """Get the appropriate model for a decision type."""
    for tier, decisions in DECISION_TIERS.items():
        if decision_type in decisions:
            # Get first available model for this tier
            tier_models = [m for m in OPENROUTER_MODELS.values() if m.tier == tier]
            if tier_models:
                return tier_models[0].name
    
    # Default to DeepSeek model if decision not found
    return 'deepseek/deepseek-v3.2-exp'

def get_model_config(model_name: str) -> Optional[ModelConfig]:
    """Get configuration for a specific model."""
    for config in OPENROUTER_MODELS.values():
        if config.name == model_name:
            return config
    return None

def estimate_cost(prompt_tokens: int, completion_tokens: int, model_name: str) -> float:
    """Estimate cost for a request."""
    config = get_model_config(model_name)
    if not config:
        return 0.0
    
    # OpenRouter typically charges for input + output tokens
    total_tokens = prompt_tokens + completion_tokens
    return (total_tokens / 1000) * config.cost_per_1k_tokens
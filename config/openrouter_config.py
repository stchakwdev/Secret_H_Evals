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

# OpenRouter model configurations with current pricing
OPENROUTER_MODELS = {
    # Critical decisions - High cost, high capability
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
    
    # Strategic decisions - Medium cost, good capability
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
    
    # Free/Low-cost models - Perfect for testing (prioritized first)
    'deepseek-chat': ModelConfig(
        name='deepseek/deepseek-v3.2-exp',
        tier='routine',
        cost_per_1k_tokens=0.00014,  # $0.14/1M input tokens (very low cost)
        max_tokens=8192,
        temperature=0.7
    ),
    
    # Routine decisions - Low cost, adequate capability
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
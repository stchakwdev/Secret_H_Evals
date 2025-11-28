"""
Model Comparison Configuration for Phase 4 Multi-Model Evaluation

Defines all models for the multi-model comparison study with current
OpenRouter pricing (November 2025).

Author: Samuel Chakwera (stchakdev)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


class ModelTier(Enum):
    """Model pricing tiers."""
    FREE = "free"
    BUDGET = "budget"
    PREMIUM = "premium"


class ModelProvider(Enum):
    """Model providers."""
    XAI = "xAI"
    ZAI = "Z.AI"
    OPENROUTER = "OpenRouter"
    META = "Meta"
    DEEPSEEK = "DeepSeek"
    GOOGLE = "Google"
    MISTRAL = "Mistral"
    OPENAI = "OpenAI"
    ANTHROPIC = "Anthropic"


class ModelArchitecture(Enum):
    """Model architecture types."""
    MOE = "mixture_of_experts"
    DENSE = "dense"
    UNKNOWN = "unknown"


@dataclass
class ModelConfig:
    """Configuration for a single model."""
    name: str
    openrouter_id: str
    provider: ModelProvider
    tier: ModelTier
    architecture: ModelArchitecture
    context_length: int
    input_cost_per_million: float  # USD per million tokens
    output_cost_per_million: float  # USD per million tokens
    parameters: Optional[str] = None  # e.g., "400B", "24B"
    notes: str = ""
    is_reasoning: bool = False
    is_multimodal: bool = False
    is_cloaked: bool = False  # Mystery/stealth model

    @property
    def is_free(self) -> bool:
        return self.tier == ModelTier.FREE

    def estimate_game_cost(self, input_tokens: int = 75000, output_tokens: int = 75000) -> float:
        """Estimate cost for a single game."""
        if self.is_free:
            return 0.0
        input_cost = (input_tokens / 1_000_000) * self.input_cost_per_million
        output_cost = (output_tokens / 1_000_000) * self.output_cost_per_million
        return input_cost + output_cost

    def estimate_batch_cost(self, num_games: int, input_tokens: int = 75000, output_tokens: int = 75000) -> float:
        """Estimate cost for a batch of games."""
        return self.estimate_game_cost(input_tokens, output_tokens) * num_games


# ============================================================================
# TIER 1: FREE MODELS (9 models)
# ============================================================================

GROK_4_1_FAST = ModelConfig(
    name="Grok 4.1 Fast",
    openrouter_id="x-ai/grok-4.1-fast:free",
    provider=ModelProvider.XAI,
    tier=ModelTier.FREE,
    architecture=ModelArchitecture.DENSE,
    context_length=2_000_000,
    input_cost_per_million=0.0,
    output_cost_per_million=0.0,
    notes="xAI's best agentic model, 2M context",
    is_reasoning=True
)

GLM_4_5_AIR = ModelConfig(
    name="GLM 4.5 Air",
    openrouter_id="z-ai/glm-4.5-air:free",
    provider=ModelProvider.ZAI,
    tier=ModelTier.FREE,
    architecture=ModelArchitecture.MOE,
    context_length=131_000,
    input_cost_per_million=0.0,
    output_cost_per_million=0.0,
    parameters="106B",
    notes="Chinese MoE model, thinking mode support",
    is_reasoning=True
)

BERT_NEBULON_ALPHA = ModelConfig(
    name="Bert-Nebulon Alpha",
    openrouter_id="openrouter/bert-nebulon-alpha",
    provider=ModelProvider.OPENROUTER,
    tier=ModelTier.FREE,
    architecture=ModelArchitecture.UNKNOWN,
    context_length=256_000,
    input_cost_per_million=0.0,
    output_cost_per_million=0.0,
    notes="Cloaked multimodal model, mystery architecture",
    is_multimodal=True,
    is_cloaked=True
)

LLAMA_4_MAVERICK = ModelConfig(
    name="Llama 4 Maverick",
    openrouter_id="meta-llama/llama-4-maverick:free",
    provider=ModelProvider.META,
    tier=ModelTier.FREE,
    architecture=ModelArchitecture.MOE,
    context_length=256_000,
    input_cost_per_million=0.0,
    output_cost_per_million=0.0,
    parameters="400B",
    notes="Meta flagship MoE, 17B active per pass",
    is_multimodal=True
)

LLAMA_4_SCOUT = ModelConfig(
    name="Llama 4 Scout",
    openrouter_id="meta-llama/llama-4-scout:free",
    provider=ModelProvider.META,
    tier=ModelTier.FREE,
    architecture=ModelArchitecture.MOE,
    context_length=512_000,
    input_cost_per_million=0.0,
    output_cost_per_million=0.0,
    parameters="109B",
    notes="Meta optimized MoE, 512K context"
)

DEEPSEEK_R1_FREE = ModelConfig(
    name="DeepSeek R1 (free)",
    openrouter_id="deepseek/deepseek-r1:free",
    provider=ModelProvider.DEEPSEEK,
    tier=ModelTier.FREE,
    architecture=ModelArchitecture.MOE,
    context_length=164_000,
    input_cost_per_million=0.0,
    output_cost_per_million=0.0,
    parameters="671B",
    notes="Open-source reasoning model, 37B active",
    is_reasoning=True
)

GEMINI_2_FLASH_EXP = ModelConfig(
    name="Gemini 2.0 Flash Exp",
    openrouter_id="google/gemini-2.0-flash-exp:free",
    provider=ModelProvider.GOOGLE,
    tier=ModelTier.FREE,
    architecture=ModelArchitecture.DENSE,
    context_length=1_050_000,
    input_cost_per_million=0.0,
    output_cost_per_million=0.0,
    notes="Google experimental, 1M+ context"
)

MISTRAL_SMALL_3_1 = ModelConfig(
    name="Mistral Small 3.1",
    openrouter_id="mistralai/mistral-small-3.1-24b-instruct:free",
    provider=ModelProvider.MISTRAL,
    tier=ModelTier.FREE,
    architecture=ModelArchitecture.DENSE,
    context_length=128_000,
    input_cost_per_million=0.0,
    output_cost_per_million=0.0,
    parameters="24B",
    notes="Mistral multimodal, sliding window attention",
    is_multimodal=True
)

OPTIMUS_ALPHA = ModelConfig(
    name="Optimus Alpha",
    openrouter_id="openrouter/optimus-alpha",
    provider=ModelProvider.OPENROUTER,
    tier=ModelTier.FREE,
    architecture=ModelArchitecture.UNKNOWN,
    context_length=256_000,
    input_cost_per_million=0.0,
    output_cost_per_million=0.0,
    notes="OpenRouter in-house stealth model",
    is_cloaked=True
)


# ============================================================================
# TIER 2: BUDGET MODELS (2 models)
# ============================================================================

GPT_5_NANO = ModelConfig(
    name="GPT-5 Nano",
    openrouter_id="openai/gpt-5-nano",
    provider=ModelProvider.OPENAI,
    tier=ModelTier.BUDGET,
    architecture=ModelArchitecture.DENSE,
    context_length=400_000,
    input_cost_per_million=0.05,
    output_cost_per_million=0.40,
    notes="OpenAI's fastest/cheapest, 400K context"
)

DEEPSEEK_V3 = ModelConfig(
    name="DeepSeek V3",
    openrouter_id="deepseek/deepseek-chat",
    provider=ModelProvider.DEEPSEEK,
    tier=ModelTier.BUDGET,
    architecture=ModelArchitecture.MOE,
    context_length=131_000,
    input_cost_per_million=0.20,
    output_cost_per_million=0.80,
    parameters="671B",
    notes="Chinese flagship, 37B active"
)


# ============================================================================
# TIER 3: PREMIUM MODELS (Reserved for future runs)
# ============================================================================

CLAUDE_4_5_SONNET = ModelConfig(
    name="Claude 4.5 Sonnet",
    openrouter_id="anthropic/claude-sonnet-4.5",
    provider=ModelProvider.ANTHROPIC,
    tier=ModelTier.PREMIUM,
    architecture=ModelArchitecture.DENSE,
    context_length=200_000,
    input_cost_per_million=3.00,
    output_cost_per_million=15.00,
    notes="Anthropic flagship, best coding/agents"
)

DEEPSEEK_R1_PAID = ModelConfig(
    name="DeepSeek R1 (paid)",
    openrouter_id="deepseek/deepseek-reasoner",
    provider=ModelProvider.DEEPSEEK,
    tier=ModelTier.PREMIUM,
    architecture=ModelArchitecture.MOE,
    context_length=164_000,
    input_cost_per_million=0.20,
    output_cost_per_million=4.50,
    parameters="671B",
    notes="Paid reasoning tier, higher limits",
    is_reasoning=True
)


# ============================================================================
# MODEL COLLECTIONS
# ============================================================================

# All models in the study (current run: 11 models)
ALL_MODELS: List[ModelConfig] = [
    # Free tier (9)
    GROK_4_1_FAST,
    GLM_4_5_AIR,
    BERT_NEBULON_ALPHA,
    LLAMA_4_MAVERICK,
    LLAMA_4_SCOUT,
    DEEPSEEK_R1_FREE,
    GEMINI_2_FLASH_EXP,
    MISTRAL_SMALL_3_1,
    OPTIMUS_ALPHA,
    # Budget tier (2)
    GPT_5_NANO,
    DEEPSEEK_V3,
]

# Models for current run (excluding premium)
CURRENT_RUN_MODELS: List[ModelConfig] = ALL_MODELS

# Free models only
FREE_MODELS: List[ModelConfig] = [m for m in ALL_MODELS if m.is_free]

# Paid models only
PAID_MODELS: List[ModelConfig] = [m for m in ALL_MODELS if not m.is_free]

# Reasoning models
REASONING_MODELS: List[ModelConfig] = [m for m in ALL_MODELS if m.is_reasoning]

# Cloaked/mystery models
CLOAKED_MODELS: List[ModelConfig] = [m for m in ALL_MODELS if m.is_cloaked]

# MoE architecture models
MOE_MODELS: List[ModelConfig] = [m for m in ALL_MODELS if m.architecture == ModelArchitecture.MOE]

# Chinese models (DeepSeek, GLM)
CHINESE_MODELS: List[ModelConfig] = [
    m for m in ALL_MODELS
    if m.provider in [ModelProvider.DEEPSEEK, ModelProvider.ZAI]
]

# Western models
WESTERN_MODELS: List[ModelConfig] = [
    m for m in ALL_MODELS
    if m.provider not in [ModelProvider.DEEPSEEK, ModelProvider.ZAI]
]

# Model lookup by OpenRouter ID
MODEL_BY_ID: Dict[str, ModelConfig] = {m.openrouter_id: m for m in ALL_MODELS}

# Reserved for future premium run
PREMIUM_MODELS: List[ModelConfig] = [
    CLAUDE_4_5_SONNET,
    DEEPSEEK_R1_PAID,
]


# ============================================================================
# BATCH CONFIGURATION
# ============================================================================

@dataclass
class BatchConfig:
    """Configuration for a comparison batch."""
    name: str
    models: List[ModelConfig]
    games_per_model: int
    description: str = ""

    @property
    def total_games(self) -> int:
        return len(self.models) * self.games_per_model

    @property
    def estimated_cost(self) -> float:
        return sum(m.estimate_batch_cost(self.games_per_model) for m in self.models)

    @property
    def free_games(self) -> int:
        return sum(self.games_per_model for m in self.models if m.is_free)

    @property
    def paid_games(self) -> int:
        return sum(self.games_per_model for m in self.models if not m.is_free)


# Default batch configuration for Phase 4
DEFAULT_BATCH = BatchConfig(
    name="phase4_multimodel",
    models=CURRENT_RUN_MODELS,
    games_per_model=500,
    description="Phase 4 multi-model comparison: 11 models, 5500 games, ~$55"
)


# ============================================================================
# COMPARISON GROUPS
# ============================================================================

@dataclass
class ComparisonGroup:
    """A group of models for head-to-head comparison."""
    name: str
    models: List[ModelConfig]
    hypothesis: str

    def get_model_ids(self) -> List[str]:
        return [m.openrouter_id for m in self.models]


# Predefined comparison groups
COMPARISON_GROUPS: List[ComparisonGroup] = [
    ComparisonGroup(
        name="chinese_vs_western",
        models=CHINESE_MODELS + WESTERN_MODELS[:3],  # Sample western
        hypothesis="Chinese models show different deception patterns than Western models"
    ),
    ComparisonGroup(
        name="reasoning_vs_standard",
        models=REASONING_MODELS + [GPT_5_NANO, LLAMA_4_MAVERICK],
        hypothesis="Reasoning models exhibit more sophisticated deception strategies"
    ),
    ComparisonGroup(
        name="free_vs_paid",
        models=FREE_MODELS[:3] + PAID_MODELS,
        hypothesis="Paid models outperform free models in strategic gameplay"
    ),
    ComparisonGroup(
        name="moe_vs_dense",
        models=MOE_MODELS[:3] + [m for m in ALL_MODELS if m.architecture == ModelArchitecture.DENSE][:3],
        hypothesis="MoE architecture affects deception detection differently than dense"
    ),
    ComparisonGroup(
        name="cloaked_models",
        models=CLOAKED_MODELS,
        hypothesis="Cloaked models provide unique behavioral signatures"
    ),
]


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_model_by_id(openrouter_id: str) -> Optional[ModelConfig]:
    """Get model config by OpenRouter ID."""
    return MODEL_BY_ID.get(openrouter_id)


def get_models_by_tier(tier: ModelTier) -> List[ModelConfig]:
    """Get all models in a specific tier."""
    return [m for m in ALL_MODELS if m.tier == tier]


def get_models_by_provider(provider: ModelProvider) -> List[ModelConfig]:
    """Get all models from a specific provider."""
    return [m for m in ALL_MODELS if m.provider == provider]


def estimate_total_cost(models: List[ModelConfig], games_per_model: int) -> float:
    """Estimate total cost for a list of models."""
    return sum(m.estimate_batch_cost(games_per_model) for m in models)


def print_cost_summary(batch: BatchConfig = DEFAULT_BATCH):
    """Print a summary of costs for a batch."""
    print(f"\n{'='*60}")
    print(f"Batch: {batch.name}")
    print(f"{'='*60}")
    print(f"Models: {len(batch.models)}")
    print(f"Games per model: {batch.games_per_model}")
    print(f"Total games: {batch.total_games}")
    print(f"Free games: {batch.free_games}")
    print(f"Paid games: {batch.paid_games}")
    print(f"Estimated cost: ${batch.estimated_cost:.2f}")
    print(f"{'='*60}")

    print("\nCost breakdown by model:")
    print(f"{'Model':<25} {'Tier':<10} {'Games':>8} {'Cost':>10}")
    print("-" * 55)
    for model in batch.models:
        cost = model.estimate_batch_cost(batch.games_per_model)
        tier = "FREE" if model.is_free else model.tier.value
        print(f"{model.name:<25} {tier:<10} {batch.games_per_model:>8} ${cost:>9.2f}")
    print("-" * 55)
    print(f"{'TOTAL':<25} {'':<10} {batch.total_games:>8} ${batch.estimated_cost:>9.2f}")


if __name__ == "__main__":
    # Print cost summary when run directly
    print_cost_summary()

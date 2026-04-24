"""Token cost calculation utilities for LLM operations."""

from typing import Optional


# Pricing per 1 million tokens (USD)
# Updated January 2025
PRICING = {
    # Gemini models
    "gemini-2.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-3-flash-preview": {"input": 0.10, "output": 0.40},
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
    "text-embedding-004": {"input": 0.00, "output": 0.00},  # Free tier
    # Claude models
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    "claude-3-5-sonnet-latest": {"input": 3.00, "output": 15.00},
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
    "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},
    # OpenAI models (for future support)
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}


def calculate_cost(
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> float:
    """Calculate the cost for token usage.

    Args:
        model: Model identifier (e.g., "gemini-2.5-flash")
        input_tokens: Number of input/prompt tokens
        output_tokens: Number of output/completion tokens

    Returns:
        Estimated cost in USD
    """
    pricing = PRICING.get(model, {"input": 0.0, "output": 0.0})
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    return input_cost + output_cost


def format_cost(cost: float) -> str:
    """Format cost as a human-readable string.

    Args:
        cost: Cost in USD

    Returns:
        Formatted string like "$0.0023" or "<$0.01"
    """
    if cost == 0:
        return "$0.00"
    elif cost < 0.01:
        return f"<$0.01 (${cost:.4f})"
    else:
        return f"${cost:.2f}"


def format_tokens(tokens: int) -> str:
    """Format token count as a human-readable string.

    Args:
        tokens: Number of tokens

    Returns:
        Formatted string like "1.2K" or "15.3M"
    """
    if tokens >= 1_000_000:
        return f"{tokens / 1_000_000:.1f}M"
    elif tokens >= 1_000:
        return f"{tokens / 1_000:.1f}K"
    else:
        return str(tokens)


def get_model_pricing(model: str) -> Optional[dict[str, float]]:
    """Get pricing for a specific model.

    Args:
        model: Model identifier

    Returns:
        Dict with "input" and "output" prices per 1M tokens, or None if unknown
    """
    return PRICING.get(model)

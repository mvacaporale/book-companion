"""Factory for creating LLM clients."""

from typing import Literal, Optional

from .base import LLMClient
from .gemini_client import GeminiClient
from .claude_client import ClaudeClient


ProviderType = Literal["gemini", "claude"]


def get_llm_client(
    provider: ProviderType = "gemini",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> LLMClient:
    """Get an LLM client for the specified provider.

    Args:
        provider: The LLM provider ("gemini" or "claude")
        api_key: Optional API key (uses env var if not provided)
        model: Optional model override

    Returns:
        An LLMClient instance

    Raises:
        ValueError: If the provider is not supported
    """
    if provider == "gemini":
        return GeminiClient(api_key=api_key, model=model)
    elif provider == "claude":
        return ClaudeClient(api_key=api_key, model=model)
    else:
        raise ValueError(f"Unsupported provider: {provider}. Use 'gemini' or 'claude'.")


def list_providers() -> list[str]:
    """List available LLM providers."""
    return ["gemini", "claude"]


def get_default_model(provider: ProviderType) -> str:
    """Get the default model for a provider."""
    if provider == "gemini":
        return GeminiClient.model
    elif provider == "claude":
        return ClaudeClient.model
    else:
        raise ValueError(f"Unsupported provider: {provider}")

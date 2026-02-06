"""Base class for LLM clients."""

from abc import ABC, abstractmethod
from typing import Generator, Optional

from pydantic import BaseModel


class LLMResponse(BaseModel):
    """Response from an LLM."""
    content: str
    model: str
    provider: str
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None


class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    provider: str
    model: str

    @abstractmethod
    def chat(
        self,
        messages: list[dict[str, str]],
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        """Send a chat request to the LLM.

        Args:
            messages: List of {"role": "user"|"assistant", "content": str}
            system_prompt: Optional system prompt

        Returns:
            LLMResponse with the assistant's response
        """
        pass

    @abstractmethod
    def chat_stream(
        self,
        messages: list[dict[str, str]],
        system_prompt: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """Stream a chat response from the LLM.

        Args:
            messages: List of {"role": "user"|"assistant", "content": str}
            system_prompt: Optional system prompt

        Yields:
            Chunks of the response text
        """
        pass

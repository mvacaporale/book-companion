"""Claude LLM client."""

import os
from typing import Generator, Optional

import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import LLMClient, LLMResponse


class ClaudeClient(LLMClient):
    """LLM client for Anthropic Claude."""

    provider = "claude"
    model = "claude-sonnet-4-20250514"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """Initialize the Claude client.

        Args:
            api_key: Anthropic API key. Defaults to ANTHROPIC_API_KEY env var.
            model: Model to use. Defaults to claude-sonnet-4.
        """
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable not set. "
                "Please set it in ~/.zshrc or pass api_key parameter."
            )

        self.client = anthropic.Anthropic(api_key=api_key)
        if model:
            self.model = model

    def _convert_messages(
        self,
        messages: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        """Convert messages to Claude format (already compatible)."""
        return [
            {"role": msg["role"], "content": msg["content"]}
            for msg in messages
        ]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    def chat(
        self,
        messages: list[dict[str, str]],
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        """Send a chat request to Claude."""
        converted_messages = self._convert_messages(messages)

        kwargs = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": converted_messages,
        }

        if system_prompt:
            kwargs["system"] = system_prompt

        response = self.client.messages.create(**kwargs)

        return LLMResponse(
            content=response.content[0].text,
            model=self.model,
            provider=self.provider,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

    def chat_stream(
        self,
        messages: list[dict[str, str]],
        system_prompt: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """Stream a chat response from Claude."""
        converted_messages = self._convert_messages(messages)

        kwargs = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": converted_messages,
        }

        if system_prompt:
            kwargs["system"] = system_prompt

        with self.client.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                yield text

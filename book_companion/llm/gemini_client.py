"""Gemini LLM client."""

import os
from typing import Generator, Optional

from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import LLMClient, LLMResponse


class GeminiClient(LLMClient):
    """LLM client for Google Gemini."""

    provider = "gemini"
    model = "gemini-2.5-flash"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """Initialize the Gemini client.

        Args:
            api_key: Gemini API key. Defaults to GEMINI_API_KEY env var.
            model: Model to use. Defaults to gemini-2.5-flash.
        """
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable not set. "
                "Please set it in ~/.zshrc or pass api_key parameter."
            )

        self.client = genai.Client(api_key=api_key)
        if model:
            self.model = model

    def _convert_messages(
        self,
        messages: list[dict[str, str]],
    ) -> list[types.Content]:
        """Convert messages to Gemini format."""
        contents = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part(text=msg["content"])],
                )
            )
        return contents

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    def chat(
        self,
        messages: list[dict[str, str]],
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        """Send a chat request to Gemini."""
        contents = self._convert_messages(messages)

        config = None
        if system_prompt:
            config = types.GenerateContentConfig(
                system_instruction=system_prompt,
            )

        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )

        # Extract token counts if available
        input_tokens = None
        output_tokens = None
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            input_tokens = getattr(response.usage_metadata, "prompt_token_count", None)
            output_tokens = getattr(response.usage_metadata, "candidates_token_count", None)

        return LLMResponse(
            content=response.text,
            model=self.model,
            provider=self.provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def chat_stream(
        self,
        messages: list[dict[str, str]],
        system_prompt: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """Stream a chat response from Gemini."""
        contents = self._convert_messages(messages)

        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
        ) if system_prompt else None

        response = self.client.models.generate_content_stream(
            model=self.model,
            contents=contents,
            config=config,
        )

        for chunk in response:
            if chunk.text:
                yield chunk.text

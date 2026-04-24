"""Embedding client using Gemini text-embedding-004."""

import os
from typing import Optional

from google import genai
from tenacity import retry, stop_after_attempt, wait_exponential

from book_companion.models import Chunk


class EmbeddingClient:
    """Client for generating embeddings using Gemini."""

    MODEL = "gemini-embedding-001"
    EMBEDDING_DIM = 768  # Using MRL to output 768 dims for compatibility
    MAX_BATCH_SIZE = 100  # Gemini's batch limit

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the embedding client.

        Args:
            api_key: Gemini API key. If not provided, reads from GEMINI_API_KEY env var.
        """
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable not set. "
                "Please set it in ~/.zshrc or pass api_key parameter."
            )
        self.client = genai.Client(api_key=api_key)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    def embed_text(self, text: str) -> list[float]:
        """Generate an embedding for a single text.

        Args:
            text: The text to embed

        Returns:
            768-dimensional embedding vector
        """
        # Handle empty text gracefully
        if not text or not text.strip():
            return [0.0] * self.EMBEDDING_DIM

        result = self.client.models.embed_content(
            model=self.MODEL,
            contents=[text],
            config={"output_dimensionality": self.EMBEDDING_DIM},
        )
        return list(result.embeddings[0].values)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of 768-dimensional embedding vectors
        """
        if not texts:
            return []

        # Track which texts are non-empty and their original indices
        non_empty_indices = []
        non_empty_texts = []
        for i, text in enumerate(texts):
            if text and text.strip():
                non_empty_indices.append(i)
                non_empty_texts.append(text)

        # If all texts are empty, return zero vectors
        if not non_empty_texts:
            return [[0.0] * self.EMBEDDING_DIM for _ in texts]

        # Embed non-empty texts
        non_empty_embeddings = []
        for i in range(0, len(non_empty_texts), self.MAX_BATCH_SIZE):
            batch = non_empty_texts[i : i + self.MAX_BATCH_SIZE]
            result = self.client.models.embed_content(
                model=self.MODEL,
                contents=batch,
                config={"output_dimensionality": self.EMBEDDING_DIM},
            )
            batch_embeddings = [list(e.values) for e in result.embeddings]
            non_empty_embeddings.extend(batch_embeddings)

        # Reconstruct full embeddings list with zero vectors for empty texts
        all_embeddings = [[0.0] * self.EMBEDDING_DIM for _ in texts]
        for orig_idx, embedding in zip(non_empty_indices, non_empty_embeddings):
            all_embeddings[orig_idx] = embedding

        return all_embeddings

    def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """Generate embeddings for a list of chunks.

        Modifies chunks in place and also returns them.

        Args:
            chunks: List of Chunk objects

        Returns:
            The same chunks with embeddings populated
        """
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embed_texts(texts)

        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding

        return chunks

    def embed_query(self, query: str) -> list[float]:
        """Generate an embedding for a query.

        This is a semantic alias for embed_text, but could be
        modified later if query embeddings need different handling.

        Args:
            query: The query text to embed

        Returns:
            768-dimensional embedding vector
        """
        return self.embed_text(query)

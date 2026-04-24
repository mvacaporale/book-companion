"""RAG chat engine for book conversations."""

from typing import Generator, Optional

from book_companion.models import Book, BookIndex, ChatRole, RetrievedContext, Session
from book_companion.processing import EmbeddingClient
from book_companion.storage import VectorStore, get_vector_store
from book_companion.llm import LLMClient, LLMResponse, get_llm_client

from .prompts import build_system_prompt, build_context_prompt
from .session import SessionManager


class ChatEngine:
    """RAG-powered chat engine for book conversations."""

    def __init__(
        self,
        book: Book,
        provider: str = "gemini",
        book_index: Optional[BookIndex] = None,
        vector_store: Optional[VectorStore] = None,
        embedding_client: Optional[EmbeddingClient] = None,
        llm_client: Optional[LLMClient] = None,
        session_manager: Optional[SessionManager] = None,
        n_context_chunks: int = 8,
    ):
        """Initialize the chat engine.

        Args:
            book: The book to chat about
            provider: LLM provider ("gemini" or "claude")
            book_index: Pre-computed book index with summaries and narratives
            vector_store: VectorStore instance
            embedding_client: EmbeddingClient instance
            llm_client: LLMClient instance
            session_manager: SessionManager instance
            n_context_chunks: Number of chunks to retrieve for context
        """
        self.book = book
        self.provider = provider
        self.book_index = book_index
        self.n_context_chunks = n_context_chunks

        # Initialize components
        self.vector_store = vector_store or get_vector_store()
        self.embedding_client = embedding_client or EmbeddingClient()
        self.llm_client = llm_client or get_llm_client(provider)
        self.session_manager = session_manager or SessionManager()

        # Build system prompt (with book index if available)
        self.system_prompt = build_system_prompt(
            title=book.title,
            author=book.author,
            book_index=book_index,
        )

        # Current session
        self._session: Optional[Session] = None

    @property
    def session(self) -> Session:
        """Get or create the current session."""
        if self._session is None:
            self._session = self.session_manager.create_session(
                book_id=self.book.id,
                provider=self.provider,
            )
        return self._session

    def load_session(self, session_id: str) -> bool:
        """Load an existing session.

        Args:
            session_id: The session ID to load

        Returns:
            True if session was loaded successfully
        """
        session = self.session_manager.load_session(self.book.id, session_id)
        if session:
            self._session = session
            return True
        return False

    def retrieve_context(self, query: str) -> RetrievedContext:
        """Retrieve relevant context for a query.

        Args:
            query: The user's question

        Returns:
            RetrievedContext with chunks and formatted text
        """
        # Generate query embedding
        query_embedding = self.embedding_client.embed_query(query)

        # Query vector store
        context = self.vector_store.query(
            book_id=self.book.id,
            query_embedding=query_embedding,
            n_results=self.n_context_chunks,
        )

        return context

    def chat(self, user_message: str) -> tuple[str, RetrievedContext]:
        """Send a message and get a response.

        Args:
            user_message: The user's message

        Returns:
            Tuple of (response text, retrieved context)
        """
        # Retrieve context
        context = self.retrieve_context(user_message)

        # Build the contextual message
        contextual_message = build_context_prompt(
            context=context.formatted_context,
            user_query=user_message,
        )

        # Add user message to session (original, not contextual)
        self.session_manager.add_message(
            self.session,
            ChatRole.USER,
            user_message,
        )

        # Get conversation history (excluding the message we just added)
        history = self.session_manager.get_message_history(
            self.session,
            max_messages=10,
        )

        # Replace the last user message with the contextual one
        if history and history[-1]["role"] == "user":
            history[-1]["content"] = contextual_message

        # Get LLM response
        response = self.llm_client.chat(
            messages=history,
            system_prompt=self.system_prompt,
        )

        # Add assistant response to session with token tracking
        self.session_manager.add_message(
            self.session,
            ChatRole.ASSISTANT,
            response.content,
            citations=context.chunk_ids,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )

        return response.content, context

    def chat_stream(
        self,
        user_message: str,
    ) -> Generator[tuple[str, Optional[RetrievedContext]], None, None]:
        """Stream a response to a message.

        Args:
            user_message: The user's message

        Yields:
            Tuples of (text chunk, context). Context is only included in first yield.
        """
        # Retrieve context
        context = self.retrieve_context(user_message)

        # Build the contextual message
        contextual_message = build_context_prompt(
            context=context.formatted_context,
            user_query=user_message,
        )

        # Add user message to session
        self.session_manager.add_message(
            self.session,
            ChatRole.USER,
            user_message,
        )

        # Get conversation history
        history = self.session_manager.get_message_history(
            self.session,
            max_messages=10,
        )

        # Replace the last user message with the contextual one
        if history and history[-1]["role"] == "user":
            history[-1]["content"] = contextual_message

        # Stream response
        full_response = ""
        first_chunk = True

        for chunk in self.llm_client.chat_stream(
            messages=history,
            system_prompt=self.system_prompt,
        ):
            full_response += chunk
            if first_chunk:
                yield chunk, context
                first_chunk = False
            else:
                yield chunk, None

        # Add assistant response to session
        self.session_manager.add_message(
            self.session,
            ChatRole.ASSISTANT,
            full_response,
            citations=context.chunk_ids,
        )

    def get_session_summary(self) -> dict:
        """Get a summary of the current session.

        Returns:
            Dict with session info
        """
        return {
            "session_id": self.session.id,
            "book_id": self.book.id,
            "provider": self.provider,
            "message_count": len(self.session.messages),
            "created_at": self.session.created_at.isoformat(),
            "updated_at": self.session.updated_at.isoformat(),
            "has_index": self.book_index is not None,
        }

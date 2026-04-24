"""Chat session management."""

from typing import Optional

from book_companion.models import ChatMessage, ChatRole, Session
from book_companion.storage import SessionStore, get_session_store


class SessionManager:
    """Manages chat sessions for books."""

    def __init__(self, store: Optional[SessionStore] = None):
        """Initialize the session manager.

        Args:
            store: SessionStore instance. Creates one if not provided.
        """
        self.store = store or get_session_store()

    def create_session(self, book_id: str, provider: str = "gemini") -> Session:
        """Create a new chat session.

        Args:
            book_id: The book ID
            provider: The LLM provider

        Returns:
            New Session instance
        """
        session = Session(book_id=book_id, provider=provider)
        self.store.save(session)
        return session

    def load_session(self, book_id: str, session_id: str) -> Optional[Session]:
        """Load an existing session.

        Args:
            book_id: The book ID
            session_id: The session ID

        Returns:
            Session if found, None otherwise
        """
        return self.store.load(book_id, session_id)

    def save_session(self, session: Session) -> None:
        """Save a session.

        Args:
            session: The session to save
        """
        self.store.save(session)

    def list_sessions(self, book_id: str) -> list[Session]:
        """List all sessions for a book.

        Args:
            book_id: The book ID

        Returns:
            List of sessions, most recent first
        """
        return self.store.list_sessions(book_id)

    def get_latest_session(self, book_id: str) -> Optional[Session]:
        """Get the most recent session for a book.

        Args:
            book_id: The book ID

        Returns:
            Most recent session or None
        """
        return self.store.get_latest_session(book_id)

    def delete_session(self, book_id: str, session_id: str) -> bool:
        """Delete a session.

        Args:
            book_id: The book ID
            session_id: The session ID

        Returns:
            True if deleted
        """
        return self.store.delete_session(book_id, session_id)

    def add_message(
        self,
        session: Session,
        role: ChatRole,
        content: str,
        citations: Optional[list[str]] = None,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
    ) -> ChatMessage:
        """Add a message to a session and save.

        Args:
            session: The session
            role: Message role (user/assistant)
            content: Message content
            citations: Optional chunk IDs used for context
            input_tokens: Optional token count for input (assistant messages)
            output_tokens: Optional token count for output (assistant messages)

        Returns:
            The created ChatMessage
        """
        message = session.add_message(
            role,
            content,
            citations,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        self.store.save(session)
        return message

    def get_message_history(
        self,
        session: Session,
        max_messages: int = 20,
    ) -> list[dict[str, str]]:
        """Get message history in LLM-compatible format.

        Args:
            session: The session
            max_messages: Maximum number of messages to include

        Returns:
            List of {"role": str, "content": str} dicts
        """
        messages = session.messages[-max_messages:]
        return [
            {"role": msg.role.value, "content": msg.content}
            for msg in messages
            if msg.role != ChatRole.SYSTEM
        ]

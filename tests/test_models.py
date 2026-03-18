"""Tests for data models."""

import pytest

from book_companion.models import (
    TokenUsage,
    Session,
    ChatMessage,
    ChatRole,
)


class TestTokenUsage:
    """Test cases for TokenUsage model."""

    def test_empty_usage(self) -> None:
        """Empty usage should have zero tokens."""
        usage = TokenUsage()
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.total_tokens == 0

    def test_initial_values(self) -> None:
        """TokenUsage should accept initial values."""
        usage = TokenUsage(input_tokens=100, output_tokens=50)
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.total_tokens == 150

    def test_add_tokens(self) -> None:
        """Adding tokens should accumulate correctly."""
        usage = TokenUsage()
        usage.add(input_tokens=100, output_tokens=50)
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50

        usage.add(input_tokens=200, output_tokens=100)
        assert usage.input_tokens == 300
        assert usage.output_tokens == 150
        assert usage.total_tokens == 450

    def test_add_none_values(self) -> None:
        """Adding None values should be ignored."""
        usage = TokenUsage(input_tokens=100, output_tokens=50)
        usage.add(input_tokens=None, output_tokens=None)
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50


class TestChatMessage:
    """Test cases for ChatMessage model."""

    def test_user_message(self) -> None:
        """User messages should be created correctly."""
        msg = ChatMessage(role=ChatRole.USER, content="Hello")
        assert msg.role == ChatRole.USER
        assert msg.content == "Hello"
        assert msg.input_tokens is None
        assert msg.output_tokens is None

    def test_assistant_message_with_tokens(self) -> None:
        """Assistant messages can have token counts."""
        msg = ChatMessage(
            role=ChatRole.ASSISTANT,
            content="Hi there!",
            input_tokens=50,
            output_tokens=25,
        )
        assert msg.role == ChatRole.ASSISTANT
        assert msg.input_tokens == 50
        assert msg.output_tokens == 25


class TestSession:
    """Test cases for Session model."""

    def test_empty_session(self) -> None:
        """Empty session should have no messages."""
        session = Session(id="test-session", book_id="test-book", provider="gemini")
        assert len(session.messages) == 0

    def test_get_total_usage_empty(self) -> None:
        """Empty session should have zero usage."""
        session = Session(id="test-session", book_id="test-book", provider="gemini")
        usage = session.get_total_usage()
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0

    def test_get_total_usage_with_messages(self) -> None:
        """Session usage should sum all message tokens."""
        session = Session(id="test-session", book_id="test-book", provider="gemini")
        session.messages = [
            ChatMessage(role=ChatRole.USER, content="Q1"),
            ChatMessage(
                role=ChatRole.ASSISTANT,
                content="A1",
                input_tokens=100,
                output_tokens=50,
            ),
            ChatMessage(role=ChatRole.USER, content="Q2"),
            ChatMessage(
                role=ChatRole.ASSISTANT,
                content="A2",
                input_tokens=150,
                output_tokens=75,
            ),
        ]
        usage = session.get_total_usage()
        assert usage.input_tokens == 250
        assert usage.output_tokens == 125
        assert usage.total_tokens == 375

    def test_get_total_usage_with_none_tokens(self) -> None:
        """Messages without token counts should be handled gracefully."""
        session = Session(id="test-session", book_id="test-book", provider="gemini")
        session.messages = [
            ChatMessage(role=ChatRole.USER, content="Q1"),
            ChatMessage(role=ChatRole.ASSISTANT, content="A1"),
            ChatMessage(
                role=ChatRole.ASSISTANT,
                content="A2",
                input_tokens=100,
                output_tokens=50,
            ),
        ]
        usage = session.get_total_usage()
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50

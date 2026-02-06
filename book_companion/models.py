"""Pydantic models for book-companion."""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field


def generate_id() -> str:
    """Generate a short unique ID."""
    return uuid4().hex[:8]


class BookFormat(str, Enum):
    """Supported book formats."""
    PDF = "pdf"
    EPUB = "epub"
    MARKDOWN = "markdown"


class TokenUsage(BaseModel):
    """Token usage tracking for LLM operations."""
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return self.input_tokens + self.output_tokens

    def add(self, input_tokens: int = 0, output_tokens: int = 0) -> None:
        """Add token counts to the running total."""
        self.input_tokens += input_tokens or 0
        self.output_tokens += output_tokens or 0

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        """Combine two TokenUsage instances."""
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )


class ChunkMetadata(BaseModel):
    """Metadata for a text chunk."""
    book_id: str
    chapter_title: Optional[str] = None
    chapter_number: Optional[int] = None
    page_number: Optional[int] = None
    start_char: int
    end_char: int


class Chunk(BaseModel):
    """A chunk of text from a book with metadata."""
    id: str = Field(default_factory=generate_id)
    text: str
    metadata: ChunkMetadata
    embedding: Optional[list[float]] = None


class Chapter(BaseModel):
    """A chapter from a book."""
    number: int
    title: Optional[str] = None
    content: str
    start_page: Optional[int] = None
    end_page: Optional[int] = None


class ParsedBook(BaseModel):
    """A parsed book with chapters and metadata."""
    title: str
    author: Optional[str] = None
    chapters: list[Chapter]
    format: BookFormat
    total_pages: Optional[int] = None
    raw_text: str  # Full text for non-chapter-aware processing


class Book(BaseModel):
    """A book record in the registry."""
    id: str = Field(default_factory=generate_id)
    title: str
    author: Optional[str] = None
    format: BookFormat
    file_path: str
    file_hash: str  # MD5 hash for deduplication
    total_chunks: int = 0
    total_pages: Optional[int] = None
    ingested_at: datetime = Field(default_factory=datetime.now)
    # Token tracking for ingestion
    embedding_tokens: int = 0
    summarization_input_tokens: int = 0
    summarization_output_tokens: int = 0

    @property
    def display_name(self) -> str:
        """Return a display name for the book."""
        if self.author:
            return f"{self.title} by {self.author}"
        return self.title

    @property
    def total_ingestion_tokens(self) -> int:
        """Total tokens used during ingestion."""
        return self.embedding_tokens + self.summarization_input_tokens + self.summarization_output_tokens


class ChatRole(str, Enum):
    """Chat message roles."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChatMessage(BaseModel):
    """A chat message with role and content."""
    role: ChatRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    citations: Optional[list[str]] = None  # Chunk IDs used for context
    input_tokens: Optional[int] = None  # Tokens in prompt (for ASSISTANT messages)
    output_tokens: Optional[int] = None  # Tokens in response (for ASSISTANT messages)


class Session(BaseModel):
    """A chat session for a book."""
    id: str = Field(default_factory=generate_id)
    book_id: str
    provider: str = "gemini"  # gemini or claude
    messages: list[ChatMessage] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    def add_message(
        self,
        role: ChatRole,
        content: str,
        citations: Optional[list[str]] = None,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
    ) -> ChatMessage:
        """Add a message to the session."""
        message = ChatMessage(
            role=role,
            content=content,
            citations=citations,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        self.messages.append(message)
        self.updated_at = datetime.now()
        return message

    def get_total_usage(self) -> TokenUsage:
        """Calculate total token usage for this session."""
        usage = TokenUsage()
        for msg in self.messages:
            if msg.role == ChatRole.ASSISTANT:
                usage.add(msg.input_tokens, msg.output_tokens)
        return usage


class BookRegistry(BaseModel):
    """Registry of all ingested books."""
    books: dict[str, Book] = Field(default_factory=dict)

    def add_book(self, book: Book) -> None:
        """Add a book to the registry."""
        self.books[book.id] = book

    def get_book(self, book_id: str) -> Optional[Book]:
        """Get a book by ID."""
        return self.books.get(book_id)

    def remove_book(self, book_id: str) -> Optional[Book]:
        """Remove a book from the registry."""
        return self.books.pop(book_id, None)

    def find_by_hash(self, file_hash: str) -> Optional[Book]:
        """Find a book by file hash."""
        for book in self.books.values():
            if book.file_hash == file_hash:
                return book
        return None

    def list_books(self) -> list[Book]:
        """List all books."""
        return list(self.books.values())


class RetrievedContext(BaseModel):
    """Context retrieved from the vector store."""
    chunks: list[Chunk]
    formatted_context: str
    chunk_ids: list[str]


# =============================================================================
# Hierarchical Summarization Models
# =============================================================================


class NarrativeType(str, Enum):
    """Types of narratives/examples in a book."""
    STORY = "story"          # Anecdotes, personal narratives
    STUDY = "study"          # Research studies, experiments
    CASE_STUDY = "case_study"  # Real-world examples, business cases
    EXAMPLE = "example"      # Illustrative examples
    QUOTE = "quote"          # Notable quotes


class Narrative(BaseModel):
    """A story, study, or example from the book."""
    type: NarrativeType
    title: str  # Brief descriptive title
    description: str  # What happened / what it illustrates
    source: Optional[str] = None  # Researcher name, person in story, etc.
    chapter_number: int
    page_reference: Optional[str] = None  # "pp. 45-48" or "p. 45"


class ChapterSummary(BaseModel):
    """Summary of a single chapter."""
    chapter_number: int
    chapter_title: Optional[str] = None
    page_range: Optional[str] = None  # "pp. 1-24"
    summary: str  # 2-3 paragraph summary
    key_concepts: list[str]  # Main ideas/arguments
    narratives: list[Narrative] = Field(default_factory=list)


class BookSummary(BaseModel):
    """High-level summary of the entire book."""
    overview: str  # 2-3 paragraph overview
    main_thesis: str  # The book's central argument
    key_themes: list[str]  # Major recurring themes
    target_audience: Optional[str] = None


class ChapterIndexEntry(BaseModel):
    """Entry in the book's navigational index."""
    chapter_number: int
    title: Optional[str] = None
    page_range: Optional[str] = None
    core_argument: str  # One-line summary
    key_topics: list[str]  # 3-5 topics covered
    notable_content: list[str] = Field(default_factory=list)  # Stories/studies (brief)


class BookIndex(BaseModel):
    """Complete index for a book, used for LLM navigation."""
    book_id: str
    title: str
    author: Optional[str] = None
    book_summary: BookSummary
    chapter_summaries: list[ChapterSummary] = Field(default_factory=list)
    chapter_index: list[ChapterIndexEntry] = Field(default_factory=list)
    all_narratives: list[Narrative] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    model_used: str = "gemini-2.5-flash"  # Track which model generated summaries
    # Token tracking for summarization
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """Total tokens used for summarization."""
        return self.total_input_tokens + self.total_output_tokens

    def get_navigation_prompt(self) -> str:
        """Generate a structured index for the LLM system prompt."""
        lines = [
            f"BOOK: {self.title}" + (f" by {self.author}" if self.author else ""),
            "",
            "OVERVIEW:",
            self.book_summary.overview,
            "",
            f"MAIN THESIS: {self.book_summary.main_thesis}",
            "",
            "KEY THEMES: " + ", ".join(self.book_summary.key_themes),
            "",
            "CHAPTER INDEX:",
        ]

        for entry in self.chapter_index:
            title_str = f'"{entry.title}"' if entry.title else f"Chapter {entry.chapter_number}"
            page_str = f" ({entry.page_range})" if entry.page_range else ""
            lines.append(f"\n{entry.chapter_number}. {title_str}{page_str}")
            lines.append(f"   Core argument: {entry.core_argument}")
            lines.append(f"   Topics: {', '.join(entry.key_topics)}")
            if entry.notable_content:
                lines.append(f"   Notable: {'; '.join(entry.notable_content)}")

        if self.all_narratives:
            lines.append("\n\nKEY STORIES & STUDIES:")
            for n in self.all_narratives[:15]:  # Limit to top 15
                type_label = n.type.value.replace("_", " ").title()
                lines.append(f"- [{type_label}] {n.title} (Ch. {n.chapter_number})")

        return "\n".join(lines)

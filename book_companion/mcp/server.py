"""MCP server for book-companion.

Provides tools for searching books, accessing summaries, and RAG chat.
"""

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from book_companion.models import (
    Book,
    BookIndex,
    ChatRole,
    ChapterSummary,
    Chunk,
    NarrativeType,
    Session,
)
from book_companion.utils.costs import calculate_cost, format_cost, format_tokens

# Initialize FastMCP server with transport security settings for Cloud Run
# Disable DNS rebinding protection - Cloud Run provides its own security layer
mcp = FastMCP(
    "book-companion",
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=False,
    ),
)

# Thread pool for running sync code
_executor = ThreadPoolExecutor(max_workers=4)

# Lazy-initialized global context
_ctx: Optional[dict] = None


def get_context() -> dict:
    """Get or create shared storage context."""
    global _ctx
    if _ctx is None:
        from book_companion.storage import (
            BookRegistryStore,
            BookIndexStore,
            SessionStore,
            VectorStore,
        )
        from book_companion.processing import EmbeddingClient

        _ctx = {
            "registry": BookRegistryStore(),
            "index_store": BookIndexStore(),
            "session_store": SessionStore(),
            "vector_store": VectorStore(),
            "embedding_client": EmbeddingClient(),
        }
    return _ctx


async def run_sync(func, *args, **kwargs):
    """Run a sync function in the thread pool."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _executor, lambda: func(*args, **kwargs)
    )


# =============================================================================
# Tool: list_books
# =============================================================================


@mcp.tool()
async def list_books(topic: Optional[str] = None) -> str:
    """List all ingested books with metadata.

    Args:
        topic: Optional topic/theme to filter by (fuzzy matches against
               book themes and chapter topics from the book index)

    Returns a JSON array of books with their IDs, titles, authors, formats,
    chunk counts, and whether they have an index (summaries).
    """
    from rapidfuzz import fuzz

    ctx = get_context()
    books = await run_sync(ctx["registry"].list_books)

    result = []
    for book in books:
        has_index = await run_sync(ctx["index_store"].exists, book.id)

        book_data = {
            "id": book.id,
            "title": book.title,
            "author": book.author,
            "format": book.format.value,
            "total_chunks": book.total_chunks,
            "total_pages": book.total_pages,
            "has_index": has_index,
            "ingested_at": book.ingested_at.isoformat(),
        }

        # If topic filter provided, check for matches
        if topic:
            if not has_index:
                continue  # Skip books without index when filtering by topic

            index = await run_sync(ctx["index_store"].load, book.id)
            matched_themes = []
            matched_topics = []
            threshold = 70

            for theme in index.book_summary.key_themes:
                score = fuzz.token_set_ratio(topic.lower(), theme.lower())
                if score >= threshold:
                    matched_themes.append(theme)

            for entry in index.chapter_index:
                for t in entry.key_topics:
                    score = fuzz.token_set_ratio(topic.lower(), t.lower())
                    if score >= threshold:
                        matched_topics.append(f"Ch.{entry.chapter_number}: {t}")

            if not matched_themes and not matched_topics:
                continue  # No matches, skip this book

            book_data["matched_themes"] = matched_themes
            book_data["matched_topics"] = matched_topics

        result.append(book_data)

    return json.dumps(result, indent=2)


# =============================================================================
# Tool: search_books
# =============================================================================


@mcp.tool()
async def search_books(
    query: str,
    book_ids: Optional[list[str]] = None,
    n_results: int = 8,
) -> str:
    """Search for relevant passages across books using semantic search.

    Args:
        query: The search query
        book_ids: Optional list of book IDs to search. If omitted, searches all books.
        n_results: Number of results per book (default 8)

    Returns JSON array of matching chunks with text, chapter, page, and book info.
    """
    ctx = get_context()

    # Get books to search
    if book_ids:
        books = []
        for book_id in book_ids:
            book = await run_sync(ctx["registry"].get_book, book_id)
            if book:
                books.append(book)
        if not books:
            return json.dumps({"error": "No valid book IDs provided"})
    else:
        books = await run_sync(ctx["registry"].list_books)

    if not books:
        return json.dumps({"error": "No books ingested. Use the CLI to ingest books first."})

    # Generate query embedding
    query_embedding = await run_sync(ctx["embedding_client"].embed_query, query)

    # Search each book
    all_results = []
    for book in books:
        try:
            retrieved = await run_sync(
                ctx["vector_store"].query,
                book.id,
                query_embedding,
                n_results,
            )
            for chunk in retrieved.chunks:
                all_results.append({
                    "book_id": book.id,
                    "book_title": book.title,
                    "text": chunk.text,
                    "chapter_title": chunk.metadata.chapter_title,
                    "chapter_number": chunk.metadata.chapter_number,
                    "page_number": chunk.metadata.page_number,
                    "chunk_id": chunk.id,
                })
        except Exception:
            # Skip books that fail (e.g., missing collection)
            continue

    return json.dumps(all_results, indent=2)


# =============================================================================
# Tool: get_book_index
# =============================================================================


@mcp.tool()
async def get_book_index(book_id: str) -> str:
    """Get the full navigation index for a book.

    Returns the book's summary, chapter index, chapter summaries, and narratives.
    This provides a structured overview of the book's content.

    Args:
        book_id: The book ID

    Returns JSON with book overview, thesis, themes, and chapter-by-chapter index.
    """
    ctx = get_context()

    # Verify book exists
    book = await run_sync(ctx["registry"].get_book, book_id)
    if not book:
        return json.dumps({"error": f"Book not found: {book_id}"})

    # Load index
    index = await run_sync(ctx["index_store"].load, book_id)
    if not index:
        return json.dumps({
            "error": "No index found for this book. Re-ingest with summarization to generate index.",
            "book_id": book_id,
            "book_title": book.title,
        })

    # Return full index as JSON
    return json.dumps({
        "book_id": index.book_id,
        "title": index.title,
        "author": index.author,
        "book_summary": {
            "overview": index.book_summary.overview,
            "main_thesis": index.book_summary.main_thesis,
            "key_themes": index.book_summary.key_themes,
            "target_audience": index.book_summary.target_audience,
        },
        "chapter_index": [
            {
                "chapter_number": entry.chapter_number,
                "title": entry.title,
                "page_range": entry.page_range,
                "core_argument": entry.core_argument,
                "key_topics": entry.key_topics,
                "notable_content": entry.notable_content,
            }
            for entry in index.chapter_index
        ],
        "narratives_count": len(index.all_narratives),
        "model_used": index.model_used,
        "created_at": index.created_at.isoformat(),
    }, indent=2)


# =============================================================================
# Tool: get_chapter_summary
# =============================================================================


@mcp.tool()
async def get_chapter_summary(book_id: str, chapter_number: int) -> str:
    """Get the summary for a specific chapter.

    Args:
        book_id: The book ID
        chapter_number: The chapter number (1-indexed)

    Returns JSON with chapter summary, key concepts, and narratives.
    """
    ctx = get_context()

    # Verify book exists
    book = await run_sync(ctx["registry"].get_book, book_id)
    if not book:
        return json.dumps({"error": f"Book not found: {book_id}"})

    # Load index
    index = await run_sync(ctx["index_store"].load, book_id)
    if not index:
        return json.dumps({
            "error": "No index found for this book. Re-ingest with summarization to generate index.",
            "book_id": book_id,
        })

    # Find chapter
    chapter_summary = None
    for cs in index.chapter_summaries:
        if cs.chapter_number == chapter_number:
            chapter_summary = cs
            break

    if not chapter_summary:
        return json.dumps({
            "error": f"Chapter {chapter_number} not found",
            "book_id": book_id,
            "available_chapters": [cs.chapter_number for cs in index.chapter_summaries],
        })

    return json.dumps({
        "book_id": book_id,
        "book_title": book.title,
        "chapter_number": chapter_summary.chapter_number,
        "chapter_title": chapter_summary.chapter_title,
        "page_range": chapter_summary.page_range,
        "summary": chapter_summary.summary,
        "key_concepts": chapter_summary.key_concepts,
        "narratives": [
            {
                "type": n.type.value,
                "title": n.title,
                "description": n.description,
                "source": n.source,
                "page_reference": n.page_reference,
            }
            for n in chapter_summary.narratives
        ],
    }, indent=2)


# =============================================================================
# Tool: get_narratives
# =============================================================================


@mcp.tool()
async def get_narratives(
    book_id: str,
    narrative_type: Optional[str] = None,
) -> str:
    """Get stories, studies, and examples from a book.

    Args:
        book_id: The book ID
        narrative_type: Optional filter by type (story, study, case_study, example, quote)

    Returns JSON array of narratives with their details.
    """
    ctx = get_context()

    # Verify book exists
    book = await run_sync(ctx["registry"].get_book, book_id)
    if not book:
        return json.dumps({"error": f"Book not found: {book_id}"})

    # Load index
    index = await run_sync(ctx["index_store"].load, book_id)
    if not index:
        return json.dumps({
            "error": "No index found for this book. Re-ingest with summarization to generate index.",
            "book_id": book_id,
        })

    # Filter narratives
    narratives = index.all_narratives
    if narrative_type:
        try:
            nt = NarrativeType(narrative_type)
            narratives = [n for n in narratives if n.type == nt]
        except ValueError:
            valid_types = [t.value for t in NarrativeType]
            return json.dumps({
                "error": f"Invalid narrative_type: {narrative_type}",
                "valid_types": valid_types,
            })

    return json.dumps({
        "book_id": book_id,
        "book_title": book.title,
        "total_narratives": len(narratives),
        "filter": narrative_type,
        "narratives": [
            {
                "type": n.type.value,
                "title": n.title,
                "description": n.description,
                "source": n.source,
                "chapter_number": n.chapter_number,
                "page_reference": n.page_reference,
            }
            for n in narratives
        ],
    }, indent=2)


# =============================================================================
# Tool: get_stats
# =============================================================================


@mcp.tool()
async def get_stats(book_id: Optional[str] = None) -> str:
    """Get token usage and cost statistics.

    Args:
        book_id: Optional book ID. If provided, shows stats for that book.
                 Otherwise, shows aggregate stats across all books.

    Returns JSON with ingestion tokens/cost and chat tokens/cost.
    """
    ctx = get_context()

    if book_id:
        # Stats for a specific book
        book = await run_sync(ctx["registry"].get_book, book_id)
        if not book:
            return json.dumps({"error": f"Book not found: {book_id}"})

        # Get index for summarization tokens
        index = await run_sync(ctx["index_store"].load, book_id)
        summarization_model = index.model_used if index else "gemini-2.5-flash"

        # Calculate ingestion cost
        ingestion_input = book.summarization_input_tokens
        ingestion_output = book.summarization_output_tokens
        ingestion_cost = calculate_cost(summarization_model, ingestion_input, ingestion_output)

        # Get chat sessions
        sessions = await run_sync(ctx["session_store"].list_sessions, book_id)
        chat_input = 0
        chat_output = 0
        for session in sessions:
            usage = session.get_total_usage()
            chat_input += usage.input_tokens
            chat_output += usage.output_tokens

        # Estimate chat cost (use gemini as default)
        chat_cost = calculate_cost("gemini-2.5-flash", chat_input, chat_output)

        return json.dumps({
            "book_id": book_id,
            "book_title": book.title,
            "ingestion": {
                "model": summarization_model,
                "input_tokens": ingestion_input,
                "output_tokens": ingestion_output,
                "input_tokens_formatted": format_tokens(ingestion_input),
                "output_tokens_formatted": format_tokens(ingestion_output),
                "cost_usd": round(ingestion_cost, 4),
                "cost_formatted": format_cost(ingestion_cost),
            },
            "chat": {
                "total_sessions": len(sessions),
                "total_messages": sum(len(s.messages) for s in sessions),
                "input_tokens": chat_input,
                "output_tokens": chat_output,
                "input_tokens_formatted": format_tokens(chat_input),
                "output_tokens_formatted": format_tokens(chat_output),
                "cost_usd": round(chat_cost, 4),
                "cost_formatted": format_cost(chat_cost),
            },
            "total_cost_usd": round(ingestion_cost + chat_cost, 4),
            "total_cost_formatted": format_cost(ingestion_cost + chat_cost),
        }, indent=2)
    else:
        # Aggregate stats across all books
        books = await run_sync(ctx["registry"].list_books)

        total_ingestion_input = 0
        total_ingestion_output = 0
        total_chat_input = 0
        total_chat_output = 0
        total_sessions = 0
        total_messages = 0

        for book in books:
            total_ingestion_input += book.summarization_input_tokens
            total_ingestion_output += book.summarization_output_tokens

            sessions = await run_sync(ctx["session_store"].list_sessions, book.id)
            total_sessions += len(sessions)
            for session in sessions:
                total_messages += len(session.messages)
                usage = session.get_total_usage()
                total_chat_input += usage.input_tokens
                total_chat_output += usage.output_tokens

        ingestion_cost = calculate_cost(
            "gemini-2.5-flash", total_ingestion_input, total_ingestion_output
        )
        chat_cost = calculate_cost(
            "gemini-2.5-flash", total_chat_input, total_chat_output
        )

        return json.dumps({
            "book_id": None,
            "total_books": len(books),
            "ingestion": {
                "input_tokens": total_ingestion_input,
                "output_tokens": total_ingestion_output,
                "input_tokens_formatted": format_tokens(total_ingestion_input),
                "output_tokens_formatted": format_tokens(total_ingestion_output),
                "cost_usd": round(ingestion_cost, 4),
                "cost_formatted": format_cost(ingestion_cost),
            },
            "chat": {
                "total_sessions": total_sessions,
                "total_messages": total_messages,
                "input_tokens": total_chat_input,
                "output_tokens": total_chat_output,
                "input_tokens_formatted": format_tokens(total_chat_input),
                "output_tokens_formatted": format_tokens(total_chat_output),
                "cost_usd": round(chat_cost, 4),
                "cost_formatted": format_cost(chat_cost),
            },
            "total_cost_usd": round(ingestion_cost + chat_cost, 4),
            "total_cost_formatted": format_cost(ingestion_cost + chat_cost),
        }, indent=2)


# =============================================================================
# Tool: chat
# =============================================================================


@mcp.tool()
async def chat(
    message: str,
    book_ids: list[str],
    session_id: Optional[str] = None,
    provider: str = "gemini",
) -> str:
    """Chat with books using RAG (Retrieval Augmented Generation).

    Searches the specified books for relevant passages and uses them as context
    for the LLM to generate a response.

    Args:
        message: Your question or message
        book_ids: List of book IDs to chat with (required)
        session_id: Optional session ID to resume a previous conversation
        provider: LLM provider - "gemini" (default) or "claude"

    Returns JSON with the response, session info, citations, and token usage.
    """
    ctx = get_context()

    if not book_ids:
        return json.dumps({"error": "book_ids is required. Use list_books to see available books."})

    # Validate book IDs
    books = []
    for book_id in book_ids:
        book = await run_sync(ctx["registry"].get_book, book_id)
        if book:
            books.append(book)

    if not books:
        return json.dumps({"error": "No valid book IDs provided. Use list_books to see available books."})

    # Single book: use existing ChatEngine
    if len(books) == 1:
        return await _chat_single_book(
            ctx, books[0], message, session_id, provider
        )

    # Multi-book: custom handling
    return await _chat_multi_book(
        ctx, books, message, provider
    )


async def _chat_single_book(
    ctx: dict,
    book: Book,
    message: str,
    session_id: Optional[str],
    provider: str,
) -> str:
    """Handle chat with a single book using ChatEngine."""
    from book_companion.chat.engine import ChatEngine
    from book_companion.chat.session import SessionManager

    # Load book index if available
    book_index = await run_sync(ctx["index_store"].load, book.id)

    # Create chat engine
    engine = ChatEngine(
        book=book,
        provider=provider,
        book_index=book_index,
        vector_store=ctx["vector_store"],
        embedding_client=ctx["embedding_client"],
    )

    # Load session if provided
    if session_id:
        loaded = engine.load_session(session_id)
        if not loaded:
            return json.dumps({
                "error": f"Session not found: {session_id}",
                "book_id": book.id,
            })

    # Get response
    response_text, retrieved_context = await run_sync(engine.chat, message)

    # Get token usage from the last message
    session = engine.session
    last_msg = session.messages[-1] if session.messages else None
    input_tokens = last_msg.input_tokens or 0 if last_msg else 0
    output_tokens = last_msg.output_tokens or 0 if last_msg else 0

    # Calculate cost
    model = "gemini-2.5-flash" if provider == "gemini" else "claude-sonnet-4-20250514"
    cost = calculate_cost(model, input_tokens, output_tokens)

    return json.dumps({
        "response": response_text,
        "session_id": session.id,
        "book_id": book.id,
        "book_title": book.title,
        "provider": provider,
        "context_chunks": len(retrieved_context.chunks),
        "citations": [
            {
                "chunk_id": c.id,
                "chapter_title": c.metadata.chapter_title,
                "chapter_number": c.metadata.chapter_number,
                "page_number": c.metadata.page_number,
            }
            for c in retrieved_context.chunks
        ],
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": round(cost, 4),
            "cost_formatted": format_cost(cost),
        },
    }, indent=2)


async def _chat_multi_book(
    ctx: dict,
    books: list[Book],
    message: str,
    provider: str,
) -> str:
    """Handle chat with multiple books."""
    from book_companion.llm import get_llm_client
    from book_companion.chat.prompts import build_context_prompt

    # Get query embedding
    query_embedding = await run_sync(ctx["embedding_client"].embed_query, message)

    # Retrieve context from all books (4 chunks each)
    all_chunks = []
    chunks_per_book = max(4, 8 // len(books))

    for book in books:
        try:
            retrieved = await run_sync(
                ctx["vector_store"].query,
                book.id,
                query_embedding,
                chunks_per_book,
            )
            for chunk in retrieved.chunks:
                # Tag chunk with book title for multi-book context
                all_chunks.append((book, chunk))
        except Exception:
            continue

    if not all_chunks:
        return json.dumps({
            "error": "No relevant content found in the specified books.",
            "book_ids": [b.id for b in books],
        })

    # Build combined context with book attribution
    context_parts = []
    for book, chunk in all_chunks:
        citation_parts = [f"Book: {book.title}"]
        if chunk.metadata.chapter_title:
            citation_parts.append(f"Chapter: {chunk.metadata.chapter_title}")
        elif chunk.metadata.chapter_number:
            citation_parts.append(f"Chapter {chunk.metadata.chapter_number}")
        if chunk.metadata.page_number:
            citation_parts.append(f"Page {chunk.metadata.page_number}")

        citation = ", ".join(citation_parts)
        context_parts.append(f"[{citation}]\n{chunk.text}")

    formatted_context = "\n\n---\n\n".join(context_parts)

    # Build system prompt with all book indices
    system_prompt_parts = [
        "You are a knowledgeable reading companion helping discuss multiple books.",
        "",
        "Response Strategy - Match your response depth to the question:",
        "",
        "1. BROAD QUESTIONS (e.g., 'summarize this book', 'main lessons'):",
        "   - Start with the main thesis and key themes",
        "   - List major structural elements briefly (chapters, principles, etc.)",
        "   - Offer to elaborate: 'Would you like me to go deeper on any of these?'",
        "",
        "2. SPECIFIC QUESTIONS: Give focused, detailed answers with examples",
        "",
        "3. EXPLORATORY QUESTIONS: Use the chapter indices to identify relevant chapters",
        "",
        "Guidelines:",
        "- Base responses on the provided context passages",
        "- When citing, always include which book the information comes from",
        "- Use the chapter indices to understand each book's structure",
        "- Offer to elaborate when giving overviews",
        "",
        "Books in this conversation:",
    ]
    for book in books:
        index = await run_sync(ctx["index_store"].load, book.id)
        if index:
            system_prompt_parts.append(f"\n--- {book.title} ---")
            system_prompt_parts.append(index.get_navigation_prompt())
        else:
            system_prompt_parts.append(f"\n- {book.title}" + (f" by {book.author}" if book.author else ""))

    system_prompt = "\n".join(system_prompt_parts)

    # Build user message with context
    user_message = build_context_prompt(formatted_context, message)

    # Get LLM response
    llm_client = get_llm_client(provider)
    response = await run_sync(
        llm_client.chat,
        messages=[{"role": "user", "content": user_message}],
        system_prompt=system_prompt,
    )

    # Calculate cost
    model = "gemini-2.5-flash" if provider == "gemini" else "claude-sonnet-4-20250514"
    cost = calculate_cost(model, response.input_tokens or 0, response.output_tokens or 0)

    return json.dumps({
        "response": response.content,
        "session_id": None,  # Multi-book sessions not persisted
        "book_ids": [b.id for b in books],
        "book_titles": [b.title for b in books],
        "provider": provider,
        "context_chunks": len(all_chunks),
        "citations": [
            {
                "book_id": book.id,
                "book_title": book.title,
                "chunk_id": chunk.id,
                "chapter_title": chunk.metadata.chapter_title,
                "chapter_number": chunk.metadata.chapter_number,
                "page_number": chunk.metadata.page_number,
            }
            for book, chunk in all_chunks
        ],
        "usage": {
            "input_tokens": response.input_tokens or 0,
            "output_tokens": response.output_tokens or 0,
            "cost_usd": round(cost, 4),
            "cost_formatted": format_cost(cost),
        },
        "note": "Multi-book sessions are not persisted. Each call starts fresh.",
    }, indent=2)


# =============================================================================
# Tool: find_book_in_drive
# =============================================================================


@mcp.tool()
async def find_book_in_drive(
    query: str = "",
    folder_id: Optional[str] = None,
) -> str:
    """Search Google Drive for books matching the query.

    Uses fuzzy matching to find books even if the filename differs from the title.
    Searches for PDF, EPUB, and Markdown files.

    Args:
        query: Book title or keywords to search for. If empty, lists all books.
        folder_id: Optional Drive folder ID to search in (uses configured default if omitted)

    Returns:
        JSON list of matching files with id, name, mime_type, match score,
        and whether the book is already ingested locally.

    Note: Requires Google Drive to be configured. Run 'bookrc setup-drive' first.
    """
    from rapidfuzz import fuzz
    from book_companion.google_drive import GoogleDriveClient
    from book_companion.google_drive.auth import is_authenticated

    # Check authentication
    if not is_authenticated():
        return json.dumps({
            "error": "Google Drive not configured",
            "help": "Run 'bookrc setup-drive' in the CLI to connect Google Drive.",
        })

    ctx = get_context()

    try:
        client = GoogleDriveClient()

        # Get files - either search or list all
        if query:
            results = await run_sync(client.search_books, query, folder_id)
        else:
            # List all books in folder
            files = await run_sync(client._list_book_files, folder_id)
            results = [(f, 100) for f in files]  # Score 100 for "list all" mode

        # Get ingested books for comparison
        ingested_books = await run_sync(ctx["registry"].list_books)

        matches = []
        for f, score in results:
            clean_name = client._clean_book_filename(f.name)

            # Check if already ingested
            best_match_id = None
            best_match_score = 0
            for book in ingested_books:
                match_score = fuzz.token_set_ratio(clean_name.lower(), book.title.lower())
                if match_score > best_match_score:
                    best_match_score = match_score
                    best_match_id = book.id

            is_ingested = best_match_score >= 80

            match_data = {
                "id": f.id,
                "name": f.name,
                "mime_type": f.mime_type,
                "size_bytes": f.size,
                "is_ingested": is_ingested,
            }

            if query:
                match_data["score"] = score

            if is_ingested:
                match_data["ingested_book_id"] = best_match_id

            matches.append(match_data)

        return json.dumps({
            "query": query or "(all books)",
            "matches": matches,
            "help": "Use ingest_book_from_drive(file_id) for books with is_ingested=false.",
        }, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


# =============================================================================
# Tool: load_book_from_drive
# =============================================================================


@mcp.tool()
async def load_book_from_drive(file_id: str) -> str:
    """Download a book from Google Drive and return its full text.

    Use this for quick questions about a book without full ingestion.
    The content is loaded into context but not persisted - it won't be
    available for future searches or chat sessions.

    For deeper exploration with RAG search and persistence, use
    ingest_book_from_drive instead.

    Args:
        file_id: Google Drive file ID (from find_book_in_drive results)

    Returns:
        JSON with title, author, chapter count, and full text content.
        Content is truncated at ~500K characters if longer.

    Note: Large books may not fit in context. Consider ingesting instead.
    """
    import tempfile
    from pathlib import Path

    from book_companion.google_drive import GoogleDriveClient
    from book_companion.google_drive.auth import is_authenticated
    from book_companion.parsers import get_parser

    # Check authentication
    if not is_authenticated():
        return json.dumps({
            "error": "Google Drive not configured",
            "help": "Run 'bookrc setup-drive' in the CLI to connect Google Drive.",
        })

    try:
        client = GoogleDriveClient()

        # Get file metadata
        metadata = await run_sync(client.get_file_metadata, file_id)

        # Download to temp file
        temp_dir = Path(tempfile.mkdtemp())
        temp_path = temp_dir / metadata.name
        await run_sync(client.download_file, file_id, temp_path)

        # Parse the book
        parser = get_parser(temp_path)
        if parser is None:
            return json.dumps({
                "error": f"Unsupported file format: {temp_path.suffix}",
                "file_name": metadata.name,
            })

        parsed = await run_sync(parser.parse, temp_path)

        # Clean up temp file
        try:
            temp_path.unlink()
            temp_dir.rmdir()
        except Exception:
            pass

        # Truncate content if too long
        max_chars = 500000
        content = parsed.raw_text
        truncated = len(content) > max_chars
        if truncated:
            content = content[:max_chars]

        return json.dumps({
            "title": parsed.title,
            "author": parsed.author,
            "format": parsed.format.value,
            "chapters": len(parsed.chapters),
            "total_chars": len(parsed.raw_text),
            "truncated": truncated,
            "content": content,
            "note": "This content is not persisted. For deeper exploration, use ingest_book_from_drive.",
        }, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


# =============================================================================
# Tool: ingest_book_from_drive
# =============================================================================


@mcp.tool()
async def ingest_book_from_drive(
    file_id: str,
    skip_summary: bool = False,
) -> str:
    """Download a book from Google Drive and fully ingest it.

    This performs full ingestion: parsing, chunking, embedding, and
    optionally summarization. The book will be available for RAG queries
    via search_books and chat tools.

    Args:
        file_id: Google Drive file ID (from find_book_in_drive results)
        skip_summary: If True, skip summarization (faster but no chapter index)

    Returns:
        JSON with book_id, title, chunk count, and summary stats.
        The book_id can be used with search_books, chat, and other tools.

    Note: Ingestion may take a few minutes for large books with summarization.
    """
    from pathlib import Path

    from book_companion.google_drive import GoogleDriveClient
    from book_companion.google_drive.auth import is_authenticated
    from book_companion.ingestion import ingest_book

    # Check authentication
    if not is_authenticated():
        return json.dumps({
            "error": "Google Drive not configured",
            "help": "Run 'bookrc setup-drive' in the CLI to connect Google Drive.",
        })

    try:
        client = GoogleDriveClient()

        # Get file metadata
        metadata = await run_sync(client.get_file_metadata, file_id)

        # Download to persistent location
        downloads_dir = Path.home() / ".bookrc" / "downloads"
        downloads_dir.mkdir(parents=True, exist_ok=True)
        dest_path = downloads_dir / metadata.name

        await run_sync(client.download_file, file_id, dest_path)

        # Run ingestion
        result = await run_sync(
            ingest_book,
            path=dest_path,
            skip_summary=skip_summary,
            drive_file_id=file_id,
        )

        if result is None:
            return json.dumps({
                "error": "Book already ingested",
                "file_name": metadata.name,
                "help": "Use list_books to find the existing book ID.",
            })

        return json.dumps({
            "success": True,
            "book_id": result["book_id"],
            "title": result["title"],
            "author": result["author"],
            "format": result["format"],
            "chapters": result["chapters"],
            "chunks": result["chunks"],
            "narratives": result["narratives"],
            "has_index": result["has_index"],
            "tokens_used": result["tokens_used"],
            "help": f"Use book_id '{result['book_id']}' with search_books or chat.",
        }, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


# =============================================================================
# Main entry point
# =============================================================================


def main():
    """Run the MCP server.

    Supports multiple transports:
    - stdio (default): For Claude Desktop local integration
    - sse: For remote access via HTTP/SSE (Claude web, tunnels, Cloud Run)
    - http: For remote access via Streamable HTTP

    All HTTP modes also include a REST API at /api/* for Obsidian plugin support.

    Usage:
        python -m book_companion.mcp.server          # stdio (default)
        python -m book_companion.mcp.server sse      # SSE on PORT or 8765
        python -m book_companion.mcp.server http     # HTTP on PORT or 8765

    Environment:
        PORT: Override default port (used by Cloud Run)
    """
    import os
    import sys

    transport = "stdio"
    if len(sys.argv) > 1:
        transport = sys.argv[1].lower()

    # Use PORT env var (Cloud Run) or default to 8765
    port = int(os.environ.get("PORT", "8765"))

    if transport == "sse":
        import uvicorn
        from starlette.middleware.cors import CORSMiddleware

        print(f"Starting MCP server with SSE transport on http://0.0.0.0:{port}/sse")
        app = mcp.sse_app()
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["app://obsidian.md"],
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["Content-Type"],
        )
        uvicorn.run(app, host="0.0.0.0", port=port)
    elif transport == "http":
        import uvicorn
        from starlette.middleware.cors import CORSMiddleware

        print(f"Starting MCP server with HTTP transport on http://0.0.0.0:{port}/mcp")
        app = mcp.streamable_http_app()
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["app://obsidian.md"],
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["Content-Type"],
        )
        uvicorn.run(app, host="0.0.0.0", port=port)
    else:
        # Default: stdio for Claude Desktop
        mcp.run()


if __name__ == "__main__":
    main()

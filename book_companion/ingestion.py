"""Reusable book ingestion logic.

This module provides the core ingestion functionality that can be used
by both the CLI and MCP server.
"""

import hashlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Any

from book_companion.models import Book, generate_id
from book_companion.parsers import get_parser
from book_companion.processing import Chunker, EmbeddingClient, Summarizer
from book_companion.storage import (
    get_vector_store,
    get_book_registry_store,
    get_book_index_store,
)

logger = logging.getLogger(__name__)


@dataclass
class IngestResult:
    """Result of book ingestion."""

    book_id: str
    title: str
    author: Optional[str]
    format: str
    chapter_count: int
    chunk_count: int
    narrative_count: int
    tokens_used: int
    has_index: bool
    drive_file_id: Optional[str] = None


def get_file_hash(file_path: Path) -> str:
    """Calculate MD5 hash of a file."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def ingest_book(
    path: Path,
    title: Optional[str] = None,
    author: Optional[str] = None,
    force: bool = False,
    model: str = "gemini-2.5-flash",
    max_workers: int = 2,
    skip_summary: bool = False,
    drive_file_id: Optional[str] = None,
    console: Optional[Any] = None,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> Optional[dict]:
    """
    Ingest a book file into the system.

    This is the core ingestion logic used by both CLI and MCP tools.

    Args:
        path: Path to the book file (PDF, EPUB, or Markdown)
        title: Override the book title
        author: Override the book author
        force: Re-ingest even if book already exists
        model: Model to use for summarization (default: claude-sonnet-4)
        max_workers: Number of parallel workers for chapter summarization (default: 4)
        skip_summary: Skip the summarization step
        drive_file_id: Optional Google Drive file ID for tracking
        console: Optional Rich console for CLI output
        progress_callback: Optional callback for progress updates
            Signature: (message: str, current: int, total: int) -> None

    Returns:
        Dict with ingestion results, or None if failed/skipped

    Raises:
        ValueError: If file format is unsupported
        FileNotFoundError: If file doesn't exist
    """
    path = Path(path).resolve()

    # Lazy import rich components for optional CLI output
    if console:
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
        from rich.panel import Panel

    # Check if parser exists
    parser = get_parser(path)
    if parser is None:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    # Check for duplicates
    registry = get_book_registry_store()
    file_hash = get_file_hash(path)

    existing = registry.find_by_hash(file_hash)
    if existing and not force:
        if console:
            console.print(f"[yellow]Book already ingested:[/yellow] {existing.display_name}")
            console.print(f"  ID: {existing.id}")
        return None

    # Create book ID
    book_id = existing.id if existing else generate_id()

    if console:
        console.print()
        console.print(f"[bold]Ingesting:[/bold] {path.name}")
        console.print()

    # Parse the book
    if progress_callback:
        progress_callback("Parsing book...", 0, 5)

    if console:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Parsing book...", total=None)
            parsed = parser.parse(path)

            if title:
                parsed.title = title
            if author:
                parsed.author = author

            progress.update(task, description=f"Parsed: {parsed.title} ({len(parsed.chapters)} chapters)")

            # Chunk the book
            progress.update(task, description="Chunking text...")
            chunker = Chunker()
            chunks = chunker.chunk_book(parsed, book_id)
            progress.update(task, description=f"Created {len(chunks)} chunks")

            # Generate embeddings
            progress.update(task, description="Generating embeddings...")
            embedding_client = EmbeddingClient()
            chunks = embedding_client.embed_chunks(chunks)
            progress.update(task, description="Embeddings generated")

            # Store in vector database
            progress.update(task, description="Storing in database...")
            vector_store = get_vector_store()

            # Delete existing if re-ingesting
            if existing:
                vector_store.delete_book(existing.id)
                registry.remove_book(existing.id)
                index_store = get_book_index_store()
                index_store.delete(existing.id)

            # Create preliminary book record BEFORE adding chunks (required for PostgreSQL foreign keys)
            preliminary_book = Book(
                id=book_id,
                title=parsed.title,
                author=parsed.author,
                format=parsed.format,
                file_path=str(path),
                file_hash=file_hash,
                total_chunks=len(chunks),
                total_pages=parsed.total_pages,
                summarization_input_tokens=0,
                summarization_output_tokens=0,
            )
            registry.add_book(preliminary_book)

            vector_store.add_chunks(book_id, chunks)
            progress.update(task, completed=True, description="Vectors stored")
    else:
        # Non-CLI mode (for MCP)
        logger.info("Parsing book: %s", path)
        parsed = parser.parse(path)

        if title:
            parsed.title = title
        if author:
            parsed.author = author

        logger.info("Parsed: %s (%d chapters)", parsed.title, len(parsed.chapters))
        if progress_callback:
            progress_callback(f"Parsed: {parsed.title}", 1, 5)

        # Chunk the book
        logger.info("Chunking text...")
        chunker = Chunker()
        chunks = chunker.chunk_book(parsed, book_id)
        logger.info("Created %d chunks", len(chunks))

        if progress_callback:
            progress_callback(f"Created {len(chunks)} chunks", 2, 5)

        # Generate embeddings
        logger.info("Generating embeddings...")
        embedding_client = EmbeddingClient()
        chunks = embedding_client.embed_chunks(chunks)
        logger.info("Embeddings generated for %d chunks", len(chunks))

        if progress_callback:
            progress_callback("Embeddings generated", 3, 5)

        # Store in vector database
        logger.info("Storing vectors in database...")
        vector_store = get_vector_store()

        # Delete existing if re-ingesting
        if existing:
            logger.info("Re-ingesting: deleting existing data for book %s", existing.id)
            vector_store.delete_book(existing.id)
            registry.remove_book(existing.id)
            index_store = get_book_index_store()
            index_store.delete(existing.id)

        # Create preliminary book record BEFORE adding chunks (required for PostgreSQL foreign keys)
        # This will be updated later with summarization token counts
        logger.info("Creating book record...")
        preliminary_book = Book(
            id=book_id,
            title=parsed.title,
            author=parsed.author,
            format=parsed.format,
            file_path=str(path),
            file_hash=file_hash,
            total_chunks=len(chunks),
            total_pages=parsed.total_pages,
            summarization_input_tokens=0,
            summarization_output_tokens=0,
        )
        registry.add_book(preliminary_book)
        logger.info("Book record created: %s", book_id)

        vector_store.add_chunks(book_id, chunks)
        logger.info("Vectors stored successfully")

        if progress_callback:
            progress_callback("Vectors stored", 4, 5)

    # Summarization step
    book_index = None
    if not skip_summary:
        if console:
            console.print()
            console.print(f"[bold]Generating summaries with {model}...[/bold]")

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
            ) as progress:
                total_steps = len(parsed.chapters) + 1
                task = progress.add_task("Summarizing...", total=total_steps)

                # ETA tracking
                start_time = time.time()
                chapter_times: list[float] = []
                last_step_time = start_time

                def cli_progress_callback(step: int, total: int, message: str):
                    nonlocal last_step_time
                    current_time = time.time()

                    if step > 0 and step <= len(parsed.chapters):
                        chapter_times.append(current_time - last_step_time)
                    last_step_time = current_time

                    eta_str = ""
                    if chapter_times:
                        avg_time = sum(chapter_times) / len(chapter_times)
                        remaining = total - step
                        eta_seconds = remaining * avg_time
                        if eta_seconds > 60:
                            eta_str = f" (~{int(eta_seconds/60)}m remaining)"
                        elif eta_seconds > 5:
                            eta_str = f" (~{int(eta_seconds)}s remaining)"

                    progress.update(task, completed=step, description=f"{message}{eta_str}")

                try:
                    summarizer = Summarizer(model=model, max_workers=max_workers)
                    book_index = summarizer.process_book(
                        parsed_book=parsed,
                        book_id=book_id,
                        progress_callback=cli_progress_callback,
                    )

                    index_store = get_book_index_store()
                    index_store.save(book_index)

                    elapsed = time.time() - start_time
                    elapsed_str = f"{int(elapsed/60)}m {int(elapsed%60)}s" if elapsed >= 60 else f"{int(elapsed)}s"
                    progress.update(task, completed=total_steps, description=f"Summaries complete ({elapsed_str})")
                except Exception as e:
                    console.print(f"[yellow]Warning:[/yellow] Summarization failed: {e}")
                    console.print("[dim]Book ingested without summaries. Chat will still work.[/dim]")
        else:
            # Non-CLI mode
            try:
                summarizer = Summarizer(model=model, max_workers=max_workers)

                def mcp_summarize_callback(step: int, total: int, message: str):
                    if progress_callback:
                        # Map to overall progress (step 5 is summarization)
                        progress_callback(message, 4, 5)

                book_index = summarizer.process_book(
                    parsed_book=parsed,
                    book_id=book_id,
                    progress_callback=mcp_summarize_callback,
                )

                index_store = get_book_index_store()
                index_store.save(book_index)
            except Exception:
                pass  # Silently continue without summaries in MCP mode

    if progress_callback:
        progress_callback("Complete", 5, 5)

    # Update book record with summarization tokens (book was created earlier before chunks)
    summarization_input = book_index.total_input_tokens if book_index else 0
    summarization_output = book_index.total_output_tokens if book_index else 0

    book = Book(
        id=book_id,
        title=parsed.title,
        author=parsed.author,
        format=parsed.format,
        file_path=str(path),
        file_hash=file_hash,
        total_chunks=len(chunks),
        total_pages=parsed.total_pages,
        summarization_input_tokens=summarization_input,
        summarization_output_tokens=summarization_output,
    )

    # Update the book record with summarization tokens (uses upsert)
    registry.add_book(book)

    # Build result
    result = IngestResult(
        book_id=book_id,
        title=parsed.title,
        author=parsed.author,
        format=parsed.format.value,
        chapter_count=len(parsed.chapters),
        chunk_count=len(chunks),
        narrative_count=len(book_index.all_narratives) if book_index else 0,
        tokens_used=summarization_input + summarization_output,
        has_index=book_index is not None,
        drive_file_id=drive_file_id,
    )

    # CLI summary output
    if console:
        from book_companion.utils.costs import calculate_cost, format_cost, format_tokens

        console.print()
        summary_lines = [
            f"[green]Successfully ingested:[/green] {book.display_name}",
            f"[dim]ID:[/dim] {book.id}",
            f"[dim]Chapters:[/dim] {len(parsed.chapters)}",
            f"[dim]Chunks:[/dim] {book.total_chunks}",
            f"[dim]Format:[/dim] {book.format.value.upper()}",
        ]

        if book_index:
            summary_lines.append(f"[dim]Stories/Studies:[/dim] {len(book_index.all_narratives)}")
            summary_lines.append(f"[dim]Summary Model:[/dim] {model}")
            if book_index.total_input_tokens > 0 or book_index.total_output_tokens > 0:
                total_tokens = book_index.total_input_tokens + book_index.total_output_tokens
                cost = calculate_cost(model, book_index.total_input_tokens, book_index.total_output_tokens)
                summary_lines.append(f"[dim]Tokens:[/dim] {format_tokens(total_tokens)} ({format_cost(cost)})")

        console.print(Panel(
            "\n".join(summary_lines),
            title="Book Ingested",
        ))

    return {
        "book_id": result.book_id,
        "title": result.title,
        "author": result.author,
        "format": result.format,
        "chapters": result.chapter_count,
        "chunks": result.chunk_count,
        "narratives": result.narrative_count,
        "tokens_used": result.tokens_used,
        "has_index": result.has_index,
    }

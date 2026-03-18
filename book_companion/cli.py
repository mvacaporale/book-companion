"""CLI commands for book-companion."""

import hashlib
import sys
import time
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.markdown import Markdown

from book_companion.models import Book, BookFormat
from book_companion.parsers import get_parser
from book_companion.processing import Chunker, EmbeddingClient, Summarizer
from book_companion.storage import VectorStore, BookRegistryStore, SessionStore, BookIndexStore
from book_companion.chat import ChatEngine
from book_companion.llm import get_llm_client, list_providers
from book_companion.utils.costs import calculate_cost, format_cost, format_tokens
from book_companion.google_drive import GoogleDriveClient, setup_drive_auth, get_credentials
from book_companion.google_drive.auth import is_authenticated, get_config, save_config

console = Console()


def get_file_hash(file_path: Path) -> str:
    """Calculate MD5 hash of a file."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


@click.group()
@click.version_option(package_name="book-companion")
def cli():
    """Book Reading Companion - Chat with your books using RAG and LLMs."""
    pass


@cli.command()
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option("--title", "-t", help="Override the book title")
@click.option("--author", "-a", help="Override the book author")
@click.option("--force", "-f", is_flag=True, help="Re-ingest even if already exists")
@click.option("--model", "-m", default="gemini-3-flash",
              help="Model for summarization (claude-sonnet-4, gemini-2.5-flash)")
@click.option("--workers", "-j", default=2, help="Parallel workers for chapter summarization (default: 2)")
@click.option("--skip-summary", is_flag=True, help="Skip summarization step (faster, but no book index)")
def ingest(path: Path, title: Optional[str], author: Optional[str], force: bool,
           model: str, workers: int, skip_summary: bool):
    """Ingest a book (PDF, EPUB, or Markdown) for chatting.

    This command parses the book, chunks the text, generates embeddings,
    and creates a hierarchical summary with key stories and studies.
    """
    path = path.resolve()

    # Check if parser exists
    parser = get_parser(path)
    if parser is None:
        console.print(f"[red]Error:[/red] Unsupported file format: {path.suffix}")
        console.print("Supported formats: PDF, EPUB, Markdown (.md, .txt)")
        sys.exit(1)

    # Check for duplicates
    registry = BookRegistryStore()
    file_hash = get_file_hash(path)

    existing = registry.find_by_hash(file_hash)
    if existing and not force:
        console.print(f"[yellow]Book already ingested:[/yellow] {existing.display_name}")
        console.print(f"  ID: {existing.id}")
        console.print(f"  Use [bold]--force[/bold] to re-ingest")
        return

    # Create book ID
    temp_id = existing.id if existing else None
    if temp_id is None:
        from book_companion.models import generate_id
        temp_id = generate_id()

    console.print()
    console.print(f"[bold]Ingesting:[/bold] {path.name}")
    console.print()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        # Parse the book
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
        chunks = chunker.chunk_book(parsed, temp_id)
        progress.update(task, description=f"Created {len(chunks)} chunks")

        # Generate embeddings
        progress.update(task, description="Generating embeddings...")
        embedding_client = EmbeddingClient()
        chunks = embedding_client.embed_chunks(chunks)
        progress.update(task, description="Embeddings generated")

        # Store in vector database
        progress.update(task, description="Storing in database...")
        vector_store = VectorStore()

        # Delete existing if re-ingesting
        if existing:
            vector_store.delete_book(existing.id)
            registry.remove_book(existing.id)
            # Also delete existing index
            index_store = BookIndexStore()
            index_store.delete(existing.id)

        vector_store.add_chunks(temp_id, chunks)
        progress.update(task, completed=True, description="Vectors stored")

    # Summarization step (separate progress for better visibility)
    book_index = None
    if not skip_summary:
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

            def progress_callback(step: int, total: int, message: str):
                nonlocal last_step_time
                current_time = time.time()

                # Track per-chapter time (for chapters, not book summary)
                if step > 0 and step <= len(parsed.chapters):
                    chapter_times.append(current_time - last_step_time)
                last_step_time = current_time

                # Calculate ETA
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
                summarizer = Summarizer(model=model, max_workers=workers)
                book_index = summarizer.process_book(
                    parsed_book=parsed,
                    book_id=temp_id,
                    progress_callback=progress_callback,
                )

                # Save the index
                index_store = BookIndexStore()
                index_store.save(book_index)

                # Calculate elapsed time
                elapsed = time.time() - start_time
                elapsed_str = f"{int(elapsed/60)}m {int(elapsed%60)}s" if elapsed >= 60 else f"{int(elapsed)}s"
                progress.update(task, completed=total_steps, description=f"Summaries complete ({elapsed_str})")
            except Exception as e:
                console.print(f"[yellow]Warning:[/yellow] Summarization failed: {e}")
                console.print("[dim]Book ingested without summaries. Chat will still work.[/dim]")

    # Create book record with token tracking
    summarization_input = book_index.total_input_tokens if book_index else 0
    summarization_output = book_index.total_output_tokens if book_index else 0

    book = Book(
        id=temp_id,
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

    registry.add_book(book)

    # Summary panel
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
    console.print()
    console.print(f"Start chatting: [bold]bookrc chat {book.id}[/bold]")


@cli.command()
@click.argument("book_id")
@click.option("--provider", "-p", type=click.Choice(["gemini", "claude"]), default="gemini",
              help="LLM provider to use")
@click.option("--resume", "-r", "session_id", help="Resume a previous session")
def chat(book_id: str, provider: str, session_id: Optional[str]):
    """Start a chat session with a book."""
    # Load book
    registry = BookRegistryStore()
    book = registry.get_book(book_id)

    if book is None:
        console.print(f"[red]Error:[/red] Book not found: {book_id}")
        console.print("Run [bold]bookrc list books[/bold] to see available books.")
        sys.exit(1)

    # Load book index if available
    index_store = BookIndexStore()
    book_index = index_store.load(book_id)

    # Initialize chat engine
    try:
        engine = ChatEngine(book=book, provider=provider, book_index=book_index)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)

    # Load existing session if requested
    if session_id:
        if not engine.load_session(session_id):
            console.print(f"[red]Error:[/red] Session not found: {session_id}")
            sys.exit(1)
        console.print(f"[dim]Resumed session: {session_id}[/dim]")

    # Print welcome
    console.print()
    index_status = "with index" if book_index else "no index"
    console.print(Panel(
        f"[bold]{book.display_name}[/bold]\n"
        f"[dim]Provider: {provider} | Session: {engine.session.id} | {index_status}[/dim]",
        title="Chat Session",
    ))
    console.print("[dim]Type 'quit' or 'exit' to end the session.[/dim]")
    console.print()

    # Chat loop
    while True:
        try:
            user_input = console.input("[bold blue]You:[/bold blue] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            break

        # Show thinking indicator
        console.print()
        with console.status("[dim]Thinking...[/dim]"):
            try:
                response, context = engine.chat(user_input)
            except Exception as e:
                console.print(f"[red]Error:[/red] {e}")
                continue

        # Display response
        console.print("[bold green]Assistant:[/bold green]")
        console.print(Markdown(response))
        console.print()

    # Session summary
    summary = engine.get_session_summary()
    console.print()
    console.print(f"[dim]Session saved: {summary['session_id']} ({summary['message_count']} messages)[/dim]")


@cli.group()
def list():
    """List books or sessions."""
    pass


@list.command("books")
def list_books():
    """List all ingested books."""
    registry = BookRegistryStore()
    books = registry.list_books()

    if not books:
        console.print("[dim]No books ingested yet.[/dim]")
        console.print("Run [bold]bookrc ingest <path>[/bold] to add a book.")
        return

    index_store = BookIndexStore()

    table = Table(title="Ingested Books")
    table.add_column("ID", style="cyan")
    table.add_column("Title")
    table.add_column("Author")
    table.add_column("Format")
    table.add_column("Chunks", justify="right")
    table.add_column("Index", justify="center")
    table.add_column("Ingested")

    for book in books:
        has_index = index_store.exists(book.id)
        table.add_row(
            book.id,
            book.title[:40] + "..." if len(book.title) > 40 else book.title,
            book.author or "-",
            book.format.value.upper(),
            str(book.total_chunks),
            "[green]Yes[/green]" if has_index else "[dim]No[/dim]",
            book.ingested_at.strftime("%Y-%m-%d"),
        )

    console.print(table)


@list.command("sessions")
@click.argument("book_id")
def list_sessions(book_id: str):
    """List chat sessions for a book."""
    registry = BookRegistryStore()
    book = registry.get_book(book_id)

    if book is None:
        console.print(f"[red]Error:[/red] Book not found: {book_id}")
        sys.exit(1)

    store = SessionStore()
    sessions = store.list_sessions(book_id)

    if not sessions:
        console.print(f"[dim]No sessions for book: {book.title}[/dim]")
        return

    table = Table(title=f"Sessions for: {book.title}")
    table.add_column("Session ID", style="cyan")
    table.add_column("Provider")
    table.add_column("Messages", justify="right")
    table.add_column("Last Updated")

    for session in sessions:
        table.add_row(
            session.id,
            session.provider,
            str(len(session.messages)),
            session.updated_at.strftime("%Y-%m-%d %H:%M"),
        )

    console.print(table)
    console.print()
    console.print(f"Resume with: [bold]bookrc chat {book_id} --resume <session_id>[/bold]")


@cli.command()
@click.argument("book_id")
def info(book_id: str):
    """Show detailed information about a book."""
    registry = BookRegistryStore()
    book = registry.get_book(book_id)

    if book is None:
        console.print(f"[red]Error:[/red] Book not found: {book_id}")
        sys.exit(1)

    # Get session count
    session_store = SessionStore()
    sessions = session_store.list_sessions(book_id)

    # Get index info
    index_store = BookIndexStore()
    book_index = index_store.load(book_id)

    info_lines = [
        f"[bold]Title:[/bold] {book.title}",
        f"[bold]Author:[/bold] {book.author or 'Unknown'}",
        f"[bold]Format:[/bold] {book.format.value.upper()}",
        f"[bold]Pages:[/bold] {book.total_pages or 'N/A'}",
        f"[bold]Chunks:[/bold] {book.total_chunks}",
        f"[bold]Sessions:[/bold] {len(sessions)}",
        f"[bold]File:[/bold] {book.file_path}",
        f"[bold]Ingested:[/bold] {book.ingested_at.strftime('%Y-%m-%d %H:%M')}",
    ]

    if book_index:
        info_lines.append("")
        info_lines.append("[bold]Index Information:[/bold]")
        info_lines.append(f"  Chapters summarized: {len(book_index.chapter_summaries)}")
        info_lines.append(f"  Stories/studies extracted: {len(book_index.all_narratives)}")
        info_lines.append(f"  Key themes: {', '.join(book_index.book_summary.key_themes[:5])}")
        info_lines.append(f"  Model used: {book_index.model_used}")

        # Token usage for summarization
        if book_index.total_input_tokens > 0 or book_index.total_output_tokens > 0:
            info_lines.append("")
            info_lines.append("[bold]Summarization Usage:[/bold]")
            info_lines.append(f"  Input tokens: {format_tokens(book_index.total_input_tokens)}")
            info_lines.append(f"  Output tokens: {format_tokens(book_index.total_output_tokens)}")
            cost = calculate_cost(
                book_index.model_used,
                book_index.total_input_tokens,
                book_index.total_output_tokens,
            )
            info_lines.append(f"  Est. cost: {format_cost(cost)}")
    else:
        info_lines.append("")
        info_lines.append("[dim]No index available. Re-ingest with --force to generate summaries.[/dim]")

    console.print(Panel(
        "\n".join(info_lines),
        title=f"Book: {book.id}",
    ))

    # Show narratives if available
    if book_index and book_index.all_narratives:
        console.print()
        console.print("[bold]Key Stories & Studies:[/bold]")
        for n in book_index.all_narratives[:10]:
            type_label = n.type.value.replace("_", " ").title()
            console.print(f"  [{type_label}] {n.title} (Ch. {n.chapter_number})")
        if len(book_index.all_narratives) > 10:
            console.print(f"  [dim]... and {len(book_index.all_narratives) - 10} more[/dim]")


@cli.command()
@click.argument("book_id")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation")
def delete(book_id: str, yes: bool):
    """Delete a book and all its data."""
    registry = BookRegistryStore()
    book = registry.get_book(book_id)

    if book is None:
        console.print(f"[red]Error:[/red] Book not found: {book_id}")
        sys.exit(1)

    index_store = BookIndexStore()
    has_index = index_store.exists(book_id)

    if not yes:
        console.print(f"[yellow]Warning:[/yellow] This will delete:")
        console.print(f"  - Book: {book.display_name}")
        console.print(f"  - {book.total_chunks} chunks from vector database")
        if has_index:
            console.print(f"  - Book index (summaries, narratives)")
        console.print(f"  - All chat sessions")
        console.print()

        if not click.confirm("Are you sure?"):
            console.print("[dim]Cancelled.[/dim]")
            return

    # Delete vector data
    vector_store = VectorStore()
    vector_store.delete_book(book_id)

    # Delete index
    if has_index:
        index_store.delete(book_id)

    # Delete sessions
    session_store = SessionStore()
    deleted_sessions = session_store.delete_book_sessions(book_id)

    # Delete book record
    registry.remove_book(book_id)

    console.print(f"[green]Deleted:[/green] {book.display_name}")
    console.print(f"  - {book.total_chunks} chunks")
    if has_index:
        console.print(f"  - Book index")
    console.print(f"  - {deleted_sessions} sessions")


@cli.command()
@click.argument("book_id")
def index(book_id: str):
    """Show the book's navigation index (for debugging)."""
    registry = BookRegistryStore()
    book = registry.get_book(book_id)

    if book is None:
        console.print(f"[red]Error:[/red] Book not found: {book_id}")
        sys.exit(1)

    index_store = BookIndexStore()
    book_index = index_store.load(book_id)

    if not book_index:
        console.print(f"[dim]No index available for: {book.title}[/dim]")
        console.print("Re-ingest with --force to generate summaries.")
        return

    # Display the navigation prompt that would be sent to the LLM
    console.print(Panel(
        book_index.get_navigation_prompt(),
        title="Book Navigation Index",
    ))


@cli.command()
@click.argument("book_id", required=False)
def stats(book_id: Optional[str]):
    """Show token usage and cost statistics.

    If BOOK_ID is provided, shows stats for that book only.
    Otherwise, shows aggregate stats across all books.
    """
    registry = BookRegistryStore()
    index_store = BookIndexStore()
    session_store = SessionStore()

    if book_id:
        # Stats for a specific book
        book = registry.get_book(book_id)
        if book is None:
            console.print(f"[red]Error:[/red] Book not found: {book_id}")
            sys.exit(1)

        book_index = index_store.load(book_id)
        sessions = session_store.list_sessions(book_id)

        # Ingestion stats
        info_lines = [
            f"[bold]{book.display_name}[/bold]",
            "",
            "[bold]Ingestion Usage:[/bold]",
        ]

        if book_index and (book_index.total_input_tokens > 0 or book_index.total_output_tokens > 0):
            ingest_cost = calculate_cost(
                book_index.model_used,
                book_index.total_input_tokens,
                book_index.total_output_tokens,
            )
            info_lines.extend([
                f"  Model: {book_index.model_used}",
                f"  Input tokens: {format_tokens(book_index.total_input_tokens)}",
                f"  Output tokens: {format_tokens(book_index.total_output_tokens)}",
                f"  Cost: {format_cost(ingest_cost)}",
            ])
        elif book.summarization_input_tokens > 0 or book.summarization_output_tokens > 0:
            model = book_index.model_used if book_index else "gemini-2.5-flash"
            ingest_cost = calculate_cost(
                model,
                book.summarization_input_tokens,
                book.summarization_output_tokens,
            )
            info_lines.extend([
                f"  Input tokens: {format_tokens(book.summarization_input_tokens)}",
                f"  Output tokens: {format_tokens(book.summarization_output_tokens)}",
                f"  Cost: {format_cost(ingest_cost)}",
            ])
        else:
            info_lines.append("  [dim]No token data available (book may have been ingested before tracking)[/dim]")

        # Chat stats
        info_lines.append("")
        info_lines.append("[bold]Chat Usage:[/bold]")

        total_chat_input = 0
        total_chat_output = 0
        session_stats = []

        for session in sessions:
            usage = session.get_total_usage()
            if usage.total_tokens > 0:
                session_stats.append({
                    "id": session.id,
                    "provider": session.provider,
                    "messages": len(session.messages),
                    "input": usage.input_tokens,
                    "output": usage.output_tokens,
                })
                total_chat_input += usage.input_tokens
                total_chat_output += usage.output_tokens

        if session_stats:
            info_lines.append(f"  Sessions with usage data: {len(session_stats)}")
            info_lines.append(f"  Total input tokens: {format_tokens(total_chat_input)}")
            info_lines.append(f"  Total output tokens: {format_tokens(total_chat_output)}")

            # Calculate chat cost (estimate with most recent provider)
            if sessions:
                provider = sessions[0].provider
                model = "gemini-2.5-flash" if provider == "gemini" else "claude-sonnet-4-20250514"
                chat_cost = calculate_cost(model, total_chat_input, total_chat_output)
                info_lines.append(f"  Est. cost ({provider}): {format_cost(chat_cost)}")
        else:
            info_lines.append("  [dim]No chat token data available[/dim]")

        console.print(Panel("\n".join(info_lines), title="Usage Statistics"))

    else:
        # Aggregate stats across all books
        books = registry.list_books()

        if not books:
            console.print("[dim]No books ingested yet.[/dim]")
            return

        total_ingest_input = 0
        total_ingest_output = 0
        total_chat_input = 0
        total_chat_output = 0
        books_with_data = 0

        table = Table(title="Token Usage by Book")
        table.add_column("Book", style="cyan", max_width=30)
        table.add_column("Ingest Tokens", justify="right")
        table.add_column("Chat Tokens", justify="right")
        table.add_column("Est. Cost", justify="right")

        for book in books:
            book_index = index_store.load(book.id)
            sessions = session_store.list_sessions(book.id)

            # Ingestion tokens
            ingest_input = 0
            ingest_output = 0
            ingest_model = "gemini-2.5-flash"

            if book_index:
                ingest_input = book_index.total_input_tokens
                ingest_output = book_index.total_output_tokens
                ingest_model = book_index.model_used
            elif book.summarization_input_tokens > 0:
                ingest_input = book.summarization_input_tokens
                ingest_output = book.summarization_output_tokens

            # Chat tokens
            chat_input = 0
            chat_output = 0
            chat_provider = "gemini"

            for session in sessions:
                usage = session.get_total_usage()
                chat_input += usage.input_tokens
                chat_output += usage.output_tokens
                chat_provider = session.provider

            # Calculate costs
            ingest_cost = calculate_cost(ingest_model, ingest_input, ingest_output)
            chat_model = "gemini-2.5-flash" if chat_provider == "gemini" else "claude-sonnet-4-20250514"
            chat_cost = calculate_cost(chat_model, chat_input, chat_output)
            total_cost = ingest_cost + chat_cost

            if ingest_input > 0 or chat_input > 0:
                books_with_data += 1
                total_ingest_input += ingest_input
                total_ingest_output += ingest_output
                total_chat_input += chat_input
                total_chat_output += chat_output

            ingest_str = format_tokens(ingest_input + ingest_output) if ingest_input > 0 else "-"
            chat_str = format_tokens(chat_input + chat_output) if chat_input > 0 else "-"
            cost_str = format_cost(total_cost) if total_cost > 0 else "-"

            table.add_row(
                book.title[:28] + ".." if len(book.title) > 30 else book.title,
                ingest_str,
                chat_str,
                cost_str,
            )

        console.print(table)

        # Show totals
        if books_with_data > 0:
            total_ingest = total_ingest_input + total_ingest_output
            total_chat = total_chat_input + total_chat_output
            total_all = total_ingest + total_chat

            # Estimate total cost (using gemini-2.5-flash as default)
            ingest_cost = calculate_cost("gemini-2.5-flash", total_ingest_input, total_ingest_output)
            chat_cost = calculate_cost("gemini-2.5-flash", total_chat_input, total_chat_output)

            console.print()
            console.print(f"[bold]Totals ({books_with_data} books with data):[/bold]")
            console.print(f"  Ingestion: {format_tokens(total_ingest)} ({format_cost(ingest_cost)})")
            console.print(f"  Chat: {format_tokens(total_chat)} ({format_cost(chat_cost)})")
            console.print(f"  All: {format_tokens(total_all)} ({format_cost(ingest_cost + chat_cost)})")
        else:
            console.print()
            console.print("[dim]No token usage data available. Token tracking was added recently.[/dim]")


@cli.command("setup-drive")
def setup_drive():
    """Set up Google Drive authentication.

    This command initiates the OAuth flow to connect your Google Drive.
    You only need to run this once - credentials are saved for future use.

    Prerequisites:
    1. Create OAuth credentials at https://console.cloud.google.com/apis/credentials
    2. Save the JSON file to ~/.bookrc/google_credentials.json
    """
    console.print()
    console.print("[bold]Google Drive Setup[/bold]")
    console.print()

    # Check if already authenticated
    if is_authenticated():
        console.print("[green]Already authenticated with Google Drive.[/green]")
        config = get_config()
        if config.get("default_folder_id"):
            console.print(f"[dim]Default folder: {config['default_folder_id']}[/dim]")
        console.print()
        console.print("[dim]To re-authenticate, delete ~/.bookrc/google_token.json[/dim]")
        return

    try:
        console.print("Opening browser for Google authentication...")
        console.print()
        success = setup_drive_auth()

        if success:
            console.print()
            console.print("[green]Successfully connected to Google Drive![/green]")
            console.print()

            # Prompt for folder configuration
            folder_id = click.prompt(
                "Enter a folder ID to search for books (or press Enter to search all of Drive)",
                default="",
                show_default=False,
            )

            if folder_id:
                config = get_config()
                config["default_folder_id"] = folder_id
                save_config(config)
                console.print(f"[dim]Saved default folder: {folder_id}[/dim]")

            console.print()
            console.print("Test with: [bold]bookrc drive search 'Atomic Habits'[/bold]")
        else:
            console.print("[red]Authentication failed.[/red]")
            sys.exit(1)

    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


@cli.group()
def drive():
    """Google Drive book operations."""
    pass


@drive.command("search")
@click.argument("query")
@click.option("--folder", "-f", "folder_id", help="Folder ID to search in")
@click.option("--threshold", "-t", default=60, help="Minimum match score (0-100)")
def drive_search(query: str, folder_id: Optional[str], threshold: int):
    """Search for books in Google Drive.

    Uses fuzzy matching to find books even if filenames differ from titles.
    """
    if not is_authenticated():
        console.print("[red]Error:[/red] Not authenticated with Google Drive.")
        console.print("Run [bold]bookrc setup-drive[/bold] first.")
        sys.exit(1)

    console.print(f"[dim]Searching for: {query}[/dim]")
    console.print()

    try:
        client = GoogleDriveClient()
        results = client.search_books(query, folder_id=folder_id, threshold=threshold)

        if not results:
            console.print("[dim]No matching books found.[/dim]")
            return

        table = Table(title="Matching Books")
        table.add_column("Score", style="cyan", justify="right")
        table.add_column("Name")
        table.add_column("Type")
        table.add_column("ID", style="dim")

        for file, score in results:
            table.add_row(
                f"{score}%",
                file.name[:50] + "..." if len(file.name) > 50 else file.name,
                file.mime_type.split("/")[-1].upper(),
                file.id[:20] + "...",
            )

        console.print(table)
        console.print()
        console.print("[dim]Use the file ID to ingest: bookrc drive ingest <file_id>[/dim]")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


@drive.command("list")
@click.option("--folder", "-f", "folder_id", help="Folder ID to list")
def drive_list(folder_id: Optional[str]):
    """List all book files in Google Drive."""
    if not is_authenticated():
        console.print("[red]Error:[/red] Not authenticated with Google Drive.")
        console.print("Run [bold]bookrc setup-drive[/bold] first.")
        sys.exit(1)

    try:
        client = GoogleDriveClient()
        # Use internal method to list all files
        files = client._list_book_files(folder_id)

        if not files:
            console.print("[dim]No book files found.[/dim]")
            return

        table = Table(title=f"Books in Drive ({len(files)} files)")
        table.add_column("Name")
        table.add_column("Type", justify="center")
        table.add_column("Size", justify="right")
        table.add_column("Modified")
        table.add_column("ID", style="dim")

        for file in files[:50]:  # Limit display
            size_str = "-"
            if file.size:
                if file.size > 1024 * 1024:
                    size_str = f"{file.size / (1024 * 1024):.1f} MB"
                else:
                    size_str = f"{file.size / 1024:.1f} KB"

            modified = file.modified_time[:10] if file.modified_time else "-"

            table.add_row(
                file.name[:40] + "..." if len(file.name) > 40 else file.name,
                file.mime_type.split("/")[-1].upper()[:6],
                size_str,
                modified,
                file.id[:15] + "...",
            )

        console.print(table)

        if len(files) > 50:
            console.print(f"[dim]... and {len(files) - 50} more files[/dim]")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


@drive.command("ingest")
@click.argument("file_id")
@click.option("--title", "-t", help="Override the book title")
@click.option("--author", "-a", help="Override the book author")
@click.option("--model", "-m", default="gemini-3-flash",
              help="Model for summarization (claude-sonnet-4, gemini-2.5-flash)")
@click.option("--workers", "-j", default=2, help="Parallel workers for chapter summarization (default: 2)")
@click.option("--skip-summary", is_flag=True, help="Skip summarization step")
def drive_ingest(file_id: str, title: Optional[str], author: Optional[str],
                 model: str, workers: int, skip_summary: bool):
    """Download and ingest a book from Google Drive.

    FILE_ID is the Google Drive file ID (shown in search/list results).
    """
    if not is_authenticated():
        console.print("[red]Error:[/red] Not authenticated with Google Drive.")
        console.print("Run [bold]bookrc setup-drive[/bold] first.")
        sys.exit(1)

    try:
        client = GoogleDriveClient()

        # Get file metadata
        console.print("[dim]Fetching file info...[/dim]")
        metadata = client.get_file_metadata(file_id)
        console.print(f"[bold]File:[/bold] {metadata.name}")

        # Download to persistent location
        from book_companion.security import sanitize_filename

        downloads_dir = Path.home() / ".bookrc" / "downloads"
        safe_filename = sanitize_filename(metadata.name)
        dest_path = downloads_dir / safe_filename

        console.print(f"[dim]Downloading to {dest_path}...[/dim]")
        client.download_file(file_id, dest_path)
        console.print("[green]Download complete.[/green]")
        console.print()

        # Run the normal ingest process using the downloaded file
        # Import the ingest_book function we'll create
        from book_companion.ingestion import ingest_book

        result = ingest_book(
            path=dest_path,
            title=title,
            author=author,
            model=model,
            max_workers=workers,
            skip_summary=skip_summary,
            drive_file_id=file_id,
            console=console,
        )

        if result:
            console.print()
            console.print(f"Start chatting: [bold]bookrc chat {result['book_id']}[/bold]")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


if __name__ == "__main__":
    cli()

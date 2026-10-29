# Book Reading Companion

A CLI application to chat with books (PDF, EPUB, Markdown) using RAG and LLMs, with intelligent summarization that preserves key stories and studies.

## Features

- **Multi-format support**: Parse PDF, EPUB, and Markdown files
- **Semantic search**: Find relevant passages using vector embeddings
- **Hierarchical summarization**: Auto-generate chapter summaries, book overview, and extract key narratives
- **Story & study extraction**: Preserve the author's examples, research studies, and case studies
- **Multiple LLM providers**: Chat using Gemini or Claude
- **Configurable models**: Use any Gemini model for summarization (gemini-2.5-flash, gemini-3-flash, etc.)
- **Session management**: Save and resume chat sessions
- **Chapter-aware chunking**: Intelligent text splitting that respects document structure
- **Token usage tracking**: Track tokens and costs for all LLM operations (ingestion and chat)
- **Progress transparency**: Real-time progress with chapter titles and ETA during ingestion

## Installation

```bash
# Clone the repository
cd book-companion

# Install with uv
uv sync

# Verify installation
uv run bookrc --help
```

## Setup

### API Keys

Set up your API keys in `~/.zshrc`:

```bash
# Required for embeddings, summarization, and Gemini chat
export GEMINI_API_KEY="your-gemini-api-key"

# Optional: for Claude chat
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

Reload your shell or run `source ~/.zshrc`.

### Google Drive (optional)

Drive access reuses the Google Workspace CLI (`gws`) OAuth token: export it
once with `gws auth export --unmasked > ~/.config/gws/credentials.json`, and set
the books folder ID (the part of the folder URL after `/folders/`) via
`GOOGLE_DRIVE_FOLDER_ID` or `default_folder_id` in `~/.bookrc/config.json`.
`uv run bookrc setup-drive` is a fallback personal OAuth flow.

### Automatic ingestion from Drive

Drop a book in the Drive books folder and it's ingested within about 5–10
minutes, with nothing running on your own machines:

1. `drive-watcher/Code.gs`, a Google Apps Script, checks the folder every 5
   minutes. When its contents change, it starts the `drive-sync.yml` workflow.
2. The workflow runs `bookrc drive sync` from `main`. It skips files that are
   already in the library (the Drive checksum matches a stored file hash, or
   the filename matches an existing title) and ingests up to 5 new books into
   Neon, about $0.03 each. A daily run at 10:00 UTC is a backstop in case the
   watcher missed a change.
   Books added to Drive before 2026-10-29 00:30 UTC are skipped on purpose
   (`DRIVE_SYNC_SINCE` in the workflow), so the older backlog stays out. A file
   *moved* into the folder keeps its original Drive creation date, so an old
   file moved in is skipped too; ingest those with `bookrc drive ingest <id>`.

You can also start a run by hand from the Actions tab (inputs: `dry_run`,
`max_books`). GitHub emails you if a run fails; Apps Script emails you if the
watcher fails.

To rebuild books that are already stored after a parser fix, run the **Re-ingest
books** workflow (Actions tab; dry run by default) or `bookrc drive reingest`.
It also runs by itself: a dry run on any PR that changes `reingest.yml`, and a
real run of every book when that change merges.
It re-ingests each book from its Drive file and keeps book IDs and chat sessions.

One-time setup:

1. Add three repo secrets (Settings → Secrets and variables → Actions). Until
   they exist, the workflow skips with a warning.

   | Secret | Value |
   |--------|-------|
   | `GEMINI_API_KEY` | Same key as `~/.bookrc/mcp.env` (Proton Pass `GEMINI_API_KEY`) |
   | `DATABASE_URL` | Neon connection string (Proton Pass `BOOKRC_DATABASE_URL`) |
   | `GWS_CREDENTIALS_JSON` | Contents of `~/.config/gws/credentials.json` |

2. Set up the watcher: follow [drive-watcher/README.md](drive-watcher/README.md).

## Usage

### Ingest a Book

```bash
# Ingest a PDF (includes summarization)
uv run bookrc ingest ~/Books/my-book.pdf

# Use a different model for summarization
uv run bookrc ingest ~/Books/book.pdf --model gemini-3-flash

# Skip summarization for faster ingestion
uv run bookrc ingest ~/Books/book.pdf --skip-summary

# Ingest with custom title/author
uv run bookrc ingest ~/Books/book.epub --title "Custom Title" --author "Author Name"

# Re-ingest an existing book
uv run bookrc ingest ~/Books/book.pdf --force
```

### Chat with a Book

```bash
# Start a new chat session (uses Gemini by default)
uv run bookrc chat abc123

# Use Claude instead
uv run bookrc chat abc123 --provider claude

# Resume a previous session
uv run bookrc chat abc123 --resume session_id
```

### Manage Books and Sessions

```bash
# List all ingested books (shows index status)
uv run bookrc list books

# List sessions for a book
uv run bookrc list sessions abc123

# Show book details and extracted stories/studies
uv run bookrc info abc123

# View the navigation index (debugging)
uv run bookrc index abc123

# Fix a wrong title or author without re-ingesting
uv run bookrc edit abc123 --title "Affluence Without Abundance" --author "James Suzman"

# Delete a book and all its data
uv run bookrc delete abc123
```

### Token Usage & Cost Tracking

```bash
# Show token usage for all books
uv run bookrc stats

# Show token usage for a specific book
uv run bookrc stats abc123
```

The `stats` command shows:
- Ingestion token usage (summarization)
- Chat token usage per session
- Estimated costs based on model pricing

## How It Works

### Ingestion Pipeline

1. **Parsing**: Books are parsed to extract text and chapter structure, in reading order.
   Chapters keep the book's own numbers ("CHAPTER 12" is chapter 12); unnumbered
   sections like the introduction or epilogue are numbered after the last chapter.
   Front and back matter (contents, notes, index, bibliography, acknowledgments) is
   skipped, and placeholder metadata (`[No data]`, ID-like authors) falls back to the
   filename for the title and to no author.
2. **Chunking**: Text is split into ~1500 character chunks with semantic awareness
3. **Embedding**: Chunks are embedded using Gemini's `text-embedding-004` model
4. **Summarization**: Each chapter is summarized, extracting:
   - Key concepts and arguments
   - Stories and anecdotes
   - Research studies
   - Case studies and examples
5. **Book Summary**: Overall thesis, themes, and target audience
6. **Storage**: Everything stored in local ChromaDB + JSON files

### Chat Pipeline

1. **Query Embedding**: Your question is embedded
2. **Retrieval**: Top 8 relevant chunks are retrieved via cosine similarity. Index pages,
   endnotes and bare headings are dropped at query time, so books ingested before that
   rule existed are filtered too. The MCP `search_books` limit (`n_results`) is the total
   across all books, best matches first.
3. **Context Building**: Chunks + book navigation index form the context
4. **Generation**: LLM generates a response grounded in the book content

The **navigation index** gives the LLM a "map" of the book—it can see all chapters, their core arguments, and notable stories/studies before answering.

## Data Storage

All data is stored in `~/.bookrc/`:

```
~/.bookrc/
├── db/              # ChromaDB vector database
├── books.json       # Book metadata registry
├── indices/         # Book summaries and navigation indices
│   └── {book_id}.json
└── sessions/        # Chat session history
    └── {book_id}/
        └── {session_id}.json
```

You can customize the location with the `BOOKRC_DB_PATH` environment variable.

## Supported Formats

| Format | Extensions | Parser |
|--------|------------|--------|
| PDF | `.pdf` | PyMuPDF |
| EPUB | `.epub` | ebooklib |
| Markdown | `.md`, `.markdown`, `.txt` | Built-in |

## LLM Providers

| Provider | Model | Environment Variable |
|----------|-------|---------------------|
| Gemini (default) | gemini-2.5-flash | `GEMINI_API_KEY` |
| Claude | claude-sonnet-4 | `ANTHROPIC_API_KEY` |

## Cost Estimates

| Operation | Approximate Cost (Gemini 2.5 Flash) |
|-----------|-------------------------------------|
| Ingest book (with summaries) | ~$0.02-0.03 |
| Chat query | ~$0.001-0.01 |

Costs are automatically tracked and can be viewed with `bookrc stats`. Token counts are stored per-session and per-book for full transparency.

### Supported Model Pricing

| Model | Input (per 1M tokens) | Output (per 1M tokens) |
|-------|----------------------|------------------------|
| gemini-2.5-flash | $0.075 | $0.30 |
| gemini-3-flash | $0.10 | $0.40 |
| claude-sonnet-4 | $3.00 | $15.00 |

## Troubleshooting

### "GEMINI_API_KEY not set"

Make sure you've added the API key to your shell config:

```bash
echo 'export GEMINI_API_KEY="your-key"' >> ~/.zshrc
source ~/.zshrc
```

### "Book not found"

Run `uv run bookrc list books` to see available book IDs.

### Slow ingestion

Ingestion with summarization processes each chapter through the LLM. For a 20-chapter book, expect:
- Parsing: seconds
- Embeddings: 10-30 seconds
- Summarization: 1-3 minutes

Use `--skip-summary` for faster ingestion if you don't need the index.

### Summarization failed

If summarization fails, the book is still ingested and usable for chat—you just won't have the navigation index. Re-ingest with `--force` to try again.

## Development

```bash
# Install dev dependencies
uv sync

# Run tests
uv run pytest

# Run CLI directly
uv run python -m book_companion.cli --help
```

## License

MIT

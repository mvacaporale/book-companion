# Book Reading Companion - Claude Instructions

## Project Overview

CLI application for chatting with books using RAG (Retrieval Augmented Generation) and LLMs. Supports PDF, EPUB, and Markdown formats with Gemini and Claude as LLM backends. Features hierarchical summarization with story/study extraction for enhanced comprehension.

## Architecture

```
book_companion/
├── cli.py                 # Click CLI commands (incl. stats command)
├── models.py              # Pydantic data models (incl. token tracking)
├── ingestion.py           # Reusable book ingestion logic
├── drive_sync.py          # Auto-ingest new books from the Drive folder
├── parsers/               # Book format parsers
│   ├── base.py            # BookParser ABC, get_parser()
│   ├── pdf_parser.py      # PyMuPDF-based PDF parser
│   ├── epub_parser.py     # ebooklib-based EPUB parser
│   └── markdown_parser.py # Markdown/text parser
├── processing/            # Text processing
│   ├── chunker.py         # Semantic chunking with chapter awareness
│   ├── embeddings.py      # Gemini text-embedding-004 client
│   └── summarizer.py      # Hierarchical summarization with token tracking
├── storage/               # Persistence layer
│   ├── vector_store.py    # ChromaDB wrapper
│   └── session_store.py   # Book registry, session, and index storage
├── llm/                   # LLM clients
│   ├── base.py            # LLMClient ABC
│   ├── gemini_client.py   # Gemini client (configurable model)
│   ├── claude_client.py   # Claude Sonnet client
│   └── factory.py         # get_llm_client() factory
├── chat/                  # Chat functionality
│   ├── engine.py          # RAG orchestrator with book index
│   ├── session.py         # Session management
│   └── prompts.py         # System prompts (index-aware)
├── google_drive/          # Google Drive integration
│   ├── __init__.py
│   ├── auth.py            # OAuth flow and token management
│   └── client.py          # GoogleDriveClient API wrapper
├── auth/                  # OAuth 2.1 for MCP server (disabled)
│   ├── __init__.py
│   ├── config.py          # OAuth configuration
│   ├── models.py          # Token, AuthCode, Client models
│   ├── store.py           # File-based token storage
│   ├── server.py          # OAuth endpoint handlers
│   └── middleware.py      # Starlette auth middleware
├── security/              # Security utilities
│   ├── __init__.py
│   └── sanitize.py        # Filename sanitization
├── mcp/                   # MCP Server for Claude Desktop
│   ├── __init__.py
│   └── server.py          # FastMCP server with all tools
└── utils/                 # Utility modules
    ├── __init__.py
    └── costs.py           # Token cost calculation utilities
```

## Key Components

### Models (models.py)

**Core Models:**
- `Book` - Book metadata record (includes token tracking fields)
- `Chunk` - Text chunk with metadata
- `Session` - Chat session with messages (includes `get_total_usage()`)
- `ChatMessage` - Message with optional `input_tokens`/`output_tokens`
- `TokenUsage` - Token usage tracking helper class
- `BookRegistry` - In-memory book collection
- `RetrievedContext` - RAG retrieval results

**Summarization Models:**
- `BookIndex` - Complete navigation index (includes token tracking)
- `BookSummary` - High-level book overview, thesis, themes
- `ChapterSummary` - Per-chapter summary with key concepts
- `ChapterIndexEntry` - Concise entry for LLM navigation
- `Narrative` - Story, study, case study, or example
- `NarrativeType` - Enum: story, study, case_study, example, quote

**Token Tracking:**
- `TokenUsage` - Tracks `input_tokens`, `output_tokens` with `add()` method
- `Book.summarization_input_tokens/output_tokens` - Ingestion token usage
- `BookIndex.total_input_tokens/output_tokens` - Summarization token usage
- `ChatMessage.input_tokens/output_tokens` - Per-message token usage
- `Session.get_total_usage()` - Aggregate usage for a session

### Processing

- `Chunker` - 1500 char chunks with 200 char overlap, chapter-aware
- `quality.py` - `is_non_content_title()` (front/back matter, matched as whole titles) and `is_low_value_chunk()` (index pages, endnotes, bare headings). Both vector stores over-fetch 3x and drop low-value chunks in `query()`, which also returns `distances`.

### Parsers

- Chapters are numbered by `number_chapters()` (`parsers/base.py`): if at least two sections carry explicit, increasing numbers ("CHAPTER 12", "12. Title", "Chapter XII", "Twelve"), those keep them and unnumbered sections are numbered after the last chapter; otherwise 1..N. Numbers must stay unique (the summarizer keys on them).
- EPUBs are read in spine order. PDF chapters end at the next TOC entry, even a skipped one, so notes/index aren't glued onto the last chapter.
- `clean_metadata_value()` drops placeholder titles/authors (`[No data]`, `Unknown`, digits-only) before they're stored.
- Books ingested before these rules (2026-10-29) keep their old numbers and metadata until re-ingested with `bookrc drive reingest` (`reingest.yml`: dry run on a PR that touches the workflow file, a real run of the books in its `REINGEST_BOOKS` env (blank = all) when such a change merges to main, or by hand from Actions). It finds each book's Drive file (checksum, else title match) and rebuilds it in place via `ingest_book(replace_book_id=...)`; books with no Drive file (ingested from a local copy on the Mac) are listed and skipped, so re-ingest those on the Mac with `bookrc ingest <path> --force`.
- Re-ingesting never deletes the `books` row (it upserts), because in Postgres that cascades to the book's chat sessions.
- `EmbeddingClient` - Gemini text-embedding-004, 768 dimensions
- `Summarizer` - Hierarchical summarization with configurable model

### Storage

Backend is auto-selected in `storage/__init__.py` by `is_postgres_configured()`:
**Postgres/pgvector when `DATABASE_URL` is set (current — Neon), else file-based.**

- Postgres: `PgVectorStore` (pgvector), `PgBookRegistryStore`, `PgSessionStore`, `PgBookIndexStore`
- File-based: `VectorStore` (ChromaDB), `BookRegistryStore` (books.json), `SessionStore` (~/.bookrc/sessions/), `BookIndexStore` (~/.bookrc/indices/)
- `get_connection()` discards pooled connections Neon has already closed (it drops idle ones, e.g. during a long summarization) and the pool uses TCP keepalives.
- Always get stores from the `get_*_store()` factories (CLI, ingestion, MCP); constructing `VectorStore()`/`BookRegistryStore()` directly silently bypasses Neon.
- Migrate file → Postgres with `scripts/migrate_to_postgres.py` (`export`/`import`/`verify`)

### Chat

- `ChatEngine` - Main RAG orchestrator with book index integration
- `SessionManager` - Session CRUD operations

## Hierarchical Summarization

During ingestion, the `Summarizer` generates:

1. **Chapter Summaries** - 2-3 paragraph summary per chapter
2. **Key Concepts** - Main ideas/arguments per chapter
3. **Narratives** - Stories, studies, case studies, examples
4. **Book Summary** - Overall thesis, themes, target audience
5. **Navigation Index** - Concise chapter entries for LLM context

The navigation index is included in the system prompt, giving the LLM a "map" of the book to reason over.

### Narrative Types

| Type | Description |
|------|-------------|
| `story` | Anecdotes, personal narratives |
| `study` | Research experiments, academic findings |
| `case_study` | Real-world examples (companies, people, events) |
| `example` | Illustrative examples |
| `quote` | Notable quotes |

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `GEMINI_API_KEY` | Required for embeddings, summarization, and Gemini chat |
| `ANTHROPIC_API_KEY` | Required for Claude chat |
| `DATABASE_URL` | Postgres/pgvector connection string. **When set, ALL storage (embeddings, registry, indices, sessions) goes to Postgres**; unset → file-based `~/.bookrc/`. This is the current backend (Neon). |
| `BOOKRC_GWS_CREDENTIALS` | Path to the gws (Google Workspace CLI) credentials file used for Drive auth (default: `~/.config/gws/credentials.json`) |
| `BOOKRC_DB_PATH` | Custom file-based data directory (default: ~/.bookrc/); ignored when `DATABASE_URL` is set |

**Default Models:**
- Summarization: `gemini-3.8-flash` (latest; configurable via `--model`)
- Chat: `gemini-2.5-flash` (configurable via `--provider`)

## Google Drive Integration

Allows searching and ingesting books directly from Google Drive.

### Drive auth (current: reuse the gws token)

`get_credentials()` in `google_drive/auth.py` tries, in order:
1. **Service account** via Application Default Credentials (legacy Cloud Run; returns `None` off-GCP).
2. **gws (Google Workspace CLI) OAuth token** — `_get_gws_credentials()` reads
   `~/.config/gws/credentials.json` (client id/secret + long-lived refresh token),
   mints a Drive access token, and caches it to `~/.bookrc/google_token.json`
   without mutating the gws file. This is the path in use — the gws app is
   published to production, so the refresh token does not hit Google's 7-day
   testing-mode expiry (the old re-auth pain).
3. **Local OAuth flow** (`bookrc setup-drive`) as a fallback.

**Setup per machine:**
- Mac: `gws auth export --unmasked > ~/.config/gws/credentials.json` (one-time).
- miciato: already has a plaintext `~/.config/gws/credentials.json`.
- Folder id comes from `GOOGLE_DRIVE_FOLDER_ID` or `~/.bookrc/config.json`
  (`1ip13wKCBGznT2S2HSESUseiNciJBMFCm`).

### Configuration Files

| File | Purpose |
|------|---------|
| `~/.config/gws/credentials.json` | gws OAuth client + refresh token (Drive auth source) |
| `~/.bookrc/google_token.json` | Cached Drive access token (auto-generated) |
| `~/.bookrc/config.json` | Drive settings (default_folder_id, cache_ttl) |
| `~/.bookrc/mcp.env` | MCP launch env — holds only `pass://` refs (no secrets) |
| `~/.bookrc/downloads/` | Downloaded book files from Drive |

### Automatic ingestion (Drive watcher → GitHub Actions)

`drive-watcher/Code.gs` (Google Apps Script, user-installed at script.google.com,
not deployed from the repo) fingerprints the books folder's file IDs every 5
minutes and, on change, dispatches `.github/workflows/drive-sync.yml` using a
fine-grained token (`GITHUB_TOKEN` script property, Actions read/write on this
repo only). The workflow also runs daily at 10:00 UTC as a backstop, and by hand
(`dry_run`, `max_books` inputs). It runs `bookrc drive sync` from `main` (merge =
deploy) with repo secrets `GEMINI_API_KEY`, `DATABASE_URL`, `GWS_CREDENTIALS_JSON`,
and skips with a warning without them. `drive_sync.sync_drive_folder` treats a
Drive file as already ingested when its Drive `md5Checksum` equals a book's
`file_hash`, or its cleaned filename fuzzy-matches a title (>= 80, same rule as
`find_book_in_drive`), and ingests at most 5 new books per run. Files whose Drive `createdTime` is before
`DRIVE_SYNC_SINCE` (workflow env, `--added-since`; set to 2026-10-29T00:30:00Z because
the user chose not to ingest the 41-book backlog) are skipped; moved-in files keep
their old createdTime, so those need `bookrc drive ingest`. Per-book
failures don't stop the run but exit non-zero, so GitHub emails the user.
Drive push notifications were rejected (need a public HTTPS endpoint, channels
expire), as were a miciato timer (user wants it self-contained in the project)
and Inngest (needs a hosted endpoint).

### Workflow

When a user asks about a book that isn't ingested:
1. Use `find_book_in_drive("Book Title")` to search Google Drive
2. Offer the user a choice:
   - **Load to context**: Use `load_book_from_drive(file_id)` for quick Q&A
   - **Full ingest**: Use `ingest_book_from_drive(file_id)` for deeper exploration

## Common Commands

```bash
# Development
uv sync                              # Install dependencies
uv run bookrc --help                 # Show CLI help
uv run pytest                        # Run tests

# Ingestion (uses Gemini 3 Flash with parallel processing by default)
uv run bookrc ingest <path>                          # Full ingest with summaries (2 workers)
uv run bookrc ingest <path> -j 4                     # Use 4 parallel workers
uv run bookrc ingest <path> --model claude-sonnet-4-20250514  # Use Claude instead
uv run bookrc ingest <path> --skip-summary           # Fast ingest, no summaries
uv run bookrc ingest <path> --force                  # Re-ingest existing book

# Chat
uv run bookrc chat <book_id>                    # Start chat (Gemini)
uv run bookrc chat <book_id> --provider claude  # Use Claude
uv run bookrc chat <book_id> --resume <session> # Resume session

# Management
uv run bookrc list books             # List books (shows index status)
uv run bookrc list sessions <id>     # List sessions for a book
uv run bookrc info <book_id>         # Book details + narratives + token usage
uv run bookrc index <book_id>        # Show navigation index (debug)
uv run bookrc edit <book_id> -t "Title" -a "Author"  # Fix metadata without re-ingesting
uv run bookrc delete <book_id>       # Delete book and all data

# Token Usage & Costs
uv run bookrc stats                  # Show usage for all books
uv run bookrc stats <book_id>        # Show usage for specific book

# Google Drive Integration
uv run bookrc setup-drive            # One-time OAuth setup
uv run bookrc drive search "Atomic Habits"  # Search Drive for books
uv run bookrc drive list             # List all book files in Drive
uv run bookrc drive ingest <file_id> # Download and ingest from Drive (--force to re-ingest)
uv run bookrc drive sync --dry-run   # List Drive books not ingested yet (drop flag to ingest)
uv run bookrc drive reingest --dry-run  # Rebuild stored books from Drive in place (--book ID, repeatable)

# MCP Server
mcp dev book_companion/mcp/server.py # Test with MCP inspector
uv run python -m book_companion.mcp.server  # Run server directly
```

## Data Flow

1. **Ingest**:
   ```
   path → Parser → ParsedBook → Chunker → Chunks → EmbeddingClient → VectorStore
                            ↓
                     Summarizer → BookIndex → BookIndexStore
   ```

2. **Chat**:
   ```
   query → EmbeddingClient → VectorStore.query() → context
                                                      ↓
   BookIndex (system prompt) + context → LLMClient → response
   ```

## RAG Configuration

- Chunk size: 1500 characters (~300-400 tokens)
- Chunk overlap: 200 characters
- Embedding model: text-embedding-004 (768 dimensions)
- Retrieval: Top 8 chunks by cosine similarity, low-value chunks (index, endnotes, bare headings) dropped
- Context format: Includes chapter/page citations
- System prompt: Includes full book navigation index (when available)

## Cost Estimates & Token Tracking

Token usage is tracked automatically for all LLM operations:
- Summarization tokens stored in `BookIndex`
- Chat tokens stored per `ChatMessage`
- View with `bookrc stats` or `bookrc info <book_id>`

**Summarization** (Gemini 3 Flash, per book - default):
- Input: ~200K tokens → ~$0.02
- Output: ~26K tokens → ~$0.01
- **Total: ~$0.03 per book** (cheap, use 2 workers to avoid rate limits)

**Summarization** (Claude Sonnet 4, per book - alternative):
- Input: ~200K tokens → ~$0.60
- Output: ~26K tokens → ~$0.39
- **Total: ~$1.00 per book** (higher cost but no rate limiting, can use more workers)

**Chat** (per query):
- Embedding: ~100 tokens → negligible
- LLM: ~2-4K tokens → ~$0.001-0.01

**Model Pricing** (per 1M tokens):
| Model | Input | Output |
|-------|-------|--------|
| gemini-3-flash | $0.10 | $0.40 |
| gemini-2.5-flash | $0.075 | $0.30 |
| claude-sonnet-4 | $3.00 | $15.00 |

## MCP Server

The MCP (Model Context Protocol) server enables Claude Desktop and other MCP clients to interact with book-companion.

### Available Tools

| Tool | Description |
|------|-------------|
| `list_books` | List all ingested books with metadata. Optional `topic` filter for fuzzy matching against themes/topics |
| `search_books` | Semantic search across books; `n_results` is the overall limit, results sorted by `score` |
| `get_book_index` | Full navigation index with summaries |
| `get_chapter_summary` | Specific chapter summary and narratives |
| `get_narratives` | Stories, studies, examples (filterable by type) |
| `get_stats` | Token usage and cost statistics |
| `chat` | RAG chat with session support (single or multi-book) |
| `find_book_in_drive` | Search Google Drive for books (fuzzy matching). Empty query lists all. Shows `is_ingested` status |
| `load_book_from_drive` | Load book content into context (no persistence) |
| `ingest_book_from_drive` | Download and fully ingest a book from Drive |

### Deployment (current): local stdio + Neon + gws + Proton Pass

The server runs as a **local stdio process** in Claude Code on both the Mac and
miciato (no hosting, no public endpoint). Secrets are pulled from Proton Pass at
launch via `pass-cli run`, so `~/.claude.json` holds no secrets.

`~/.bookrc/mcp.env` (references only, no secret values):
```
GEMINI_API_KEY=pass://Agent Passwords/GEMINI_API_KEY/password
DATABASE_URL=pass://Agent Passwords/BOOKRC_DATABASE_URL/password
GOOGLE_DRIVE_FOLDER_ID=1ip13wKCBGznT2S2HSESUseiNciJBMFCm
```

Register (Mac). The server runs straight from GitHub `main` with `uvx`, so merging
a PR is the deploy: the next Claude Code session that starts the server picks it up
(`--refresh` re-checks `main` on every launch; no local checkout or `git pull` needed):
```bash
claude mcp add book-companion --scope user -- \
  pass-cli run --env-file ~/.bookrc/mcp.env -- \
  uvx --refresh --from git+https://github.com/mvacaporale/book-companion bookrc-mcp
```

miciato is identical but adds `-e PROTON_PASS_KEY_PROVIDER=fs` and uses absolute
paths (`/root/.local/bin/pass-cli`, `/root/.local/bin/uvx`).
See the parent memory `pass-cli-miciato-headless` for the box's headless pass-cli rules.

`uvx` resolves dependencies from `pyproject.toml`, not `uv.lock`, so keep upper
bounds on anything with breaking majors (e.g. `mcp<2`: mcp 2.x removed `FastMCP`).
To run a local checkout instead (e.g. testing a branch), swap the last line for
`uv run --directory <checkout> bookrc-mcp`.

State lives in one **Neon Postgres** project (`DATABASE_URL`), shared by both machines.

Quick local test (MCP inspector): `mcp dev book_companion/mcp/server.py`.

### HTTP/SSE transports (legacy)

`server.py` still supports `sse`/`http` transports and a `stateless_http=True`
FastMCP setting (needed by claude.ai connectors, which call `tools/list` without
an `mcp-session-id`). These are unused now that deployment is local stdio; kept
for optional future self-hosting behind a tunnel.

### Google Drive Tools Broken?

See `docs/drive-reauth-runbook.md`. Usual cause: the refresh token in the
`GOOGLE_DRIVE_TOKEN` secret expired because the OAuth consent screen is in
*Testing* status (Google kills those after 7 days).

### OAuth Authentication (Disabled - Future Feature)

OAuth 2.1 authentication is fully implemented but currently **disabled** due to bugs in Claude.ai's web interface OAuth handling ([Issue #11814](https://github.com/anthropics/claude-code/issues/11814)).

**To enable OAuth** (when Claude.ai fixes their implementation):

1. Set environment variable: `MCP_OAUTH_ENABLED=true`
2. Add to deploy workflow in `.github/workflows/deploy-simple.yml`:
   ```yaml
   --set-env-vars "MCP_OAUTH_ENABLED=true" \
   ```

**OAuth Implementation Details:**
- Location: `book_companion/auth/`
- Endpoints: `/.well-known/oauth-authorization-server`, `/register`, `/authorize`, `/token`
- Implements OAuth 2.1 with PKCE support
- Dynamic client registration (RFC 7591)
- Uses Google Drive credentials as the authentication source
- Tokens stored in `~/.bookrc/oauth_tokens.json` (local) or GCS bucket (Cloud Run)

**OAuth works correctly with:**
- Claude Code CLI: `claude mcp add --transport http book-companion <url>`
- MCP Inspector
- Manual curl testing

**Known issues with Claude.ai web:**
- OAuth flow doesn't complete ([#11814](https://github.com/anthropics/claude-code/issues/11814))
- Ignores external auth server endpoints ([#82](https://github.com/anthropics/claude-ai-mcp/issues/82))

## Future Work

- Obsidian plugin support
- Additional LLM providers (OpenAI, local models)
- Thematic retrieval using summaries first
- Section-level summarization (H2/H3)
- Multi-book session persistence
- Google Drive sync detection for *changed* files (new files are auto-ingested within ~10 minutes; edits to an already-ingested file are not)

## Cloud Run Configuration (RETIRED 2026-09-14)

The Cloud Run deployment has been **decommissioned**. The service
`book-companion-mcp`, its GCS bucket `general-477905-bookrc-data`, and the
`book-companion-drive` service account were deleted. Deployment is now local
stdio + Neon (see **Deployment (current)** above). The `deploy-simple.yml` and
`setup-drive-sa.yml` workflows were removed. Kept only as history — do not
redeploy to Cloud Run.

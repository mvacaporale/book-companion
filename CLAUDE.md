# Book Reading Companion - Claude Instructions

## Project Overview

CLI application for chatting with books using RAG (Retrieval Augmented Generation) and LLMs. Supports PDF, EPUB, and Markdown formats with Gemini and Claude as LLM backends. Features hierarchical summarization with story/study extraction for enhanced comprehension.

## Architecture

```
book_companion/
├── cli.py                 # Click CLI commands (incl. stats command)
├── models.py              # Pydantic data models (incl. token tracking)
├── ingestion.py           # Reusable book ingestion logic
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
- `EmbeddingClient` - Gemini text-embedding-004, 768 dimensions
- `Summarizer` - Hierarchical summarization with configurable model

### Storage

- `VectorStore` - ChromaDB with cosine similarity
- `BookRegistryStore` - books.json persistence
- `SessionStore` - Session JSON files in ~/.bookrc/sessions/
- `BookIndexStore` - Book indices in ~/.bookrc/indices/

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
| `BOOKRC_DB_PATH` | Custom data directory (default: ~/.bookrc/) |

## Google Drive Integration

Allows searching and ingesting books directly from Google Drive.

### Setup (One-Time)

1. Create OAuth credentials in [Google Cloud Console](https://console.cloud.google.com/apis/credentials)
2. Enable the Google Drive API
3. Create OAuth 2.0 Client ID (Desktop app type)
4. Download JSON and save to `~/.bookrc/google_credentials.json`
5. Run `uv run bookrc setup-drive` to complete OAuth flow

### Configuration Files

| File | Purpose |
|------|---------|
| `~/.bookrc/google_credentials.json` | OAuth client credentials (from GCP) |
| `~/.bookrc/google_token.json` | OAuth refresh token (auto-generated) |
| `~/.bookrc/config.json` | Drive settings (default_folder_id, cache_ttl) |
| `~/.bookrc/downloads/` | Downloaded book files from Drive |

### Workflow

When a user asks about a book that isn't ingested:
1. Use `find_book_in_drive("Book Title")` to search Google Drive
2. Offer the user a choice:
   - **Load to context**: Use `load_book_from_drive(file_id)` for quick Q&A
   - **Full ingest**: Use `ingest_book_from_drive(file_id)` for deeper exploration

### Cloud Run Deployment

For Cloud Run, set these environment variables instead of using files:

| Variable | Purpose | How to Get |
|----------|---------|------------|
| `GOOGLE_DRIVE_TOKEN` | OAuth token JSON | Run `cat ~/.bookrc/google_token.json` after local setup |
| `GOOGLE_DRIVE_FOLDER_ID` | Default folder to search | Your folder ID (already configured: `1ip13wKCBGznT2S2HSESUseiNciJBMFCm`) |

**Setup steps for Cloud Run:**
1. Run `bookrc setup-drive` locally to authenticate
2. Copy the token: `cat ~/.bookrc/google_token.json`
3. Set as Cloud Run secret: `GOOGLE_DRIVE_TOKEN`
4. Set folder ID: `GOOGLE_DRIVE_FOLDER_ID=1QaYnMzEc2JiiWheX5j4wuoaVgi9raHNQ`

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
uv run bookrc delete <book_id>       # Delete book and all data

# Token Usage & Costs
uv run bookrc stats                  # Show usage for all books
uv run bookrc stats <book_id>        # Show usage for specific book

# Google Drive Integration
uv run bookrc setup-drive            # One-time OAuth setup
uv run bookrc drive search "Atomic Habits"  # Search Drive for books
uv run bookrc drive list             # List all book files in Drive
uv run bookrc drive ingest <file_id> # Download and ingest from Drive

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
- Retrieval: Top 8 chunks by cosine similarity
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
| `search_books` | Semantic search across books (supports multi-book) |
| `get_book_index` | Full navigation index with summaries |
| `get_chapter_summary` | Specific chapter summary and narratives |
| `get_narratives` | Stories, studies, examples (filterable by type) |
| `get_stats` | Token usage and cost statistics |
| `chat` | RAG chat with session support (single or multi-book) |
| `find_book_in_drive` | Search Google Drive for books (fuzzy matching). Empty query lists all. Shows `is_ingested` status |
| `load_book_from_drive` | Load book content into context (no persistence) |
| `ingest_book_from_drive` | Download and fully ingest a book from Drive |

### Running the MCP Server

```bash
# Test with MCP inspector
cd /Users/michaelangelocaporale/Documents/Projects/claude/book-companion
mcp dev book_companion/mcp/server.py
```

### Claude Desktop Configuration

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "book-companion": {
      "command": "uv",
      "args": [
        "run", "--directory",
        "/Users/michaelangelocaporale/Documents/Projects/claude/book-companion",
        "python", "-m", "book_companion.mcp.server"
      ]
    }
  }
}
```

After adding, restart Claude Desktop. The "book-companion" server will appear in connectors.

## Future Work

- Obsidian plugin support
- Additional LLM providers (OpenAI, local models)
- Thematic retrieval using summaries first
- Section-level summarization (H2/H3)
- Multi-book session persistence
- Google Drive sync detection (check if Drive file has been updated)

## Cloud Run Configuration

**Service:** `book-companion-mcp`

### Required Environment Variables

| Variable | Value/Source |
|----------|--------------|
| `GEMINI_API_KEY` | From `~/.zshrc` |
| `GOOGLE_DRIVE_TOKEN_B64` | `cat ~/.bookrc/google_token.json \| base64` |
| `GOOGLE_DRIVE_FOLDER_ID` | `1ip13wKCBGznT2S2HSESUseiNciJBMFCm` (Books folder) |

### Deploy Commands

```bash
# Redeploy with new code
gcloud run deploy book-companion-mcp --project=general-477905 --region=us-central1 --source=.

# Update env vars only
gcloud run services update book-companion-mcp --project=general-477905 --region=us-central1 --update-env-vars="KEY=value"
```

### OAuth Details

- **Client:** "GTD Client" in GCP project `general-477905`
- **Credentials:** `~/.bookrc/google_credentials.json`
- **Token:** `~/.bookrc/google_token.json`

See parent `CLAUDE.md` for general Cloud Run and OAuth process notes.

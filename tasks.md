# Book Companion - Future Tasks

## High Priority

### 1. Improve processing transparency
**Problem**: During ingestion, there's no visibility into which chapter is being processed or progress percentage when running in background.

**Requirements**:
- Show "Processing chapter X of Y: [Chapter Title]" during summarization
- Log progress to file when running in background (Rich progress doesn't write to files)
- Show estimated time remaining based on average chapter processing time

**Status**: Partially addressed (progress bar shows, but not in background mode)

### 2. Token/cost tracking - `bookrc stats` command
**Problem**: Token tracking is in place but no way to view it.

**Implemented**:
- [x] Track input/output tokens for summarization (in BookIndex)
- [x] Track tokens per chat message (in Session)
- [x] Store cumulative usage per book in metadata

**Remaining**:
- [ ] Add `bookrc stats` command to show:
  - Total tokens used across all books
  - Estimated cost breakdown by operation
  - Per-book usage statistics

### 3. MCP Server / RAG Connector
**Problem**: Currently CLI-only. Need to expose as a connector for other tools (Claude Desktop, Obsidian, etc.)

**Requirements**: (needs planning discussion)
- [ ] Define MCP server interface
- [ ] Determine which operations to expose:
  - `search_book(query, book_id)` - Semantic search
  - `get_chapter_summary(book_id, chapter)` - Get pre-computed summary
  - `get_book_index(book_id)` - Get navigation index
  - `list_books()` - Available books
  - `chat(book_id, message, session_id?)` - Full RAG chat
- [ ] Consider authentication/security
- [ ] Plan Obsidian integration approach

**Notes**: This is the big one. Need to discuss architecture and what integrations matter most.

---

## Medium Priority

### 4. Better chapter detection
- Improve PDF TOC extraction heuristics
- Handle books without clear chapter markers
- Support section-level (H2/H3) summarization

### 5. Streaming chat responses
- Currently waits for full response
- Implement `chat_stream` in CLI for real-time output

### 6. Multiple book chat
- Chat across multiple books simultaneously
- Compare concepts between books

---

## Low Priority / Nice to Have

### 7. Local LLM support
- Add Ollama/llama.cpp backend
- Allow offline operation

### 8. Export features
- Export book summaries to Markdown
- Export Anki flashcards from key concepts
- Export highlights/annotations

### 9. Web UI
- Simple web interface alternative to CLI
- Shareable book libraries

---

## Completed

- [x] Basic CLI (ingest, chat, list, info, delete)
- [x] PDF, EPUB, Markdown parsing
- [x] Semantic chunking with chapter awareness
- [x] Gemini embeddings (text-embedding-004)
- [x] ChromaDB vector storage
- [x] Gemini and Claude chat backends
- [x] Session persistence
- [x] Hierarchical summarization
- [x] Story/study extraction (347 extracted from "Origins of Efficiency")
- [x] Configurable summarization model
- [x] Book navigation index in system prompt
- [x] Token tracking in models (Book, Session, BookIndex)

---

## Test Books

| Book | ID | Chapters | Chunks | Stories/Studies | Model |
|------|-----|----------|--------|-----------------|-------|
| The Origins of Efficiency | 3f32ebf3 | 15 | 628 | 347 | gemini-2.5-flash |

# Book Companion - Future Work

## High Priority

### Background Mode Progress
**Problem**: During ingestion in background mode, there's no visibility into progress (Rich progress doesn't write to files).

**Idea**: Log progress to file when running in background, so users can tail the log.

---

## Medium Priority

### Better Chapter Detection
- Improve PDF TOC extraction heuristics
- Handle books without clear chapter markers
- Support section-level (H2/H3) summarization

### Streaming Chat Responses
- Currently waits for full response
- Implement `chat_stream` in CLI for real-time output

### Multi-book Session Persistence
- Currently multi-book chats aren't saved
- Add session persistence for cross-book conversations

---

## Low Priority / Nice to Have

### Local LLM Support
- Add Ollama/llama.cpp backend
- Allow offline operation

### Export Features
- Export book summaries to Markdown
- Export Anki flashcards from key concepts
- Export highlights/annotations

### Web UI
- Simple web interface alternative to CLI
- Shareable book libraries

### Obsidian Plugin
- Direct integration with Obsidian notes
- Link book content to notes

---

## Test Books

| Book | ID | Chapters | Chunks | Stories/Studies | Model |
|------|-----|----------|--------|-----------------|-------|
| The Origins of Efficiency | 3f32ebf3 | 15 | 628 | 347 | gemini-2.5-flash |

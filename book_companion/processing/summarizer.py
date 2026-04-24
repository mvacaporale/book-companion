"""Hierarchical book summarization using LLMs."""

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import anthropic
from google import genai
from google.genai import types
from pydantic import ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential

from book_companion.models import (
    BookIndex,
    BookSummary,
    Chapter,
    ChapterIndexEntry,
    ChapterSummary,
    Narrative,
    NarrativeType,
    ParsedBook,
)


# Prompts for summarization

CHAPTER_SUMMARY_PROMPT = """Analyze this chapter and provide a structured summary.

CHAPTER {chapter_number}: {chapter_title}
{page_info}

TEXT:
{content}

---

Provide your analysis as JSON with this exact structure:
{{
    "summary": "A 2-3 paragraph summary of the chapter's main content and arguments",
    "key_concepts": ["concept 1", "concept 2", "concept 3"],
    "narratives": [
        {{
            "type": "story|study|case_study|example|quote",
            "title": "Brief descriptive title",
            "description": "What happened and what it illustrates (2-3 sentences)",
            "source": "Person/researcher name if applicable, or null"
        }}
    ]
}}

Guidelines:
- The summary should capture the chapter's main arguments and how they connect
- Key concepts should be 3-7 of the most important ideas
- For narratives, extract ALL stories, research studies, case studies, and notable examples
- Stories = anecdotes, personal narratives the author shares
- Studies = research experiments, academic findings (include researcher names)
- Case studies = real-world examples (companies, people, events)
- Examples = illustrative examples the author uses to explain concepts
- Quotes = particularly notable or memorable quotes

Respond ONLY with valid JSON, no other text."""


BOOK_SUMMARY_PROMPT = """Based on these chapter summaries, provide an overall book summary.

BOOK: {title}
AUTHOR: {author}

CHAPTER SUMMARIES:
{chapter_summaries}

---

Provide your analysis as JSON with this exact structure:
{{
    "overview": "A 2-3 paragraph overview of the entire book",
    "main_thesis": "The book's central argument in 1-2 sentences",
    "key_themes": ["theme 1", "theme 2", "theme 3"],
    "target_audience": "Who this book is written for"
}}

Respond ONLY with valid JSON, no other text."""


class Summarizer:
    """Generates hierarchical summaries of books using LLMs."""

    DEFAULT_MODEL = "gemini-2.5-flash"

    # Model provider detection
    CLAUDE_MODELS = ["claude-sonnet-4-20250514", "claude-sonnet-4", "claude-haiku", "claude-opus"]
    GEMINI_MODELS = ["gemini-2.5-flash", "gemini-3-flash-preview", "gemini-2.0-flash"]

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        max_workers: int = 4,
    ):
        """Initialize the summarizer.

        Args:
            model: Model to use for summarization. Defaults to claude-sonnet-4.
                   Supports: claude-sonnet-4, gemini-2.5-flash, gemini-3-flash-preview
            api_key: API key. Defaults to ANTHROPIC_API_KEY or GEMINI_API_KEY env var.
            max_workers: Number of parallel workers for chapter summarization (default: 4)
        """
        self.model = model or self.DEFAULT_MODEL
        self.max_workers = max_workers

        # Determine provider from model name
        self.provider = "claude" if any(m in self.model for m in ["claude", "sonnet", "haiku", "opus"]) else "gemini"

        if self.provider == "claude":
            api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError(
                    "ANTHROPIC_API_KEY environment variable not set. "
                    "Please set it in ~/.zshrc or pass api_key parameter."
                )
            self.claude_client = anthropic.Anthropic(api_key=api_key)
            self.gemini_client = None
        else:
            api_key = api_key or os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError(
                    "GEMINI_API_KEY environment variable not set. "
                    "Please set it in ~/.zshrc or pass api_key parameter."
                )
            self.gemini_client = genai.Client(api_key=api_key)
            self.claude_client = None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
    )
    def _generate(self, prompt: str) -> tuple[str, int, int]:
        """Generate text using the LLM.

        Returns:
            Tuple of (response_text, input_tokens, output_tokens)
        """
        if self.provider == "claude":
            return self._generate_claude(prompt)
        else:
            return self._generate_gemini(prompt)

    def _generate_claude(self, prompt: str) -> tuple[str, int, int]:
        """Generate text using Claude."""
        response = self.claude_client.messages.create(
            model=self.model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )

        content = response.content[0].text
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

        return content, input_tokens, output_tokens

    def _generate_gemini(self, prompt: str) -> tuple[str, int, int]:
        """Generate text using Gemini."""
        response = self.gemini_client.models.generate_content(
            model=self.model,
            contents=[prompt],
            config=types.GenerateContentConfig(
                temperature=0.3,
                response_mime_type="application/json",
            ),
        )

        input_tokens = 0
        output_tokens = 0
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            input_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
            output_tokens = getattr(response.usage_metadata, "candidates_token_count", 0) or 0

        return response.text, input_tokens, output_tokens

    def _parse_json_response(self, text: str) -> dict:
        """Parse JSON from LLM response, handling common issues."""
        # Clean up common issues
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        return json.loads(text)

    def summarize_chapter(
        self,
        chapter: Chapter,
        book_title: str,
    ) -> tuple[ChapterSummary, int, int]:
        """Generate a summary for a single chapter.

        Args:
            chapter: The chapter to summarize
            book_title: Title of the book (for context)

        Returns:
            Tuple of (ChapterSummary, input_tokens, output_tokens)
        """
        # Build page info string
        page_info = ""
        if chapter.start_page and chapter.end_page:
            page_info = f"Pages {chapter.start_page}-{chapter.end_page}"
        elif chapter.start_page:
            page_info = f"Starting page {chapter.start_page}"

        # Truncate very long chapters to avoid token limits
        content = chapter.content
        max_chars = 50000  # ~12k tokens, well within limits
        if len(content) > max_chars:
            content = content[:max_chars] + "\n\n[Content truncated for summarization...]"

        prompt = CHAPTER_SUMMARY_PROMPT.format(
            chapter_number=chapter.number,
            chapter_title=chapter.title or "Untitled",
            page_info=page_info,
            content=content,
        )

        response_text, input_tokens, output_tokens = self._generate(prompt)
        data = self._parse_json_response(response_text)

        # Parse narratives
        narratives = []
        for n in data.get("narratives", []):
            try:
                narrative_type = NarrativeType(n.get("type", "example"))
            except ValueError:
                narrative_type = NarrativeType.EXAMPLE

            narratives.append(
                Narrative(
                    type=narrative_type,
                    title=n.get("title", "Untitled"),
                    description=n.get("description", ""),
                    source=n.get("source"),
                    chapter_number=chapter.number,
                    page_reference=page_info if page_info else None,
                )
            )

        # Build page range string
        page_range = None
        if chapter.start_page and chapter.end_page:
            page_range = f"pp. {chapter.start_page}-{chapter.end_page}"
        elif chapter.start_page:
            page_range = f"p. {chapter.start_page}"

        return (
            ChapterSummary(
                chapter_number=chapter.number,
                chapter_title=chapter.title,
                page_range=page_range,
                summary=data.get("summary", ""),
                key_concepts=data.get("key_concepts", []),
                narratives=narratives,
            ),
            input_tokens,
            output_tokens,
        )

    def summarize_book(
        self,
        title: str,
        author: Optional[str],
        chapter_summaries: list[ChapterSummary],
    ) -> tuple[BookSummary, int, int]:
        """Generate an overall book summary from chapter summaries.

        Args:
            title: Book title
            author: Book author
            chapter_summaries: List of chapter summaries

        Returns:
            Tuple of (BookSummary, input_tokens, output_tokens)
        """
        # Format chapter summaries for the prompt
        summaries_text = []
        for cs in chapter_summaries:
            title_str = f'"{cs.chapter_title}"' if cs.chapter_title else f"Chapter {cs.chapter_number}"
            summaries_text.append(f"Chapter {cs.chapter_number}: {title_str}")
            summaries_text.append(cs.summary)
            summaries_text.append(f"Key concepts: {', '.join(cs.key_concepts)}")
            summaries_text.append("")

        prompt = BOOK_SUMMARY_PROMPT.format(
            title=title,
            author=author or "Unknown",
            chapter_summaries="\n".join(summaries_text),
        )

        response_text, input_tokens, output_tokens = self._generate(prompt)
        data = self._parse_json_response(response_text)

        return (
            BookSummary(
                overview=data.get("overview", ""),
                main_thesis=data.get("main_thesis", ""),
                key_themes=data.get("key_themes", []),
                target_audience=data.get("target_audience"),
            ),
            input_tokens,
            output_tokens,
        )

    def create_chapter_index_entry(
        self,
        chapter_summary: ChapterSummary,
    ) -> ChapterIndexEntry:
        """Create a navigation index entry from a chapter summary.

        Args:
            chapter_summary: The chapter summary

        Returns:
            ChapterIndexEntry for navigation
        """
        # Extract first sentence of summary as core argument
        summary = chapter_summary.summary
        first_sentence = summary.split(". ")[0] + "." if ". " in summary else summary[:200]

        # Get notable content from narratives
        notable = []
        for n in chapter_summary.narratives[:3]:  # Top 3
            type_label = n.type.value.replace("_", " ").title()
            notable.append(f"{type_label}: {n.title}")

        return ChapterIndexEntry(
            chapter_number=chapter_summary.chapter_number,
            title=chapter_summary.chapter_title,
            page_range=chapter_summary.page_range,
            core_argument=first_sentence,
            key_topics=chapter_summary.key_concepts[:5],
            notable_content=notable,
        )

    def process_book(
        self,
        parsed_book: ParsedBook,
        book_id: str,
        progress_callback: Optional[callable] = None,
    ) -> BookIndex:
        """Process an entire book and generate the full index.

        Uses parallel processing for chapter summarization when max_workers > 1.

        Args:
            parsed_book: The parsed book
            book_id: ID for the book
            progress_callback: Optional callback(step, total, message) for progress

        Returns:
            Complete BookIndex with summaries, navigation, and token usage
        """
        total_steps = len(parsed_book.chapters) + 1  # chapters + book summary
        completed_chapters = 0

        all_narratives = []
        total_input_tokens = 0
        total_output_tokens = 0

        # Parallel chapter summarization
        chapter_results: dict[int, ChapterSummary] = {}

        if progress_callback:
            progress_callback(0, total_steps, f"Summarizing {len(parsed_book.chapters)} chapters ({self.max_workers} workers)...")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all chapter tasks
            future_to_chapter = {
                executor.submit(self._summarize_chapter_safe, chapter, parsed_book.title): chapter
                for chapter in parsed_book.chapters
            }

            # Process completed futures
            for future in as_completed(future_to_chapter):
                chapter = future_to_chapter[future]
                try:
                    summary, input_tokens, output_tokens = future.result()
                    chapter_results[chapter.number] = summary
                    total_input_tokens += input_tokens
                    total_output_tokens += output_tokens
                except Exception as e:
                    # Create minimal summary on error
                    chapter_results[chapter.number] = ChapterSummary(
                        chapter_number=chapter.number,
                        chapter_title=chapter.title,
                        summary=f"[Summary generation failed: {e}]",
                        key_concepts=[],
                    )

                completed_chapters += 1
                if progress_callback:
                    progress_callback(
                        completed_chapters,
                        total_steps,
                        f"Chapter {completed_chapters}/{len(parsed_book.chapters)} complete",
                    )

        # Sort summaries by chapter number
        chapter_summaries = [chapter_results[ch.number] for ch in parsed_book.chapters]

        # Collect all narratives
        for summary in chapter_summaries:
            all_narratives.extend(summary.narratives)

        # Generate book summary (sequential - needs all chapter summaries)
        if progress_callback:
            progress_callback(completed_chapters, total_steps, "Generating book summary...")

        book_summary, book_input_tokens, book_output_tokens = self.summarize_book(
            title=parsed_book.title,
            author=parsed_book.author,
            chapter_summaries=chapter_summaries,
        )
        total_input_tokens += book_input_tokens
        total_output_tokens += book_output_tokens

        # Create chapter index entries
        chapter_index = [
            self.create_chapter_index_entry(cs)
            for cs in chapter_summaries
        ]

        return BookIndex(
            book_id=book_id,
            title=parsed_book.title,
            author=parsed_book.author,
            book_summary=book_summary,
            chapter_summaries=chapter_summaries,
            chapter_index=chapter_index,
            all_narratives=all_narratives,
            model_used=self.model,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
        )

    def _summarize_chapter_safe(
        self,
        chapter: Chapter,
        book_title: str,
    ) -> tuple[ChapterSummary, int, int]:
        """Thread-safe wrapper for summarize_chapter."""
        return self.summarize_chapter(chapter, book_title)

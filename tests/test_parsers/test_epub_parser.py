"""EPUB parser tests against a small generated EPUB."""

from pathlib import Path

from ebooklib import epub

from book_companion.parsers.epub_parser import EPUBParser


def _page(name: str, heading: str, body: str) -> epub.EpubHtml:
    item = epub.EpubHtml(title=heading, file_name=f"{name}.xhtml")
    item.content = f"<html><body><h1>{heading}</h1><p>{body}</p></body></html>"
    return item


def test_epub_reads_spine_order_and_book_numbering(tmp_path: Path) -> None:
    book = epub.EpubBook()
    book.set_identifier("test")
    book.set_title("Situated")
    book.add_author("443186283")

    pages = {
        "notes": _page("notes", "Notes", "1. Ibid., 45."),
        "intro": _page("intro", "Introduction", "Why minds are situated."),
        "ch1": _page("ch1", "CHAPTER 1", "Bodies."),
        "ch2": _page("ch2", "CHAPTER 2", "Tools."),
        "disc": _page("disc", "Discovering Context", "Not a cover page."),
    }
    # Added to the manifest out of reading order on purpose
    for key in ["notes", "ch2", "disc", "intro", "ch1"]:
        book.add_item(pages[key])
    book.spine = [pages[k] for k in ["intro", "ch1", "ch2", "disc", "notes"]]
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())

    path = tmp_path / "situated.epub"
    epub.write_epub(str(path), book)

    parsed = EPUBParser().parse(path)
    assert parsed.author is None
    titles = [c.title for c in parsed.chapters]
    assert titles == ["Introduction", "CHAPTER 1", "CHAPTER 2", "Discovering Context"]
    numbers = {c.title: c.number for c in parsed.chapters}
    assert numbers["CHAPTER 1"] == 1 and numbers["CHAPTER 2"] == 2
    assert numbers["Introduction"] == 3 and numbers["Discovering Context"] == 4

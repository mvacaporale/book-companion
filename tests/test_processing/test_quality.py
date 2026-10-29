"""Tests for the junk/back-matter heuristics."""

from book_companion.processing.quality import is_low_value_chunk, is_non_content_title

PROSE = (
    "Hunter-gatherers worked far fewer hours than we do, and they spent the rest "
    "of their time resting, socialising and playing. Suzman argues that this was "
    "not poverty but a different kind of affluence, built on wanting little and "
    "sharing what there was. The puzzle is why we abandoned it."
)

INDEX_PAGE = "\n".join([
    "habit loops, 23, 45–47",
    "habit stacking, 74–76, 112n",
    "Hebb, Donald, 201",
    "identity-based habits, 30–41",
    "implementation intentions, 69–72",
    "incentives, 150, 188–90",
    "inversion, 167",
    "Ivy League, 210",
])

ENDNOTES = "\n".join([
    "1. Marshall Sahlins, Stone Age Economics (Chicago: Aldine, 1972), p. 14.",
    "2. Richard Lee, The !Kung San (Cambridge University Press, 1979).",
    "3. Ibid., 45.",
    "4. James Suzman, Affluence Without Abundance (Bloomsbury, 2017), pp. 88-90.",
    "5. John Maynard Keynes, Essays in Persuasion (London: Macmillan, 1931).",
    "6. Lee and DeVore, Man the Hunter (Aldine, 1968), 3.",
])


def test_real_prose_is_kept() -> None:
    assert not is_low_value_chunk(PROSE, "CHAPTER 12")


def test_index_page_is_dropped() -> None:
    assert is_low_value_chunk(INDEX_PAGE)


def test_endnotes_are_dropped() -> None:
    assert is_low_value_chunk(ENDNOTES)


def test_bare_part_heading_is_dropped() -> None:
    assert is_low_value_chunk("PART TWO\nThe Fire", "Part Two")


def test_back_matter_titles_are_dropped() -> None:
    for title in ["Index", "Notes", "Endnotes", "Bibliography", "Acknowledgments", "Also by Ken Robinson"]:
        assert is_low_value_chunk(PROSE, title), title


def test_titles_are_matched_whole_not_as_substrings() -> None:
    # The old substring rules skipped these ("cover", "index", "toc").
    for title in ["Discovering Fire", "Indexing the World", "Stockpiling", "Notes on a Revolution"]:
        assert not is_non_content_title(title), title

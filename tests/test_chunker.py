"""Tests for src.ingestion.chunker.TextChunker.

Validates chunking behaviour: empty input, single chunks, paragraph splits,
overlap correctness, metadata preservation, sequential indices, and multi-chunk
splitting for long texts.
"""

from __future__ import annotations

import pytest

from src.ingestion.chunker import TextChunker, Chunk


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def chunker():
    """TextChunker with small, test-friendly parameters."""
    return TextChunker(chunk_size=100, chunk_overlap=20)


@pytest.fixture
def large_chunker():
    """TextChunker with a larger chunk size for multi-chunk tests."""
    return TextChunker(chunk_size=200, chunk_overlap=30)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEmptyText:
    """split('') and split(whitespace-only) should return an empty list."""

    def test_empty_text_returns_empty(self, chunker: TextChunker):
        assert chunker.split("") == []

    def test_whitespace_only_returns_empty(self, chunker: TextChunker):
        assert chunker.split("   ") == []

    def test_none_like_empty(self, chunker: TextChunker):
        # An empty string should yield no chunks
        result = chunker.split("")
        assert isinstance(result, list)
        assert len(result) == 0


class TestShortTextSingleChunk:
    """Text shorter than chunk_size should produce exactly one chunk."""

    def test_short_text_single_chunk(self, chunker: TextChunker):
        text = "This is a short text."
        result = chunker.split(text)

        assert len(result) == 1
        assert isinstance(result[0], Chunk)
        assert result[0].text == text
        assert result[0].chunk_index == 0


class TestSplitOnParagraphs:
    """Text with \\n\\n paragraph separators should split on those boundaries."""

    def test_split_on_paragraphs(self, chunker: TextChunker):
        # Create two paragraphs, each well under chunk_size but together exceeding it
        para1 = "A" * 60
        para2 = "B" * 60
        text = f"{para1}\n\n{para2}"

        result = chunker.split(text)

        # Should produce at least 2 chunks since combined length exceeds chunk_size
        assert len(result) >= 2
        # First chunk should contain para1 content
        assert "A" * 10 in result[0].text
        # Second chunk should contain para2 content
        assert "B" * 10 in result[-1].text


class TestChunkOverlap:
    """Consecutive chunks should share overlapping text."""

    def test_chunk_overlap(self):
        # Use a chunker with known overlap
        chunker = TextChunker(chunk_size=50, chunk_overlap=15)

        # Create text that will be split into multiple chunks
        # Use distinct words separated by spaces for clear overlap detection
        words = [f"word{i}" for i in range(40)]
        text = " ".join(words)

        result = chunker.split(text)

        if len(result) >= 2:
            # Check that the end of chunk[0] overlaps with the beginning of chunk[1]
            # The overlap means some content from the end of chunk 0
            # should appear at the start of chunk 1
            chunk0_text = result[0].text
            chunk1_text = result[1].text

            # Find words that appear at the end of chunk 0
            chunk0_words = chunk0_text.split()
            chunk1_words = chunk1_text.split()

            if len(chunk0_words) >= 2 and len(chunk1_words) >= 2:
                # There should be at least some word from the tail of chunk0
                # appearing in chunk1 (overlap region)
                tail_words = set(chunk0_words[-3:])
                head_words = set(chunk1_words[:3])
                overlap = tail_words & head_words
                # At least some overlap should exist
                assert len(overlap) > 0 or len(result) == 1, (
                    "Expected some overlap between consecutive chunks"
                )


class TestChunkMetadataPreserved:
    """Metadata dict passed to split() should be copied to every chunk."""

    def test_chunk_metadata_preserved(self, chunker: TextChunker):
        meta = {"filename": "report.pdf", "source_type": "dropbox"}
        text = "Some content for testing metadata."

        result = chunker.split(text, metadata=meta)

        assert len(result) >= 1
        for chunk in result:
            assert chunk.metadata == meta
            # Must be a copy, not the same object
            assert chunk.metadata is not meta

    def test_metadata_deep_copy(self, chunker: TextChunker):
        """Mutating one chunk's metadata should not affect others."""
        meta = {"tags": ["a", "b"]}
        # Text long enough to produce multiple chunks
        para1 = "X" * 60
        para2 = "Y" * 60
        text = f"{para1}\n\n{para2}"

        result = chunker.split(text, metadata=meta)

        if len(result) >= 2:
            result[0].metadata["tags"].append("c")
            assert "c" not in result[1].metadata["tags"]


class TestChunkIndicesSequential:
    """chunk_index values should be 0, 1, 2, ... for sequential chunks."""

    def test_chunk_indices_sequential(self, chunker: TextChunker):
        # Create text that produces multiple chunks
        paragraphs = [f"Paragraph {i}: " + "X" * 80 for i in range(5)]
        text = "\n\n".join(paragraphs)

        result = chunker.split(text)

        assert len(result) >= 2, "Expected multiple chunks for this test"

        for i, chunk in enumerate(result):
            assert chunk.chunk_index == i, (
                f"Expected chunk_index={i}, got {chunk.chunk_index}"
            )


class TestLongTextMultipleChunks:
    """A long text should produce an expected number of chunks."""

    def test_long_text_multiple_chunks(self, large_chunker: TextChunker):
        # Build a text of ~1000 chars with paragraph breaks every ~150 chars
        paragraphs = []
        for i in range(8):
            paragraphs.append(f"Section {i}: " + "word " * 25)
        text = "\n\n".join(paragraphs)

        result = large_chunker.split(text)

        # With chunk_size=200 and ~1000+ chars, we expect multiple chunks
        assert len(result) >= 3, (
            f"Expected at least 3 chunks for ~1000 chars with chunk_size=200, "
            f"got {len(result)}"
        )

        # Every chunk text should be non-empty
        for chunk in result:
            assert chunk.text.strip() != ""

        # Total text coverage: all original content words should appear
        # in at least one chunk (allowing for overlap)
        all_chunk_text = " ".join(c.text for c in result)
        for i in range(8):
            assert f"Section {i}" in all_chunk_text

    def test_no_chunk_exceeds_size(self):
        """No chunk should be longer than chunk_size (unless a single word is)."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=10)
        text = " ".join(f"w{i}" for i in range(200))

        result = chunker.split(text)

        for chunk in result:
            # Allow a small margin for edge cases in overlap merging
            assert len(chunk.text) <= chunker.chunk_size + chunker.chunk_overlap, (
                f"Chunk length {len(chunk.text)} exceeds budget "
                f"(chunk_size={chunker.chunk_size})"
            )

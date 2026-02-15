"""
Recursive character text splitter for chunking documents.

Splits text using a priority-ordered list of separators, falling back to
finer-grained splits when chunks exceed the configured size.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field

from src.config import settings

logger = logging.getLogger(__name__)

# Default separator priority (coarsest to finest)
DEFAULT_SEPARATORS: list[str] = ["\n\n", "\n", ". ", " ", ""]


@dataclass
class Chunk:
    """A single chunk of text with its index and associated metadata."""

    text: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)


class TextChunker:
    """
    RecursiveCharacterTextSplitter-style chunker.

    Splits text by trying separators in priority order. Each resulting chunk
    carries a ``chunk_index`` and a copy of the supplied metadata.

    Args:
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of overlapping characters between consecutive chunks.
        separators: Ordered list of separator strings (coarsest first).
                    An empty string ``""`` as the last element triggers
                    character-level splitting.
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        separators: list[str] | None = None,
    ) -> None:
        self.chunk_size = chunk_size if chunk_size is not None else settings.chunk_size
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
        self.separators = separators if separators is not None else list(DEFAULT_SEPARATORS)

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than "
                f"chunk_size ({self.chunk_size})"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def split(self, text: str, metadata: dict | None = None) -> list[Chunk]:
        """
        Split *text* into a list of :class:`Chunk` objects.

        Args:
            text: The text to split.
            metadata: Optional metadata dict to attach to every chunk.
                      Each chunk receives its own copy of this dict.

        Returns:
            List of Chunk objects.  Empty list if *text* is empty or blank.
        """
        if not text or not text.strip():
            return []

        base_meta = metadata or {}
        raw_chunks = self._recursive_split(text, self.separators)
        merged = self._merge_with_overlap(raw_chunks)

        chunks: list[Chunk] = []
        for idx, chunk_text in enumerate(merged):
            stripped = chunk_text.strip()
            if not stripped:
                continue
            chunks.append(
                Chunk(
                    text=stripped,
                    chunk_index=idx,
                    metadata=copy.deepcopy(base_meta),
                )
            )

        # Re-index after filtering out empty chunks
        for i, chunk in enumerate(chunks):
            chunk.chunk_index = i

        return chunks

    # ------------------------------------------------------------------
    # Internal splitting logic
    # ------------------------------------------------------------------

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        """
        Recursively split *text* using the first effective separator.

        A separator is *effective* if it actually occurs in *text*.  When no
        separator produces sub-strings short enough, we fall through to the
        next (finer) separator.  An empty-string separator triggers
        character-level splitting as a last resort.
        """
        if len(text) <= self.chunk_size:
            return [text]

        # Find the best (coarsest) separator that exists in the text
        chosen_sep: str | None = None
        remaining_seps: list[str] = []

        for i, sep in enumerate(separators):
            if sep == "":
                # Character-level: always works
                chosen_sep = sep
                remaining_seps = []
                break
            if sep in text:
                chosen_sep = sep
                remaining_seps = separators[i + 1 :]
                break

        if chosen_sep is None:
            # No separator found at all -- return the text as-is
            return [text]

        # Split with the chosen separator
        if chosen_sep == "":
            # Character-level splitting
            pieces = [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]
        else:
            pieces = text.split(chosen_sep)

        # Merge small pieces and recursively split oversized ones
        result: list[str] = []
        current_buffer = ""

        for piece in pieces:
            # Build a candidate by appending this piece to the buffer
            if current_buffer:
                candidate = current_buffer + chosen_sep + piece
            else:
                candidate = piece

            if len(candidate) <= self.chunk_size:
                current_buffer = candidate
            else:
                # Flush the buffer if it has content
                if current_buffer:
                    result.append(current_buffer)
                    current_buffer = ""

                if len(piece) <= self.chunk_size:
                    current_buffer = piece
                elif remaining_seps:
                    # Piece is too large -- recurse with finer separators
                    sub_chunks = self._recursive_split(piece, remaining_seps)
                    result.extend(sub_chunks)
                else:
                    # No finer separator available -- just add as-is
                    result.append(piece)

        if current_buffer:
            result.append(current_buffer)

        return result

    def _merge_with_overlap(self, chunks: list[str]) -> list[str]:
        """
        Apply ``chunk_overlap`` between consecutive chunks.

        For each chunk after the first, prepend the last *overlap* characters
        from the previous chunk (trying to break at a word boundary within
        the overlap region for readability).
        """
        if not chunks or self.chunk_overlap <= 0:
            return chunks

        merged: list[str] = [chunks[0]]

        for i in range(1, len(chunks)):
            prev = chunks[i - 1]
            curr = chunks[i]

            # Determine overlap text from the end of the previous chunk
            overlap_text = prev[-self.chunk_overlap :] if len(prev) > self.chunk_overlap else prev

            # Try to start the overlap at a word boundary (space) for cleanliness
            space_idx = overlap_text.find(" ")
            if space_idx != -1 and space_idx < len(overlap_text) - 1:
                overlap_text = overlap_text[space_idx + 1 :]

            # Only prepend overlap if the combined length stays within budget
            combined = overlap_text + curr
            if len(combined.strip()) <= self.chunk_size:
                merged.append(combined)
            else:
                # If overlap would push us over the limit, skip it
                merged.append(curr)

        return merged

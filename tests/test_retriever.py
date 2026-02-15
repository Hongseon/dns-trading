"""Tests for src.rag.retriever.Retriever helper methods.

The Supabase client and Embedder are fully mocked so no external services or
API keys are required.  Only the formatting and source-extraction logic is
tested here -- the actual vector search round-trip is covered by integration
tests.
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def retriever():
    """Create a Retriever with mocked Embedder and Supabase client."""
    with patch("src.rag.retriever.Embedder") as MockEmbedder, \
         patch("src.rag.retriever.get_client") as mock_get_client:
        # Mock the embedder
        mock_embedder_instance = MagicMock()
        mock_embedder_instance.embed.return_value = [0.0] * 768
        MockEmbedder.return_value = mock_embedder_instance

        # Mock the Supabase client
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        from src.rag.retriever import Retriever
        r = Retriever()

        yield r


def _dropbox_doc(
    filename: str = "A사_계약서.pdf",
    folder_path: str = "/계약서",
    content: str = "납품 기한은 2025년 3월 15일입니다.",
    similarity: float = 0.92,
    created_date: str = "2025-02-10T09:00:00+09:00",
) -> dict:
    """Build a mock Dropbox document result dict."""
    return {
        "id": 1,
        "source_type": "dropbox",
        "content": content,
        "filename": filename,
        "folder_path": folder_path,
        "email_subject": None,
        "email_from": None,
        "created_date": created_date,
        "similarity": similarity,
    }


def _email_doc(
    email_subject: str = "2월 정기회의 안건",
    email_from: str = "kim@company.com",
    content: str = "회의 날짜: 2/20(목) 14:00 본사 3층",
    similarity: float = 0.88,
    created_date: str = "2025-02-10T10:30:00+09:00",
) -> dict:
    """Build a mock email document result dict."""
    return {
        "id": 2,
        "source_type": "email",
        "content": content,
        "filename": None,
        "folder_path": None,
        "email_subject": email_subject,
        "email_from": email_from,
        "created_date": created_date,
        "similarity": similarity,
    }


# ---------------------------------------------------------------------------
# Tests -- format_context
# ---------------------------------------------------------------------------


class TestFormatContextWithFile:
    """format_context should produce a labelled entry for Dropbox documents."""

    def test_format_context_with_file(self, retriever):
        doc = _dropbox_doc()
        result = retriever.format_context([doc])

        # Should contain the document number
        assert "[문서 1]" in result
        # Should contain the filename label
        assert "[파일: A사_계약서.pdf]" in result
        # Should contain the similarity score
        assert "0.92" in result
        # Should contain the document content
        assert "납품 기한은 2025년 3월 15일입니다." in result


class TestFormatContextWithEmail:
    """format_context should produce a labelled entry for email documents."""

    def test_format_context_with_email(self, retriever):
        doc = _email_doc()
        result = retriever.format_context([doc])

        assert "[문서 1]" in result
        # Should show email label with subject and sender
        assert "[이메일: 2월 정기회의 안건 - kim@company.com]" in result
        assert "0.88" in result
        assert "회의 날짜" in result


class TestFormatContextEmpty:
    """format_context with an empty results list should return an empty string."""

    def test_format_context_empty(self, retriever):
        result = retriever.format_context([])
        assert result == ""


class TestFormatContextMultiple:
    """format_context with multiple results should number them sequentially."""

    def test_format_context_multiple(self, retriever):
        docs = [_dropbox_doc(), _email_doc()]
        result = retriever.format_context(docs)

        assert "[문서 1]" in result
        assert "[문서 2]" in result
        assert "[파일: A사_계약서.pdf]" in result
        assert "[이메일: 2월 정기회의 안건" in result


# ---------------------------------------------------------------------------
# Tests -- extract_sources
# ---------------------------------------------------------------------------


class TestExtractSourcesDeduplication:
    """Same filename appearing twice should produce only one source entry."""

    def test_extract_sources_deduplication(self, retriever):
        doc1 = _dropbox_doc(filename="report.pdf", folder_path="/docs")
        doc2 = _dropbox_doc(
            filename="report.pdf",
            folder_path="/docs",
            content="Another chunk from the same file.",
            similarity=0.85,
        )

        sources = retriever.extract_sources([doc1, doc2])

        # The same file should appear only once
        assert len(sources) == 1
        assert "report.pdf" in sources[0]


class TestExtractSourcesMixed:
    """Both Dropbox file and email results should be correctly formatted."""

    def test_extract_sources_mixed(self, retriever):
        docs = [
            _dropbox_doc(filename="계약서.pdf", folder_path="/계약"),
            _email_doc(email_subject="회의 안건", created_date="2025-02-10T00:00:00+00:00"),
        ]

        sources = retriever.extract_sources(docs)

        assert len(sources) == 2

        # Check Dropbox source format
        dropbox_sources = [s for s in sources if "Dropbox" in s]
        assert len(dropbox_sources) == 1
        assert "계약서.pdf" in dropbox_sources[0]
        assert "계약" in dropbox_sources[0]  # folder path should appear

        # Check email source format
        email_sources = [s for s in sources if "이메일" in s]
        assert len(email_sources) == 1
        assert "회의 안건" in email_sources[0]
        assert "2025-02-10" in email_sources[0]


class TestExtractSourcesEmpty:
    """extract_sources with no results should return an empty list."""

    def test_extract_sources_empty(self, retriever):
        sources = retriever.extract_sources([])
        assert sources == []


class TestExtractSourcesEmailDateFormatting:
    """Email sources should format dates as YYYY-MM-DD."""

    def test_extract_sources_email_date(self, retriever):
        doc = _email_doc(created_date="2025-03-15T14:30:00+09:00")
        sources = retriever.extract_sources([doc])

        assert len(sources) == 1
        assert "2025-03-15" in sources[0]


class TestExtractSourcesNoDate:
    """Email source with None created_date should handle gracefully."""

    def test_extract_sources_no_date(self, retriever):
        doc = _email_doc(created_date=None)
        sources = retriever.extract_sources([doc])

        assert len(sources) == 1
        # Should contain some fallback text for the date
        assert "날짜 없음" in sources[0]


class TestExtractSourcesDropboxNoFolder:
    """Dropbox source with no folder_path should still produce a valid citation."""

    def test_extract_sources_no_folder(self, retriever):
        doc = _dropbox_doc(filename="standalone.pdf", folder_path="")
        sources = retriever.extract_sources([doc])

        assert len(sources) == 1
        assert "Dropbox/standalone.pdf" in sources[0]

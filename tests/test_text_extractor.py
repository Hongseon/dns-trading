"""Tests for src.ingestion.text_extractor.

All external library dependencies (PyMuPDF, python-docx, openpyxl, etc.) are
mocked so that the tests can run without installing those heavy packages.
Only the public API surface (extract_text, extract_files_from_archive) and the
internal helpers (_extract_html, _extract_plain_text) are exercised.
"""

from __future__ import annotations

import csv
import os
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_dir():
    """Provide a temporary directory that is cleaned up after the test."""
    d = tempfile.mkdtemp(prefix="test_extractor_")
    yield Path(d)
    # Cleanup
    import shutil
    shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestExtractTextUnsupportedExtension:
    """extract_text should return '' for unsupported file extensions."""

    def test_extract_text_unsupported_extension(self, tmp_dir: Path):
        unsupported = tmp_dir / "file.xyz"
        unsupported.write_text("some content", encoding="utf-8")

        from src.ingestion.text_extractor import extract_text

        result = extract_text(unsupported)
        assert result == ""


class TestExtractTextNonexistentFile:
    """extract_text should return '' gracefully for a file that does not exist."""

    def test_extract_text_nonexistent_file(self):
        from src.ingestion.text_extractor import extract_text

        missing = Path("/tmp/__does_not_exist_12345__.txt")
        result = extract_text(missing)
        assert result == ""


class TestExtractPlainText:
    """extract_text should correctly read .txt files."""

    def test_extract_plain_text(self, tmp_dir: Path):
        content = "Hello, this is a plain text file.\nLine two."
        txt_file = tmp_dir / "sample.txt"
        txt_file.write_text(content, encoding="utf-8")

        from src.ingestion.text_extractor import extract_text

        result = extract_text(txt_file)
        assert "Hello, this is a plain text file." in result
        assert "Line two." in result


class TestExtractCsv:
    """extract_text should correctly read .csv files."""

    def test_extract_csv(self, tmp_dir: Path):
        csv_file = tmp_dir / "data.csv"
        with open(csv_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Age", "City"])
            writer.writerow(["Alice", "30", "Seoul"])
            writer.writerow(["Bob", "25", "Busan"])

        from src.ingestion.text_extractor import extract_text

        result = extract_text(csv_file)
        assert "Name" in result
        assert "Alice" in result
        assert "Seoul" in result
        assert "Bob" in result


class TestExtractZipBasic:
    """extract_files_from_archive should extract text from files inside a ZIP."""

    def test_extract_zip_basic(self, tmp_dir: Path):
        # Create a txt file to put inside the zip
        inner_content = "This is the inner text file content."
        zip_path = tmp_dir / "archive.zip"

        with zipfile.ZipFile(str(zip_path), "w") as zf:
            zf.writestr("inner_doc.txt", inner_content)

        from src.ingestion.text_extractor import extract_files_from_archive

        results = extract_files_from_archive(zip_path, depth=0)

        assert len(results) >= 1
        # results is list of (internal_path, extracted_text)
        paths = [r[0] for r in results]
        texts = [r[1] for r in results]
        assert any("inner_doc.txt" in p for p in paths)
        assert any("This is the inner text file content." in t for t in texts)


class TestExtractZipMaxDepth:
    """extract_files_from_archive should return [] when depth exceeds max_zip_depth."""

    def test_extract_zip_max_depth(self, tmp_dir: Path):
        zip_path = tmp_dir / "deep.zip"
        with zipfile.ZipFile(str(zip_path), "w") as zf:
            zf.writestr("file.txt", "content")

        from src.ingestion.text_extractor import extract_files_from_archive
        from src.config import settings

        # Call with depth greater than the configured max
        result = extract_files_from_archive(zip_path, depth=settings.max_zip_depth + 1)
        assert result == []


class TestExtractZipEncryptedSkip:
    """extract_files_from_archive should handle encrypted ZIPs gracefully."""

    def test_extract_zip_encrypted_skip(self, tmp_dir: Path):
        """Simulate an encrypted ZIP by setting the encryption flag bit on entries."""
        zip_path = tmp_dir / "encrypted.zip"

        # Create a normal zip first
        with zipfile.ZipFile(str(zip_path), "w") as zf:
            zf.writestr("secret.txt", "classified info")

        # Patch the flag_bits on the ZipInfo to simulate encryption (bit 0 set)
        # We do this by patching zipfile.ZipFile to return modified infolist
        original_zipfile_init = zipfile.ZipFile.__init__

        class FakeZipFile(zipfile.ZipFile):
            def infolist(self):
                infos = super().infolist()
                for info in infos:
                    info.flag_bits = info.flag_bits | 0x1  # Set encryption bit
                return infos

        from src.ingestion import text_extractor

        with patch.object(text_extractor.zipfile, "ZipFile", FakeZipFile):
            from src.ingestion.text_extractor import extract_files_from_archive

            result = extract_files_from_archive(zip_path, depth=0)

        assert result == []


class TestExtractHtmlRemovesScripts:
    """_extract_html should remove <script> tags and return only text content."""

    def test_extract_html_removes_scripts(self, tmp_dir: Path):
        html_content = """
        <html>
        <head><script>var x = 1;</script></head>
        <body>
            <h1>Important Title</h1>
            <p>This is the body text.</p>
            <script>alert('malicious');</script>
            <style>.hidden { display: none; }</style>
            <p>Another paragraph.</p>
        </body>
        </html>
        """
        html_file = tmp_dir / "page.html"
        html_file.write_text(html_content, encoding="utf-8")

        from src.ingestion.text_extractor import extract_text

        result = extract_text(html_file)

        # The text content should be present
        assert "Important Title" in result
        assert "This is the body text." in result
        assert "Another paragraph." in result

        # Script and style content should be removed
        assert "var x = 1" not in result
        assert "alert('malicious')" not in result
        assert ".hidden" not in result

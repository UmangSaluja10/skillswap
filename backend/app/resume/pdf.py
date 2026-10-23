"""PDF -> plain text, with clear errors the app can show to the student."""
from __future__ import annotations

import re
from pathlib import Path

import pymupdf as fitz  # PyMuPDF

MAX_BYTES = 5 * 1024 * 1024
MAX_PAGES = 10
MIN_TEXT_CHARS = 150  # less than this almost always means a scanned/image-only PDF


class ResumeError(Exception):
    """Base class. `str(e)` is safe to show to the user."""


class ScannedPdfError(ResumeError):
    pass


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"(?<=[A-Za-z])-[ \t]*\n[ \t]*(?=[a-z])", "", text)  # "Learn-\ning" -> "Learning"
    text = re.sub(r"[ \t ]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _page_text(page) -> str:
    """Reads a page in human order. Two-column layouts are read column by column
    (plain top-to-bottom sorting would interleave the columns line by line)."""
    width, height = page.rect.width, page.rect.height
    blocks = [(b[0], b[1], b[2], b[3], b[4]) for b in page.get_text("blocks") if b[6] == 0 and b[4].strip()]
    if not blocks:
        return ""

    def centered_header(b) -> bool:  # a centered name/contact line at the top is not a column
        return b[1] < 0.2 * height and abs((b[0] + b[2]) / 2 - width / 2) < 0.08 * width

    narrow = [b for b in blocks if (b[2] - b[0]) < 0.62 * width and not centered_header(b)]
    left = [b for b in narrow if b[0] < 0.4 * width]
    right = [b for b in narrow if b[0] >= 0.4 * width]
    chars = lambda bs: sum(len(b[4].strip()) for b in bs)  # noqa: E731
    total = chars(blocks)
    two_columns = (len(left) >= 2 and len(right) >= 2
                   and chars(left) >= 0.25 * total and chars(right) >= 0.25 * total)
    if not two_columns:
        ordered = sorted(blocks, key=lambda b: (round(b[1] / 4), b[0]))
    else:
        col_top = min(b[1] for b in left + right)
        column_ids = {id(b) for b in left + right}
        header = [b for b in blocks if id(b) not in column_ids and b[3] <= col_top + 2]
        footer = [b for b in blocks if id(b) not in column_ids and b[3] > col_top + 2]
        ordered = (sorted(header, key=lambda b: b[1]) + sorted(left, key=lambda b: b[1])
                   + sorted(right, key=lambda b: b[1]) + sorted(footer, key=lambda b: b[1]))
    return "\n".join(b[4].strip() for b in ordered)


def extract_text(source: bytes | str | Path) -> str:
    data = Path(source).read_bytes() if isinstance(source, (str, Path)) else source
    if not data:
        raise ResumeError("The file is empty.")
    if len(data) > MAX_BYTES:
        raise ResumeError("The PDF is larger than 5 MB. Please upload a smaller file.")
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:  # PyMuPDF raises several unrelated types
        raise ResumeError("That file could not be read as a PDF.") from exc
    with doc:
        if doc.needs_pass:
            raise ResumeError("The PDF is password-protected. Remove the password and try again.")
        pages = [_page_text(page) for page in doc.pages(0, min(len(doc), MAX_PAGES))]
    text = clean_text("\n\n".join(pages))
    if len(text) < MIN_TEXT_CHARS:
        raise ScannedPdfError(
            "This PDF looks like a scan or image, so no text could be read. "
            "Export a text-based PDF from your editor, or paste your resume text instead.")
    return text

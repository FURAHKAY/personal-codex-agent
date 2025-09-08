from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Tuple, List

try:
    from pypdf import PdfReader
except Exception:  # pypdf is optional at runtime if you don't load PDFs
    PdfReader = None  # type: ignore


# ---------------------------
# Text loading & normalization
# ---------------------------

def _read_text(fp: Path) -> str:
    """Read UTF-8 text with lenient error handling."""
    return fp.read_text(encoding="utf-8", errors="ignore")


def _read_pdf(fp: Path) -> str:
    """
    Extract text from a PDF (best-effort).
    Includes simple de-hyphenation and line-wrap fixes common in PDFs.
    """
    if PdfReader is None:
        print(f"[warn] pypdf not installed; skipping PDF: {fp.name}")
        return ""
    try:
        reader = PdfReader(str(fp))
        pages: List[str] = []
        for p in reader.pages:
            t = p.extract_text() or ""
            pages.append(t)
        raw = "\n".join(pages)
        return _normalize_pdf_text(raw)
    except Exception as e:
        print(f"[warn] Could not read PDF: {fp.name} ({e})")
        return ""


def _strip_front_matter(text: str) -> str:
    """
    Remove basic Markdown front matter (--- yaml ---) if present
    to avoid polluting embeddings with metadata.
    """
    return re.sub(r"(?s)\A---\n.*?\n---\n", "", text, count=1)


def _normalize_pdf_text(text: str) -> str:
    """
    PDF-specific cleanups:
    - join hyphenated line breaks: "computa-\ntion" -> "computation"
    - collapse hard line wraps where appropriate
    Then fall through to generic whitespace normalization.
    """
    # de-hyphenation when a hyphen is followed by a line break then a lowercase letter
    text = re.sub(r"-\n(?=[a-z])", "", text)
    # soften hard wraps: keep paragraph breaks, but reduce excessive single newlines
    # convert Windows newlines and tabs first
    text = text.replace("\r\n", "\n").replace("\t", " ")
    # join single newlines inside sentences (heuristic: no blank line around)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    return _normalize_whitespace(text)


def _normalize_whitespace(text: str) -> str:
    """
    Collapse weird spacing while preserving blank lines as paragraph breaks.
    """
    text = text.replace("\r\n", "\n").replace("\t", " ")
    # collapse non-breaking spaces, multiples
    text = re.sub(r"[ \u00A0]+", " ", text)
    # cap consecutive blank lines to 1
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------
# Public API
# ---------------------------

def load_texts(dirpath: str) -> Iterable[Tuple[str, str]]:
    """
    Yield (doc_id, text) from all files in dirpath.
    Supports: .txt, .md, .pdf
    Hidden files are ignored.
    """
    p = Path(dirpath)
    for fp in sorted(p.glob("*")):
        if fp.name.startswith("."):
            continue

        ext = fp.suffix.lower()
        if ext in {".txt", ".md"}:
            raw = _read_text(fp)
            raw = _strip_front_matter(raw)
            text = _normalize_whitespace(raw)
        elif ext == ".pdf":
            text = _read_pdf(fp)
        else:
            continue  # skip unsupported

        if not text:
            print(f"[warn] Skipping empty/unsupported: {fp.name}")
            continue

        yield fp.name, text


def _word_tokenize(s: str) -> List[str]:
    """Very light tokenizer: whitespace split."""
    return s.split()


def chunk(text: str, max_tokens: int = 350, overlap: int = 80) -> Iterable[str]:
    """
    Paragraph-aware chunking with token windows and overlap.
    - Pack paragraphs together up to ~max_tokens.
    - For very long paragraphs, fall back to windowed splitting with `overlap`.
    - Keep `max_tokens` & `overlap` aligned with your index/build settings.
    """
    assert max_tokens > 0 and overlap >= 0 and overlap < max_tokens, \
        "overlap must be 0 <= overlap < max_tokens"

    # Prefer paragraph boundaries when we can
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    buf: List[str] = []
    buf_len = 0

    def flush_buf():
        nonlocal buf, buf_len
        if buf:
            yield " ".join(buf).strip()
            buf = []
            buf_len = 0

    for para in paragraphs:
        toks = _word_tokenize(para)
        if not toks:
            continue

        # Huge paragraph → split into windows with overlap
        if len(toks) > max_tokens:
            # flush whatever is pending
            yield from flush_buf()
            i = 0
            while i < len(toks):
                window = toks[i: i + max_tokens]
                yield " ".join(window)
                if i + max_tokens >= len(toks):
                    break
                i += max_tokens - overlap
            continue

        # Otherwise pack paragraphs into a window
        if buf_len + len(toks) <= max_tokens:
            buf.append(para)
            buf_len += len(toks)
        else:
            # flush and start new buffer with this paragraph
            yield from flush_buf()
            buf = [para]
            buf_len = len(toks)

    # flush tail
    yield from flush_buf()


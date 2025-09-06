# import os, re
# from pathlib import Path
# from pypdf import PdfReader
#
# def load_texts(dirpath: str):
#     """Yield (doc_id, text) for .txt, .md, .pdf files."""
#     p = Path(dirpath)
#     for fp in p.glob("*"):
#         if fp.name.startswith("."):
#             continue
#         if fp.suffix.lower() in [".txt", ".md"]:
#             text = fp.read_text(errors="ignore")
#             yield fp.name, text
#         elif fp.suffix.lower() == ".pdf":
#             try:
#                 reader = PdfReader(str(fp))
#                 pages = [page.extract_text() or "" for page in reader.pages]
#                 yield fp.name, "\n".join(pages)
#             except Exception as e:
#                 print(f"[warn] Could not read {fp.name}: {e}")
#
#
# def chunk(text: str, max_tokens: int = 350, overlap: int = 80):
#     """Token-window chunking with overlap for better context recall."""
#     toks = text.split()
#     i = 0
#     while i < len(toks):
#         window = toks[i:i+max_tokens]
#         yield " ".join(window)
#         if i + max_tokens >= len(toks):
#             break
#         i += max_tokens - overlap

# rag.py
import re
from pathlib import Path
from typing import Iterable, Tuple

from pypdf import PdfReader


# ---------------------------
# Text loading & normalization
# ---------------------------

def _read_text(fp: Path) -> str:
    """Read UTF-8 text with lenient error handling."""
    return fp.read_text(encoding="utf-8", errors="ignore")

def _read_pdf(fp: Path) -> str:
    """Extract text from a PDF (best-effort)."""
    try:
        reader = PdfReader(str(fp))
        pages = []
        for p in reader.pages:
            t = p.extract_text() or ""
            pages.append(t)
        return "\n".join(pages)
    except Exception as e:
        print(f"[warn] Could not read PDF: {fp.name} ({e})")
        return ""

def _strip_front_matter(text: str) -> str:
    """
    Remove basic Markdown front matter (--- yaml ---) if present.
    Keeps things simple; avoids embedding metadata noise.
    """
    return re.sub(r"(?s)\A---\n.*?\n---\n", "", text, count=1)

def _normalize_whitespace(text: str) -> str:
    """Collapse weird spacing while preserving paragraphs."""
    # Replace Windows newlines, tabs, multiple spaces; keep blank lines
    text = text.replace("\r\n", "\n").replace("\t", " ")
    text = re.sub(r"[ \u00A0]+", " ", text)            # collapse spaces
    text = re.sub(r"\n{3,}", "\n\n", text)             # max two newlines
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
        elif ext == ".pdf":
            raw = _read_pdf(fp)
        else:
            continue  # skip unsupported

        text = _normalize_whitespace(raw)
        if not text:
            print(f"[warn] Skipping empty/unsupported: {fp.name}")
            continue

        yield fp.name, text


def _word_tokenize(s: str) -> list:
    """
    Very light tokenizer: split on whitespace.
    We keep it simple to avoid heavy deps; embedding model will still work great.
    """
    return s.split()


def chunk(text: str, max_tokens: int = 350, overlap: int = 80) -> Iterable[str]:
    """
    Token-window chunking with overlap.
    - Keeps semantic continuity across chunks
    - Resistant to long paragraphs / uneven newlines
    - max_tokens & overlap should match your embed/index assumptions
    """
    # Prefer paragraph boundaries when we can, then fall back to token windows.
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    buf = []
    buf_len = 0

    def flush_buf():
        nonlocal buf, buf_len
        if buf:
            yield " ".join(buf).strip()
            buf = []
            buf_len = 0

    # First pass: build windows that respect paragraph groups up to ~max_tokens
    for para in paragraphs:
        toks = _word_tokenize(para)
        if not toks:
            continue

        # If a single paragraph is huge, split it into windows directly.
        if len(toks) > max_tokens:
            # Flush whatever we had accumulated before splitting the long para
            yield from flush_buf()

            i = 0
            while i < len(toks):
                window = toks[i : i + max_tokens]
                yield " ".join(window)
                if i + max_tokens >= len(toks):
                    break
                i += max_tokens - overlap
            continue

        # Otherwise, try to pack paragraphs together up to max_tokens
        if buf_len + len(toks) <= max_tokens:
            buf.append(para)
            buf_len += len(toks)
        else:
            # flush current chunk
            yield from flush_buf()
            # start new buffer with this paragraph
            buf = [para]
            buf_len = len(toks)

    # Flush tail
    yield from flush_buf()

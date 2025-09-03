import os, re
from pathlib import Path
from pypdf import PdfReader

def load_texts(dirpath: str):
    """Yield (doc_id, text) for .txt, .md, .pdf files."""
    p = Path(dirpath)
    for fp in p.glob("*"):
        if fp.name.startswith("."):
            continue
        if fp.suffix.lower() in [".txt", ".md"]:
            text = fp.read_text(errors="ignore")
            yield fp.name, text
        elif fp.suffix.lower() == ".pdf":
            try:
                reader = PdfReader(str(fp))
                pages = [page.extract_text() or "" for page in reader.pages]
                yield fp.name, "\n".join(pages)
            except Exception as e:
                print(f"[warn] Could not read {fp.name}: {e}")

# def chunk(text: str, max_tokens: int = 400):
#     """Basic chunking by paragraphs/lines (rough)."""
#     text = re.sub(r"\n{3,}", "\n\n", text)
#     parts = text.split("\n\n")
#     buf, count = [], 0
#     for part in parts:
#         tokens = part.split()
#         if count + len(tokens) > max_tokens and buf:
#             yield " ".join(buf)
#             buf, count = [], 0
#         buf.append(part)
#         count += len(tokens)
#     if buf:
#         yield " ".join(buf)
def chunk(text: str, max_tokens: int = 350, overlap: int = 80):
    """Token-window chunking with overlap for better context recall."""
    toks = text.split()
    i = 0
    while i < len(toks):
        window = toks[i:i+max_tokens]
        yield " ".join(window)
        if i + max_tokens >= len(toks):
            break
        i += max_tokens - overlap
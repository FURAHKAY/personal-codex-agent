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

def chunk(text: str, max_tokens: int = 400):
    """Basic chunking by paragraphs/lines (rough)."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    parts = text.split("\n\n")
    buf, count = [], 0
    for part in parts:
        tokens = part.split()
        if count + len(tokens) > max_tokens and buf:
            yield " ".join(buf)
            buf, count = [], 0
        buf.append(part)
        count += len(tokens)
    if buf:
        yield " ".join(buf)

# import os, pickle, faiss
# from typing import List, Tuple
# from tiktoken import get_encoding
# from pypdf import PdfReader
#
# def load_texts(data_dir="data") -> List[Tuple[str,str]]:
#     """Return list of (doc_id, text)."""
#     items=[]
#     for name in os.listdir(data_dir):
#         p=os.path.join(data_dir,name)
#         if name.endswith(".pdf"):
#             text=[]
#             for page in PdfReader(p).pages:
#                 text.append(page.extract_text() or "")
#             items.append((name, "\n".join(text)))
#         elif name.endswith((".md",".txt")):
#             items.append((name, open(p,"r",encoding="utf-8").read()))
#     return items
#
# def chunk(text, max_tokens=400):
#     enc = get_encoding("cl100k_base")
#     toks = enc.encode(text)
#     chunks=[]
#     for i in range(0, len(toks), max_tokens):
#         chunk_tokens = toks[i:i+max_tokens]
#         chunks.append(enc.decode(chunk_tokens))
#     return chunks
#
# def top_k(query, k=4, client=None, embed_model=None, use_azure=False):
#     import numpy as np, pickle, faiss, os
#     index = faiss.read_index("index/faiss.index")
#     records = pickle.load(open("index/docs.pkl","rb"))
#     if use_azure:
#         vec = client.embeddings.create(input=[query], model=embed_model).data[0].embedding
#     else:
#         vec = client.embeddings.create(input=[query], model=embed_model).data[0].embedding
#     D, I = index.search(np.array([vec]).astype("float32"), k)
#     hits = [records[i] for i in I[0]]
#     return hits

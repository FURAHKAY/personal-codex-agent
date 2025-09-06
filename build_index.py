# import os, json, pickle, faiss, numpy as np
# from pathlib import Path
# import os
# from openai import OpenAI
# from rag import load_texts, chunk  # your existing helpers
#
# # --- Load config ---
# if os.path.exists(".env"):
#     from dotenv import load_dotenv
#     load_dotenv()
# EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# assert OPENAI_API_KEY, "Missing OPENAI_API_KEY in .env"
#
# client = OpenAI(api_key=OPENAI_API_KEY)
#
# # --- Collect chunks ---
# docs = load_texts("data")
# records = []  # [{doc_id, text}]
# for doc_id, text in docs:
#     for c in chunk(text, max_tokens=350, overlap=80):
#         c = c.strip()
#         if c:
#             records.append({"doc_id": doc_id, "text": c})
#
# if not records:
#     raise SystemExit("No text found under ./data. Add your CV & 2–3 docs, then retry.")
#
# # --- Embed in batches ---
# def embed_texts(texts):
#     resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
#     return [d.embedding for d in resp.data]
#
# vecs = []
# BATCH = 64
# for i in range(0, len(records), BATCH):
#     batch = [r["text"] for r in records[i:i+BATCH]]
#     vecs.extend(embed_texts(batch))
#
# xb = np.array(vecs, dtype="float32")
# dim = xb.shape[1]
#
# # --- Build FAISS index ---
# index = faiss.IndexFlatL2(dim)
# index.add(xb)
#
# # --- Persist everything (index + records + metadata) ---
# Path("index").mkdir(exist_ok=True)
# faiss.write_index(index, "index/faiss.index")
# with open("index/records.pkl", "wb") as f:
#     pickle.dump(records, f)
#
# meta = {
#     "embed_model": EMBED_MODEL,
#     "dim": int(dim),
#     "count": len(records),
#     "notes": "Built via build_index.py; app will assert model & dim match at runtime."
# }
#
# with open("index/meta.json", "w") as f:
#     json.dump(meta, f, indent=2)
#
# print(f"Indexed {len(records)} chunks with model='{EMBED_MODEL}', dim={dim}.")
# print("   Files written to ./index: faiss.index, records.pkl, meta.json")
#
# build_index.py
import os
import json
import pickle
import time
import argparse
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from rag import load_texts, chunk


def load_config():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Missing OPENAI_API_KEY in environment (.env or Streamlit secrets).")
    embed_model = os.getenv("EMBED_MODEL", "text-embedding-3-small")
    data_dir = os.getenv("DATA_DIR", "data")
    out_dir = os.getenv("INDEX_DIR", "index")
    return api_key, embed_model, data_dir, out_dir

def infer_type(name: str) -> str:
    n = name.lower()
    if any(w in n for w in ["cv", "resume", "education", "experience"]): return "cv"
    if any(w in n for w in ["project", "readme", "repo", "code"]):       return "project"
    if any(w in n for w in ["team", "values", "culture", "collaborat"]):  return "values"
    if any(w in n for w in ["about", "bio", "profile"]):                  return "about"
    return "any"

def collect_records(data_dir: str, max_tokens: int = 350, overlap: int = 80):
    records = []  # [{doc_id, text}]
    docs = load_texts("data")
    for doc_id, text in docs:
        src_type = infer_type(doc_id)
        for c in chunk(text, max_tokens=400):
            c = c.strip()
            if c:
                records.append({"doc_id": doc_id, "type": infer_type(doc_id), "text": c})
    return records


def embed_batches(client: OpenAI, model: str, texts, batch_size: int = 64, sleep: float = 0.0):
    """Return list[list[float]] of embeddings."""
    vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        # Basic retry for transient failures
        for attempt in range(3):
            try:
                resp = client.embeddings.create(model=model, input=batch)
                vecs.extend([d.embedding for d in resp.data])
                break
            except Exception as e:
                if attempt == 2:
                    raise
                time.sleep(1.5 * (attempt + 1))
    return vecs


def write_index(out_dir: str, index: faiss.Index, records, meta):
    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    faiss.write_index(index, str(out / "faiss.index"))
    with open(out / "records.pkl", "wb") as f:
        pickle.dump(records, f)
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Build FAISS index for Personal Codex Agent.")
    parser.add_argument("--rebuild", action="store_true", help="Delete existing index/ and rebuild.")
    parser.add_argument("--data", default=None, help="Override data directory (default: env DATA_DIR or ./data).")
    parser.add_argument("--batch", type=int, default=64, help="Embedding batch size (default: 64).")
    args = parser.parse_args()

    api_key, embed_model, data_dir, out_dir = load_config()
    if args.data:
        data_dir = args.data

    if args.rebuild and Path(out_dir).exists():
        for fp in Path(out_dir).glob("*"):
            try:
                fp.unlink()
            except Exception:
                pass

    print(f"Data directory: {data_dir}")
    print(f"Embed model:   {embed_model}")
    print("Collecting chunks …")

    records = collect_records(data_dir, max_tokens=350, overlap=80)
    if not records:
        raise SystemExit("No text found under ./data. Add your CV & 2–3 docs, then retry.")

    texts = [r["text"] for r in records]
    print(f"Collected {len(records)} chunks.")

    print("Embedding …")
    client = OpenAI(api_key=api_key)
    vecs = embed_batches(client, embed_model, texts, batch_size=args.batch)

    xb = np.array(vecs, dtype="float32")
    if xb.ndim != 2 or xb.shape[0] != len(records):
        raise SystemExit(f"Embedding shape mismatch: got {xb.shape}, expected ({len(records)}, dim)")

    dim = xb.shape[1]
    print(f"Embedding dim: {dim}")

    print("Building FAISS (IndexFlatL2) …")
    index = faiss.IndexFlatL2(dim)
    index.add(xb)

    meta = {
        "embed_model": embed_model,
        "dim": int(dim),
        "count": len(records),
        "data_dir": data_dir,
        "overlap": 80,
        "max_tokens": 350,
    }

    print(f"Writing files to ./{out_dir}")
    write_index(out_dir, index, records, meta)

    print("Done.")
    print("index/faiss.index")
    print("index/records.pkl")
    print("index/meta.json")


if __name__ == "__main__":
    main()

from __future__ import annotations

import os, json, pickle, time, argparse
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from rag import load_texts, chunk

# ---------------------------
# Config & helpers
# ---------------------------

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
    if any(w in n for w in ["cv","resume","education","experience"]): return "cv"
    if any(w in n for w in ["project","readme","repo","code"]):       return "project"
    if any(w in n for w in ["team","values","culture","collaborat"]): return "values"
    if any(w in n for w in ["about","bio","profile"]):                 return "about"
    return "any"

def collect_records(data_dir: str, max_tokens: int = 350, overlap: int = 80):
    records = []
    for doc_id, text in load_texts(data_dir):
        dtype = infer_type(doc_id)
        for c in chunk(text, max_tokens=max_tokens, overlap=overlap):
            c = c.strip()
            if c:
                records.append({"doc_id": doc_id, "type": dtype, "text": c})
    return records

def embed_batches(client: OpenAI, model: str, texts, batch_size: int = 64, pause: float = 0.0):
    vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        for attempt in range(3):
            try:
                resp = client.embeddings.create(model=model, input=batch)
                vecs.extend([d.embedding for d in resp.data])
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(1.5 * (attempt + 1) if pause == 0.0 else pause)
    return vecs

def write_index(out_dir: str, index: faiss.Index, records, meta):
    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    faiss.write_index(index, str(out / "faiss.index"))
    with open(out / "records.pkl", "wb") as f:
        pickle.dump(records, f)
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

# ---------------------------
# CLI
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Build FAISS index for Personal Codex Agent.")
    parser.add_argument("--rebuild", action="store_true", help="Delete existing index/ and rebuild.")
    parser.add_argument("--data", default=None, help="Override data dir (default: env DATA_DIR or ./data)")
    parser.add_argument("--out", default=None, help="Override output dir (default: env INDEX_DIR or ./index)")
    parser.add_argument("--batch", type=int, default=64, help="Embedding batch size (default: 64)")
    parser.add_argument("--max_tokens", type=int, default=350, help="Chunk size (default: 350)")
    parser.add_argument("--overlap", type=int, default=80, help="Chunk overlap (default: 80)")
    args = parser.parse_args()

    api_key, embed_model, env_data_dir, env_out_dir = load_config()
    data_dir = args.data or env_data_dir
    out_dir  = args.out  or env_out_dir

    if args.rebuild and Path(out_dir).exists():
        for fp in Path(out_dir).glob("*"):
            try: fp.unlink()
            except Exception: pass

    print(f"Data directory: {data_dir}")
    print(f"Output dir:     {out_dir}")
    print(f"Embed model:    {embed_model}")
    print(f"Chunking:       max_tokens={args.max_tokens}, overlap={args.overlap}")

    # Collect
    t0 = time.time()
    records = collect_records(data_dir, max_tokens=args.max_tokens, overlap=args.overlap)
    if not records:
        raise SystemExit(f"No text found under ./{data_dir}. Add your CV & a few docs, then retry.")
    print(f"Collected {len(records)} chunks in {time.time() - t0:.2f}s.")

    # Embed
    client = OpenAI(api_key=api_key)
    texts = [r["text"] for r in records]
    t1 = time.time()
    vecs = embed_batches(client, embed_model, texts, batch_size=args.batch)
    xb = np.array(vecs, dtype="float32")
    if xb.ndim != 2 or xb.shape[0] != len(records):
        raise SystemExit(f"Embedding shape mismatch: got {xb.shape}, expected ({len(records)}, dim)")
    dim = xb.shape[1]
    print(f"Embedded {len(records)} chunks @ dim={dim} in {time.time() - t1:.2f}s.")

    # Index
    t2 = time.time()
    index = faiss.IndexFlatL2(dim)
    index.add(xb)
    print(f"Built FAISS index in {time.time() - t2:.2f}s.")

    # Persist
    meta = {
        "embed_model": embed_model,
        "dim": int(dim),
        "count": len(records),
        "data_dir": data_dir,
        "overlap": args.overlap,
        "max_tokens": args.max_tokens,
    }
    write_index(out_dir, index, records, meta)
    print("Wrote files:", str(Path(out_dir) / "faiss.index"),
          str(Path(out_dir) / "records.pkl"), str(Path(out_dir) / "meta.json"))

if __name__ == "__main__":
    main()

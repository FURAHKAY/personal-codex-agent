import os, json, pickle, faiss, numpy as np
from pathlib import Path
import os
from openai import OpenAI
from rag import load_texts, chunk  # your existing helpers

# --- Load config ---
if os.path.exists(".env"):
    from dotenv import load_dotenv
    load_dotenv()
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
assert OPENAI_API_KEY, "Missing OPENAI_API_KEY in .env"

client = OpenAI(api_key=OPENAI_API_KEY)

# --- Collect chunks ---
docs = load_texts("data")
records = []  # [{doc_id, text}]
for doc_id, text in docs:
    for c in chunk(text, max_tokens=350, overlap=80):
        c = c.strip()
        if c:
            records.append({"doc_id": doc_id, "text": c})

if not records:
    raise SystemExit("No text found under ./data. Add your CV & 2–3 docs, then retry.")

# --- Embed in batches ---
def embed_texts(texts):
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]

vecs = []
BATCH = 64
for i in range(0, len(records), BATCH):
    batch = [r["text"] for r in records[i:i+BATCH]]
    vecs.extend(embed_texts(batch))

xb = np.array(vecs, dtype="float32")
dim = xb.shape[1]

# --- Build FAISS index ---
index = faiss.IndexFlatL2(dim)
index.add(xb)

# --- Persist everything (index + records + metadata) ---
Path("index").mkdir(exist_ok=True)
faiss.write_index(index, "index/faiss.index")
with open("index/records.pkl", "wb") as f:
    pickle.dump(records, f)

meta = {
    "embed_model": EMBED_MODEL,
    "dim": int(dim),
    "count": len(records),
}
with open("index/meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print(f"Indexed {len(records)} chunks with model='{EMBED_MODEL}', dim={dim}.")
print("   Files written to ./index: faiss.index, records.pkl, meta.json")


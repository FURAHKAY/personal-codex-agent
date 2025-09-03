import os, json, pickle, faiss, numpy as np, streamlit as st
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from prompts import SYSTEM_PROMPT, MODES

load_dotenv()

# --- Config from .env (one source of truth) ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL  = os.getenv("CHAT_MODEL",  "gpt-4o-mini")
assert OPENAI_API_KEY, "Missing OPENAI_API_KEY in .env"

client = OpenAI(api_key=OPENAI_API_KEY)

# --- Load index & metadata ---
IDX_DIR = Path("index")
META_FP = IDX_DIR / "meta.json"
IDX_FP  = IDX_DIR / "faiss.index"
RECS_FP = IDX_DIR / "records.pkl"

if not (META_FP.exists() and IDX_FP.exists() and RECS_FP.exists()):
    st.error("Index not found. Run `python build_index.py` first.")
    st.stop()

meta = json.loads(META_FP.read_text())
index = faiss.read_index(str(IDX_FP))
with open(RECS_FP, "rb") as f:
    records = pickle.load(f)

# --- Safety checks: model + dimension ---
if meta.get("embed_model") != EMBED_MODEL:
    st.error(
        f"Embed model mismatch:\n"
        f" • Index built with: {meta.get('embed_model')}\n"
        f" • App configured as: {EMBED_MODEL}\n\n"
        f"Fix: set EMBED_MODEL in .env to '{meta.get('embed_model')}' or rebuild the index."
    )
    st.stop()

if index.d != int(meta.get("dim", -1)):
    st.error(
        f"Embedding dimension mismatch (index.d={index.d}, meta.dim={meta.get('dim')}).\n"
        f"Rebuild the index: `rm -rf index && python build_index.py`."
    )
    st.stop()

# --- RAG helpers ---
def embed_one(text: str) -> np.ndarray:
    v = client.embeddings.create(model=EMBED_MODEL, input=[text]).data[0].embedding
    return np.array(v, dtype="float32")[None, :]

def retrieve(query: str, k: int = 4):
    qv = embed_one(query)
    D, I = index.search(qv, k)
    out = []
    for rank, idx in enumerate(I[0]):
        if idx == -1:
            continue
        out.append({"rank": rank+1, "score": float(D[0][rank]), **records[idx]})
    return out

def format_citations(ctx_items):
    return " | ".join(f"[{c['doc_id']}#{c['rank']}]" for c in ctx_items)

def answer(query: str, ctx_text: str, mode_instr: str = "") -> str:
    sys = SYSTEM_PROMPT + ("\n\nMode instructions: " + mode_instr if mode_instr else "")
    msgs = [
        {"role": "system", "content": sys},
        {
            "role": "user",
            "content": f"Context:\n{ctx_text}\n\nQuestion: {query}\n\n"
                       f"Instructions: Cite sources inline like [doc#rank] when appropriate."
        },
    ]
    resp = client.chat.completions.create(model=CHAT_MODEL, messages=msgs)
    return resp.choices[0].message.content


# --- UI ---
st.set_page_config(page_title="Personal Codex Agent", page_icon="🗂️")
st.title("Personal Codex Agent")
st.caption(f"Embed model: `{EMBED_MODEL}` · Chat model: `{CHAT_MODEL}` · Chunks: {meta.get('count')}")

col1, col2 = st.columns([2,1])
with col1:
    query = st.text_input("Ask about me:", placeholder="e.g., What kind of engineer am I?")
with col2:
    mode = st.selectbox("Mode", options=list(MODES.keys()))
k = st.slider("How many chunks to retrieve", min_value=2, max_value=8, value=4, step=1)

if st.button("Ask") and query.strip():
    with st.spinner("Retrieving…"):
        ctx_items = retrieve(query, k=k)
    if not ctx_items:
        st.warning("No relevant context found.")
    else:
        ctx_text = "\n\n---\n\n".join(f"[{c['doc_id']}#{c['rank']}] {c['text']}" for c in ctx_items)
        mode_instr = MODES.get(mode, "")
        st.subheader("Answer")
        with st.spinner("Thinking…"):
            out = answer(query, ctx_text, mode_instr=mode_instr)
        st.write(out)

        st.caption("Sources: " + format_citations(ctx_items))

        with st.expander("View retrieved context"):
            for c in ctx_items:
                st.markdown(f"**{c['rank']}. {c['doc_id']}**  \nScore: `{c['score']:.4f}`\n\n{c['text']}")

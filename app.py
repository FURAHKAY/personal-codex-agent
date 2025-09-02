# app.py
import os, json, pickle, faiss, numpy as np, streamlit as st
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

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

def answer(query: str, ctx_text: str) -> str:
    sys = (
        "You are a helpful, honest assistant that answers questions about the candidate "
        "based ONLY on the provided context. If unsure, say you don't know."
    )
    msgs = [
        {"role": "system", "content": sys},
        {"role": "user", "content": f"Context:\n{ctx_text}\n\nQuestion: {query}"},
    ]
    resp = client.chat.completions.create(model=CHAT_MODEL, messages=msgs)
    return resp.choices[0].message.content

# --- UI ---
st.set_page_config(page_title="Personal Codex Agent", page_icon="🗂️")
st.title("Personal Codex Agent")
st.caption(f"Embed model: `{EMBED_MODEL}` · Chat model: `{CHAT_MODEL}` · Chunks: {meta.get('count')}")

query = st.text_input("Ask about me:", placeholder="e.g., What kind of engineer am I?")
if st.button("Ask") and query.strip():
    with st.spinner("Retrieving…"):
        ctx_items = retrieve(query, k=4)
    if not ctx_items:
        st.warning("No relevant context found.")
    else:
        ctx_text = "\n\n---\n\n".join(f"[{c['doc_id']}] {c['text']}" for c in ctx_items)
        st.subheader("Answer")
        with st.spinner("Thinking…"):
            out = answer(query, ctx_text)
        st.write(out)

        with st.expander("View retrieved context"):
            for c in ctx_items:
                st.markdown(f"**{c['rank']}. {c['doc_id']}**\n\n{c['text']}")

# import os
# import streamlit as st
# from dotenv import load_dotenv
# from prompts import INTERVIEW_STYLE, STORY_STYLE, FAST_FACTS_STYLE
# from rag import top_k
# from openai import OpenAI
#
# load_dotenv()
#
# use_azure = bool(os.getenv("AZURE_OPENAI_API_KEY"))
# if use_azure:
#     from openai import AzureOpenAI
#     client = AzureOpenAI(
#         api_key=os.getenv("AZURE_OPENAI_API_KEY"),
#         api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
#         azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
#     )
#     CHAT_MODEL = os.getenv("AZURE_OPENAI_DEPLOYMENT")
#     EMBED_MODEL = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT")
# else:
#     client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
#     CHAT_MODEL = os.getenv("MODEL", "gpt-4o-mini")
#     EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
#
# st.set_page_config(page_title="Furaha Codex", page_icon="✨")
# st.title("Furaha — Personal Codex Agent")
#
# mode = st.selectbox("Answer style", ["Interview", "Story", "Fast Facts"])
# use_rag = st.toggle("Use my documents (RAG)", value=True)
#
# style_map = {
#     "Interview": INTERVIEW_STYLE,
#     "Story": STORY_STYLE,
#     "Fast Facts": FAST_FACTS_STYLE
# }
#
# if "messages" not in st.session_state:
#     st.session_state.messages = []
#
# for m in st.session_state.messages:
#     with st.chat_message(m["role"]):
#         st.markdown(m["content"])
#
# q = st.chat_input("Ask about me… (e.g., “What are your strongest skills?”)")
# if q:
#     st.session_state.messages.append({"role":"user","content":q})
#
#     # Build system prompt
#     system = style_map[mode] + "\nOnly speak as me (Furaha). Keep answers truthful and grounded."
#
#     # Optional RAG
#     rag_context = ""
#     sources = []
#     if use_rag:
#         hits = top_k(q, k=4, client=client, embed_model=EMBED_MODEL, use_azure=use_azure)
#         for h in hits:
#             rag_context += f"\n[Source:{h['doc_id']}] {h['text']}"
#             sources.append(h["doc_id"])
#         system += "\n\nUse the following personal snippets if relevant:\n" + rag_context
#
#     with st.chat_message("assistant"):
#         stream = client.chat.completions.create(
#             model=CHAT_MODEL,
#             messages=[
#                 {"role":"system","content":system},
#                 *st.session_state.messages
#             ],
#             stream=True
#         )
#         full=""
#         for chunk in stream:
#             delta = chunk.choices[0].delta.content or ""
#             full += delta
#             st.write(delta, end="")
#
#     st.session_state.messages.append({"role":"assistant","content":full})
#
#     if use_rag and sources:
#         st.caption("Sources: " + ", ".join(sorted(set(sources))[:5]))

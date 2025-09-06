# import os, json, pickle, faiss, numpy as np, streamlit as st
# from pathlib import Path
# import os
# from openai import OpenAI
# from prompts import SYSTEM_PROMPT, MODES
#
# if os.path.exists(".env"):
#     from dotenv import load_dotenv
#     load_dotenv()
#
#
# # --- Config from .env (one source of truth) ---
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
# CHAT_MODEL  = os.getenv("CHAT_MODEL",  "gpt-4o-mini")
# assert OPENAI_API_KEY, "Missing OPENAI_API_KEY in .env"
#
# client = OpenAI(api_key=OPENAI_API_KEY)
#
# # --- Load index & metadata ---
# IDX_DIR = Path("index")
# META_FP = IDX_DIR / "meta.json"
# IDX_FP  = IDX_DIR / "faiss.index"
# RECS_FP = IDX_DIR / "records.pkl"
#
# if not (META_FP.exists() and IDX_FP.exists() and RECS_FP.exists()):
#     st.error("Index not found. Run `python build_index.py` first.")
#     st.stop()
#
# meta = json.loads(META_FP.read_text())
# index = faiss.read_index(str(IDX_FP))
# with open(RECS_FP, "rb") as f:
#     records = pickle.load(f)
#
# # --- Safety checks: model + dimension ---
# if meta.get("embed_model") != EMBED_MODEL:
#     st.error(
#         f"Embed model mismatch:\n"
#         f" • Index built with: {meta.get('embed_model')}\n"
#         f" • App configured as: {EMBED_MODEL}\n\n"
#         f"Fix: set EMBED_MODEL in .env to '{meta.get('embed_model')}' or rebuild the index."
#     )
#     st.stop()
#
# if index.d != int(meta.get("dim", -1)):
#     st.error(
#         f"Embedding dimension mismatch (index.d={index.d}, meta.dim={meta.get('dim')}).\n"
#         f"Rebuild the index: `rm -rf index && python build_index.py`."
#     )
#     st.stop()
#
# # --- RAG helpers ---
# def embed_one(text: str) -> np.ndarray:
#     v = client.embeddings.create(model=EMBED_MODEL, input=[text]).data[0].embedding
#     return np.array(v, dtype="float32")[None, :]
#
# def retrieve(query: str, k: int = 4):
#     qv = embed_one(query)
#     D, I = index.search(qv, k)
#     out = []
#     for rank, idx in enumerate(I[0]):
#         if idx == -1:
#             continue
#         out.append({"rank": rank+1, "score": float(D[0][rank]), **records[idx]})
#     return out
#
# def format_citations(ctx_items):
#     return " | ".join(f"[{c['doc_id']}#{c['rank']}]" for c in ctx_items)
#
# def answer(query: str, ctx_text: str, mode_instr: str = "") -> str:
#     sys = SYSTEM_PROMPT + ("\n\nMode instructions: " + mode_instr if mode_instr else "")
#     msgs = [
#         {"role": "system", "content": sys},
#         {
#             "role": "user",
#             "content": f"Context:\n{ctx_text}\n\nQuestion: {query}\n\n"
#                        f"Instructions: Cite sources inline like [doc#rank] when appropriate."
#         },
#     ]
#     resp = client.chat.completions.create(model=CHAT_MODEL, messages=msgs)
#     return resp.choices[0].message.content
#
#
# # --- UI ---
# st.set_page_config(page_title="Personal Codex Agent", page_icon="🗂️")
# st.title("Personal Codex Agent")
# st.caption(f"Embed model: `{EMBED_MODEL}` · Chat model: `{CHAT_MODEL}` · Chunks: {meta.get('count')}")
#
# col1, col2 = st.columns([2,1])
# with col1:
#     query = st.text_input("Ask about me:", placeholder="e.g., What kind of engineer am I?")
# with col2:
#     mode = st.selectbox("Mode", options=list(MODES.keys()))
# k = st.slider("How many chunks to retrieve", min_value=2, max_value=8, value=4, step=1)
#
# if st.button("Ask") and query.strip():
#     with st.spinner("Retrieving…"):
#         ctx_items = retrieve(query, k=k)
#     if not ctx_items:
#         st.warning("No relevant context found.")
#     else:
#         ctx_text = "\n\n---\n\n".join(f"[{c['doc_id']}#{c['rank']}] {c['text']}" for c in ctx_items)
#         mode_instr = MODES.get(mode, "")
#         st.subheader("Answer")
#         with st.spinner("Thinking…"):
#             out = answer(query, ctx_text, mode_instr=mode_instr)
#         st.write(out)
#
#         st.caption("Sources: " + format_citations(ctx_items))
#
#         with st.expander("View retrieved context"):
#             for c in ctx_items:
#                 st.markdown(f"**{c['rank']}. {c['doc_id']}**  \nScore: `{c['score']:.4f}`\n\n{c['text']}")

# app.py
import os, json, pickle
from pathlib import Path

import faiss
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from prompts import build_messages, MODE_INSTRUCTIONS, MODE_PARAMS

# --------------------------
# Page config (once, at top)
# --------------------------
st.set_page_config(page_title="Personal Codex Agent", page_icon="🗂️")

# --------------------------
# Env / secrets (safe local & cloud)
# --------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Only touch st.secrets if a secrets.toml exists (prevents Streamlit error locally)
home_secrets = Path.home() / ".streamlit" / "secrets.toml"
proj_secrets = Path.cwd() / ".streamlit" / "secrets.toml"
if not OPENAI_API_KEY and (home_secrets.exists() or proj_secrets.exists()):
    try:
        OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]  # on Streamlit Cloud
    except Exception:
        OPENAI_API_KEY = None

if not OPENAI_API_KEY:
    st.error("No API key found. Set OPENAI_API_KEY in .env (local) or Streamlit Secrets (cloud).")
    st.stop()

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL  = os.getenv("CHAT_MODEL",  "gpt-4o-mini")

client = OpenAI(api_key=OPENAI_API_KEY)

# --------------------------
# Load index + metadata
# --------------------------
IDX_DIR = Path("index")
META_FP = IDX_DIR / "meta.json"
IDX_FP  = IDX_DIR / "faiss.index"
RECS_FP = IDX_DIR / "records.pkl"

if not (META_FP.exists() and IDX_FP.exists() and RECS_FP.exists()):
    st.error("Index not found. Run `python build_index.py` locally, commit the `index/` folder, then redeploy.")
    st.stop()

meta = json.loads(META_FP.read_text())
index = faiss.read_index(str(IDX_FP))
with open(RECS_FP, "rb") as f:
    records = pickle.load(f)

# Safety checks
if meta.get("embed_model") != EMBED_MODEL:
    st.error(
        "Embed model mismatch.\n"
        f"• Built with: {meta.get('embed_model')}  • Configured: {EMBED_MODEL}\n"
        "Fix: set EMBED_MODEL to match or rebuild the index."
    )
    st.stop()

if index.d != int(meta.get("dim", -1)):
    st.error(
        f"Embedding dimension mismatch (index.d={index.d}, meta.dim={meta.get('dim')}). "
        "Rebuild: `rm -rf index && python build_index.py`."
    )
    st.stop()

# --------------------------
# RAG helpers
# --------------------------
def embed_one(text: str) -> np.ndarray:
    v = client.embeddings.create(model=EMBED_MODEL, input=[text]).data[0].embedding
    return np.array(v, dtype="float32")[None, :]

def guess_intent(q: str) -> str:
    ql = q.lower()
    if any(k in ql for k in ["role", "engineer", "background", "resume", "cv", "experience", "years"]):
        return "cv"
    if any(k in ql for k in ["project", "built", "repo", "code", "readme"]):
        return "project"
    if any(k in ql for k in ["team", "culture", "values", "collaborat"]):
        return "values"
    if any(k in ql for k in ["about", "who are you", "who am i", "bio"]):
        return "about"
    return "any"

def retrieve(query: str, k: int = 4):
    qv = embed_one(query)
    # over-fetch candidates to allow filtering/boosting
    over_k = max(k * 3, 8)
    D, I = index.search(qv, over_k)

    intent = guess_intent(query)
    picked = []
    for rank, idx_ in enumerate(I[0]):
        if idx_ == -1:
            continue
        rec = records[idx_]
        # keep if type matches intent, else skip
        if intent != "any" and rec.get("type") != intent:
            continue
        picked.append({"rank": len(picked) + 1, "score": float(D[0][rank]), **rec})
        if len(picked) >= k:
            break

    # fallback if filter too strict: take top-k unfiltered
    if not picked:
        for rank, idx_ in enumerate(I[0][:k]):
            if idx_ == -1:
                continue
            rec = records[idx_]
            picked.append({"rank": rank + 1, "score": float(D[0][rank]), **rec})

    return picked


def build_history_messages(history, limit=3):
    """Return alternating user/assistant turns for the last `limit` exchanges."""
    msgs = []
    for turn in history[-limit:]:
        # Keep it tight; the full history is visible in UI already.
        msgs.append({"role": "user", "content": turn["q"]})
        msgs.append({"role": "assistant", "content": turn["a"]})
    return msgs



def answer(query: str, ctx_text: str, mode: str):
    # base system + current-turn messages
    msgs = build_messages(query, ctx_text, mode=mode)

    # insert recent conversation turns after the system message
    prior = build_history_messages(st.session_state.history, limit=3)
    msgs = [msgs[0]] + prior + msgs[1:]

    params = MODE_PARAMS.get(mode, MODE_PARAMS["Interview"])
    resp = client.chat.completions.create(model=CHAT_MODEL, messages=msgs, **params)
    return resp.choices[0].message.content



# --------------------------
# UI — chat-first, persistent
# --------------------------

# 0) state

if "history" not in st.session_state:
    # each turn: {"q": str, "a": str, "mode": str, "ctx": list[dict]}
    st.session_state.history = []
if "prefill" not in st.session_state:
    st.session_state.prefill = ""
if "auto_submit" not in st.session_state:
    st.session_state.auto_submit = False

if not st.session_state.history:
    st.info("👋 I’m your Personal Codex. Ask about my skills, projects, values, or work style.")

# 1) sidebar: profile, controls, quick-asks
with st.sidebar:
    st.markdown("### Furaha Kabeya")
    st.caption("MSc ML/AI · Google DeepMind Scholar")
    st.divider()

    st.markdown("**Answer settings**")
    mode = st.selectbox(
        "Answer mode",
        options=list(MODE_INSTRUCTIONS.keys()),
        index=0,
        help="Interview (concise) · Storytelling (narrative) · Fast Facts (bullets) · Reflective (insights)"
    )
    k = st.slider("Context passages (k)", 2, 6, 4)
    show_sources = st.toggle("Show retrieved sources", True)

    st.divider()
    st.markdown("**Try these**")
    def _quick(q: str):
        st.session_state.prefill = q
        st.session_state.auto_submit = True
    st.button("What kind of engineer are you?", on_click=_quick,
              args=("What kind of engineer are you?",))
    st.button("How do you debug new problems?", on_click=_quick,
              args=("How do you debug new problems?",))
    st.button("What do you value in a team?", on_click=_quick,
              args=("What do you value in a team?",))

# 2) header
st.title("Personal Codex Agent")
st.caption(f"Embed: `{EMBED_MODEL}` · Chat: `{CHAT_MODEL}` · Chunks: {meta.get('count')}")

# 3) one QA turn and persist
def run_qa(user_query: str, k_ctx: int, mode_name: str):
    ctx_items = retrieve(user_query, k=k_ctx)
    if not ctx_items:
        st.session_state.history.append(
            {"q": user_query, "a": "_No relevant context found._", "mode": mode_name, "ctx": []}
        )
        return
    ctx_text = "\n\n---\n\n".join(f"[{c['doc_id']}] {c['text']}" for c in ctx_items)
    out = answer(user_query, ctx_text, mode=mode_name)
    st.session_state.history.append(
        {"q": user_query, "a": out, "mode": mode_name, "ctx": ctx_items}
    )

# 4) auto-submit if a sidebar quick-ask was pressed this rerun
if st.session_state.auto_submit and st.session_state.prefill.strip():
    run_qa(st.session_state.prefill.strip(), k_ctx=k, mode_name=mode)
    st.session_state.auto_submit = False
    st.session_state.prefill = ""

# 5) render full conversation so far
for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["q"])
    with st.chat_message("assistant"):
        st.write(turn["a"])
        if show_sources and turn["ctx"]:
            with st.expander("Sources for this answer"):
                for c in turn["ctx"]:
                    st.markdown(f"**{c['rank']}. {c['doc_id']}**\n\n{c['text']}")

# 6) chat input at the bottom (always present)
user_q = st.chat_input("Ask about me…")

if user_q:
    # Show the user message immediately (no waiting)
    with st.chat_message("user"):
        st.write(user_q)

    # Compute answer with spinner, then render once
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            ctx_items = retrieve(user_q, k=k)
            if ctx_items:
                ctx_text = "\n\n---\n\n".join(f"[{c['doc_id']}] {c['text']}" for c in ctx_items)
                answer_text = answer(user_q, ctx_text, mode=mode)
            else:
                answer_text = "_No relevant context found._"
                ctx_items = []

        st.write(answer_text)

        # Sources for THIS turn (if any)
        if show_sources and ctx_items:
            with st.expander("Sources for this answer"):
                for c in ctx_items:
                    st.markdown(f"**{c['rank']}. {c['doc_id']}**\n\n{c['text']}")

    # Persist into history so it appears on subsequent reruns
    st.session_state.history.append(
        {"q": user_q, "a": answer_text, "mode": mode, "ctx": ctx_items}
    )



# import os, json, pickle
# from pathlib import Path
#
# import faiss
# import numpy as np
# import streamlit as st
# from dotenv import load_dotenv
# from openai import OpenAI
#
# from prompts import build_messages, MODE_INSTRUCTIONS, MODE_PARAMS
#
# # --------------------------
# # Page config
# # --------------------------
# st.set_page_config(page_title="Personal Codex Agent", page_icon="🗂️")
#
# # --------------------------
# # Env / secrets (safe local & cloud)
# # --------------------------
# load_dotenv()
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
#
# # only touch st.secrets if a secrets.toml actually exists
# home_secrets = Path.home() / ".streamlit" / "secrets.toml"
# proj_secrets = Path.cwd() / ".streamlit" / "secrets.toml"
# if not OPENAI_API_KEY and (home_secrets.exists() or proj_secrets.exists()):
#     try:
#         OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]  # Streamlit Cloud path
#     except Exception:
#         OPENAI_API_KEY = None
#
# if not OPENAI_API_KEY:
#     st.error("No API key found. Set OPENAI_API_KEY in .env (local) or Streamlit Secrets (cloud).")
#     st.stop()
#
# EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
# CHAT_MODEL  = os.getenv("CHAT_MODEL",  "gpt-4o-mini")
#
# client = OpenAI(api_key=OPENAI_API_KEY)
#
# # --------------------------
# # Load index + metadata
# # --------------------------
# IDX_DIR = Path("index")
# META_FP = IDX_DIR / "meta.json"
# IDX_FP  = IDX_DIR / "faiss.index"
# RECS_FP = IDX_DIR / "records.pkl"
#
# if not (META_FP.exists() and IDX_FP.exists() and RECS_FP.exists()):
#     st.error("Index not found. Run `python build_index.py` locally, commit the `index/` folder, then redeploy.")
#     st.stop()
#
# meta = json.loads(META_FP.read_text())
# index = faiss.read_index(str(IDX_FP))
# with open(RECS_FP, "rb") as f:
#     records = pickle.load(f)
#
# # Safety checks
# if meta.get("embed_model") != EMBED_MODEL:
#     st.error(
#         "Embed model mismatch.\n"
#         f"• Built with: {meta.get('embed_model')}  • Configured: {EMBED_MODEL}\n"
#         "Fix: set EMBED_MODEL to match or rebuild the index."
#     )
#     st.stop()
#
# if index.d != int(meta.get("dim", -1)):
#     st.error(
#         f"Embedding dimension mismatch (index.d={index.d}, meta.dim={meta.get('dim')}). "
#         "Rebuild: `rm -rf index && python build_index.py`."
#     )
#     st.stop()
#
# # --------------------------
# # RAG helpers
# # --------------------------
# def embed_one(text: str) -> np.ndarray:
#     v = client.embeddings.create(model=EMBED_MODEL, input=[text]).data[0].embedding
#     return np.array(v, dtype="float32")[None, :]
#
# def retrieve(query: str, k: int = 4):
#     qv = embed_one(query)
#     D, I = index.search(qv, k)
#     out = []
#     for rank, idx_ in enumerate(I[0]):
#         if idx_ == -1:
#             continue
#         rec = records[idx_]
#         out.append({"rank": rank + 1, "score": float(D[0][rank]), **rec})
#     return out
#
# def answer(query: str, ctx_text: str, mode: str):
#     msgs = build_messages(query, ctx_text, mode=mode)
#     params = MODE_PARAMS.get(mode, MODE_PARAMS["Interview"])
#     resp = client.chat.completions.create(model=CHAT_MODEL, messages=msgs, **params)
#     return resp.choices[0].message.content
#
# # --------------------------
# # UI — chat-first, persistent
# # --------------------------
#
# # state
# if "history" not in st.session_state:
#     # each turn: {"q": str, "a": str, "mode": str, "ctx": list[dict]}
#     st.session_state.history = []
# if "prefill" not in st.session_state:
#     st.session_state.prefill = ""
# if "auto_submit" not in st.session_state:
#     st.session_state.auto_submit = False
#
# # sidebar: profile, controls, quick-asks
# with st.sidebar:
#     st.markdown("### Furaha Kabeya")
#     st.caption("MSc ML/AI · Google DeepMind Scholar")
#     st.divider()
#
#     st.markdown("**Answer settings**")
#     mode = st.selectbox(
#         "Answer mode",
#         options=list(MODE_INSTRUCTIONS.keys()),
#         index=0,
#         help="Interview (concise) · Storytelling (narrative) · Fast Facts (bullets) · Reflective (insights) · Humble Brag"
#     )
#     k = st.slider("Context passages (k)", 2, 6, 4)
#     show_sources = st.toggle("Show retrieved sources", True)
#
#     st.divider()
#     st.markdown("**Try these**")
#     def _quick(q: str):
#         st.session_state.prefill = q
#         st.session_state.auto_submit = True
#     st.button("What kind of engineer are you?", on_click=_quick,
#               args=("What kind of engineer are you?",))
#     st.button("How do you debug new problems?", on_click=_quick,
#               args=("How do you debug new problems?",))
#     st.button("What do you value in a team?", on_click=_quick,
#               args=("What do you value in a team?",))
#
# # header
# st.title("Personal Codex Agent")
# st.caption(f"Embed: `{EMBED_MODEL}` · Chat: `{CHAT_MODEL}` · Chunks: {meta.get('count')}")
#
# # qa runner
# def run_qa(user_query: str, k_ctx: int, mode_name: str):
#     ctx_items = retrieve(user_query, k=k_ctx)
#     if not ctx_items:
#         st.session_state.history.append(
#             {"q": user_query, "a": "_No relevant context found._", "mode": mode_name, "ctx": []}
#         )
#         return
#     ctx_text = "\n\n---\n\n".join(f"[{c['doc_id']}] {c['text']}" for c in ctx_items)
#     out = answer(user_query, ctx_text, mode=mode_name)
#     st.session_state.history.append(
#         {"q": user_query, "a": out, "mode": mode_name, "ctx": ctx_items}
#     )
#
# # auto-submit if a quick-ask was pressed this rerun
# if st.session_state.auto_submit and st.session_state.prefill.strip():
#     run_qa(st.session_state.prefill.strip(), k_ctx=k, mode_name=mode)
#     st.session_state.auto_submit = False
#     st.session_state.prefill = ""
#
# # render full conversation so far
# for turn in st.session_state.history:
#     with st.chat_message("user"):
#         st.write(turn["q"])
#     with st.chat_message("assistant"):
#         st.write(turn["a"])
#         if show_sources and turn["ctx"]:
#             with st.expander("Sources for this answer"):
#                 for c in turn["ctx"]:
#                     st.markdown(f"**{c['rank']}. {c['doc_id']}**\n\n{c['text']}")
#
# # chat input at the bottom (always present)
# user_q = st.chat_input("Ask about me…", key="chat_input", value=st.session_state.prefill or "")
# if user_q:
#     run_qa(user_q, k_ctx=k, mode_name=mode)
#     # clear any prefill now that the user has asked
#     st.session_state.prefill = ""
#
#     # render the just-added turn immediately
#     last = st.session_state.history[-1]
#     with st.chat_message("user"):
#         st.write(last["q"])
#     with st.chat_message("assistant"):
#         st.write(last["a"])
#         if show_sources and last["ctx"]:
#             with st.expander("Sources for this answer"):
#                 for c in last["ctx"]:
#                     st.markdown(f"**{c['rank']}. {c['doc_id']}**\n\n{c['text']}")

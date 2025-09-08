import os, json, pickle, re, uuid, time, datetime
from pathlib import Path

import faiss
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from prompts import build_messages, MODE_INSTRUCTIONS, MODE_PARAMS, SYSTEM_PROMPT
import json, os
from pathlib import Path

CHATS_DIR = Path(".chats")
CHATS_DIR.mkdir(exist_ok=True)

def _get_query_params():
    # Streamlit renamed experimental APIs over time; support both
    try:
        return st.query_params  # new API
    except Exception:
        return st.experimental_get_query_params()  # fallback

def _set_query_params(**kwargs):
    try:
        st.query_params.clear()
        for k, v in kwargs.items():
            st.query_params[k] = v
    except Exception:
        st.experimental_set_query_params(**kwargs)

def get_chat_id():
    qp = _get_query_params()
    cid = (qp.get("cid") or [""])[0]
    if not cid:
        cid = str(uuid.uuid4())[:8]
        _set_query_params(cid=cid)
    return cid

def chat_path(cid: str) -> Path:
    return CHATS_DIR / f"{cid}.json"

def load_chat(cid: str):
    fp = chat_path(cid)
    if fp.exists():
        try:
            obj = json.loads(fp.read_text())
        except Exception:
            obj = {"title": "New chat", "history": []}
    else:
        obj = {"title": "New chat", "history": []}

    # Ensure defaults
    if "mode" not in obj or not obj["mode"]:
        obj["mode"] = "Interview"  # default once
    if "history" not in obj:
        obj["history"] = []
    if "title" not in obj or not obj["title"]:
        obj["title"] = "New chat"
    return obj

def load_all_chats() -> dict:
    sessions = {}
    for fp in CHATS_DIR.glob("*.json"):
        cid = fp.stem
        try:
            sessions[cid] = load_chat(cid)
        except Exception:
            continue
    return sessions


def save_chat(cid: str, session_obj: dict):
    fp = chat_path(cid)
    try:
        fp.write_text(json.dumps(session_obj, ensure_ascii=False, indent=2))
    except Exception:
        pass

# --------------------------
# Page config + light CSS
# --------------------------
st.set_page_config(page_title="Personal Codex Agent", page_icon="🗂️", layout="wide")
st.markdown(
    """
    <style>
      .block-container {max-width: 920px;}
      .top-menu { position: sticky; top: 0; z-index: 12; background: rgba(255,255,255,0.85);
                  -webkit-backdrop-filter: blur(6px); backdrop-filter: blur(6px); padding: 8px 0 12px; }
      .question-pill { border: 1px solid #e5e7eb; padding: 8px 12px; border-radius: 10px;
                       background: #fff; cursor: pointer; }
      .question-pill:hover { background: #f8fafc; }
      .bubble-user, .bubble-assistant { border-radius: 14px; padding: 12px 14px; margin: 8px 0 4px 0;
                       border: 1px solid rgba(0,0,0,0.06); }
      .bubble-user { background: #f6f7f9; }
      .bubble-assistant { background: #fffef7; }
      .meta-chip { font-size: 12px; color: #6b7280; margin: 2px 0 8px 0; }
      .sources { font-size: 12px; color: #6b7280; margin-top: 6px; font-style: italic; }
      .hint { font-size: 12px; color:#6b7280; margin-top: 6px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------
# Env / secrets
# --------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

home_secrets = Path.home() / ".streamlit" / "secrets.toml"
proj_secrets = Path.cwd() / ".streamlit" / "secrets.toml"
if not OPENAI_API_KEY and (home_secrets.exists() or proj_secrets.exists()):
    try:
        OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
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

if meta.get("embed_model") != EMBED_MODEL:
    st.error(
        "Embed model mismatch.\n"
        f"• Built with: {meta.get('embed_model')}  • Configured: {EMBED_MODEL}\n"
        "Fix: set EMBED_MODEL to match or rebuild the index."
    ); st.stop()

if index.d != int(meta.get("dim", -1)):
    st.error(
        f"Embedding dimension mismatch (index.d={index.d}, meta.dim={meta.get('dim')}). "
        "Rebuild: `rm -rf index && python build_index.py`."
    ); st.stop()

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
    over_k = max(k * 3, 8)
    D, I = index.search(qv, over_k)
    intent = guess_intent(query)
    picked = []
    for rank, idx_ in enumerate(I[0]):
        if idx_ == -1: continue
        rec = records[idx_]
        if intent != "any" and rec.get("type") != intent:
            continue
        picked.append({"rank": len(picked)+1, "score": float(D[0][rank]), **rec})
        if len(picked) >= k: break
    if not picked:
        for rank, idx_ in enumerate(I[0][:k]):
            if idx_ == -1: continue
            rec = records[idx_]
            picked.append({"rank": rank+1, "score": float(D[0][rank]), **rec})
    return picked

def build_history_messages(history, limit=3):
    msgs = []
    for turn in history[-limit:]:
        msgs.append({"role": "user", "content": turn["q"]})
        msgs.append({"role": "assistant", "content": turn["a"]})
    return msgs

import re

GREET_RE = re.compile(r"^\s*(hi|hello|hey|howdy|yo|morning|evening|afternoon)\b[.!?]?\s*$", re.I)
THANKS_RE = re.compile(r"\b(thanks|thank\s*you|appreciate\s*it)\b", re.I)
HOWAREYOU_RE = re.compile(r"^(how\s+are\s+you|how’s\s+it\s+going|how\s+are\s+things)\b", re.I)
BYE_RE = re.compile(r"^\s*(bye|goodbye|see\s+you|cheers)\b", re.I)
NICE2MEET_RE = re.compile(r"(nice|pleased|great)\s+to\s+meet\s+you", re.I)

def is_smalltalk(q: str) -> dict:
    """Return a dict tag -> bool so we can compose nuanced replies."""
    ql = q.strip().lower()
    return {
        "greet": bool(GREET_RE.search(ql)),
        "thanks": bool(THANKS_RE.search(ql)),
        "howru": bool(HOWAREYOU_RE.search(ql)),
        "bye": bool(BYE_RE.search(ql)),
        "meet": bool(NICE2MEET_RE.search(ql)),
    }

def smalltalk_reply(q: str, mode: str) -> str:
    tags = is_smalltalk(q)
    parts = []

    # order matters for tone
    if tags["greet"]:
        if mode == "Storytelling":
            parts.append("Hey there! 👋 I’m Furaha — happy to chat.")
        elif mode == "Fast Facts":
            parts.append("Hi! 👋")
        elif mode == "Reflective":
            parts.append("Hello! 👋 Grateful for the chat.")
        else:
            parts.append("Hey! 👋 I’m Furaha.")

    if tags["meet"]:
        parts.append("Nice to meet you too.")

    if tags["howru"]:
        parts.append("I’m doing well — focused on building thoughtful AI products.")

    if tags["thanks"]:
        parts.append("You’re welcome!")

    if tags["bye"]:
        parts.append("Bye for now! 👋")

    # If it was only small talk, nudge them with a friendly suggestion.
    if parts:
        if mode == "Fast Facts":
            parts.append("\nTry one:\n- What kind of engineer are you?\n- How do you debug new problems?\n- What do you value in a team?")
        else:
            parts.append("Ask me about my projects, experience, or how I work if you’d like.")
        return " ".join(parts)

    # Not recognized as small talk, but still vague → gentle clarify:
    return "Happy to help! Want to ask about my experience, a project I built, or how I approach problems?"
def label_for_doc(doc_id: str) -> str:
    lid = doc_id.lower()
    if "cv" in lid: return "CV"
    if "workstyle" in lid: return "WorkStyle"
    if "about" in lid: return "AboutMe"
    if "teamvalues" in lid or "values" in lid: return "TeamValues"
    if "debug" in lid: return "Debugging"
    if "growth" in lid: return "Growth"
    return Path(doc_id).stem[:12]

def inline_citations(ctx_items):
    if not ctx_items: return ""
    tags = []
    seen = set()
    for c in ctx_items:
        lab = label_for_doc(c.get("doc_id", "Source"))
        if lab not in seen:
            tags.append(f"[{lab}]")
            seen.add(lab)
    return "  \n\n*Sources: " + " ".join(tags) + "*"

def postprocess_by_mode(text: str, mode: str, query: str = "") -> str:
    """
    Normalize the model output per mode for consistent, polished UX.
    `query` is used to craft a natural intro line (esp. Fast Facts / Reflective).
    """
    t = (text or "").strip()
    if not t:
        return t

    import re

    t = re.sub(r"\*\*(.*?)\*\*", r"\1", t)

    def _sentences(s: str):
        return [x.strip() for x in re.split(r"(?<=[.!?])\s+", s) if x.strip()]

    # ---------- FAST FACTS ----------
    if mode == "Fast Facts":
        # use existing bullets if present; otherwise split sentences
        raw_lines = [ln.strip() for ln in t.split("\n") if ln.strip()]
        bullets = [re.sub(r"^[\-\•\–]\s*", "", ln) for ln in raw_lines
                   if ln.lstrip().startswith(("-", "•", "–"))]
        if not bullets:
            bullets = _sentences(t)

        # clean: strip stray bold markers & trim to 6
        bullets = [re.sub(r"\*\*(.*?)\*\*", r"\1", b).strip() for b in bullets][:6]

        intro = (f"Here’s the quick version of **{query.rstrip(' ?!.')}**:"
                 if query else "Here’s the quick version:")
        body = "\n".join(f"- {b}" for b in bullets if b)  # use Markdown list dashes
        return f"{intro}\n\n{body}".strip()

    # ---------- INTERVIEW ----------
    if mode == "Interview":
        return " ".join(_sentences(t)[:6])

    # ---------- STORYTELLING ----------
    if mode == "Storytelling":
        sents = _sentences(t)
        if len(sents) <= 3:
            return re.sub(r"\*\*(.*?)\*\*", r"\1", " ".join(sents))
        mid = max(2, min(len(sents) - 2, 4))
        p1 = " ".join(sents[:mid])
        p2 = " ".join(sents[mid:][:5])
        # remove stray bolds
        p1 = re.sub(r"\*\*(.*?)\*\*", r"\1", p1)
        p2 = re.sub(r"\*\*(.*?)\*\*", r"\1", p2)
        if not re.search(r"(so|therefore|this (taught|reinforced)|the takeaway|as a result)\b", p2, re.I):
            p2 += " This reinforced my belief in building carefully while iterating quickly."
        return f"{p1}\n\n{p2}"

    # ---------- REFLECTIVE ----------
    if mode == "Reflective":
        sents = _sentences(t)
        para = " ".join(sents[:3])
        raw_lines = [ln.strip("•- ").strip() for ln in t.split("\n") if ln.strip()]
        if len(raw_lines) < 3:
            raw_lines = sents[3:] + sents[:3]
        pts = [re.sub(r"\*\*(.*?)\*\*", r"\1", x) for x in raw_lines][:9]

        def take(label, start):
            chunk = pts[start:start+2]
            return f"- **{label}:** " + "; ".join(chunk) if chunk else f"- **{label}:** —"

        intro = (f"On **{query.rstrip(' ?!.')}**, here’s how I think about it:"
                 if query else "Here’s how I think about it:")
        return "\n".join([intro, "", para, "", take("Energizes me", 0),
                          take("Drains me", 2), take("Growth focus", 4)]).strip()

    # ---------- HUMBLE BRAG ----------
    if mode == "Humble Brag":
        raw_lines = [ln.strip() for ln in t.split("\n") if ln.strip()]
        bullets = [re.sub(r"^[\-\•\–]\s*", "", ln) for ln in raw_lines
                   if ln.lstrip().startswith((" ", "•", "–"))]
        if not bullets:
            bullets = _sentences(t)
        bullets = [re.sub(r"\*\*(.*?)\*\*", r"\1", b).strip() for b in bullets][:5]

        shaped = []
        for b in bullets:
            if not re.search(r"(impact|result|improved|increased|reduced|shipped|launched|accuracy|latency|cost|outcomes|growth)", b, re.I):
                b += " — delivered measurable impact."
            shaped.append(b)
        return "\n".join(f" {b}" for b in shaped)

    # ---------- FALLBACK ----------
    return re.sub(r"\*\*(.*?)\*\*", r"\1", t)



def format_citations(ctx_items):
    tags, seen = [], set()
    for c in ctx_items:
        label = c.get("doc_id", "Source")
        low = label.lower()
        if "cv" in low: label = "CV"
        elif "workstyle" in low: label = "WorkStyle"
        elif "about" in low: label = "AboutMe"
        elif "debug" in low: label = "Debugging"
        elif "teamvalues" in low or "values" in low: label = "TeamValues"
        tag = f"[{label}]"
        if tag not in seen:
            tags.append(tag); seen.add(tag)
    return " ".join(tags)

def answer(query: str, ctx_text: str, mode: str, history):
    # If the user is small-talking, reply immediately (don’t require context)
    if any(is_smalltalk(query).values()):
        return smalltalk_reply(query, mode)

    # If no useful context, still try a friendly fallback (don’t stonewall)
    if not ctx_text.strip():
        return smalltalk_reply(query, mode)

    # Normal grounded answer with short history
    msgs = build_messages(query, ctx_text, mode=mode)
    prior = build_history_messages(history, limit=3)
    msgs = [msgs[0]] + prior + msgs[1:]
    params = MODE_PARAMS.get(mode, MODE_PARAMS.get("Interview", {}))
    resp = client.chat.completions.create(model=CHAT_MODEL, messages=msgs, **params)
    raw = resp.choices[0].message.content
    return postprocess_by_mode(raw, mode)

# app.py (helpers)
def restyle_last_answer(last_answer: str, target_mode: str) -> str:
    # Use the same model to rewrite the text to target mode
    sys = SYSTEM_PROMPT  # import from prompts.py or duplicate the same string
    mode_note = MODE_INSTRUCTIONS.get(target_mode, "")
    msgs = [
        {"role": "system", "content": sys},
        {"role": "user", "content": (
            "Rewrite the answer below into the requested mode.\n"
            "Keep factual content; do not invent new claims.\n"
            f"Mode: {target_mode}\n"
            f"Mode note: {mode_note}\n\n"
            f"Answer:\n{last_answer}"
        )},
    ]
    params = MODE_PARAMS.get(target_mode, {"temperature": 0.5, "max_tokens": 350})
    resp = client.chat.completions.create(model=CHAT_MODEL, messages=msgs, **params)
    raw = resp.choices[0].message.content
    return postprocess_by_mode(raw, target_mode)

# --------------------------
# Semantic search (Option C)
# --------------------------
def semantic_search(q: str, top_k: int = 6):
    if not q.strip():
        return []
    qv = embed_one(q)
    D, I = index.search(qv, top_k)
    out = []
    for rank, idx_ in enumerate(I[0]):
        if idx_ == -1: continue
        rec = records[idx_]
        out.append({"rank": rank+1, "score": float(D[0][rank]), **rec})
    return out
# --------------------------
# Multi-session state
# --------------------------
def new_session(title="New chat"):
    sid = str(uuid.uuid4())[:8]
    st.session_state.sessions[sid] = {"title": title, "history": [], "mode": "Interview"}
    st.session_state.active_session_id = sid
    _set_query_params(cid=sid)
    save_chat(sid, st.session_state.sessions[sid])


def active_session():
    sid = st.session_state.active_session_id
    return st.session_state.sessions[sid]

def rename_active_if_first_turn(q: str):
    sess = active_session()
    if sess["title"] == "New chat" and not sess["history"]:
        title = q.strip()
        if len(title) > 40: title = title[:37] + "…"
        sess["title"] = title or "New chat"

def clear_active_history():
    sess = active_session()
    sess["history"].clear()
    sess["title"] = "New chat"
    st.rerun()

def delete_active_session():
    sid = st.session_state.active_session_id
    st.session_state.sessions.pop(sid, None)
    if not st.session_state.sessions:
        new_session()
    else:
        st.session_state.active_session_id = next(iter(st.session_state.sessions.keys()))
    st.rerun()

# --- Chat persistence bootstrapping (single source of truth) ---
def prune_empty_chats(keep_cid: str):
    """Delete only truly orphaned empty chats; never touch the active one."""
    for fp in CHATS_DIR.glob("*.json"):
        cid = fp.stem
        if cid == keep_cid:
            continue  # never delete the active chat, even if empty
        try:
            obj = json.loads(fp.read_text())
        except Exception:
            # unreadable → delete
            fp.unlink(missing_ok=True)
            continue
        if (obj.get("title") in (None, "", "New chat")) and not obj.get("history"):
            fp.unlink(missing_ok=True)


# --- Chat persistence bootstrapping (single source of truth) ---
st.session_state.sessions = st.session_state.get("sessions", load_all_chats())
if not st.session_state.sessions:
    new_session()
    st.stop()

cid = get_chat_id()
if cid not in st.session_state.sessions:
    if chat_path(cid).exists():
        st.session_state.sessions[cid] = load_chat(cid)
    else:
        st.session_state.sessions[cid] = {"title": "New chat", "history": [], "mode": "Interview"}
        save_chat(cid, st.session_state.sessions[cid])

st.session_state.active_session_id = cid

# prune AFTER we know the active chat, and keep it
prune_empty_chats(keep_cid=st.session_state.active_session_id)

# reload sidebar list from disk after pruning
st.session_state.sessions = load_all_chats()
if st.session_state.active_session_id not in st.session_state.sessions:
    first = sorted(st.session_state.sessions.keys())[0]
    st.session_state.active_session_id = first
    _set_query_params(cid=first)

st.session_state.setdefault("pending_q", "")
st.session_state.setdefault("confirm_delete", False)
st.session_state.setdefault("prefill_from_search", "")

# --------------------------
# Sidebar: profile + mode + chat list (NO delete/clear here)
# --------------------------
with st.sidebar:
    st.markdown("### Furaha Kabeya")
    st.caption("MSc ML/AI · Google DeepMind Scholar")
    st.divider()

    current = active_session()  # <-- keep a distinct name, don't shadow later
    options = list(MODE_INSTRUCTIONS.keys())
    selected_mode = st.selectbox(
        "Answer mode",
        options=options,
        index=options.index(current.get("mode", "Interview")) if current.get("mode", "Interview") in options else 0,
        help="Interview (concise) · Storytelling (narrative) · Fast Facts (bullets) · Reflective (insights)"
    )
    if selected_mode != current.get("mode"):
        current["mode"] = selected_mode
        save_chat(get_chat_id(), current)

    st.divider()
    st.markdown("**Chats**")
    to_switch = None
    for sid, s_item in st.session_state.sessions.items():
        label = "• " + (s_item.get("title") or "New chat")
        if st.button(label, key=f"switch_{sid}", use_container_width=True):
            to_switch = sid
    if to_switch:
        st.session_state.active_session_id = to_switch
        _set_query_params(cid=to_switch)
        st.rerun()

    if st.button("➕ New chat", use_container_width=True, key="btn_new_chat"):
        new_session()  # creates + selects + saves + updates ?cid
        st.rerun()

    st.divider()
    st.markdown("**Search my docs**")
    ss_q = st.text_input("Find mentions of…", key="sem_srch", label_visibility="collapsed", placeholder="e.g., DeepMind")
    if ss_q:
        hits = semantic_search(ss_q, top_k=6)
        if not hits:
            st.caption("_No matches found._")
        else:
            for h in hits:
                lab = label_for_doc(h["doc_id"])
                snippet = (h["text"][:140] + "…") if len(h["text"]) > 150 else h["text"]
                if st.button(f"[{lab}] {snippet}", key=f"hit_{h['rank']}", use_container_width=True):
                    st.session_state.pending_q = ss_q
                    st.rerun()

# --------------------------
# Main: top menu + suggestions
# --------------------------

with st.container():
    st.markdown('<div class="top-menu">', unsafe_allow_html=True)
    _, _, col_del = st.columns([2, 2, 1])
    with col_del:
        if st.button("🗑️ Delete chat", type="secondary", use_container_width=True, key="delete_chat"):
            sid = st.session_state.active_session_id
            try:
                chat_path(sid).unlink(missing_ok=True)
            except Exception:
                pass
            st.session_state.sessions.pop(sid, None)

            if not st.session_state.sessions:
                # start a brand-new one with default mode
                new_session()
            else:
                # pick any remaining chat deterministically
                new_active = sorted(st.session_state.sessions.keys())[0]
                st.session_state.active_session_id = new_active
                _set_query_params(cid=new_active)

            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


st.title("Personal Codex Agent")

# --------------------------
# Render conversation
# --------------------------
sess = active_session()
if not sess["history"]:
    st.info("👋 Welcome to your Personal Codex. Ask about my skills, projects, values, or work style.")

    # suggestions row
    suggested = [
        "What kind of engineer are you?",
        "How do you debug new problems?",
        "What do you value in a team?",
    ]
    sc = st.columns(3)
    for i, q in enumerate(suggested):
        if sc[i].button(q, key=f"sugg_{i}", use_container_width=True):
            # store for this turn; we will render the user bubble immediately below
            st.session_state.pending_q = q
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


def mode_chip_and_time(turn):
    ts = turn.get("ts")
    tstr = datetime.datetime.fromtimestamp(ts).strftime("%H:%M") if ts else ""
    return f"{tstr} · {turn['mode']}" if tstr else turn["mode"]

for i, turn in enumerate(sess["history"]):
    with st.chat_message("user"):
        st.markdown(f"<div class='chat-user'>{turn['q']}</div>", unsafe_allow_html=True)
    with st.chat_message("assistant"):
        meta = mode_chip_and_time(turn)
        st.markdown(f"<div class='meta-chip'>{meta}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='chat-assistant'>{turn['a']}</div>", unsafe_allow_html=True)

        # Actions row: Retell in…  |  Save as Insight ⭐
        ac = st.columns([2.5,2.5,2.5,2.5])
        # Cross-mode reframe (Option B)
        with ac[0]:

            st.write("")  # spacer
        with ac[1]:
            st.write("")  # spacer
        with ac[2]:
            st.write("")  # spacer
        # Insight tracker (Option A)
        with ac[3]:
            if st.button("⭐ Save as Insight", key=f"sv_{i}", use_container_width=True):
                insight = f"### {turn['q']}\n\n{turn['a']}\n\n---\n"
                data_dir = Path("data")
                data_dir.mkdir(exist_ok=True)
                out_path = data_dir / "GrowthNotes.md"
                try:
                    # Try to append locally (works when running local or with write access)
                    with open(out_path, "a", encoding="utf-8") as f:
                        f.write(insight)
                    st.success("Saved to data/GrowthNotes.md")
                except Exception:
                    # Cloud fallback: let the reviewer download the insight
                    st.download_button("Download Insight.md",
                                       data=insight.encode("utf-8"),
                                       file_name="Insight.md",
                                       mime="text/markdown",
                                       use_container_width=True)

# --------------------------
# Input row (always visible)
# --------------------------
typed_q = st.chat_input("Ask about me…")
st.markdown('<div class="hint">⏎ to send · ⇧⏎ for newline</div>', unsafe_allow_html=True)

current_q = st.session_state.get("pending_q", "") or typed_q
chat_mode = active_session().get("mode", "Interview")

if current_q:
    with st.chat_message("user"):
        st.markdown(f'<div class="bubble-user">{current_q}</div>', unsafe_allow_html=True)

    # Smalltalk short-circuit so "hi" doesn't go through RAG/LLM
    if any(is_smalltalk(current_q).values()):
        answer_text = smalltalk_reply(current_q, chat_mode)
        ctx_items = []
        with st.chat_message("assistant"):
            st.markdown(f'<div class="bubble-assistant">{answer_text}</div>', unsafe_allow_html=True)
    else:
        # Normal grounded path (streaming)
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                ctx_items = retrieve(current_q, k=4)
                ctx_text = "\n\n---\n\n".join(f"[{c['doc_id']}] {c['text']}" for c in ctx_items) if ctx_items else ""

                params = MODE_PARAMS.get(chat_mode, MODE_PARAMS["Interview"])
                msgs = build_messages(current_q, ctx_text, mode=chat_mode)
                prior = build_history_messages(active_session()["history"], limit=3)
                msgs = [msgs[0]] + prior + msgs[1:]

                resp = client.chat.completions.create(
                    model=CHAT_MODEL, messages=msgs, stream=True, **params
                )

                full = ""
                placeholder = st.empty()
                for chunk in resp:
                    delta = (chunk.choices[0].delta.content or "")
                    full += delta
                    placeholder.markdown(f'<div class="bubble-assistant">{full}▌</div>', unsafe_allow_html=True)
                placeholder.markdown(f'<div class="bubble-assistant">{full}</div>', unsafe_allow_html=True)

                answer_text = postprocess_by_mode(full, chat_mode, query=current_q)

            if ctx_items:
                st.markdown(f'<div class="sources">Sources: {format_citations(ctx_items)}</div>', unsafe_allow_html=True)

    # persist
    rename_active_if_first_turn(current_q)
    sess_obj = active_session()
    sess_obj["history"].append(
        {"q": current_q, "a": answer_text, "mode": chat_mode, "ctx": ctx_items, "ts": time.time()}
    )
    save_chat(get_chat_id(), sess_obj)

    st.session_state.pending_q = ""
    st.rerun()


if active_session()["history"]:
    last = active_session()["history"][-1]
    with st.expander("Retell this answer in another mode"):
        cols = st.columns(4)
        targets = ["Fast Facts", "Storytelling", "Interview", "Humble Brag"]
        for i, m in enumerate(targets):
            if cols[i].button(m, key=f"retell_{i}"):
                retold = restyle_last_answer(last["a"], m)
                active_session()["history"].append(
                    {"q": f"(Retell last answer as {m})",
                     "a": retold, "mode": m, "ctx": last["ctx"], "ts": time.time()}
                )
                save_chat(cid, active_session())
                st.rerun()

# --------------------------
# Export current chat (Markdown)
# --------------------------
def transcript_markdown(sess_obj) -> str:
    title = sess_obj["title"]
    lines = [f"# {title}", ""]
    for t in sess_obj["history"]:
        ts = datetime.datetime.fromtimestamp(t.get("ts", time.time())).strftime("%Y-%m-%d %H:%M")
        lines.append(f"**You ({ts})**"); lines.append(t["q"]); lines.append("")
        lines.append(f"**Furaha ({t['mode']})**")
        body = t["a"]
        if t["ctx"]:
            body += f"\n\n*Sources: {format_citations(t['ctx'])}*"
        lines.append(body); lines.append("")
    return "\n".join(lines).strip()

if sess["history"]:
    md = transcript_markdown(sess)
    filename = (sess["title"] or "chat").replace(" ", "_").replace("/", "_")
    st.download_button("📥 Download Markdown", md, file_name=f"{filename}.md")



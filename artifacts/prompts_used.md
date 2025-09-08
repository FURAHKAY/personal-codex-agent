# Prompts Used (Selected)

## 1) Chunking helper
**Ask:** “Paragraph-aware chunker ~350 tokens with 80 overlap; preserve blank lines.”
**Use:** Seeded `rag.py:chunk`, later hand-tuned for long paras.

## 2) System prompt shaping
**Ask:** “System prompt that only answers about my personal background/skills/style; says I don’t know when context is thin.”
**Use:** Basis for `SYSTEM_PROMPT`; edited tone + explicit ‘what to add’.

## 3) Streamlit chat scaffold
**Ask:** “Streamlit chat with FAISS retrieval and OpenAI chat; show citations.”
**Use:** Seeded `app.py` structure; I added sessions, dark mode, chips, regen.

## 4) Mode rewrites
**Ask:** “Rewrite into Storytelling/Fast Facts while preserving facts.”
**Use:** Implemented `restyle_last_answer` with post-processing to enforce format.

## 5) Small-talk UX
**Ask:** “Detect greetings/thanks/bye and respond briefly.”
**Use:** Regex guards and friendly nudges back to work topics.

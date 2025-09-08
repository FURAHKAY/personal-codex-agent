# Personal Codex Agent

## 📌 Overview
A context-aware chatbot that answers questions **about me (Furaha)** using my own docs (CV, project READMEs, work-style notes). Retrieval via **OpenAI embeddings + FAISS**, generation via **GPT-4o-mini**, and a Streamlit UI with **modes** (Interview, Storytelling, Fast Facts, Reflective, Humble Brag).

- **Deployed app**: https://codex-agent.streamlit.app/
* **Video walkthrough**: https://drive.google.com/file/d/1yA68q3XZ-2JHP4AjI0BoO9zUOmTpt7Cq/view
- **Architecture diagram:** `artifacts/architecture.png` 

---

## Quickstart

```bash
git clone https://github.com/FURAHKAY/personal-codex-agent.git
cd personal-codex-agent
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
````

Create `.env`:

```env
OPENAI_API_KEY=sk-...
EMBED_MODEL=text-embedding-3-small
CHAT_MODEL=gpt-4o-mini
```

Build index + run:

```bash
python build_index.py
streamlit run app.py
```

---

## Features

* **RAG**: FAISS (IndexFlatL2) + `text-embedding-3-small`
* **Modes**: Interview, Storytelling, Fast Facts, Reflective, Humble Brag
* **Citations**: inline chips that expand to snippets
* **Dark theme toggle**
* **Multi-session history** with export to Markdown
* **Admin pane**: upload docs → re-index from the UI

---

## Project Structure

```
app.py                  # Streamlit UI + chat loop + retrieval + UX
build_index.py          # Index builder (embeds → FAISS + metadata)
rag.py                  # Loading/normalization + chunking
prompts.py              # System prompt + mode instructions + params
data/                   # Personal corpus (CV, AboutMe, WorkStyle, READMEs…)
index/                  # Generated FAISS + metadata (meta.json, records.pkl)
artifacts/              # Docs: architecture, eval plan, metrics, security, etc.
```

---

## Design Choices (TL;DR)

* **FAISS** for simplicity + portability (<10k chunks sweet spot).
* **Chunking** \~350 tokens with 80 overlap (paragraph-aware) for continuity.
* **Intent filters** (cv/project/values/about) to prefer relevant sources.
* **Post-processing** enforces mode formatting (bullets/length/tone).
* **Small-talk guardrails**: friendly, then steer back to work topics.

Full rationale in `artifacts/architecture.md`.
- Known issues & fixes in `artifacts/troubleshooting.md`
- Security & Privacy: No PII beyond my documents. Keys via `.env`/Streamlit Secrets, never committed. See `artifacts/security_privacy.md`.

---

## Author

Furaha Kabeya


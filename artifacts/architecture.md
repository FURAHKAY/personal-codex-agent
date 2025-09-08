
---

# artifacts/architecture.md

```markdown
# Architecture

**Goal:** answer questions about Furaha, grounded in personal docs.

## Diagram (textual)
1) User → Streamlit UI (`app.py`)
2) Query → OpenAI Embeddings (EMBED_MODEL) → FAISS search (IndexFlatL2)
3) Top-k chunks (+ intent filter) → `prompts.build_messages(...)`
4) OpenAI Chat (CHAT_MODEL) → mode post-processing → UI render
5) Source chips → expandable snippet panel
6) Admin: file upload → re-index (`build_index.py` logic mirrored in UI)

## Key Components
- `rag.py`
  - `load_texts`: .txt/.md/.pdf with whitespace normalization + YAML front-matter strip
  - `chunk`: paragraph-aware ~350 tokens, 80 overlap
- `build_index.py`
  - batches embeddings, writes `index/faiss.index`, `index/records.pkl`, `index/meta.json`
- `app.py`
  - chat sessions in `.chats/`
  - intent heuristics → prefer CV/Project/Values/About
  - small-talk fallback; streaming responses; dark theme; source chips

## Design Tradeoffs
- **Pros:** simple infra, reproducible, fast iteration
- **Cons:** no semantic reranking; no query expansion; limited metadata
- **Next:** switch to Qdrant/Pinecone with scalar + text filters; add reranker; per-doc abstractive summaries.

## Risks & Mitigations
- **Model mismatch** → assert `meta.embed_model` & dimension
- **Empty corpus** → explicit guard + UI guidance
- **Hallucination** → strict system prompt; inline source chips; “I don’t know” path

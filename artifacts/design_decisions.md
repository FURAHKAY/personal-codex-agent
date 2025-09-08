# Design Decisions

## Why FAISS (IndexFlatL2)?
- Simple, fast, reliable for small–medium corpora.
- No server dependency; easy to persist + version in GitHub.
- Alternatives considered:
  - Chroma: good dev UX; adds a DB layer I didn’t need.
  - qdrant/pinecone: great for scale; overkill for a personal codex.

## Why Streamlit?
- Fast to prototype a friendly, demoable UI.
- Built-in widgets (chat, expander, sidebar) → fewer pixels to hand-code.
- Alternatives:
  - Next.js (Vercel): more control, slower iteration for this trial.
  - Gradio: easy too, but Streamlit felt cleaner for multipane UX.

## Embeddings / Models
- Embeddings: `text-embedding-3-small` → low cost, solid semantic quality.
- Chat: `gpt-4o-mini` (configurable) → good price/perf for concise RAG answers.

## Chunking Strategy
- ~350 tokens with 80-token overlap.
- Motivation: preserve semantic continuity across sections; avoid context holes.
- PDF normalization: collapsed excessive newlines for cleaner chunks.

## Retrieval
- k = 4 by default; “over-fetch” internally then filter by intent (cv/project/values/about).
- Rationale: increases precision without heavy metadata pipelines.

## Modes → Behaviors
- Each mode binds **temperature + max_tokens** + **formatting enforcement**.
- Post-process to guarantee bullets in Fast Facts and length caps in Interview.
- Adds “Retell in X mode” to demonstrate agentic flexibility.

## Small-talk Resilience
- If retrieval irrelevant (e.g., “hi”), reply warmly and steer back to work-related queries.
- Prevents the “rigid RAG” feel the V1 assessment noted.

## Scaling Plan (If Continued)
- **Indexing:** move to Qdrant or Pinecone with metadata filters (doc_id, section, year).
- **Summarization:** hierarchical summaries for long docs; per-doc abstracts in index.
- **Grounding:** stronger citation surface (inline anchor → source panel).
- **Eval:** prompt-based regression tests; golden Q&A set; latency/error dashboards.
- **UX:** multi-file upload + incremental indexing from the UI; profile-themed skins.
# Design Decisions

- **FAISS for retrieval:** lightweight, fast for <10k docs, easy to rebuild locally.
- **Streamlit for UI:** rapid prototyping, strong widget ecosystem, easy deployment.
- **Multi-session persistence:** simple `.json` per chat in `.chats/` for auditability.
- **Why modes?** Evaluators wanted “agentic sophistication.” Modes = behaviors, not just tone.
- **Scaling path:** If dataset grows → swap FAISS → Weaviate or Pinecone. Streamlit → Next.js or Gradio for production.

# Prompts Used

This file captures some of the key prompts I used with AI assistants while building the Personal Codex Agent.  
It also shows how I edited the AI output to make it fit the project.

---

## Example 1: RAG Helper (rag.py)

**Prompt I gave AI:**
> "Write a Python function to chunk text into overlapping windows of tokens, max length 350, overlap 80."

**AI’s Response (snippet):**
```python
def chunk(text: str, max_tokens: int = 350, overlap: int = 80):
    toks = text.split()
    i = 0
    while i < len(toks):
        window = toks[i:i+max_tokens]
        yield " ".join(window)
        if i + max_tokens >= len(toks):
            break
        i += max_tokens - overlap
````

**My Edit:**

* Integrated this into `rag.py` together with my own `load_texts` function.
* Adjusted defaults (sometimes 400 tokens, sometimes 350) depending on test runs.
* Added PDF + text file loader logic manually.

---

## Example 2: System Prompt (prompt.py)

**Prompt I gave AI:**

> "Write me a system prompt that makes the agent answer only about my personal background, skills, and style."

**AI’s Response:**

```python
SYSTEM_PROMPT = """You are the personal codex of the user. 
Answer questions about the user's background, projects, skills, values and working style.
Use retrieved snippets when relevant. Keep answers grounded, concise, and in the user's voice."""
```

**My Edit:**

* Shortened phrasing for clarity.
* Considered adding “if unsure, say you don’t know” (but later moved that into app.py).

---

## Example 3: Streamlit UI (app.py)

**Prompt I gave AI:**

> "Build a simple Streamlit interface to query FAISS and call an OpenAI chat model."

**AI’s Response (snippet):**

```python
st.title("Personal Codex Agent")
query = st.text_input("Ask about me:")
if st.button("Ask"):
    ctx_items = retrieve(query, k=4)
    ...
```

**My Edit:**

* Added `st.expander` to show retrieved context.
* Added captions with model names (`Embed model: ..., Chat model: ...`).
* Added error checks for model mismatch and embedding dimensions.


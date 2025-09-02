# Prompts Used (Sample)

These are a few representative prompts I used with ChatGPT while building the project.  
They show how I collaborated with AI tools, not every single interaction.

---

**Prompt → Response → My Action**

**Prompt:**  
"How do I structure a personal codex project that uses RAG and Streamlit?"  

**AI Response:**  
Suggested a folder layout (`app.py`, `rag.py`, `build_index.py`, `data/`, `index/`) with FAISS for embeddings.  

**My Action:**  
I adopted the suggested structure but added my own `artifacts/` folder to track build notes, since Ubundi required “show your thinking”.

---

**Prompt:**  
"I’m getting a dotenv error. Why is `OPENAI_API_KEY` not loading?"  

**AI Response:**  
Told me to check if `.env` was inside the venv or system-wide, and suggested reinstalling `python-dotenv`.  

**My Action:**  
Manually reinstalled `python-dotenv` inside my venv and confirmed it worked by testing with `os.getenv`.

---

**Prompt:**  
"Why is FAISS giving `assert d == self.d` error?"  

**AI Response:**  
Explained that query embedding dimension didn’t match the index. Suggested rebuilding the index with the same embedding model.  

**My Action:**  
Rebuilt the FAISS index after correcting `AZURE_OPENAI_EMBED_DEPLOYMENT`, and confirmed that embeddings matched.

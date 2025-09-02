````markdown
# Personal Codex Agent

## 📌 Overview
This project is a **trial build for Ubundi**: a context-aware chatbot that answers questions about me as a candidate.  
It is powered by my personal documents (CV, project READMEs, and reflective notes) and serves as a lightweight first version of my **personal Codex**.

The system uses **retrieval-augmented generation (RAG)** with OpenAI embeddings + FAISS to ground answers in my data, then generates responses in my own voice.

---

## ⚙️ Setup

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/personal-codex-agent.git
cd personal-codex-agent
````

### 2. Create virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Environment variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_key_here
# or Azure
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com
AZURE_OPENAI_API_KEY=your_azure_key
AZURE_OPENAI_API_VERSION=2024-06-01
AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini-v1
AZURE_OPENAI_EMBED_DEPLOYMENT=text-embedding-3-small
```

### 4. Build the index

```bash
python build_index.py
```

### 5. Run the app

```bash
streamlit run app.py
```

---

## 🏗️ System Design

* **Data layer (`data/`)**: My CV and supporting documents (project READMEs, personal notes).
* **Indexing (`build_index.py`)**: Splits docs into chunks → embeds → stores in FAISS index.
* **Retriever (`rag.py`)**: Finds top-k relevant chunks for a query.
* **Chat interface (`app.py`)**: Streamlit UI where users can ask questions.
* **Artifacts (`artifacts/`)**: AI prompts, agent instructions, and commit notes to show my process.

---

## 🎯 Example Questions

Here are some queries the agent can answer:

* “What kind of engineer are you?”
* “What are your strongest technical skills?”
* “What projects or experiences are you most proud of?”
* “What do you value in a team or company culture?”
* “What’s your approach to learning or debugging something new?”

---

## 🔮 Future Improvements

Given more time, I would:

* Add **modes** (e.g., Interview mode vs Storytelling mode).
* Extend the dataset automatically (upload new docs on the fly).
* Improve grounding with richer metadata and cross-document linking.
* Deploy on **Vercel** or **Replit** with a public demo URL.

---

## 🧠 Show Your Thinking

See the `artifacts/` folder for:

* Prompts I used with AI coding agents.
* How I guided sub-agents (tasks delegated, rules given).
* Notes on what was AI-generated vs hand-edited.

---

## 🚀 Deployment

* Local: via `streamlit run app.py`
* Cloud: (to be updated after deployment to Streamlit Cloud)

---

## 🎥 Walkthrough Video

A short demo video will accompany submission. It will show:

* Project structure + README
* Live Q\&A with the agent
* Artifacts folder

---

## 👤 Author

Furaha Kabeya
```

---

✅ This README covers: **setup, design choices, examples, improvements, artifacts, and video** — exactly what Ubundi asked.  

Do you want me to also **auto-draft the artifacts files** (`prompts_used.md`, `agent_instructions.md`, `commit_log.txt`) based on our chat so you can just drop them in?
```

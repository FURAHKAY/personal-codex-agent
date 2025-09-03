# Personal Codex Agent

## 📌 Overview
This project is a **trial build for Ubundi**: a context-aware chatbot that answers questions about me as a candidate.  
It is powered by my personal documents (CV, project READMEs, and reflective notes) and serves as a lightweight first version of my **personal Codex**.

The system uses **retrieval-augmented generation (RAG)** with OpenAI embeddings + FAISS to ground answers in my data, then generates responses in my own voice.

## Features
- **Retrieval-Augmented Generation (RAG)** with FAISS + OpenAI embeddings
- **Personal dataset** (`/data/`) including CV, AboutMe, work style notes, and project READMEs
- **Streamlit UI** for interactive Q&A
- **Citations**: see which documents/chunks were used
- **Agentic modes** (optional stretch): interview / storytelling / fast facts / humble-brag


## Project Structure
```
personal-codex-agent/
│
├── app.py              # Streamlit UI
├── build\_index.py      # Splits docs into chunks -> Build embeddings -> stores in FAISS index. 
├── rag.py              #  Finds top-k relevant chunks for a query.
├── prompt.py           # System prompt definition
│
├── data/               # My personal documents
│   ├── Furaha kabeya-ML-CV.pdf
│   ├── AboutMe.md
│   ├── WorkStyle.txt
│   ├── README\_DiabetesPredictor.md
│   └── README\_RL\_Agent.md
│
├── index/              # Generated FAISS index + metadata
│
├── artifacts/          # "Show Your Thinking" AI build logs
│   ├── prompts\_used.md
│   ├── agent\_instructions.md
│   └── commit\_log.txt
│
├── requirements.txt
└── README.md

````
## Setup

### Clone the repo
```bash
git clone git@github.com:FURAHKAY/personal-codex-agent.git
cd personal-codex-agent
````

### Create virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Environment variables

`.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_key_here
MODEL=gpt-4o-mini
EMBED_MODEL=text-embedding-3-small

```

### Build the index

```bash
python build_index.py
```

### Run the app

```bash
streamlit run app.py
```

---

## Design Choices

* **RAG (FAISS + OpenAI embeddings)** → ensures answers are grounded in my actual documents.
* **Chunking with overlap** → balances recall vs relevance.
* **Streamlit** → quick, interactive UI.
* **Temperature = 0** → deterministic answers for consistency.
* **Personal dataset** → blends technical + reflective docs so answers feel authentic.


## Example Questions

Here are some queries the agent can answer:

* “What kind of engineer are you?”
* “What are your strongest technical skills?”
* “What projects or experiences are you most proud of?”
* “What do you value in a team or company culture?”
* “What’s your approach to learning or debugging something new?”

---

## Future Improvements

Given more time, I would:

* Improve grounding with richer metadata and cross-document linking.
* Incremental indexing (upload new docs from UI).
* Fine-grained metadata filtering (e.g., “only from CV”).
* Add **self-reflective agent mode** for growth/values questions.



## Show Your Thinking

See the `artifacts/` folder for:

* Prompts I used with AI coding agents.
* How I guided sub-agents (tasks delegated, rules given).
* Notes on what was AI-generated vs hand-edited.
---

## Demo

* **Deployed app**: \[Streamlit link here once deployed]
* **Video walkthrough**: \[5-min screen recording link here]

## 📊 Evaluation Mapping

* **Context handling** → FAISS + overlap, retrieved snippets shown
* **Agentic thinking** → modes scaffolded, expandable
* **Personal data** → CV + project READMEs + reflective notes
* **Build quality** → index checks, robust error handling
* **Voice & reflection** → first-person answers, grounded in my docs
* **Bonus effort** → citations, mode design, artifacts folder
* **AI build artifacts** → included in `/artifacts/`
* **RAG usage** → effective FAISS + OpenAI embedding pipeline

---

## 👤 Author
Furaha Kabeya

# Agent Instructions (How I guided AI)

I treated AI as a coding assistant, while I made design and content decisions.  

---

### My Role
- Selected and authored authentic personal documents (`AboutMe.md`, `WorkStyle.txt`, project READMEs).  
- Chose what content to expose as “training data” (to make sure the agent speaks in my voice).  
- Debugged environment issues (`dotenv`, Azure endpoints, rate limits).  
- Designed the system flow: document ingestion → embeddings → retrieval → Streamlit chat.  

### AI’s Role
- Suggested initial project scaffolding.  
- Generated boilerplate code (Streamlit UI, FAISS usage, dotenv integration).  
- Explained errors when code failed.  
- Offered prompt templates for retrieval-augmented generation.  

### Shared Ownership
- I **curated all content** that goes into the system.  
- I **edited** AI-generated code to work with my setup (Azure vs OpenAI).  
- I **decided** what to keep simple vs. what to extend (e.g., not adding bonus “modes” due to time).  
- I added extra files for clarity (artifacts/ folder) to meet Ubundi’s trial requirements.  

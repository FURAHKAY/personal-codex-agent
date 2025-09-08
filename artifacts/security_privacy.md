# Security & Privacy

- **Secrets**: `OPENAI_API_KEY` via `.env` or Streamlit Secrets; never committed.
- **Data scope**: only files in `/data`; user uploads are local to the session/repo.
- **No PII processing** beyond my own docs.
- **Outbound**: OpenAI API only; no third-party telemetry.
- **Opt-out**: Remove any doc by deleting from `/data` and re-indexing.

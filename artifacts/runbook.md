# Runbook

## Local
1) `python3 build_index.py`
2) `streamlit run app.py`
3) Test: ask 5 canned questions (see `eval_plan.md`), verify source chips.

## Reindex from UI
Sidebar → Admin → Upload → **Save & Re-index** → confirm **Index dim/count** increased in Settings.

## Deployment (Streamlit Cloud)
- Add `OPENAI_API_KEY` to Secrets
- Commit `index/` for deterministic startup or allow UI reindex on first run

## Recovery
- Model/Dim mismatch → rebuild index
- Broken index → `rm -rf index && python build_index.py`

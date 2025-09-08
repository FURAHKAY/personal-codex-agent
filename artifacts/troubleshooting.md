# Troubleshooting

## “Index not found”
Run `python build_index.py`, commit `index/` or rebuild via UI (Admin → Save & Re-index).

## “Embed model mismatch”
Set `EMBED_MODEL` in `.env` to match `index/meta.json` or `rm -rf index && build`.

## “Upload worked but not used”
After uploading, click **Save file(s) and Re-index**. Check **Settings**: it now shows updated **Index dim/count**. Use “Search my docs” to confirm phrases appear.

## “Dark theme looks off”
Ensure only one CSS token block is active (see CSS in `app.py`); header background must use `var(--card)` not hardcoded white.

## “Copy / Regenerate do nothing”
Be sure the UI helpers are wired (see `ui_copy_button` + `ui_regen_button` usage in the assistant turn loop).

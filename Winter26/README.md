# Winter26 — self-contained lab standardization workflow

1. Translate selected columns to English (Ollama, with `translation_cache.json`).
2. Normalize text and build match keys.
3. **Tier 1** — exact mapping from bundled JSON dictionaries (with original-key fallback after translation).
4. **Tier 2** — Sentence-BERT retrieval + Ollama in-context standardization for uncovered rows.

## Layout

| Path | Purpose |
|------|---------|
| `winter26/` | Python package |
| `mapping/` | `combined_mapping.json`, `specimen_mapping.json`, `item_mapping.json`, `unit_mapping.json` |
| `sample_data/` | Example input (`AIPHAS_raw_data_2026-2.csv`) |
| `new_data_output/` | Generated artifacts (CSV/cache; gitignored except `.gitkeep`) |

## Setup

**Use the bundle root** — the directory that contains `pyproject.toml`, `requirements.txt`, and the `winter26/` package folder. Do **not** create the virtualenv inside `winter26/` (that subdirectory is only the importable package).

```bash
# If you cloned AIPHAS-W26-F26:
cd /path/to/AIPHAS-W26-F26/Winter26

# Sanity check (should print pyproject.toml):
ls pyproject.toml

python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python3 -m pip install -U pip
python3 -m pip install -e .
# or: pip install -r requirements.txt
```

From the **repository root** (`AIPHAS-W26-F26/`), you can instead run `make setup`, which creates `Winter26/.venv` and installs the project there.

Install and start **Ollama**, then pull a model (e.g. `ollama pull llama3.2`).

## Run

From `Winter26/` after `pip install -e .`:

```bash
python -m winter26.new_data_pipeline \
  --input sample_data/AIPHAS_raw_data_2026-2.csv \
  --output-dir new_data_output \
  --translate-workers 8 \
  --max-llm-records 300
```

Quick test (no Tier 2):

```bash
python -m winter26.new_data_pipeline --max-rows 50 --skip-tier2
```

Same entry point:

```bash
python -m winter26 --max-rows 50 --skip-tier2
```

Direct script (from `Winter26/`):

```bash
python winter26/new_data_pipeline.py --max-rows 50 --skip-tier2
```

## Environment variables

- `OLLAMA_HOST` — default `http://127.0.0.1:11434`
- `OLLAMA_MODEL` — overrides `--ollama-model`

## Notes

- Mapping JSON files are **large**; consider Git LFS if needed.
- First Tier 2 run downloads the Sentence-BERT model from Hugging Face.
- See [WORKFLOW.md](WORKFLOW.md) for step-by-step objectives and how the code fulfills them.

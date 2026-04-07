# Winter26 workflow — steps, objectives, and validation

This document describes each stage of [`winter26/new_data_pipeline.py`](winter26/new_data_pipeline.py), the **objective** of that stage, whether the **implementation satisfies** that objective, and **caveats**.

## Flow overview

```mermaid
flowchart LR
  load[Load_CSV]
  translate[Translate_Ollama]
  normalize[Normalize_Keys]
  tier1[Tier1_Exact]
  tier2[Tier2_SBERT_Ollama]
  final[Final_Schema_CSV]
  load --> translate --> normalize --> tier1 --> tier2 --> final
```

## Step-by-step

| Step | Objective | What the code does | Objective met? | Caveats |
|------|-----------|-------------------|------------------|---------|
| **0 — Load** | Ingest a new lab export reliably across encodings. | `read_csv_robust()` tries `utf-8-sig`, `utf-8`, `latin1`; skips bad lines. | **Yes** for typical exports. | Malformed rows are dropped (`on_bad_lines="skip"`). |
| **1 — Translate** | Produce English text for mapping/Tier2 while keeping originals for fallback. | For each configured column, unique strings are translated via Ollama in parallel (`ThreadPoolExecutor`); results cached in `new_data_output/translation_cache.json`; adds `specimen_en`, `item_en`, `unit_en`; working copy uses English for downstream normalization. | **Yes** if Ollama is reachable. **Partial** if Ollama is down (passthrough, no real translation). | LLM output can vary; cache makes reruns stable. |
| **2 — Normalize** | Stable `raw_*`, `*_norm`, `combo_key` for Tier 1/2; preserve original combo for dictionary match. | `prepare_for_mapping()` applies encoding recovery + normalization on translated working columns; `build_original_norm_keys()` builds `combo_key_original` from `*_original` fields. | **Yes**. | Recovery heuristics assume latin1-style garbling for some edge cases. |
| **3 — Tier 1** | Maximize deterministic matches using bundled JSON maps. | `run_tier1_exact_mapping()` on translated `combo_key`; `run_tier1_with_original_fallback()` retries combined + component maps on `combo_key_original` with labels `*_original_fallback`. | **Yes** for keys present in `mapping/*.json`. **No** for unseen triples (expected until Tier 2). | Translation can drift vocabulary; fallback mitigates but does not guarantee a hit. |
| **4 — Tier 2** | For uncovered rows, retrieve similar known mappings and ask Ollama for a standardized triple + confidence; gate auto-accept. | `run_tier2_bert_llm()`: embed reference keys from combined map, top-k cosine similarity, prompt Ollama with examples, parse JSON, set `ollama_auto` vs `manual_review_low_confidence` vs `bert_only_manual`. | **Yes** when `sentence_transformers` + PyTorch work and Ollama responds. **Degraded** if SBERT missing (`manual_review_missing_sbert`) or Ollama unreachable (BERT hint only). | CPU inference can be slow; use `--max-llm-records` for trials. |
| **5 — Final export** | Deliver a fixed column schema for downstream tools. | `finalize_output()` selects and normalizes `OUTPUT_COLUMNS`. | **Yes**. | Dropped intermediate columns (e.g. `*_en`, `combo_key`) are only in step CSVs, not in `final_output.csv`. |

## Artifacts

| File | Contents |
|------|----------|
| `new_data_output/step1_translated.csv` | Original columns + `*_en` translated columns. |
| `new_data_output/step2_normalized.csv` | Normalized keys, `combo_key`, `combo_key_original`, etc. |
| `new_data_output/step3_tier1.csv` | After exact mapping + original fallback. |
| `new_data_output/step4_tier2.csv` | After retrieval + Ollama. |
| `new_data_output/final_output.csv` | Schema-aligned export. |
| `new_data_output/translation_cache.json` | String → translated string (speed + reproducibility). |

## How to confirm objectives in practice

1. **Tier 1 coverage**: In `step3_tier1.csv`, count non-empty `std_specimen`/`std_item`/`std_unit` or `label_source` starting with `exact_`.
2. **Translation**: Compare `specimen` vs `specimen_en` in `step1_translated.csv` when Ollama is running.
3. **Tier 2**: With Ollama + SBERT, expect some `ollama_auto` in `final_output.csv` for previously uncovered rows.

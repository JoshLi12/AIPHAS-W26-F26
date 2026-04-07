"""Tier 2: Sentence-BERT retrieval + in-context learning via a local Ollama model."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

from winter26.config import DEFAULT_OLLAMA_HOST, DEFAULT_OLLAMA_MODEL, DEFAULT_SBERT_MODEL
from winter26.io_raw import normalize_text
from winter26.ollama_chat import extract_json_object, ollama_chat, ollama_server_reachable

try:
    from sentence_transformers import SentenceTransformer, util
except Exception:  # pragma: no cover
    SentenceTransformer = None
    util = None


@dataclass
class Tier2Result:
    std_specimen: Optional[str]
    std_item: Optional[str]
    std_unit: Optional[str]
    confidence: float
    bert_std_specimen: Optional[str]
    source: str


def build_tier2_references(
    combined_map: Dict[str, Dict[str, str]],
) -> Tuple[List[str], List[str], List[str]]:
    norm_keys = list(combined_map.keys())
    raw_inputs: List[str] = []
    std_outputs: List[str] = []
    bert_std_specimens: List[str] = []

    for norm_key in norm_keys:
        std = combined_map[norm_key]
        raw_inputs.append(norm_key)
        std_outputs.append(
            f"{std.get('std_specimen', '')}|{std.get('std_item', '')}|{std.get('std_unit', '')}"
        )
        bert_std_specimens.append(std.get("std_specimen", ""))

    return raw_inputs, std_outputs, bert_std_specimens


def parse_llm_json_response(response_text: str) -> Tuple[Optional[str], Optional[str], Optional[str], float]:
    try:
        payload = json.loads(extract_json_object(response_text))
        specimen = normalize_text(payload.get("std_specimen"))
        item = normalize_text(payload.get("std_item"))
        unit = normalize_text(payload.get("std_unit"))
        confidence = float(payload.get("confidence", 0.0))
        confidence = max(0.0, min(100.0, confidence))
        if not (specimen and item and unit):
            return None, None, None, 0.0
        return specimen, item, unit, confidence
    except Exception:
        return None, None, None, 0.0


def call_ollama_with_examples(
    ollama_host: str,
    ollama_model: str,
    raw_specimen: str,
    raw_item: str,
    raw_unit: str,
    examples: List[Tuple[str, str, float]],
) -> Tuple[Optional[str], Optional[str], Optional[str], float]:
    examples_block = ""
    for i, (inp, out, sim) in enumerate(examples, start=1):
        examples_block += (
            f"Example {i}\nInput: {inp}\nOutput: {out}\nSimilarity: {sim:.4f}\n\n"
        )

    prompt = f"""You standardize clinical lab fields. Use examples to map specimen, item, and unit.

{examples_block}Target input:
{raw_specimen}|{raw_item}|{raw_unit}

Return STRICT JSON only:
{{
  "std_specimen": "<string>",
  "std_item": "<string>",
  "std_unit": "<string>",
  "confidence": <0-100 number>
}}
"""

    content = ollama_chat(
        ollama_host,
        ollama_model,
        system="You are an expert in medical lab terminology standardization. Output valid JSON only.",
        user=prompt,
    )
    if not content:
        return None, None, None, 0.0
    return parse_llm_json_response(content)


def run_tier2_bert_llm(
    df: pd.DataFrame,
    combined_map: Dict[str, Dict[str, str]],
    *,
    sbert_model_name: str = DEFAULT_SBERT_MODEL,
    confidence_threshold: float,
    top_k: int,
    ollama_model: str = DEFAULT_OLLAMA_MODEL,
    ollama_host: str = DEFAULT_OLLAMA_HOST,
    max_llm_records: Optional[int],
) -> pd.DataFrame:
    uncovered_mask = (df["std_specimen"] == "") | (df["std_item"] == "") | (df["std_unit"] == "")
    uncovered_indices = df[uncovered_mask].index.tolist()
    if not uncovered_indices:
        return df

    out = df.copy()
    print(f"Tier 2 uncovered rows: {len(uncovered_indices):,}")

    if SentenceTransformer is None or util is None:
        out.loc[uncovered_indices, "label_source"] = "manual_review_missing_sbert"
        return out

    ref_inputs, ref_outputs, ref_bert_specimens = build_tier2_references(combined_map)
    if not ref_inputs:
        out.loc[uncovered_indices, "label_source"] = "manual_review_empty_reference"
        return out

    print("Loading Sentence-BERT model...")
    bert_model = SentenceTransformer(sbert_model_name)
    print(f"Encoding {len(ref_inputs):,} reference combinations...")
    ref_embeddings = bert_model.encode(ref_inputs, convert_to_tensor=True)

    process_indices = uncovered_indices
    if max_llm_records is not None:
        process_indices = process_indices[: int(max_llm_records)]
        skip_indices = list(set(uncovered_indices) - set(process_indices))
        if skip_indices:
            out.loc[skip_indices, "label_source"] = "manual_review_llm_limit"
            print(f"Tier 2 capped by --max-llm-records: processing {len(process_indices):,}, skipping {len(skip_indices):,}")

    if process_indices and not ollama_server_reachable(ollama_host):
        print(
            f"Ollama not reachable at {ollama_host!r}. Tier 2 will use BERT hints only "
            f"(pull a model with: ollama pull {ollama_model})."
        )

    total = len(process_indices)
    if total == 0:
        return out
    print(f"Tier 2 processing rows: {total:,}")

    for i, idx in enumerate(process_indices, start=1):
        row = out.loc[idx]
        raw_combo = f"{row['specimen_norm']}|{row['item_norm']}|{row['unit_norm']}"
        query_embedding = bert_model.encode(raw_combo, convert_to_tensor=True)
        similarities = util.cos_sim(query_embedding, ref_embeddings)[0]

        tk = min(top_k, len(ref_inputs))
        top_indices = similarities.topk(tk).indices.tolist()
        examples: List[Tuple[str, str, float]] = []
        for ref_idx in top_indices:
            examples.append(
                (ref_inputs[ref_idx], ref_outputs[ref_idx], float(similarities[ref_idx].item()))
            )

        best_ref_idx = top_indices[0]
        best_similarity = float(similarities[best_ref_idx].item())
        best_std_specimen = ref_bert_specimens[best_ref_idx]
        out.at[idx, "bert_std_specimen"] = best_std_specimen

        tier2_result = Tier2Result(
            std_specimen=None,
            std_item=None,
            std_unit=None,
            confidence=best_similarity * 100.0,
            bert_std_specimen=best_std_specimen,
            source="bert_only_manual",
        )

        pred_specimen, pred_item, pred_unit, llm_conf = call_ollama_with_examples(
            ollama_host=ollama_host,
            ollama_model=ollama_model,
            raw_specimen=str(row["raw_specimen"]),
            raw_item=str(row["raw_item"]),
            raw_unit=str(row["raw_unit"]),
            examples=examples,
        )

        if pred_specimen and pred_item and pred_unit:
            tier2_result = Tier2Result(
                std_specimen=pred_specimen,
                std_item=pred_item,
                std_unit=pred_unit,
                confidence=llm_conf,
                bert_std_specimen=best_std_specimen,
                source="ollama_prediction",
            )

        if tier2_result.std_specimen and tier2_result.std_item and tier2_result.std_unit:
            out.at[idx, "std_specimen"] = tier2_result.std_specimen
            out.at[idx, "std_item"] = tier2_result.std_item
            out.at[idx, "std_unit"] = tier2_result.std_unit
            if tier2_result.confidence >= confidence_threshold:
                out.at[idx, "label_source"] = "ollama_auto"
            else:
                out.at[idx, "label_source"] = "manual_review_low_confidence"
        else:
            out.at[idx, "label_source"] = tier2_result.source

        if i == 1 or i % 25 == 0 or i == total:
            print(f"Tier 2 progress: {i:,}/{total:,}")

    return out

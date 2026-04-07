"""Full data pipeline"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

# Allow direct execution: python winter26/new_data_pipeline.py from Winter26/
if __package__ in (None, ""):
    _pkg_root = Path(__file__).resolve().parent.parent
    if str(_pkg_root) not in sys.path:
        sys.path.insert(0, str(_pkg_root))

from winter26.config import (
    DEFAULT_OLLAMA_HOST,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_SBERT_MODEL,
    default_paths,
    ollama_host_from_env,
    ollama_model_from_env,
)
from winter26.io_raw import normalize_text, prepare_for_mapping
from winter26.mappings import load_mappings
from winter26.ollama_chat import ollama_chat, ollama_server_reachable
from winter26.output_schema import finalize_output
from winter26.tier1_exact import run_tier1_exact_mapping
from winter26.tier2_bert_llm import run_tier2_bert_llm


def read_csv_robust(path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc, engine="python", on_bad_lines="skip")
        except Exception:
            continue
    return pd.read_csv(path, engine="python", on_bad_lines="skip")


def ensure_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def load_json_dict(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            return {str(k): str(v) for k, v in payload.items()}
    except Exception:
        pass
    return {}


def save_json_dict(path: Path, data: Dict[str, str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dict(sorted(data.items())), f, ensure_ascii=False, indent=2)


def translate_text_ollama(ollama_host: str, ollama_model: str, text: str) -> str:
    t = normalize_text(text)
    if not t:
        return ""
    prompt = (
        "Translate to concise English for clinical lab data. "
        "Keep abbreviations, units, symbols, and numbers unchanged. "
        "Return only translated text.\n\n"
        f"Input: {t}"
    )
    out = ollama_chat(
        ollama_host,
        ollama_model,
        system="You are a precise medical translator. Output plain text only.",
        user=prompt,
        timeout=90.0,
    )
    return normalize_text(out) if out else t


def translate_columns(
    df: pd.DataFrame,
    *,
    columns: Iterable[str],
    ollama_host: str,
    ollama_model: str,
    cache_path: Path,
    workers: int,
) -> pd.DataFrame:
    out = df.copy()
    cache = load_json_dict(cache_path)

    use_ollama = ollama_server_reachable(ollama_host)
    if not use_ollama:
        print(f"Ollama not reachable at {ollama_host!r}; translation step will keep original text.")

    for col in columns:
        if col not in out.columns:
            continue
        out[col] = out[col].fillna("").map(normalize_text)
        uniques: List[str] = sorted(v for v in out[col].unique().tolist() if v)
        if not uniques:
            continue

        print(f"Translating column '{col}' ({len(uniques):,} unique values)...")
        translated_lookup: Dict[str, str] = {u: cache.get(u, "") for u in uniques}
        pending = [u for u in uniques if not translated_lookup[u]]

        if use_ollama and pending:
            max_workers = max(1, int(workers))
            print(f"  {col}: translating {len(pending):,} cache misses with {max_workers} workers...")
            done = 0
            flush_every = 100
            with cf.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(translate_text_ollama, ollama_host, ollama_model, raw): raw
                    for raw in pending
                }
                for future in cf.as_completed(futures):
                    raw = futures[future]
                    try:
                        translated = future.result()
                    except Exception:
                        translated = raw
                    translated_lookup[raw] = translated or raw
                    cache[raw] = translated_lookup[raw]
                    done += 1
                    if done == 1 or done % 50 == 0 or done == len(pending):
                        print(f"  {col}: {done:,}/{len(pending):,} translated")
                    if done % flush_every == 0:
                        save_json_dict(cache_path, cache)
        else:
            for raw in pending:
                translated_lookup[raw] = raw

        out[f"{col}_en"] = out[col].map(translated_lookup).fillna("")
        save_json_dict(cache_path, cache)

    return out


def build_original_norm_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    specimen_src = "specimen_original" if "specimen_original" in out.columns else "specimen"
    item_src = "item_original" if "item_original" in out.columns else "item"
    unit_src = "unit_original" if "unit_original" in out.columns else "unit"
    out["raw_specimen_original"] = out.get(specimen_src, "").fillna("").map(normalize_text)
    out["raw_item_original"] = out.get(item_src, "").fillna("").map(normalize_text)
    out["raw_unit_original"] = out.get(unit_src, "").fillna("").map(normalize_text)
    out["specimen_norm_original"] = out["raw_specimen_original"].str.lower()
    out["item_norm_original"] = out["raw_item_original"].str.lower()
    out["unit_norm_original"] = out["raw_unit_original"].str.lower()
    out["combo_key_original"] = (
        out["specimen_norm_original"] + "|" + out["item_norm_original"] + "|" + out["unit_norm_original"]
    )
    return out


def run_tier1_with_original_fallback(
    normalized_df: pd.DataFrame,
    combined_map: Dict[str, Dict[str, str]],
    specimen_map: Dict[str, str],
    item_map: Dict[str, str],
    unit_map: Dict[str, str],
) -> pd.DataFrame:
    out = run_tier1_exact_mapping(normalized_df, combined_map, specimen_map, item_map, unit_map)
    missing = (out["std_specimen"] == "") | (out["std_item"] == "") | (out["std_unit"] == "")
    if not missing.any():
        return out

    miss_idx = out.index[missing]
    combo_hits = out.loc[miss_idx, "combo_key_original"].map(combined_map)
    for idx, hit in combo_hits.items():
        if not isinstance(hit, dict):
            continue
        s = str(hit.get("std_specimen", "") or "")
        i = str(hit.get("std_item", "") or "")
        u = str(hit.get("std_unit", "") or "")
        if s and i and u:
            out.at[idx, "std_specimen"] = s
            out.at[idx, "std_item"] = i
            out.at[idx, "std_unit"] = u
            out.at[idx, "label_source"] = "exact_combined_mapping_original_fallback"

    missing2 = (out["std_specimen"] == "") | (out["std_item"] == "") | (out["std_unit"] == "")
    if missing2.any():
        idx2 = out.index[missing2]
        sp = out.loc[idx2, "specimen_norm_original"].map(specimen_map).fillna("")
        it = out.loc[idx2, "item_norm_original"].map(item_map).fillna("")
        un = out.loc[idx2, "unit_norm_original"].map(unit_map).fillna("")
        ok = (sp != "") & (it != "") & (un != "")
        ok_idx = sp[ok].index
        out.loc[ok_idx, "std_specimen"] = sp[ok].to_numpy()
        out.loc[ok_idx, "std_item"] = it[ok].to_numpy()
        out.loc[ok_idx, "std_unit"] = un[ok].to_numpy()
        out.loc[ok_idx, "label_source"] = "exact_component_mapping_original_fallback"
    return out


def run_new_data_pipeline(
    *,
    input_csv: Path,
    mapping_dir: Path,
    output_dir: Path,
    translate_cols: List[str],
    ollama_model: str,
    ollama_host: str,
    sbert_model: str,
    confidence_threshold: float,
    top_k: int,
    max_llm_records: Optional[int],
    max_rows: Optional[int],
    skip_tier2: bool,
    translate_workers: int,
) -> Path:
    ensure_output_dir(output_dir)
    translation_cache_path = output_dir / "translation_cache.json"

    print("Step 0/4: load raw data...")
    raw_df = read_csv_robust(input_csv)
    if max_rows is not None:
        raw_df = raw_df.head(int(max_rows)).copy()
    print(f"Rows loaded: {len(raw_df):,}")

    print("Step 1/4: translate configured columns to English...")
    translated_df = translate_columns(
        raw_df,
        columns=translate_cols,
        ollama_host=ollama_host,
        ollama_model=ollama_model,
        cache_path=translation_cache_path,
        workers=translate_workers,
    )
    translated_working_df = translated_df.copy()
    for col in translate_cols:
        en_col = f"{col}_en"
        if en_col in translated_working_df.columns and col in translated_working_df.columns:
            translated_working_df[f"{col}_original"] = translated_working_df[col]
            translated_working_df[col] = translated_working_df[en_col]
    step1 = output_dir / "step1_translated.csv"
    translated_df.to_csv(step1, index=False)
    print(f"Wrote: {step1}")

    print("Step 2/4: normalize data and create matching keys...")
    normalized_df = prepare_for_mapping(translated_working_df)
    normalized_df = build_original_norm_keys(normalized_df)
    step2 = output_dir / "step2_normalized.csv"
    normalized_df.to_csv(step2, index=False)
    print(f"Wrote: {step2}")

    print("Load mapping dictionaries...")
    combined_map, specimen_map, item_map, unit_map = load_mappings(mapping_dir)
    print(
        f"Mappings: combined={len(combined_map):,}, specimen={len(specimen_map):,}, "
        f"item={len(item_map):,}, unit={len(unit_map):,}"
    )

    print("Step 3/4: Tier 1 exact mapping...")
    tier1_df = run_tier1_with_original_fallback(
        normalized_df,
        combined_map,
        specimen_map,
        item_map,
        unit_map,
    )
    step3 = output_dir / "step3_tier1.csv"
    tier1_df.to_csv(step3, index=False)
    print(f"Wrote: {step3}")

    print("Step 4/4: Tier 2 Sentence-BERT + Ollama...")
    if skip_tier2:
        tier2_df = tier1_df.copy()
        uncovered = (
            (tier2_df["std_specimen"] == "")
            | (tier2_df["std_item"] == "")
            | (tier2_df["std_unit"] == "")
        )
        tier2_df.loc[uncovered & (tier2_df["label_source"] == ""), "label_source"] = "manual_review_tier2_skipped"
    else:
        tier2_df = run_tier2_bert_llm(
            tier1_df,
            combined_map,
            sbert_model_name=sbert_model,
            confidence_threshold=confidence_threshold,
            top_k=top_k,
            ollama_model=ollama_model,
            ollama_host=ollama_host,
            max_llm_records=max_llm_records,
        )
    step4 = output_dir / "step4_tier2.csv"
    tier2_df.to_csv(step4, index=False)
    print(f"Wrote: {step4}")

    final_df = finalize_output(tier2_df)
    final_path = output_dir / "final_output.csv"
    final_df.to_csv(final_path, index=False)
    print(f"Wrote: {final_path}")
    print("\nFinal label_source counts:")
    print(final_df["label_source"].value_counts(dropna=False).to_string())
    return final_path


def build_parser() -> argparse.ArgumentParser:
    paths = default_paths()
    p = argparse.ArgumentParser(
        description="Translate -> normalize -> Tier1 -> Tier2 pipeline with per-step outputs."
    )
    p.add_argument(
        "--input",
        type=Path,
        default=paths["input_csv"],
        help="Input CSV for new data run (default: sample_data/AIPHAS_raw_data_2026-2.csv if present).",
    )
    p.add_argument("--mapping-dir", type=Path, default=paths["mapping_dir"])
    p.add_argument(
        "--output-dir",
        type=Path,
        default=paths["output_dir"],
        help="Folder for step-wise outputs.",
    )
    p.add_argument(
        "--translate-cols",
        nargs="+",
        default=["specimen", "item", "unit"],
        help="Columns translated to English before normalization.",
    )
    p.add_argument("--ollama-model", type=str, default=DEFAULT_OLLAMA_MODEL)
    p.add_argument("--ollama-host", type=str, default=DEFAULT_OLLAMA_HOST)
    p.add_argument("--sbert-model", type=str, default=DEFAULT_SBERT_MODEL)
    p.add_argument("--threshold", type=float, default=85.0)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--max-llm-records", type=int, default=None)
    p.add_argument("--max-rows", type=int, default=None, help="Optional row cap for quick test runs.")
    p.add_argument(
        "--translate-workers",
        type=int,
        default=4,
        help="Parallel workers for translation cache misses (increase for faster runs).",
    )
    p.add_argument("--skip-tier2", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    run_new_data_pipeline(
        input_csv=args.input,
        mapping_dir=args.mapping_dir,
        output_dir=args.output_dir,
        translate_cols=args.translate_cols,
        ollama_model=ollama_model_from_env(args.ollama_model),
        ollama_host=ollama_host_from_env(args.ollama_host),
        sbert_model=args.sbert_model,
        confidence_threshold=args.threshold,
        top_k=args.top_k,
        max_llm_records=args.max_llm_records,
        max_rows=args.max_rows,
        skip_tier2=args.skip_tier2,
        translate_workers=args.translate_workers,
    )


if __name__ == "__main__":
    main()

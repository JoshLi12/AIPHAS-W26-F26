"""Normalizes text fields and builds match keys for mapping"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ORDERCODE_PATTERN = re.compile(r"^\d{5}[A-Z]$")


def normalize_text(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).replace("\u3000", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_for_match(value) -> str:
    return normalize_text(value).lower()


def score_text_quality(text: str) -> float:
    if not text:
        return 0.0
    printable = sum(1 for ch in text if ch.isprintable())
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    replacement = text.count("\ufffd")
    control = sum(1 for ch in text if ord(ch) < 32 and ch not in ("\t", "\n", "\r"))
    return printable + (cjk * 1.5) - (replacement * 2.0) - (control * 2.0)


def recover_encoding_value(value) -> str:
    text = normalize_text(value)
    if not text:
        return ""

    best = text
    best_score = score_text_quality(text)

    for target_encoding in ("cp950", "big5", "utf-8"):
        try:
            candidate = text.encode("latin1", errors="ignore").decode(target_encoding, errors="ignore")
        except Exception:
            continue
        candidate = normalize_text(candidate)
        score = score_text_quality(candidate)
        if score > best_score:
            best = candidate
            best_score = score

    return best


def load_raw_2026_csv(input_csv: Path) -> pd.DataFrame:
    return pd.read_csv(
        input_csv,
        encoding="latin1",
        engine="python",
        on_bad_lines="skip",
    )


def filter_valid_ordercodes(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    working["ordercode"] = working["ordercode"].apply(normalize_text).str.upper()
    return working[working["ordercode"].str.match(ORDERCODE_PATTERN, na=False)].copy()


def prepare_for_mapping(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    for col in ("specimen", "item", "unit", "hospital", "ref"):
        if col in working.columns:
            working[col] = working[col].apply(recover_encoding_value).apply(normalize_text)

    working["raw_specimen"] = working.get("specimen", "").apply(normalize_text)
    working["raw_item"] = working.get("item", "").apply(normalize_text)
    working["raw_unit"] = working.get("unit", "").apply(normalize_text)

    working["specimen_norm"] = working["raw_specimen"].apply(normalize_for_match)
    working["item_norm"] = working["raw_item"].apply(normalize_for_match)
    working["unit_norm"] = working["raw_unit"].apply(normalize_for_match)
    working["combo_key"] = (
        working["specimen_norm"] + "|" + working["item_norm"] + "|" + working["unit_norm"]
    )
    return working

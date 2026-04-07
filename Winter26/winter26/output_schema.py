"""Finalizes output schema"""

from __future__ import annotations

import pandas as pd

from winter26.io_raw import normalize_text

OUTPUT_COLUMNS = [
    "ordercode",
    "raw_specimen",
    "std_specimen",
    "raw_item",
    "std_item",
    "raw_unit",
    "std_unit",
    "multiplier_unit",
    "hospital",
    "ref",
    "label_source",
    "bert_std_specimen",
    "nn_std_specimen",
]


def finalize_output(df: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame()
    output["ordercode"] = df["ordercode"].apply(normalize_text)
    output["raw_specimen"] = df["raw_specimen"].apply(normalize_text)
    output["std_specimen"] = df["std_specimen"].apply(normalize_text)
    output["raw_item"] = df["raw_item"].apply(normalize_text)
    output["std_item"] = df["std_item"].apply(normalize_text)
    output["raw_unit"] = df["raw_unit"].apply(normalize_text)
    output["std_unit"] = df["std_unit"].apply(normalize_text)
    output["multiplier_unit"] = df["multiplier_unit"].apply(normalize_text)
    output["hospital"] = df.get("hospital", "").apply(normalize_text)
    output["ref"] = df.get("ref", "").apply(normalize_text)
    output["label_source"] = df["label_source"].apply(normalize_text)
    output["bert_std_specimen"] = df["bert_std_specimen"].apply(normalize_text)
    output["nn_std_specimen"] = df["nn_std_specimen"].apply(normalize_text)
    return output[OUTPUT_COLUMNS]

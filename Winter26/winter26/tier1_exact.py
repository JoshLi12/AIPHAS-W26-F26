"""Tier 1: exact lookup on (specimen, item, unit), then component-level fallback."""

from __future__ import annotations

from typing import Dict, Union

import pandas as pd


def run_tier1_exact_mapping(
    df: pd.DataFrame,
    combined_map: Dict[str, Dict[str, str]],
    specimen_map: Dict[str, str],
    item_map: Dict[str, str],
    unit_map: Dict[str, str],
) -> pd.DataFrame:
    out = df.copy()
    n = len(out)

    out["std_specimen"] = ""
    out["std_item"] = ""
    out["std_unit"] = ""
    out["label_source"] = ""
    out["bert_std_specimen"] = ""
    out["nn_std_specimen"] = ""
    out["multiplier_unit"] = ""

    combo_series = out["combo_key"].map(combined_map)

    def _std_col(hit: Union[dict, float], field: str) -> str:
        if not isinstance(hit, dict):
            return ""
        return str(hit.get(field, "") or "")

    std_spec_c = combo_series.map(lambda h: _std_col(h, "std_specimen"))
    std_item_c = combo_series.map(lambda h: _std_col(h, "std_item"))
    std_unit_c = combo_series.map(lambda h: _std_col(h, "std_unit"))
    combined_hit = std_spec_c.ne("") & std_item_c.ne("") & std_unit_c.ne("")

    out.loc[combined_hit, "std_specimen"] = std_spec_c[combined_hit]
    out.loc[combined_hit, "std_item"] = std_item_c[combined_hit]
    out.loc[combined_hit, "std_unit"] = std_unit_c[combined_hit]
    out.loc[combined_hit, "label_source"] = "exact_combined_mapping"

    miss = ~combined_hit
    if miss.any():
        miss_idx = out.index[miss]
        sp = out.loc[miss_idx, "specimen_norm"].map(specimen_map).fillna("")
        it = out.loc[miss_idx, "item_norm"].map(item_map).fillna("")
        un = out.loc[miss_idx, "unit_norm"].map(unit_map).fillna("")
        comp_ok = (sp != "") & (it != "") & (un != "")
        ok_idx = sp[comp_ok].index
        out.loc[ok_idx, "std_specimen"] = sp[comp_ok].to_numpy()
        out.loc[ok_idx, "std_item"] = it[comp_ok].to_numpy()
        out.loc[ok_idx, "std_unit"] = un[comp_ok].to_numpy()
        out.loc[ok_idx, "label_source"] = "exact_component_mapping"

    assert len(out) == n
    return out

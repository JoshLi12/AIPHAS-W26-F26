"""Loads combined + component JSON mappings with normalized lookup keys"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

from winter26.io_raw import normalize_for_match


def load_mappings(
    mapping_dir: Path,
) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str], Dict[str, str], Dict[str, str]]:
    with open(mapping_dir / "combined_mapping.json", "r", encoding="utf-8") as f:
        combined_map_raw = json.load(f)
    with open(mapping_dir / "specimen_mapping.json", "r", encoding="utf-8") as f:
        specimen_map_raw = json.load(f)
    with open(mapping_dir / "item_mapping.json", "r", encoding="utf-8") as f:
        item_map_raw = json.load(f)
    with open(mapping_dir / "unit_mapping.json", "r", encoding="utf-8") as f:
        unit_map_raw = json.load(f)

    combined_norm: Dict[str, Dict[str, str]] = {}
    for key, std_values in combined_map_raw.items():
        parts = key.split("|", 2)
        raw_specimen = parts[0] if len(parts) > 0 else ""
        raw_item = parts[1] if len(parts) > 1 else ""
        raw_unit = parts[2] if len(parts) > 2 else ""
        norm_key = (
            f"{normalize_for_match(raw_specimen)}|"
            f"{normalize_for_match(raw_item)}|"
            f"{normalize_for_match(raw_unit)}"
        )
        combined_norm[norm_key] = std_values

    specimen_norm = {normalize_for_match(k): v for k, v in specimen_map_raw.items()}
    item_norm = {normalize_for_match(k): v for k, v in item_map_raw.items()}
    unit_norm = {normalize_for_match(k): v for k, v in unit_map_raw.items()}

    return combined_norm, specimen_norm, item_norm, unit_norm

"""Target table contract shared by the dashboard export and the model export."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    role: str  # id, categorical, numeric, datetime, text
    required: bool = False
    sources: Sequence[str] = field(default_factory=tuple)
    vocabulary: Sequence[str] = field(default_factory=tuple)

    def source_names(self) -> List[str]:
        names = [self.name, *list(self.sources)]
        return names


@dataclass(frozen=True)
class TargetSchema:
    name: str
    columns: Sequence[ColumnSpec]
    description: str = ""

    def column(self, name: str) -> Optional[ColumnSpec]:
        for spec in self.columns:
            if spec.name == name:
                return spec
        return None

    def names(self) -> List[str]:
        return [spec.name for spec in self.columns]


def load_schema(path: Path) -> TargetSchema:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    columns = []
    for raw in payload.get("columns", []):
        role = str(raw.get("role", "text")).strip().lower()
        if role not in {"id", "categorical", "numeric", "datetime", "text"}:
            raise ValueError(f"Unsupported role {role!r} on column {raw.get('name')!r}")
        columns.append(
            ColumnSpec(
                name=str(raw["name"]).strip(),
                role=role,
                required=bool(raw.get("required", False)),
                sources=tuple(str(item) for item in raw.get("sources", [])),
                vocabulary=tuple(str(item) for item in raw.get("vocabulary", [])),
            )
        )
    if not columns:
        raise ValueError(f"Schema {path} has no columns")
    return TargetSchema(
        name=str(payload.get("name") or "dataset"),
        description=str(payload.get("description") or ""),
        columns=tuple(columns),
    )


def value_is_allowed(spec: ColumnSpec, value: str) -> bool:
    """Return whether a proposed value may be written for this column."""
    text = "" if value is None else str(value).strip()
    if text == "":
        return not spec.required
    if spec.role == "categorical" and spec.vocabulary:
        allowed = {item.casefold() for item in spec.vocabulary}
        return text.casefold() in allowed
    if spec.role == "numeric":
        return parse_number(text) is not None
    if spec.role == "datetime":
        return parse_datetime(text) is not None
    return True


def canonical_value(spec: ColumnSpec, value: str) -> str:
    """Normalize a value that has already been accepted as allowed."""
    text = "" if value is None else str(value).strip()
    if text == "":
        return ""
    if spec.role == "categorical" and spec.vocabulary:
        for item in spec.vocabulary:
            if item.casefold() == text.casefold():
                return item
        return text
    if spec.role == "numeric":
        parsed = parse_number(text)
        return parsed if parsed is not None else text
    if spec.role == "datetime":
        parsed = parse_datetime(text)
        return parsed if parsed is not None else text
    return collapse_space(text)


def exact_vocabulary_match(spec: ColumnSpec, raw_value: str) -> Optional[str]:
    """Return the canonical vocabulary item when the raw text already matches."""
    if spec.role != "categorical" or not spec.vocabulary:
        return None
    folded = collapse_space(raw_value).casefold()
    if not folded:
        return None
    for item in spec.vocabulary:
        if item.casefold() == folded:
            return item
    return None


def parse_number(value: str) -> Optional[str]:
    text = collapse_space(value).replace(",", "")
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return format(number, ".12g")


def parse_datetime(value: str) -> Optional[str]:
    text = collapse_space(value)
    if not text:
        return None
    formats = (
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%m/%d/%Y",
        "%m-%d-%Y",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
    )
    for fmt in formats:
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def collapse_space(value: str) -> str:
    return " ".join(str(value or "").replace("\u00a0", " ").split())

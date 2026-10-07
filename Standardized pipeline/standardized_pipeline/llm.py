"""Model clients. Each proposal is a value plus a confidence in [0, 1]."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Dict, Optional, Protocol, Tuple

from standardized_pipeline.schema import ColumnSpec, TargetSchema, collapse_space


Proposal = Tuple[str, float, str]


class LLMClient(Protocol):
    name: str

    def propose(self, *, task: str, payload: Dict[str, object]) -> Proposal:
        """Return (proposed_value, confidence, rationale)."""


# Stand-in scores used when no model server is running. They are fixed so a
# run can be reviewed and tested. Values at or above 0.80 are trusted by the
# default gate. Values below 0.80 are left for a person.
_OFFLINE_VALUES = {
    "specimen": {
        "urin": ("urine", 0.88, "offline model: urin is a common shortening of urine"),
        "血": ("blood", 0.61, "offline model: character may mean blood, but confidence is low"),
        "unknown fluid": ("", 0.22, "offline model: no specimen in the vocabulary fits"),
        "血液": ("blood", 0.93, "offline model: 血液 is blood"),
        "尿液": ("urine", 0.93, "offline model: 尿液 is urine"),
        "全血": ("blood", 0.90, "offline model: 全血 is whole blood"),
        "血清": ("serum", 0.92, "offline model: 血清 is serum"),
        "血液(blood)": ("blood", 0.86, "offline model: label already contains blood"),
        "urine-m. stream": ("urine", 0.66, "offline model: midstream urine needs a person to confirm"),
        "抽血/blood/blood": ("blood", 0.70, "offline model: mixed blood label"),
        "blood(豐濱)": ("blood", 0.68, "offline model: blood plus a site qualifier"),
        "血液常規": ("blood", 0.64, "offline model: this reads as a panel name, not only a specimen"),
        "nil": ("", 0.20, "offline model: nil is a placeholder, not a specimen"),
        "test": ("", 0.12, "offline model: test is not a specimen"),
        "-": ("", 0.10, "offline model: dash is not a specimen"),
    },
    "item": {
        "a2-glo": ("a2-globulin", 0.86, "offline model: A2-Glo abbreviates a2-globulin"),
        "glu": ("glucose", 0.91, "offline model: glu abbreviates glucose"),
        "mystery test": ("", 0.18, "offline model: test name is not in the vocabulary"),
    },
    "unit": {
        "percent": ("%", 0.93, "offline model: percent is the percent sign"),
        "???": ("", 0.05, "offline model: unit text is not a unit"),
    },
}


class OfflineLLM:
    """Deterministic client with the same proposal contract as a live model."""

    name = "offline"

    def propose(self, *, task: str, payload: Dict[str, object]) -> Proposal:
        if task == "map_column":
            header = collapse_space(str(payload.get("header", ""))).casefold()
            if "note" in header or "comment" in header:
                return (
                    "note",
                    0.55,
                    "offline model: header looks like a note, below the default threshold",
                )
            return (
                "DROP",
                0.40,
                "offline model: no schema column is a clear match for this header",
            )
        if task == "standardize_value":
            field = str(payload.get("field", ""))
            raw = collapse_space(str(payload.get("raw_value", ""))).casefold()
            known = _OFFLINE_VALUES.get(field, {}).get(raw)
            if known:
                return known
            if payload.get("role") == "datetime":
                return ("", 0.20, "offline model: value is not a recognizable date")
            return ("", 0.15, "offline model: no confident mapping for this value")
        return ("", 0.0, f"offline model: unknown task {task}")


class OllamaLLM:
    """Live local model. Unusable responses become confidence 0 so a person finishes them."""

    def __init__(self, host: str, model: str, timeout: float = 120.0) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.name = f"ollama:{model}"

    def propose(self, *, task: str, payload: Dict[str, object]) -> Proposal:
        system = (
            "You standardize one messy data value for a dashboard and a machine-learning table. "
            "Return a single JSON object and nothing else. "
            "confidence is a number from 0 to 1. Use a low confidence when you are unsure. "
            "Do not invent a value outside the allowed list."
        )
        user = _prompt(task, payload)
        text = _chat(self.host, self.model, system, user, self.timeout)
        return parse_proposal(text)


def parse_proposal(text: Optional[str]) -> Proposal:
    if not text:
        return ("", 0.0, "model returned no usable text")
    try:
        payload = json.loads(extract_json_object(text))
        proposed = str(payload.get("proposed_value", "") or "").strip()
        confidence = float(payload.get("confidence", 0.0))
        if 1.0 < confidence <= 100.0:
            confidence = confidence / 100.0
        rationale = str(payload.get("rationale", "") or "").strip()
        return (proposed, confidence, rationale or "model proposal")
    except (TypeError, ValueError, json.JSONDecodeError):
        return ("", 0.0, "model response was not usable JSON")


def extract_json_object(text: str) -> str:
    stripped = (text or "").strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", stripped, re.IGNORECASE)
    if fence:
        stripped = fence.group(1).strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        return stripped[start : end + 1]
    return stripped


def ollama_reachable(host: str, timeout: float = 3.0) -> bool:
    url = f"{host.rstrip('/')}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status == 200
    except Exception:
        return False


def build_client(kind: str, host: str, model: str) -> LLMClient:
    """Return the requested client.

    ``auto`` uses Ollama when the server answers, and the offline client otherwise.
    """
    selected = (kind or "auto").strip().lower()
    if selected == "offline":
        return OfflineLLM()
    if selected == "ollama":
        return OllamaLLM(host, model)
    if selected == "auto":
        if ollama_reachable(host):
            return OllamaLLM(host, model)
        return OfflineLLM()
    raise ValueError("llm must be auto, offline, or ollama")


def _prompt(task: str, payload: Dict[str, object]) -> str:
    if task == "map_column":
        columns = payload.get("columns") or []
        return (
            "Map this raw column header onto one target column, or use DROP if it should not be kept.\n"
            f"Header: {payload.get('header')}\n"
            f"Target columns: {', '.join(str(name) for name in columns)}\n"
            'JSON: {"proposed_value": "<column or DROP>", "confidence": 0.0, "rationale": "..."}'
        )
    vocabulary = payload.get("vocabulary") or []
    allowed = ", ".join(str(item) for item in vocabulary) if vocabulary else "(any cleaned value)"
    return (
        "Map this raw value onto the target field.\n"
        f"Field: {payload.get('field')}\n"
        f"Role: {payload.get('role')}\n"
        f"Allowed values: {allowed}\n"
        f"Raw value: {payload.get('raw_value')}\n"
        "Use an empty proposed_value if none of the allowed values fit, and set confidence low.\n"
        'JSON: {"proposed_value": "<value or empty>", "confidence": 0.0, "rationale": "..."}'
    )


def _chat(host: str, model: str, system: str, user: str, timeout: float) -> Optional[str]:
    url = f"{host}/api/chat"
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {"temperature": 0},
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError, OSError):
        return None
    message = payload.get("message") or {}
    content = message.get("content")
    if content is None:
        return None
    return str(content).strip()


def column_prompt_names(schema: TargetSchema) -> list:
    return [spec.name for spec in schema.columns]


def value_payload(spec: ColumnSpec, raw_value: str) -> Dict[str, object]:
    return {
        "field": spec.name,
        "role": spec.role,
        "vocabulary": list(spec.vocabulary),
        "raw_value": raw_value,
    }

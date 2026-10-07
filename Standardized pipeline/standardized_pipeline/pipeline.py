"""Run raw rows through model decisions, the confidence gate, and exports."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from standardized_pipeline.events import Emit, EventLog, decision_event, emitter_or_noop
from standardized_pipeline.gate import ConfidenceGate, Decision, RoutedDecision
from standardized_pipeline.llm import (
    LLMClient,
    build_client,
    column_prompt_names,
    value_payload,
)
from standardized_pipeline.schema import (
    ColumnSpec,
    TargetSchema,
    canonical_value,
    collapse_space,
    exact_vocabulary_match,
    load_schema,
    parse_datetime,
    parse_number,
    value_is_allowed,
)


def run_pipeline(
    *,
    input_csv: Path,
    schema_path: Path,
    output_dir: Path,
    threshold: float = 0.80,
    llm: str = "auto",
    ollama_host: str = "http://127.0.0.1:11434",
    ollama_model: str = "llama3.2",
    client: Optional[LLMClient] = None,
    echo: bool = False,
) -> Dict[str, object]:
    """Standardize a raw table and write dashboard, model, and review files.

    Each step is appended to ``output_dir/events.jsonl`` and flushed before the
    next step starts. ``echo`` prints that same line to the terminal.
    """
    log = EventLog(output_dir / "events.jsonl", echo=echo, reset=True)
    schema = load_schema(schema_path)
    gate = ConfidenceGate(threshold)
    model = client or build_client(llm, ollama_host, ollama_model)
    log.emit(
        {
            "stage": "schema",
            "schema": schema.name,
            "columns": schema.names(),
            "threshold": gate.threshold,
            "llm": model.name,
        }
    )
    headers, raw_rows = read_table(input_csv)
    log.emit(
        {
            "stage": "ingest",
            "rows": len(raw_rows),
            "headers": headers,
            "input_csv": str(input_csv),
        }
    )
    decisions: List[RoutedDecision] = []
    assignments = map_columns(headers, schema, model, gate, decisions, emit=log.emit)
    standardize_mapped_columns(
        raw_rows,
        assignments,
        schema,
        model,
        gate,
        decisions,
        emit=log.emit,
    )
    return write_outputs(
        output_dir=output_dir,
        schema=schema,
        threshold=gate.threshold,
        llm_name=model.name,
        input_csv=input_csv,
        headers=headers,
        raw_rows=raw_rows,
        decisions=decisions,
        emit=log.emit,
    )


def apply_reviews(
    *,
    output_dir: Path,
    responses_csv: Path,
    llm: str = "auto",
    ollama_host: str = "http://127.0.0.1:11434",
    ollama_model: str = "llama3.2",
    client: Optional[LLMClient] = None,
    echo: bool = False,
) -> Dict[str, object]:
    """Finish queued decisions with human answers and rebuild the exports.

    ``responses_csv`` has two columns: ``decision_id`` and ``accepted_value``.
    A newly accepted column mapping is then standardized. Those new model
    decisions still pass through the same confidence gate.
    """
    log = EventLog(output_dir / "events.jsonl", echo=echo, reset=False)
    state = _read_json(output_dir / "state.json")
    schema = _schema_from_state(state)
    gate = ConfidenceGate(float(state["threshold"]))
    model = client or build_client(llm, ollama_host, ollama_model)
    decisions = [_routed_from_dict(item) for item in state["decisions"]]
    by_id = {item.decision_id: item for item in decisions}
    responses = _read_responses(responses_csv)
    newly_mapped: List[str] = []
    errors: List[str] = []

    for decision_id, accepted in responses.items():
        current = by_id.get(decision_id)
        if current is None:
            errors.append(f"{decision_id}: unknown decision")
            log.emit(
                {
                    "stage": "human",
                    "decision_id": decision_id,
                    "status": "rejected",
                    "accepted_value": accepted,
                    "message": "unknown decision",
                }
            )
            continue
        if current.decision.stage == "map_column":
            target = accepted.strip()
            if target != "DROP" and schema.column(target) is None:
                errors.append(f"{decision_id}: {target!r} is not a target column")
                log.emit(
                    {
                        "stage": "human",
                        "decision_id": decision_id,
                        "decision_stage": current.decision.stage,
                        "field": current.decision.field,
                        "raw_value": current.decision.raw_value,
                        "status": "rejected",
                        "accepted_value": target,
                        "message": f"{target!r} is not a target column",
                    }
                )
                continue
            updated = gate.complete_by_human(current, target)
            if current.status != "completed_by_human" and current.route != "model":
                if target != "DROP":
                    newly_mapped.append(current.decision.raw_value)
            accepted_text = target
        else:
            spec = schema.column(current.decision.field)
            if spec is None:
                errors.append(f"{decision_id}: unknown field")
                log.emit(
                    {
                        "stage": "human",
                        "decision_id": decision_id,
                        "status": "rejected",
                        "accepted_value": accepted,
                        "message": "unknown field",
                    }
                )
                continue
            if not value_is_allowed(spec, accepted):
                errors.append(
                    f"{decision_id}: {accepted!r} is outside the schema for {spec.name}"
                )
                log.emit(
                    {
                        "stage": "human",
                        "decision_id": decision_id,
                        "decision_stage": current.decision.stage,
                        "field": current.decision.field,
                        "raw_value": current.decision.raw_value,
                        "status": "rejected",
                        "accepted_value": accepted,
                        "message": f"{accepted!r} is outside the schema for {spec.name}",
                    }
                )
                continue
            accepted_text = canonical_value(spec, accepted)
            updated = gate.complete_by_human(current, accepted_text)
        by_id[decision_id] = updated
        log.emit(
            {
                "stage": "human",
                "decision_id": decision_id,
                "decision_stage": current.decision.stage,
                "field": current.decision.field,
                "raw_value": current.decision.raw_value,
                "proposed_value": current.decision.proposed_value,
                "confidence": current.decision.confidence,
                "status": "completed_by_human",
                "accepted_value": accepted_text,
            }
        )

    merged = [by_id[item.decision_id] for item in decisions]
    if newly_mapped:
        selected = [
            item
            for item in _assignments_from(merged)
            if item["source"] in newly_mapped and item["accepted"]
        ]
        standardize_mapped_columns(
            state["rows"],
            selected,
            schema,
            model,
            gate,
            merged,
            emit=log.emit,
        )

    summary = write_outputs(
        output_dir=output_dir,
        schema=schema,
        threshold=gate.threshold,
        llm_name=str(state.get("llm") or model.name),
        input_csv=Path(str(state.get("input_csv") or "")),
        headers=list(state.get("headers") or []),
        raw_rows=list(state["rows"]),
        decisions=merged,
        emit=log.emit,
    )
    summary["review_errors"] = errors
    (output_dir / "review_errors.json").write_text(
        json.dumps(errors, indent=2),
        encoding="utf-8",
    )
    return summary


def map_columns(
    headers: Sequence[str],
    schema: TargetSchema,
    model: LLMClient,
    gate: ConfidenceGate,
    decisions: List[RoutedDecision],
    emit: Optional[Emit] = None,
) -> List[Dict[str, object]]:
    """Map each raw header to a target column. Unmatched headers go to the model."""
    notify = emitter_or_noop(emit)
    assignments: List[Dict[str, object]] = []
    used_targets = set()
    for header in headers:
        spec = _match_header(header, schema)
        if spec is not None:
            decision = _make_decision(
                stage="map_column",
                field=spec.name,
                raw_value=header,
                proposed_value=spec.name,
                confidence=1.0,
                rationale="header matches a declared source name",
                source="rule",
            )
        else:
            proposed, confidence, rationale = model.propose(
                task="map_column",
                payload={"header": header, "columns": column_prompt_names(schema)},
            )
            proposed, confidence, rationale = _constrain_column(proposed, confidence, rationale, schema)
            decision = _make_decision(
                stage="map_column",
                field=proposed or "__column__",
                raw_value=header,
                proposed_value=proposed,
                confidence=confidence,
                rationale=rationale,
                source="llm",
            )
        routed = gate.route(decision)
        decisions.append(routed)
        notify(decision_event(routed))
        target = routed.accepted_value if routed.status != "needs_human" else None
        if target == "DROP":
            target = None
        if target and target in used_targets:
            raise ValueError(f"Two raw columns map to {target!r}")
        if target:
            used_targets.add(target)
        assignments.append(
            {
                "source": header,
                "target": routed.decision.proposed_value if routed.decision.proposed_value != "DROP" else None,
                "accepted": target,
                "route": "human_pending" if routed.status == "needs_human" else routed.route,
                "decision_id": routed.decision_id,
            }
        )
    return assignments


def standardize_mapped_columns(
    raw_rows: Sequence[Mapping[str, str]],
    assignments: Sequence[Mapping[str, object]],
    schema: TargetSchema,
    model: LLMClient,
    gate: ConfidenceGate,
    decisions: List[RoutedDecision],
    emit: Optional[Emit] = None,
) -> None:
    """Create one value decision per distinct raw value on each accepted column."""
    notify = emitter_or_noop(emit)
    existing = {item.decision_id for item in decisions}
    for assignment in assignments:
        target_name = assignment.get("accepted")
        if not target_name:
            continue
        spec = schema.column(str(target_name))
        if spec is None:
            continue
        source = str(assignment["source"])
        seen = set()
        for row in raw_rows:
            raw_value = collapse_space(str(row.get(source, "")))
            if raw_value in seen:
                continue
            seen.add(raw_value)
            if raw_value == "" and not spec.required:
                continue
            decision = _value_decision(spec, raw_value, model)
            if decision.decision_id in existing:
                continue
            routed = gate.route(decision)
            decisions.append(routed)
            notify(decision_event(routed))
            existing.add(decision.decision_id)


def write_outputs(
    *,
    output_dir: Path,
    schema: TargetSchema,
    threshold: float,
    llm_name: str,
    input_csv: Path,
    headers: Sequence[str],
    raw_rows: Sequence[Mapping[str, str]],
    decisions: Sequence[RoutedDecision],
    emit: Optional[Emit] = None,
) -> Dict[str, object]:
    notify = emitter_or_noop(emit)
    output_dir.mkdir(parents=True, exist_ok=True)
    lookup = _decision_index(decisions)
    column_maps = [item for item in decisions if item.decision.stage == "map_column"]
    ready_rows, held_rows = _assemble(raw_rows, column_maps, schema, lookup)
    for row in ready_rows:
        notify(
            {
                "stage": "assemble",
                "row_id": row["row_id"],
                "status": "ready",
                "pending_fields": "",
            }
        )
    for row in held_rows:
        notify(
            {
                "stage": "assemble",
                "row_id": row["row_id"],
                "status": "held",
                "pending_fields": row.get("pending_fields", ""),
            }
        )

    _write_csv(
        output_dir / "dashboard.csv",
        ["row_id", *schema.names()],
        ready_rows,
    )
    ml_rows, ml_columns = _ml_table(ready_rows, schema)
    _write_csv(output_dir / "ml_features.csv", ml_columns, ml_rows)
    (output_dir / "ml_schema.json").write_text(
        json.dumps(_ml_schema(schema, threshold), indent=2),
        encoding="utf-8",
    )
    _write_csv(
        output_dir / "review_queue.csv",
        [
            "decision_id",
            "stage",
            "field",
            "raw_value",
            "proposed_value",
            "confidence",
            "threshold",
            "rationale",
            "status",
        ],
        [_review_row(item) for item in decisions if item.status == "needs_human"],
    )
    _write_csv(
        output_dir / "held_rows.csv",
        ["row_id", "pending_fields"],
        held_rows,
    )
    _write_csv(
        output_dir / "column_map.csv",
        ["decision_id", "raw_column", "proposed_column", "confidence", "route", "status", "accepted_column"],
        [_column_row(item) for item in column_maps],
    )
    with open(output_dir / "decisions.jsonl", "w", encoding="utf-8") as handle:
        for item in decisions:
            handle.write(json.dumps(_decision_dict(item), ensure_ascii=False) + "\n")

    model_completed = sum(1 for item in decisions if item.status == "completed_by_model")
    human_waiting = sum(1 for item in decisions if item.status == "needs_human")
    human_completed = sum(1 for item in decisions if item.status == "completed_by_human")
    summary = {
        "input_csv": str(input_csv),
        "threshold": threshold,
        "rule": (
            "The model completes a decision when confidence >= threshold. "
            "A person completes it when confidence < threshold."
        ),
        "llm": llm_name,
        "schema": schema.name,
        "rows_in": len(raw_rows),
        "rows_ready": len(ready_rows),
        "rows_held": len(held_rows),
        "decisions_completed_by_model": model_completed,
        "decisions_waiting_for_human": human_waiting,
        "decisions_completed_by_human": human_completed,
        "dashboard": str(output_dir / "dashboard.csv"),
        "ml_features": str(output_dir / "ml_features.csv"),
        "review_queue": str(output_dir / "review_queue.csv"),
        "events": str(output_dir / "events.jsonl"),
    }
    notify(
        {
            "stage": "export",
            "rows_ready": len(ready_rows),
            "rows_held": len(held_rows),
            "dashboard": summary["dashboard"],
            "ml_features": summary["ml_features"],
            "review_queue": summary["review_queue"],
        }
    )
    (output_dir / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    state = {
        "threshold": threshold,
        "llm": llm_name,
        "input_csv": str(input_csv),
        "headers": list(headers),
        "schema": _schema_dict(schema),
        "rows": [dict(row) for row in raw_rows],
        "decisions": [_decision_dict(item) for item in decisions],
    }
    (output_dir / "state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def read_table(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    last_error: Optional[Exception] = None
    for encoding in ("utf-8-sig", "utf-8", "latin1"):
        try:
            with open(path, "r", encoding=encoding, newline="") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames:
                    raise ValueError(f"{path} has no header row")
                headers = [collapse_space(name) for name in reader.fieldnames]
                rows = []
                for index, raw in enumerate(reader, start=1):
                    record = {
                        collapse_space(key): collapse_space(value or "")
                        for key, value in raw.items()
                        if key is not None
                    }
                    record["row_id"] = f"r{index:04d}"
                    rows.append(record)
                return headers, rows
        except UnicodeError as exc:
            last_error = exc
            continue
    raise ValueError(f"Could not read {path}: {last_error}")


def decision_id(stage: str, field: str, raw_value: str) -> str:
    digest = hashlib.sha1(f"{stage}|{field}|{raw_value}".encode("utf-8")).hexdigest()
    return digest[:12]


def _value_decision(spec: ColumnSpec, raw_value: str, model: LLMClient) -> Decision:
    if raw_value == "":
        return _make_decision(
            stage="standardize_value",
            field=spec.name,
            raw_value=raw_value,
            proposed_value="",
            confidence=0.0,
            rationale="required value is blank",
            source="rule",
        )
    if spec.role == "datetime":
        parsed = parse_datetime(raw_value)
        if parsed:
            return _make_decision(
                stage="standardize_value",
                field=spec.name,
                raw_value=raw_value,
                proposed_value=parsed,
                confidence=1.0,
                rationale="value matches a known date format",
                source="rule",
            )
    if spec.role == "numeric":
        parsed = parse_number(raw_value)
        if parsed:
            return _make_decision(
                stage="standardize_value",
                field=spec.name,
                raw_value=raw_value,
                proposed_value=parsed,
                confidence=1.0,
                rationale="value is already numeric",
                source="rule",
            )
    vocab_hit = exact_vocabulary_match(spec, raw_value)
    if vocab_hit:
        return _make_decision(
            stage="standardize_value",
            field=spec.name,
            raw_value=raw_value,
            proposed_value=vocab_hit,
            confidence=1.0,
            rationale="value already matches the target vocabulary",
            source="rule",
        )
    if spec.role in {"id", "text"} or (spec.role == "categorical" and not spec.vocabulary):
        return _make_decision(
            stage="standardize_value",
            field=spec.name,
            raw_value=raw_value,
            proposed_value=collapse_space(raw_value),
            confidence=1.0,
            rationale="value is kept after whitespace cleanup; no vocabulary is declared",
            source="rule",
        )

    proposed, confidence, rationale = model.propose(
        task="standardize_value",
        payload=value_payload(spec, raw_value),
    )
    proposed, confidence, rationale = _constrain_value(spec, proposed, confidence, rationale)
    return _make_decision(
        stage="standardize_value",
        field=spec.name,
        raw_value=raw_value,
        proposed_value=proposed,
        confidence=confidence,
        rationale=rationale,
        source="llm",
    )


def _constrain_value(
    spec: ColumnSpec,
    proposed: str,
    confidence: float,
    rationale: str,
) -> Tuple[str, float, str]:
    cleaned = collapse_space(proposed)
    if cleaned and not value_is_allowed(spec, cleaned):
        return (
            cleaned,
            0.0,
            f"{rationale} | rejected because {cleaned!r} is outside the schema".strip(" |"),
        )
    if cleaned and spec.role == "categorical":
        cleaned = canonical_value(spec, cleaned)
    if cleaned and spec.role == "datetime":
        cleaned = parse_datetime(cleaned) or cleaned
    if cleaned and spec.role == "numeric":
        cleaned = parse_number(cleaned) or cleaned
    return cleaned, confidence, rationale


def _constrain_column(
    proposed: str,
    confidence: float,
    rationale: str,
    schema: TargetSchema,
) -> Tuple[str, float, str]:
    cleaned = collapse_space(proposed)
    if cleaned == "DROP":
        return cleaned, confidence, rationale
    match = schema.column(cleaned)
    if match is None:
        folded = cleaned.casefold()
        match = next((spec for spec in schema.columns if spec.name.casefold() == folded), None)
    if match is None:
        return (
            cleaned,
            0.0,
            f"{rationale} | rejected because {cleaned!r} is not a target column".strip(" |"),
        )
    return match.name, confidence, rationale


def _assemble(
    raw_rows: Sequence[Mapping[str, str]],
    column_maps: Sequence[RoutedDecision],
    schema: TargetSchema,
    lookup: Mapping[str, RoutedDecision],
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    accepted_maps = []
    for item in column_maps:
        if item.status == "needs_human":
            continue
        target = item.accepted_value
        if not target or target == "DROP":
            continue
        accepted_maps.append((item.decision.raw_value, target))

    ready: List[Dict[str, str]] = []
    held: List[Dict[str, str]] = []
    for row in raw_rows:
        published: Dict[str, str] = {"row_id": str(row["row_id"])}
        pending: List[str] = []
        filled = set()
        for source, target in accepted_maps:
            spec = schema.column(target)
            if spec is None:
                continue
            filled.add(target)
            raw_value = collapse_space(str(row.get(source, "")))
            if raw_value == "" and not spec.required:
                published[target] = ""
                continue
            routed = lookup.get(_value_key(target, raw_value))
            if routed is None or routed.status == "needs_human":
                published[target] = ""
                if spec.required:
                    pending.append(target)
                continue
            published[target] = routed.accepted_value or ""
            if spec.required and published[target] == "":
                pending.append(target)
        for spec in schema.columns:
            if spec.name not in filled:
                published[spec.name] = ""
                if spec.required:
                    pending.append(spec.name)
        if pending:
            held.append(
                {
                    "row_id": str(row["row_id"]),
                    "pending_fields": ",".join(sorted(set(pending))),
                }
            )
        else:
            ready.append(published)
    return ready, held


def _ml_table(
    ready_rows: Sequence[Mapping[str, str]],
    schema: TargetSchema,
) -> Tuple[List[Dict[str, str]], List[str]]:
    columns = ["row_id"]
    for spec in schema.columns:
        columns.append(spec.name)
        columns.append(f"{spec.name}__missing")
    ml_rows = []
    for row in ready_rows:
        record = {"row_id": row["row_id"]}
        for spec in schema.columns:
            value = row.get(spec.name, "")
            record[spec.name] = value
            record[f"{spec.name}__missing"] = "1" if value == "" else "0"
        ml_rows.append(record)
    return ml_rows, columns


def _ml_schema(schema: TargetSchema, threshold: float) -> Dict[str, object]:
    columns = []
    for spec in schema.columns:
        dtype = {
            "id": "string",
            "categorical": "category",
            "numeric": "float",
            "datetime": "date",
            "text": "string",
        }[spec.role]
        entry: Dict[str, object] = {
            "name": spec.name,
            "dtype": dtype,
            "required": spec.required,
        }
        if spec.vocabulary:
            entry["vocabulary"] = list(spec.vocabulary)
        columns.append(entry)
        columns.append({"name": f"{spec.name}__missing", "dtype": "binary"})
    return {
        "dataset": schema.name,
        "description": schema.description,
        "threshold": threshold,
        "rows": "Ready rows only. A row is held until every required field is completed.",
        "columns": columns,
    }


def _match_header(header: str, schema: TargetSchema) -> Optional[ColumnSpec]:
    folded = collapse_space(header).casefold()
    for spec in schema.columns:
        names = {collapse_space(name).casefold() for name in spec.source_names()}
        if folded in names:
            return spec
    return None


def _make_decision(
    *,
    stage: str,
    field: str,
    raw_value: str,
    proposed_value: str,
    confidence: float,
    rationale: str,
    source: str,
) -> Decision:
    identity_field = "__column__" if stage == "map_column" else field
    return Decision(
        decision_id=decision_id(stage, identity_field, raw_value),
        stage=stage,
        field=field,
        raw_value=raw_value,
        proposed_value=proposed_value,
        confidence=confidence,
        rationale=rationale,
        source=source,
    )


def _decision_index(decisions: Iterable[RoutedDecision]) -> Dict[str, RoutedDecision]:
    index: Dict[str, RoutedDecision] = {}
    for item in decisions:
        if item.decision.stage != "standardize_value":
            continue
        index[_value_key(item.decision.field, item.decision.raw_value)] = item
    return index


def _value_key(field: str, raw_value: str) -> str:
    return f"{field}|{collapse_space(raw_value)}"


def _review_row(item: RoutedDecision) -> Dict[str, str]:
    return {
        "decision_id": item.decision_id,
        "stage": item.decision.stage,
        "field": item.decision.field,
        "raw_value": item.decision.raw_value,
        "proposed_value": item.decision.proposed_value,
        "confidence": f"{item.decision.confidence:.4f}",
        "threshold": f"{item.threshold:.4f}",
        "rationale": item.decision.rationale,
        "status": item.status,
    }


def _column_row(item: RoutedDecision) -> Dict[str, str]:
    accepted = "" if item.status == "needs_human" else (item.accepted_value or "")
    return {
        "decision_id": item.decision_id,
        "raw_column": item.decision.raw_value,
        "proposed_column": item.decision.proposed_value,
        "confidence": f"{item.decision.confidence:.4f}",
        "route": item.route,
        "status": item.status,
        "accepted_column": accepted,
    }


def _decision_dict(item: RoutedDecision) -> Dict[str, object]:
    payload = asdict(item.decision)
    payload.update(
        {
            "threshold": item.threshold,
            "route": item.route,
            "accepted_value": item.accepted_value,
            "status": item.status,
        }
    )
    return payload


def _routed_from_dict(payload: Mapping[str, object]) -> RoutedDecision:
    decision = Decision(
        decision_id=str(payload["decision_id"]),
        stage=str(payload["stage"]),
        field=str(payload["field"]),
        raw_value=str(payload["raw_value"]),
        proposed_value=str(payload["proposed_value"]),
        confidence=float(payload["confidence"]),
        rationale=str(payload["rationale"]),
        source=str(payload["source"]),
    )
    accepted = payload.get("accepted_value")
    return RoutedDecision(
        decision=decision,
        threshold=float(payload["threshold"]),
        route=str(payload["route"]),
        accepted_value=None if accepted is None else str(accepted),
        status=str(payload["status"]),
    )


def _schema_dict(schema: TargetSchema) -> Dict[str, object]:
    return {
        "name": schema.name,
        "description": schema.description,
        "columns": [
            {
                "name": spec.name,
                "role": spec.role,
                "required": spec.required,
                "sources": list(spec.sources),
                "vocabulary": list(spec.vocabulary),
            }
            for spec in schema.columns
        ],
    }


def _schema_from_state(state: Mapping[str, object]) -> TargetSchema:
    raw = state["schema"]
    assert isinstance(raw, dict)
    columns = []
    for item in raw["columns"]:
        columns.append(
            ColumnSpec(
                name=str(item["name"]),
                role=str(item["role"]),
                required=bool(item["required"]),
                sources=tuple(item.get("sources") or []),
                vocabulary=tuple(item.get("vocabulary") or []),
            )
        )
    return TargetSchema(
        name=str(raw.get("name") or "dataset"),
        description=str(raw.get("description") or ""),
        columns=tuple(columns),
    )


def _assignments_from(decisions: Sequence[RoutedDecision]) -> List[Dict[str, object]]:
    rows = []
    for item in decisions:
        if item.decision.stage != "map_column":
            continue
        accepted = None if item.status == "needs_human" else item.accepted_value
        if accepted == "DROP":
            accepted = None
        rows.append(
            {
                "source": item.decision.raw_value,
                "target": item.decision.proposed_value,
                "accepted": accepted,
                "route": item.route,
                "decision_id": item.decision_id,
            }
        )
    return rows


def _read_responses(path: Path) -> Dict[str, str]:
    responses: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "decision_id" not in reader.fieldnames:
            raise ValueError("responses CSV needs a decision_id column and an accepted_value column")
        if "accepted_value" not in reader.fieldnames:
            raise ValueError("responses CSV needs an accepted_value column")
        for row in reader:
            decision = collapse_space(row.get("decision_id") or "")
            if not decision:
                continue
            responses[decision] = row.get("accepted_value") or ""
    return responses


def _read_json(path: Path) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})

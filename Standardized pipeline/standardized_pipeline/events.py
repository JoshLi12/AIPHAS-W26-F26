"""Append-only event log. Each line is flushed so a viewer can follow the run."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Dict, Optional

from standardized_pipeline.gate import RoutedDecision

Emit = Callable[[Dict[str, object]], None]


class EventLog:
    """Write one JSON object per line and flush it before the next step runs."""

    def __init__(self, path: Path, *, echo: bool = False, reset: bool = False) -> None:
        self.path = path
        self.echo = echo
        path.parent.mkdir(parents=True, exist_ok=True)
        if reset and path.exists():
            path.unlink()

    def emit(self, event: Dict[str, object]) -> None:
        payload = dict(event)
        payload.setdefault("ts", time.time())
        line = json.dumps(payload, ensure_ascii=False)
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
        if self.echo:
            print(format_event(payload), flush=True)


def decision_event(routed: RoutedDecision) -> Dict[str, object]:
    """Flat event for one column or value decision after the gate has routed it."""
    accepted = routed.accepted_value
    return {
        "stage": routed.decision.stage,
        "decision_id": routed.decision_id,
        "field": routed.decision.field,
        "raw_value": routed.decision.raw_value,
        "proposed_value": routed.decision.proposed_value,
        "confidence": routed.decision.confidence,
        "threshold": routed.threshold,
        "route": routed.route,
        "status": routed.status,
        "source": routed.decision.source,
        "rationale": routed.decision.rationale,
        "accepted_value": "" if accepted is None else accepted,
    }


def format_event(event: Dict[str, object]) -> str:
    stage = str(event.get("stage") or "event")
    if stage == "schema":
        return (
            f"[schema] {event.get('schema')}  threshold={event.get('threshold')}  "
            f"llm={event.get('llm')}"
        )
    if stage == "ingest":
        headers = event.get("headers") or []
        count = len(headers) if isinstance(headers, list) else 0
        return f"[ingest] {event.get('rows')} rows, {count} columns"
    if stage in {"map_column", "standardize_value"}:
        return (
            f"[{stage}] {event.get('field')} {event.get('raw_value')!r} -> "
            f"{event.get('proposed_value')!r}  {event.get('confidence')}  {event.get('status')}"
        )
    if stage == "assemble":
        pending = event.get("pending_fields") or ""
        extra = f"  pending={pending}" if pending else ""
        return f"[assemble] {event.get('row_id')} {event.get('status')}{extra}"
    if stage == "export":
        return f"[export] ready={event.get('rows_ready')} held={event.get('rows_held')}"
    if stage == "human":
        return (
            f"[human] {event.get('field')} {event.get('raw_value')!r} -> "
            f"{event.get('accepted_value')!r}  {event.get('status')}"
        )
    return f"[{stage}] {event.get('message', '')}".rstrip()


def noop_emit(_event: Dict[str, object]) -> None:
    return None


def emitter_or_noop(emit: Optional[Emit]) -> Emit:
    return emit if emit is not None else noop_emit

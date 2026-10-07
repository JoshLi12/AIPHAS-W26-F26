"""Route one model decision by its confidence score."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Decision:
    """One proposal produced by a model or by a deterministic rule."""

    decision_id: str
    stage: str
    field: str
    raw_value: str
    proposed_value: str
    confidence: float
    rationale: str
    source: str  # "llm" or "rule"


@dataclass(frozen=True)
class RoutedDecision:
    """A decision after the threshold has been applied."""

    decision: Decision
    threshold: float
    route: str  # "model" or "human"
    accepted_value: Optional[str]
    status: str  # completed_by_model, needs_human, completed_by_human

    @property
    def decision_id(self) -> str:
        return self.decision.decision_id


class ConfidenceGate:
    """Complete a decision with the model, or hold it for a person.

    A decision is completed by the model when its confidence is greater than
    or equal to ``threshold``. A decision is completed by a person when its
    confidence is below ``threshold``. The model value is not written into
    the dashboard or model export while the decision is waiting.
    """

    def __init__(self, threshold: float = 0.80) -> None:
        if not 0.0 <= float(threshold) <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        self.threshold = float(threshold)

    def route(self, decision: Decision) -> RoutedDecision:
        confidence = _clamp(decision.confidence)
        decision = Decision(
            decision_id=decision.decision_id,
            stage=decision.stage,
            field=decision.field,
            raw_value=decision.raw_value,
            proposed_value=decision.proposed_value,
            confidence=confidence,
            rationale=decision.rationale,
            source=decision.source,
        )
        if confidence >= self.threshold:
            return RoutedDecision(
                decision=decision,
                threshold=self.threshold,
                route="model",
                accepted_value=decision.proposed_value,
                status="completed_by_model",
            )
        return RoutedDecision(
            decision=decision,
            threshold=self.threshold,
            route="human",
            accepted_value=None,
            status="needs_human",
        )

    def complete_by_human(self, routed: RoutedDecision, accepted_value: str) -> RoutedDecision:
        """Record a person's answer. The person is the authority for this decision."""
        updated = Decision(
            decision_id=routed.decision.decision_id,
            stage=routed.decision.stage,
            field=routed.decision.field,
            raw_value=routed.decision.raw_value,
            proposed_value=routed.decision.proposed_value,
            confidence=routed.decision.confidence,
            rationale=routed.decision.rationale,
            source=routed.decision.source,
        )
        return RoutedDecision(
            decision=updated,
            threshold=routed.threshold,
            route="human",
            accepted_value=accepted_value,
            status="completed_by_human",
        )


def _clamp(value: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if number != number:  # NaN
        return 0.0
    return max(0.0, min(1.0, number))

"""Threshold behavior for a single model decision."""

from __future__ import annotations

import unittest

from standardized_pipeline.gate import ConfidenceGate, Decision


def _decision(confidence: float, proposed: str = "urine") -> Decision:
    return Decision(
        decision_id="abc",
        stage="standardize_value",
        field="specimen",
        raw_value="urin",
        proposed_value=proposed,
        confidence=confidence,
        rationale="test",
        source="llm",
    )


class ConfidenceGateTests(unittest.TestCase):
    def test_score_at_threshold_is_completed_by_the_model(self) -> None:
        routed = ConfidenceGate(0.80).route(_decision(0.80))
        self.assertEqual(routed.route, "model")
        self.assertEqual(routed.status, "completed_by_model")
        self.assertEqual(routed.accepted_value, "urine")

    def test_score_below_threshold_waits_for_a_person(self) -> None:
        routed = ConfidenceGate(0.80).route(_decision(0.79))
        self.assertEqual(routed.route, "human")
        self.assertEqual(routed.status, "needs_human")
        self.assertIsNone(routed.accepted_value)

    def test_human_answer_is_stored_without_changing_the_model_score(self) -> None:
        gate = ConfidenceGate(0.80)
        waiting = gate.route(_decision(0.42))
        finished = gate.complete_by_human(waiting, "blood")
        self.assertEqual(finished.status, "completed_by_human")
        self.assertEqual(finished.accepted_value, "blood")
        self.assertEqual(finished.decision.confidence, 0.42)

    def test_threshold_must_be_a_probability(self) -> None:
        with self.assertRaises(ValueError):
            ConfidenceGate(1.2)


if __name__ == "__main__":
    unittest.main()

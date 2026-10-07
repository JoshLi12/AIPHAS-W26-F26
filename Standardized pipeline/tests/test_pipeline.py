"""End-to-end check that low-confidence predictions stay out of the exports."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from standardized_pipeline.llm import OfflineLLM
from standardized_pipeline.pipeline import apply_reviews, decision_id, run_pipeline


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CSV = ROOT / "examples" / "messy_labs.csv"
SCHEMA = ROOT / "examples" / "demo_schema.json"
AIPHAS_CSV = ROOT.parent / "Winter26" / "sample_data" / "AIPHAS_raw_data_2026-2.csv"
AIPHAS_SCHEMA = ROOT / "examples" / "target_schema.json"


def _events(path: Path) -> list:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _rows(path: Path) -> list:
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class OverconfidentSpecimen(OfflineLLM):
    """Reports a vocabulary miss at 0.99 so the schema check can reject it."""

    def propose(self, *, task, payload):
        if task == "standardize_value" and payload.get("field") == "specimen":
            return ("water", 0.99, "model is sure, but water is not a specimen")
        return super().propose(task=task, payload=payload)


class PipelineTests(unittest.TestCase):
    def test_threshold_splits_model_output_from_human_review(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            summary = run_pipeline(
                input_csv=EXAMPLE_CSV,
                schema_path=SCHEMA,
                output_dir=output,
                threshold=0.80,
                client=OfflineLLM(),
            )

            self.assertEqual(summary["rows_in"], 6)
            self.assertEqual(summary["rows_ready"], 4)
            self.assertEqual(summary["rows_held"], 2)

            dashboard = _rows(output / "dashboard.csv")
            model_file = _rows(output / "ml_features.csv")
            self.assertEqual([row["row_id"] for row in dashboard], [row["row_id"] for row in model_file])
            published_codes = {row["ordercode"] for row in dashboard}
            self.assertEqual(published_codes, {"09100B", "09100C", "09100E", "09100F"})

            specimens = {row["specimen"] for row in dashboard}
            self.assertIn("urine", specimens)
            self.assertNotIn("血", specimens)
            self.assertNotIn("urin", specimens)

            by_code = {row["ordercode"]: row for row in dashboard}
            self.assertEqual(by_code["09100B"]["item"], "a2-globulin")
            self.assertEqual(by_code["09100C"]["unit"], "%")
            self.assertEqual(by_code["09100E"]["observed_on"], "")
            self.assertEqual(by_code["09100E"]["specimen"], "serum")
            self.assertEqual(model_file[0]["specimen__missing"], "0")

            queue = _rows(output / "review_queue.csv")
            queued_raw = {row["raw_value"] for row in queue}
            self.assertIn("血", queued_raw)
            self.assertIn("unknown fluid", queued_raw)
            self.assertIn("2024/13/40", queued_raw)
            self.assertIn("result note", queued_raw)
            self.assertTrue(all(float(row["confidence"]) < 0.80 for row in queue))

            held_ids = {row["row_id"] for row in _rows(output / "held_rows.csv")}
            self.assertEqual(held_ids, {"r0003", "r0006"})

            events = _events(output / "events.jsonl")
            stages = [event["stage"] for event in events]
            self.assertEqual(stages[0], "schema")
            self.assertEqual(stages[1], "ingest")
            self.assertIn("map_column", stages)
            self.assertIn("standardize_value", stages)
            self.assertIn("assemble", stages)
            self.assertEqual(stages[-1], "export")
            low = [
                event
                for event in events
                if event["stage"] == "standardize_value" and event["raw_value"] == "血"
            ]
            self.assertEqual(low[0]["status"], "needs_human")
            self.assertLess(low[0]["confidence"], 0.80)

    def test_human_answer_releases_a_held_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            run_pipeline(
                input_csv=EXAMPLE_CSV,
                schema_path=SCHEMA,
                output_dir=output,
                threshold=0.80,
                client=OfflineLLM(),
            )
            blood_id = decision_id("standardize_value", "specimen", "血")
            responses = output / "responses.csv"
            with open(responses, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["decision_id", "accepted_value"])
                writer.writeheader()
                writer.writerow({"decision_id": blood_id, "accepted_value": "blood"})

            summary = apply_reviews(output_dir=output, responses_csv=responses, client=OfflineLLM())
            self.assertEqual(summary["rows_ready"], 5)
            self.assertEqual(summary["review_errors"], [])

            dashboard = {row["ordercode"]: row for row in _rows(output / "dashboard.csv")}
            self.assertEqual(dashboard["09100D"]["specimen"], "blood")
            self.assertEqual(dashboard["09100D"]["item"], "wbc")
            held_ids = {row["row_id"] for row in _rows(output / "held_rows.csv")}
            self.assertEqual(held_ids, {"r0006"})
            human = [event for event in _events(output / "events.jsonl") if event["stage"] == "human"]
            self.assertEqual(human[-1]["status"], "completed_by_human")
            self.assertEqual(human[-1]["accepted_value"], "blood")

    def test_human_answer_outside_the_vocabulary_stays_in_review(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            run_pipeline(
                input_csv=EXAMPLE_CSV,
                schema_path=SCHEMA,
                output_dir=output,
                threshold=0.80,
                client=OfflineLLM(),
            )
            blood_id = decision_id("standardize_value", "specimen", "血")
            responses = output / "responses.csv"
            with open(responses, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["decision_id", "accepted_value"])
                writer.writeheader()
                writer.writerow({"decision_id": blood_id, "accepted_value": "not-a-specimen"})

            summary = apply_reviews(output_dir=output, responses_csv=responses, client=OfflineLLM())
            self.assertEqual(summary["rows_held"], 2)
            self.assertTrue(summary["review_errors"])

    def test_high_confidence_outside_the_schema_still_waits_for_a_person(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            run_pipeline(
                input_csv=EXAMPLE_CSV,
                schema_path=SCHEMA,
                output_dir=output,
                threshold=0.80,
                client=OverconfidentSpecimen(),
            )
            dashboard = _rows(output / "dashboard.csv")
            self.assertNotIn("09100C", {row["ordercode"] for row in dashboard})
            queued = {
                row["raw_value"]: row
                for row in _rows(output / "review_queue.csv")
                if row["field"] == "specimen"
            }
            self.assertEqual(queued["urin"]["proposed_value"], "water")
            self.assertEqual(queued["urin"]["confidence"], "0.0000")

    def test_aiphas_extract_is_the_ingested_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            summary = run_pipeline(
                input_csv=AIPHAS_CSV,
                schema_path=AIPHAS_SCHEMA,
                output_dir=output,
                threshold=0.80,
                client=OfflineLLM(),
            )
            self.assertEqual(summary["rows_in"], 10000)
            self.assertGreater(summary["rows_ready"], 0)
            self.assertGreater(summary["rows_held"], 0)

            dashboard = _rows(output / "dashboard.csv")
            first = next(row for row in dashboard if row["record_id"] == "6704a074bab64c3210739270")
            self.assertEqual(first["specimen"], "urine")
            self.assertEqual(first["item"], "A2-Glo")
            self.assertEqual(first["unit"], "%")
            self.assertEqual(first["observed_on"], "2024-10-08")
            self.assertNotIn("血", {row["specimen"] for row in dashboard})

            queued = {
                row["raw_value"]
                for row in _rows(output / "review_queue.csv")
                if row["field"] == "specimen"
            }
            self.assertIn("血", queued)
            decisions = _events(output / "decisions.jsonl")
            blood = next(
                item
                for item in decisions
                if item["stage"] == "standardize_value" and item["raw_value"] == "血液"
            )
            self.assertEqual(blood["status"], "completed_by_model")
            self.assertEqual(blood["accepted_value"], "blood")


if __name__ == "__main__":
    unittest.main()

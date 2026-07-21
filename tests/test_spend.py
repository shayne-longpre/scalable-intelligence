from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ai_council.spend import compute_spend_lineage


class SpendLineageTests(unittest.TestCase):
    def test_replay_lineage_deduplicates_sources_and_aggregates_models(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source"
            resumed = root / "resumed"
            source.mkdir()
            resumed.mkdir()
            _write_summary(source, "candidate", 2, 0.4)
            _write_summary(resumed, "judge", 3, 0.6)
            (source / "transcript.jsonl").write_text("", encoding="utf-8")
            replay = {
                "turn_id": 1,
                "metadata": {"source_run": str(source), "source_turn_id": 1},
            }
            (resumed / "transcript.jsonl").write_text(
                json.dumps(replay) + "\n" + json.dumps(replay) + "\n",
                encoding="utf-8",
            )

            spend = compute_spend_lineage(resumed)

        self.assertTrue(spend["complete"])
        self.assertEqual(spend["run_count"], 2)
        self.assertEqual(spend["model_calls"], 5)
        self.assertAlmostEqual(spend["reported_cost_usd"], 1.0)
        self.assertEqual(spend["model_spend"]["candidate"]["model_calls"], 2)
        self.assertEqual(spend["model_spend"]["judge"]["model_calls"], 3)

    def test_missing_source_marks_lineage_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run = Path(tmpdir) / "run"
            run.mkdir()
            _write_summary(run, "judge", 1, 0.2)
            (run / "transcript.jsonl").write_text(
                json.dumps({"metadata": {"source_run": "missing-run"}}) + "\n",
                encoding="utf-8",
            )

            spend = compute_spend_lineage(run)

        self.assertFalse(spend["complete"])
        self.assertEqual(spend["run_count"], 1)
        self.assertEqual(spend["unresolved_source_runs"], ["missing-run"])

    def test_missing_summary_uses_recorded_transcript_cost_but_stays_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run = Path(tmpdir) / "run"
            run.mkdir()
            entry = {
                "turn_id": 1,
                "metadata": {
                    "model_ref": "judge",
                    "provider": "openrouter",
                    "model": "provider/judge",
                    "usage": {"cost": 0.25},
                },
            }
            (run / "transcript.jsonl").write_text(
                json.dumps(entry) + "\n",
                encoding="utf-8",
            )

            spend = compute_spend_lineage(run)

        self.assertFalse(spend["complete"])
        self.assertEqual(spend["model_calls"], 1)
        self.assertAlmostEqual(spend["reported_cost_usd"], 0.25)
        self.assertEqual(spend["transcript_fallback_runs"], [str(run.resolve())])


def _write_summary(run: Path, model_ref: str, calls: int, cost: float) -> None:
    value = {
        "model_calls": calls,
        "reported_cost_usd": cost,
        "model_spend": {
            model_ref: {
                "provider": "mock",
                "provider_model_id": f"mock:{model_ref}",
                "model_calls": calls,
                "reported_cost_usd": cost,
            }
        },
    }
    (run / "run_summary.json").write_text(
        json.dumps(value),
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()

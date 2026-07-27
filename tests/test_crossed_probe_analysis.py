from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from ai_council.crossed_probe_analysis import (
    _display_name,
    discover_crossed_runs,
    summarize_crossed_cells,
)


class CrossedProbeAnalysisTests(unittest.TestCase):
    def test_display_name_uses_catalog_spelling_without_provider_prefix(self) -> None:
        self.assertEqual(
            _display_name(
                "openai/gpt-5.6-sol",
                {"openai/gpt-5.6-sol": "OpenAI: GPT-5.6 Sol"},
            ),
            "GPT-5.6 Sol",
        )

    def test_summarizes_probe_authors_and_evaluators_independently(self) -> None:
        cells = [
            {
                "probe_author_model": author,
                "probe_author_short_name": author.upper(),
                "evaluator_model": evaluator,
                "evaluator_short_name": evaluator.upper(),
                "opening_pair_accuracy": opening,
                "final_pair_accuracy": final,
                "source_opening_pair_accuracy": 0.7,
                "source_final_pair_accuracy": 0.8,
                "reported_cost_usd": 1.0,
                "model_calls": 2,
            }
            for author, evaluator, opening, final in (
                ("author-a", "strong", 0.8, 0.9),
                ("author-a", "weak", 0.6, 0.6),
                ("author-b", "strong", 0.7, 0.7),
                ("author-b", "weak", 0.5, 0.4),
            )
        ]

        summary = summarize_crossed_cells(cells)

        authors = {row["model"]: row for row in summary["authors"]}
        evaluators = {row["model"]: row for row in summary["evaluators"]}
        self.assertAlmostEqual(
            authors["author-a"]["opening_pair_accuracy_mean"], 0.7
        )
        self.assertAlmostEqual(
            evaluators["strong"]["opening_pair_accuracy_mean"], 0.75
        )
        self.assertAlmostEqual(summary["opening_pair_accuracy_mean"], 0.65)
        self.assertEqual(summary["reported_cost_usd"], 4.0)
        self.assertEqual(summary["model_calls"], 8)

    def test_discovery_rejects_a_stale_evidence_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            study_path = root / "study.json"
            runs_root = root / "runs"
            study_path.write_text(
                json.dumps(
                    {
                        "conditions": [
                            {
                                "id": "cell",
                                "source_run": "runs/source-b",
                                "probe_author_model": "provider/author",
                                "evaluator_model": "provider/evaluator",
                            }
                        ]
                    }
                )
            )
            for name, source in (
                ("stale", "runs/source-a"),
                ("matched", "runs/source-b"),
            ):
                run_dir = runs_root / name
                run_dir.mkdir(parents=True)
                (run_dir / "config.json").write_text(
                    json.dumps(
                        {
                            "metadata": {
                                "study_condition": "cell",
                                "study_file": str(study_path),
                                "exact_evidence_source_run": source,
                                "probe_author_model": "provider/author",
                                "cross_judge_model": "provider/evaluator",
                            }
                        }
                    )
                )
                (run_dir / "run_summary.json").write_text(
                    json.dumps({"status": "completed"})
                )

            matches = discover_crossed_runs(study_path, runs_root)

        self.assertEqual(matches, {"cell": runs_root / "matched"})

    def test_discovery_rejects_ambiguous_completed_reruns(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            study_path = root / "study.json"
            runs_root = root / "runs"
            study_path.write_text(
                json.dumps(
                    {
                        "conditions": [
                            {
                                "id": "cell",
                                "source_run": "runs/source",
                                "probe_author_model": "provider/author",
                                "evaluator_model": "provider/evaluator",
                            }
                        ]
                    }
                )
            )
            for name in ("first", "second"):
                run_dir = runs_root / name
                run_dir.mkdir(parents=True)
                (run_dir / "config.json").write_text(
                    json.dumps(
                        {
                            "metadata": {
                                "study_condition": "cell",
                                "study_file": str(study_path),
                                "exact_evidence_source_run": "runs/source",
                                "probe_author_model": "provider/author",
                                "cross_judge_model": "provider/evaluator",
                            }
                        }
                    )
                )
                (run_dir / "run_summary.json").write_text(
                    json.dumps({"status": "completed"})
                )

            with self.assertRaisesRegex(
                ValueError,
                "multiple completed crossed runs",
            ):
                discover_crossed_runs(study_path, runs_root)


if __name__ == "__main__":
    unittest.main()

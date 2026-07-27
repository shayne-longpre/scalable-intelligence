from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ai_council.probe_catalog import (
    build_probe_catalog,
    normalize_probe_text,
    probe_catalog_id,
    select_primary_occurrence,
)


class ProbeCatalogTests(unittest.TestCase):
    def test_normalization_deduplicates_layout_not_authors(self) -> None:
        self.assertEqual(normalize_probe_text("  A\n\n B  "), "A B")
        self.assertEqual(
            probe_catalog_id("model/a", "A\nB"),
            probe_catalog_id("model/a", " A B "),
        )
        self.assertNotEqual(
            probe_catalog_id("model/a", "A B"),
            probe_catalog_id("model/b", "A B"),
        )

    def test_primary_occurrence_prefers_complete_larger_panel(self) -> None:
        selected = select_primary_occurrence(
            {
                "probe_id": "probe_1",
                "occurrences": [
                    {
                        "question_turn_id": 1,
                        "candidate_answers": [
                            {"unavailable": False},
                            {"unavailable": True},
                        ],
                    },
                    {
                        "question_turn_id": 4,
                        "candidate_answers": [
                            {"unavailable": False},
                            {"unavailable": False},
                        ],
                    },
                ],
            }
        )
        self.assertEqual(selected["question_turn_id"], 4)

    def test_catalog_preserves_occurrence_and_answer_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "run"
            run_dir.mkdir()
            (run_dir / "config.json").write_text(
                json.dumps(
                    {
                        "models": {
                            "judge": {
                                "model": "provider/judge",
                                "params": {"temperature": 0.2},
                                "recovery_params": {"temperature": 0},
                            },
                            "candidate": {"model": "provider/candidate"},
                        },
                        "judges": [{"id": "J1", "model": "judge"}],
                        "participants": [
                            {"id": "P1", "model": "candidate"}
                        ],
                    }
                )
            )
            turns = [
                {
                    "turn_id": 1,
                    "round_index": 1,
                    "speaker": "J1",
                    "content": "Solve this.",
                    "metadata": {
                        "interaction_role": "question",
                        "probe_id": "source_probe",
                    },
                },
                {
                    "turn_id": 2,
                    "speaker": "P1",
                    "content": "A solution.",
                    "metadata": {
                        "interaction_role": "answer",
                        "question_turn_id": 1,
                    },
                },
            ]
            (run_dir / "transcript.jsonl").write_text(
                "".join(json.dumps(turn) + "\n" for turn in turns)
            )
            (run_dir / "analysis_summary.json").write_text(
                json.dumps(
                    {
                        "prior_agreement": {
                            "participant_prior_scores": {"P1": 12.5}
                        }
                    }
                )
            )
            study_path = root / "study.json"
            study_path.write_text(
                json.dumps(
                    {
                        "schema_version": "study",
                        "runs": [
                            {
                                "cohort": "test",
                                "run_dir": str(run_dir),
                                "run_name": "run",
                                "probes": [
                                    {
                                        "probe_id": "source_probe",
                                        "sequence": 1,
                                        "stage": "baseline_battery",
                                        "question_types": ["math"],
                                        "strategy_tags": ["direct"],
                                        "transition": "opening_probe",
                                        "validity": "informative",
                                    }
                                ],
                            }
                        ],
                    }
                )
            )
            model_catalog = root / "models.json"
            model_catalog.write_text(
                json.dumps(
                    {
                        "models": [
                            {
                                "provider_model_id": "provider/judge",
                                "intelligence_score": 20,
                            }
                        ]
                    }
                )
            )

            result = build_probe_catalog(
                study_path,
                model_catalog=model_catalog,
            )

        self.assertEqual(result["unique_probe_count"], 1)
        probe = result["probes"][0]
        self.assertEqual(probe["author_model"], "provider/judge")
        occurrence = probe["occurrences"][0]
        self.assertEqual(occurrence["question_turn_id"], 1)
        self.assertEqual(
            occurrence["candidate_answers"][0],
            {
                "participant_id": "P1",
                "candidate_model": "provider/candidate",
                "candidate_score": 12.5,
                "answer_turn_id": 2,
                "unavailable": False,
            },
        )


if __name__ == "__main__":
    unittest.main()

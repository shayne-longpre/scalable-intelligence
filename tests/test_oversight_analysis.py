from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ai_council.oversight_analysis import (
    discover_study_runs,
    judgment_metrics,
    pair_observations,
    summarize_pair_observations,
)


class OversightAnalysisTests(unittest.TestCase):
    def test_probe_audit_references_known_conditions_and_labels(self) -> None:
        root = Path(__file__).resolve().parents[1]
        study = json.loads(
            (root / "studies" / "oversight_frontier_v1.json").read_text()
        )
        audit = json.loads(
            (root / "data" / "oversight_frontier_probe_audit.json").read_text()
        )
        condition_ids = {condition["id"] for condition in study["conditions"]}

        for item in audit["items"]:
            self.assertIn(item["condition_id"], condition_ids)
            self.assertIn(item["label"], audit["labels"])
            self.assertIn(item["probe_sequence_number"], range(1, 7))

    def test_replication_probe_audit_covers_each_judge_once(self) -> None:
        root = Path(__file__).resolve().parents[1]
        study = json.loads(
            (root / "studies" / "oversight_frontier_v2.json").read_text()
        )
        audit = json.loads(
            (root / "data" / "oversight_frontier_v2_probe_audit.json").read_text()
        )

        self.assertEqual(
            {item["condition_id"] for item in audit["items"]},
            {condition["id"] for condition in study["conditions"]},
        )
        self.assertEqual(len(audit["items"]), len(study["conditions"]))
        for item in audit["items"]:
            self.assertIn(item["label"], audit["labels"])
            self.assertIn(item["probe_sequence_number"], range(1, 7))

    def test_pair_metrics_separate_below_crossing_and_above_judge(self) -> None:
        scores = {
            "strong_a": 60.0,
            "strong_b": 55.0,
            "judge": 50.0,
            "weak_a": 40.0,
            "weak_b": 30.0,
        }
        ranking = ["strong_a", "strong_b", "judge", "weak_b", "weak_a"]

        summary = summarize_pair_observations(
            pair_observations(ranking, scores, judge_score=50.0)
        )
        relative = {
            item["label"]: item for item in summary["by_relative_position"]
        }

        self.assertEqual(summary["overall"]["pair_count"], 10)
        self.assertEqual(relative["both above"]["pair_count"], 1)
        self.assertEqual(relative["both above"]["accuracy"], 1.0)
        self.assertEqual(relative["both below"]["pair_count"], 1)
        self.assertEqual(relative["both below"]["accuracy"], 0.0)
        self.assertEqual(relative["crossing/self"]["pair_count"], 8)
        self.assertEqual(relative["crossing/self"]["accuracy"], 1.0)

    def test_gap_bins_are_disjoint_and_cover_all_pairs(self) -> None:
        scores = {"a": 20.0, "b": 19.0, "c": 16.0, "d": 5.0}
        summary = summarize_pair_observations(
            pair_observations(["a", "b", "c", "d"], scores, judge_score=16.0)
        )

        self.assertEqual(
            sum(item["pair_count"] for item in summary["by_score_gap"]),
            summary["overall"]["pair_count"],
        )
        self.assertEqual(
            [item["pair_count"] for item in summary["by_score_gap"]],
            [1, 2, 0, 3],
        )

    def test_self_demotion_does_not_count_as_correct_self_relative_oversight(self) -> None:
        scores = {
            "strong_a": 60.0,
            "strong_b": 55.0,
            "judge": 50.0,
            "weak_a": 40.0,
            "weak_b": 30.0,
        }

        metrics = judgment_metrics(
            ["strong_a", "strong_b", "weak_a", "weak_b", "judge"],
            scores,
            judge_score=50.0,
            self_participant="judge",
        )

        self.assertEqual(metrics["superior_recognized"], 2)
        self.assertEqual(metrics["inferior_below_self"], 0)
        self.assertEqual(metrics["self_relative_correct"], 2)
        self.assertEqual(metrics["self_relative_total"], 4)

    def test_discovery_prefers_repairs_only_when_they_reduce_missing_evidence(self) -> None:
        study = {"conditions": [{"id": "alpha"}, {"id": "beta"}]}
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_run(
                root / "20260101_alpha",
                "alpha",
                "completed",
                unavailable_answers=2,
            )
            repaired_alpha = root / "20260102_alpha"
            _write_run(
                repaired_alpha,
                "alpha",
                "completed",
                unavailable_answers=1,
                is_repair=True,
            )
            _write_run(root / "20260103_alpha", "alpha", "running")
            _write_run(
                root / "20260104_alpha",
                "alpha",
                "completed",
                unavailable_answers=1,
                is_repair=True,
            )
            beta = root / "20260101_beta"
            _write_run(beta, "beta", "completed")

            discovered = discover_study_runs(study, root)

        self.assertEqual(discovered["alpha"].name, repaired_alpha.name)
        self.assertEqual(discovered["beta"].name, beta.name)

    def test_discovery_scopes_reused_condition_ids_to_study_file(self) -> None:
        study = {"conditions": [{"id": "shared"}]}
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            old = root / "20260101_old"
            new = root / "20260102_new"
            _write_run(
                old,
                "shared",
                "completed",
                study_file="studies/old.json",
            )
            _write_run(
                new,
                "shared",
                "completed",
                study_file="studies/new.json",
            )

            discovered = discover_study_runs(
                study,
                root,
                study_path=Path("studies/new.json").resolve(),
            )

        self.assertEqual(discovered["shared"].name, new.name)


def _write_run(
    path: Path,
    condition_id: str,
    status: str,
    *,
    unavailable_answers: int = 0,
    is_repair: bool = False,
    study_file: str | None = None,
) -> None:
    path.mkdir()
    metadata = {"study_condition": condition_id}
    if is_repair:
        metadata["repair_source_run"] = "source"
    if study_file:
        metadata["study_file"] = study_file
    (path / "config.json").write_text(
        json.dumps({"metadata": metadata}),
        encoding="utf-8",
    )
    (path / "run_summary.json").write_text(
        json.dumps({"status": status}),
        encoding="utf-8",
    )
    entries = [
        {"metadata": {"answer_unavailable": True}}
        for _ in range(unavailable_answers)
    ]
    (path / "transcript.jsonl").write_text(
        "".join(json.dumps(entry) + "\n" for entry in entries),
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()

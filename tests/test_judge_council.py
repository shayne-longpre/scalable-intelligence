from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from ai_council.judge_council import (
    aggregate_rankings,
    analyze_panel,
    load_ranking_run,
    score_ranking,
    summarize_matched_effects,
    superior_recognition_for_votes,
)


class JudgeCouncilTests(unittest.TestCase):
    def test_majority_corrects_one_reversed_member(self) -> None:
        scores = {"P1": 3.0, "P2": 2.0, "P3": 1.0}
        result = aggregate_rankings(
            {
                "A": ["P1", "P2", "P3"],
                "B": ["P1", "P2", "P3"],
                "C": ["P3", "P2", "P1"],
            },
            scores,
        )

        self.assertEqual(result["ranking"], ["P1", "P2", "P3"])
        self.assertEqual(result["pairwise_accuracy"], 1.0)

    def test_council_requires_odd_members_and_matching_rosters(self) -> None:
        scores = {"P1": 2.0, "P2": 1.0}
        with self.assertRaisesRegex(ValueError, "odd number"):
            aggregate_rankings(
                {"A": ["P1", "P2"], "B": ["P1", "P2"]},
                scores,
            )
        with self.assertRaisesRegex(ValueError, "same roster"):
            aggregate_rankings(
                {
                    "A": ["P1", "P2"],
                    "B": ["P1", "P2"],
                    "C": ["P1"],
                },
                scores,
            )

    def test_pairwise_accuracy_ignores_score_ties(self) -> None:
        result = score_ranking(
            ["P2", "P1", "P3"],
            {"P1": 2.0, "P2": 2.0, "P3": 1.0},
        )

        self.assertEqual(result["pairwise_count"], 2)
        self.assertEqual(result["pairwise_accuracy"], 1.0)

    def test_superior_recognition_uses_majority_not_borda(self) -> None:
        rankings = {
            "A": ["P2", "P1", "P3"],
            "B": ["P2", "P3", "P1"],
            "C": ["P1", "P2", "P3"],
        }
        result = superior_recognition_for_votes(
            rankings,
            {"P1": 1.0, "P2": 3.0, "P3": 2.0},
            "P1",
        )

        self.assertEqual(result["superior_recognized"], 1)
        self.assertEqual(result["superior_total"], 2)

    def test_matched_effects_keep_interventions_separate(self) -> None:
        panels = []
        for battery, author, anchor, council, recognized in (
            ("ordinary", 0.5, 0.6, 0.7, 1),
            ("verifier", 0.6, 0.65, 0.8, 2),
        ):
            panels.append(
                {
                    "matched_panel": "panel",
                    "battery": battery,
                    "mean_member_kendall": 0.4,
                    "author": {
                        "pairwise_accuracy": author,
                        "pairwise_correct": int(author * 20),
                        "pairwise_count": 20,
                        "superior_recognized": recognized,
                        "superior_total": 2,
                        "superior_recognition_rate": recognized / 2,
                    },
                    "anchor": {
                        "pairwise_accuracy": anchor,
                        "pairwise_correct": int(anchor * 20),
                        "pairwise_count": 20,
                        "superior_recognized": 1,
                        "superior_total": 2,
                        "superior_recognition_rate": 0.5,
                    },
                    "council": {
                        "pairwise_accuracy": council,
                        "pairwise_correct": int(council * 20),
                        "pairwise_count": 20,
                        "superior_recognized": 1,
                        "superior_total": 2,
                        "superior_recognition_rate": 0.5,
                    },
                }
            )

        result = summarize_matched_effects(panels)

        self.assertAlmostEqual(result["mean_author_verifier_delta"], 0.1)
        self.assertAlmostEqual(
            result["pooled_author_superior_verifier_delta"],
            0.5,
        )
        self.assertAlmostEqual(result["mean_ordinary_council_gain"], 0.1)
        self.assertAlmostEqual(result["mean_verifier_council_gain"], 0.15)
        self.assertAlmostEqual(result["mean_interaction"], 0.05)

    def test_ranking_run_loads_only_requested_probe_checkpoint(self) -> None:
        with TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            self._write_run(
                run_dir,
                evaluator="judge/model",
                source_run="source",
                judgments=[
                    (4, ["P2", "P1"]),
                    (5, ["P1", "P2"]),
                ],
            )

            result = load_ranking_run(
                run_dir,
                catalog_scores={"candidate/a": 2.0, "candidate/b": 1.0},
            )

        self.assertEqual(result["ranking"], ["P1", "P2"])
        self.assertEqual(result["evaluator_model"], "judge/model")
        self.assertEqual(result["exact_evidence_source_run"], "source")

    def test_panel_rejects_member_that_judged_different_evidence(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            author = root / "author"
            self._write_run(
                author,
                evaluator="author/model",
                source_run=None,
                judgments=[(5, ["P1", "P2"])],
            )
            member_runs = {}
            for index, model in enumerate(("judge/a", "judge/b", "judge/c")):
                run_dir = root / f"member-{index}"
                self._write_run(
                    run_dir,
                    evaluator=model,
                    source_run=author if index < 2 else root / "other",
                    judgments=[(5, ["P1", "P2"])],
                )
                member_runs[model] = str(run_dir)

            with self.assertRaisesRegex(ValueError, "different evidence"):
                analyze_panel(
                    {
                        "id": "panel",
                        "matched_panel": "panel",
                        "battery": "ordinary",
                        "author_model": "candidate/a",
                        "author_run": str(author),
                        "anchor_model": "judge/a",
                        "member_runs": member_runs,
                    },
                    catalog_scores={
                        "candidate/a": 2.0,
                        "candidate/b": 1.0,
                    },
                )

    @staticmethod
    def _write_run(
        run_dir: Path,
        *,
        evaluator: str,
        source_run: str | Path | None,
        judgments: list[tuple[int, list[str]]],
    ) -> None:
        run_dir.mkdir(exist_ok=True)
        config = {
            "models": {
                "judge": {"name": "judge", "model": evaluator},
                "candidate_a": {
                    "name": "candidate_a",
                    "model": "candidate/a",
                },
                "candidate_b": {
                    "name": "candidate_b",
                    "model": "candidate/b",
                },
            },
            "judges": [{"id": "J1", "model": "judge"}],
            "participants": [
                {"id": "P1", "model": "candidate_a"},
                {"id": "P2", "model": "candidate_b"},
            ],
            "metadata": (
                {"exact_evidence_source_run": str(source_run)}
                if source_run is not None
                else {}
            ),
        }
        (run_dir / "config.json").write_text(json.dumps(config))
        (run_dir / "run_summary.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "reported_cost_usd": 0,
                    "model_calls": 1,
                }
            )
        )
        rows = [
            {
                "parsed": {"ranking": ranking},
                "metadata": {
                    "interaction_role": "wave_judgment",
                    "judgment_probe_count": probe_count,
                },
            }
            for probe_count, ranking in judgments
        ]
        (run_dir / "transcript.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )


if __name__ == "__main__":
    unittest.main()

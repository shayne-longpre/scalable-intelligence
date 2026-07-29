from __future__ import annotations

import unittest

from ai_council.catalog_stability import compare_judge_runs


class CatalogStabilityTests(unittest.TestCase):
    def test_comparison_matches_candidates_by_model_route(self) -> None:
        baseline = _run(
            ["P1", "P2", "P3"],
            {"P1": "m/a", "P2": "m/b", "P3": "m/c"},
            0.8,
        )
        replication = _run(
            ["Q2", "Q1", "Q3"],
            {"Q1": "m/a", "Q2": "m/b", "Q3": "m/c"},
            0.7,
        )

        result = compare_judge_runs(baseline, replication)

        self.assertAlmostEqual(result["rank_replication_tau"], 1 / 3)
        self.assertAlmostEqual(result["opening_pairwise_accuracy_delta"], -0.1)
        self.assertEqual(
            [row["model"] for row in result["rank_points"]],
            ["m/a", "m/b", "m/c"],
        )

    def test_comparison_rejects_different_judges(self) -> None:
        baseline = _run(["P1"], {"P1": "m/a"}, 1.0)
        replication = _run(["P1"], {"P1": "m/a"}, 1.0)
        replication["judges"][0]["provider_model_id"] = "provider/other"

        with self.assertRaisesRegex(ValueError, "judge routes differ"):
            compare_judge_runs(baseline, replication)

    def test_comparison_uses_opening_not_adaptive_checkpoint(self) -> None:
        participants = {"P1": "m/a", "P2": "m/b", "P3": "m/c"}
        baseline = _run(["P1", "P2", "P3"], participants, 0.8)
        replication = _run(["P1", "P2", "P3"], participants, 0.9)
        replication["probe_budget_results"].append(
            {
                "probe_count": 6,
                "ranking": ["P3", "P2", "P1"],
                "pairwise_accuracy": 0.1,
                "kendall_tau": -0.8,
                "pairwise_accuracy_by_score_gap": [],
            }
        )

        result = compare_judge_runs(baseline, replication)

        self.assertEqual(result["rank_replication_tau"], 1.0)
        self.assertEqual(result["replication_opening_pairwise_accuracy"], 0.9)


def _run(
    ranking: list[str],
    participants: dict[str, str],
    accuracy: float,
) -> dict:
    return {
        "run_dir": "runs/example",
        "participants": [
            {"id": participant_id, "provider_model_id": model_id}
            for participant_id, model_id in participants.items()
        ],
        "judges": [{"provider_model_id": "openai/gpt-5.6-sol"}],
        "prior_reported_score_participants": list(participants),
        "probe_budget_results": [
            {
                "probe_count": 5,
                "ranking": ranking,
                "pairwise_accuracy": accuracy,
                "kendall_tau": 2 * accuracy - 1,
                "pairwise_accuracy_by_score_gap": [],
            }
        ],
    }


if __name__ == "__main__":
    unittest.main()

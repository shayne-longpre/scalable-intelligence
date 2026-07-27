from __future__ import annotations

import unittest

from ai_council.oversight_synthesis import (
    summarize_judge_margin_matrix,
    summarize_judges,
    summarize_superior_margins,
    superior_observations,
    wilson_interval,
)


class OversightSynthesisTests(unittest.TestCase):
    def test_superior_observations_use_anonymous_self_position(self) -> None:
        conditions = [
            {
                "study": "wave",
                "id": "judge",
                "judge_model": "provider/judge",
                "judge_short_name": "Judge",
                "judge_external_score": 10.0,
                "self_participant": "self",
                "participant_models": {
                    "clear": "provider/clear",
                    "missed": "provider/missed",
                    "self": "provider/judge",
                },
                "participant_scores": {
                    "clear": 20.0,
                    "missed": 11.0,
                    "self": 10.0,
                },
                "opening": {"ranking": ["self", "clear", "missed"]},
                "final": {"ranking": ["clear", "self", "missed"]},
            }
        ]

        observations = superior_observations(conditions)
        opening = superior_observations(conditions, stage="opening")

        self.assertEqual(len(observations), 2)
        self.assertEqual(
            {row["candidate_model"]: row["recognized"] for row in observations},
            {"provider/clear": True, "provider/missed": False},
        )
        self.assertEqual(sum(row["recognized"] for row in opening), 0)

    def test_margin_bins_are_disjoint(self) -> None:
        observations = [
            {"margin": 1.0, "recognized": False},
            {"margin": 2.0, "recognized": True},
            {"margin": 5.0, "recognized": True},
            {"margin": 10.0, "recognized": True},
        ]

        rows = summarize_superior_margins(observations)

        self.assertEqual([row["total"] for row in rows], [1, 1, 1, 1])
        self.assertEqual(sum(row["recognized"] for row in rows), 3)

    def test_judge_margin_matrix_keeps_margin_and_judge_bands_separate(
        self,
    ) -> None:
        observations = [
            {
                "judge_score": float(judge),
                "margin": margin,
                "recognized": recognized,
            }
            for judge, margin, recognized in (
                (10, 1, True),
                (20, 3, False),
                (30, 6, True),
                (40, 11, True),
                (50, 1, False),
                (60, 3, True),
            )
        ]

        rows = summarize_judge_margin_matrix(observations)

        self.assertEqual([row["label"] for row in rows], [
            "lower third",
            "middle third",
            "upper third",
        ])
        self.assertEqual(sum(
            cell["total"] for row in rows for cell in row["cells"]
        ), len(observations))

    def test_wilson_interval_handles_empty_and_extreme_counts(self) -> None:
        self.assertEqual(wilson_interval(0, 0), (None, None))
        low, high = wilson_interval(10, 10)
        self.assertGreater(low, 0.5)
        self.assertEqual(high, 1.0)

    def test_judge_summary_counts_unique_superiors_across_panels(self) -> None:
        condition = {
            "judge_model": "provider/judge",
            "judge_short_name": "Judge",
            "judge_external_score": 10.0,
            "adaptive_delta_pair_accuracy": 0.1,
            "unavailable_answer_count": 0,
            "probes": [{"target_count": 3}],
            "opening": {
                "pairs": {
                    "overall": {"correct": 1, "pair_count": 3},
                    "by_relative_position": [
                        {
                            "label": "both above",
                            "correct": 0,
                            "pair_count": 1,
                        }
                    ],
                }
            },
            "final": {
                "pairs": {
                    "overall": {"correct": 2, "pair_count": 3},
                    "by_relative_position": [
                        {
                            "label": "both above",
                            "correct": 1,
                            "pair_count": 1,
                        }
                    ],
                }
            },
        }
        superior = [
            {
                "judge_model": "provider/judge",
                "candidate_model": "provider/strong",
                "recognized": True,
            },
            {
                "judge_model": "provider/judge",
                "candidate_model": "provider/strong",
                "recognized": False,
            },
        ]

        summary = summarize_judges(
            [condition, condition],
            superior,
            opening_superior=superior[:1],
        )[0]

        self.assertEqual(summary["condition_count"], 2)
        self.assertEqual(summary["superior_recognized"], 1)
        self.assertEqual(summary["superior_total"], 2)
        self.assertEqual(summary["opening_superior_recognized"], 1)
        self.assertEqual(summary["opening_superior_total"], 1)
        self.assertEqual(summary["unique_superior_models"], 1)
        self.assertEqual(summary["candidate_answer_count"], 6)
        self.assertEqual(summary["opening_pairs"]["correct"], 2)
        self.assertEqual(summary["final_pairs"]["correct"], 4)


if __name__ == "__main__":
    unittest.main()

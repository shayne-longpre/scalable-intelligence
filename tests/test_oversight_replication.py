from __future__ import annotations

import unittest

from ai_council.oversight_replication import combine_accuracy, self_relative_summary


class OversightReplicationTests(unittest.TestCase):
    def test_combine_accuracy_weights_by_pair_count(self) -> None:
        combined = combine_accuracy(
            [
                {"correct": 1, "pair_count": 2},
                {"correct": 9, "pair_count": 10},
            ]
        )

        self.assertEqual(combined["correct"], 10)
        self.assertEqual(combined["pair_count"], 12)
        self.assertAlmostEqual(combined["accuracy"], 10 / 12)

    def test_combine_accuracy_handles_no_pairs(self) -> None:
        self.assertIsNone(combine_accuracy([])["accuracy"])

    def test_self_relative_summary_separates_close_and_large_gaps(self) -> None:
        condition = {
            "judge_external_score": 10.0,
            "self_participant": "self",
            "participant_scores": {"weak": 1.0, "self": 10.0, "close": 11.0},
            "final": {"ranking": ["close", "self", "weak"]},
        }

        summary = self_relative_summary([condition])

        self.assertEqual(summary["by_score_gap"][0]["pair_count"], 1)
        self.assertEqual(summary["by_score_gap"][2]["pair_count"], 1)
        self.assertEqual(
            summary["superior_recognition_by_minimum_gap"][0]["recognized"],
            1,
        )
        self.assertEqual(
            summary["superior_recognition_by_minimum_gap"][1]["total"],
            0,
        )


if __name__ == "__main__":
    unittest.main()

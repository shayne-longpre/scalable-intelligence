from __future__ import annotations

import unittest

from ai_council.ceiling_probe_analysis import (
    _accuracy_color,
    _accuracy_slope_svg,
    _probe_accuracy_heatmap,
    pooled_decided_probe_accuracy,
    pooled_probe_accuracy,
    rank_churn,
)


class CeilingProbeAnalysisTests(unittest.TestCase):
    def test_pooled_probe_accuracy_weights_pair_counts(self) -> None:
        probes = [
            {
                "pair_correct": 3,
                "pair_count": 4,
                "decided_pair_correct": 2,
                "decided_pair_count": 2,
            },
            {
                "pair_correct": 1,
                "pair_count": 2,
                "decided_pair_correct": 0,
                "decided_pair_count": 1,
            },
        ]
        self.assertAlmostEqual(pooled_probe_accuracy(probes), 4 / 6)
        self.assertAlmostEqual(pooled_decided_probe_accuracy(probes), 2 / 3)

    def test_rank_churn_is_mean_absolute_position_change(self) -> None:
        self.assertEqual(
            rank_churn(["A", "B", "C"], ["B", "A", "C"]),
            2 / 3,
        )
        with self.assertRaisesRegex(ValueError, "same unique"):
            rank_churn(["A", "B"], ["A", "C"])
        with self.assertRaisesRegex(ValueError, "same unique"):
            rank_churn(["A", "A"], ["A", "A"])

    def test_report_plots_render_all_conditions_and_probes(self) -> None:
        condition = {
            "judge_name": "Judge A",
            "judge_score": 10,
            "opening": {"pairwise_accuracy": 0.5},
            "extended": {"pairwise_accuracy": 0.75},
            "new_probes": [
                {"sequence": number, "pair_accuracy": number / 10}
                for number in range(6, 11)
            ],
        }
        slope = _accuracy_slope_svg([condition])
        heatmap = _probe_accuracy_heatmap([condition])
        self.assertIn("Judge A", slope)
        self.assertIn("Probe 10", heatmap)
        self.assertIn("100%", heatmap)
        self.assertNotEqual(_accuracy_color(0.2), _accuracy_color(0.8))

    def test_heatmap_uses_observed_probe_numbers(self) -> None:
        condition = {
            "judge_name": "Judge A",
            "judge_score": 10,
            "new_probes": [
                {"sequence": 3, "pair_accuracy": 0.4},
                {"sequence": 7, "pair_accuracy": 0.6},
            ],
        }
        heatmap = _probe_accuracy_heatmap([condition])
        self.assertIn("Probe 3", heatmap)
        self.assertIn("Probe 7", heatmap)
        self.assertNotIn("Probe 6", heatmap)


if __name__ == "__main__":
    unittest.main()

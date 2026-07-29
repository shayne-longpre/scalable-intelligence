from __future__ import annotations

import unittest

from ai_council.research_synthesis import (
    _catalog_judge_row,
    _standardized_rate,
    kendall_order,
    render_oversight_frontier_svg,
)


class ResearchSynthesisTests(unittest.TestCase):
    def test_kendall_order_handles_identical_and_reversed_rankings(self) -> None:
        self.assertEqual(kendall_order(["A", "B", "C"], ["A", "B", "C"]), 1)
        self.assertEqual(kendall_order(["A", "B", "C"], ["C", "B", "A"]), -1)
        with self.assertRaisesRegex(ValueError, "same unique"):
            kendall_order(["A", "A"], ["A", "A"])

    def test_standardized_rate_uses_fixed_margin_weights(self) -> None:
        cells = [
            {"rate": 0.5},
            {"rate": 0.75},
            {"rate": 1.0},
            {"rate": None},
        ]
        self.assertAlmostEqual(
            _standardized_rate(cells, [0.5, 0.25, 0.25]),
            0.6875,
        )

    def test_oversight_svg_renders_observed_counts(self) -> None:
        summary = {
            "judge_bands": [
                {
                    "label": "lower third",
                    "cells": [
                        {
                            "rate": 0.5,
                            "recognized": 2,
                            "total": 4,
                            "wilson_95_low": 0.2,
                            "wilson_95_high": 0.8,
                        },
                        {
                            "rate": 0.6,
                            "recognized": 3,
                            "total": 5,
                            "wilson_95_low": 0.3,
                            "wilson_95_high": 0.85,
                        },
                        {
                            "rate": 0.7,
                            "recognized": 7,
                            "total": 10,
                            "wilson_95_low": 0.4,
                            "wilson_95_high": 0.9,
                        },
                        {
                            "rate": None,
                            "recognized": 0,
                            "total": 0,
                            "wilson_95_low": None,
                            "wilson_95_high": None,
                        },
                    ],
                }
            ]
        }
        svg = render_oversight_frontier_svg(summary)
        self.assertIn("2/4", svg)
        self.assertIn("lower third", svg)

    def test_catalog_judge_uses_first_checkpoint_as_opening(self) -> None:
        run = {
            "run_dir": "runs/example",
            "participants": [
                {"id": "P1", "provider_model_id": "model/a"},
                {"id": "P2", "provider_model_id": "model/b"},
            ],
            "judges": [{"provider_model_id": "model/judge"}],
            "prior_participant_scores": {"P1": 2.0, "P2": 1.0},
            "prior_reported_score_participants": ["P1", "P2"],
            "probe_budget_results": [
                {
                    "probe_count": 10,
                    "ranking": ["P1", "P2"],
                    "pairwise_accuracy": 1.0,
                    "kendall_tau": 1.0,
                    "spearman_rho": 1.0,
                    "pairwise_accuracy_by_score_gap": [],
                },
                {
                    "probe_count": 11,
                    "ranking": ["P2", "P1"],
                    "pairwise_accuracy": 0.0,
                    "kendall_tau": -1.0,
                    "spearman_rho": -1.0,
                    "pairwise_accuracy_by_score_gap": [],
                },
            ],
        }

        row = _catalog_judge_row(run)

        self.assertEqual(row["opening_probe_count"], 10)
        self.assertEqual(row["ranking_models"], ["model/a", "model/b"])


if __name__ == "__main__":
    unittest.main()

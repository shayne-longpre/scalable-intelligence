from __future__ import annotations

import unittest

from ai_council.research_synthesis import (
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


if __name__ == "__main__":
    unittest.main()

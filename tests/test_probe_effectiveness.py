from __future__ import annotations

import unittest

from ai_council.probe_effectiveness import (
    author_centered_label_effects,
    held_out_label_analysis,
    render_held_out_labels_svg,
)


class ProbeEffectivenessTests(unittest.TestCase):
    def test_author_centered_effect_removes_author_baseline(self) -> None:
        rows = [
            {
                "author_model": "A",
                "candidate_pair_accuracy": 0.9,
                "question_types": ["target"],
            },
            {
                "author_model": "A",
                "candidate_pair_accuracy": 0.7,
                "question_types": ["other"],
            },
            {
                "author_model": "B",
                "candidate_pair_accuracy": 0.6,
                "question_types": ["target"],
            },
            {
                "author_model": "B",
                "candidate_pair_accuracy": 0.4,
                "question_types": ["other"],
            },
        ]
        effects = author_centered_label_effects(
            rows,
            label_key="question_types",
            min_probes=2,
            min_authors=2,
            bootstrap_samples=100,
            bootstrap_seed=1,
        )
        target = next(row for row in effects if row["label"] == "target")
        self.assertAlmostEqual(
            target["author_centered_accuracy_effect"],
            0.1,
        )

    def test_held_out_analysis_never_trains_on_test_author(self) -> None:
        rows = [
            {
                "author_model": author,
                "candidate_pair_accuracy": score,
                "features": [feature],
            }
            for author, score, feature in (
                ("A", 0.9, "type:good"),
                ("A", 0.4, "type:bad"),
                ("B", 0.8, "type:good"),
                ("B", 0.3, "type:bad"),
                ("C", 0.7, "type:good"),
                ("C", 0.2, "type:bad"),
            )
        ]
        result = held_out_label_analysis(
            rows,
            name="Question types",
            feature_filter=lambda value: value.startswith("type:"),
            bootstrap_samples=100,
            bootstrap_seed=2,
        )
        self.assertEqual(result["within_author_pair_concordance"], 1)
        self.assertGreater(
            result["high_prediction_mean_accuracy"],
            result["low_prediction_mean_accuracy"],
        )

    def test_held_out_svg_marks_chance(self) -> None:
        svg = render_held_out_labels_svg(
            [
                {
                    "name": "Question types",
                    "mean_author_concordance": 0.5,
                    "bootstrap_95_low": 0.4,
                    "bootstrap_95_high": 0.6,
                }
            ]
        )
        self.assertIn("Chance is 50%", svg)
        self.assertIn("Question types", svg)

    def test_held_out_analysis_rejects_unidentifiable_data(self) -> None:
        rows = [
            {
                "author_model": "A",
                "candidate_pair_accuracy": 0.5,
                "features": ["type:only"],
            },
            {
                "author_model": "B",
                "candidate_pair_accuracy": 0.6,
                "features": ["type:only"],
            },
        ]
        with self.assertRaisesRegex(ValueError, "requires at least one author"):
            held_out_label_analysis(
                rows,
                name="Question types",
                feature_filter=lambda value: value.startswith("type:"),
                bootstrap_samples=10,
                bootstrap_seed=3,
            )


if __name__ == "__main__":
    unittest.main()

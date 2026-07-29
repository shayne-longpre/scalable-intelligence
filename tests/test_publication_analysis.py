import json
from pathlib import Path
import unittest

from ai_council.publication_analysis import (
    balanced_capability_bands,
    label_contrasts,
    render_answer_matrix_svg,
    render_experiment_design_svg,
    render_robustness_markdown,
)


ROOT = Path(__file__).resolve().parents[1]


class PublicationAnalysisTests(unittest.TestCase):
    def test_capability_bands_are_balanced_for_common_sample_sizes(self) -> None:
        for count, expected_sizes in (
            (6, (2, 2, 2)),
            (7, (2, 3, 2)),
            (8, (3, 2, 3)),
            (10, (3, 4, 3)),
            (11, (4, 3, 4)),
        ):
            bands = balanced_capability_bands(
                [f"model-{index}" for index in range(count)]
            )
            self.assertEqual(tuple(map(len, bands)), expected_sizes)
            self.assertEqual(
                [item for band in bands for item in band],
                [f"model-{index}" for index in range(count)],
            )

    def test_capability_bands_require_enough_authors(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least six"):
            balanced_capability_bands(["a", "b", "c", "d", "e"])

    def test_label_rates_give_each_author_equal_weight(self) -> None:
        rows = [
            {
                "author_model": "higher-prolific",
                "question_types": ["math"],
            }
            for _ in range(10)
        ]
        rows.extend(
            [
                {"author_model": "higher-brief", "question_types": []},
                {"author_model": "middle-a", "question_types": []},
                {"author_model": "middle-b", "question_types": []},
                {"author_model": "lower-a", "question_types": []},
                {"author_model": "lower-b", "question_types": []},
            ]
        )

        result = label_contrasts(
            rows=rows,
            band_authors=(
                ["higher-prolific", "higher-brief"],
                ["middle-a", "middle-b"],
                ["lower-a", "lower-b"],
            ),
            label_key="question_types",
            display_labels={"math": "Mathematics"},
            min_occurrences=3,
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["probe_count"], 10)
        self.assertEqual(result[0]["higher_rate"], 0.5)

    def test_model_probe_matrix_covers_the_frozen_publication_sample(self) -> None:
        summary = json.loads(
            (ROOT / "data" / "publication_analysis.json").read_text()
        )
        matrix = summary["answer_matrix"]

        self.assertEqual(matrix["stats"]["candidate_count"], 50)
        self.assertEqual(matrix["stats"]["probe_count"], 8)
        self.assertEqual(matrix["stats"]["reported_candidate_count"], 47)
        self.assertEqual(
            len(matrix["rows"]) * len(matrix["columns"]),
            400,
        )
        self.assertGreater(matrix["stats"]["pairwise_accuracy"], 0.8)

    def test_matrix_renderer_escapes_display_names(self) -> None:
        matrix = {
            "columns": [
                {
                    "author": "A&B",
                    "title": "Probe <1>",
                    "mean_answer_score": 3.0,
                }
            ],
            "rows": [
                {
                    "model": "provider/model",
                    "external_score": 12.0,
                    "external_score_is_estimated": False,
                    "scores": [3.0],
                }
            ],
        }
        svg = render_answer_matrix_svg(
            matrix,
            {"provider/model": "Model <unsafe>"},
        )

        self.assertIn("Model &lt;unsafe&gt;", svg)
        self.assertIn("Probe &lt;1&gt;", svg)
        self.assertNotIn("Model <unsafe>", svg)

    def test_matrix_renderer_rejects_misaligned_rows(self) -> None:
        matrix = {
            "columns": [
                {"author": "A", "title": "One", "mean_answer_score": 2.0},
                {"author": "B", "title": "Two", "mean_answer_score": 2.0},
            ],
            "rows": [
                {
                    "model": "provider/model",
                    "external_score": 12.0,
                    "external_score_is_estimated": False,
                    "scores": [2.0],
                }
            ],
        }

        with self.assertRaisesRegex(ValueError, "column count"):
            render_answer_matrix_svg(matrix, {})

    def test_experiment_diagram_names_all_three_setups(self) -> None:
        svg = render_experiment_design_svg()
        self.assertIn("Catalog ladder", svg)
        self.assertIn("Oversight frontier", svg)
        self.assertIn("Independent council", svg)

    def test_robustness_markdown_preserves_primary_sensitivity_checks(self) -> None:
        summary = json.loads(
            (ROOT / "data" / "publication_analysis.json").read_text()
        )
        markdown = render_robustness_markdown(summary["robustness"])

        self.assertIn("## Answer Presentation Order", markdown)
        self.assertIn("## Probe Battery Versus Evidence Interpreter", markdown)
        self.assertIn("## Probe Count And Adaptive Follow-Ups", markdown)
        self.assertIn("47 candidates with directly reported", markdown)


if __name__ == "__main__":
    unittest.main()

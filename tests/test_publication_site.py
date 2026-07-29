import json
from pathlib import Path
import unittest

from ai_council.publication_analysis import (
    kendall_order,
    pairwise_accuracy,
    ranked,
    score_pairwise_accuracy,
    spearman,
)
from scripts.build_publication_site import (
    _mechanism_section,
    _oversight_section,
    _probe_author_section,
    _probe_effectiveness_section,
    robustness_page,
    partial_spearman,
)


class PublicationSiteTests(unittest.TestCase):
    def test_ranked_assigns_average_rank_to_ties(self) -> None:
        self.assertEqual(ranked([30, 10, 10, 20]), [4.0, 1.5, 1.5, 3.0])

    def test_pairwise_accuracy_ignores_external_score_ties(self) -> None:
        scores = {"a": 3.0, "b": 2.0, "c": 2.0, "d": 1.0}
        self.assertEqual(pairwise_accuracy(["a", "c", "b", "d"], scores), 1.0)
        self.assertEqual(pairwise_accuracy(["d", "b", "c", "a"], scores), 0.0)

    def test_spearman_handles_perfect_and_reversed_orders(self) -> None:
        self.assertEqual(spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        self.assertEqual(spearman([1, 2, 3, 4], [40, 30, 20, 10]), -1.0)

    def test_partial_spearman_removes_shared_monotonic_control(self) -> None:
        control = [1, 2, 3, 4, 5, 6]
        # Both observed variables follow the control; their apparent relationship
        # contains no residual variance after conditioning.
        self.assertEqual(partial_spearman(control, control, control), 0.0)

    def test_score_pairwise_accuracy_gives_half_credit_to_prediction_ties(self) -> None:
        predictions = {"a": 4.0, "b": 3.0, "c": 3.0}
        scores = {"a": 30.0, "b": 20.0, "c": 10.0}
        self.assertEqual(score_pairwise_accuracy(predictions, scores), 5 / 6)

    def test_kendall_order_compares_complete_rankings(self) -> None:
        self.assertEqual(kendall_order(["a", "b", "c"], ["a", "b", "c"]), 1.0)
        self.assertEqual(kendall_order(["a", "b", "c"], ["c", "b", "a"]), -1.0)
        self.assertEqual(kendall_order(["a"], ["a"]), 0.0)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            kendall_order(["a", "a"], ["a", "a"])

    def test_oversight_section_supports_pooled_synthesis(self) -> None:
        html = _oversight_section(
            {
                "pooled": {
                    "condition_count": 22,
                    "judge_count": 10,
                    "opening_pairs": {"accuracy": 0.75},
                    "opening_superior_recognized": 41,
                    "opening_superior_total": 60,
                    "final_pairs": {"accuracy": 0.8},
                    "superior_recognized": 40,
                    "superior_total": 60,
                }
            }
        )

        self.assertIn("22 panels", html)
        self.assertIn("10 distinct judges", html)
        self.assertIn("75.0%", html)
        self.assertIn("41 of", html)
        self.assertIn("adaptive follow-up is a separate", html)

    def test_oversight_section_accepts_report_card_repo_prefix(self) -> None:
        from ai_council.oversight_synthesis import render_frontier_synthesis

        summary = json.loads(
            (
                Path(__file__).resolve().parents[1]
                / "data"
                / "oversight_frontier_synthesis_matched_results.json"
            ).read_text()
        )
        html = render_frontier_synthesis(
            summary,
            repo_prefix="../../..",
        )

        self.assertIn(
            "../../../data/oversight_frontier_synthesis_matched_results.json",
            html,
        )

    def test_oversight_section_reports_synthesis_with_uncertainty(self) -> None:
        root = Path(__file__).resolve().parents[1]
        summary = json.loads(
            (root / "data" / "research_question_synthesis.json").read_text()
        )

        html = _oversight_section(None, summary)

        self.assertIn("77 of 118", html)
        self.assertIn("panel-bootstrap intervals", html)
        self.assertIn("suggests judge capability matters", html)
        self.assertIn("oversight-frontier.svg", html)

    def test_probe_effectiveness_section_preserves_key_caveats(self) -> None:
        root = Path(__file__).resolve().parents[1]
        summary = json.loads(
            (root / "data" / "probe_effectiveness_results.json").read_text()
        )

        html = _probe_effectiveness_section(summary)

        self.assertIn(f"{summary['probe_count']} probes", html)
        self.assertIn("not yet a", html)
        self.assertIn("validated recipe", html)
        self.assertIn("One fixed evaluator", html)
        self.assertIn("cannot establish", html)
        self.assertIn("held-out-labels.svg", html)

    def test_probe_author_section_preserves_weighting_and_causal_caveat(
        self,
    ) -> None:
        summary = json.loads(
            (Path(__file__).resolve().parents[1] / "data"
             / "publication_analysis.json").read_text()
        )["probe_author_analysis"]

        html = _probe_author_section(summary)
        text = " ".join(html.split())

        self.assertIn("four higher-capability authors", text)
        self.assertIn("Rates first average within author", text)
        self.assertIn("observationally entangled", text)
        self.assertIn("probe-types-by-capability.svg", html)
        self.assertIn("probe-design-by-capability.svg", html)

    def test_robustness_page_includes_all_primary_checks(self) -> None:
        summary = json.loads(
            (Path(__file__).resolve().parents[1] / "data"
             / "publication_analysis.json").read_text()
        )["robustness"]

        html = robustness_page(summary)

        self.assertIn("Reference sensitivity", html)
        self.assertIn("Replication", html)
        self.assertIn("Answer order", html)
        self.assertIn("What varies?", html)
        self.assertIn("Evidence budget", html)
        self.assertIn("Missing evidence", html)
        self.assertIn("probe-self-solvability.svg", html)

    def test_mechanism_section_keeps_recognition_and_ranking_separate(
        self,
    ) -> None:
        summary = {
            "conditions": [
                {
                    "battery": "ordinary",
                    "author_superior_recognition_rate": 0.6,
                    "anchor_pairwise_accuracy": 0.65,
                    "council_pairwise_accuracy": 0.7,
                },
                {
                    "battery": "verifier",
                    "author_superior_recognition_rate": 0.5,
                    "anchor_pairwise_accuracy": 0.7,
                    "council_pairwise_accuracy": 0.72,
                },
            ],
            "matched_effects": {"mean_interaction": -0.03},
        }

        html = _mechanism_section(summary)
        text = " ".join(html.split())

        self.assertIn("60.0% to", text)
        self.assertIn("50.0%", text)
        self.assertIn("65.0% to", text)
        self.assertIn("70.0%", text)
        self.assertIn("broad ordering", text)
        self.assertIn("externally stronger model", text)

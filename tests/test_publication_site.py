import unittest

from scripts.build_publication_site import (
    _oversight_section,
    kendall_order,
    pairwise_accuracy,
    partial_spearman,
    ranked,
    score_pairwise_accuracy,
    spearman,
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

    def test_oversight_section_supports_pooled_synthesis(self) -> None:
        html = _oversight_section(
            {
                "pooled": {
                    "condition_count": 22,
                    "judge_count": 10,
                    "final_pairs": {"accuracy": 0.8},
                    "superior_recognized": 40,
                    "superior_total": 60,
                }
            }
        )

        self.assertIn("22 panels", html)
        self.assertIn("10 distinct judges", html)
        self.assertIn("40 of", html)

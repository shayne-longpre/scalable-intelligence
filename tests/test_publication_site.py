import unittest

from scripts.build_publication_site import (
    pairwise_accuracy,
    partial_spearman,
    ranked,
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

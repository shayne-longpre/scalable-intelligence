from __future__ import annotations

import unittest

from ai_council.probe_study import (
    adaptive_decision_records,
    adaptive_intents,
    assign_score_bands,
    score_tied_ordering,
    spearman,
    summarize_cohort,
)
from scripts.analyze_probe_study import _parse_run_spec


class ProbeStudyTests(unittest.TestCase):
    def test_spearman_handles_monotonic_values_and_missing_rows(self) -> None:
        self.assertAlmostEqual(spearman([1, 2, 3], [10, 20, 30]), 1.0)
        self.assertAlmostEqual(spearman([1, 2, 3], [30, 20, 10]), -1.0)
        self.assertAlmostEqual(
            spearman([1, 2, 3, 4], [None, 20, 30, 40]), 1.0
        )

    def test_score_bands_partition_ordered_sample(self) -> None:
        rows = [{"judge_score": value} for value in range(10)]
        self.assertEqual(
            assign_score_bands(rows),
            [
                "lower third",
                "lower third",
                "lower third",
                "lower third",
                "middle third",
                "middle third",
                "middle third",
                "upper third",
                "upper third",
                "upper third",
            ],
        )

    def test_cohort_summary_counts_labels_without_losing_multiplicity(self) -> None:
        rows = [
            _record(10, {"math": 2}, {"direct": 1}),
            _record(20, {"math": 1, "code": 1}, {"direct": 2}),
            _record(30, {"code": 2}, {"edge": 1}),
        ]
        summary = summarize_cohort(rows)

        self.assertEqual(summary["question_type_counts"], {"math": 3, "code": 3})
        self.assertEqual(summary["strategy_counts"], {"direct": 3, "edge": 1})
        self.assertEqual(summary["probe_count"], 3)
        self.assertAlmostEqual(
            summary["correlations"]["opening_type_breadth"]["spearman_rho"],
            1.0,
        )

    def test_run_spec_requires_explicit_cohort(self) -> None:
        self.assertEqual(
            _parse_run_spec("frontier=runs/example"),
            {"cohort": "frontier", "run_dir": "runs/example"},
        )
        with self.assertRaises(ValueError):
            _parse_run_spec("runs/example")

    def test_adaptive_intents_code_explicit_plan_language(self) -> None:
        labels = adaptive_intents(
            [
                "Change domain and increase difficulty.",
                "Retest the observed error with adversarial edge cases.",
                "Require proof of correctness for mechanical scoring.",
            ]
        )
        self.assertEqual(
            labels,
            [
                "raise_difficulty",
                "change_domain",
                "retest_weakness",
                "adversarial_validation",
                "mechanical_scoring",
            ],
        )

    def test_adaptive_decisions_preserve_each_round(self) -> None:
        records = adaptive_decision_records(
            [
                {
                    "round_index": 2,
                    "probe_sequence_number": 6,
                    "actual_candidates": ["P1", "P2"],
                    "covered_uncertain_pairs": [["P1", "P2"]],
                    "planned_strategy": ["Change domain."],
                    "follow_up_rationale": ["Retest the weakness."],
                },
                {
                    "round_index": 3,
                    "probe_sequence_number": 7,
                    "actual_candidates": ["P2"],
                    "planned_strategy": ["Increase difficulty."],
                },
            ]
        )

        self.assertEqual([row["round_index"] for row in records], [2, 3])
        self.assertEqual(records[0]["target_count"], 2)
        self.assertEqual(records[0]["uncertain_pairs_covered"], 1)
        self.assertEqual(
            records[0]["intents"], ["change_domain", "retest_weakness"]
        )
        self.assertEqual(records[1]["intents"], ["raise_difficulty"])

    def test_tied_probe_pairs_receive_half_credit(self) -> None:
        result = score_tied_ordering(
            ["A", "B", "C"],
            [["A", "B"]],
            {"A": 3.0, "B": 2.0, "C": 1.0},
            judge_score=0.0,
        )

        self.assertEqual(result["pair_count"], 3)
        self.assertEqual(result["pair_correct"], 2.5)
        self.assertEqual(result["tie_pair_count"], 1)
        self.assertAlmostEqual(result["pair_accuracy"], 5 / 6)
        self.assertAlmostEqual(result["tie_pair_rate"], 1 / 3)
        self.assertEqual(result["decided_pair_count"], 2)
        self.assertEqual(result["decided_pair_correct"], 2)
        self.assertEqual(result["decided_pair_accuracy"], 1.0)

    def test_judge_ties_do_not_restore_external_score_ties(self) -> None:
        result = score_tied_ordering(
            ["A", "B", "C"],
            [["A", "B"]],
            {"A": 2.0, "B": 2.0, "C": 1.0},
            judge_score=0.0,
        )

        self.assertEqual(result["pair_count"], 2)
        self.assertEqual(result["tie_pair_count"], 0)
        self.assertEqual(result["decided_pair_accuracy"], 1.0)


def _record(score: float, question_types: dict[str, int], strategies: dict[str, int]):
    return {
        "run_dir": f"runs/{score}",
        "judge_model": f"provider/{score}",
        "judge_name": str(score),
        "judge_score": score,
        "probes": [{}],
        "opening_type_breadth": score / 10,
        "opening_probe_pair_accuracy": 0.5 + score / 100,
        "opening_decided_pair_accuracy": 0.6 + score / 100,
        "opening_tie_pair_rate": 0.0,
        "informative_probe_rate": 0.5,
        "adaptive_probe_pair_accuracy": 0.5,
        "adaptive_decided_pair_accuracy": 0.5,
        "adaptive_tie_pair_rate": 0.0,
        "adaptive_ranking_delta": 0.0,
        "adaptive_plan_action_count": 1,
        "adaptive_intent_breadth": 1,
        "adaptive_intents": ["raise_difficulty"],
        "question_type_counts": question_types,
        "strategy_counts": strategies,
        "transition_counts": {"opening_probe": 1},
        "validity_counts": {"informative": 1},
    }


if __name__ == "__main__":
    unittest.main()

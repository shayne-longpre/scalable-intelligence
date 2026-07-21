from __future__ import annotations

import unittest

from ai_council.metrics import compute_evolution_metrics, compute_ranking_metrics


class MetricsTests(unittest.TestCase):
    def test_ranking_metrics_measure_final_agreement_and_churn(self) -> None:
        metrics = compute_ranking_metrics(
            {
                "memory_update_1": [
                    {"speaker": "P1", "ranking": ["P1", "P2", "P3"]},
                    {"speaker": "P2", "ranking": ["P1", "P2", "P3"]},
                ],
                "final_judgment": [
                    {"speaker": "P1", "ranking": ["P2", "P1", "P3"]},
                    {"speaker": "P2", "ranking": ["P2", "P1", "P3"]},
                ],
            }
        )

        agreement = metrics["final_agreement"]
        self.assertEqual(agreement["pair_count"], 1)
        self.assertEqual(agreement["mean_pairwise_tau"], 1.0)
        self.assertEqual(agreement["same_top1_pairs"], 1)
        self.assertEqual(agreement["exact_match_pairs"], 1)

        p1_churn = metrics["churn_by_speaker"]["P1"]
        self.assertEqual(p1_churn["transition_count"], 1)
        self.assertEqual(p1_churn["top1_changes"], 1)
        self.assertAlmostEqual(p1_churn["mean_adjacent_tau"], 1 / 3)

    def test_evolution_metrics_label_followups_and_switches(self) -> None:
        extraction = {
            "probe_events": [
                {
                    "turn_id": 1,
                    "speaker": "P1",
                    "round_index": 1,
                    "generation_stage": "baseline_probe",
                    "evidence_turn_ids_available": [],
                    "content": "Solve this algebra pattern and explain the rule.",
                    "question_type_tags": [{"tag": "quantitative_math_reasoning"}],
                    "strategy_tags": [{"tag": "direct_task_probe"}],
                },
                {
                    "turn_id": 2,
                    "speaker": "P1",
                    "round_index": 2,
                    "generation_stage": "iterative_round_robin",
                    "evidence_turn_ids_available": [1],
                    "content": "Follow-up based on that algebra answer: explain the edge case.",
                    "question_type_tags": [{"tag": "quantitative_math_reasoning"}],
                    "strategy_tags": [{"tag": "adaptive_followup"}],
                },
                {
                    "turn_id": 3,
                    "speaker": "P2",
                    "round_index": 1,
                    "generation_stage": "baseline_probe",
                    "evidence_turn_ids_available": [],
                    "content": "Define intelligence with a concise analogy.",
                    "question_type_tags": [{"tag": "verbal_abstraction_similarity"}],
                    "strategy_tags": [{"tag": "criterion_negotiation"}],
                },
                {
                    "turn_id": 4,
                    "speaker": "P2",
                    "round_index": 2,
                    "generation_stage": "iterative_round_robin",
                    "evidence_turn_ids_available": [3],
                    "content": "Now write pseudocode for a scheduler.",
                    "question_type_tags": [{"tag": "coding_algorithmic_reasoning"}],
                    "strategy_tags": [{"tag": "direct_task_probe"}],
                },
            ]
        }

        metrics = compute_evolution_metrics(extraction)
        self.assertEqual(metrics["probe_event_count"], 4)
        self.assertEqual(metrics["transition_counts"]["opening_probe"], 2)
        self.assertEqual(metrics["transition_counts"]["adaptive_deepening"], 1)
        self.assertEqual(metrics["transition_counts"]["adaptive_broadening"], 1)
        self.assertEqual(metrics["dependency_counts"]["evidence_conditioned"], 2)
        self.assertEqual(
            metrics["question_types_by_round"]["round_2"]["coding_algorithmic_reasoning"],
            1,
        )

    def test_evolution_does_not_call_preplanned_battery_adaptive(self) -> None:
        metrics = compute_evolution_metrics(
            {
                "probe_events": [
                    {
                        "turn_id": 1,
                        "speaker": "J1",
                        "round_index": 1,
                        "generation_stage": "baseline_battery",
                        "evidence_turn_ids_available": [],
                        "content": "Solve this algebra problem.",
                        "question_type_tags": [{"tag": "quantitative_math_reasoning"}],
                        "strategy_tags": [{"tag": "direct_task_probe"}],
                    },
                    {
                        "turn_id": 2,
                        "speaker": "J1",
                        "round_index": 1,
                        "generation_stage": "baseline_battery",
                        "evidence_turn_ids_available": [],
                        "content": "Solve a second algebra problem with an edge case.",
                        "question_type_tags": [{"tag": "quantitative_math_reasoning"}],
                        "strategy_tags": [{"tag": "adaptive_followup"}],
                    },
                ]
            }
        )

        self.assertEqual(metrics["transition_counts"]["preplanned_same_area"], 1)
        self.assertEqual(metrics["dependency_counts"], {"preplanned_without_answers": 2})
        self.assertNotIn("adaptive_deepening", metrics["transition_counts"])

    def test_incidental_format_tags_do_not_make_distinct_topics_look_repeated(self) -> None:
        shared_modifier = {"tag": "instruction_following_format_control"}
        metrics = compute_evolution_metrics(
            {
                "probe_events": [
                    {
                        "turn_id": 1,
                        "speaker": "J1",
                        "generation_stage": "baseline_battery",
                        "question_type_tags": [
                            {"tag": "coding_algorithmic_reasoning"},
                            {"tag": "working_memory_state_tracking"},
                            shared_modifier,
                        ],
                        "strategy_tags": [{"tag": "direct_task_probe"}],
                    },
                    {
                        "turn_id": 2,
                        "speaker": "J1",
                        "generation_stage": "baseline_battery",
                        "question_type_tags": [
                            {"tag": "stem_scientific_reasoning"},
                            {"tag": "working_memory_state_tracking"},
                            shared_modifier,
                        ],
                        "strategy_tags": [{"tag": "direct_task_probe"}],
                    },
                ]
            }
        )

        self.assertEqual(metrics["transition_counts"]["preplanned_broadening"], 1)
        self.assertEqual(
            metrics["probe_events"][1]["primary_question_type"],
            "stem_scientific_reasoning",
        )

    def test_evolution_flags_adaptive_stage_without_evidence_provenance(self) -> None:
        metrics = compute_evolution_metrics(
            {
                "probe_events": [
                    {
                        "turn_id": 1,
                        "speaker": "J1",
                        "generation_stage": "baseline_battery",
                        "evidence_turn_ids_available": [],
                        "question_type_tags": [],
                        "strategy_tags": [],
                    },
                    {
                        "turn_id": 2,
                        "speaker": "J1",
                        "generation_stage": "adaptive_followup",
                        "evidence_turn_ids_available": [],
                        "question_type_tags": [],
                        "strategy_tags": [{"tag": "adaptive_followup"}],
                    },
                ]
            }
        )

        self.assertEqual(metrics["transition_counts"]["adaptive_provenance_missing"], 1)
        self.assertEqual(metrics["dependency_counts"]["evidence_missing"], 1)

    def test_adaptive_metrics_distinguish_targeting_from_probe_yield(self) -> None:
        metrics = compute_evolution_metrics(
            {
                "probe_events": [
                    {
                        "turn_id": 11,
                        "speaker": "J1",
                        "round_index": 2,
                        "generation_stage": "adaptive_followup",
                        "evidence_turn_ids_available": [10],
                        "question_type_tags": [
                            {"tag": "working_memory_state_tracking"}
                        ],
                        "strategy_tags": [{"tag": "adaptive_followup"}],
                    }
                ],
                "adaptive_decisions": [
                    {
                        "question_turn_id": 11,
                        "actual_candidates": ["P1", "P2"],
                        "requested_candidates": ["P1", "P2"],
                        "selection_matches_request": True,
                        "retained_candidates": ["P1", "P2"],
                        "added_candidates": [],
                        "dropped_candidates": ["P3"],
                        "prior_uncertain_pairs": [["P1", "P2"]],
                        "covered_uncertain_pairs": [["P1", "P2"]],
                        "probe_validity": "limited",
                        "ranking_changed": False,
                        "confidence_delta": -0.05,
                    }
                ],
            }
        )

        adaptive = metrics["adaptive"]
        self.assertEqual(adaptive["selection_match_count"], 1)
        self.assertEqual(adaptive["uncertainty_coverage_count"], 1)
        self.assertEqual(adaptive["target_change_counts"], {"narrowed": 1})
        self.assertEqual(adaptive["evidence_outcome_counts"], {"limited": 1})
        self.assertEqual(adaptive["ranking_change_count"], 0)
        self.assertAlmostEqual(adaptive["mean_confidence_delta"], -0.05)
        self.assertEqual(
            adaptive["decision_trace"][0]["question_types"],
            ["working_memory_state_tracking"],
        )

    def test_primary_probe_budget_ranking_is_the_final_snapshot(self) -> None:
        metrics = compute_ranking_metrics(
            {
                "judge_ranking": [
                    {
                        "speaker": "J1",
                        "ranking": ["P2", "P1"],
                        "scores": {"P1": 40, "P2": 60},
                        "judgment_probe_count": 2,
                        "judgment_probe_total": 6,
                        "is_primary_judgment": False,
                    },
                    {
                        "speaker": "J1",
                        "ranking": ["P1", "P2"],
                        "scores": {"P1": 70, "P2": 30},
                        "judgment_probe_count": 6,
                        "judgment_probe_total": 6,
                        "is_primary_judgment": True,
                    },
                ]
            }
        )

        self.assertEqual(metrics["final_rankings"][0]["ranking"], ["P1", "P2"])
        self.assertEqual(metrics["final_rankings"][0]["scores"], {"P1": 70, "P2": 30})


if __name__ == "__main__":
    unittest.main()

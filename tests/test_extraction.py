from __future__ import annotations

import unittest

from ai_council.extraction import extract_posthoc_interactions


class ExtractionTests(unittest.TestCase):
    def test_free_discussion_probe_events_feed_summary_taxonomy_counts(self) -> None:
        extraction = extract_posthoc_interactions(
            [
                {
                    "turn_id": 1,
                    "phase": "discussion_round_1",
                    "round_index": 1,
                    "speaker": "P1",
                    "visibility": "public",
                    "content": (
                        "Question: design a falsifiable experiment with predicted outcomes "
                        "and confounding factors."
                    ),
                    "parsed": None,
                    "metadata": {"interaction_mode": "interactive_discussion", "interaction_role": "discussion"},
                }
            ]
        )

        self.assertEqual(extraction["summary"]["probe_event_count"], 1)
        self.assertIn("stem_scientific_reasoning", extraction["summary"]["question_type_frequency"])
        self.assertIn("direct_task_probe", extraction["summary"]["strategy_frequency"])

    def test_qa_strategy_labels_come_from_question_not_candidate_answer(self) -> None:
        extraction = extract_posthoc_interactions(
            [
                {
                    "turn_id": 1,
                    "phase": "probe",
                    "round_index": 1,
                    "speaker": "J1",
                    "visibility": "private",
                    "content": "Solve this geometry problem and explain your reasoning.",
                    "parsed": None,
                    "metadata": {
                        "interaction_mode": "independent_judge_ranking",
                        "interaction_role": "question",
                        "probe_id": "q1",
                    },
                },
                {
                    "turn_id": 2,
                    "phase": "probe",
                    "round_index": 1,
                    "speaker": "P1",
                    "visibility": "private",
                    "content": "I could bluff or manipulate the ranking, but here is the proof.",
                    "parsed": None,
                    "metadata": {
                        "interaction_mode": "independent_judge_ranking",
                        "interaction_role": "answer",
                        "interviewer": "J1",
                        "respondent": "P1",
                        "question_turn_id": 1,
                        "probe_id": "q1",
                    },
                },
            ]
        )

        pair = extraction["qa_pairs"][0]
        strategy_ids = {tag["tag"] for tag in pair["strategy_tags"]}
        self.assertNotIn("deception_or_manipulation", strategy_ids)
        self.assertNotIn("evasion_or_performativity", strategy_ids)

    def test_preplanned_probe_cannot_be_labeled_as_adaptive_from_keywords(self) -> None:
        extraction = extract_posthoc_interactions(
            [
                {
                    "turn_id": 1,
                    "phase": "judge_ranking",
                    "round_index": 1,
                    "speaker": "J1",
                    "visibility": "private",
                    "content": "Quantify uncertainty, then refine the estimate.",
                    "parsed": None,
                    "metadata": {
                        "interaction_mode": "independent_judge_ranking",
                        "interaction_role": "question",
                        "generation_stage": "baseline_battery",
                        "probe_id": "q1",
                    },
                }
            ]
        )

        strategy_ids = {
            tag["tag"] for tag in extraction["probe_events"][0]["strategy_tags"]
        }
        self.assertNotIn("adaptive_followup", strategy_ids)

    def test_adaptive_decision_links_targets_evidence_and_outcome(self) -> None:
        extraction = extract_posthoc_interactions(
            [
                {
                    "turn_id": 10,
                    "phase": "judge_ranking",
                    "round_index": 1,
                    "speaker": "J1",
                    "visibility": "private",
                    "content": "{}",
                    "parsed": {
                        "ranking": ["P1", "P2", "P3"],
                        "confidence": 0.6,
                        "uncertain_pairs": [["P1", "P2"]],
                        "follow_up_candidates": ["P1", "P2"],
                        "follow_up_rationale": ["Separate the close pair"],
                        "next_probe_strategy": ["Change domain"],
                    },
                    "metadata": {
                        "interaction_role": "wave_judgment",
                        "probe_comparison_turn_ids": [9],
                    },
                },
                {
                    "turn_id": 11,
                    "phase": "judge_ranking",
                    "round_index": 2,
                    "speaker": "J1",
                    "visibility": "private",
                    "content": "Solve this state update.",
                    "parsed": None,
                    "metadata": {
                        "interaction_mode": "independent_judge_ranking",
                        "interaction_role": "question",
                        "generation_stage": "adaptive_followup",
                        "probe_id": "q2",
                        "probe_sequence_number": 5,
                        "respondents": ["P1", "P2"],
                        "prior_ranking_turn_id": 10,
                    },
                },
                {
                    "turn_id": 12,
                    "phase": "judge_ranking",
                    "round_index": 2,
                    "speaker": "J1",
                    "visibility": "private",
                    "content": "{}",
                    "parsed": {
                        "probe_validity": "informative",
                        "confidence": 0.8,
                    },
                    "metadata": {
                        "interaction_role": "probe_comparison",
                        "probe_id": "q2",
                    },
                },
                {
                    "turn_id": 13,
                    "phase": "judge_ranking",
                    "round_index": 2,
                    "speaker": "J1",
                    "visibility": "private",
                    "content": "{}",
                    "parsed": {
                        "ranking": ["P2", "P1", "P3"],
                        "confidence": 0.75,
                    },
                    "metadata": {"interaction_role": "wave_judgment"},
                },
            ]
        )

        decision = extraction["adaptive_decisions"][0]
        self.assertEqual(decision["requested_candidates"], ["P1", "P2"])
        self.assertTrue(decision["selection_matches_request"])
        self.assertEqual(decision["covered_uncertain_pairs"], [["P1", "P2"]])
        self.assertEqual(decision["probe_validity"], "informative")
        self.assertTrue(decision["ranking_changed"])
        self.assertAlmostEqual(decision["confidence_delta"], 0.15)


if __name__ == "__main__":
    unittest.main()

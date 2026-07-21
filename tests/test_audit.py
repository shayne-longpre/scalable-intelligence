from __future__ import annotations

import unittest

from ai_council.audit import audit_experiment_behavior
from ai_council.extraction import extract_posthoc_interactions


class BehaviorAuditTests(unittest.TestCase):
    def test_flags_answer_that_appears_to_answer_another_same_round_probe(self) -> None:
        entries = [
            _entry(
                1,
                "P1",
                "Explain how emotions function as currency in a fictional city.",
                role="question",
                interviewer="P1",
                respondents=["P2"],
                probe_id="r1:p1",
            ),
            _entry(
                2,
                "P1",
                "Write a palindrome algorithm with two pointers and return false on mismatch.",
                role="question",
                interviewer="P1",
                respondents=["P2"],
                probe_id="r1:p2",
            ),
            _entry(
                3,
                "P2",
                "A palindrome algorithm uses left and right pointers, compares characters, "
                "normalizes casing, skips punctuation, handles unicode cautiously, tracks "
                "loop invariants, returns false on mismatch, returns true after crossing, "
                "and explains complexity, edge cases, empty strings, and testing strategy.",
                role="answer",
                interviewer="P1",
                respondent="P2",
                question_turn_id=1,
                probe_id="r1:p1",
            ),
        ]
        extraction = extract_posthoc_interactions(entries)
        audit = audit_experiment_behavior(entries, extraction)
        self.assertEqual(audit["summary"]["codes"]["possible_wrong_question_answered"], 1)

    def test_does_not_compare_private_questions_from_independent_judges(self) -> None:
        entries = [
            _entry(
                1,
                "J1",
                "Analyze an integer array rewrite system and prove termination.",
                role="question",
                interviewer="J1",
                respondents=["P1"],
                probe_id="j1:r2",
            ),
            _entry(
                2,
                "J2",
                "Characterize all functions satisfying a nested functional equation.",
                role="question",
                interviewer="J2",
                respondents=["P1"],
                probe_id="j2:r2",
            ),
            _entry(
                3,
                "P1",
                "The function is additive and involutive. Injectivity and surjectivity follow, "
                "then a rational vector-space decomposition gives every solution. This also "
                "covers discontinuous examples built from a Hamel basis and proves the converse.",
                role="answer",
                interviewer="J2",
                respondent="P1",
                question_turn_id=2,
                probe_id="j2:r2",
            ),
        ]
        extraction = extract_posthoc_interactions(entries)
        audit = audit_experiment_behavior(entries, extraction)

        self.assertNotIn("possible_wrong_question_answered", audit["summary"]["codes"])

    def test_does_not_flag_weak_shared_vocabulary_across_battery_probes(self) -> None:
        entries = [
            _entry(
                1,
                "J1",
                "Compute an observational probability and an intervention probability.",
                role="question",
                interviewer="J1",
                respondents=["P1"],
                probe_id="j1:p1",
            ),
            _entry(
                2,
                "J1",
                "Choose an action by calculating posterior probability and expected utility.",
                role="question",
                interviewer="J1",
                respondents=["P1"],
                probe_id="j1:p2",
            ),
            _entry(
                3,
                "P1",
                "The observational probability differs from the intervention because conditioning "
                "updates the common cause while intervention cuts the incoming edge. Bayes gives "
                "the first value, and marginalizing the unchanged cause gives the second value.",
                role="answer",
                interviewer="J1",
                respondent="P1",
                question_turn_id=1,
                probe_id="j1:p1",
            ),
        ]
        extraction = extract_posthoc_interactions(entries)
        audit = audit_experiment_behavior(entries, extraction)

        self.assertNotIn("possible_wrong_question_answered", audit["summary"]["codes"])

    def test_records_structured_json_repair_metadata(self) -> None:
        entries = [
            {
                "turn_id": 1,
                "phase": "final",
                "round_index": None,
                "speaker": "P1",
                "visibility": "private",
                "content": '{"participant_id":"P1"}',
                "parsed": {"participant_id": "P1"},
                "metadata": {
                    "structured_json_repair": {
                        "attempted": True,
                        "repaired": True,
                        "original_parse_error": "no JSON object found",
                    }
                },
            }
        ]
        audit = audit_experiment_behavior(entries, {"qa_pairs": []})
        self.assertEqual(audit["summary"]["codes"]["structured_json_repaired"], 1)

    def test_flags_expected_observed_qa_mismatch_and_missing_final_judgment(self) -> None:
        entries = [
            _entry(
                1,
                "P1",
                "Ask a diagnostic question.",
                role="question",
                interviewer="P1",
                respondents=["P2", "P3"],
                probe_id="r1:p1",
            ),
            {
                "turn_id": 2,
                "phase": "final_judgment",
                "round_index": None,
                "speaker": "P1",
                "visibility": "private",
                "content": "{}",
                "parsed": {"ranking": ["P1", "P2", "P3"], "evidence": ["Observed reasoning."]},
                "metadata": {},
            },
        ]
        config = {
            "participants": [
                {"id": "P1"},
                {"id": "P2"},
                {"id": "P3"},
            ]
        }
        audit = audit_experiment_behavior(
            entries,
            {"qa_pairs": [], "summary": {"probe_event_count": 1, "question_type_frequency": {}}},
            config,
        )
        self.assertEqual(audit["summary"]["codes"]["expected_observed_qa_mismatch"], 1)
        self.assertEqual(audit["summary"]["codes"]["missing_final_judgment"], 2)
        self.assertEqual(audit["summary"]["codes"]["no_question_type_labels"], 1)

    def test_flags_memory_update_with_unrouted_or_missing_respondents(self) -> None:
        entries = [
            _entry(
                1,
                "P1",
                "Ask a diagnostic question.",
                role="question",
                interviewer="P1",
                respondents=["P2", "P3"],
                probe_id="r1:p1",
            ),
            {
                "turn_id": 2,
                "phase": "probe_rounds",
                "round_index": 1,
                "speaker": "P1",
                "visibility": "private",
                "content": "{}",
                "parsed": {
                    "qa_assessment_summaries": [
                        {"respondent_id": "P1", "assessment_summary": "Self was not routed."},
                        {"respondent_id": "P2", "assessment_summary": "P2 answered."},
                    ]
                },
                "metadata": {
                    "interaction_mode": "round_robin_probes",
                    "interaction_role": "memory_update",
                    "interviewer": "P1",
                    "probe_id": "r1:p1",
                },
            },
        ]
        audit = audit_experiment_behavior(
            entries,
            {"qa_pairs": [], "summary": {"probe_event_count": 1, "question_type_frequency": {"logic": 1}}},
        )
        self.assertEqual(audit["summary"]["codes"]["memory_update_unexpected_respondent"], 1)
        self.assertEqual(audit["summary"]["codes"]["memory_update_missing_respondent"], 1)

    def test_flags_truncated_model_completion(self) -> None:
        entries = [
            {
                "turn_id": 1,
                "phase": "probe_rounds",
                "round_index": 1,
                "speaker": "P1",
                "visibility": "private",
                "content": "partial answer",
                "parsed": None,
                "metadata": {
                    "finish_reason": "length",
                    "interaction_role": "answer",
                    "request_params": {"max_tokens": 12},
                },
            }
        ]
        audit = audit_experiment_behavior(entries, {"qa_pairs": []})
        self.assertEqual(audit["summary"]["codes"]["completion_truncated"], 1)

    def test_wave_judgment_comparative_evidence_satisfies_final_evidence_check(self) -> None:
        entries = [
            {
                "turn_id": 1,
                "phase": "judge_ranking",
                "round_index": 1,
                "speaker": "J1",
                "visibility": "private",
                "content": "{}",
                "parsed": {
                    "ranking": ["P1", "P2"],
                    "comparative_evidence": ["P1 corrected a premise that P2 accepted."],
                },
                "metadata": {
                    "interaction_mode": "independent_judge_ranking",
                    "interaction_role": "wave_judgment",
                },
            }
        ]
        config = {
            "participants": [{"id": "P1"}, {"id": "P2"}],
            "judges": [{"id": "J1"}],
            "protocol": {"phases": [{"kind": "independent_judge_ranking"}]},
        }

        audit = audit_experiment_behavior(entries, {"qa_pairs": []}, config)

        self.assertNotIn("final_judgment_missing_evidence", audit["summary"]["codes"])

    def test_thin_answer_uses_word_count_not_unique_content_vocabulary(self) -> None:
        long_numeric_answer = " ".join(
            f"step {index}: value {index * 2}; therefore continue"
            for index in range(1, 9)
        )
        entries = [
            _entry(
                1,
                "J1",
                "Calculate each state and show the intermediate values.",
                role="question",
                interviewer="J1",
                respondents=["P1"],
                probe_id="r1:j1",
            ),
            _entry(
                2,
                "P1",
                long_numeric_answer,
                role="answer",
                interviewer="J1",
                respondent="P1",
                question_turn_id=1,
                probe_id="r1:j1",
            ),
        ]

        extraction = extract_posthoc_interactions(entries)
        audit = audit_experiment_behavior(entries, extraction)

        self.assertNotIn("thin_answer", audit["summary"]["codes"])


def _entry(
    turn_id: int,
    speaker: str,
    content: str,
    *,
    role: str,
    interviewer: str,
    respondent: str | None = None,
    respondents: list[str] | None = None,
    question_turn_id: int | None = None,
    probe_id: str,
) -> dict:
    metadata = {
        "interaction_mode": "round_robin_probes",
        "interaction_role": role,
        "stream_id": probe_id,
        "probe_id": probe_id,
        "interviewer": interviewer,
    }
    if respondent is not None:
        metadata["respondent"] = respondent
    if respondents is not None:
        metadata["respondents"] = respondents
    if question_turn_id is not None:
        metadata["question_turn_id"] = question_turn_id
    return {
        "turn_id": turn_id,
        "phase": "probe_rounds",
        "round_index": 1,
        "speaker": speaker,
        "visibility": "private",
        "content": content,
        "parsed": None,
        "metadata": metadata,
    }


if __name__ == "__main__":
    unittest.main()

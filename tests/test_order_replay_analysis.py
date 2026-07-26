from __future__ import annotations

import unittest

from ai_council.order_replay_analysis import (
    candidate_call_count,
    compare_rankings,
    exact_evidence_match,
)


class OrderReplayAnalysisTests(unittest.TestCase):
    def test_compare_rankings_reports_stability_and_displacement(self) -> None:
        comparison = compare_rankings(
            ["P1", "P2", "P3", "P4"],
            ["P1", "P3", "P2", "P4"],
        )

        self.assertAlmostEqual(comparison["kendall_tau"], 2 / 3)
        self.assertTrue(comparison["top_rank_stable"])
        self.assertEqual(comparison["top_three_overlap"], 3)
        self.assertEqual(comparison["mean_absolute_displacement"], 0.5)

    def test_exact_evidence_match_uses_stream_ids_and_content(self) -> None:
        source = [
            _event("probe", "probe:1", "Question"),
            _event("answer", "probe:1:P1", "Answer one"),
            _event("answer", "probe:1:P2", "Answer two"),
        ]
        replay = [source[2], source[0], source[1]]

        self.assertTrue(exact_evidence_match(source, replay)["exact"])

        replay[0] = _event("answer", "probe:1:P2", "Changed")
        comparison = exact_evidence_match(source, replay)
        self.assertFalse(comparison["exact"])
        self.assertEqual(comparison["mismatched_stream_ids"], ["probe:1:P2"])

    def test_exact_evidence_match_detects_missing_stream(self) -> None:
        source = [
            _event("probe", "probe:1", "Question"),
            _event("answer", "probe:1:P1", "Answer"),
        ]
        comparison = exact_evidence_match(source, source[:1])

        self.assertFalse(comparison["exact"])
        self.assertEqual(comparison["missing_stream_ids"], ["probe:1:P1"])

    def test_candidate_call_count_uses_persisted_model_spend(self) -> None:
        config = {
            "participants": [
                {"id": "P1", "model": "candidate_p1"},
                {"id": "P2", "model": "candidate_p2"},
            ]
        }
        run_summary = {
            "model_spend": {
                "judge": {"model_calls": 8},
                "candidate_p1": {"model_calls": 2},
            }
        }

        self.assertEqual(candidate_call_count(run_summary, config), 2)


def _event(role: str, stream_id: str, content: str) -> dict:
    return {
        "content": content,
        "metadata": {
            "interaction_role": role,
            "stream_id": stream_id,
        },
    }


if __name__ == "__main__":
    unittest.main()

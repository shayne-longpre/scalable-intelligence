from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ai_council.config import ExperimentConfig
from ai_council.core import TranscriptEntry
from ai_council.report_card import (
    _final_prior_agreement,
    _highlight_candidates,
    _judge_condition_summary,
    _probe_budget_results,
    build_report_card,
)
from ai_council.storage import RunStore


class ReportCardTests(unittest.TestCase):
    def test_judge_condition_summary_compares_final_rankings_across_runs(self) -> None:
        cards = [
            {
                "name": "run-a",
                "participants": [
                    {"id": "P1", "provider_model_id": "candidate/1"},
                    {"id": "P2", "provider_model_id": "candidate/2"},
                    {"id": "P3", "provider_model_id": "candidate/3"},
                ],
                "judges": [
                    {"id": "J1", "model_ref": "judge", "provider_model_id": "model/a"}
                ],
                "final_rankings": [{"speaker": "J1", "ranking": ["P1", "P2", "P3"]}],
            },
            {
                "name": "run-b",
                "participants": [
                    {"id": "P1", "provider_model_id": "candidate/1"},
                    {"id": "P2", "provider_model_id": "candidate/2"},
                    {"id": "P3", "provider_model_id": "candidate/3"},
                ],
                "judges": [
                    {"id": "J1", "model_ref": "judge", "provider_model_id": "model/b"}
                ],
                "final_rankings": [{"speaker": "J1", "ranking": ["P1", "P3", "P2"]}],
            },
        ]

        summary = _judge_condition_summary(cards)

        self.assertAlmostEqual(summary["mean_final_interjudge_tau"], 1 / 3)
        self.assertEqual(len(summary["final_interjudge_pairs"]), 1)
        self.assertTrue(summary["final_interjudge_pairs"][0]["same_top1"])

    def test_judge_condition_summary_does_not_compare_different_rosters(self) -> None:
        cards = [
            {
                "name": "run-a",
                "participants": [{"id": "P1", "provider_model_id": "candidate/a"}],
                "judges": [{"id": "J1", "provider_model_id": "judge/a"}],
                "final_rankings": [{"speaker": "J1", "ranking": ["P1"]}],
            },
            {
                "name": "run-b",
                "participants": [{"id": "P1", "provider_model_id": "candidate/b"}],
                "judges": [{"id": "J1", "provider_model_id": "judge/b"}],
                "final_rankings": [{"speaker": "J1", "ranking": ["P1"]}],
            },
        ]

        summary = _judge_condition_summary(cards)

        self.assertEqual(summary["final_interjudge_pairs"], [])
        self.assertIsNone(summary["mean_final_interjudge_tau"])

    def test_reported_score_subset_is_primary_but_all_candidate_metrics_remain(self) -> None:
        judgment = {
            "phase": "judge_ranking",
            "speaker": "J1",
            "round_index": 1,
            "judgment_probe_count": 4,
            "is_primary_judgment": True,
            "ranking": ["P1", "P2", "P3"],
            "kendall_tau": 0.2,
            "spearman_rho": 0.3,
            "pairwise_accuracy": 0.6,
            "rank_score_r_squared": 0.4,
            "top1_matches_prior": False,
            "reported_score_subset": {
                "candidate_count": 2,
                "kendall_tau": 1.0,
                "spearman_rho": 1.0,
                "pairwise_accuracy": 1.0,
                "rank_score_r_squared": 0.9,
                "top1_matches_prior": True,
            },
        }
        prior = {"judgments": [judgment]}

        final = _final_prior_agreement(prior)
        budget = _probe_budget_results(prior)[0]

        self.assertEqual(final["mean_tau"], 1.0)
        self.assertEqual(final["all_candidate_mean_tau"], 0.2)
        self.assertEqual(final["basis"], "reported-score subset")
        self.assertEqual(budget["pairwise_accuracy"], 1.0)
        self.assertEqual(budget["all_candidate_pairwise_accuracy"], 0.6)

    def test_judge_condition_summary_aggregates_per_model_not_anonymous_id(self) -> None:
        cards = []
        for run_index, pairwise in enumerate((1.0, 0.5), start=1):
            cards.append(
                {
                    "judges": [
                        {
                            "id": f"J{run_index}",
                            "model_ref": "judge_model",
                            "provider_model_id": "provider/judge",
                        }
                    ],
                    "final_agreement": {"mean_pairwise_tau": 0.75},
                    "probe_budget_results": [
                        {
                            "speaker": f"J{run_index}",
                            "probe_count": 4,
                            "pairwise_accuracy": pairwise,
                            "kendall_tau": pairwise,
                            "confidence": 0.8,
                            "top1_matches_prior": pairwise == 1.0,
                        }
                    ],
                    "probe_comparisons": [
                        {
                            "speaker": f"J{run_index}",
                            "parsed": {"probe_validity": "informative"},
                        }
                    ],
                    "adaptive_metrics": {
                        "decision_trace": [
                            {
                                "judge_id": f"J{run_index}",
                                "actual_candidates": ["P1", "P2"],
                                "ranking_changed": run_index == 2,
                            }
                        ]
                    },
                    "model_spend": {
                        "judge_model": {
                            "model_calls": 3,
                            "reported_cost_usd": 0.25,
                        }
                    },
                }
            )

        summary = _judge_condition_summary(cards)
        accuracy = summary["accuracy_by_probe_count"][0]
        behavior = summary["judge_behavior"][0]

        self.assertEqual(accuracy["n"], 2)
        self.assertEqual(accuracy["mean_pairwise_accuracy"], 0.75)
        self.assertEqual(accuracy["top1_rate"], 0.5)
        self.assertEqual(behavior["probe_validity"]["informative"], 2)
        self.assertEqual(behavior["adaptive_decisions"], 2)
        self.assertEqual(behavior["mean_target_size"], 2)
        self.assertEqual(behavior["rank_changes"], 1)
        self.assertEqual(behavior["direct_model_calls"], 6)
        self.assertEqual(behavior["direct_reported_cost_usd"], 0.5)

    def test_highlights_are_balanced_across_independent_judges(self) -> None:
        qa_pairs = []
        for turn_id in range(1, 5):
            qa_pairs.append(
                {
                    "question_turn_id": turn_id,
                    "interviewer_id": "J1",
                    "respondent_id": "P1",
                    "question_text": f"J1 probe {turn_id}",
                    "question_type_tags": [],
                }
            )
        for turn_id in range(5, 9):
            qa_pairs.append(
                {
                    "question_turn_id": turn_id,
                    "interviewer_id": "J2",
                    "respondent_id": "P1",
                    "question_text": f"J2 probe {turn_id}",
                    "question_type_tags": [],
                }
            )

        highlights = _highlight_candidates({"qa_pairs": qa_pairs}, {}, limit=4)

        self.assertEqual([item["speaker"] for item in highlights], ["J1", "J2", "J1", "J2"])

    def test_report_card_compares_runs_and_writes_artifacts(self) -> None:
        config = ExperimentConfig.from_dict(
            {
                "name": "report_card_fixture",
                "providers": [{"name": "mock", "kind": "mock"}],
                "models": [
                    {"name": "strong", "provider": "mock", "model": "provider/strong"},
                    {"name": "weak", "provider": "mock", "model": "provider/weak"},
                ],
                "participants": [
                    {"id": "P1", "model": "strong"},
                    {"id": "P2", "model": "weak"},
                ],
                "protocol": {
                    "name": "fixture_protocol",
                    "phases": [
                        {
                            "name": "final_judgment",
                            "kind": "private_judgment",
                            "prompt": "final_judgment",
                            "visibility": "private",
                            "require_json": True,
                        }
                    ],
                },
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            prior_path = Path(tmpdir) / "priors.json"
            prior_path.write_text(
                '{"name":"test","version":"1","models":['
                '{"provider_model_id":"provider/strong","estimated_rank":1},'
                '{"provider_model_id":"provider/weak","estimated_rank":2}'
                "]}",
                encoding="utf-8",
            )
            first = _run_with_rankings(tmpdir, config, [["P1", "P2"], ["P1", "P2"]])
            second = _run_with_rankings(tmpdir, config, [["P2", "P1"], ["P1", "P2"]])
            first.write_json(
                "run_summary.json",
                {
                    "model_calls": 2,
                    "reported_cost_usd": 0.3,
                    "model_spend": {
                        "strong": {
                            "provider": "mock",
                            "provider_model_id": "provider/strong",
                            "model_calls": 1,
                            "reported_cost_usd": 0.2,
                        },
                        "weak": {
                            "provider": "mock",
                            "provider_model_id": "provider/weak",
                            "model_calls": 1,
                            "reported_cost_usd": 0.1,
                        },
                    },
                },
            )
            output_dir = Path(tmpdir) / "card"

            report = build_report_card(
                [first.run_dir, second.run_dir],
                prior_ranking_file=prior_path,
                output_dir=output_dir,
            )

            self.assertEqual(len(report["runs"]), 2)
            self.assertTrue((output_dir / "report_card.md").exists())
            self.assertTrue((output_dir / "report_card.html").exists())
            self.assertTrue((output_dir / "report_card_summary.json").exists())
            html = (output_dir / "report_card.html").read_text(encoding="utf-8")
            self.assertIn("Model Priors", html)
            self.assertIn("At A Glance", html)
            self.assertIn("Reported Spend", html)
            self.assertIn("$0.200000", html)
            self.assertEqual(report["runs"][0]["model_spend"]["strong"]["model_calls"], 1)
            self.assertEqual(report["runs"][0]["final_agreement"]["mean_pairwise_tau"], 1.0)
            self.assertEqual(report["runs"][0]["final_prior_agreement"]["top1_matches"], 2)

    def test_report_card_can_include_opt_in_llm_summary(self) -> None:
        config = ExperimentConfig.from_dict(
            {
                "name": "report_card_llm_fixture",
                "providers": [{"name": "mock", "kind": "mock"}],
                "models": [{"name": "model", "provider": "mock", "model": "mock:model"}],
                "participants": [{"id": "P1", "model": "model"}, {"id": "P2", "model": "model"}],
                "protocol": {
                    "name": "fixture_protocol",
                    "phases": [
                        {
                            "name": "final_judgment",
                            "kind": "private_judgment",
                            "prompt": "final_judgment",
                            "visibility": "private",
                            "require_json": True,
                        }
                    ],
                },
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_config = Path(tmpdir) / "summary.json"
            summary_config.write_text(
                '{"provider":{"name":"mock","kind":"mock"},'
                '"model":"mock:summary","params":{"max_tokens":120}}',
                encoding="utf-8",
            )
            run = _run_with_rankings(tmpdir, config, [["P1", "P2"], ["P2", "P1"]])
            output_dir = Path(tmpdir) / "card"

            report = build_report_card(
                [run.run_dir],
                output_dir=output_dir,
                llm_summary_config=summary_config,
            )

            self.assertIn("llm_summary", report)
            self.assertIn("mock:summary", report["llm_summary"]["model"])
            html = (output_dir / "report_card.html").read_text(encoding="utf-8")
            self.assertIn("LLM Highlights", html)

    def test_report_card_adds_paired_mode_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            prior_path = Path(tmpdir) / "priors.json"
            prior_path.write_text(
                '{"name":"test","version":"1","models":['
                '{"provider_model_id":"provider/strong","estimated_rank":1},'
                '{"provider_model_id":"provider/weak","estimated_rank":2}'
                "]}",
                encoding="utf-8",
            )
            free = _run_with_rankings(
                tmpdir,
                _mode_config("free_fixture", "interactive_discussion"),
                [["P1", "P2"], ["P1", "P2"]],
            )
            structured = _run_with_rankings(
                tmpdir,
                _mode_config("structured_fixture", "round_robin_probes"),
                [["P2", "P1"], ["P1", "P2"]],
            )
            output_dir = Path(tmpdir) / "card"

            report = build_report_card(
                [free.run_dir, structured.run_dir],
                prior_ranking_file=prior_path,
                output_dir=output_dir,
            )

            self.assertIn("paired_comparison", report)
            html = (output_dir / "report_card.html").read_text(encoding="utf-8")
            self.assertIn("Paired Mode Comparison", html)
            self.assertIn("Free avg rank", html)
            self.assertIn("Structured avg rank", html)

    def test_round_robin_report_uses_round_labels_for_timeline_and_highlights(self) -> None:
        config = ExperimentConfig.from_dict(
            {
                "name": "report_card_round_fixture",
                "providers": [{"name": "mock", "kind": "mock"}],
                "models": [{"name": "model", "provider": "mock", "model": "mock:model"}],
                "participants": [{"id": "P1", "model": "model"}, {"id": "P2", "model": "model"}],
                "protocol": {
                    "name": "fixture_protocol",
                    "phases": [
                        {
                            "name": "probe_rounds",
                            "kind": "round_robin_probes",
                            "prompt": "round_robin_probe_question",
                            "question_prompt": "round_robin_probe_question",
                            "answer_prompt": "round_robin_probe_answer",
                            "assessment_prompt": "round_robin_probe_assessment",
                            "ranking_prompt": "round_robin_round_ranking",
                            "memory_prompt": "round_robin_memory_update",
                            "visibility": "public",
                            "rounds": 2,
                            "include_self": False,
                            "require_json": True,
                        },
                        {
                            "name": "final_judgment",
                            "kind": "private_judgment",
                            "prompt": "final_judgment",
                            "visibility": "private",
                            "require_json": True,
                        },
                    ],
                },
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            store = RunStore.create(tmpdir, config)
            store.append_entry(
                TranscriptEntry(
                    turn_id=1,
                    phase="probe_rounds",
                    speaker="P1",
                    visibility="private",
                    round_index=1,
                    content="Probe: explain a hidden rule from examples.",
                    metadata={
                        "interaction_mode": "round_robin_probes",
                        "interaction_role": "question",
                        "stream_id": "probe_rounds:round_1:P1:probe",
                        "probe_id": "probe_rounds:round_1:P1:probe",
                        "interviewer": "P1",
                        "respondents": ["P2"],
                    },
                )
            )
            store.append_entry(
                TranscriptEntry(
                    turn_id=2,
                    phase="probe_rounds",
                    speaker="P2",
                    visibility="private",
                    round_index=1,
                    content="The hidden rule is parity.",
                    metadata={
                        "interaction_mode": "round_robin_probes",
                        "interaction_role": "answer",
                        "stream_id": "probe_rounds:P1->P2",
                        "probe_id": "probe_rounds:round_1:P1:probe",
                        "interviewer": "P1",
                        "respondent": "P2",
                        "question_turn_id": 1,
                    },
                )
            )
            store.append_entry(
                TranscriptEntry(
                    turn_id=3,
                    phase="final_judgment",
                    speaker="P1",
                    visibility="private",
                    content="",
                    parsed={"ranking": ["P1", "P2"], "criteria": []},
                )
            )

            output_dir = Path(tmpdir) / "card"
            build_report_card([store.run_dir], output_dir=output_dir)

            html = (output_dir / "report_card.html").read_text(encoding="utf-8")
            self.assertIn("Round 1", html)
            self.assertIn("turn 1", html)
            self.assertIn("Expected Q/A", html)

    def test_report_taxonomy_counts_exclude_routed_answer_content(self) -> None:
        config = _mode_config("probe_count_fixture", "round_robin_probes")
        with tempfile.TemporaryDirectory() as tmpdir:
            store = RunStore.create(tmpdir, config)
            store.append_entry(
                TranscriptEntry(
                    turn_id=1,
                    phase="probe_rounds",
                    speaker="P1",
                    visibility="private",
                    round_index=1,
                    content="Write Python code that implements binary search.",
                    metadata={
                        "interaction_mode": "round_robin_probes",
                        "interaction_role": "question",
                        "stream_id": "probe_rounds:round_1:P1:probe",
                        "probe_id": "probe_rounds:round_1:P1:probe",
                        "interviewer": "P1",
                        "respondents": ["P2"],
                    },
                )
            )
            store.append_entry(
                TranscriptEntry(
                    turn_id=2,
                    phase="probe_rounds",
                    speaker="P2",
                    visibility="private",
                    round_index=1,
                    content="Python code: def binary_search(items, target): return -1",
                    metadata={
                        "interaction_mode": "round_robin_probes",
                        "interaction_role": "answer",
                        "stream_id": "probe_rounds:P1->P2",
                        "probe_id": "probe_rounds:round_1:P1:probe",
                        "interviewer": "P1",
                        "respondent": "P2",
                        "question_turn_id": 1,
                    },
                )
            )

            report = build_report_card([store.run_dir], output_dir=Path(tmpdir) / "card")
            card = report["runs"][0]

            self.assertEqual(card["probe_event_count"], 1)
            self.assertEqual(
                card["taxonomy_counts"]["question_type_frequency"][
                    "coding_algorithmic_reasoning"
                ],
                1,
            )
            self.assertEqual(
                card["transcript_taxonomy_counts"]["question_type_frequency"][
                    "coding_algorithmic_reasoning"
                ],
                1,
            )

    def test_adaptive_judge_report_shows_schedule_comparisons_and_round_checkpoints(self) -> None:
        config = ExperimentConfig.from_dict(
            {
                "name": "adaptive_report_fixture",
                "providers": [{"name": "mock", "kind": "mock"}],
                "models": [
                    {"name": "candidate", "provider": "mock", "model": "mock:candidate"},
                    {"name": "judge", "provider": "mock", "model": "mock:judge"},
                ],
                "participants": [
                    {"id": "P1", "model": "candidate"},
                    {"id": "P2", "model": "candidate"},
                ],
                "judges": [{"id": "J1", "model": "judge"}],
                "protocol": {
                    "name": "adaptive_report_protocol",
                    "phases": [
                        {
                            "name": "judge_ranking",
                            "kind": "independent_judge_ranking",
                            "prompt": "independent_judge_probe",
                            "visibility": "private",
                            "probe_schedule": [1, 1],
                            "adaptive_targeting": "judge_selected",
                            "require_json": True,
                        }
                    ],
                },
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            store = RunStore.create(tmpdir, config)
            store.append_entry(
                TranscriptEntry(
                    turn_id=1,
                    phase="judge_ranking",
                    speaker="J1",
                    visibility="private",
                    round_index=1,
                    content="Solve this logic problem and justify the result.",
                    metadata={
                        "interaction_mode": "independent_judge_ranking",
                        "interaction_role": "question",
                        "probe_id": "judge_ranking:J1:round_1:probe_1",
                        "probe_sequence_number": 1,
                        "respondents": ["P1", "P2"],
                    },
                )
            )
            for turn_id, candidate_id in [(2, "P1"), (3, "P2")]:
                store.append_entry(
                    TranscriptEntry(
                        turn_id=turn_id,
                        phase="judge_ranking",
                        speaker=candidate_id,
                        visibility="private",
                        round_index=1,
                        content=f"{candidate_id} answer",
                        metadata={
                            "interaction_mode": "independent_judge_ranking",
                            "interaction_role": "answer",
                            "probe_id": "judge_ranking:J1:round_1:probe_1",
                            "respondent": candidate_id,
                            "question_turn_id": 1,
                        },
                    )
                )
            store.append_entry(
                TranscriptEntry(
                    turn_id=4,
                    phase="judge_ranking",
                    speaker="J1",
                    visibility="private",
                    round_index=1,
                    content="",
                    parsed={
                        "probe_id": "judge_ranking:J1:round_1:probe_1",
                        "ordering": ["P1", "P2"],
                        "ties": [],
                        "confidence": 0.7,
                        "probe_validity": "informative",
                        "candidate_summaries": {"P1": "Strong.", "P2": "Weaker."},
                        "comparative_evidence": ["P1 found the contradiction."],
                        "uncertainties": [],
                    },
                    metadata={
                        "interaction_mode": "independent_judge_ranking",
                        "interaction_role": "probe_comparison",
                        "probe_id": "judge_ranking:J1:round_1:probe_1",
                        "probe_sequence_number": 1,
                        "question_turn_id": 1,
                        "answer_turn_ids": [2, 3],
                    },
                )
            )
            store.append_entry(
                TranscriptEntry(
                    turn_id=5,
                    phase="judge_ranking",
                    speaker="J1",
                    visibility="private",
                    round_index=1,
                    content="",
                    parsed={
                        "ranking": ["P1", "P2"],
                        "confidence": 0.7,
                        "criteria": ["Reasoning"],
                        "candidate_dossiers": {"P1": "Strong.", "P2": "Weaker."},
                        "comparative_evidence": ["P1 found the contradiction."],
                        "uncertainties": [],
                        "uncertain_pairs": [["P1", "P2"]],
                        "follow_up_candidates": ["P1", "P2"],
                        "follow_up_rationale": ["Resolve the close pair."],
                        "next_probe_strategy": ["Use a harder common probe."],
                    },
                    metadata={
                        "interaction_mode": "independent_judge_ranking",
                        "interaction_role": "wave_judgment",
                        "probe_comparison_turn_ids": [4],
                        "judgment_probe_count": 1,
                    },
                )
            )

            output_dir = Path(tmpdir) / "card"
            report = build_report_card([store.run_dir], output_dir=output_dir)
            card = report["runs"][0]
            html = (output_dir / "report_card.html").read_text(encoding="utf-8")

            self.assertEqual(card["structure"]["probe_schedule"], [1, 1])
            self.assertEqual(
                card["probe_comparisons"][0]["parsed"]["probe_validity"],
                "informative",
            )
            self.assertIn("Round-by-Round Ranking", html)
            self.assertIn("Per-Probe Comparisons", html)
            self.assertIn("[1, 1]", html)
            self.assertIn("Logic And Consistency", html)


def _run_with_rankings(
    tmpdir: str,
    config: ExperimentConfig,
    rankings: list[list[str]],
) -> RunStore:
    store = RunStore.create(tmpdir, config)
    for offset, ranking in enumerate(rankings, start=1):
        store.append_entry(
            TranscriptEntry(
                turn_id=offset,
                phase="final_judgment",
                speaker=f"P{offset}",
                visibility="private",
                content="",
                parsed={"ranking": ranking, "criteria": []},
            )
        )
    return store


def _mode_config(name: str, mode_kind: str) -> ExperimentConfig:
    public_phase = {
        "name": "discussion_round_1",
        "kind": "interactive_discussion",
        "prompt": "interactive_discussion_turn",
        "visibility": "public",
    }
    if mode_kind == "round_robin_probes":
        public_phase = {
            "name": "probe_rounds",
            "kind": "round_robin_probes",
            "prompt": "round_robin_probe_question",
            "question_prompt": "round_robin_probe_question",
            "answer_prompt": "round_robin_probe_answer",
            "assessment_prompt": "round_robin_probe_assessment",
            "ranking_prompt": "round_robin_round_ranking",
            "memory_prompt": "round_robin_memory_update",
            "visibility": "public",
            "rounds": 1,
            "include_self": False,
        }
    return ExperimentConfig.from_dict(
        {
            "name": name,
            "providers": [{"name": "mock", "kind": "mock"}],
            "models": [
                {"name": "strong", "provider": "mock", "model": "provider/strong"},
                {"name": "weak", "provider": "mock", "model": "provider/weak"},
            ],
            "participants": [
                {"id": "P1", "model": "strong"},
                {"id": "P2", "model": "weak"},
            ],
            "protocol": {
                "name": f"{name}_protocol",
                "phases": [
                    public_phase,
                    {
                        "name": "final_judgment",
                        "kind": "private_judgment",
                        "prompt": "final_judgment",
                        "visibility": "private",
                        "require_json": True,
                    },
                ],
            },
        }
    )


if __name__ == "__main__":
    unittest.main()

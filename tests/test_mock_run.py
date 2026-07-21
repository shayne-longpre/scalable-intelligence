from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ai_council.analysis import analyze_run
from ai_council.clients import build_clients
from ai_council.clients.base import ModelClient
from ai_council.config import ExperimentConfig, load_experiment_config
from ai_council.core import ModelRequest, ModelResponse, TranscriptEntry
from ai_council.orchestrator import BudgetExceededError, CouncilRunner, _extract_reported_cost
from ai_council.storage import RunStore, load_jsonl


ROOT = Path(__file__).resolve().parents[1]


class MockRunTests(unittest.TestCase):
    def test_extract_reported_cost_handles_strings_and_missing_values(self) -> None:
        self.assertEqual(_extract_reported_cost({}), 0.0)
        self.assertEqual(_extract_reported_cost({"cost": None}), 0.0)
        self.assertEqual(_extract_reported_cost({"cost": "0.125"}), 0.125)
        self.assertEqual(_extract_reported_cost({"cost": "not-a-number"}), 0.0)
        self.assertEqual(_extract_reported_cost({"cost": True}), 0.0)
        self.assertEqual(_extract_reported_cost({"cost": "nan"}), 0.0)
        self.assertEqual(_extract_reported_cost({"cost": float("inf")}), 0.0)
        self.assertEqual(_extract_reported_cost({"cost": -0.1}), 0.0)

    def test_mock_run_writes_transcript_and_analysis(self) -> None:
        config = load_experiment_config(ROOT / "examples" / "blind_council.mock.json")
        with tempfile.TemporaryDirectory() as tmpdir:
            store = RunStore.create(tmpdir, config)
            CouncilRunner(config, build_clients(config), store).run()
            entries = load_jsonl(store.transcript_path)
            self.assertEqual(len(entries), 77)
            self.assertTrue((store.run_dir / "run_summary.json").exists())
            run_summary = json.loads(
                (store.run_dir / "run_summary.json").read_text(encoding="utf-8")
            )
            self.assertGreaterEqual(run_summary["elapsed_seconds"], 0)
            self.assertEqual(run_summary["status"], "completed")
            self.assertEqual(run_summary["max_parallel_calls"], 1)
            self.assertLessEqual(run_summary["started_at"], run_summary["completed_at"])

            summary = analyze_run(store.run_dir)
            self.assertEqual(summary["turn_count"], 77)
            self.assertIn("question_quality", summary["criteria_frequency"])
            self.assertTrue((store.run_dir / "analysis_summary.json").exists())

    def test_run_and_analysis_log_reported_spend_by_model(self) -> None:
        config = ExperimentConfig.from_dict(
            {
                "name": "model_spend_test",
                "providers": [{"name": "cost", "kind": "mock"}],
                "models": [
                    {"name": "strong", "provider": "cost", "model": "provider/strong"},
                    {"name": "weak", "provider": "cost", "model": "provider/weak"},
                ],
                "participants": [
                    {"id": "P1", "model": "strong"},
                    {"id": "P2", "model": "weak"},
                ],
                "protocol": {
                    "name": "model_spend_protocol",
                    "phases": [
                        {
                            "name": "opening",
                            "kind": "public_round_robin",
                            "prompt": "opening_council",
                            "rounds": 1,
                            "visibility": "public",
                        }
                    ],
                },
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            store = RunStore.create(tmpdir, config)
            CouncilRunner(config, {"cost": PerModelCostClient()}, store).run()
            run_summary = json.loads(
                (store.run_dir / "run_summary.json").read_text(encoding="utf-8")
            )
            analysis = analyze_run(store.run_dir)
            report = (store.run_dir / "analysis_report.md").read_text(encoding="utf-8")

        self.assertEqual(run_summary["model_calls"], 2)
        self.assertAlmostEqual(run_summary["reported_cost_usd"], 0.3)
        self.assertEqual(run_summary["model_spend"]["strong"]["model_calls"], 1)
        self.assertAlmostEqual(
            run_summary["model_spend"]["strong"]["reported_cost_usd"],
            0.2,
        )
        self.assertAlmostEqual(
            run_summary["model_spend"]["weak"]["reported_cost_usd"],
            0.1,
        )
        self.assertEqual(analysis["model_spend"], run_summary["model_spend"])
        self.assertIn("## Reported Spend by Model", report)
        self.assertIn("| strong | `provider/strong` | 1 | $0.200000 |", report)

    def test_analysis_can_compare_rankings_to_prior(self) -> None:
        config = ExperimentConfig.from_dict(
            {
                "name": "prior_test",
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
                    "name": "prior_protocol",
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
            store = RunStore.create(tmpdir, config)
            store.append_entry(
                TranscriptEntry(
                    turn_id=1,
                    phase="final_judgment",
                    speaker="P1",
                    visibility="private",
                    content="",
                    created_at="2026-01-01T00:00:00+00:00",
                    parsed={"ranking": ["P1", "P2"], "criteria": []},
                )
            )
            summary = analyze_run(store.run_dir, prior_ranking_file=prior_path)
            agreement = summary["prior_agreement"]
            self.assertEqual(agreement["expected_order"], ["P1", "P2"])
            self.assertEqual(agreement["judgments"][0]["kendall_tau"], 1.0)
            self.assertTrue(agreement["judgments"][0]["top1_matches_prior"])

    def test_analysis_can_compare_rank_map_to_prior(self) -> None:
        config = ExperimentConfig.from_dict(
            {
                "name": "prior_rank_map_test",
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
                    "name": "prior_protocol",
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
            store = RunStore.create(tmpdir, config)
            store.append_entry(
                TranscriptEntry(
                    turn_id=1,
                    phase="final_judgment",
                    speaker="P2",
                    visibility="private",
                    content="",
                    created_at="2026-01-01T00:00:00+00:00",
                    parsed={"ranking": {"P2": 2, "P1": 1}, "criteria": []},
                )
            )
            summary = analyze_run(store.run_dir, prior_ranking_file=prior_path)
            judgment = summary["prior_agreement"]["judgments"][0]
            self.assertEqual(judgment["ranking"], ["P1", "P2"])
            self.assertEqual(judgment["kendall_tau"], 1.0)

    def test_analysis_counts_string_criteria_as_one_item(self) -> None:
        config = ExperimentConfig.from_dict(
            {
                "name": "criteria_test",
                "providers": [{"name": "mock", "kind": "mock"}],
                "models": [{"name": "mock_model", "provider": "mock", "model": "mock:model"}],
                "participants": [{"id": "P1", "model": "mock_model"}],
                "protocol": {
                    "name": "criteria_protocol",
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
            store = RunStore.create(tmpdir, config)
            store.append_entry(
                TranscriptEntry(
                    turn_id=1,
                    phase="final_judgment",
                    speaker="P1",
                    visibility="private",
                    content="",
                    parsed={"ranking": ["P1"], "criteria": "single criterion"},
                )
            )
            summary = analyze_run(store.run_dir)
            self.assertEqual(summary["criteria_frequency"], {"single criterion": 1})

    def test_call_budget_stops_run_before_extra_turns(self) -> None:
        config = ExperimentConfig.from_dict(
            {
                "name": "budget_test",
                "run": {"max_model_calls": 1},
                "providers": [{"name": "mock", "kind": "mock"}],
                "models": [{"name": "mock_model", "provider": "mock", "model": "mock:model"}],
                "participants": [
                    {"id": "P1", "model": "mock_model"},
                    {"id": "P2", "model": "mock_model"},
                ],
                "protocol": {
                    "name": "budget_protocol",
                    "phases": [
                        {
                            "name": "opening",
                            "kind": "public_round_robin",
                            "prompt": "opening_council",
                            "rounds": 1,
                            "visibility": "public",
                        }
                    ],
                },
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            store = RunStore.create(tmpdir, config)
            with self.assertRaises(BudgetExceededError):
                CouncilRunner(config, build_clients(config), store).run()
            self.assertEqual(len(load_jsonl(store.transcript_path)), 1)

    def test_private_structured_json_parse_failure_gets_same_model_repair(self) -> None:
        config = ExperimentConfig.from_dict(
            {
                "name": "json_repair_test",
                "providers": [{"name": "mock", "kind": "mock"}],
                "models": [
                    {
                        "name": "mock_model",
                        "provider": "mock",
                        "model": "mock:model",
                        "params": {"mock_malformed_json_once": True},
                    }
                ],
                "participants": [{"id": "P1", "model": "mock_model"}],
                "protocol": {
                    "name": "json_repair_protocol",
                    "phases": [
                        {
                            "name": "final_judgment",
                            "kind": "private_judgment",
                            "prompt": "final_judgment",
                            "visibility": "private",
                            "require_json": True,
                            "required_keys": ["participant_id", "phase", "ranking", "confidence"],
                        }
                    ],
                },
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            store = RunStore.create(tmpdir, config)
            CouncilRunner(config, build_clients(config), store).run()
            entries = load_jsonl(store.transcript_path)
            self.assertEqual(len(entries), 1)
            self.assertIsInstance(entries[0]["parsed"], dict)
            self.assertIsNone(entries[0]["metadata"]["parse_error"])
            repair = entries[0]["metadata"]["structured_json_repair"]
            self.assertTrue(repair["attempted"])
            self.assertTrue(repair["repaired"])
            self.assertIn("original_content", repair)
            self.assertEqual(load_jsonl(store.findings_path), [])
            run_summary = json.loads(
                (store.run_dir / "run_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(run_summary["model_calls"], 2)
            self.assertEqual(run_summary["model_spend"]["mock_model"]["model_calls"], 2)

    def test_private_structured_json_missing_key_gets_same_model_repair(self) -> None:
        config = ExperimentConfig.from_dict(
            {
                "name": "json_structural_repair_test",
                "providers": [{"name": "mock", "kind": "mock"}],
                "models": [
                    {
                        "name": "mock_model",
                        "provider": "mock",
                        "model": "mock:model",
                        "params": {"mock_missing_json_key_once": True},
                    }
                ],
                "participants": [{"id": "P1", "model": "mock_model"}],
                "protocol": {
                    "name": "json_structural_repair_protocol",
                    "phases": [
                        {
                            "name": "final_judgment",
                            "kind": "private_judgment",
                            "prompt": "final_judgment",
                            "visibility": "private",
                            "require_json": True,
                            "required_keys": ["participant_id", "phase", "ranking", "confidence"],
                        }
                    ],
                },
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            store = RunStore.create(tmpdir, config)
            CouncilRunner(config, build_clients(config), store).run()
            entries = load_jsonl(store.transcript_path)
            self.assertIsInstance(entries[0]["parsed"], dict)
            self.assertIn("confidence", entries[0]["parsed"])
            repair = entries[0]["metadata"]["structured_json_repair"]
            self.assertTrue(repair["repaired"])
            self.assertIn("missing_required_keys", repair["original_parse_error"])
            self.assertEqual(load_jsonl(store.findings_path), [])

    def test_private_structured_json_repair_can_be_disabled(self) -> None:
        config = ExperimentConfig.from_dict(
            {
                "name": "json_repair_disabled_test",
                "run": {"structured_json_retries": 0},
                "providers": [{"name": "mock", "kind": "mock"}],
                "models": [
                    {
                        "name": "mock_model",
                        "provider": "mock",
                        "model": "mock:model",
                        "params": {"mock_malformed_json_once": True},
                    }
                ],
                "participants": [{"id": "P1", "model": "mock_model"}],
                "protocol": {
                    "name": "json_repair_protocol",
                    "phases": [
                        {
                            "name": "final_judgment",
                            "kind": "private_judgment",
                            "prompt": "final_judgment",
                            "visibility": "private",
                            "require_json": True,
                            "required_keys": ["participant_id", "phase", "ranking", "confidence"],
                        }
                    ],
                },
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            store = RunStore.create(tmpdir, config)
            CouncilRunner(config, build_clients(config), store).run()
            entries = load_jsonl(store.transcript_path)
            self.assertIsNone(entries[0]["parsed"])
            self.assertNotIn("structured_json_repair", entries[0]["metadata"])
            self.assertEqual(load_jsonl(store.findings_path)[0]["code"], "missing_structured_json")

    def test_rotate_turn_order_changes_public_order_by_round(self) -> None:
        config = ExperimentConfig.from_dict(
            {
                "name": "rotate_test",
                "providers": [{"name": "mock", "kind": "mock"}],
                "models": [{"name": "mock_model", "provider": "mock", "model": "mock:model"}],
                "participants": [
                    {"id": "P1", "model": "mock_model"},
                    {"id": "P2", "model": "mock_model"},
                    {"id": "P3", "model": "mock_model"},
                ],
                "protocol": {
                    "name": "rotate_protocol",
                    "turn_order": "rotate",
                    "phases": [
                        {
                            "name": "public_round",
                            "kind": "public_round_robin",
                            "prompt": "opening_council",
                            "rounds": 2,
                            "visibility": "public",
                        }
                    ],
                },
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            store = RunStore.create(tmpdir, config)
            CouncilRunner(config, build_clients(config), store).run()
            speakers = [entry["speaker"] for entry in load_jsonl(store.transcript_path)]
            self.assertEqual(speakers, ["P1", "P2", "P3", "P2", "P3", "P1"])

    def test_generated_test_matrix_routes_answers_and_evaluations(self) -> None:
        config = ExperimentConfig.from_dict(
            {
                "name": "matrix_test",
                "providers": [{"name": "mock", "kind": "mock"}],
                "models": [{"name": "mock_model", "provider": "mock", "model": "mock:model"}],
                "participants": [
                    {"id": "P1", "model": "mock_model"},
                    {"id": "P2", "model": "mock_model"},
                    {"id": "P3", "model": "mock_model"},
                ],
                "protocol": {
                    "name": "matrix_protocol",
                    "turn_order": "fixed",
                    "phases": [
                        {
                            "name": "public_test_proposal",
                            "kind": "public_round_robin",
                            "prompt": "public_test_proposal",
                            "rounds": 1,
                            "visibility": "public",
                        },
                        {
                            "name": "test_application",
                            "kind": "public_test_matrix",
                            "prompt": "test_answer",
                            "visibility": "public",
                            "response_visibility": "private",
                            "source_phase": "public_test_proposal",
                        },
                        {
                            "name": "test_evaluation",
                            "kind": "public_test_evaluation",
                            "prompt": "test_evaluation",
                            "visibility": "public",
                            "source_phase": "public_test_proposal",
                            "answer_phase": "test_application",
                        },
                    ],
                },
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            store = RunStore.create(tmpdir, config)
            CouncilRunner(config, build_clients(config), store).run()
            entries = load_jsonl(store.transcript_path)
            application_entries = [entry for entry in entries if entry["phase"] == "test_application"]
            evaluation_entries = [entry for entry in entries if entry["phase"] == "test_evaluation"]

            self.assertEqual(len(application_entries), 9)
            self.assertEqual(len(evaluation_entries), 3)
            self.assertEqual({entry["visibility"] for entry in application_entries}, {"private"})
            self.assertEqual(
                sorted((entry["metadata"]["test_originator"], entry["metadata"]["respondent"]) for entry in application_entries),
                [
                    ("P1", "P1"),
                    ("P1", "P2"),
                    ("P1", "P3"),
                    ("P2", "P1"),
                    ("P2", "P2"),
                    ("P2", "P3"),
                    ("P3", "P1"),
                    ("P3", "P2"),
                    ("P3", "P3"),
                ],
            )
            self.assertEqual(
                {entry["metadata"]["test_originator"]: len(entry["metadata"]["answer_turn_ids"]) for entry in evaluation_entries},
                {"P1": 3, "P2": 3, "P3": 3},
            )


class PerModelCostClient(ModelClient):
    def generate(self, request: ModelRequest) -> ModelResponse:
        cost = 0.2 if request.model == "provider/strong" else 0.1
        return ModelResponse(
            content="cost fixture response",
            raw={},
            usage={"cost": cost},
            model=request.model,
            provider="cost",
        )


if __name__ == "__main__":
    unittest.main()

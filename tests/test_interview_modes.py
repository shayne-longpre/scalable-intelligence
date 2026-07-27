from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

from ai_council.analysis import analyze_run
from ai_council.clients import build_clients
from ai_council.clients.base import ModelClient, ModelClientError
from ai_council.clients.mock import MockModelClient
from ai_council.config import ExperimentConfig
from ai_council.core import ModelRequest, ModelResponse
from ai_council.orchestrator import (
    BudgetExceededError,
    CouncilRunner,
    ExperimentViolationError,
    _PreparedCall,
    _comparison_presentation_order,
    _load_preauthored_answers,
    _load_preauthored_probes,
    _preauthored_round_candidates,
)
from ai_council.storage import RunStore, load_jsonl
from ai_council.validation import revalidate_run


class InterviewModeTests(unittest.TestCase):
    def test_crossed_replay_preserves_source_adaptive_targets(self) -> None:
        config = _parallel_judge_config(max_parallel_calls=2, probe_schedule=[1])
        with tempfile.TemporaryDirectory() as tmpdir:
            store = RunStore.create(tmpdir, config)
            runner = CouncilRunner(config, build_clients(config), store)
            probes = {
                ("J1", 2, 1): {"metadata": {"respondents": ["P3", "P1"]}},
                ("J1", 2, 2): {"metadata": {"respondents": ["P3", "P1"]}},
            }

            selected = _preauthored_round_candidates(
                probes,
                "J1",
                2,
                2,
                runner.agents,
            )

        self.assertEqual([agent.spec.id for agent in selected], ["P3", "P1"])

    def test_preauthored_answer_directory_merges_transcript_and_pending_journal(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            entries = []
            for participant, stream_id in [("P1", "probe:P1"), ("P2", "probe:P2")]:
                entries.append(
                    {
                        "turn_id": 1 if participant == "P1" else 0,
                        "speaker": participant,
                        "content": f"Answer from {participant}",
                        "metadata": {
                            "interaction_role": "answer",
                            "stream_id": stream_id,
                            "model_ref": f"candidate_{participant.lower()}",
                            "finish_reason": "stop",
                        },
                    }
                )
            (run_dir / "transcript.jsonl").write_text(
                json.dumps(entries[0]) + "\n",
                encoding="utf-8",
            )
            (run_dir / "pending_batch_entries.jsonl").write_text(
                json.dumps(entries[1]) + "\n",
                encoding="utf-8",
            )

            loaded = _load_preauthored_answers(str(run_dir), set())

        self.assertEqual(set(loaded), {"probe:P1", "probe:P2"})
        self.assertTrue(loaded["probe:P1"]["source_file"].endswith("transcript.jsonl"))
        self.assertTrue(
            loaded["probe:P2"]["source_file"].endswith("pending_batch_entries.jsonl")
        )

    def test_preauthored_answer_directory_deduplicates_committed_journal_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            metadata = {
                "interaction_role": "answer",
                "stream_id": "probe:P1",
                "model_ref": "candidate_p1",
                "finish_reason": "stop",
            }
            committed = {
                "turn_id": 5,
                "speaker": "P1",
                "content": "same answer",
                "metadata": metadata,
            }
            pending = {
                **committed,
                "turn_id": 0,
                "metadata": {**metadata, "batch_position": 0},
            }
            (run_dir / "transcript.jsonl").write_text(
                json.dumps(committed) + "\n",
                encoding="utf-8",
            )
            (run_dir / "pending_batch_entries.jsonl").write_text(
                json.dumps(pending) + "\n",
                encoding="utf-8",
            )

            loaded = _load_preauthored_answers(str(run_dir), set())

        self.assertEqual(list(loaded), ["probe:P1"])
        self.assertEqual(loaded["probe:P1"]["source_turn_id"], 5)

    def test_preauthored_answer_loader_replays_explicitly_unavailable_answer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            transcript = Path(tmpdir) / "transcript.jsonl"
            transcript.write_text(
                json.dumps(
                    {
                        "turn_id": 3,
                        "speaker": "P1",
                        "content": "",
                        "metadata": {
                            "interaction_role": "answer",
                            "stream_id": "probe:P1",
                            "model_ref": "candidate_p1",
                            "finish_reason": "provider_error",
                            "answer_unavailable": True,
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            loaded = _load_preauthored_answers(str(transcript), set())
            included = _load_preauthored_answers(
                str(transcript),
                set(),
                include_unavailable=True,
            )

        self.assertNotIn("probe:P1", loaded)
        self.assertTrue(included["probe:P1"]["metadata"]["answer_unavailable"])

    def test_preauthored_answer_loader_retries_unavailable_selected_rounds(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            transcript = Path(tmpdir) / "transcript.jsonl"
            entries = [
                {
                    "turn_id": round_index,
                    "round_index": round_index,
                    "speaker": "P1",
                    "content": "",
                    "metadata": {
                        "interaction_role": "answer",
                        "stream_id": f"round-{round_index}:P1",
                        "answer_unavailable": True,
                    },
                }
                for round_index in (1, 2)
            ]
            transcript.write_text(
                "".join(json.dumps(entry) + "\n" for entry in entries),
                encoding="utf-8",
            )

            loaded = _load_preauthored_answers(
                str(transcript),
                set(),
                include_unavailable=True,
                retry_unavailable_rounds={2},
            )

        self.assertIn("round-1:P1", loaded)
        self.assertNotIn("round-2:P1", loaded)

    def test_independent_judge_can_replay_preauthored_probes_with_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            battery_path = Path(tmpdir) / "battery.json"
            battery_path.write_text(
                json.dumps(
                    {
                        "probes": [
                            {
                                "judge_id": "J1",
                                "round_index": 1,
                                "probe_number": 1,
                                "content": "A previously authored diagnostic probe.",
                                "source_run": "runs/source",
                                "source_turn_id": 7,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            config = ExperimentConfig.from_dict(
                {
                    "name": "preauthored_probe_test",
                    "providers": [{"name": "mock", "kind": "mock"}],
                    "models": [
                        {"name": "mock_model", "provider": "mock", "model": "mock:model"}
                    ],
                    "participants": [
                        {
                            "id": candidate_id,
                            "model": "mock_model",
                            "system_prompt": "blind_evaluation_candidate",
                        }
                        for candidate_id in ["P1", "P2"]
                    ],
                    "judges": [
                        {
                            "id": "J1",
                            "model": "mock_model",
                            "system_prompt": "independent_intelligence_judge",
                        }
                    ],
                    "protocol": {
                        "name": "preauthored_probe_protocol",
                        "phases": [
                            {
                                "name": "judge_ranking",
                                "kind": "independent_judge_ranking",
                                "prompt": "independent_judge_probe",
                                "probes_per_round": 1,
                                "preauthored_probe_file": str(battery_path),
                                "visibility": "private",
                            }
                        ],
                    },
                }
            )
            store = RunStore.create(tmpdir, config)
            CouncilRunner(config, build_clients(config), store).run()
            entries = load_jsonl(store.transcript_path)
            summary = json.loads((store.run_dir / "run_summary.json").read_text())
            shifted_transcript = Path(tmpdir) / "shifted_transcript.jsonl"
            shifted_transcript.write_text(
                "".join(
                    json.dumps(_shift_turn_ids(entry, offset=100)) + "\n"
                    for entry in entries
                ),
                encoding="utf-8",
            )

            replay_config = ExperimentConfig.from_dict(
                {
                    "name": "preauthored_judgment_test",
                    "providers": [{"name": "mock", "kind": "mock"}],
                    "models": [
                        {"name": "mock_model", "provider": "mock", "model": "mock:model"}
                    ],
                    "participants": [
                        {
                            "id": candidate_id,
                            "model": "mock_model",
                            "system_prompt": "blind_evaluation_candidate",
                        }
                        for candidate_id in ["P1", "P2"]
                    ],
                    "judges": [
                        {
                            "id": "J1",
                            "model": "mock_model",
                            "system_prompt": "independent_intelligence_judge",
                        }
                    ],
                    "protocol": {
                        "name": "preauthored_judgment_protocol",
                        "phases": [
                            {
                                "name": "judge_ranking",
                                "kind": "independent_judge_ranking",
                                "prompt": "independent_judge_probe",
                                "probes_per_round": 1,
                                "preauthored_probe_file": str(shifted_transcript),
                                "preauthored_answer_file": str(shifted_transcript),
                                "preauthored_answer_participants": ["P1", "P2"],
                                "preauthored_evidence_file": str(shifted_transcript),
                                "preauthored_ranking_file": str(shifted_transcript),
                                "visibility": "private",
                            }
                        ],
                    },
                }
            )
            replay_store = RunStore.create(tmpdir, replay_config)
            CouncilRunner(
                replay_config,
                build_clients(replay_config),
                replay_store,
            ).run()
            replay_entries = load_jsonl(replay_store.transcript_path)
            replay_summary = json.loads(
                (replay_store.run_dir / "run_summary.json").read_text()
            )

        question = entries[0]
        self.assertEqual(question["content"], "A previously authored diagnostic probe.")
        self.assertTrue(question["metadata"]["preauthored_probe"])
        self.assertEqual(question["metadata"]["source_run"], "runs/source")
        self.assertEqual(question["metadata"]["source_turn_id"], 7)
        self.assertEqual(summary["model_calls"], 5)
        self.assertEqual(replay_summary["model_calls"], 0)
        self.assertTrue(replay_entries[0]["metadata"]["preauthored_probe"])
        self.assertTrue(replay_entries[1]["metadata"]["preauthored_answer"])
        evidence = [
            entry
            for entry in replay_entries
            if entry["metadata"]["interaction_role"] == "evidence_card"
        ]
        self.assertTrue(all(entry["metadata"]["preauthored_evidence"] for entry in evidence))
        rankings = [
            entry
            for entry in replay_entries
            if entry["metadata"]["interaction_role"] == "judge_ranking"
        ]
        self.assertTrue(all(entry["metadata"]["preauthored_ranking"] for entry in rankings))

    def test_preauthored_probe_loader_skips_truncated_turns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            transcript = Path(tmpdir) / "transcript.jsonl"
            rows = [
                {
                    "turn_id": 1,
                    "speaker": "J1",
                    "round_index": 1,
                    "content": "Incomplete probe",
                    "metadata": {
                        "interaction_role": "question",
                        "probe_number": 1,
                        "finish_reason": "length",
                    },
                },
                {
                    "turn_id": 2,
                    "speaker": "J1",
                    "round_index": 1,
                    "content": "Complete probe",
                    "metadata": {
                        "interaction_role": "question",
                        "probe_number": 2,
                        "finish_reason": "stop",
                    },
                },
            ]
            transcript.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )

            probes = _load_preauthored_probes(str(transcript))

        self.assertNotIn(("J1", 1, 1), probes)
        self.assertEqual(probes[("J1", 1, 2)]["content"], "Complete probe")

    def test_independent_judge_parallel_batches_overlap_and_commit_deterministically(self) -> None:
        serial_entries, serial_peak, serial_requests = _run_parallel_judge_fixture(
            max_parallel_calls=1
        )
        parallel_entries, parallel_peak, parallel_requests = _run_parallel_judge_fixture(
            max_parallel_calls=3
        )

        self.assertEqual(serial_peak, 1)
        self.assertGreaterEqual(parallel_peak, 3)
        self.assertEqual(
            [_stable_entry_fields(entry) for entry in parallel_entries],
            [_stable_entry_fields(entry) for entry in serial_entries],
        )
        self.assertEqual(
            sorted(_stable_request_fields(request) for request in parallel_requests),
            sorted(_stable_request_fields(request) for request in serial_requests),
        )

    def test_independent_judge_parallel_recovery_matches_serial_execution(self) -> None:
        serial_entries, _, serial_requests = _run_parallel_judge_fixture(
            max_parallel_calls=1,
            client_class=DeterministicRecoveryMockClient,
        )
        parallel_entries, _, parallel_requests = _run_parallel_judge_fixture(
            max_parallel_calls=3,
            client_class=DeterministicRecoveryMockClient,
        )

        self.assertEqual(
            [_stable_entry_fields(entry) for entry in parallel_entries],
            [_stable_entry_fields(entry) for entry in serial_entries],
        )
        self.assertEqual(
            sorted(_stable_request_fields(request) for request in parallel_requests),
            sorted(_stable_request_fields(request) for request in serial_requests),
        )

    def test_independent_judge_serializes_clients_without_parallel_opt_in(self) -> None:
        _, peak_calls, _ = _run_parallel_judge_fixture(
            max_parallel_calls=3,
            client_class=SerializedTrackingMockClient,
        )

        self.assertEqual(peak_calls, 1)

    def test_parallel_scheduler_does_not_let_serial_client_starve_other_clients(self) -> None:
        config = _parallel_judge_config(max_parallel_calls=2)
        parallel_started = threading.Event()
        serial_client = BlockingSerialClient(parallel_started)
        parallel_client = SignalingParallelClient(parallel_started)
        calls = [
            _PreparedCall(serial_client, _scheduler_request("serial-1"), "mock_model"),
            _PreparedCall(serial_client, _scheduler_request("serial-2"), "mock_model"),
            _PreparedCall(parallel_client, _scheduler_request("parallel"), "mock_model"),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            store = RunStore.create(tmpdir, config)
            runner = CouncilRunner(config, {"mock": parallel_client}, store)
            responses = runner._run_prepared_calls(calls, workers=2)

        self.assertTrue(serial_client.parallel_started_before_release)
        self.assertEqual(serial_client.peak_calls, 1)
        self.assertEqual(
            [response.model for response in responses],
            ["serial-1", "serial-2", "parallel"],
        )

    def test_parallel_scheduler_stops_submitting_after_cost_budget_is_crossed(self) -> None:
        config = _parallel_judge_config(
            max_parallel_calls=2,
            probes_per_round=1,
            max_reported_cost_usd=0.15,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            store = RunStore.create(tmpdir, config)
            client = CostTrackingMockClient(config.providers["mock"], cost_per_call=0.1)
            runner = CouncilRunner(config, {"mock": client}, store)

            with self.assertRaises(BudgetExceededError):
                runner.run()

            entries = load_jsonl(store.transcript_path)
            run_summary = json.loads(
                (store.run_dir / "run_summary.json").read_text(encoding="utf-8")
            )

        self.assertEqual(runner.model_calls, 3)
        self.assertAlmostEqual(runner.reported_cost_usd, 0.3)
        self.assertEqual(len(client.requests), 3)
        self.assertEqual(client.peak_calls, 2)
        self.assertEqual(
            [entry["metadata"]["interaction_role"] for entry in entries],
            ["question", "answer", "answer"],
        )
        self.assertEqual(run_summary["status"], "failed")
        self.assertEqual(run_summary["model_calls"], 3)
        self.assertEqual(run_summary["error"]["type"], "BudgetExceededError")

    def test_parallel_scheduler_stops_queued_work_after_provider_failure(self) -> None:
        config = _parallel_judge_config(max_parallel_calls=2, probes_per_round=1)
        with tempfile.TemporaryDirectory() as tmpdir:
            store = RunStore.create(tmpdir, config)
            client = FailingParallelMockClient(config.providers["mock"])
            runner = CouncilRunner(config, {"mock": client}, store)

            with self.assertRaises(ModelClientError):
                runner.run()

            entries = load_jsonl(store.transcript_path)
            run_summary = json.loads(
                (store.run_dir / "run_summary.json").read_text(encoding="utf-8")
            )
            failures = load_jsonl(store.batch_failures_path)

        self.assertEqual(len(client.requests), 3)
        self.assertEqual(client.peak_calls, 2)
        self.assertEqual(runner.model_calls, 2)
        self.assertEqual(
            [entry["metadata"]["interaction_role"] for entry in entries],
            ["question", "answer"],
        )
        self.assertEqual(entries[-1]["speaker"], "P2")
        self.assertEqual(run_summary["status"], "failed")
        self.assertEqual(run_summary["error"]["type"], "ModelClientError")
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["speaker"], "P1")
        self.assertEqual(failures[0]["error_type"], "ModelClientError")

    def test_parallel_scheduler_can_finish_batch_after_provider_failure(self) -> None:
        config = _parallel_judge_config(
            max_parallel_calls=2,
            probes_per_round=1,
            continue_batch_on_call_error=True,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            store = RunStore.create(tmpdir, config)
            client = FailingParallelMockClient(config.providers["mock"])
            runner = CouncilRunner(config, {"mock": client}, store)

            with self.assertRaises(ModelClientError):
                runner.run()

            entries = load_jsonl(store.transcript_path)

        answers = [
            entry
            for entry in entries
            if entry["metadata"].get("interaction_role") == "answer"
        ]
        self.assertEqual(len(client.requests), 4)
        self.assertEqual(runner.model_calls, 3)
        self.assertEqual([entry["speaker"] for entry in answers], ["P2", "P3"])

    def test_incomplete_answer_does_not_cancel_other_candidate_calls(self) -> None:
        config = _parallel_judge_config(
            max_parallel_calls=2,
            probes_per_round=1,
            visible_text_retries=0,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            store = RunStore.create(tmpdir, config)
            client = IncompleteParallelMockClient(config.providers["mock"])
            with self.assertRaisesRegex(ExperimentViolationError, "incomplete answer"):
                CouncilRunner(config, {"mock": client}, store).run()
            entries = load_jsonl(store.transcript_path)

        answers = [
            entry
            for entry in entries
            if entry["metadata"].get("interaction_role") == "answer"
        ]
        self.assertEqual(len(client.requests), 4)
        self.assertEqual([entry["speaker"] for entry in answers], ["P1", "P2", "P3"])
        self.assertEqual(answers[0]["content"], "")

    def test_incomplete_answer_can_be_recorded_as_unavailable_after_qualification(self) -> None:
        config = _parallel_judge_config(
            max_parallel_calls=2,
            probes_per_round=1,
            visible_text_retries=0,
            incomplete_answer_policy="record_unavailable",
            probe_schedule=[1],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            store = RunStore.create(tmpdir, config)
            client = IncompleteParallelMockClient(config.providers["mock"])
            CouncilRunner(config, {"mock": client}, store).run()
            entries = load_jsonl(store.transcript_path)
            analyze_run(store.run_dir)
            archive = json.loads(
                (store.run_dir / "probe_answer_archive.json").read_text(encoding="utf-8")
            )

        unavailable = next(
            entry
            for entry in entries
            if entry["metadata"].get("interaction_role") == "answer"
            and entry["speaker"] == "P1"
        )
        comparison_request = next(
            request
            for request in client.requests
            if request.metadata.get("interaction_role") == "probe_comparison"
        )
        self.assertEqual(unavailable["content"], "")
        self.assertTrue(unavailable["metadata"]["answer_unavailable"])
        self.assertIn("missing evidence, not evidence of low capability", comparison_request.messages[-1]["content"])
        archived = next(
            answer
            for answer in archive["probes"][0]["answers"]
            if answer["candidate_id"] == "P1"
        )
        self.assertTrue(archived["answer_unavailable"])

    def test_provider_failure_can_be_recorded_as_unavailable_after_qualification(self) -> None:
        config = _parallel_judge_config(
            max_parallel_calls=2,
            probes_per_round=1,
            continue_batch_on_call_error=True,
            incomplete_answer_policy="record_unavailable",
            probe_schedule=[1],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            store = RunStore.create(tmpdir, config)
            client = FailingParallelMockClient(config.providers["mock"])
            CouncilRunner(config, {"mock": client}, store).run()
            entries = load_jsonl(store.transcript_path)
            failures = load_jsonl(store.batch_failures_path)

        unavailable = next(
            entry
            for entry in entries
            if entry["metadata"].get("interaction_role") == "answer"
            and entry["speaker"] == "P1"
        )
        self.assertEqual(unavailable["content"], "")
        self.assertTrue(unavailable["metadata"]["answer_unavailable"])
        self.assertEqual(unavailable["metadata"]["provider_error_type"], "ModelClientError")
        self.assertEqual(len(failures), 1)

    def test_serial_provider_failure_uses_the_same_unavailable_policy(self) -> None:
        config = _parallel_judge_config(
            max_parallel_calls=1,
            probes_per_round=1,
            continue_batch_on_call_error=True,
            incomplete_answer_policy="record_unavailable",
            probe_schedule=[1],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            store = RunStore.create(tmpdir, config)
            client = FailingParallelMockClient(config.providers["mock"])
            CouncilRunner(config, {"mock": client}, store).run()
            entries = load_jsonl(store.transcript_path)

        answers = [
            entry
            for entry in entries
            if entry["metadata"].get("interaction_role") == "answer"
        ]
        self.assertEqual([entry["speaker"] for entry in answers], ["P1", "P2", "P3"])
        self.assertTrue(answers[0]["metadata"]["answer_unavailable"])
        self.assertTrue(answers[1]["content"].strip())

    def test_parallel_batch_journals_completed_entries_before_slowest_call_finishes(self) -> None:
        config = _parallel_judge_config(max_parallel_calls=2, probes_per_round=1)
        with tempfile.TemporaryDirectory() as tmpdir:
            store = RunStore.create(tmpdir, config)
            client = CheckpointBlockingMockClient(config.providers["mock"])
            runner = CouncilRunner(config, {"mock": client}, store)
            error: list[Exception] = []

            def run() -> None:
                try:
                    runner.run()
                except Exception as exc:
                    error.append(exc)

            thread = threading.Thread(target=run)
            thread.start()
            self.assertTrue(client.first_answer_completed.wait(1))
            for _ in range(100):
                if store.pending_batch_path.exists():
                    break
                time.sleep(0.01)
            pending = load_jsonl(store.pending_batch_path)
            client.release_answers.set()
            thread.join(3)
            pending_cleared = not store.pending_batch_path.exists()

        self.assertFalse(thread.is_alive())
        self.assertEqual(error, [])
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["metadata"]["interaction_role"], "answer")
        self.assertEqual(pending[0]["metadata"]["respondent"], "P1")
        self.assertTrue(pending_cleared)

    def test_batch_preflights_the_full_call_budget_at_every_concurrency(self) -> None:
        for max_parallel_calls in (1, 2):
            with self.subTest(max_parallel_calls=max_parallel_calls):
                config = _parallel_judge_config(
                    max_parallel_calls=max_parallel_calls,
                    probes_per_round=1,
                    max_model_calls=2,
                )
                with tempfile.TemporaryDirectory() as tmpdir:
                    store = RunStore.create(tmpdir, config)
                    client = ConcurrencyTrackingMockClient(config.providers["mock"])
                    runner = CouncilRunner(config, {"mock": client}, store)

                    with self.assertRaises(BudgetExceededError):
                        runner.run()

                    entries = load_jsonl(store.transcript_path)

                self.assertEqual(len(client.requests), 1)
                self.assertEqual(runner.model_calls, 1)
                self.assertEqual(
                    [entry["metadata"]["interaction_role"] for entry in entries],
                    ["question"],
                )

    def test_independent_judge_aborts_before_scoring_incomplete_candidate_answer(self) -> None:
        config = ExperimentConfig.from_dict(
            {
                "name": "independent_judge_incomplete_answer_test",
                "run": {"visible_text_retries": 1, "max_model_calls": 3},
                "providers": [{"name": "mock", "kind": "mock"}],
                "models": [
                    {"name": "judge_model", "provider": "mock", "model": "mock:judge"},
                    {
                        "name": "candidate_model",
                        "provider": "mock",
                        "model": "mock:candidate",
                        "params": {"mock_empty_visible_once": True},
                    },
                ],
                "participants": [
                    {
                        "id": candidate_id,
                        "model": "candidate_model",
                        "system_prompt": "blind_evaluation_candidate",
                    }
                    for candidate_id in ["P1", "P2"]
                ],
                "judges": [
                    {
                        "id": "J1",
                        "model": "judge_model",
                        "system_prompt": "independent_intelligence_judge",
                    }
                ],
                "protocol": {
                    "name": "incomplete_answer_protocol",
                    "phases": [
                        {
                            "name": "judge_ranking",
                            "kind": "independent_judge_ranking",
                            "prompt": "independent_judge_probe",
                            "rounds": 1,
                            "visibility": "private",
                        }
                    ],
                },
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            store = RunStore.create(tmpdir, config)
            client = ConcurrencyTrackingMockClient(config.providers["mock"])
            with self.assertRaisesRegex(ExperimentViolationError, "incomplete answer"):
                CouncilRunner(config, {"mock": client}, store).run()
            entries = load_jsonl(store.transcript_path)

        self.assertEqual(len(client.requests), 2)
        self.assertEqual([entry["metadata"]["interaction_role"] for entry in entries], ["question", "answer"])
        self.assertEqual(entries[-1]["content"], "")

    def test_independent_judge_recovers_empty_visible_output_before_routing(self) -> None:
        config = ExperimentConfig.from_dict(
            {
                "name": "independent_judge_empty_recovery_test",
                "run": {"visible_text_retries": 1, "max_parallel_calls": 2},
                "providers": [{"name": "mock", "kind": "mock"}],
                "models": [
                    {
                        "name": "judge_model",
                        "provider": "mock",
                        "model": "mock:judge",
                        "params": {"mock_empty_visible_once": True},
                        "recovery_params": {"reasoning": {"effort": "low"}},
                    },
                    {"name": "candidate_model", "provider": "mock", "model": "mock:candidate"},
                ],
                "participants": [
                    {
                        "id": candidate_id,
                        "model": "candidate_model",
                        "system_prompt": "blind_evaluation_candidate",
                    }
                    for candidate_id in ["P1", "P2"]
                ],
                "judges": [
                    {
                        "id": "J1",
                        "model": "judge_model",
                        "system_prompt": "independent_intelligence_judge",
                    }
                ],
                "protocol": {
                    "name": "empty_recovery_protocol",
                    "phases": [
                        {
                            "name": "judge_ranking",
                            "kind": "independent_judge_ranking",
                            "prompt": "independent_judge_probe",
                            "rounds": 1,
                            "probes_per_round": 1,
                            "visibility": "private",
                            "recovery_model_params": {"reasoning": {"effort": "none"}},
                        }
                    ],
                },
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            store = RunStore.create(tmpdir, config)
            client = ConcurrencyTrackingMockClient(config.providers["mock"])
            CouncilRunner(config, {"mock": client}, store).run()
            entries = load_jsonl(store.transcript_path)
            summary = analyze_run(store.run_dir)

        recovered = [entry for entry in entries if entry["metadata"].get("visible_text_retry")]
        self.assertEqual(len(recovered), 3)
        self.assertTrue(all(entry["metadata"]["visible_text_retry"]["recovered"] for entry in recovered))
        question = next(entry for entry in entries if entry["metadata"]["interaction_role"] == "question")
        self.assertTrue(question["content"].strip())
        self.assertEqual(
            question["metadata"]["visible_text_retry"]["attempts"][0]["request_params"]["reasoning"],
            {"effort": "low"},
        )
        self.assertEqual(
            summary["behavior_audit"]["codes"].get("empty_visible_output_recovered"),
            3,
        )
        self.assertNotIn("missing_final_judgment", summary["behavior_audit"]["codes"])

    def test_visible_recovery_falls_back_when_provider_rejects_override(self) -> None:
        config = ExperimentConfig.from_dict(
            {
                "name": "rejected_recovery_override_test",
                "run": {"visible_text_retries": 1, "max_parallel_calls": 2},
                "providers": [{"name": "mock", "kind": "mock"}],
                "models": [
                    {"name": "judge", "provider": "mock", "model": "mock:judge"},
                    {
                        "name": "candidate",
                        "provider": "mock",
                        "model": "mock:candidate",
                        "params": {
                            "mock_empty_visible_once": True,
                            "reasoning": {"effort": "medium"},
                        },
                    },
                ],
                "participants": [
                    {
                        "id": candidate_id,
                        "model": "candidate",
                        "system_prompt": "blind_evaluation_candidate",
                    }
                    for candidate_id in ["P1", "P2"]
                ],
                "judges": [
                    {
                        "id": "J1",
                        "model": "judge",
                        "system_prompt": "independent_intelligence_judge",
                    }
                ],
                "protocol": {
                    "name": "rejected_recovery_override_protocol",
                    "phases": [
                        {
                            "name": "judge_ranking",
                            "kind": "independent_judge_ranking",
                            "prompt": "independent_judge_probe",
                            "probes_per_round": 1,
                            "recovery_model_params": {
                                "reasoning": {"effort": "none"}
                            },
                            "visibility": "private",
                        }
                    ],
                },
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            store = RunStore.create(tmpdir, config)
            client = RecoveryOverrideRejectingMockClient(config.providers["mock"])
            CouncilRunner(config, {"mock": client}, store).run()
            entries = load_jsonl(store.transcript_path)

        recovered_answers = [
            entry
            for entry in entries
            if entry["metadata"].get("interaction_role") == "answer"
            and entry["metadata"].get("visible_text_retry")
        ]
        self.assertEqual(len(recovered_answers), 2)
        for entry in recovered_answers:
            attempt = entry["metadata"]["visible_text_retry"]["attempts"][0]
            self.assertIn("HTTP 400", attempt["rejected_recovery_override"])
            self.assertEqual(attempt["request_params"]["reasoning"], {"effort": "medium"})

    def test_independent_judges_rank_candidates_without_cross_judge_state(self) -> None:
        config = ExperimentConfig.from_dict(
            {
                "name": "independent_judge_test",
                "context": {
                    "mode": "private_memory",
                    "max_public_turns": 0,
                    "max_private_turns": 12,
                    "max_stream_turns": 8,
                },
                "providers": [{"name": "mock", "kind": "mock"}],
                "models": [{"name": "mock_model", "provider": "mock", "model": "mock:model"}],
                "participants": [
                    {
                        "id": candidate_id,
                        "model": "mock_model",
                        "system_prompt": "blind_evaluation_candidate",
                    }
                    for candidate_id in ["P1", "P2", "P3"]
                ],
                "judges": [
                    {
                        "id": judge_id,
                        "model": "mock_model",
                        "system_prompt": "independent_intelligence_judge",
                    }
                    for judge_id in ["J1", "J2"]
                ],
                "protocol": {
                    "name": "independent_judge_protocol",
                    "phases": [
                        {
                            "name": "judge_ranking",
                            "kind": "independent_judge_ranking",
                            "prompt": "independent_judge_probe",
                            "rounds": 1,
                            "probes_per_round": 2,
                            "visibility": "private",
                        }
                    ],
                },
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            store = RunStore.create(tmpdir, config)
            CouncilRunner(config, build_clients(config), store).run()
            entries = load_jsonl(store.transcript_path)
            findings = load_jsonl(store.findings_path)

            role_counts = {}
            for entry in entries:
                role = entry["metadata"]["interaction_role"]
                role_counts[role] = role_counts.get(role, 0) + 1
                self.assertEqual(
                    entry["metadata"]["interaction_mode"],
                    "independent_judge_ranking",
                )
            self.assertEqual(
                role_counts,
                {"question": 4, "answer": 12, "evidence_card": 6, "judge_ranking": 2},
            )
            self.assertEqual(findings, [])

            rankings = [
                entry for entry in entries if entry["metadata"]["interaction_role"] == "judge_ranking"
            ]
            self.assertEqual({entry["speaker"] for entry in rankings}, {"J1", "J2"})
            self.assertTrue(all(entry["parsed"]["ranking"] == ["P1", "P2", "P3"] for entry in rankings))
            self.assertTrue(all("J1" not in entry["parsed"]["ranking"] for entry in rankings))
            self.assertTrue(all("J2" not in entry["parsed"]["ranking"] for entry in rankings))

            summary = analyze_run(store.run_dir)
            self.assertEqual(summary["posthoc_extraction"]["qa_pair_count"], 12)
            self.assertEqual(
                summary["posthoc_extraction"]["qa_pairs_by_mode"],
                {"independent_judge_ranking": 12},
            )

    def test_independent_judge_probe_prefixes_create_isolated_judgment_branches(self) -> None:
        config = ExperimentConfig.from_dict(
            {
                "name": "independent_judge_prefix_test",
                "run": {"max_parallel_calls": 4},
                "providers": [{"name": "mock", "kind": "mock"}],
                "models": [{"name": "mock_model", "provider": "mock", "model": "mock:model"}],
                "participants": [
                    {
                        "id": candidate_id,
                        "model": "mock_model",
                        "system_prompt": "blind_evaluation_candidate",
                    }
                    for candidate_id in ["P1", "P2", "P3"]
                ],
                "judges": [
                    {
                        "id": "J1",
                        "model": "mock_model",
                        "system_prompt": "independent_intelligence_judge",
                    }
                ],
                "protocol": {
                    "name": "independent_judge_prefix_protocol",
                    "phases": [
                        {
                            "name": "judge_ranking",
                            "kind": "independent_judge_ranking",
                            "prompt": "independent_judge_probe",
                            "rounds": 1,
                            "probes_per_round": 6,
                            "judgment_probe_counts": [2, 4, 6],
                            "visibility": "private",
                        }
                    ],
                },
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            store = RunStore.create(tmpdir, config)
            CouncilRunner(config, build_clients(config), store).run()
            entries = load_jsonl(store.transcript_path)
            summary = analyze_run(store.run_dir)

        evidence_cards = [
            entry for entry in entries if entry["metadata"]["interaction_role"] == "evidence_card"
        ]
        rankings = [
            entry for entry in entries if entry["metadata"]["interaction_role"] == "judge_ranking"
        ]
        self.assertEqual(len(evidence_cards), 9)
        self.assertEqual(len(rankings), 3)
        self.assertEqual(
            {entry["metadata"]["judgment_probe_count"] for entry in rankings},
            {2, 4, 6},
        )
        for entry in evidence_cards:
            count = entry["metadata"]["judgment_probe_count"]
            self.assertEqual(len(entry["metadata"]["question_turn_ids"]), count)
            self.assertEqual(len(entry["metadata"]["answer_turn_ids"]), count)
            self.assertEqual(entry["metadata"]["is_primary_judgment"], count == 6)
        self.assertEqual(len(summary["metrics"]["rankings"]["final_rankings"]), 1)
        self.assertEqual(
            summary["metrics"]["rankings"]["final_rankings"][0]["judgment_probe_count"],
            6,
        )

    def test_independent_judge_adaptive_round_routes_only_selected_candidates(self) -> None:
        config = ExperimentConfig.from_dict(
            {
                "name": "independent_judge_adaptive_test",
                "providers": [{"name": "mock", "kind": "mock"}],
                "models": [{"name": "mock_model", "provider": "mock", "model": "mock:model"}],
                "participants": [
                    {
                        "id": candidate_id,
                        "model": "mock_model",
                        "system_prompt": "blind_evaluation_candidate",
                    }
                    for candidate_id in ["P1", "P2", "P3", "P4"]
                ],
                "judges": [
                    {
                        "id": "J1",
                        "model": "mock_model",
                        "system_prompt": "independent_intelligence_judge",
                    }
                ],
                "protocol": {
                    "name": "adaptive_judge_protocol",
                    "phases": [
                        {
                            "name": "judge_ranking",
                            "kind": "independent_judge_ranking",
                            "prompt": "independent_judge_probe",
                            "rounds": 2,
                            "probes_per_round": 1,
                            "adaptive_probes_per_round": 1,
                            "max_adaptive_candidates": 2,
                            "visibility": "private",
                        }
                    ],
                },
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            store = RunStore.create(tmpdir, config)
            client = ConcurrencyTrackingMockClient(config.providers["mock"])
            CouncilRunner(config, {"mock": client}, store).run()
            entries = load_jsonl(store.transcript_path)

        round_two_answers = [
            entry
            for entry in entries
            if entry["round_index"] == 2 and entry["metadata"]["interaction_role"] == "answer"
        ]
        self.assertEqual({entry["speaker"] for entry in round_two_answers}, {"P1", "P2"})
        evidence_counts = {}
        for entry in entries:
            if entry["metadata"]["interaction_role"] != "evidence_card":
                continue
            candidate_id = entry["metadata"]["candidate"]
            evidence_counts[candidate_id] = evidence_counts.get(candidate_id, 0) + 1
        self.assertEqual(evidence_counts, {"P1": 2, "P2": 2, "P3": 1, "P4": 1})
        rankings = [
            entry for entry in entries if entry["metadata"]["interaction_role"] == "judge_ranking"
        ]
        self.assertEqual(len(rankings), 2)
        self.assertTrue(all(entry["parsed"]["ranking"] == ["P1", "P2", "P3", "P4"] for entry in rankings))
        round_two_question = next(
            entry
            for entry in entries
            if entry["round_index"] == 2 and entry["metadata"]["interaction_role"] == "question"
        )
        self.assertEqual(round_two_question["metadata"]["generation_stage"], "adaptive_followup")
        self.assertEqual(len(round_two_question["metadata"]["evidence_turn_ids_available"]), 3)
        round_two_request = next(
            request
            for request in client.requests
            if request.metadata.get("round_index") == 2
            and request.metadata.get("interaction_role") == "question"
        )
        user_prompt = round_two_request.messages[-1]["content"]
        self.assertIn("Prior full-evidence ranking", user_prompt)
        self.assertIn("Relevant prior evidence cards", user_prompt)

    def test_adaptive_probe_schedule_compares_each_probe_then_merges_each_round(self) -> None:
        config = ExperimentConfig.from_dict(
            {
                "name": "adaptive_probe_wave_test",
                "run": {"max_parallel_calls": 8},
                "providers": [{"name": "mock", "kind": "mock"}],
                "models": [{"name": "mock_model", "provider": "mock", "model": "mock:model"}],
                "participants": [
                    {
                        "id": candidate_id,
                        "model": "mock_model",
                        "system_prompt": "blind_evaluation_candidate",
                    }
                    for candidate_id in ["P1", "P2", "P3", "P4"]
                ],
                "judges": [
                    {
                        "id": "J1",
                        "model": "mock_model",
                        "system_prompt": "independent_intelligence_judge",
                    }
                ],
                "protocol": {
                    "name": "adaptive_probe_wave_protocol",
                    "phases": [
                        {
                            "name": "judge_ranking",
                            "kind": "independent_judge_ranking",
                            "prompt": "adaptive_judge_probe",
                            "probe_schedule": [4, 1, 2],
                            "max_adaptive_candidates": 2,
                            "visibility": "private",
                        }
                    ],
                },
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            store = RunStore.create(tmpdir, config)
            client = ConcurrencyTrackingMockClient(config.providers["mock"])
            CouncilRunner(config, {"mock": client}, store).run()
            entries = load_jsonl(store.transcript_path)
            summary = analyze_run(store.run_dir)

        role_counts = {}
        for entry in entries:
            role = entry["metadata"]["interaction_role"]
            role_counts[role] = role_counts.get(role, 0) + 1
        self.assertEqual(
            role_counts,
            {"question": 7, "answer": 22, "probe_comparison": 7, "wave_judgment": 3},
        )

        first_answer_index = next(
            index
            for index, entry in enumerate(entries)
            if entry["metadata"]["interaction_role"] == "answer"
        )
        self.assertTrue(
            all(
                entry["metadata"]["interaction_role"] == "question"
                for entry in entries[:first_answer_index]
            )
        )
        self.assertEqual(first_answer_index, 4)

        comparisons = [
            entry
            for entry in entries
            if entry["metadata"]["interaction_role"] == "probe_comparison"
        ]
        self.assertTrue(all(len(entry["metadata"]["answer_turn_ids"]) == 4 for entry in comparisons[:4]))
        self.assertTrue(all(len(entry["metadata"]["answer_turn_ids"]) == 2 for entry in comparisons[4:]))
        self.assertTrue(all("ability_score" not in entry["parsed"] for entry in comparisons))

        judgments = [
            entry
            for entry in entries
            if entry["metadata"]["interaction_role"] == "wave_judgment"
        ]
        self.assertEqual(
            [entry["metadata"]["judgment_probe_count"] for entry in judgments],
            [4, 5, 7],
        )
        self.assertEqual(
            [entry["metadata"].get("prior_judgment_turn_id") for entry in judgments],
            [None, judgments[0]["turn_id"], judgments[1]["turn_id"]],
        )
        self.assertTrue(all(set(entry["parsed"]["candidate_dossiers"]) == {"P1", "P2", "P3", "P4"} for entry in judgments))
        self.assertTrue(all("scores" not in entry["parsed"] for entry in judgments))

        later_answers = [
            entry
            for entry in entries
            if entry["round_index"] in {2, 3}
            and entry["metadata"]["interaction_role"] == "answer"
        ]
        self.assertEqual({entry["speaker"] for entry in later_answers}, {"P1", "P2"})
        round_two_probe_request = next(
            request
            for request in client.requests
            if request.metadata.get("interaction_role") == "question"
            and request.metadata.get("round_index") == 2
        )
        self.assertIn("Prior cumulative judgment", round_two_probe_request.messages[-1]["content"])
        self.assertEqual(summary["posthoc_extraction"]["qa_pair_count"], 22)
        self.assertEqual(summary["posthoc_extraction"]["probe_comparison_count"], 7)
        self.assertEqual(summary["posthoc_extraction"]["wave_judgment_count"], 3)
        self.assertEqual(summary["metrics"]["rankings"]["churn_by_speaker"]["J1"]["snapshot_count"], 3)

    def test_adaptive_probe_schedule_can_replay_every_stage(self) -> None:
        data = {
            "name": "adaptive_wave_replay_source",
            "providers": [{"name": "mock", "kind": "mock"}],
            "models": [{"name": "mock_model", "provider": "mock", "model": "mock:model"}],
            "participants": [
                {"id": candidate_id, "model": "mock_model", "system_prompt": "blind_evaluation_candidate"}
                for candidate_id in ["P1", "P2", "P3"]
            ],
            "judges": [
                {"id": "J1", "model": "mock_model", "system_prompt": "independent_intelligence_judge"}
            ],
            "protocol": {
                "name": "adaptive_wave_replay",
                "phases": [
                    {
                        "name": "judge_ranking",
                        "kind": "independent_judge_ranking",
                        "prompt": "adaptive_judge_probe",
                        "probe_schedule": [2, 1],
                        "max_adaptive_candidates": 2,
                        "visibility": "private",
                    }
                ],
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            source_config = ExperimentConfig.from_dict(data)
            source_store = RunStore.create(tmpdir, source_config)
            CouncilRunner(source_config, build_clients(source_config), source_store).run()

            transcript_path = str(source_store.transcript_path)
            replay_data = json.loads(json.dumps(data))
            replay_data["name"] = "adaptive_wave_replay_target"
            replay_phase = replay_data["protocol"]["phases"][0]
            replay_phase.update(
                {
                    "preauthored_probe_file": transcript_path,
                    "preauthored_answer_file": transcript_path,
                    "preauthored_evidence_file": transcript_path,
                    "preauthored_ranking_file": transcript_path,
                }
            )
            replay_config = ExperimentConfig.from_dict(replay_data)
            replay_store = RunStore.create(tmpdir, replay_config)
            CouncilRunner(replay_config, build_clients(replay_config), replay_store).run()
            replay_entries = load_jsonl(replay_store.transcript_path)
            replay_summary = json.loads((replay_store.run_dir / "run_summary.json").read_text())

        self.assertEqual(replay_summary["model_calls"], 0)
        self.assertTrue(
            all(
                entry["metadata"].get("preauthored_probe")
                for entry in replay_entries
                if entry["metadata"]["interaction_role"] == "question"
            )
        )
        self.assertTrue(
            all(
                entry["metadata"].get("preauthored_evidence")
                for entry in replay_entries
                if entry["metadata"]["interaction_role"] == "probe_comparison"
            )
        )
        self.assertTrue(
            all(
                entry["metadata"].get("preauthored_ranking")
                for entry in replay_entries
                if entry["metadata"]["interaction_role"] == "wave_judgment"
            )
        )

    def test_adaptive_probe_extension_sees_replayed_probes_and_guidance(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source_config = _parallel_judge_config(
                max_parallel_calls=4,
                probe_schedule=[1],
            )
            source_store = RunStore.create(tmpdir, source_config)
            CouncilRunner(
                source_config,
                build_clients(source_config),
                source_store,
            ).run()
            replay_config = _parallel_judge_config(
                max_parallel_calls=4,
                probe_schedule=[2],
                preauthored_probe_file=str(source_store.transcript_path),
                preauthored_answer_file=str(source_store.transcript_path),
                preauthored_evidence_file=str(source_store.transcript_path),
                probe_generation_guidance="Seek evidence above your own ceiling.",
            )
            replay_store = RunStore.create(tmpdir, replay_config)
            client = ConcurrencyTrackingMockClient(
                replay_config.providers["mock"]
            )
            CouncilRunner(
                replay_config,
                {"mock": client},
                replay_store,
            ).run()
            entries = load_jsonl(replay_store.transcript_path)

        questions = [
            entry
            for entry in entries
            if entry["metadata"].get("interaction_role") == "question"
        ]
        self.assertEqual(len(questions), 2)
        self.assertTrue(questions[0]["metadata"]["preauthored_probe"])
        self.assertFalse(questions[1]["metadata"].get("preauthored_probe", False))
        fresh_request = next(
            request
            for request in client.requests
            if request.metadata.get("interaction_role") == "question"
        )
        prompt = fresh_request.messages[-1]["content"]
        self.assertIn("Additional study guidance:", prompt)
        self.assertIn("Seek evidence above your own ceiling.", prompt)
        self.assertIn("Probes already chosen for this round", prompt)
        self.assertIn(questions[0]["content"], prompt)

    def test_adaptive_probe_omits_empty_study_guidance_block(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _parallel_judge_config(
                max_parallel_calls=2,
                probe_schedule=[1],
            )
            store = RunStore.create(tmpdir, config)
            client = ConcurrencyTrackingMockClient(config.providers["mock"])
            CouncilRunner(config, {"mock": client}, store).run()

        request = next(
            request
            for request in client.requests
            if request.metadata.get("interaction_role") == "question"
        )
        self.assertNotIn(
            "Additional study guidance",
            request.messages[-1]["content"],
        )

    def test_adaptive_probe_replay_survives_fresh_later_stage_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source_config = _parallel_judge_config(
                max_parallel_calls=2,
                probe_schedule=[2, 1],
            )
            source_store = RunStore.create(tmpdir, source_config)
            CouncilRunner(
                source_config,
                build_clients(source_config),
                source_store,
            ).run()
            source_entries = load_jsonl(source_store.transcript_path)
            round_one_outputs = Path(tmpdir) / "round_one_outputs.jsonl"
            round_one_outputs.write_text(
                "".join(
                    json.dumps(entry) + "\n"
                    for entry in source_entries
                    if entry["round_index"] == 1
                    and entry["metadata"].get("interaction_role")
                    in {"probe_comparison", "wave_judgment"}
                ),
                encoding="utf-8",
            )

            replay_config = _parallel_judge_config(
                max_parallel_calls=2,
                probe_schedule=[2, 1],
                preauthored_probe_file=str(source_store.transcript_path),
                preauthored_answer_file=str(source_store.transcript_path),
                preauthored_evidence_file=str(round_one_outputs),
                preauthored_ranking_file=str(round_one_outputs),
                replay_source_targets=True,
            )
            replay_store = RunStore.create(tmpdir, replay_config)
            CouncilRunner(
                replay_config,
                build_clients(replay_config),
                replay_store,
            ).run()
            replay_entries = load_jsonl(replay_store.transcript_path)
            replay_summary = json.loads(
                (replay_store.run_dir / "run_summary.json").read_text(encoding="utf-8")
            )

        self.assertEqual(replay_summary["model_calls"], 2)
        self.assertTrue(
            all(
                entry["metadata"].get("preauthored_probe")
                for entry in replay_entries
                if entry["metadata"].get("interaction_role") == "question"
            )
        )
        self.assertTrue(
            all(
                entry["metadata"].get("preauthored_answer")
                for entry in replay_entries
                if entry["metadata"].get("interaction_role") == "answer"
            )
        )
        round_two_roles = {
            entry["metadata"].get("interaction_role"):
            entry["metadata"].get("preauthored_evidence")
            or entry["metadata"].get("preauthored_ranking")
            for entry in replay_entries
            if entry["round_index"] == 2
            and entry["metadata"].get("interaction_role")
            in {"probe_comparison", "wave_judgment"}
        }
        self.assertEqual(
            round_two_roles,
            {"probe_comparison": None, "wave_judgment": None},
        )

    def test_adaptive_comparisons_can_replay_a_sparse_probe_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source_config = _parallel_judge_config(
                max_parallel_calls=2,
                probe_schedule=[2],
            )
            source_store = RunStore.create(tmpdir, source_config)
            CouncilRunner(
                source_config,
                build_clients(source_config),
                source_store,
            ).run()
            source_entries = load_jsonl(source_store.transcript_path)
            second_comparison = next(
                entry
                for entry in source_entries
                if entry["metadata"].get("interaction_role") == "probe_comparison"
                and entry["metadata"].get("probe_sequence_number") == 2
            )
            partial_evidence = Path(tmpdir) / "partial_evidence.jsonl"
            partial_evidence.write_text(
                json.dumps(second_comparison) + "\n",
                encoding="utf-8",
            )
            replay_config = _parallel_judge_config(
                max_parallel_calls=2,
                probe_schedule=[2],
                preauthored_probe_file=str(source_store.transcript_path),
                preauthored_answer_file=str(source_store.transcript_path),
                preauthored_evidence_file=str(partial_evidence),
            )
            replay_store = RunStore.create(tmpdir, replay_config)
            CouncilRunner(
                replay_config,
                build_clients(replay_config),
                replay_store,
            ).run()
            replay_entries = load_jsonl(replay_store.transcript_path)
            replay_summary = json.loads(
                (replay_store.run_dir / "run_summary.json").read_text(encoding="utf-8")
            )

        comparisons = [
            entry
            for entry in replay_entries
            if entry["metadata"].get("interaction_role") == "probe_comparison"
        ]
        comparison_by_probe = {
            entry["metadata"]["probe_sequence_number"]: entry
            for entry in comparisons
        }
        self.assertEqual(replay_summary["model_calls"], 2)
        self.assertFalse(
            comparison_by_probe[1]["metadata"].get("preauthored_evidence", False)
        )
        self.assertTrue(comparison_by_probe[2]["metadata"]["preauthored_evidence"])

    def test_fifty_candidate_comparison_is_complete_and_seeded(self) -> None:
        candidate_ids = [f"P{index:02d}" for index in range(1, 51)]
        config = ExperimentConfig.from_dict(
            {
                "name": "fifty_candidate_comparison",
                "run": {"max_parallel_calls": 16},
                "providers": [{"name": "mock", "kind": "mock"}],
                "models": [{"name": "mock_model", "provider": "mock", "model": "mock:model"}],
                "participants": [
                    {"id": candidate_id, "model": "mock_model", "system_prompt": "blind_evaluation_candidate"}
                    for candidate_id in candidate_ids
                ],
                "judges": [
                    {"id": "J1", "model": "mock_model", "system_prompt": "independent_intelligence_judge"}
                ],
                "protocol": {
                    "name": "global_comparison_stress",
                    "phases": [
                        {
                            "name": "judge_ranking",
                            "kind": "independent_judge_ranking",
                            "prompt": "adaptive_judge_probe",
                            "probe_schedule": [1],
                            "comparison_order": "seeded_shuffle",
                            "comparison_seed": 2718,
                            "visibility": "private",
                        }
                    ],
                },
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            store = RunStore.create(tmpdir, config)
            CouncilRunner(config, build_clients(config), store).run()
            analyze_run(store.run_dir)
            entries = load_jsonl(store.transcript_path)
            archive = json.loads((store.run_dir / "probe_answer_archive.json").read_text())

        comparison = next(
            entry
            for entry in entries
            if entry["metadata"].get("interaction_role") == "probe_comparison"
        )
        presented = comparison["metadata"]["answer_presentation_order"]
        self.assertEqual(set(presented), set(candidate_ids))
        self.assertNotEqual(presented, candidate_ids)
        self.assertEqual(set(comparison["parsed"]["candidate_summaries"]), set(candidate_ids))
        self.assertEqual(archive["probe_count"], 1)
        self.assertEqual(archive["answer_count"], 50)

    def test_seeded_comparison_order_is_repeatable_by_probe_number(self) -> None:
        candidates = [f"P{index}" for index in range(1, 9)]
        first = _comparison_presentation_order(candidates, "seeded_shuffle", 41, 1)
        self.assertEqual(
            first,
            _comparison_presentation_order(candidates, "seeded_shuffle", 41, 1),
        )
        self.assertNotEqual(
            first,
            _comparison_presentation_order(candidates, "seeded_shuffle", 41, 2),
        )
        self.assertEqual(candidates, [f"P{index}" for index in range(1, 9)])

    def test_round_robin_probes_use_one_probe_per_interviewer_and_round_outputs(self) -> None:
        config = ExperimentConfig.from_dict(
            {
                "name": "round_robin_probe_test",
                "context": {
                    "mode": "private_memory",
                    "max_public_turns": 0,
                    "max_private_turns": 12,
                    "max_stream_turns": 8,
                },
                "providers": [{"name": "mock", "kind": "mock"}],
                "models": [{"name": "mock_model", "provider": "mock", "model": "mock:model"}],
                "participants": [
                    {"id": "P1", "model": "mock_model"},
                    {"id": "P2", "model": "mock_model"},
                    {"id": "P3", "model": "mock_model"},
                ],
                "protocol": {
                    "name": "round_robin_probe_protocol",
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
                            "rounds": 1,
                            "visibility": "public",
                            "include_self": False,
                        }
                    ],
                },
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            store = RunStore.create(tmpdir, config)
            CouncilRunner(config, build_clients(config), store).run()
            entries = load_jsonl(store.transcript_path)
            findings = load_jsonl(store.findings_path)

            self.assertEqual(len(entries), 21)
            self.assertEqual(findings, [])
            self.assertEqual({entry["visibility"] for entry in entries}, {"private"})

            role_counts = {}
            question_turn_by_interviewer = {}
            for entry in entries:
                metadata = entry["metadata"]
                role = metadata["interaction_role"]
                role_counts[role] = role_counts.get(role, 0) + 1
                self.assertEqual(metadata["interaction_mode"], "round_robin_probes")
                if role == "question":
                    question_turn_by_interviewer[metadata["interviewer"]] = entry["turn_id"]

            self.assertEqual(
                role_counts,
                {
                    "question": 3,
                    "answer": 6,
                    "assessment": 6,
                    "round_ranking": 3,
                    "memory_update": 3,
                },
            )
            self.assertEqual(set(question_turn_by_interviewer), {"P1", "P2", "P3"})
            self.assertEqual(
                [entry["metadata"]["interaction_role"] for entry in entries[:3]],
                ["question", "question", "question"],
            )

            answers = [
                entry for entry in entries if entry["metadata"]["interaction_role"] == "answer"
            ]
            for answer in answers:
                interviewer = answer["metadata"]["interviewer"]
                self.assertEqual(
                    answer["metadata"]["question_turn_id"],
                    question_turn_by_interviewer[interviewer],
                )

            rankings = [
                entry for entry in entries if entry["metadata"]["interaction_role"] == "round_ranking"
            ]
            self.assertTrue(all("ranking" in entry["parsed"] for entry in rankings))

            memories = [
                entry for entry in entries if entry["metadata"]["interaction_role"] == "memory_update"
            ]
            self.assertTrue(
                all("qa_assessment_summaries" in entry["parsed"] for entry in memories)
            )

            summary = analyze_run(store.run_dir)
            self.assertEqual(summary["posthoc_extraction"]["qa_pair_count"], 6)
            self.assertEqual(summary["posthoc_extraction"]["probe_event_count"], 3)
            self.assertEqual(summary["metrics"]["evolution"]["probe_event_count"], 3)
            self.assertEqual(
                summary["posthoc_extraction"]["qa_pairs_by_mode"],
                {"round_robin_probes": 6},
            )
            self.assertTrue((store.run_dir / "posthoc_extraction.json").exists())
            self.assertTrue((store.run_dir / "run_metrics.json").exists())
            self.assertTrue((store.run_dir / "analysis_report.md").exists())

            revalidation = revalidate_run(store.run_dir)
            self.assertEqual(revalidation["codes"], {})

    def test_round_robin_ranking_context_contains_prior_memory_and_current_comparison_set(self) -> None:
        config = ExperimentConfig.from_dict(
            {
                "name": "round_robin_context_test",
                "context": {
                    "mode": "private_memory",
                    "max_public_turns": 0,
                    "max_private_turns": 20,
                    "max_stream_turns": 12,
                },
                "providers": [{"name": "capture", "kind": "mock"}],
                "models": [{"name": "capture_model", "provider": "capture", "model": "capture:model"}],
                "participants": [
                    {"id": "P1", "model": "capture_model"},
                    {"id": "P2", "model": "capture_model"},
                    {"id": "P3", "model": "capture_model"},
                ],
                "protocol": {
                    "name": "round_robin_context_protocol",
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
                            "rounds": 2,
                            "visibility": "public",
                            "include_self": False,
                        }
                    ],
                },
            }
        )
        client = CapturingClient()
        with tempfile.TemporaryDirectory() as tmpdir:
            store = RunStore.create(tmpdir, config)
            CouncilRunner(config, {"capture": client}, store).run()

        ranking_request = next(
            request
            for request in client.requests
            if request.metadata.get("interaction_role") == "round_ranking"
            and request.metadata.get("participant_id") == "P1"
            and request.metadata.get("round_index") == 2
        )
        user_prompt = ranking_request.messages[1]["content"]
        self.assertIn("memory marker round 1 for P1", user_prompt)
        self.assertIn("Respondent P2", user_prompt)
        self.assertIn("Respondent P3", user_prompt)
        self.assertIn("Your assessment turn", user_prompt)
        self.assertIn("Answer by P2 to P1 in round 2", user_prompt)
        self.assertIn("Answer by P3 to P1 in round 2", user_prompt)

    def test_round_robin_memory_update_uses_fallback_when_json_fails(self) -> None:
        config = ExperimentConfig.from_dict(
            {
                "name": "round_robin_memory_fallback_test",
                "context": {
                    "mode": "private_memory",
                    "max_public_turns": 0,
                    "max_private_turns": 12,
                    "max_stream_turns": 8,
                },
                "providers": [{"name": "capture", "kind": "mock"}],
                "models": [{"name": "capture_model", "provider": "capture", "model": "capture:model"}],
                "participants": [
                    {"id": "P1", "model": "capture_model"},
                    {"id": "P2", "model": "capture_model"},
                ],
                "protocol": {
                    "name": "round_robin_memory_fallback_protocol",
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
                            "rounds": 1,
                            "visibility": "public",
                            "include_self": False,
                        }
                    ],
                },
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            store = RunStore.create(tmpdir, config)
            CouncilRunner(config, {"capture": FailingMemoryClient()}, store).run()
            memories = [
                entry
                for entry in load_jsonl(store.transcript_path)
                if entry["metadata"]["interaction_role"] == "memory_update"
            ]
            self.assertEqual(len(memories), 2)
            self.assertTrue(all(entry["parsed"] for entry in memories))
            self.assertTrue(
                all(entry["metadata"].get("structured_json_fallback", {}).get("applied") for entry in memories)
            )
            self.assertEqual(load_jsonl(store.findings_path), [])

    def test_separate_interviews_create_streamed_question_answer_assessment_rows(self) -> None:
        config = ExperimentConfig.from_dict(
            {
                "name": "separate_interview_test",
                "context": {
                    "mode": "private_memory",
                    "max_public_turns": 0,
                    "max_private_turns": 8,
                    "max_stream_turns": 8,
                },
                "providers": [{"name": "mock", "kind": "mock"}],
                "models": [{"name": "mock_model", "provider": "mock", "model": "mock:model"}],
                "participants": [
                    {"id": "P1", "model": "mock_model"},
                    {"id": "P2", "model": "mock_model"},
                    {"id": "P3", "model": "mock_model"},
                ],
                "protocol": {
                    "name": "separate_interview_protocol",
                    "phases": [
                        {
                            "name": "interviews",
                            "kind": "separate_interviews",
                            "prompt": "separate_interview_question",
                            "question_prompt": "separate_interview_question",
                            "answer_prompt": "separate_interview_answer",
                            "assessment_prompt": "separate_interview_assessment",
                            "rounds": 1,
                            "visibility": "public",
                            "include_self": False,
                            "require_json": True,
                            "required_keys": [
                                "participant_id",
                                "phase",
                                "interviewer_id",
                                "respondent_id",
                                "target_participant_id",
                                "question_summary",
                                "answer_summary",
                                "assessment",
                                "current_ranking",
                                "confidence",
                                "criteria",
                                "evidence",
                                "uncertainties",
                                "updates",
                                "next_probe",
                            ],
                        }
                    ],
                },
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            store = RunStore.create(tmpdir, config)
            CouncilRunner(config, build_clients(config), store).run()
            entries = load_jsonl(store.transcript_path)
            findings = load_jsonl(store.findings_path)

            self.assertEqual(len(entries), 18)
            self.assertEqual(findings, [])
            self.assertEqual({entry["visibility"] for entry in entries}, {"private"})

            role_counts = {}
            stream_ids = set()
            for entry in entries:
                metadata = entry["metadata"]
                role = metadata["interaction_role"]
                role_counts[role] = role_counts.get(role, 0) + 1
                stream_ids.add(metadata["stream_id"])
                self.assertEqual(metadata["interaction_mode"], "separate_interviews")

            self.assertEqual(role_counts, {"question": 6, "answer": 6, "assessment": 6})
            self.assertEqual(len(stream_ids), 6)

            assessments = [entry for entry in entries if entry["metadata"]["interaction_role"] == "assessment"]
            self.assertTrue(all("current_ranking" in entry["parsed"] for entry in assessments))

            summary = analyze_run(store.run_dir)
            self.assertEqual(len(summary["interaction_streams"]), 6)
            self.assertEqual(summary["posthoc_extraction"]["qa_pair_count"], 6)
            self.assertTrue((store.run_dir / "posthoc_extraction.json").exists())
            self.assertTrue((store.run_dir / "analysis_report.md").exists())
            for stream in summary["interaction_streams"].values():
                self.assertEqual(stream["role_counts"], {"question": 1, "answer": 1, "assessment": 1})
                self.assertEqual(len(stream["assessments"]), 1)

            revalidation = revalidate_run(store.run_dir)
            self.assertEqual(revalidation["codes"], {})

    def test_interactive_discussion_marks_shared_stream(self) -> None:
        config = ExperimentConfig.from_dict(
            {
                "name": "interactive_discussion_test",
                "providers": [{"name": "mock", "kind": "mock"}],
                "models": [{"name": "mock_model", "provider": "mock", "model": "mock:model"}],
                "participants": [
                    {"id": "P1", "model": "mock_model"},
                    {"id": "P2", "model": "mock_model"},
                ],
                "protocol": {
                    "name": "interactive_protocol",
                    "phases": [
                        {
                            "name": "discussion",
                            "kind": "interactive_discussion",
                            "prompt": "interactive_discussion_turn",
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
            entries = load_jsonl(store.transcript_path)

            self.assertEqual(len(entries), 4)
            self.assertEqual({entry["metadata"]["stream_id"] for entry in entries}, {"discussion"})
            self.assertEqual({entry["metadata"]["interaction_role"] for entry in entries}, {"discussion"})
            summary = analyze_run(store.run_dir)
            self.assertEqual(summary["metrics"]["evolution"]["probe_event_count"], 4)
            self.assertEqual(
                summary["metrics"]["evolution"]["transition_counts"]["opening_probe"],
                2,
            )

class CapturingClient(ModelClient):
    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        metadata = request.metadata
        participant_id = str(metadata.get("participant_id", "P1"))
        phase = str(metadata.get("phase", "phase"))
        role = metadata.get("interaction_role")
        round_index = metadata.get("round_index")
        participants = list(metadata.get("participants", []))
        if metadata.get("require_json"):
            content = self._json_content(participant_id, phase, role, round_index, participants, metadata)
        elif role == "question":
            content = f"Probe by {participant_id} in round {round_index}: compare mechanisms and edge cases."
        elif role == "answer":
            content = (
                f"Answer by {participant_id} to {metadata.get('interviewer_id')} in round {round_index}: "
                "mechanism, edge case, counterexample, synthesis, calibration."
            )
        else:
            content = f"Discussion by {participant_id} in {phase}."
        return ModelResponse(content=content, raw={"mock": True}, usage={}, model=request.model, provider="capture")

    def _json_content(
        self,
        participant_id: str,
        phase: str,
        role: object,
        round_index: object,
        participants: list[str],
        metadata: dict[str, object],
    ) -> str:
        if role == "assessment":
            return json.dumps(
                {
                    "participant_id": participant_id,
                    "phase": phase,
                    "round_index": round_index,
                    "interviewer_id": metadata.get("interviewer_id"),
                    "respondent_id": metadata.get("respondent_id"),
                    "question_summary": "Asked for mechanisms and edge cases.",
                    "answer_summary": "Answered with mechanism and calibration.",
                    "assessment": "Sufficient for context testing.",
                    "confidence": 0.5,
                    "criteria": ["reasoning"],
                    "evidence": ["context fixture"],
                    "uncertainties": ["mock client"],
                    "updates": ["none"],
                    "next_probe": "Continue.",
                }
            )
        if role == "round_ranking":
            return json.dumps(
                {
                    "participant_id": participant_id,
                    "phase": phase,
                    "round_index": round_index,
                    "interviewer_id": metadata.get("interviewer_id"),
                    "ranking": participants,
                    "confidence": 0.5,
                    "criteria": ["comparison"],
                    "evidence": ["current round answers"],
                    "uncertainties": ["mock client"],
                    "updates": ["none"],
                    "next_probe_strategy": ["Continue."],
                }
            )
        if role == "memory_update":
            return json.dumps(
                {
                    "participant_id": participant_id,
                    "phase": phase,
                    "round_index": round_index,
                    "interviewer_id": metadata.get("interviewer_id"),
                    "qa_assessment_summaries": [
                        {
                            "respondent_id": "P2",
                            "question_summary": "Mechanism probe.",
                            "answer_summary": "Mechanism answer.",
                            "assessment_summary": "Mock assessment.",
                            "evidence_to_remember": ["mock evidence"],
                        }
                    ],
                    "ranking_summary": f"memory marker round {round_index} for {participant_id}",
                    "uncertainties": ["mock uncertainty"],
                    "next_round_plan": "Continue.",
                }
            )
        return json.dumps(
            {
                "participant_id": participant_id,
                "phase": phase,
                "ranking": participants,
                "confidence": 0.5,
                "criteria": ["mock"],
                "evidence": ["mock"],
                "uncertainties": ["mock"],
                "updates": ["mock"],
                "next_evidence_needed": ["mock"],
            }
        )


class FailingMemoryClient(CapturingClient):
    def generate(self, request: ModelRequest) -> ModelResponse:
        if request.metadata.get("interaction_role") == "memory_update":
            self.requests.append(request)
            return ModelResponse(
                content='{"participant_id": "P1", "phase": "probe_rounds", ',
                raw={"mock": True, "choices": [{"finish_reason": "length", "message": {"content": ""}}]},
                usage={"completion_tokens": 10},
                model=request.model,
                provider="capture",
            )
        return super().generate(request)


class ConcurrencyTrackingMockClient(MockModelClient):
    supports_parallel_requests = True

    def __init__(self, provider) -> None:
        super().__init__(provider)
        self._lock = threading.Lock()
        self.active_calls = 0
        self.peak_calls = 0
        self.requests: list[ModelRequest] = []
        self._mock_state_lock = threading.Lock()

    def _start_request(self, request: ModelRequest) -> None:
        with self._lock:
            self.active_calls += 1
            self.peak_calls = max(self.peak_calls, self.active_calls)
            self.requests.append(request)

    def _finish_request(self) -> None:
        with self._lock:
            self.active_calls -= 1

    def generate(self, request: ModelRequest) -> ModelResponse:
        self._start_request(request)
        try:
            time.sleep(0.02)
            with self._mock_state_lock:
                return super().generate(request)
        finally:
            self._finish_request()


class SerializedTrackingMockClient(ConcurrencyTrackingMockClient):
    supports_parallel_requests = False


class RecoveryOverrideRejectingMockClient(ConcurrencyTrackingMockClient):
    def generate(self, request: ModelRequest) -> ModelResponse:
        if request.params.get("reasoning") == {"effort": "none"}:
            raise ModelClientError("HTTP 400: reasoning is mandatory")
        return super().generate(request)


class CostTrackingMockClient(ConcurrencyTrackingMockClient):
    def __init__(self, provider, *, cost_per_call: float) -> None:
        super().__init__(provider)
        self.cost_per_call = cost_per_call

    def generate(self, request: ModelRequest) -> ModelResponse:
        response = super().generate(request)
        return ModelResponse(
            content=response.content,
            raw=response.raw,
            usage={"cost": self.cost_per_call},
            model=response.model,
            provider=response.provider,
        )


class FailingParallelMockClient(ConcurrencyTrackingMockClient):
    def generate(self, request: ModelRequest) -> ModelResponse:
        self._start_request(request)
        try:
            is_first_answer = (
                request.metadata.get("interaction_role") == "answer"
                and request.metadata.get("respondent_id") == "P1"
            )
            time.sleep(0.005 if is_first_answer else 0.03)
            if is_first_answer:
                raise ModelClientError("intentional parallel provider failure")
            with self._mock_state_lock:
                return MockModelClient.generate(self, request)
        finally:
            self._finish_request()


class IncompleteParallelMockClient(ConcurrencyTrackingMockClient):
    def generate(self, request: ModelRequest) -> ModelResponse:
        if (
            request.metadata.get("interaction_role") == "answer"
            and request.metadata.get("respondent_id") == "P1"
        ):
            self._start_request(request)
            try:
                return ModelResponse(
                    content="",
                    raw={"choices": [{"finish_reason": "length"}]},
                    usage={},
                    model=request.model,
                    provider="mock",
                )
            finally:
                self._finish_request()
        return super().generate(request)


class CheckpointBlockingMockClient(ConcurrencyTrackingMockClient):
    def __init__(self, provider) -> None:
        super().__init__(provider)
        self.first_answer_completed = threading.Event()
        self.release_answers = threading.Event()

    def generate(self, request: ModelRequest) -> ModelResponse:
        self._start_request(request)
        try:
            if request.metadata.get("interaction_role") == "answer":
                if request.metadata.get("respondent_id") == "P1":
                    with self._mock_state_lock:
                        response = MockModelClient.generate(self, request)
                    self.first_answer_completed.set()
                    return response
                self.release_answers.wait(2)
            with self._mock_state_lock:
                return MockModelClient.generate(self, request)
        finally:
            self._finish_request()


class DeterministicRecoveryMockClient(ConcurrencyTrackingMockClient):
    def __init__(self, provider) -> None:
        super().__init__(provider)
        self._failed_target = False

    def generate(self, request: ModelRequest) -> ModelResponse:
        self._start_request(request)
        try:
            time.sleep(0.01)
            is_target = (
                request.metadata.get("interaction_role") == "answer"
                and request.metadata.get("respondent_id") == "P1"
                and str(request.metadata.get("probe_id", "")).endswith("probe_1")
            )
            with self._mock_state_lock:
                if is_target and not self._failed_target:
                    self._failed_target = True
                    return ModelResponse(
                        content="",
                        raw={"mock": True},
                        usage={},
                        model=request.model,
                        provider="mock",
                    )
                return MockModelClient.generate(self, request)
        finally:
            self._finish_request()


class BlockingSerialClient(ModelClient):
    def __init__(self, parallel_started: threading.Event) -> None:
        self.parallel_started = parallel_started
        self.parallel_started_before_release = False
        self.active_calls = 0
        self.peak_calls = 0
        self._lock = threading.Lock()

    def generate(self, request: ModelRequest) -> ModelResponse:
        with self._lock:
            self.active_calls += 1
            self.peak_calls = max(self.peak_calls, self.active_calls)
        try:
            if request.model == "serial-1":
                self.parallel_started_before_release = self.parallel_started.wait(0.5)
            time.sleep(0.005)
            return ModelResponse(
                content="ok",
                raw={},
                usage={},
                model=request.model,
                provider="serial",
            )
        finally:
            with self._lock:
                self.active_calls -= 1


class SignalingParallelClient(ModelClient):
    supports_parallel_requests = True

    def __init__(self, parallel_started: threading.Event) -> None:
        self.parallel_started = parallel_started

    def generate(self, request: ModelRequest) -> ModelResponse:
        self.parallel_started.set()
        return ModelResponse(
            content="ok",
            raw={},
            usage={},
            model=request.model,
            provider="parallel",
        )


def _scheduler_request(model: str) -> ModelRequest:
    return ModelRequest(
        model=model,
        messages=[{"role": "user", "content": "test"}],
    )


def _shift_turn_ids(entry: dict[str, object], *, offset: int) -> dict[str, object]:
    shifted = json.loads(json.dumps(entry))
    if isinstance(shifted.get("turn_id"), int):
        shifted["turn_id"] += offset
    metadata = shifted.get("metadata")
    if not isinstance(metadata, dict):
        return shifted
    for field, value in list(metadata.items()):
        if field.endswith("_turn_id") and isinstance(value, int):
            metadata[field] = value + offset
        elif field.endswith("_turn_ids") and isinstance(value, list):
            metadata[field] = [
                item + offset if isinstance(item, int) else item for item in value
            ]
    return shifted


def _run_parallel_judge_fixture(
    *,
    max_parallel_calls: int,
    client_class: type[ConcurrencyTrackingMockClient] = ConcurrencyTrackingMockClient,
) -> tuple[list[dict[str, object]], int, list[ModelRequest]]:
    config = _parallel_judge_config(max_parallel_calls=max_parallel_calls)
    with tempfile.TemporaryDirectory() as tmpdir:
        store = RunStore.create(tmpdir, config)
        client = client_class(config.providers["mock"])
        CouncilRunner(config, {"mock": client}, store).run()
        entries = load_jsonl(store.transcript_path)
    return entries, client.peak_calls, client.requests


def _parallel_judge_config(
    *,
    max_parallel_calls: int,
    probes_per_round: int = 2,
    max_reported_cost_usd: float | None = None,
    max_model_calls: int | None = None,
    visible_text_retries: int | None = None,
    continue_batch_on_call_error: bool = False,
    incomplete_answer_policy: str = "fail",
    probe_schedule: list[int] | None = None,
    preauthored_probe_file: str | None = None,
    preauthored_answer_file: str | None = None,
    preauthored_evidence_file: str | None = None,
    preauthored_ranking_file: str | None = None,
    replay_source_targets: bool = False,
    probe_generation_guidance: str = "",
) -> ExperimentConfig:
    run = {
        "max_parallel_calls": max_parallel_calls,
        "continue_batch_on_call_error": continue_batch_on_call_error,
    }
    if max_reported_cost_usd is not None:
        run["max_reported_cost_usd"] = max_reported_cost_usd
    if max_model_calls is not None:
        run["max_model_calls"] = max_model_calls
    if visible_text_retries is not None:
        run["visible_text_retries"] = visible_text_retries
    return ExperimentConfig.from_dict(
        {
            "name": f"parallel_judge_{max_parallel_calls}",
            "run": run,
            "providers": [{"name": "mock", "kind": "mock"}],
            "models": [{"name": "mock_model", "provider": "mock", "model": "mock:model"}],
            "participants": [
                {
                    "id": candidate_id,
                    "model": "mock_model",
                    "system_prompt": "blind_evaluation_candidate",
                }
                for candidate_id in ["P1", "P2", "P3"]
            ],
            "judges": [
                {
                    "id": "J1",
                    "model": "mock_model",
                    "system_prompt": "independent_intelligence_judge",
                }
            ],
            "protocol": {
                "name": "parallel_judge_protocol",
                "phases": [
                    {
                        "name": "judge_ranking",
                        "kind": "independent_judge_ranking",
                        "prompt": "independent_judge_probe",
                        "rounds": len(probe_schedule) if probe_schedule else 1,
                        **(
                            {"probes_per_round": probes_per_round}
                            if probe_schedule is None
                            else {}
                        ),
                        "incomplete_answer_policy": incomplete_answer_policy,
                        **(
                            {
                                "prompt": "adaptive_judge_probe",
                                "probe_schedule": probe_schedule,
                            }
                            if probe_schedule is not None
                            else {}
                        ),
                        **(
                            {"preauthored_probe_file": preauthored_probe_file}
                            if preauthored_probe_file is not None
                            else {}
                        ),
                        **(
                            {"preauthored_answer_file": preauthored_answer_file}
                            if preauthored_answer_file is not None
                            else {}
                        ),
                        **(
                            {"preauthored_evidence_file": preauthored_evidence_file}
                            if preauthored_evidence_file is not None
                            else {}
                        ),
                        **(
                            {"preauthored_ranking_file": preauthored_ranking_file}
                            if preauthored_ranking_file is not None
                            else {}
                        ),
                        **(
                            {"replay_source_targets": True}
                            if replay_source_targets
                            else {}
                        ),
                        **(
                            {
                                "probe_generation_guidance": (
                                    probe_generation_guidance
                                )
                            }
                            if probe_generation_guidance
                            else {}
                        ),
                        "visibility": "private",
                    }
                ],
            },
        }
    )


def _stable_entry_fields(entry: dict[str, object]) -> str:
    stable = dict(entry)
    stable.pop("created_at", None)
    return json.dumps(stable, sort_keys=True)


def _stable_request_fields(request: ModelRequest) -> tuple[str, ...]:
    metadata = request.metadata
    return (
        str(metadata.get("interaction_role")),
        str(metadata.get("participant_id")),
        str(metadata.get("round_index")),
        str(metadata.get("probe_id")),
        str(metadata.get("candidate_id")),
        str(metadata.get("respondent_id")),
        json.dumps(request.messages, sort_keys=True),
        json.dumps(request.params, sort_keys=True),
        json.dumps(request.metadata, sort_keys=True),
    )


if __name__ == "__main__":
    unittest.main()

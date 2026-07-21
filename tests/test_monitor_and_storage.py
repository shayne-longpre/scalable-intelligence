from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from ai_council.clients.base import ModelClient
from ai_council.config import ExperimentConfig
from ai_council.core import ModelRequest, ModelResponse, TranscriptEntry
from ai_council.monitors import RuleBasedMonitor
from ai_council.orchestrator import CouncilRunner, ExperimentViolationError
from ai_council.storage import RunStore, _unique_run_dir, load_jsonl


class StaticClient(ModelClient):
    def __init__(
        self,
        content: str,
        usage: dict | None = None,
        raw: dict | None = None,
    ):
        self.content = content
        self.usage = usage or {}
        self.raw = raw or {}
        self.requests: list[ModelRequest] = []

    def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return ModelResponse(
            content=self.content,
            raw=self.raw,
            usage=self.usage,
            model=request.model,
            provider="test",
        )


class MonitorAndStorageTests(unittest.TestCase):
    def test_strict_monitor_stops_on_identity_leak(self) -> None:
        config = _minimal_config(
            phase={
                "name": "public_probe",
                "kind": "public_round_robin",
                "prompt": "opening_council",
                "rounds": 1,
                "visibility": "public",
            },
            monitor={"enabled": True, "strict": True},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            store = RunStore.create(tmpdir, config)
            runner = CouncilRunner(config, {"test": StaticClient("I am OpenAI model X.")}, store)
            with self.assertRaises(ExperimentViolationError):
                runner.run()

            findings = load_jsonl(store.findings_path)
            self.assertEqual(findings[0]["code"], "possible_identity_leak")

    def test_monitor_does_not_treat_task_training_data_as_identity_leak(self) -> None:
        monitor = RuleBasedMonitor()
        entry = TranscriptEntry(
            turn_id=1,
            phase="probe",
            speaker="P1",
            visibility="private",
            content="The stock algorithm was tested on training data from bull markets.",
        )
        self.assertEqual(monitor.check_entry(entry), [])

    def test_monitor_flags_first_person_training_disclosure(self) -> None:
        monitor = RuleBasedMonitor()
        entry = TranscriptEntry(
            turn_id=1,
            phase="probe",
            speaker="P1",
            visibility="private",
            content="My training data includes material through 2025.",
        )
        findings = monitor.check_entry(entry)
        self.assertEqual([finding.code for finding in findings], ["possible_identity_leak"])

    def test_required_json_parse_failure_is_recorded_without_required_keys(self) -> None:
        config = _minimal_config(
            phase={
                "name": "json_checkpoint",
                "kind": "private_judgment",
                "prompt": "checkpoint_ranking",
                "visibility": "private",
                "require_json": True,
                "required_keys": [],
            },
            monitor={"enabled": True, "strict": False},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            store = RunStore.create(tmpdir, config)
            CouncilRunner(config, {"test": StaticClient("not json")}, store).run()
            findings = load_jsonl(store.findings_path)
            self.assertEqual(findings[0]["code"], "missing_structured_json")

    def test_empty_visible_output_after_reasoning_is_recorded(self) -> None:
        config = _minimal_config(
            phase={
                "name": "public_probe",
                "kind": "public_round_robin",
                "prompt": "opening_council",
                "rounds": 1,
                "visibility": "public",
            },
            monitor={"enabled": True, "strict": False},
        )
        usage = {
            "completion_tokens": 750,
            "completion_tokens_details": {"reasoning_tokens": 749},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            store = RunStore.create(tmpdir, config)
            CouncilRunner(config, {"test": StaticClient("", usage=usage)}, store).run()
            findings = load_jsonl(store.findings_path)
            self.assertEqual(findings[0]["code"], "empty_visible_output_after_reasoning")
            self.assertIn("reasoning_tokens=749", findings[0]["evidence"])

    def test_reasoning_dominated_structured_parse_failure_is_recorded(self) -> None:
        config = _minimal_config(
            phase={
                "name": "json_checkpoint",
                "kind": "private_judgment",
                "prompt": "checkpoint_ranking",
                "visibility": "private",
                "require_json": True,
                "required_keys": [],
            },
            monitor={"enabled": True, "strict": False},
        )
        usage = {
            "completion_tokens": 750,
            "completion_tokens_details": {"reasoning_tokens": 615},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            store = RunStore.create(tmpdir, config)
            CouncilRunner(
                config,
                {"test": StaticClient('{"participant_id": "P1", "phase": ', usage=usage)},
                store,
            ).run()
            findings = load_jsonl(store.findings_path)
            self.assertEqual(
                [finding["code"] for finding in findings],
                [
                    "reasoning_dominated_structured_output_failure",
                    "missing_structured_json",
                ],
            )

    def test_structured_parse_failure_at_max_tokens_is_recorded(self) -> None:
        config = ExperimentConfig.from_dict(
            {
                "name": "truncation_test",
                "providers": [{"name": "test", "kind": "mock"}],
                "models": [
                    {
                        "name": "model",
                        "provider": "test",
                        "model": "test:model",
                        "params": {"max_tokens": 5},
                    }
                ],
                "participants": [{"id": "P1", "model": "model"}],
                "monitor": {"enabled": True, "strict": False},
                "protocol": {
                    "name": "test_protocol",
                    "phases": [
                        {
                            "name": "json_checkpoint",
                            "kind": "private_judgment",
                            "prompt": "checkpoint_ranking",
                            "visibility": "private",
                            "require_json": True,
                        }
                    ],
                },
            }
        )
        usage = {"completion_tokens": 5}
        with tempfile.TemporaryDirectory() as tmpdir:
            store = RunStore.create(tmpdir, config)
            CouncilRunner(config, {"test": StaticClient('{"ranking": [', usage=usage)}, store).run()
            findings = load_jsonl(store.findings_path)
            self.assertEqual(
                [finding["code"] for finding in findings],
                ["structured_output_may_be_truncated", "missing_structured_json"],
            )

    def test_provider_finish_reason_is_recorded_without_full_raw_response(self) -> None:
        config = _minimal_config(
            phase={
                "name": "public_probe",
                "kind": "public_round_robin",
                "prompt": "opening_council",
                "rounds": 1,
                "visibility": "public",
            },
            monitor={"enabled": False},
        )
        raw = {
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {"content": "ok", "reasoning": "hidden"},
                }
            ]
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            store = RunStore.create(tmpdir, config)
            CouncilRunner(config, {"test": StaticClient("ok", raw=raw)}, store).run()
            entries = load_jsonl(store.transcript_path)
            metadata = entries[0]["metadata"]
            self.assertEqual(metadata["finish_reason"], "length")
            self.assertEqual(metadata["response_message_keys"], ["content", "reasoning"])
            self.assertEqual(metadata["request_params"], {})

    def test_phase_model_params_override_model_defaults(self) -> None:
        config = ExperimentConfig.from_dict(
            {
                "name": "phase_params_test",
                "providers": [{"name": "test", "kind": "mock"}],
                "models": [
                    {
                        "name": "model",
                        "provider": "test",
                        "model": "test:model",
                        "params": {"temperature": 0.7, "max_tokens": 500},
                    }
                ],
                "participants": [{"id": "P1", "model": "model"}],
                "monitor": {"enabled": False},
                "protocol": {
                    "name": "test_protocol",
                    "phases": [
                        {
                            "name": "public_probe",
                            "kind": "public_round_robin",
                            "prompt": "opening_council",
                            "rounds": 1,
                            "visibility": "public",
                            "model_params": {
                                "max_tokens": 1000,
                                "reasoning": {"effort": "none"},
                            },
                        }
                    ],
                },
            }
        )
        client = StaticClient("ok")
        with tempfile.TemporaryDirectory() as tmpdir:
            store = RunStore.create(tmpdir, config)
            CouncilRunner(config, {"test": client}, store).run()
        self.assertEqual(
            client.requests[0].params,
            {"temperature": 0.7, "max_tokens": 1000, "reasoning": {"effort": "none"}},
        )

    def test_prompts_render_actual_participant_roster_for_json_rankings(self) -> None:
        config = ExperimentConfig.from_dict(
            {
                "name": "participant_roster_prompt_test",
                "providers": [{"name": "test", "kind": "mock"}],
                "models": [{"name": "model", "provider": "test", "model": "test:model"}],
                "participants": [
                    {"id": "P1", "model": "model"},
                    {"id": "P2", "model": "model"},
                ],
                "monitor": {"enabled": False},
                "protocol": {
                    "name": "test_protocol",
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
        client = StaticClient("{}")
        with tempfile.TemporaryDirectory() as tmpdir:
            store = RunStore.create(tmpdir, config)
            CouncilRunner(config, {"test": client}, store).run()
        prompt = client.requests[0].messages[1]["content"]
        self.assertIn('["P1", "P2"]', prompt)
        self.assertNotIn('["P1", "P2", "P3"]', prompt)

    def test_unique_run_dir_uses_suffix_on_collision(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            (base / "run").mkdir()
            self.assertEqual(_unique_run_dir(base, "run").name, "run_2")
            (base / "run_2").mkdir()
            self.assertEqual(_unique_run_dir(base, "run").name, "run_3")

    def test_run_store_snapshots_effective_prompts_with_hash(self) -> None:
        config = _minimal_config(
            phase={
                "name": "public_probe",
                "kind": "public_round_robin",
                "prompt": "opening_council",
                "rounds": 1,
                "visibility": "public",
            },
            monitor={"enabled": False},
        )
        config.prompt_overrides["opening_council"] = "Exact test prompt."

        with tempfile.TemporaryDirectory() as tmpdir:
            store = RunStore.create(tmpdir, config)
            snapshot = json.loads(
                (store.run_dir / "prompt_snapshot.json").read_text(encoding="utf-8")
            )

        self.assertEqual(snapshot["prompts"]["opening_council"], "Exact test prompt.")
        canonical = json.dumps(
            snapshot["prompts"],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        self.assertEqual(snapshot["sha256"], hashlib.sha256(canonical).hexdigest())

    def test_structured_value_checks_reject_bad_ranking_and_confidence(self) -> None:
        monitor = RuleBasedMonitor()
        config = _minimal_config(
            phase={
                "name": "json_checkpoint",
                "kind": "private_judgment",
                "prompt": "checkpoint_ranking",
                "visibility": "private",
                "require_json": True,
            },
            monitor={"enabled": True, "strict": False},
        )
        entry = config.protocol.phases[0]
        transcript_entry = TranscriptEntry(
            turn_id=1,
            phase=entry.name,
            speaker="P1",
            visibility="private",
            content="{}",
        )
        findings = monitor.check_structured_values(
            transcript_entry,
            {"ranking": [{"id": "P1"}], "confidence": "high"},
        )
        self.assertEqual([finding.code for finding in findings], ["invalid_ranking_shape", "invalid_confidence"])

    def test_structured_value_checks_reject_scalar_list_fields(self) -> None:
        monitor = RuleBasedMonitor()
        transcript_entry = TranscriptEntry(
            turn_id=12,
            phase="final_judgment",
            speaker="P1",
            visibility="private",
            content="{}",
        )
        findings = monitor.check_structured_values(
            transcript_entry,
            {
                "ranking": ["P1", "P2"],
                "confidence": 0.7,
                "criteria": "reasoning depth",
                "evidence": [{"claim": "object evidence"}],
                "uncertainties": ["limited sample"],
            },
            participant_ids=["P1", "P2"],
        )
        self.assertEqual(
            [finding.code for finding in findings],
            ["invalid_criteria_shape", "invalid_evidence_shape"],
        )

    def test_structured_value_checks_use_participant_roster(self) -> None:
        monitor = RuleBasedMonitor()
        transcript_entry = TranscriptEntry(
            turn_id=7,
            phase="checkpoint",
            speaker="P1",
            visibility="private",
            content="{}",
        )
        findings = monitor.check_structured_values(
            transcript_entry,
            {
                "participant_id": "P2",
                "phase": "other_phase",
                "ranking": ["P2", "P2", "PX"],
                "confidence": 0.5,
            },
            participant_ids=["P1", "P2", "P3"],
        )
        self.assertEqual(
            [finding.code for finding in findings],
            [
                "participant_id_mismatch",
                "phase_mismatch",
                "duplicate_ranking_ids",
                "unknown_ranking_ids",
                "missing_ranking_ids",
            ],
        )

    def test_structured_value_checks_accept_participant_rank_maps(self) -> None:
        monitor = RuleBasedMonitor()
        transcript_entry = TranscriptEntry(
            turn_id=8,
            phase="memory_update",
            speaker="P1",
            visibility="private",
            content="{}",
        )
        findings = monitor.check_structured_values(
            transcript_entry,
            {"current_ranking": {"P2": 2, "P1": 1, "P3": 3}, "confidence": 0.7},
            participant_ids=["P1", "P2", "P3"],
        )
        self.assertEqual(findings, [])

    def test_structured_value_checks_reject_ranking_maps_without_participant_ranks(self) -> None:
        monitor = RuleBasedMonitor()
        transcript_entry = TranscriptEntry(
            turn_id=9,
            phase="memory_update",
            speaker="P1",
            visibility="private",
            content="{}",
        )
        findings = monitor.check_structured_values(
            transcript_entry,
            {"current_ranking": {"evaluation_criteria": ["reasoning"]}, "confidence": 0.7},
            participant_ids=["P1", "P2", "P3"],
        )
        self.assertEqual([finding.code for finding in findings], ["invalid_current_ranking_shape"])

    def test_structured_value_checks_reject_duplicate_rank_map_positions(self) -> None:
        monitor = RuleBasedMonitor()
        transcript_entry = TranscriptEntry(
            turn_id=10,
            phase="memory_update",
            speaker="P1",
            visibility="private",
            content="{}",
        )
        findings = monitor.check_structured_values(
            transcript_entry,
            {"current_ranking": {"P1": 1, "P2": 1, "P3": 3}, "confidence": 0.7},
            participant_ids=["P1", "P2", "P3"],
        )
        self.assertEqual([finding.code for finding in findings], ["duplicate_current_ranking_positions"])

    def test_structured_value_checks_use_interview_metadata(self) -> None:
        monitor = RuleBasedMonitor()
        transcript_entry = TranscriptEntry(
            turn_id=11,
            phase="interviews",
            speaker="P3",
            visibility="private",
            content="{}",
            metadata={"interviewer": "P3", "respondent": "P1"},
        )
        findings = monitor.check_structured_values(
            transcript_entry,
            {
                "participant_id": "P3",
                "phase": "interviews",
                "interviewer_id": "P1",
                "respondent_id": "P1",
                "target_participant_id": "P2",
                "current_ranking": ["P1", "P2", "P3"],
                "confidence": 0.7,
            },
            participant_ids=["P1", "P2", "P3"],
        )
        self.assertEqual(
            [finding.code for finding in findings],
            ["interviewer_id_mismatch", "target_participant_id_mismatch"],
        )


def _minimal_config(phase: dict, monitor: dict) -> ExperimentConfig:
    return ExperimentConfig.from_dict(
        {
            "name": "test",
            "providers": [{"name": "test", "kind": "mock"}],
            "models": [{"name": "model", "provider": "test", "model": "test:model"}],
            "participants": [{"id": "P1", "model": "model"}],
            "monitor": monitor,
            "protocol": {"name": "test_protocol", "phases": [phase]},
        }
    )


if __name__ == "__main__":
    unittest.main()

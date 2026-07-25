from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from ai_council.clients.base import ModelClient
from ai_council.config import ConfigError
from ai_council.core import ModelRequest, ModelResponse
from ai_council.probe_scoring import (
    ProbeScoringConfig,
    ProbeEvidence,
    ProbeSource,
    ScorePayloadError,
    ScoringJudge,
    _load_journal,
    _repair_prompt,
    _score_one,
    build_scoring_summary,
    load_probe_evidence,
    validate_score_payload,
    write_export_summary,
)


class ProbeScoringTests(unittest.TestCase):
    def test_load_probe_evidence_links_answers_by_question_turn(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            entries = [
                {"turn_id": 1, "speaker": "J1", "content": "Solve it.", "metadata": {}},
                {
                    "turn_id": 2,
                    "speaker": "P1",
                    "content": "Answer one.",
                    "metadata": {
                        "interaction_role": "answer",
                        "question_turn_id": 1,
                    },
                },
                {
                    "turn_id": 3,
                    "speaker": "P2",
                    "content": "",
                    "metadata": {
                        "interaction_role": "answer",
                        "question_turn_id": 1,
                    },
                },
                {
                    "turn_id": 4,
                    "speaker": "P3",
                    "content": "Wrong stream.",
                    "metadata": {
                        "interaction_role": "answer",
                        "question_turn_id": 99,
                    },
                },
            ]
            (run_dir / "transcript.jsonl").write_text(
                "\n".join(json.dumps(item) for item in entries)
            )
            source = ProbeSource("fixture", run_dir, (1,))

            evidence = load_probe_evidence((source,))

            self.assertEqual(len(evidence), 1)
            self.assertEqual(evidence[0].answers, {"P1": "Answer one."})
            self.assertEqual(evidence[0].unavailable_candidates, ("P2",))

    def test_validate_score_payload_requires_exact_candidate_set(self) -> None:
        payload = {
            "probe_rubric": ["Correct result"],
            "scores": {
                "P1": {
                    "score": 4,
                    "confidence": "high",
                    "summary": "Correct.",
                    "error_tags": [],
                }
            },
        }
        self.assertEqual(
            validate_score_payload(payload, {"P1"})["scores"]["P1"]["score"], 4
        )
        with self.assertRaisesRegex(ValueError, "candidate mismatch"):
            validate_score_payload(payload, {"P1", "P2"})

    def test_validate_score_payload_rejects_fractional_scores(self) -> None:
        payload = {
            "probe_rubric": [],
            "scores": {
                "P1": {
                    "score": 3.5,
                    "confidence": "medium",
                    "summary": "Mostly correct.",
                    "error_tags": ["minor gap"],
                }
            },
        }
        with self.assertRaisesRegex(ValueError, "integer from 0 to 4"):
            validate_score_payload(payload, {"P1"})

    def test_summary_averages_judges_and_records_disagreement(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            (run_dir / "transcript.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "turn_id": 1,
                                "speaker": "J1",
                                "content": "Question",
                                "metadata": {},
                            }
                        ),
                        json.dumps(
                            {
                                "turn_id": 2,
                                "speaker": "P1",
                                "content": "Answer",
                                "metadata": {
                                    "interaction_role": "answer",
                                    "question_turn_id": 1,
                                },
                            }
                        ),
                    ]
                )
            )
            evidence = load_probe_evidence(
                (ProbeSource("fixture", run_dir, (1,)),)
            )
            config = ProbeScoringConfig(
                name="fixture",
                providers={},
                judges=(),
                sources=(),
                output_dir=run_dir,
            )
            result_template = {
                "status": "ok",
                "probe_id": "fixture:turn_1",
                "judge_model": "provider/judge",
                "probe_rubric": ["Correct result"],
                "answer_order": ["P1"],
                "attempts": [],
            }
            results = [
                {
                    **result_template,
                    "judge_id": "a",
                    "scores": {"P1": {"score": 4}},
                },
                {
                    **result_template,
                    "judge_id": "b",
                    "scores": {"P1": {"score": 2}},
                },
            ]

            summary = build_scoring_summary(config, evidence, results)

            probe = summary["probes"][0]
            self.assertEqual(probe["mean_scores"]["P1"], 3.0)
            self.assertEqual(probe["mean_judge_score_range"], 2.0)
            self.assertEqual(probe["substantially_correct_rate"], 1.0)
            self.assertNotIn("attempts", probe["judge_results"][0])

    def test_config_rejects_unknown_provider(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "scoring.json"
            path.write_text(
                json.dumps(
                    {
                        "name": "bad",
                        "providers": [],
                        "judges": [
                            {
                                "id": "judge",
                                "provider": "missing",
                                "model": "provider/model",
                            }
                        ],
                        "sources": [],
                        "output_dir": "out",
                    }
                )
            )
            with self.assertRaisesRegex(ConfigError, "unknown providers"):
                ProbeScoringConfig.from_path(path)

    def test_repair_prompt_repeats_original_evidence(self) -> None:
        probe = ProbeEvidence(
            id="fixture:turn_1",
            source_label="fixture",
            run_dir="fixture",
            question_turn_id=1,
            question="What is two plus two?",
            answers={"P1": "Four.", "P2": "Five."},
            unavailable_candidates=(),
        )

        repaired = json.loads(
            _repair_prompt(probe, ["P2", "P1"], "", "no JSON object")
        )

        self.assertEqual(repaired["probe"], probe.question)
        self.assertEqual(
            [item["candidate_id"] for item in repaired["answers"]], ["P2", "P1"]
        )
        self.assertEqual(repaired["repair"]["previous_output"], "")

    def test_load_journal_ignores_only_a_truncated_final_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "journal.jsonl"
            path.write_text('{"status":"ok"}\n{"status":')
            self.assertEqual(_load_journal(path), [{"status": "ok"}])
            self.assertEqual(path.read_text(), '{"status":"ok"}\n{"status":')
            self.assertEqual(
                _load_journal(path, repair_trailing=True), [{"status": "ok"}]
            )
            with path.open("a") as handle:
                handle.write('{"status":"next"}\n')
            self.assertEqual(
                _load_journal(path),
                [{"status": "ok"}, {"status": "next"}],
            )
            path.write_text('{"status":\n{"status":"ok"}\n')
            with self.assertRaises(json.JSONDecodeError):
                _load_journal(path)

    def test_resumed_failed_job_starts_with_recovery_parameters(self) -> None:
        class RecordingClient(ModelClient):
            def __init__(self) -> None:
                self.requests: list[ModelRequest] = []

            def generate(self, request: ModelRequest) -> ModelResponse:
                self.requests.append(request)
                return ModelResponse(
                    content=json.dumps(
                        {
                            "probe_rubric": ["Correct result"],
                            "scores": {
                                "P1": {
                                    "score": 4,
                                    "confidence": "high",
                                    "summary": "Correct.",
                                    "error_tags": [],
                                }
                            },
                        }
                    )
                )

        client = RecordingClient()
        judge = ScoringJudge(
            id="judge",
            provider="provider",
            model="provider/model",
            params={"reasoning": {"effort": "xhigh"}},
            recovery_params={"reasoning": {"effort": "low"}},
        )
        probe = ProbeEvidence(
            id="fixture:turn_1",
            source_label="fixture",
            run_dir="fixture",
            question_turn_id=1,
            question="Question",
            answers={"P1": "Answer"},
            unavailable_candidates=(),
        )
        config = ProbeScoringConfig(
            name="fixture",
            providers={},
            judges=(judge,),
            sources=(),
            output_dir=Path("."),
            structured_json_retries=0,
        )

        _score_one(judge, probe, client, None, config, start_with_recovery=True)

        self.assertEqual(
            client.requests[0].params, {"reasoning": {"effort": "low"}}
        )
        self.assertTrue(client.requests[0].metadata["recovery"])

    def test_transport_failure_after_malformed_output_preserves_attempt_cost(
        self,
    ) -> None:
        class FailingRepairClient(ModelClient):
            def __init__(self) -> None:
                self.call_count = 0

            def generate(self, request: ModelRequest) -> ModelResponse:
                self.call_count += 1
                if self.call_count == 1:
                    return ModelResponse(
                        content="not json",
                        usage={"cost": 0.25},
                    )
                raise RuntimeError("provider unavailable")

        judge = ScoringJudge(
            id="judge",
            provider="provider",
            model="provider/model",
        )
        probe = ProbeEvidence(
            id="fixture:turn_1",
            source_label="fixture",
            run_dir="fixture",
            question_turn_id=1,
            question="Question",
            answers={"P1": "Answer"},
            unavailable_candidates=(),
        )
        config = ProbeScoringConfig(
            name="fixture",
            providers={},
            judges=(judge,),
            sources=(),
            output_dir=Path("."),
            structured_json_retries=1,
        )

        with self.assertRaises(ScorePayloadError) as raised:
            _score_one(judge, probe, FailingRepairClient(), None, config)

        self.assertEqual(raised.exception.attempts[0]["usage"]["cost"], 0.25)

    def test_export_summary_removes_machine_specific_run_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "scores.json"
            summary = {
                "name": "fixture",
                "probes": [
                    {
                        "run_dir": "/home/researcher/project/runs/source_run",
                        "mean_scores": {"P1": 4.0},
                    }
                ],
            }

            write_export_summary(summary, path)

            exported = json.loads(path.read_text())
            self.assertEqual(exported["probes"][0]["run_dir"], "source_run")
            self.assertEqual(
                summary["probes"][0]["run_dir"],
                "/home/researcher/project/runs/source_run",
            )


if __name__ == "__main__":
    unittest.main()

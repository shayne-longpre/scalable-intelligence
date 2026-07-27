from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ai_council.config import ConfigError, ProviderSpec
from ai_council.probe_self_analysis import candidate_pair_accuracy
from ai_council.probe_self_analysis import reference_metrics
from ai_council.probe_self_analysis import _summarize_group
from ai_council.probe_self_study import (
    ProbeSelfStudyConfig,
    ReferenceEvaluator,
    run_probe_self_study,
    validate_author_assessment,
    validate_reference_score,
    _author_assessment_prompt,
)


class ProbeSelfStudyTests(unittest.TestCase):
    def test_author_assessment_schema_is_bounded(self) -> None:
        result = validate_author_assessment(
            {
                "probe_validity": "VALID",
                "checkability": "objective",
                "intended_level": "stronger",
                "self_solvability": "uncertain",
                "predicted_score": 2,
                "confidence": 0.6,
                "rationale": ["Hard but checkable.", "I may miss the proof."],
            }
        )
        self.assertEqual(result["probe_validity"], "valid")
        self.assertEqual(result["predicted_score"], 2)

    def test_author_assessment_prompt_requires_top_level_output_keys(self) -> None:
        prompt = json.loads(_author_assessment_prompt("Question"))
        self.assertIn("output_schema", prompt)
        self.assertNotIn("fields", prompt)
        self.assertTrue(
            any("top level" in rule for rule in prompt["rules"])
        )

    def test_reference_schema_requires_every_anonymous_answer(self) -> None:
        payload = {
            "probe_validity": "valid",
            "validity_reason": "The result is checkable.",
            "probe_rubric": ["Correct conclusion", "Sound proof"],
            "scores": {
                "A01": {
                    "score": 2,
                    "confidence": "medium",
                    "summary": "Partial result.",
                    "error_tags": ["proof gap"],
                },
                "A02": {
                    "score": 4,
                    "confidence": "high",
                    "summary": "Complete solution.",
                    "error_tags": [],
                },
            },
        }
        result = validate_reference_score(payload, {"A01", "A02"})
        self.assertEqual(result["scores"]["A02"]["score"], 4)
        with self.assertRaisesRegex(ValueError, "missing"):
            validate_reference_score(payload, {"A01", "A02", "A03"})

    def test_candidate_pair_accuracy_gives_half_credit_to_score_ties(self) -> None:
        accuracy, count = candidate_pair_accuracy(
            [
                {"candidate_score": 30, "score": 3},
                {"candidate_score": 20, "score": 3},
                {"candidate_score": 10, "score": 1},
            ]
        )
        self.assertEqual(count, 3)
        self.assertAlmostEqual(accuracy, 5 / 6)

    def test_beyond_author_requires_a_stronger_candidate_to_succeed(self) -> None:
        result = reference_metrics(
            {
                "evaluation": {
                    "probe_validity": "valid",
                    "validity_reason": "checkable",
                    "scores": {
                        "author": {"score": 2},
                        "weaker": {"score": 4},
                        "stronger": {"score": 2},
                    },
                },
                "answer_map": {
                    "author": {
                        "kind": "author_fresh",
                        "candidate_score": 20,
                    },
                    "weaker": {
                        "kind": "candidate",
                        "candidate_score": 10,
                    },
                    "stronger": {
                        "kind": "candidate",
                        "candidate_score": 30,
                    },
                },
            }
        )
        self.assertEqual(result["best_candidate_score"], 4)
        self.assertFalse(result["beyond_author"])
        self.assertFalse(result["stronger_candidate_succeeds"])

    def test_author_summary_preserves_intent_and_solvability_claims(self) -> None:
        base = {
            "reference_validity": "valid",
            "beyond_author": False,
            "candidate_pair_accuracy": 0.5,
            "candidate_score_range": 2,
            "self_score_error": 0,
        }
        summary = _summarize_group(
            "author",
            [
                {
                    **base,
                    "author_assessment": {
                        "intended_level": "stronger",
                        "self_solvability": "not_solvable",
                    },
                },
                {
                    **base,
                    "author_assessment": {
                        "intended_level": "peer",
                        "self_solvability": "fully",
                    },
                },
            ],
        )
        self.assertEqual(summary["intended_stronger_rate"], 0.5)
        self.assertEqual(summary["self_reported_unsolvable_rate"], 0.5)

    def test_author_summary_keeps_unscored_self_assessments(self) -> None:
        summary = _summarize_group(
            "author",
            [
                {
                    "author_assessment": {
                        "intended_level": "stronger",
                        "self_solvability": "not_solvable",
                    }
                },
                {
                    "reference_validity": "valid",
                    "beyond_author": False,
                    "candidate_pair_accuracy": 0.5,
                    "candidate_score_range": 1,
                    "self_score_error": 0,
                    "author_assessment": {
                        "intended_level": "peer",
                        "self_solvability": "fully",
                    },
                },
            ],
        )
        self.assertEqual(summary["probe_count"], 2)
        self.assertEqual(summary["scored_probe_count"], 1)
        self.assertEqual(summary["self_reported_unsolvable_rate"], 0.5)
        self.assertEqual(summary["valid_rate"], 1)

    def test_config_rejects_duplicate_or_misordered_stages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            base = {
                "name": "test",
                "catalog_file": "catalog.json",
                "provider": {"name": "fake", "kind": "fake"},
                "reference_evaluator": {
                    "id": "reference",
                    "model": "reference-model",
                },
                "output_dir": "output",
            }
            for stages in (
                [],
                "author_solve",
                ["author_solve", "author_solve"],
                ["reference_score", "author_solve"],
            ):
                path.write_text(
                    json.dumps({**base, "stages": stages}),
                    encoding="utf-8",
                )
                with self.assertRaises(ConfigError):
                    ProbeSelfStudyConfig.from_path(path)

            for field, value in (
                ("probe_ids", "probe_1"),
                ("max_reported_cost_usd", 0),
            ):
                path.write_text(
                    json.dumps({**base, field: value}),
                    encoding="utf-8",
                )
                with self.assertRaises(ConfigError):
                    ProbeSelfStudyConfig.from_path(path)

    def test_stage_waves_preserve_dependencies_and_resume_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog_file = root / "catalog.json"
            probes = [
                {
                    "probe_id": f"probe_{index}",
                    "author_model": "author-model",
                    "question": f"Question {index}",
                    "occurrences": [
                        {
                            "stage": "baseline_battery",
                            "run_dir": "unused",
                            "question_turn_id": index,
                        }
                    ],
                }
                for index in range(3)
            ]
            catalog_file.write_text(
                json.dumps({"probes": probes}), encoding="utf-8"
            )
            config = self._config(
                root, catalog_file, stages=("author_solve", "author_assess")
            )
            events: list[str] = []

            def fake_job(stage, probe, *_args):
                events.append(stage)
                result = {
                    "status": "ok",
                    "job_key": f"{stage}:{probe['probe_id']}",
                    "stage": stage,
                    "probe_id": probe["probe_id"],
                    "author_model": probe["author_model"],
                    "attempts": [],
                }
                if stage == "author_solve":
                    result["solution"] = "answer"
                else:
                    result["assessment"] = {
                        "probe_validity": "valid",
                        "checkability": "objective",
                        "intended_level": "peer",
                        "self_solvability": "fully",
                        "predicted_score": 3,
                        "confidence": 0.8,
                        "rationale": ["test"],
                    }
                return result

            with (
                patch(
                    "ai_council.probe_self_study.build_client",
                    return_value=_ParallelClient(),
                ),
                patch(
                    "ai_council.probe_self_study._run_job",
                    side_effect=fake_job,
                ),
            ):
                summary = run_probe_self_study(config)
            self.assertEqual(
                events,
                ["author_solve"] * 3 + ["author_assess"] * 3,
            )
            self.assertEqual(summary["missing_jobs"], [])

    def test_cost_limit_stops_new_work_after_completed_calls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog_file = root / "catalog.json"
            probes = [
                {
                    "probe_id": f"probe_{index}",
                    "author_model": "author-model",
                    "question": f"Question {index}",
                    "occurrences": [
                        {
                            "stage": "baseline_battery",
                            "run_dir": "unused",
                            "question_turn_id": index,
                        }
                    ],
                }
                for index in range(5)
            ]
            catalog_file.write_text(
                json.dumps({"probes": probes}), encoding="utf-8"
            )
            config = self._config(
                root,
                catalog_file,
                stages=("author_solve",),
                max_parallel_calls=1,
                max_reported_cost_usd=2,
            )
            calls = []

            def fake_job(stage, probe, *_args):
                calls.append(probe["probe_id"])
                return {
                    "status": "ok",
                    "job_key": f"{stage}:{probe['probe_id']}",
                    "stage": stage,
                    "probe_id": probe["probe_id"],
                    "author_model": probe["author_model"],
                    "attempts": [{"usage": {"cost": 1}}],
                    "solution": "answer",
                }

            with (
                patch(
                    "ai_council.probe_self_study.build_client",
                    return_value=_ParallelClient(),
                ),
                patch(
                    "ai_council.probe_self_study._run_job",
                    side_effect=fake_job,
                ),
            ):
                summary = run_probe_self_study(config)
            self.assertEqual(len(calls), 2)
            self.assertEqual(summary["reported_cost_usd"], 2)
            self.assertEqual(len(summary["missing_jobs"]), 3)

    @staticmethod
    def _config(
        root: Path,
        catalog_file: Path,
        *,
        stages: tuple[str, ...],
        max_parallel_calls: int = 4,
        max_reported_cost_usd: float | None = None,
    ) -> ProbeSelfStudyConfig:
        return ProbeSelfStudyConfig(
            name="test",
            catalog_file=catalog_file,
            provider=ProviderSpec(name="fake", kind="fake"),
            reference_evaluator=ReferenceEvaluator(
                id="reference", model="reference-model"
            ),
            output_dir=root / "output",
            stages=stages,
            max_parallel_calls=max_parallel_calls,
            max_reported_cost_usd=max_reported_cost_usd,
        )


class _ParallelClient:
    supports_parallel_requests = True


if __name__ == "__main__":
    unittest.main()

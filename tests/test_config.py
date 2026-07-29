from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from ai_council.config import ConfigError, ExperimentConfig, load_experiment_config
from ai_council.experiment_builders import (
    AdaptiveJudgeConfigSpec,
    build_adaptive_judge_config,
    build_exact_evidence_cross_judge_config,
    build_exact_evidence_order_replay_config,
    candidate_model,
    expected_model_calls,
    judge_model_params,
)
from scripts.build_adaptive_judge_study import build_study_configs
from scripts.build_adaptive_judge_study import validate_relative_gap_requirements
from scripts.build_cross_judge_study import validate_exact_evidence_source
from scripts.build_ceiling_probe_extension import (
    build_ceiling_extension_config,
    write_opening_replay_bundle,
)
from scripts.build_partial_resume_config import (
    build_partial_resume_config,
    write_replay_bundle,
)
from scripts.build_replay_repair_config import build_replay_repair_config


ROOT = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    def test_ceiling_extension_replays_opening_evidence_and_adds_probe_budget(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source_run = Path(tmpdir)
            source = {
                "name": "source",
                "run": {"max_model_calls": 20},
                "models": {
                    "judge": {
                        "name": "judge",
                        "provider": "mock",
                        "model": "mock:judge",
                    }
                },
                "participants": [{"id": "P1"}, {"id": "P2"}],
                "judges": [{"id": "J1", "model": "judge"}],
                "protocol": {
                    "phases": [
                        {
                            "kind": "independent_judge_ranking",
                            "probe_schedule": [5, 1],
                        }
                    ]
                },
                "metadata": {
                    "repair_source_run": "old-run",
                    "repair_retry_unavailable_rounds": [1],
                },
            }
            (source_run / "config.json").write_text(json.dumps(source))
            turns = []
            turn_id = 1
            for probe_number in range(1, 6):
                probe_id = f"judge_ranking:J1:round_1:probe_{probe_number}"
                turns.append(
                    {
                        "turn_id": turn_id,
                        "round_index": 1,
                        "speaker": "J1",
                        "content": f"Probe {probe_number}",
                        "metadata": {
                            "interaction_role": "question",
                            "probe_id": probe_id,
                            "probe_number": probe_number,
                        },
                    }
                )
                turn_id += 1
                for participant_id in ("P1", "P2"):
                    turns.append(
                        {
                            "turn_id": turn_id,
                            "round_index": 1,
                            "speaker": participant_id,
                            "content": "Answer",
                            "metadata": {
                                "interaction_role": "answer",
                                "probe_id": probe_id,
                            },
                        }
                    )
                    turn_id += 1
                turns.append(
                    {
                        "turn_id": turn_id,
                        "round_index": 1,
                        "speaker": "J1",
                        "content": "{}",
                        "metadata": {
                            "interaction_role": "probe_comparison",
                            "probe_id": probe_id,
                        },
                    }
                )
                turn_id += 1
            (source_run / "transcript.jsonl").write_text(
                "".join(json.dumps(turn) + "\n" for turn in turns)
            )
            (source_run / "run_summary.json").write_text(
                json.dumps({"status": "completed"})
            )

            extended = build_ceiling_extension_config(
                source_run,
                additional_probes=5,
                guidance="Seek evidence above your own ceiling.",
            )
            opening_replay = write_opening_replay_bundle(
                source_run,
                source_run / "opening_replay.jsonl",
            )
            adaptive = build_ceiling_extension_config(
                source_run,
                additional_probes=5,
                adaptive_probe_counts=[1, 1],
                guidance="Seek evidence above your own ceiling.",
                preauthored_opening_file=opening_replay,
            )
            replay_entries = [
                json.loads(line)
                for line in opening_replay.read_text().splitlines()
            ]
            turns[1]["content"] = ""
            turns[1]["metadata"]["answer_unavailable"] = True
            (source_run / "transcript.jsonl").write_text(
                "".join(json.dumps(turn) + "\n" for turn in turns)
            )
            with self.assertRaisesRegex(ValueError, "answers are incomplete"):
                build_ceiling_extension_config(source_run)

        phase = extended["protocol"]["phases"][0]
        self.assertEqual(phase["probe_schedule"], [10])
        self.assertEqual(phase["rounds"], 1)
        self.assertEqual(
            phase["preauthored_probe_file"],
            str(source_run / "transcript.jsonl"),
        )
        self.assertEqual(phase["preauthored_answer_file"], str(source_run))
        self.assertEqual(
            phase["preauthored_evidence_file"],
            str(source_run / "transcript.jsonl"),
        )
        self.assertIsNone(phase["preauthored_ranking_file"])
        self.assertEqual(
            phase["probe_generation_guidance"],
            "Seek evidence above your own ceiling.",
        )
        self.assertNotIn("repair_source_run", extended["metadata"])
        self.assertNotIn(
            "repair_retry_unavailable_rounds",
            extended["metadata"],
        )
        self.assertEqual(extended["run"]["max_model_calls"], 41)
        adaptive_phase = adaptive["protocol"]["phases"][0]
        self.assertEqual(adaptive_phase["probe_schedule"], [10, 1, 1])
        self.assertEqual(adaptive_phase["rounds"], 3)
        self.assertEqual(
            adaptive_phase["preauthored_probe_file"],
            str(opening_replay),
        )
        self.assertEqual(
            adaptive_phase["preauthored_evidence_file"],
            str(opening_replay),
        )
        self.assertEqual(adaptive["run"]["max_model_calls"], 51)
        self.assertEqual(len(replay_entries), 10)
        self.assertTrue(
            all(entry["round_index"] == 1 for entry in replay_entries)
        )

    def test_ceiling_extension_rejects_incomplete_opening_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source_run = Path(tmpdir)
            (source_run / "config.json").write_text(
                json.dumps(
                    {
                        "name": "source",
                        "participants": [{"id": "P1"}],
                        "judges": [{"id": "J1", "model": "judge"}],
                        "models": {
                            "judge": {
                                "name": "judge",
                                "model": "mock:judge",
                            }
                        },
                        "protocol": {
                            "phases": [
                                {
                                    "kind": "independent_judge_ranking",
                                    "probe_schedule": [1],
                                }
                            ]
                        },
                    }
                )
            )
            (source_run / "transcript.jsonl").write_text("")
            (source_run / "run_summary.json").write_text(
                json.dumps({"status": "completed"})
            )

            with self.assertRaisesRegex(ValueError, "probes are incomplete"):
                build_ceiling_extension_config(source_run)

    def test_catalog_ladder_call_budget_scales_with_roster_and_schedule(self) -> None:
        self.assertEqual(
            expected_model_calls(
                candidate_count=50,
                probe_schedule=[4, 1, 1],
                max_adaptive_candidates=10,
            ),
            235,
        )
        self.assertEqual(
            expected_model_calls(
                candidate_count=100,
                probe_schedule=[4, 1, 1],
                max_adaptive_candidates=10,
            ),
            435,
        )

    def test_catalog_ladder_model_override_preserves_unspecified_parameters(self) -> None:
        model = {
            "provider_model_id": "provider/model",
            "max_completion_tokens": 100000,
            "matched_evals": [{"model_name": "Model (xhigh)"}],
        }

        configured = candidate_model(
            model,
            "P01",
            {"params": {"reasoning": {"effort": "low"}, "temperature": 0}},
        )

        self.assertEqual(configured["params"]["max_tokens"], 40000)
        self.assertEqual(configured["params"]["reasoning"], {"effort": "low"})
        self.assertEqual(configured["params"]["temperature"], 0)

    def test_non_reasoning_judge_does_not_receive_reasoning_parameters(self) -> None:
        model = {
            "provider_model_id": "provider/model",
            "max_completion_tokens": 64000,
            "matched_evals": [{"model_name": "Model"}],
        }

        params, recovery = judge_model_params(model)

        self.assertEqual(params, {"max_tokens": 20000, "temperature": 0.2})
        self.assertEqual(recovery, {"max_tokens": 8000, "temperature": 0})

    def test_catalog_ladder_configs_match_frozen_fifty_model_selection(self) -> None:
        selection = json.loads(
            (ROOT / "data" / "catalog_ladder_50.selection.json").read_text(
                encoding="utf-8"
            )
        )
        selected_routes = set(selection["provider_model_ids"])
        catalog = json.loads(
            (ROOT / "data" / "model_catalog.openrouter.json").read_text(
                encoding="utf-8"
            )
        )
        route_limits = {
            model["provider_model_id"]: model.get("max_completion_tokens")
            for model in catalog["models"]
        }

        for filename, judge_route, schedule, timeout in [
            (
                "catalog_ladder50_sol.openrouter.json",
                "openai/gpt-5.6-sol",
                [4, 1, 1],
                300,
            ),
            (
                "catalog_ladder50_fable.openrouter.json",
                "anthropic/claude-fable-5",
                [4, 1, 1],
                300,
            ),
            (
                "catalog_ladder50_sol_opening5.openrouter.json",
                "openai/gpt-5.6-sol",
                [5, 1, 1],
                600,
            ),
            (
                "catalog_ladder50_fable_opening5.openrouter.json",
                "anthropic/claude-fable-5",
                [5, 1, 1],
                600,
            ),
        ]:
            with self.subTest(filename=filename):
                config = load_experiment_config(ROOT / "examples" / filename)
                participant_ids = [participant.id for participant in config.participants]
                participant_routes = {
                    config.models[participant.model].model
                    for participant in config.participants
                }
                phase = config.protocol.phases[0]

                self.assertEqual(len(participant_ids), 50)
                self.assertEqual(len(set(participant_ids)), 50)
                self.assertEqual(participant_routes, selected_routes)
                self.assertIn(judge_route, participant_routes)
                for participant in config.participants:
                    model = config.models[participant.model]
                    route_limit = route_limits[model.model]
                    expected_limit = min(40000, route_limit) if route_limit else 40000
                    self.assertEqual(model.params.get("max_tokens"), expected_limit)
                self.assertEqual(phase.probe_schedule, schedule)
                self.assertEqual(phase.comparison_order, "seeded_shuffle")
                self.assertEqual(phase.incomplete_answer_policy, "record_unavailable")
                self.assertTrue(config.run.continue_batch_on_call_error)
                self.assertEqual(
                    config.providers["openrouter_candidates"].timeout_seconds,
                    timeout,
                )
                self.assertEqual(config.providers["openrouter_judge"].timeout_seconds, 900)
                self.assertTrue(
                    all(
                        config.models[participant.model].provider
                        == "openrouter_candidates"
                        for participant in config.participants
                    )
                )
                self.assertEqual(config.models["judge_primary"].provider, "openrouter_judge")

    def test_oversight_frontier_manifest_builds_six_diverse_judge_configs(self) -> None:
        study_path = ROOT / "studies" / "oversight_frontier_v1.json"
        study = json.loads(
            study_path.read_text()
        )
        catalog = json.loads(
            (ROOT / "data" / "model_catalog.openrouter.json").read_text()
        )
        scores = {
            model["provider_model_id"]: model["intelligence_score"]
            for model in catalog["models"]
        }

        self.assertEqual(study["protocol"]["probe_schedule"], [5, 1])
        self.assertEqual(len(study["conditions"]), 6)
        self.assertEqual(
            len(
                {
                    condition["judge"]["model"].split("/", 1)[0]
                    for condition in study["conditions"]
                }
            ),
            6,
        )
        for condition in study["conditions"]:
            routes = condition["candidate_models"]
            judge_route = condition["judge"]["model"]
            judge_score = scores[judge_route]
            self.assertEqual(len(routes), 7)
            self.assertEqual(len(set(routes)), 7)
            self.assertIn(judge_route, routes)
            self.assertTrue(any(scores[route] > judge_score for route in routes))
            self.assertTrue(any(scores[route] < judge_score for route in routes))

        with tempfile.TemporaryDirectory() as temp_dir:
            index = build_study_configs(study_path, Path(temp_dir))
            self.assertEqual(len(index["configs"]), 6)
            for built in index["configs"]:
                config = load_experiment_config(built["config"])
                phase = config.protocol.phases[0]
                condition = next(
                    row
                    for row in study["conditions"]
                    if row["id"] == built["condition_id"]
                )
                configured_routes = {
                    config.models[participant.model].model
                    for participant in config.participants
                }
                configured_judge = config.models[config.judges[0].model].model
                self.assertEqual(configured_routes, set(condition["candidate_models"]))
                self.assertEqual(configured_judge, condition["judge"]["model"])
                self.assertEqual(phase.probe_schedule, [5, 1])
                self.assertEqual(phase.max_adaptive_candidates, 4)
                self.assertEqual(config.run.visible_text_retries, 1)
                self.assertEqual(
                    config.providers["openrouter_candidates"].request_retries, 1
                )
                self.assertEqual(
                    config.models[config.judges[0].model].params,
                    condition["judge"]["params"],
                )

    def test_adaptive_judge_builder_does_not_require_checked_in_generated_config(self) -> None:
        catalog = json.loads(
            (ROOT / "data" / "model_catalog.openrouter.json").read_text()
        )
        selected = [
            "anthropic/claude-fable-5",
            "openai/gpt-5.6-sol",
        ]
        config = build_adaptive_judge_config(
            spec=AdaptiveJudgeConfigSpec(
                name="generated_test",
                judge_model="openai/gpt-5.6-sol",
                participant_seed=7,
                comparison_seed=11,
                probe_schedule=(5,),
                max_adaptive_candidates=2,
            ),
            selected_model_ids=selected,
            catalog=catalog,
            catalog_label="catalog.json",
            selection_label="study.json#condition",
        )

        self.assertEqual(config["protocol"]["phases"][0]["probe_schedule"], [5])
        self.assertEqual(len(config["participants"]), 2)

    def test_oversight_replication_changes_panels_without_changing_protocol(self) -> None:
        first = json.loads(
            (ROOT / "studies" / "oversight_frontier_v1.json").read_text()
        )
        second = json.loads(
            (ROOT / "studies" / "oversight_frontier_v2.json").read_text()
        )
        catalog = json.loads(
            (ROOT / "data" / "model_catalog.openrouter.json").read_text()
        )
        scores = {
            model["provider_model_id"]: model["intelligence_score"]
            for model in catalog["models"]
        }
        first_by_id = {condition["id"]: condition for condition in first["conditions"]}

        self.assertEqual(second["protocol"]["probe_schedule"], [5, 1])
        self.assertEqual(len(second["conditions"]), 6)
        for condition in second["conditions"]:
            source = first_by_id[condition["id"]]
            judge = condition["judge"]["model"]
            routes = condition["candidate_models"]
            self.assertEqual(condition["judge"], source["judge"])
            self.assertNotEqual(routes, source["candidate_models"])
            self.assertNotEqual(
                condition["participant_seed"], source["participant_seed"]
            )
            self.assertNotEqual(
                condition["comparison_seed"], source["comparison_seed"]
            )
            self.assertEqual(len(routes), 7)
            self.assertEqual(len(set(routes)), 7)
            self.assertIn(judge, routes)
            self.assertTrue(any(scores[route] > scores[judge] for route in routes))
            self.assertTrue(any(scores[route] < scores[judge] for route in routes))

    def test_above_heavy_frontier_has_ten_judges_and_five_superiors_when_available(
        self,
    ) -> None:
        study_path = ROOT / "studies" / "oversight_frontier_v3_above_heavy.json"
        study = json.loads(study_path.read_text())
        catalog = json.loads(
            (ROOT / "data" / "model_catalog.openrouter.json").read_text()
        )
        scores = {
            model["provider_model_id"]: model["intelligence_score"]
            for model in catalog["models"]
        }
        conditions = study["conditions"]

        self.assertEqual(study["protocol"]["probe_schedule"], [5, 1])
        self.assertEqual(len(conditions), 10)
        self.assertEqual(len({condition["id"] for condition in conditions}), 10)
        self.assertEqual(
            len({condition["judge"]["model"] for condition in conditions}), 10
        )
        self.assertEqual(
            len(
                {
                    condition["judge"]["model"].split("/", 1)[0]
                    for condition in conditions
                }
            ),
            10,
        )
        for condition in conditions:
            routes = condition["candidate_models"]
            judge = condition["judge"]["model"]
            superior_count = sum(scores[route] > scores[judge] for route in routes)
            inferior_count = sum(scores[route] < scores[judge] for route in routes)
            self.assertEqual(len(routes), 9)
            self.assertEqual(len(set(routes)), 9)
            self.assertIn(judge, routes)
            self.assertEqual(superior_count, 1 if condition["id"] == "sol" else 5)
            self.assertEqual(inferior_count, 7 if condition["id"] == "sol" else 3)

        with tempfile.TemporaryDirectory() as temp_dir:
            index = build_study_configs(study_path, Path(temp_dir))
            self.assertEqual(len(index["configs"]), 10)
            for built in index["configs"]:
                config = load_experiment_config(built["config"])
                phase = config.protocol.phases[0]
                self.assertEqual(len(config.participants), 9)
                self.assertEqual(phase.probe_schedule, [5, 1])
                self.assertEqual(phase.max_adaptive_candidates, 4)

    def test_matched_frontier_extension_enforces_relative_gap_bands(self) -> None:
        study_path = (
            ROOT / "studies" / "oversight_frontier_v4_matched_extension.json"
        )
        study = json.loads(study_path.read_text())

        self.assertEqual(len(study["conditions"]), 8)
        self.assertEqual(study["protocol"]["probe_schedule"], [5, 1])
        with tempfile.TemporaryDirectory() as temp_dir:
            index = build_study_configs(study_path, Path(temp_dir))

        self.assertEqual(len(index["configs"]), 8)
        for built in index["configs"]:
            self.assertEqual(
                built["relative_gap_counts"],
                {
                    "near above": 2,
                    "moderate above": 1,
                    "far above": 2,
                    "near below": 1,
                    "moderate below": 1,
                    "far below": 1,
                },
            )

    def test_verifier_study_changes_only_probe_design_guidance(self) -> None:
        study_path = ROOT / "studies" / "verifier_oriented_probes_v1.json"
        study = json.loads(study_path.read_text())

        self.assertEqual(study["protocol"]["probe_schedule"], [5])
        self.assertEqual(len(study["conditions"]), 4)
        with tempfile.TemporaryDirectory() as temp_dir:
            index = build_study_configs(study_path, Path(temp_dir))
            for built in index["configs"]:
                config = load_experiment_config(built["config"])
                phase = config.protocol.phases[0]
                condition = next(
                    row
                    for row in study["conditions"]
                    if row["id"] == built["condition_id"]
                )
                baseline = load_experiment_config(
                    ROOT / condition["baseline_run"] / "config.json"
                )
                baseline_phase = baseline.protocol.phases[0]

                self.assertEqual(phase.probe_schedule, [5])
                self.assertEqual(
                    phase.probe_generation_guidance,
                    study["protocol"]["probe_generation_guidance"],
                )
                self.assertEqual(
                    config.metadata["primary_endpoint"],
                    study["protocol"]["primary_endpoint"],
                )
                self.assertEqual(
                    [participant.id for participant in config.participants],
                    [participant.id for participant in baseline.participants],
                )
                self.assertEqual(
                    {
                        participant.id: config.models[
                            participant.model
                        ].model
                        for participant in config.participants
                    },
                    {
                        participant.id: baseline.models[
                            participant.model
                        ].model
                        for participant in baseline.participants
                    },
                )
                self.assertEqual(
                    phase.comparison_seed,
                    baseline_phase.comparison_seed,
                )
                self.assertEqual(
                    phase.comparison_order,
                    baseline_phase.comparison_order,
                )
                self.assertEqual(
                    phase.incomplete_answer_policy,
                    baseline_phase.incomplete_answer_policy,
                )
                self.assertEqual(
                    config.prompt_overrides,
                    baseline.prompt_overrides,
                )

    def test_relative_gap_validation_rejects_an_unmatched_panel(self) -> None:
        catalog = {
            "judge": {"intelligence_score": 10.0},
            "near": {"intelligence_score": 11.0},
            "far": {"intelligence_score": 20.0},
        }

        with self.assertRaisesRegex(ValueError, "requires 2 near above"):
            validate_relative_gap_requirements(
                condition_id="test",
                judge_model="judge",
                candidate_models=["judge", "near", "far"],
                catalog_by_route=catalog,
                requirements=[
                    {
                        "label": "near above",
                        "side": "above",
                        "min_gap": 0,
                        "max_gap": 2,
                        "minimum": 2,
                    }
                ],
            )

    def test_relative_gap_validation_rejects_ambiguous_requirements(self) -> None:
        catalog = {
            "judge": {"intelligence_score": 10.0},
            "candidate": {"intelligence_score": 11.0},
        }
        base = {
            "label": "near above",
            "side": "above",
            "min_gap": 0,
            "max_gap": 2,
            "minimum": 1,
        }

        with self.assertRaisesRegex(ValueError, "repeats gap requirement label"):
            validate_relative_gap_requirements(
                condition_id="test",
                judge_model="judge",
                candidate_models=["judge", "candidate"],
                catalog_by_route=catalog,
                requirements=[base, base],
            )

        with self.assertRaisesRegex(ValueError, "minimum must be an integer"):
            validate_relative_gap_requirements(
                condition_id="test",
                judge_model="judge",
                candidate_models=["judge", "candidate"],
                catalog_by_route=catalog,
                requirements=[{**base, "minimum": 1.5}],
            )

    def test_relative_gap_validation_requires_catalog_scores(self) -> None:
        with self.assertRaisesRegex(ValueError, "candidate candidate has no"):
            validate_relative_gap_requirements(
                condition_id="test",
                judge_model="judge",
                candidate_models=["judge", "candidate"],
                catalog_by_route={
                    "judge": {"intelligence_score": 10.0},
                    "candidate": {},
                },
                requirements=[
                    {
                        "label": "near above",
                        "side": "above",
                        "min_gap": 0,
                        "max_gap": 2,
                        "minimum": 1,
                    }
                ],
            )

    def test_exact_order_replay_reuses_answers_but_regenerates_judgments(self) -> None:
        source = {
            "name": "source",
            "protocol": {
                "phases": [
                    {
                        "kind": "independent_judge_ranking",
                        "comparison_order": "seeded_shuffle",
                        "comparison_seed": 10,
                        "preauthored_evidence_file": "old-evidence.jsonl",
                        "preauthored_ranking_file": "old-rankings.jsonl",
                    }
                ]
            },
            "metadata": {
                "study_condition": "sol",
                "repair_source_run": "old",
                "repair_retry_unavailable_rounds": [2],
            },
        }

        replay = build_exact_evidence_order_replay_config(
            source,
            source_run="runs/source",
            comparison_seed=20,
            study_condition="sol_order",
        )
        phase = replay["protocol"]["phases"][0]

        self.assertEqual(source["protocol"]["phases"][0]["comparison_seed"], 10)
        self.assertEqual(
            phase["preauthored_probe_file"], "runs/source/transcript.jsonl"
        )
        self.assertEqual(phase["preauthored_answer_file"], "runs/source")
        self.assertEqual(phase["comparison_seed"], 20)
        self.assertTrue(phase["reuse_unavailable_answers"])
        self.assertTrue(phase["replay_source_targets"])
        self.assertNotIn("preauthored_evidence_file", phase)
        self.assertNotIn("preauthored_ranking_file", phase)
        self.assertEqual(replay["metadata"]["study_condition"], "sol_order")
        self.assertEqual(replay["metadata"]["source_study_condition"], "sol")
        self.assertNotIn("repair_source_run", replay["metadata"])

        with self.assertRaisesRegex(ValueError, "must differ"):
            build_exact_evidence_order_replay_config(
                source,
                source_run="runs/source",
                comparison_seed=10,
                study_condition="sol_order",
            )

    def test_cross_judge_replay_changes_only_the_evaluator(self) -> None:
        catalog = json.loads(
            (ROOT / "data" / "model_catalog.openrouter.json").read_text()
        )
        source = {
            "name": "source",
            "models": [
                {
                    "name": "candidate_p01",
                    "provider": "openrouter_candidates",
                    "model": "meta-llama/llama-4-maverick",
                    "params": {"max_tokens": 1000},
                },
                {
                    "name": "judge_primary",
                    "provider": "openrouter_judge",
                    "model": "meta-llama/llama-4-maverick",
                    "params": {"max_tokens": 1000},
                },
            ],
            "judges": [{"id": "J1", "model": "judge_primary"}],
            "protocol": {
                "phases": [
                    {
                        "kind": "independent_judge_ranking",
                        "comparison_order": "seeded_shuffle",
                        "comparison_seed": 10,
                    }
                ]
            },
            "metadata": {"study_condition": "source"},
        }

        replay = build_exact_evidence_cross_judge_config(
            source,
            source_run="runs/source",
            comparison_seed=20,
            study_condition="source_sol",
            judge_model="openai/gpt-5.6-sol",
            catalog=catalog,
        )

        candidate = next(
            row for row in replay["models"] if row["name"] == "candidate_p01"
        )
        judge = next(
            row for row in replay["models"] if row["name"] == "judge_primary"
        )
        phase = replay["protocol"]["phases"][0]
        self.assertEqual(candidate["model"], "meta-llama/llama-4-maverick")
        self.assertEqual(judge["model"], "openai/gpt-5.6-sol")
        self.assertEqual(
            phase["preauthored_probe_file"],
            "runs/source/transcript.jsonl",
        )
        self.assertEqual(phase["preauthored_answer_file"], "runs/source")
        self.assertTrue(phase["replay_source_targets"])
        self.assertEqual(
            replay["metadata"]["source_judge_model"],
            "meta-llama/llama-4-maverick",
        )
        self.assertEqual(
            replay["metadata"]["cross_judge_model"],
            "openai/gpt-5.6-sol",
        )
        self.assertEqual(replay["metadata"]["study_condition"], "source_sol")
        self.assertEqual(replay["metadata"]["source_study_condition"], "source")

    def test_cross_judge_source_must_be_complete_and_have_all_answers(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            (run_dir / "config.json").write_text("{}")
            (run_dir / "transcript.jsonl").write_text(
                json.dumps(
                    {
                        "metadata": {
                            "answer_unavailable": True,
                        }
                    }
                )
                + "\n"
            )
            (run_dir / "run_summary.json").write_text(
                json.dumps({"status": "completed"})
            )

            with self.assertRaisesRegex(ValueError, "1 unavailable"):
                validate_exact_evidence_source(
                    run_dir,
                    condition_id="cell",
                )
            validate_exact_evidence_source(
                run_dir,
                condition_id="cell",
                allow_unavailable=True,
            )

    def test_replay_repair_reuses_evidence_but_recomputes_comparisons(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            built = build_study_configs(
                ROOT / "studies" / "oversight_frontier_v1.json",
                root / "configs",
            )
            source_run = root / "source"
            source_run.mkdir()
            source_config = json.loads(
                Path(built["configs"][0]["config"]).read_text()
            )
            phase = source_config["protocol"]["phases"][0]
            phase["preauthored_evidence_file"] = "old-comparisons.jsonl"
            phase["preauthored_ranking_file"] = "old-rankings.jsonl"
            (source_run / "config.json").write_text(json.dumps(source_config))
            participant_ref = source_config["participants"][0]["model"]
            participant_route = next(
                model["model"]
                for model in source_config["models"]
                if model["name"] == participant_ref
            )
            parameter_override = {
                "max_tokens": 12000,
                "chat_template_kwargs": {"enable_thinking": False},
            }

            repaired = build_replay_repair_config(
                source_run,
                retry_unavailable_rounds=[2, 1, 2],
                candidate_timeout_seconds=900,
                use_recovery_params=True,
                model_parameter_overrides={
                    participant_route: parameter_override
                },
            )
            source_config["providers"] = {
                provider["name"]: provider for provider in source_config["providers"]
            }
            (source_run / "config.json").write_text(json.dumps(source_config))
            repaired_from_resolved_config = build_replay_repair_config(
                source_run,
                retry_unavailable_rounds=[2],
            )
            with self.assertRaisesRegex(ValueError, "did not match"):
                build_replay_repair_config(
                    source_run,
                    retry_unavailable_rounds=[1],
                    model_parameter_overrides={
                        "provider/missing": {"max_tokens": 10}
                    },
                )
            (source_run / "transcript.jsonl").write_text(
                json.dumps(
                    {
                        "round_index": 1,
                        "metadata": {"answer_unavailable": True},
                    }
                )
                + "\n"
            )
            with self.assertRaisesRegex(
                ValueError,
                "rounds have no unavailable answers: 2",
            ):
                build_replay_repair_config(
                    source_run,
                    retry_unavailable_rounds=[2],
                )
            build_replay_repair_config(
                source_run,
                retry_unavailable_rounds=[1],
            )
            (source_run / "pending_batch_entries.jsonl").write_text(
                json.dumps(
                    {
                        "round_index": 2,
                        "metadata": {"answer_unavailable": True},
                    }
                )
                + "\n"
            )
            build_replay_repair_config(
                source_run,
                retry_unavailable_rounds=[2],
            )

        phase = repaired["protocol"]["phases"][0]
        self.assertEqual(
            phase["preauthored_probe_file"], str(source_run / "transcript.jsonl")
        )
        self.assertEqual(phase["preauthored_answer_file"], str(source_run))
        self.assertEqual(phase["retry_unavailable_rounds"], [1, 2])
        self.assertTrue(phase["reuse_unavailable_answers"])
        self.assertTrue(phase["replay_source_targets"])
        self.assertNotIn("preauthored_evidence_file", phase)
        self.assertNotIn("preauthored_ranking_file", phase)
        candidate_provider = next(
            provider
            for provider in repaired["providers"]
            if provider["name"] == "openrouter_candidates"
        )
        self.assertEqual(candidate_provider["request_retries"], 1)
        self.assertEqual(candidate_provider["timeout_seconds"], 900)
        participant_ref = repaired["participants"][0]["model"]
        participant_model = next(
            model
            for model in repaired["models"]
            if model["name"] == participant_ref
        )
        self.assertEqual(participant_model["params"], participant_model["recovery_params"])
        self.assertEqual(participant_model["params"], parameter_override)
        self.assertTrue(repaired["metadata"]["repair_uses_recovery_params"])
        self.assertEqual(
            repaired["metadata"]["repair_parameter_overrides"],
            {participant_route: parameter_override},
        )
        self.assertEqual(
            repaired_from_resolved_config["providers"]["openrouter_candidates"][
                "request_retries"
            ],
            1,
        )

    def test_partial_resume_reuses_completed_stages_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source_run = Path(temp_dir) / "source"
            source_run.mkdir()
            source_config = {
                "name": "partial",
                "models": [
                    {
                        "name": "judge_model",
                        "params": {"reasoning": {"effort": "xhigh"}},
                        "recovery_params": {
                            "reasoning": {"effort": "low"},
                            "max_tokens": 8000,
                        },
                    }
                ],
                "judges": [{"id": "J1", "model": "judge_model"}],
                "protocol": {
                    "phases": [
                        {
                            "kind": "independent_judge_ranking",
                            "replay_source_targets": True,
                            "retry_unavailable_rounds": [1],
                        }
                    ]
                },
            }
            (source_run / "config.json").write_text(json.dumps(source_config))

            resumed = build_partial_resume_config(
                source_run,
                use_judge_recovery_params=True,
            )

        phase = resumed["protocol"]["phases"][0]
        transcript = str(source_run / "transcript.jsonl")
        self.assertEqual(phase["preauthored_probe_file"], transcript)
        self.assertEqual(phase["preauthored_answer_file"], str(source_run))
        self.assertEqual(phase["preauthored_evidence_file"], transcript)
        self.assertEqual(phase["preauthored_ranking_file"], transcript)
        self.assertTrue(phase["reuse_unavailable_answers"])
        self.assertFalse(phase["replay_source_targets"])
        self.assertEqual(phase["retry_unavailable_rounds"], [])
        self.assertEqual(resumed["metadata"]["resume_source_run"], str(source_run))
        self.assertTrue(
            resumed["metadata"]["resume_judge_uses_recovery_params"]
        )
        self.assertEqual(
            resumed["models"][0]["params"],
            {
                "reasoning": {"effort": "low"},
                "max_tokens": 8000,
            },
        )

    def test_replay_bundle_preserves_primary_and_adds_missing_streams(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            primary = root / "primary"
            supplement = root / "supplement"
            primary.mkdir()
            supplement.mkdir()
            primary_rows = [
                _replay_row("question", "q1", "question"),
                _replay_row("answer", "a1", "repaired"),
                _replay_row("answer", "a3", ""),
            ]
            supplement_rows = [
                _replay_row("question", "q1", "question"),
                _replay_row("answer", "a1", "stale"),
                _replay_row("question", "q2", "follow-up"),
                _replay_row("answer", "a2", "follow-up answer"),
                {
                    **_replay_row("answer", "a3", ""),
                    "metadata": {
                        "interaction_role": "answer",
                        "stream_id": "a3",
                        "answer_unavailable": True,
                    },
                },
            ]
            _write_jsonl(primary / "transcript.jsonl", primary_rows)
            _write_jsonl(supplement / "transcript.jsonl", supplement_rows)

            bundle = write_replay_bundle(
                primary_run=primary,
                supplement_run=supplement,
                output_dir=root / "bundle",
            )
            rows = [
                json.loads(line)
                for line in (bundle / "transcript.jsonl").read_text().splitlines()
            ]

        by_stream = {
            row["metadata"]["stream_id"]: row["content"] for row in rows
        }
        self.assertEqual(by_stream["a1"], "repaired")
        self.assertEqual(by_stream["q2"], "follow-up")
        self.assertEqual(by_stream["a2"], "follow-up answer")
        self.assertTrue(
            next(
                row
                for row in rows
                if row["metadata"].get("answer_unavailable") is True
            )["metadata"]["answer_unavailable"]
        )

    def test_independent_judge_phase_requires_judge_roster(self) -> None:
        data = _minimal_config_dict()
        data["protocol"]["phases"] = [
            {
                "name": "judge_ranking",
                "kind": "independent_judge_ranking",
                "prompt": "independent_judge_probe",
                "visibility": "private",
            }
        ]
        with self.assertRaisesRegex(ConfigError, "requires at least one configured judge"):
            ExperimentConfig.from_dict(data)

    def test_loads_mock_config(self) -> None:
        config = load_experiment_config(ROOT / "examples" / "blind_council.mock.json")
        self.assertEqual(config.name, "blind_council_mock")
        self.assertEqual(len(config.participants), 7)
        self.assertEqual(config.protocol.name, "blind_council_baseline")

    def test_rejects_public_phase_with_private_visibility(self) -> None:
        data = _minimal_config_dict()
        data["protocol"]["phases"][0]["visibility"] = "private"
        with self.assertRaises(ConfigError):
            ExperimentConfig.from_dict(data)

    def test_rejects_zero_rounds(self) -> None:
        data = _minimal_config_dict()
        data["protocol"]["phases"][0]["rounds"] = 0
        with self.assertRaises(ConfigError):
            ExperimentConfig.from_dict(data)

    def test_loads_and_validates_max_parallel_calls(self) -> None:
        data = _minimal_config_dict()
        data["run"] = {"max_parallel_calls": 4}
        self.assertEqual(ExperimentConfig.from_dict(data).run.max_parallel_calls, 4)

        data["run"]["max_parallel_calls"] = 0
        with self.assertRaisesRegex(ConfigError, "max_parallel_calls must be at least 1"):
            ExperimentConfig.from_dict(data)

    def test_continue_batch_on_call_error_requires_boolean(self) -> None:
        data = _minimal_config_dict()
        data["run"] = {"continue_batch_on_call_error": True}
        self.assertTrue(ExperimentConfig.from_dict(data).run.continue_batch_on_call_error)

        data["run"]["continue_batch_on_call_error"] = "yes"
        with self.assertRaisesRegex(ConfigError, "continue_batch_on_call_error"):
            ExperimentConfig.from_dict(data)

    def test_incomplete_answer_policy_is_explicit_and_validated(self) -> None:
        data = _minimal_config_dict()
        phase = data["protocol"]["phases"][0]
        phase["incomplete_answer_policy"] = "record_unavailable"
        self.assertEqual(
            ExperimentConfig.from_dict(data).protocol.phases[0].incomplete_answer_policy,
            "record_unavailable",
        )

        phase["incomplete_answer_policy"] = "ignore"
        with self.assertRaisesRegex(ConfigError, "incomplete_answer_policy"):
            ExperimentConfig.from_dict(data)

    def test_reuse_unavailable_answers_requires_boolean(self) -> None:
        data = _minimal_config_dict()
        phase = data["protocol"]["phases"][0]
        phase["reuse_unavailable_answers"] = True
        self.assertTrue(
            ExperimentConfig.from_dict(data).protocol.phases[0].reuse_unavailable_answers
        )

        phase["reuse_unavailable_answers"] = "yes"
        with self.assertRaisesRegex(ConfigError, "reuse_unavailable_answers"):
            ExperimentConfig.from_dict(data)

    def test_retry_unavailable_rounds_are_validated(self) -> None:
        data = _minimal_config_dict()
        phase = data["protocol"]["phases"][0]
        phase["reuse_unavailable_answers"] = True
        phase["retry_unavailable_rounds"] = [2, 3]
        self.assertEqual(
            ExperimentConfig.from_dict(data).protocol.phases[0].retry_unavailable_rounds,
            [2, 3],
        )

        phase["retry_unavailable_rounds"] = [2, 2]
        with self.assertRaisesRegex(ConfigError, "must not contain duplicates"):
            ExperimentConfig.from_dict(data)

        phase["retry_unavailable_rounds"] = [2]
        phase["reuse_unavailable_answers"] = False
        with self.assertRaisesRegex(ConfigError, "requires reuse_unavailable_answers"):
            ExperimentConfig.from_dict(data)

    def test_replay_source_targets_requires_boolean(self) -> None:
        data = json.loads(
            (ROOT / "examples" / "catalog_ladder50_sol.openrouter.json").read_text(
                encoding="utf-8"
            )
        )
        phase = data["protocol"]["phases"][0]
        phase["replay_source_targets"] = True
        phase["preauthored_probe_file"] = "source.jsonl"
        self.assertTrue(
            ExperimentConfig.from_dict(data).protocol.phases[0].replay_source_targets
        )

        phase["replay_source_targets"] = "yes"
        with self.assertRaisesRegex(ConfigError, "replay_source_targets"):
            ExperimentConfig.from_dict(data)

        phase["replay_source_targets"] = True
        phase.pop("preauthored_probe_file")
        with self.assertRaisesRegex(ConfigError, "requires preauthored_probe_file"):
            ExperimentConfig.from_dict(data)

    def test_loads_and_validates_provider_request_retries(self) -> None:
        data = _minimal_config_dict()
        data["providers"][0]["request_retries"] = 0
        config = ExperimentConfig.from_dict(data)
        self.assertEqual(config.providers["test"].request_retries, 0)

        data["providers"][0]["request_retries"] = -1
        with self.assertRaisesRegex(ConfigError, "request_retries must be non-negative"):
            ExperimentConfig.from_dict(data)

    def test_loads_provider_factory_and_adapter_options(self) -> None:
        data = _minimal_config_dict()
        data["providers"][0].update(
            {
                "kind": "external_api",
                "client_factory": "package.adapter:create_client",
                "options": {"deployment": "research"},
            }
        )

        provider = ExperimentConfig.from_dict(data).providers["test"]

        self.assertEqual(provider.client_factory, "package.adapter:create_client")
        self.assertEqual(provider.options, {"deployment": "research"})

        data["providers"][0]["client_factory"] = ""
        with self.assertRaisesRegex(ConfigError, "non-empty string"):
            ExperimentConfig.from_dict(data)

    def test_rejects_unknown_phase_kind(self) -> None:
        data = _minimal_config_dict()
        data["protocol"]["phases"][0]["kind"] = "auction"
        with self.assertRaises(ConfigError):
            ExperimentConfig.from_dict(data)

    def test_rejects_unknown_prompt_id(self) -> None:
        data = _minimal_config_dict()
        data["protocol"]["phases"][0]["prompt"] = "missing_prompt"
        with self.assertRaisesRegex(ConfigError, "unknown prompt"):
            ExperimentConfig.from_dict(data)

    def test_rejects_non_boolean_phase_flags(self) -> None:
        data = _minimal_config_dict()
        data["protocol"]["phases"][0]["require_json"] = "false"
        with self.assertRaisesRegex(ConfigError, "must be a boolean"):
            ExperimentConfig.from_dict(data)

    def test_rejects_unknown_response_visibility(self) -> None:
        data = _minimal_config_dict()
        data["protocol"]["phases"][0]["response_visibility"] = "hidden"
        with self.assertRaisesRegex(ConfigError, "response_visibility"):
            ExperimentConfig.from_dict(data)

    def test_rejects_response_visibility_on_non_matrix_phase(self) -> None:
        data = _minimal_config_dict()
        data["protocol"]["phases"][0]["response_visibility"] = "private"
        with self.assertRaisesRegex(ConfigError, "only used by public_test_matrix"):
            ExperimentConfig.from_dict(data)

    def test_rejects_duplicate_phase_names(self) -> None:
        data = _minimal_config_dict()
        data["protocol"]["phases"].append(copy.deepcopy(data["protocol"]["phases"][0]))
        with self.assertRaisesRegex(ConfigError, "phase names must be unique"):
            ExperimentConfig.from_dict(data)

    def test_rejects_unknown_phase_reference(self) -> None:
        data = _minimal_config_dict()
        data["protocol"]["phases"][0].update(
            {
                "kind": "public_test_matrix",
                "prompt": "test_answer",
                "source_phase": "missing_proposal",
            }
        )
        with self.assertRaisesRegex(ConfigError, "unknown source_phase"):
            ExperimentConfig.from_dict(data)

    def test_rejects_forward_phase_reference(self) -> None:
        data = _minimal_config_dict()
        data["protocol"]["phases"] = [
            {
                "name": "test_application",
                "kind": "public_test_matrix",
                "prompt": "test_answer",
                "visibility": "public",
                "source_phase": "public_test_proposal",
            },
            {
                "name": "public_test_proposal",
                "kind": "public_round_robin",
                "prompt": "public_test_proposal",
                "rounds": 1,
                "visibility": "public",
            },
        ]
        with self.assertRaisesRegex(ConfigError, "before that phase has run"):
            ExperimentConfig.from_dict(data)

    def test_rejects_private_source_phase_for_public_test_matrix(self) -> None:
        data = _minimal_config_dict()
        data["protocol"]["phases"] = [
            {
                "name": "private_test_design",
                "kind": "private_reflection",
                "prompt": "private_test_design",
                "visibility": "private",
            },
            {
                "name": "test_application",
                "kind": "public_test_matrix",
                "prompt": "test_answer",
                "visibility": "public",
                "source_phase": "private_test_design",
            },
        ]
        with self.assertRaisesRegex(ConfigError, "must be public"):
            ExperimentConfig.from_dict(data)

    def test_rejects_unimplemented_turn_order(self) -> None:
        data = _minimal_config_dict()
        data["protocol"]["turn_order"] = "random"
        with self.assertRaises(ConfigError):
            ExperimentConfig.from_dict(data)

    def test_loads_context_spec(self) -> None:
        data = _minimal_config_dict()
        data["context"] = {
            "mode": "private_memory",
            "max_public_turns": 2,
            "max_private_turns": 8,
        }
        config = ExperimentConfig.from_dict(data)
        self.assertEqual(config.context.mode, "private_memory")
        self.assertEqual(config.context.max_public_turns, 2)
        self.assertEqual(config.context.max_private_turns, 8)

    def test_loads_phase_model_params(self) -> None:
        data = _minimal_config_dict()
        data["protocol"]["phases"][0]["model_params"] = {
            "max_tokens": 1000,
            "reasoning": {"effort": "none"},
        }
        config = ExperimentConfig.from_dict(data)
        self.assertEqual(
            config.protocol.phases[0].model_params,
            {"max_tokens": 1000, "reasoning": {"effort": "none"}},
        )

    def test_validates_comparison_presentation_order(self) -> None:
        data = _minimal_config_dict()
        data["judges"] = [
            {"id": "J1", "model": "model", "system_prompt": "independent_intelligence_judge"}
        ]
        phase = data["protocol"]["phases"][0]
        phase.update(
            {
                "kind": "independent_judge_ranking",
                "prompt": "adaptive_judge_probe",
                "visibility": "private",
                "comparison_order": "seeded_shuffle",
                "comparison_seed": 17,
            }
        )

        config = ExperimentConfig.from_dict(data)
        self.assertEqual(config.protocol.phases[0].comparison_order, "seeded_shuffle")
        self.assertEqual(config.protocol.phases[0].comparison_seed, 17)

        phase["comparison_order"] = "random"
        with self.assertRaisesRegex(ConfigError, "comparison_order"):
            ExperimentConfig.from_dict(data)

    def test_loads_structured_json_retries(self) -> None:
        data = _minimal_config_dict()
        data["run"] = {"structured_json_retries": 2}
        config = ExperimentConfig.from_dict(data)
        self.assertEqual(config.run.structured_json_retries, 2)

    def test_rejects_negative_structured_json_retries(self) -> None:
        data = _minimal_config_dict()
        data["run"] = {"structured_json_retries": -1}
        with self.assertRaisesRegex(ConfigError, "structured_json_retries"):
            ExperimentConfig.from_dict(data)

    def test_rejects_unknown_context_mode(self) -> None:
        data = _minimal_config_dict()
        data["context"] = {"mode": "telepathy"}
        with self.assertRaisesRegex(ConfigError, "context mode"):
            ExperimentConfig.from_dict(data)

    def test_loads_saved_named_item_maps(self) -> None:
        data = _minimal_config_dict()
        data["providers"] = {"test": data["providers"][0]}
        data["models"] = {"model": data["models"][0]}
        config = ExperimentConfig.from_dict(data)
        self.assertIn("test", config.providers)
        self.assertIn("model", config.models)

    def test_validates_independent_judge_probe_prefixes(self) -> None:
        data = _minimal_config_dict()
        data["judges"] = [{"id": "J1", "model": "model"}]
        data["protocol"]["phases"] = [
            {
                "name": "judge_ranking",
                "kind": "independent_judge_ranking",
                "prompt": "independent_judge_probe",
                "visibility": "private",
                "probes_per_round": 6,
                "judgment_probe_counts": [2, 4, 6],
            }
        ]
        phase = ExperimentConfig.from_dict(data).protocol.phases[0]
        self.assertEqual(phase.judgment_probe_counts, [2, 4, 6])

        data["protocol"]["phases"][0]["judgment_probe_counts"] = [2, 7]
        with self.assertRaisesRegex(ConfigError, "cannot exceed probes_per_round"):
            ExperimentConfig.from_dict(data)

    def test_validates_adaptive_probe_schedule(self) -> None:
        data = _minimal_config_dict()
        data["judges"] = [{"id": "J1", "model": "model"}]
        data["protocol"]["phases"] = [
            {
                "name": "judge_ranking",
                "kind": "independent_judge_ranking",
                "prompt": "adaptive_judge_probe",
                "visibility": "private",
                "probe_schedule": [4, 1, 2],
                "adaptive_targeting": "judge_selected",
            }
        ]
        phase = ExperimentConfig.from_dict(data).protocol.phases[0]
        self.assertEqual(phase.rounds, 3)
        self.assertEqual(phase.probe_schedule, [4, 1, 2])

        data["protocol"]["phases"][0]["rounds"] = 2
        with self.assertRaisesRegex(ConfigError, "rounds must equal"):
            ExperimentConfig.from_dict(data)

        data["protocol"]["phases"][0]["rounds"] = 3
        data["protocol"]["phases"][0]["adaptive_targeting"] = "nearest_neighbors"
        with self.assertRaisesRegex(ConfigError, "adaptive_targeting"):
            ExperimentConfig.from_dict(data)


def _replay_row(role: str, stream_id: str, content: str) -> dict:
    return {
        "content": content,
        "metadata": {
            "interaction_role": role,
            "stream_id": stream_id,
        },
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _minimal_config_dict() -> dict:
    return copy.deepcopy(
        {
            "name": "test",
            "providers": [{"name": "test", "kind": "mock"}],
            "models": [{"name": "model", "provider": "test", "model": "test:model"}],
            "participants": [{"id": "P1", "model": "model"}],
            "protocol": {
                "name": "test_protocol",
                "phases": [
                    {
                        "name": "public_probe",
                        "kind": "public_round_robin",
                        "prompt": "opening_council",
                        "rounds": 1,
                        "visibility": "public",
                    }
                ],
            },
        }
    )


if __name__ == "__main__":
    unittest.main()

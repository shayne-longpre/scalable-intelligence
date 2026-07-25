from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from ai_council.config import ConfigError, ExperimentConfig, load_experiment_config
from scripts.build_catalog_ladder_config import _candidate_model, _expected_model_calls


ROOT = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    def test_catalog_ladder_call_budget_scales_with_roster_and_schedule(self) -> None:
        self.assertEqual(
            _expected_model_calls(
                candidate_count=50,
                probe_schedule=[4, 1, 1],
                max_adaptive_candidates=10,
            ),
            235,
        )
        self.assertEqual(
            _expected_model_calls(
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

        configured = _candidate_model(
            model,
            "P01",
            {"params": {"reasoning": {"effort": "low"}, "temperature": 0}},
        )

        self.assertEqual(configured["params"]["max_tokens"], 40000)
        self.assertEqual(configured["params"]["reasoning"], {"effort": "low"})
        self.assertEqual(configured["params"]["temperature"], 0)

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

        for filename, judge_route in [
            ("catalog_ladder50_sol.openrouter.json", "openai/gpt-5.6-sol"),
            ("catalog_ladder50_fable.openrouter.json", "anthropic/claude-fable-5"),
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
                self.assertEqual(phase.probe_schedule, [4, 1, 1])
                self.assertEqual(phase.comparison_order, "seeded_shuffle")
                self.assertEqual(phase.incomplete_answer_policy, "record_unavailable")
                self.assertTrue(config.run.continue_batch_on_call_error)
                self.assertEqual(
                    config.providers["openrouter_candidates"].timeout_seconds,
                    300,
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

    def test_oversight_frontier_configs_straddle_each_judge(self) -> None:
        study = json.loads(
            (ROOT / "studies" / "oversight_frontier_v1.json").read_text()
        )
        catalog = json.loads(
            (ROOT / "data" / "model_catalog.openrouter.json").read_text()
        )
        scores = {
            model["provider_model_id"]: model["intelligence_score"]
            for model in catalog["models"]
        }

        for condition in study["conditions"]:
            selection = json.loads((ROOT / condition["selection"]).read_text())
            routes = selection["provider_model_ids"]
            judge_route = condition["judge_model"]
            judge_score = condition["judge_external_score"]
            self.assertEqual(len(routes), 7)
            self.assertEqual(len(set(routes)), 7)
            self.assertIn(judge_route, routes)
            self.assertTrue(any(scores[route] > judge_score for route in routes))
            self.assertTrue(any(scores[route] < judge_score for route in routes))

            participant_seeds = set()
            comparison_seeds = set()
            for config_path in condition["configs"]:
                config = load_experiment_config(ROOT / config_path)
                phase = config.protocol.phases[0]
                configured_routes = {
                    config.models[participant.model].model
                    for participant in config.participants
                }
                configured_judge = config.models[config.judges[0].model].model
                self.assertEqual(configured_routes, set(routes))
                self.assertEqual(configured_judge, judge_route)
                self.assertEqual(phase.probe_schedule, [4, 1])
                self.assertEqual(phase.max_adaptive_candidates, 4)
                participant_seeds.add(config.metadata["participant_seed"])
                comparison_seeds.add(config.metadata["comparison_seed"])
            self.assertEqual(len(participant_seeds), 2)
            self.assertEqual(len(comparison_seeds), 2)

    def test_five_probe_catalog_configs_isolate_the_opening_battery(self) -> None:
        for filename in (
            "catalog_ladder50_sol_p5.openrouter.json",
            "catalog_ladder50_fable_p5.openrouter.json",
        ):
            config = load_experiment_config(ROOT / "examples" / filename)
            self.assertEqual(len(config.participants), 50)
            self.assertEqual(config.protocol.phases[0].probe_schedule, [5])

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

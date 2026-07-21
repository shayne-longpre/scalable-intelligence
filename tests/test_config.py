from __future__ import annotations

import copy
import unittest
from pathlib import Path

from ai_council.config import ConfigError, ExperimentConfig, load_experiment_config


ROOT = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
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

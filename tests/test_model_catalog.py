from __future__ import annotations

import json
import unittest

from ai_council.model_catalog import build_openrouter_catalog, parse_artificial_analysis_models_html


class ModelCatalogTests(unittest.TestCase):
    def test_parse_artificial_analysis_models_chooses_payload_with_scores(self) -> None:
        low_value_payload = json.dumps({"models": [{"name": "Search row"}]})
        scored_payload = json.dumps(
            {
                "models": [
                    {
                        "id": "aa-1",
                        "name": "Strong Model",
                        "slug": "strong-model",
                        "intelligenceIndex": 55.0,
                    }
                ]
            }
        )
        html = _script_payload(low_value_payload) + _script_payload(scored_payload)

        models = parse_artificial_analysis_models_html(html)

        self.assertEqual(models[0]["name"], "Strong Model")
        self.assertEqual(models[0]["intelligenceIndex"], 55.0)

    def test_build_catalog_prefers_reported_scores_over_metadata_fallback(self) -> None:
        openrouter_payload = {
            "data": [
                {
                    "id": "provider/new-unscored",
                    "name": "Provider: New Unscored",
                    "created": 1782276303,
                    "context_length": 1000000,
                    "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
                    "top_provider": {"max_completion_tokens": 8192},
                    "pricing": {"prompt": "0.1", "completion": "0.2"},
                },
                {
                    "id": "provider/strong",
                    "name": "Provider: Strong",
                    "created": 1700000000,
                    "context_length": 128000,
                    "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
                    "top_provider": {"max_completion_tokens": 4096},
                    "pricing": {},
                },
            ]
        }
        artificial_analysis_models = [
            {
                "id": "aa-strong",
                "name": "Strong",
                "slug": "strong",
                "releaseDate": "2026-01-01",
                "modelCreatorName": "Provider",
                "intelligenceIndex": 50.0,
                "intelligenceIndexIsEstimated": False,
                "openrouterApiId": "provider/strong",
            }
        ]

        catalog = build_openrouter_catalog(
            openrouter_payload,
            artificial_analysis_models,
            generated_at="2026-06-27T00:00:00+00:00",
        )
        models = catalog["models"]

        self.assertEqual(models[0]["provider_model_id"], "provider/strong")
        self.assertEqual(models[0]["estimated_rank"], 1)
        self.assertEqual(models[0]["ranking_basis"], "artificial_analysis_reported")
        self.assertEqual(models[0]["rank_confidence"], "high")
        self.assertEqual(models[1]["ranking_basis"], "openrouter_metadata_fallback")
        self.assertEqual(catalog["summary"]["ranked_by_reported_score_count"], 1)

    def test_build_catalog_can_match_artificial_analysis_slug_when_api_id_is_missing(self) -> None:
        openrouter_payload = {
            "data": [
                {
                    "id": "anthropic/claude-opus-4.7",
                    "name": "Anthropic: Claude Opus 4.7",
                    "canonical_slug": "claude-opus-4.7",
                    "created": 1780000000,
                    "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
                    "top_provider": {},
                    "pricing": {},
                }
            ]
        }
        artificial_analysis_models = [
            {
                "id": "aa-opus",
                "name": "Claude Opus 4.7",
                "slug": "claude-opus-4-7",
                "releaseDate": "2026-04-16",
                "modelCreatorName": "Anthropic",
                "intelligenceIndex": 53.5,
                "intelligenceIndexIsEstimated": False,
                "openrouterApiId": None,
            }
        ]

        catalog = build_openrouter_catalog(openrouter_payload, artificial_analysis_models)
        entry = catalog["models"][0]

        self.assertEqual(entry["ranking_basis"], "artificial_analysis_reported")
        self.assertEqual(entry["rank_confidence"], "medium")
        self.assertEqual(entry["matched_evals"][0]["match_type"], "slug_or_name")


def _script_payload(payload: str) -> str:
    compact_payload = json.dumps(json.loads(payload), separators=(",", ":"))
    escaped = json.dumps(compact_payload)[1:-1]
    return f'<script>self.__next_f.push([1,"{escaped}"])</script>'


if __name__ == "__main__":
    unittest.main()

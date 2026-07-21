from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, patch

import httpx

from ai_council.clients.anthropic import AnthropicClient
from ai_council.clients.base import ModelClientError
from ai_council.clients.http import post_json
from ai_council.clients.mock import MockModelClient
from ai_council.clients.openai_compatible import OpenAICompatibleClient, OpenRouterClient
from ai_council.clients.registry import build_client, register_client
from ai_council.config import ProviderSpec
from ai_council.core import ModelRequest


class ProviderClientTests(unittest.TestCase):
    def test_loads_external_client_factory_from_config(self) -> None:
        provider = ProviderSpec(
            name="external",
            kind="external_api",
            client_factory="ai_council.clients.mock:MockModelClient",
            options={"deployment": "test"},
        )

        client = build_client(provider)

        self.assertIsInstance(client, MockModelClient)
        self.assertEqual(client.provider.options, {"deployment": "test"})

    def test_registered_client_factory_supports_new_provider_kinds(self) -> None:
        kind = "unit-test-provider"
        register_client(kind, MockModelClient)

        client = build_client(ProviderSpec(name="custom", kind=kind))

        self.assertIsInstance(client, MockModelClient)
        with self.assertRaisesRegex(ValueError, "already registered"):
            register_client(kind, MockModelClient)

    def test_rejects_invalid_external_client_factory_path(self) -> None:
        provider = ProviderSpec(
            name="external",
            kind="external_api",
            client_factory="missing_separator",
        )

        with self.assertRaisesRegex(ValueError, "package.module:factory"):
            build_client(provider)

    def test_only_stateless_http_clients_opt_into_parallel_requests(self) -> None:
        self.assertTrue(
            OpenAICompatibleClient(ProviderSpec(name="openai", kind="openai"))
            .supports_parallel_requests
        )
        self.assertTrue(
            OpenRouterClient(ProviderSpec(name="openrouter", kind="openrouter"))
            .supports_parallel_requests
        )
        self.assertTrue(
            AnthropicClient(ProviderSpec(name="anthropic", kind="anthropic"))
            .supports_parallel_requests
        )
        self.assertFalse(
            MockModelClient(ProviderSpec(name="mock", kind="mock"))
            .supports_parallel_requests
        )

    def test_openrouter_default_url_preserves_provider_options(self) -> None:
        provider = ProviderSpec(
            name="openrouter",
            kind="openrouter",
            options={"routing_policy": "test"},
        )

        client = OpenRouterClient(provider)

        self.assertEqual(client.base_url, "https://openrouter.ai/api/v1")
        self.assertEqual(client.provider.options, {"routing_policy": "test"})

    def test_anthropic_client_splits_system_messages_and_preserves_params(self) -> None:
        provider = ProviderSpec(
            name="anthropic",
            kind="anthropic",
            api_key_env=None,
            request_retries=0,
        )
        client = AnthropicClient(provider)

        captured = {}

        def fake_post_json(**kwargs):
            captured.update(kwargs)
            return {
                "model": "claude-test",
                "content": [{"type": "text", "text": "ok"}],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }

        with patch("ai_council.clients.anthropic.post_json", fake_post_json):
            response = client.generate(
                ModelRequest(
                    model="claude-test",
                    messages=[
                        {"role": "system", "content": "System one."},
                        {"role": "system", "content": "System two."},
                        {"role": "user", "content": "Hello"},
                    ],
                    params={"max_tokens": 77, "temperature": 0.2},
                )
            )

        self.assertEqual(response.content, "ok")
        payload = captured["payload"]
        self.assertEqual(payload["system"], "System one.\n\nSystem two.")
        self.assertEqual(payload["messages"], [{"role": "user", "content": "Hello"}])
        self.assertEqual(payload["max_tokens"], 77)
        self.assertEqual(payload["temperature"], 0.2)
        self.assertEqual(captured["retries"], 0)

    def test_post_json_retries_transport_errors(self) -> None:
        request = AsyncMock(
            side_effect=[httpx.ConnectError("connection failed"), (200, '{"ok": true}')]
        )
        with patch("ai_council.clients.http._post_json_once", request), patch("time.sleep"):
            response = post_json(
                url="https://example.test",
                headers={},
                payload={"hello": "world"},
                timeout_seconds=1,
                retries=1,
            )

        self.assertEqual(response, {"ok": True})
        self.assertEqual(request.await_count, 2)

    def test_post_json_retries_truncated_provider_json(self) -> None:
        request = AsyncMock(
            side_effect=[(200, '{"choices": ['), (200, '{"ok": true}')]
        )
        with patch("ai_council.clients.http._post_json_once", request), patch("time.sleep"):
            response = post_json(
                url="https://example.test",
                headers={},
                payload={"hello": "world"},
                timeout_seconds=1,
                retries=1,
            )

        self.assertEqual(response, {"ok": True})
        self.assertEqual(request.await_count, 2)

    def test_post_json_retries_embedded_transient_provider_error(self) -> None:
        request = AsyncMock(
            side_effect=[
                (200, '{"error":{"message":"Internal Server Error","code":500}}'),
                (200, '{"ok":true}'),
            ]
        )
        with patch("ai_council.clients.http._post_json_once", request), patch("time.sleep"):
            response = post_json(
                url="https://example.test",
                headers={},
                payload={"hello": "world"},
                timeout_seconds=1,
                retries=1,
            )

        self.assertEqual(response, {"ok": True})
        self.assertEqual(request.await_count, 2)

    def test_post_json_does_not_retry_embedded_client_error(self) -> None:
        request = AsyncMock(
            return_value=(
                200,
                '{"error":{"message":"Invalid request","code":"400"}}',
            )
        )
        with patch("ai_council.clients.http._post_json_once", request), patch("time.sleep"):
            with self.assertRaisesRegex(ModelClientError, "provider error 400"):
                post_json(
                    url="https://example.test",
                    headers={},
                    payload={"hello": "world"},
                    timeout_seconds=1,
                    retries=2,
                )

        self.assertEqual(request.await_count, 1)

    def test_post_json_enforces_total_attempt_deadline(self) -> None:
        class SlowClient:
            def __init__(self, **_: object) -> None:
                pass

            async def __aenter__(self) -> "SlowClient":
                return self

            async def __aexit__(self, *_: object) -> None:
                return None

            async def post(self, *_: object, **__: object) -> object:
                await asyncio.sleep(0.1)
                raise AssertionError("deadline did not cancel request")

        with patch("ai_council.clients.http.httpx.AsyncClient", SlowClient):
            with self.assertRaisesRegex(ModelClientError, "failed"):
                post_json(
                    url="https://example.test",
                    headers={},
                    payload={"hello": "world"},
                    timeout_seconds=0.01,
                    retries=0,
                )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import os

from ai_council.clients.base import ModelClient, ModelClientError
from ai_council.clients.http import post_json
from ai_council.config import ProviderSpec
from ai_council.core import ModelRequest, ModelResponse


class AnthropicClient(ModelClient):
    supports_parallel_requests = True

    def __init__(self, provider: ProviderSpec):
        self.provider = provider
        self.base_url = (provider.base_url or "https://api.anthropic.com/v1").rstrip("/")

    def generate(self, request: ModelRequest) -> ModelResponse:
        params = dict(request.params)
        max_tokens = int(params.pop("max_tokens", 1024))
        system, messages = _split_system_messages(request.messages)
        payload = {
            "model": request.model,
            "max_tokens": max_tokens,
            "messages": messages,
            **params,
        }
        if system:
            payload["system"] = system

        data = post_json(
            url=f"{self.base_url}/messages",
            headers=self._headers(),
            payload=payload,
            timeout_seconds=self.provider.timeout_seconds,
            retries=self.provider.request_retries,
        )
        try:
            blocks = data["content"]
            content = "".join(block.get("text", "") for block in blocks if block.get("type") == "text")
        except (KeyError, TypeError) as exc:
            raise ModelClientError(f"unexpected Anthropic response shape: {data}") from exc
        return ModelResponse(
            content=content,
            raw=data,
            usage=data.get("usage", {}),
            model=data.get("model", request.model),
            provider=self.provider.name,
        )

    def _headers(self) -> dict[str, str]:
        headers = {"anthropic-version": "2023-06-01", **self.provider.headers}
        if self.provider.api_key_env:
            api_key = os.environ.get(self.provider.api_key_env)
            if not api_key:
                raise ModelClientError(f"missing API key env var: {self.provider.api_key_env}")
            headers["x-api-key"] = api_key
        return headers


def _split_system_messages(messages: list[dict[str, str]]) -> tuple[str | None, list[dict[str, str]]]:
    system_parts = [message["content"] for message in messages if message.get("role") == "system"]
    chat_messages = [
        {"role": message["role"], "content": message["content"]}
        for message in messages
        if message.get("role") in {"user", "assistant"}
    ]
    return ("\n\n".join(system_parts) if system_parts else None, chat_messages)

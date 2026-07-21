from __future__ import annotations

import os
from dataclasses import replace

from ai_council.clients.base import ModelClient, ModelClientError
from ai_council.clients.http import post_json
from ai_council.config import ProviderSpec
from ai_council.core import ModelRequest, ModelResponse


class OpenAICompatibleClient(ModelClient):
    supports_parallel_requests = True

    def __init__(self, provider: ProviderSpec):
        self.provider = provider
        self.base_url = (provider.base_url or "https://api.openai.com/v1").rstrip("/")

    def generate(self, request: ModelRequest) -> ModelResponse:
        payload = {
            "model": request.model,
            "messages": request.messages,
            **request.params,
        }
        data = post_json(
            url=f"{self.base_url}/chat/completions",
            headers=self._headers(),
            payload=payload,
            timeout_seconds=self.provider.timeout_seconds,
            retries=self.provider.request_retries,
        )
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ModelClientError(f"unexpected OpenAI-compatible response shape: {data}") from exc
        return ModelResponse(
            content=content or "",
            raw=data,
            usage=data.get("usage", {}),
            model=data.get("model", request.model),
            provider=self.provider.name,
        )

    def _headers(self) -> dict[str, str]:
        headers = dict(self.provider.headers)
        if self.provider.api_key_env:
            api_key = os.environ.get(self.provider.api_key_env)
            if not api_key:
                raise ModelClientError(f"missing API key env var: {self.provider.api_key_env}")
            headers["Authorization"] = f"Bearer {api_key}"
        return headers


class OpenRouterClient(OpenAICompatibleClient):
    def __init__(self, provider: ProviderSpec):
        if provider.base_url is None:
            provider = replace(provider, base_url="https://openrouter.ai/api/v1")
        super().__init__(provider)

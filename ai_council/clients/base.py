from __future__ import annotations

from abc import ABC, abstractmethod

from ai_council.core import ModelRequest, ModelResponse


class ModelClientError(RuntimeError):
    """Raised when a provider request fails or returns an unexpected response."""


class ModelClient(ABC):
    # Stateful clients remain serialized unless they explicitly opt in.
    supports_parallel_requests = False

    @abstractmethod
    def generate(self, request: ModelRequest) -> ModelResponse:
        raise NotImplementedError

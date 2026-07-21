from __future__ import annotations

from collections.abc import Callable
from importlib import import_module

from ai_council.clients.anthropic import AnthropicClient
from ai_council.clients.base import ModelClient
from ai_council.clients.mock import MockModelClient
from ai_council.clients.openai_compatible import OpenAICompatibleClient, OpenRouterClient
from ai_council.config import ExperimentConfig, ProviderSpec

ClientFactory = Callable[[ProviderSpec], ModelClient]

_CLIENT_FACTORIES: dict[str, ClientFactory] = {
    "mock": MockModelClient,
    "openai": OpenAICompatibleClient,
    "openai_compatible": OpenAICompatibleClient,
    "openrouter": OpenRouterClient,
    "anthropic": AnthropicClient,
    "claude": AnthropicClient,
}


def build_clients(config: ExperimentConfig) -> dict[str, ModelClient]:
    return {name: build_client(provider) for name, provider in config.providers.items()}


def build_client(provider: ProviderSpec) -> ModelClient:
    factory = (
        _load_factory(provider.client_factory)
        if provider.client_factory
        else _CLIENT_FACTORIES.get(_normalize_kind(provider.kind))
    )
    if factory is None:
        raise ValueError(
            f"unknown provider kind {provider.kind!r}; register it or set client_factory"
        )
    client = factory(provider)
    if not isinstance(client, ModelClient):
        raise TypeError(
            f"client factory for provider {provider.name!r} returned "
            f"{type(client).__name__}, expected ModelClient"
        )
    return client


def register_client(kind: str, factory: ClientFactory, *, replace: bool = False) -> None:
    normalized = _normalize_kind(kind)
    if not normalized:
        raise ValueError("provider kind must not be empty")
    if not callable(factory):
        raise TypeError("client factory must be callable")
    if normalized in _CLIENT_FACTORIES and not replace:
        raise ValueError(f"provider kind {kind!r} is already registered")
    _CLIENT_FACTORIES[normalized] = factory


def _load_factory(path: str) -> ClientFactory:
    module_name, separator, attribute_path = path.partition(":")
    if not separator or not module_name or not attribute_path:
        raise ValueError("client_factory must use 'package.module:factory' syntax")
    target: object = import_module(module_name)
    for attribute in attribute_path.split("."):
        target = getattr(target, attribute)
    if not callable(target):
        raise TypeError(f"client_factory {path!r} is not callable")
    return target


def _normalize_kind(kind: str) -> str:
    return kind.strip().lower().replace("-", "_")

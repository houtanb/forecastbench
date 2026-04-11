"""Central model registry for LLM providers."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Final, Type

from .lab_registry import LABS, Lab
from .providers.anthropic import AnthropicProvider
from .providers.base import BaseLLMProvider
from .providers.google import GoogleProvider
from .providers.openai import OpenAIProvider
from .providers.together import TogetherProvider
from .providers.xai import XAIProvider

# Registry for API keys by provider class
_PROVIDER_API_KEYS: dict[Type[BaseLLMProvider], str] = {}

# Mapping from provider name strings to provider classes
_PROVIDER_NAME_TO_CLASS: dict[str, Type[BaseLLMProvider]] = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "google": GoogleProvider,
    "xai": XAIProvider,
    "together": TogetherProvider,
}


@dataclass(frozen=True, slots=True)
class Model:
    """Registered LLM model metadata."""

    id: str
    full_name: str
    token_limit: int
    provider_cls: Type[BaseLLMProvider]
    lab: Lab
    rate_limit_group: str = ""

    def __post_init__(self) -> None:
        """Default rate_limit_group to the provider name if not set."""
        if not self.rate_limit_group:
            # Reverse-lookup the provider name from the class
            provider_name = next(
                (name for name, cls in _PROVIDER_NAME_TO_CLASS.items() if cls is self.provider_cls),
                self.provider_cls.__name__,
            )
            object.__setattr__(self, "rate_limit_group", provider_name)

    def get_response(self, prompt: str, **options: Any) -> str:
        """Request a response from the model's provider."""
        provider = _get_provider_instance(self.provider_cls)
        return provider.get_response(self, prompt, **options)


def _get_api_key_for_provider(provider_cls: Type[BaseLLMProvider]) -> str | None:
    """Look up API key for a provider from the registry configuration.

    Returns:
        API key string if configured, None otherwise.
    """
    return _PROVIDER_API_KEYS.get(provider_cls)


@lru_cache(maxsize=None)
def _get_provider_instance(provider_cls: Type[BaseLLMProvider]) -> BaseLLMProvider:
    """Return a cached provider instance for the given provider class."""
    api_key = _get_api_key_for_provider(provider_cls)
    if api_key is not None:
        return provider_cls(api_key=api_key)
    return provider_cls()


def configure_api_keys(
    *,
    openai: str | None = None,
    anthropic: str | None = None,
    google: str | None = None,
    xai: str | None = None,
    together: str | None = None,
) -> None:
    """Configure API keys for LLM providers.

    Args:
        openai: OpenAI API key (e.g., "sk-...")
        anthropic: Anthropic API key (e.g., "sk-ant-...")
        google: Google Gemini API key
        xai: xAI API key
        together: Together AI API key
    """
    key_mapping = {
        "openai": (OpenAIProvider, openai),
        "anthropic": (AnthropicProvider, anthropic),
        "google": (GoogleProvider, google),
        "xai": (XAIProvider, xai),
        "together": (TogetherProvider, together),
    }

    for provider_cls, api_key in key_mapping.values():
        if api_key is not None:
            _PROVIDER_API_KEYS[provider_cls] = api_key

    # Clear the provider instance cache since keys have changed
    _get_provider_instance.cache_clear()


def validate_provider_keys(models: list[Model]) -> None:
    """Validate that all providers needed by the given models have API keys configured.

    Args:
        models: List of Model objects to validate.

    Raises:
        ValueError: If any model's provider lacks a configured API key.
    """
    missing_keys = []
    provider_names = {cls: name for name, cls in _PROVIDER_NAME_TO_CLASS.items()}

    for model in models:
        provider_cls = model.provider_cls
        if provider_cls not in _PROVIDER_API_KEYS:
            provider_name = provider_names.get(provider_cls, provider_cls.__name__)
            missing_keys.append(f"{provider_name} (for model {model.id})")

    if missing_keys:
        missing_list = ", ".join(missing_keys)
        raise ValueError(
            f"API keys not configured for the following providers: {missing_list}. "
            "Call configure_api_keys() to set them."
        )


MODELS: Final[list[Model]] = [
    Model(
        id="gpt-4.1-mini",
        full_name="gpt-4.1-mini",
        token_limit=128_000,
        provider_cls=OpenAIProvider,
        lab=LABS["OpenAI"],
    ),
    Model(
        id="gpt-4o-mini",
        full_name="gpt-4o-mini",
        token_limit=128_000,
        provider_cls=OpenAIProvider,
        lab=LABS["OpenAI"],
    ),
    Model(
        id="gpt-5-2025-08-07",
        full_name="gpt-5-2025-08-07",
        token_limit=128_000,
        provider_cls=OpenAIProvider,
        lab=LABS["OpenAI"],

    ),
    Model(
        id="gpt-5-mini-2025-08-07",
        full_name="gpt-5-mini-2025-08-07",
        token_limit=128_000,
        provider_cls=OpenAIProvider,
        lab=LABS["OpenAI"],

    ),
    Model(
        id="gpt-5-nano-2025-08-07",
        full_name="gpt-5-nano-2025-08-07",
        token_limit=128_000,
        provider_cls=OpenAIProvider,
        lab=LABS["OpenAI"],

    ),
    Model(
        id="gpt-5.1-2025-11-13",
        full_name="gpt-5.1-2025-11-13",
        token_limit=128_000,
        provider_cls=OpenAIProvider,
        lab=LABS["OpenAI"],

    ),
    Model(
        id="gpt-5.2-2025-12-11",
        full_name="gpt-5.2-2025-12-11",
        token_limit=128_000,
        provider_cls=OpenAIProvider,
        lab=LABS["OpenAI"],

    ),
    Model(
        id="o3-2025-04-16",
        full_name="o3-2025-04-16",
        token_limit=200_000,
        provider_cls=OpenAIProvider,
        lab=LABS["OpenAI"],

    ),
    Model(
        id="gpt-4.1-2025-04-14",
        full_name="gpt-4.1-2025-04-14",
        token_limit=128_000,
        provider_cls=OpenAIProvider,
        lab=LABS["OpenAI"],
    ),
    Model(
        id="DeepSeek-V3.1",
        full_name="deepseek-ai/DeepSeek-V3.1",
        token_limit=128_000,
        provider_cls=TogetherProvider,
        lab=LABS["DeepSeek"],
    ),
    Model(
        id="Qwen3-235B-A22B-Thinking-2507",
        full_name="Qwen/Qwen3-235B-A22B-Thinking-2507",
        token_limit=262_144,
        provider_cls=TogetherProvider,
        lab=LABS["Qwen"],
    ),
    Model(
        id="GLM-4.5-Air-FP8",
        full_name="zai-org/GLM-4.5-Air-FP8",
        token_limit=131_072,
        provider_cls=TogetherProvider,
        lab=LABS["Z.ai"],
    ),
    Model(
        id="GLM-4.6",
        full_name="zai-org/GLM-4.6",
        token_limit=202_752,
        provider_cls=TogetherProvider,
        lab=LABS["Z.ai"],

    ),
    Model(
        id="claude-sonnet-4-5-20250929",
        full_name="claude-sonnet-4-5-20250929",
        token_limit=200_000,
        provider_cls=AnthropicProvider,
        lab=LABS["Anthropic"],
    ),
    Model(
        id="claude-haiku-4-5-20251001",
        full_name="claude-haiku-4-5-20251001",
        token_limit=200_000,
        provider_cls=AnthropicProvider,
        lab=LABS["Anthropic"],
    ),
    Model(
        id="claude-opus-4-1-20250805",
        full_name="claude-opus-4-1-20250805",
        token_limit=200_000,
        provider_cls=AnthropicProvider,
        lab=LABS["Anthropic"],
    ),
    Model(
        id="claude-opus-4-5-20251101",
        full_name="claude-opus-4-5-20251101",
        token_limit=200_000,
        provider_cls=AnthropicProvider,
        lab=LABS["Anthropic"],
    ),
    Model(
        id="claude-sonnet-4-6",
        full_name="claude-sonnet-4-6",
        token_limit=200_000,
        provider_cls=AnthropicProvider,
        lab=LABS["Anthropic"],
    ),
    Model(
        id="claude-sonnet-4-20250514",
        full_name="claude-sonnet-4-20250514",
        token_limit=200_000,
        provider_cls=AnthropicProvider,
        lab=LABS["Anthropic"],
    ),
    Model(
        id="grok-4-fast-reasoning",
        full_name="grok-4-fast-reasoning",
        token_limit=2_000_000,
        provider_cls=XAIProvider,
        lab=LABS["xAI"],
    ),
    Model(
        id="grok-4-fast-non-reasoning",
        full_name="grok-4-fast-non-reasoning",
        token_limit=2_000_000,
        provider_cls=XAIProvider,
        lab=LABS["xAI"],
    ),
    Model(
        id="grok-4-0709",
        full_name="grok-4-0709",
        token_limit=256_000,
        provider_cls=XAIProvider,
        lab=LABS["xAI"],
    ),
    Model(
        id="grok-4-1-fast-reasoning",
        full_name="grok-4-1-fast-reasoning",
        token_limit=2_000_000,
        provider_cls=XAIProvider,
        lab=LABS["xAI"],

    ),
    Model(
        id="grok-4-1-fast-non-reasoning",
        full_name="grok-4-1-fast-non-reasoning",
        token_limit=2_000_000,
        provider_cls=XAIProvider,
        lab=LABS["xAI"],

    ),
    Model(
        id="gemini-2.5-pro",
        full_name="gemini-2.5-pro",
        token_limit=1_048_576,
        provider_cls=GoogleProvider,
        lab=LABS["Google"],
    ),
    Model(
        id="gemini-2.5-flash",
        full_name="models/gemini-2.5-flash",
        token_limit=1_048_576,
        provider_cls=GoogleProvider,
        lab=LABS["Google"],
    ),
    Model(
        id="gemini-3-pro-preview",
        full_name="gemini-3-pro-preview",
        token_limit=1_048_576,
        provider_cls=GoogleProvider,
        lab=LABS["Google"],

    ),
]

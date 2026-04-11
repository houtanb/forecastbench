"""LLM model registry backed by utils_llm providers."""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .utils_llm.model_registry import (
    Model,
    configure_api_keys as _utils_configure_api_keys,
)
from .utils_llm.lab_registry import LABS
from .utils_llm.providers.openai import OpenAIProvider
from .utils_llm.providers.anthropic import AnthropicProvider
from .utils_llm.providers.google import GoogleProvider
from .utils_llm.providers.together import TogetherProvider
from .utils_llm.providers.xai import XAIProvider

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provider enum — maps to utils_llm provider classes
# ---------------------------------------------------------------------------

_PROVIDER_TO_CLS = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "google": GoogleProvider,
    "xai": XAIProvider,
    "together": TogetherProvider,
}


class Provider(Enum):
    """LLM provider organizations."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    XAI = "xai"
    TOGETHER = "together"


# ---------------------------------------------------------------------------
# Rate limits — max concurrent requests per group
# ---------------------------------------------------------------------------

RATE_LIMITS: dict[str, int] = {
    "openai": 10,
    "anthropic": 10,
    "google": 10,
    "xai": 10,
    "together": 10,
}

# ---------------------------------------------------------------------------
# ModelRun dataclass
# ---------------------------------------------------------------------------


@dataclass
class ModelRun:
    """Configuration for running an LLM with specific options.

    Attributes:
        name: Display name for this configuration.
        model_id: Provider model identifier.
        provider: The provider enum value.
        org: Organization name (for leaderboard display).
        options: Default options passed to the provider SDK.
        rate_limit_group: Concurrency group (defaults to provider.value).
    """

    name: str
    model_id: str
    provider: Provider
    org: str
    options: dict[str, Any] = field(default_factory=dict)
    rate_limit_group: str = ""

    def __post_init__(self) -> None:
        """Default rate_limit_group to provider.value."""
        if not self.rate_limit_group:
            self.rate_limit_group = self.provider.value

    @property
    def id(self) -> str:
        """Return the name as the identifier."""
        return self.name

    def _get_model(self) -> Model:
        """Create a utils_llm Model for this run."""
        provider_cls = _PROVIDER_TO_CLS[self.provider.value]
        return Model(
            id=self.name,
            full_name=self.model_id,
            token_limit=0,
            provider_cls=provider_cls,
            lab=LABS.get(self.org, LABS.get("OpenAI")),
        )

    def get_response(self, prompt: str, **kwargs: Any) -> str:
        """Request a response, merging self.options with call-time kwargs."""
        merged = {**self.options, **kwargs}
        model = self._get_model()
        return model.get_response(prompt, **merged)

    def __repr__(self) -> str:
        """Return a string representation."""
        if self.options:
            return f"<ModelRun {self.name} ({self.model_id}) {self.options}>"
        return f"<ModelRun {self.name}>"


# ---------------------------------------------------------------------------
# configure_keys — bridge to utils_llm
# ---------------------------------------------------------------------------


def configure_keys() -> None:
    """Configure API keys for all LLM providers using helpers.keys."""
    from . import keys

    _utils_configure_api_keys(
        openai=keys.API_KEY_OPENAI,
        anthropic=keys.API_KEY_ANTHROPIC,
        google=keys.API_KEY_GOOGLE,
        xai=keys.API_KEY_XAI,
        together=keys.API_KEY_TOGETHERAI,
    )


# ---------------------------------------------------------------------------
# MODEL_RUNS — all model configurations
# ---------------------------------------------------------------------------

# OpenAI: https://platform.openai.com/docs/models/
OPENAI_RUNS = [
    ModelRun(
        name="gpt-4o-mini-2024-07-18",
        model_id="gpt-4o-mini",
        provider=Provider.OPENAI,
        org="OpenAI",
        options={"temperature": 0},
    ),
    ModelRun(
        name="gpt-4.1-2025-04-14",
        model_id="gpt-4.1-2025-04-14",
        provider=Provider.OPENAI,
        org="OpenAI",
        options={"temperature": 0},
    ),
    ModelRun(
        name="gpt-4.1-mini-2025-04-14",
        model_id="gpt-4.1-mini",
        provider=Provider.OPENAI,
        org="OpenAI",
        options={"temperature": 0},
    ),
    ModelRun(
        name="o3-2025-04-16",
        model_id="o3-2025-04-16",
        provider=Provider.OPENAI,
        org="OpenAI",
    ),
    ModelRun(
        name="gpt-5-nano-2025-08-07",
        model_id="gpt-5-nano-2025-08-07",
        provider=Provider.OPENAI,
        org="OpenAI",
    ),
    ModelRun(
        name="gpt-5-mini-2025-08-07",
        model_id="gpt-5-mini-2025-08-07",
        provider=Provider.OPENAI,
        org="OpenAI",
    ),
    ModelRun(
        name="gpt-5.2-2025-12-11-low",
        model_id="gpt-5.2-2025-12-11",
        provider=Provider.OPENAI,
        org="OpenAI",
        options={"reasoning_effort": "low"},
    ),
    ModelRun(
        name="gpt-5.2-2025-12-11-medium",
        model_id="gpt-5.2-2025-12-11",
        provider=Provider.OPENAI,
        org="OpenAI",
        options={"reasoning_effort": "medium"},
    ),
    ModelRun(
        name="gpt-5.2-2025-12-11-high",
        model_id="gpt-5.2-2025-12-11",
        provider=Provider.OPENAI,
        org="OpenAI",
        options={"reasoning_effort": "high"},
    ),
    ModelRun(
        name="gpt-5.2-2025-12-11-high-web-search",
        model_id="gpt-5.2-2025-12-11",
        provider=Provider.OPENAI,
        org="OpenAI",
        options={
            "reasoning_effort": "high",
            "tools": [{"type": "web_search"}],
        },
    ),
]

# Together.ai: https://docs.together.ai/docs/serverless-models
TOGETHER_RUNS = [
    ModelRun(
        name="DeepSeek-V3.1",
        model_id="deepseek-ai/DeepSeek-V3.1",
        provider=Provider.TOGETHER,
        org="DeepSeek",
        options={"temperature": 0},
    ),
    ModelRun(
        name="Kimi-K2-Instruct-0905",
        model_id="moonshotai/Kimi-K2-Instruct-0905",
        provider=Provider.TOGETHER,
        org="Moonshot",
        options={"temperature": 0},
    ),
]

# Anthropic: https://platform.claude.com/docs/en/about-claude/models/overview
ANTHROPIC_RUNS = [
    ModelRun(
        name="claude-3-7-sonnet-20250219",
        model_id="claude-3-7-sonnet-20250219",
        provider=Provider.ANTHROPIC,
        org="Anthropic",
        options={"max_tokens": 4096, "temperature": 0},
    ),
    ModelRun(
        name="claude-haiku-4-5-20251001",
        model_id="claude-haiku-4-5-20251001",
        provider=Provider.ANTHROPIC,
        org="Anthropic",
        options={"max_tokens": 4096, "temperature": 0},
    ),
    ModelRun(
        name="claude-sonnet-4-20250514",
        model_id="claude-sonnet-4-20250514",
        provider=Provider.ANTHROPIC,
        org="Anthropic",
        options={"max_tokens": 4096, "temperature": 0},
    ),
    ModelRun(
        name="claude-sonnet-4-5-20250929",
        model_id="claude-sonnet-4-5-20250929",
        provider=Provider.ANTHROPIC,
        org="Anthropic",
        options={"max_tokens": 4096, "temperature": 0},
    ),
    ModelRun(
        name="claude-opus-4-1-20250805",
        model_id="claude-opus-4-1-20250805",
        provider=Provider.ANTHROPIC,
        org="Anthropic",
        options={"max_tokens": 4096, "temperature": 0},
    ),
    ModelRun(
        name="claude-opus-4-5-20251101",
        model_id="claude-opus-4-5-20251101",
        provider=Provider.ANTHROPIC,
        org="Anthropic",
        options={"max_tokens": 4096, "temperature": 0},
    ),
    ModelRun(
        name="claude-opus-4-5-20251101-web-search",
        model_id="claude-opus-4-5-20251101",
        provider=Provider.ANTHROPIC,
        org="Anthropic",
        options={
            "max_tokens": 4096,
            "temperature": 0,
            "tools": [{"type": "web_search_20250305", "name": "web_search", "max_uses": 5}],
        },
    ),
]

# xAI: https://console.x.ai/
XAI_RUNS = [
    ModelRun(
        name="grok-4-fast-reasoning",
        model_id="grok-4-fast-reasoning",
        provider=Provider.XAI,
        org="xAI",
        options={"temperature": 0},
    ),
    ModelRun(
        name="grok-4-fast-non-reasoning",
        model_id="grok-4-fast-non-reasoning",
        provider=Provider.XAI,
        org="xAI",
        options={"temperature": 0},
    ),
    ModelRun(
        name="grok-4-1-fast-reasoning",
        model_id="grok-4-1-fast-reasoning",
        provider=Provider.XAI,
        org="xAI",
        options={"temperature": 0},
    ),
    ModelRun(
        name="grok-4-1-fast-non-reasoning",
        model_id="grok-4-1-fast-non-reasoning",
        provider=Provider.XAI,
        org="xAI",
        options={"temperature": 0},
    ),
]

# Google: https://ai.google.dev/gemini-api/docs/models
GOOGLE_RUNS = [
    ModelRun(
        name="gemini-2.5-pro",
        model_id="gemini-2.5-pro",
        provider=Provider.GOOGLE,
        org="Google",
        options={"temperature": 0},
    ),
    ModelRun(
        name="gemini-2.5-pro-web-search",
        model_id="gemini-2.5-pro",
        provider=Provider.GOOGLE,
        org="Google",
        options={"temperature": 0, "tools": [{"googleSearch": {}}]},
    ),
    ModelRun(
        name="gemini-2.5-flash",
        model_id="models/gemini-2.5-flash",
        provider=Provider.GOOGLE,
        org="Google",
        options={"temperature": 0},
    ),
    ModelRun(
        name="gemini-3-flash-preview",
        model_id="gemini-3-flash-preview",
        provider=Provider.GOOGLE,
        org="Google",
    ),
    ModelRun(
        name="gemini-3-pro-preview",
        model_id="gemini-3-pro-preview",
        provider=Provider.GOOGLE,
        org="Google",
    ),
]

MODEL_RUNS: list[ModelRun] = (
    OPENAI_RUNS + TOGETHER_RUNS + ANTHROPIC_RUNS + XAI_RUNS + GOOGLE_RUNS
)

# Validation: ensure no duplicate names
_model_run_names = [m.name for m in MODEL_RUNS]
if len(_model_run_names) != len(set(_model_run_names)):
    from collections import Counter

    _duplicates = [name for name, count in Counter(_model_run_names).items() if count > 1]
    raise ValueError(f"Duplicate ModelRun names found: {_duplicates}")

# Model used for reformatting unparseable responses
REFORMAT_MODEL = ModelRun(
    name="gpt-5-mini-2025-08-07-reformat",
    model_id="gpt-5-mini-2025-08-07",
    provider=Provider.OPENAI,
    org="OpenAI",
)

# Model used for getting metadata (validating and categorizing questions)
METADATA_MODEL = ModelRun(
    name="gpt-5-mini-2025-08-07-metadata",
    model_id="gpt-5-mini-2025-08-07",
    provider=Provider.OPENAI,
    org="OpenAI",
)

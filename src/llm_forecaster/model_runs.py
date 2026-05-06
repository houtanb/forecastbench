"""ForecastBench selected shared LLM model runs."""

import logging
import os
from typing import Sequence

from utils.helpers.constants import (
    ANTHROPIC_API_KEY_SECRET_NAME,
    GOOGLE_GEMINI_API_KEY_SECRET_NAME,
    OPENAI_API_KEY_SECRET_NAME,
    TOGETHER_API_KEY_SECRET_NAME,
    XAI_API_KEY_SECRET_NAME,
)
from utils.llm import model_runs as shared_model_runs
from utils.llm.provider_registry import PROVIDERS, Provider

logger = logging.getLogger(__name__)

ModelRun = shared_model_runs.ModelRun

FORECASTBENCH_MODEL_RUN_KEYS = [
    "gpt-5.5-2026-04-23-run-variant-01",
    "gpt-5.4-2026-03-05-run-variant-01",
    "gpt-5.4-mini-2026-03-17-run-variant-01",
    "gpt-5.4-nano-2026-03-17-run-variant-01",
    "gpt-5-mini-2025-08-07-run-variant-01",
    "gpt-5-nano-2025-08-07-run-variant-01",
    "deepseek-v4-pro-run-variant-01",
    "minimax-m2.7-run-variant-01",
    "kimi-k2.6-run-variant-01",
    "glm-5.1-run-variant-01",
    "gemma-4-31b-run-variant-01",
    "claude-haiku-4-5-20251001-run-variant-01",
    "claude-sonnet-4-5-20250929-run-variant-01",
    "claude-sonnet-4-6-run-variant-01",
    "claude-opus-4-7-run-variant-03",
    "grok-4.20-0309-reasoning-run-variant-01",
    "grok-4.20-0309-non-reasoning-run-variant-01",
    "grok-4.3-run-variant-01",
    "gemini-3.5-flash-run-variant-01",
    "gemini-3.1-pro-preview-run-variant-01",
    "gemini-3.1-flash-lite-run-variant-01",
]

MODEL_RUNS = shared_model_runs.select_model_runs(FORECASTBENCH_MODEL_RUN_KEYS)
MODEL_RUNS_BY_KEY = {run.model_run_key: run for run in MODEL_RUNS}
MODEL_RUNS_BY_SLUG = {run.slug: run for run in MODEL_RUNS}
REFORMAT_MODEL = shared_model_runs.get_model_run("gpt-5-mini-2025-08-07-run-variant-02")


PROVIDER_MAX_WORKERS = {
    PROVIDERS["OpenAI"]: 16,
    PROVIDERS["Anthropic"]: 4,
    PROVIDERS["Google"]: 8,
    PROVIDERS["xAI"]: 8,
    PROVIDERS["Together"]: 8,
}


PROVIDER_API_KEY_CONFIG = {
    PROVIDERS["OpenAI"]: ("openai", OPENAI_API_KEY_SECRET_NAME),
    PROVIDERS["Anthropic"]: ("anthropic", ANTHROPIC_API_KEY_SECRET_NAME),
    PROVIDERS["Google"]: ("google", GOOGLE_GEMINI_API_KEY_SECRET_NAME),
    PROVIDERS["xAI"]: ("xai", XAI_API_KEY_SECRET_NAME),
    PROVIDERS["Together"]: ("together", TOGETHER_API_KEY_SECRET_NAME),
}


def get_model_run(model_run_key: str) -> ModelRun:
    """Return a ForecastBench model run by immutable key; prefer this for stable references."""
    try:
        return MODEL_RUNS_BY_KEY[model_run_key]
    except KeyError as exc:
        raise KeyError(f"Unknown ForecastBench LLM model run key: {model_run_key}") from exc


def get_model_run_by_slug(slug: str) -> ModelRun:
    """Return a ForecastBench model run by slug; prefer get_model_run when possible."""
    try:
        return MODEL_RUNS_BY_SLUG[slug]
    except KeyError as exc:
        raise KeyError(f"Unknown ForecastBench LLM model run slug: {slug}") from exc


def providers_for_model_runs(model_runs: Sequence[ModelRun]) -> list[Provider]:
    """Return unique providers required for the requested model runs."""
    providers = []
    for run in list(model_runs) + [REFORMAT_MODEL]:
        if run.provider not in providers:
            providers.append(run.provider)
    return providers


def _api_key_kwargs_for_providers(providers: Sequence[Provider]) -> dict[str, str]:
    """Return configure_api_keys kwargs for the required providers only."""
    from utils import gcp

    api_key_kwargs = {}
    for provider in providers:
        kwarg_name, secret_name = PROVIDER_API_KEY_CONFIG[provider]
        api_key = os.getenv(secret_name)
        if api_key is None:
            logger.info(
                "Loading %s API key from GCP Secret Manager secret %s.",
                provider.name,
                secret_name,
            )
            api_key = gcp.get_secret(secret_name)
        else:
            logger.info(
                "Using %s API key from environment variable %s.", provider.name, secret_name
            )
        api_key_kwargs[kwarg_name] = api_key
    return api_key_kwargs


def configure_and_validate_provider_keys(model_runs: Sequence[ModelRun]) -> None:
    """Configure provider keys from GCP and validate all required providers."""
    from utils.llm.model_registry import configure_api_keys, validate_provider_keys

    providers = providers_for_model_runs(model_runs)
    configure_api_keys(**_api_key_kwargs_for_providers(providers))
    validate_provider_keys(providers)

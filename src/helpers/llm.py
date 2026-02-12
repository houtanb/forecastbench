import logging
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from pprint import pformat
from typing import Any

import litellm

from . import env, keys

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class Lab(Enum):
    """LLM provider organizations."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    XAI = "xai"
    TOGETHER = "together"
    MISTRAL = "mistral"


LAB_API_KEYS = {
    Lab.OPENAI: keys.API_KEY_OPENAI,
    Lab.ANTHROPIC: keys.API_KEY_ANTHROPIC,
    Lab.GOOGLE: keys.API_KEY_GOOGLE,
    Lab.XAI: keys.API_KEY_XAI,
    Lab.TOGETHER: keys.API_KEY_TOGETHERAI,
    Lab.MISTRAL: keys.API_KEY_MISTRAL,
}

LAB_MAX_WORKERS = {
    Lab.OPENAI: 10,
    Lab.ANTHROPIC: 10,
    Lab.GOOGLE: 10,
    Lab.XAI: 10,
    Lab.TOGETHER: 10,
    Lab.MISTRAL: 10,
}


def validate_api_key(lab: Lab) -> None:
    """Raise if the API key for the given lab is not configured."""
    key = LAB_API_KEYS.get(lab)
    if not key:
        raise ValueError(f"No API key configured for {lab.name}")


LLM_NUM_RETRIES = 3
_LITELLM_LOG_PROMPT_ENV = "LITELLM_LOG_PROMPT"
_LITELLM_LOG_FILTER_INSTALLED = False


def _should_log_litellm_payload() -> bool:
    return env.RUNNING_LOCALLY or env.TESTING


def _should_log_litellm_prompt() -> bool:
    return os.getenv(_LITELLM_LOG_PROMPT_ENV, "").lower() in {"1", "true", "yes"}


def _redact_prompt_text(text: str) -> str:
    redacted = re.sub(r'("content"\s*:\s*")(.*?)(")', r"\1<redacted>\3", text)
    return re.sub(r"('content'\s*:\s*')(.*?)(')", r"\1<redacted>\3", redacted)


def _is_litellm_log_record(record: logging.LogRecord) -> bool:
    name = record.name.lower()
    if "litellm" in name:
        return True
    message = record.getMessage()
    return "LiteLLM" in message or "litellm" in message


class _LiteLLMRedactFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if _should_log_litellm_prompt() or not _is_litellm_log_record(record):
            return True
        message = record.getMessage()
        record.msg = _redact_prompt_text(message)
        record.args = ()
        return True


def _ensure_litellm_log_filter() -> None:
    global _LITELLM_LOG_FILTER_INSTALLED
    if _LITELLM_LOG_FILTER_INSTALLED:
        return
    logging.getLogger().addFilter(_LiteLLMRedactFilter())
    _LITELLM_LOG_FILTER_INSTALLED = True


def _summarize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summarized = []
    for message in messages:
        content = message.get("content", "")
        summarized.append(
            {
                "role": message.get("role", "unknown"),
                "content_len": len(str(content)),
            }
        )
    return summarized


def _redact_payload(value: Any) -> Any:
    sensitive_keys = {"api_key", "apikey", "api-key", "authorization", "x-api-key", "x_api_key"}
    text_keys = {"content", "text", "prompt"}
    list_summary_keys = {"messages", "input"}
    if isinstance(value, dict):
        return {
            k: (
                "***"
                if k.lower() in sensitive_keys
                else (
                    f"<redacted len={len(str(v))}>"
                    if (k in text_keys and not _should_log_litellm_prompt())
                    else (
                        _summarize_messages(v)
                        if (
                            k in list_summary_keys
                            and isinstance(v, list)
                            and not _should_log_litellm_prompt()
                        )
                        else (
                            f"<redacted len={len(v)}>"
                            if (
                                k == "contents"
                                and isinstance(v, list)
                                and not _should_log_litellm_prompt()
                            )
                            else (
                                _summarize_messages(v)
                                if (
                                    k == "messages"
                                    and isinstance(v, list)
                                    and not _should_log_litellm_prompt()
                                )
                                else _redact_payload(v)
                            )
                        )
                    )
                )
            )
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_redact_payload(item) for item in value]
    return value


def _litellm_payload_logger(model_call_dict: dict) -> None:
    logger.info(
        "LiteLLM payload:\n%s",
        _format_payload_for_log(_redact_payload(model_call_dict)),
    )


def _format_payload_for_log(payload: Any) -> str:
    return pformat(payload, sort_dicts=True, width=120)


@dataclass
class ModelRun:
    """
    Configuration for running an LLM with specific options.

    Attributes:
        name: Display name for this configuration (e.g., "gpt-5-high-reasoning").
        model_id: LiteLLM model identifier (e.g., "gpt-5", "anthropic/claude-3-opus").
        lab: The organization that created the model.
        options: Additional options passed to litellm.completion.
    """

    name: str
    model_id: str
    lab: Lab
    options: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        """Return the name as the identifier."""
        return self.name

    def __repr__(self) -> str:
        """Return a string representation of the ModelRun."""
        return (
            f"<ModelRun {self.name} ({self.model_id}) {self.options}>"
            if self.options
            else f"<ModelRun {self.name}>"
        )

    def get_response(self, prompt: str, **kwargs) -> str:
        """Request a response from the model via LiteLLM."""
        if not _should_log_litellm_prompt():
            _ensure_litellm_log_filter()
        if _should_log_litellm_payload() and "logger_fn" not in kwargs:
            kwargs["logger_fn"] = _litellm_payload_logger
        response = litellm.completion(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}],
            api_key=LAB_API_KEYS[self.lab],
            num_retries=LLM_NUM_RETRIES,
            **{**self.options, **kwargs},
        )
        return response.choices[0].message.content


# Model runs grouped by lab
# OpenAI: https://platform.openai.com/docs/models/
OPENAI_RUNS = [
    ModelRun(
        name="gpt-4o-mini-2024-07-18",
        model_id="gpt-4o-mini",
        lab=Lab.OPENAI,
        options={"temperature": 0},
    ),
    ModelRun(
        name="gpt-4.1-2025-04-14",
        model_id="gpt-4.1-2025-04-14",
        lab=Lab.OPENAI,
        options={"temperature": 0},
    ),
    ModelRun(
        name="gpt-4.1-mini-2025-04-14",
        model_id="gpt-4.1-mini",
        lab=Lab.OPENAI,
        options={"temperature": 0},
    ),
    ModelRun(
        name="o3-2025-04-16",
        model_id="o3-2025-04-16",
        lab=Lab.OPENAI,
    ),
    ModelRun(
        name="gpt-5-nano-2025-08-07",
        model_id="openai/responses/gpt-5-nano-2025-08-07",
        lab=Lab.OPENAI,
    ),
    ModelRun(
        name="gpt-5-mini-2025-08-07",
        model_id="openai/responses/gpt-5-mini-2025-08-07",
        lab=Lab.OPENAI,
    ),
    ModelRun(
        name="gpt-5.2-2025-12-11-low",
        model_id="openai/responses/gpt-5.2-2025-12-11",
        lab=Lab.OPENAI,
        options={"reasoning_effort": "low"},
    ),
    ModelRun(
        name="gpt-5.2-2025-12-11-medium",
        model_id="openai/responses/gpt-5.2-2025-12-11",
        lab=Lab.OPENAI,
        options={"reasoning_effort": "medium"},
    ),
    ModelRun(
        name="gpt-5.2-2025-12-11-high",
        model_id="openai/responses/gpt-5.2-2025-12-11",
        lab=Lab.OPENAI,
        options={"reasoning_effort": "high"},
    ),
    ModelRun(
        name="gpt-5.2-2025-12-11-high-web-search",
        model_id="openai/responses/gpt-5.2-2025-12-11",
        lab=Lab.OPENAI,
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
        model_id="together_ai/deepseek-ai/DeepSeek-V3.1",
        lab=Lab.TOGETHER,
        options={"temperature": 0},
    ),
    ModelRun(
        name="Kimi-K2-Instruct-0905",
        model_id="together_ai/moonshotai/Kimi-K2-Instruct-0905",
        lab=Lab.TOGETHER,
        options={"temperature": 0},
    ),
]

# Anthropic: https://platform.claude.com/docs/en/about-claude/models/overview
ANTHROPIC_RUNS = [
    ModelRun(
        name="claude-3-7-sonnet-20250219",
        model_id="anthropic/claude-3-7-sonnet-20250219",
        lab=Lab.ANTHROPIC,
        options={"max_tokens": 4096, "temperature": 0},
    ),
    ModelRun(
        name="claude-haiku-4-5-20251001",
        model_id="anthropic/claude-haiku-4-5-20251001",
        lab=Lab.ANTHROPIC,
        options={"max_tokens": 4096, "temperature": 0},
    ),
    ModelRun(
        name="claude-sonnet-4-20250514",
        model_id="anthropic/claude-sonnet-4-20250514",
        lab=Lab.ANTHROPIC,
        options={"max_tokens": 4096, "temperature": 0},
    ),
    ModelRun(
        name="claude-sonnet-4-5-20250929",
        model_id="anthropic/claude-sonnet-4-5-20250929",
        lab=Lab.ANTHROPIC,
        options={"max_tokens": 4096, "temperature": 0},
    ),
    ModelRun(
        name="claude-opus-4-1-20250805",
        model_id="anthropic/claude-opus-4-1-20250805",
        lab=Lab.ANTHROPIC,
        options={"max_tokens": 4096, "temperature": 0},
    ),
    ModelRun(
        name="claude-opus-4-5-20251101",
        model_id="anthropic/claude-opus-4-5-20251101",
        lab=Lab.ANTHROPIC,
        options={"max_tokens": 4096, "temperature": 0},
    ),
    ModelRun(
        name="claude-opus-4-5-20251101-web-search",
        model_id="anthropic/claude-opus-4-5-20251101",
        lab=Lab.ANTHROPIC,
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
        model_id="xai/grok-4-fast-reasoning",
        lab=Lab.XAI,
        options={"temperature": 0},
    ),
    ModelRun(
        name="grok-4-fast-non-reasoning",
        model_id="xai/grok-4-fast-non-reasoning",
        lab=Lab.XAI,
        options={"temperature": 0},
    ),
    ModelRun(
        name="grok-4-1-fast-reasoning",
        model_id="xai/grok-4-1-fast-reasoning",
        lab=Lab.XAI,
        options={"temperature": 0},
    ),
    ModelRun(
        name="grok-4-1-fast-non-reasoning",
        model_id="xai/grok-4-1-fast-non-reasoning",
        lab=Lab.XAI,
        options={"temperature": 0},
    ),
]

# Google: https://ai.google.dev/gemini-api/docs/models
GOOGLE_RUNS = [
    ModelRun(
        name="gemini-2.5-pro",
        model_id="gemini/gemini-2.5-pro",
        lab=Lab.GOOGLE,
        options={"temperature": 0},
    ),
    ModelRun(
        name="gemini-2.5-pro-web-search",
        model_id="gemini/gemini-2.5-pro",
        lab=Lab.GOOGLE,
        options={"temperature": 0, "tools": [{"googleSearch": {}}]},
    ),
    ModelRun(
        name="gemini-2.5-flash",
        model_id="gemini/gemini-2.5-flash",
        lab=Lab.GOOGLE,
        options={"temperature": 0},
    ),
    ModelRun(
        name="gemini-3-flash-preview",
        model_id="gemini/gemini-3-flash-preview",
        lab=Lab.GOOGLE,
    ),
    ModelRun(
        name="gemini-3-pro-preview",
        model_id="gemini/gemini-3-pro-preview",
        lab=Lab.GOOGLE,
    ),
]

# Mistral
MISTRAL_RUNS: list[ModelRun] = []

MODEL_RUNS: list[ModelRun] = (
    OPENAI_RUNS + TOGETHER_RUNS + ANTHROPIC_RUNS + XAI_RUNS + GOOGLE_RUNS + MISTRAL_RUNS
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
    model_id="openai/responses/gpt-5-mini-2025-08-07",
    lab=Lab.OPENAI,
)

# Model used for getting metadata (validating and categorizing questions)
METADATA_MODEL = ModelRun(
    name="gpt-5-mini-2025-08-07-reformat",
    model_id="openai/responses/gpt-5-mini-2025-08-07",
    lab=Lab.OPENAI,
)

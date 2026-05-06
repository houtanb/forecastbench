# Shared LLM Model Runs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move LLM model metadata, run naming, run declarations, and release dates into `/workspace/utils`, then migrate ForecastBench and TimeSeriesBench to select shared `model_run_key` values instead of declaring local `ModelRun(...)` objects.

**Architecture:** `utils.llm.model_runs` becomes the shared source of truth for canonical `LLMModel` metadata and concrete `ModelRun` variants. ForecastBench and TimeSeriesBench keep benchmark-specific selected-key lists, provider-key setup, prompt behavior, output schemas, and orchestration logic, but import shared model-run objects from utils.

**Tech Stack:** Python 3.14+, pytest, `fri-utils`, ForecastBench LLM forecaster, ForecastBench leaderboard, TimeSeriesBench LLM worker/manager/smoke test.

---

## File Structure

Shared utils repository:

- Create `/workspace/utils/utils/llm/model_runs.py`: shared `LLMModel`, `ModelRun`, naming rules, canonical model registry, shared model-run registry, selectors, provider helpers, and release-date lookup.
- Modify `/workspace/utils/utils/llm/__init__.py`: export or leave importable the new `model_runs` module if this file already exposes public LLM helpers.
- Modify `/workspace/utils/tests/unit/test_llm_model_runs.py`: tests for naming, provider route IDs, release dates, registry uniqueness, selectors, and response routing.
- Create `/workspace/utils/tests/integration/llm/test_model_runs.py`: live smoke test that calls each shared `ModelRun` with a tiny prompt so invalid option payloads fail before benchmark runs.
- Modify `/workspace/utils/tests/unit/test_llm_routing.py`: keep existing provider-routing tests passing; no behavior changes required unless imports need adjustment.

ForecastBench:

- Modify `/workspace/forecastbench/requirements.runtime.txt`: update the `fri-utils` pin after the utils commit.
- Modify `/workspace/forecastbench/src/llm_forecaster/model_runs.py`: remove local `ModelRun` and grouped run declarations; select shared utils model runs by key; keep ForecastBench provider-key loading and concurrency helpers.
- Modify `/workspace/forecastbench/src/llm_forecaster/output.py`: keep schema unchanged; use shared `model_run_key` through `model_run.name`.
- Modify `/workspace/forecastbench/src/llm_forecaster/runner.py`: keep behavior unchanged; label transcript output as `Provider model ID` and read `model_run.provider_model_id`.
- Modify `/workspace/forecastbench/src/orchestration/func_llm_forecaster_worker/main.py`: keep task selection by index; no local declaration changes.
- Modify `/workspace/forecastbench/src/orchestration/func_llm_forecaster_manager/main.py`: keep `len(model_runs.MODEL_RUNS)`.
- Modify `/workspace/forecastbench/src/leaderboard/model_identities.py`: map current shared run names to `model_key`; map legacy names to `model_key`; read release dates from utils.
- Modify `/workspace/forecastbench/src/leaderboard/main.py`: stop reading `model_release_dates.csv`; call the model-identity helper for release-date metadata.
- Modify `/workspace/forecastbench/src/leaderboard/Makefile`: remove `model_release_dates.csv` from deploy prerequisites.
- Delete `/workspace/forecastbench/src/leaderboard/model_release_dates.csv`.
- Modify `/workspace/forecastbench/src/tests/llm_forecaster/test_model_runs.py`: expect shared selected keys and no local declarations.
- Modify `/workspace/forecastbench/src/tests/leaderboard/test_model_identities.py`: assert release-date lookup uses utils and CSV is gone.
- Modify `/workspace/forecastbench/src/tests/leaderboard/test_llm_legacy_names.py`: assert legacy names map to canonical utils `model_key` values.
- Create `/workspace/forecastbench/src/tests/test_shared_llm_model_runs.py`: anti-drift scan rejecting benchmark-local `ModelRun` declarations.

TimeSeriesBench:

- Modify `/workspace/time-series-benchmark/requirements.txt`: update the `fri-utils` pin after the utils commit.
- Modify `/workspace/time-series-benchmark/src/helpers/constants.py`: remove local `ModelRun` dataclass and grouped run declarations; select shared model runs by key; keep `MODEL_RUNS` and `REFORMAT_MODEL` compatibility constants.
- Modify `/workspace/time-series-benchmark/src/models/llms/common.py`: update type references to utils `ModelRun` and keep provider-key behavior.
- Modify `/workspace/time-series-benchmark/src/models/llms/manager/main.py`: keep task count and logging behavior.
- Modify `/workspace/time-series-benchmark/src/models/llms/worker/main.py`: keep task index selection and output filenames using `model_run.name`.
- Modify `/workspace/time-series-benchmark/src/models/llms/smoke_test/main.py`: keep key selection by `model_run.name`.
- Create `/workspace/time-series-benchmark/tests/test_shared_llm_model_runs.py`: selected-key and anti-drift tests.
- Modify any existing TimeSeriesBench LLM tests that construct local `constants.ModelRun` instances to import shared `ModelRun` or use simple fakes.

---

### Task 1: Add Shared Utils Model-Run Naming Tests

**Files:**
- Create: `/workspace/utils/tests/unit/test_llm_model_runs.py`
- Create: `/workspace/utils/utils/llm/model_runs.py`

- [ ] **Step 1: Write failing tests for model keys, provider IDs, and naming**

Create `/workspace/utils/tests/unit/test_llm_model_runs.py`:

```python
from datetime import date
from unittest.mock import patch

import pytest

from utils.llm.lab_registry import LABS
from utils.llm.provider_registry import PROVIDERS


def test_model_key_and_provider_model_id_are_distinct_for_together_models():
    from utils.llm import model_runs

    model = model_runs.LLM_MODELS_BY_KEY["deepseek-v3.1"]

    assert model.model_key == "deepseek-v3.1"
    assert model.provider_model_id == "deepseek-ai/DeepSeek-V3.1"
    assert model.lab == LABS["DeepSeek"]
    assert model.provider == PROVIDERS["Together"]
    assert model.release_date == date(2025, 8, 21)


@pytest.mark.parametrize(
    ("model_key", "options", "expected_key"),
    [
        ("gpt-5.5-2026-04-23", {}, "gpt-5.5-2026-04-23"),
        (
            "gpt-5.5-2026-04-23",
            {"reasoning": {"effort": "high"}},
            "gpt-5.5-2026-04-23-high",
        ),
        (
            "gpt-5.5-2026-04-23",
            {"reasoning": {"effort": "high"}, "tools": [{"type": "web_search"}]},
            "gpt-5.5-2026-04-23-high-web-search",
        ),
        (
            "claude-opus-4-7",
            {
                "max_tokens": 64000,
                "output_config": {"effort": "high"},
                "thinking": {"type": "adaptive"},
                "tools": [{"type": "web_search_20260209", "name": "web_search", "max_uses": 5}],
            },
            "claude-opus-4-7-adaptive-thinking-high-web-search-64000",
        ),
        (
            "grok-4.20-0309-reasoning",
            {"tools": [{"type": "web_search"}, {"type": "x_search"}], "max_tokens": 10000},
            "grok-4.20-0309-reasoning-web-search-x-search-10000",
        ),
        ("deepseek-v3.1", {"max_tokens": 10000}, "deepseek-v3.1-10000"),
    ],
)
def test_model_run_key_is_generated_from_name_relevant_options(
    model_key, options, expected_key
):
    from utils.llm import model_runs

    run = model_runs.ModelRun(
        model=model_runs.LLM_MODELS_BY_KEY[model_key],
        options=options,
    )

    assert run.model_run_key == expected_key
    assert run.name == expected_key


def test_name_neutral_options_do_not_appear_in_model_run_key():
    from utils.llm import model_runs

    run = model_runs.ModelRun(
        model=model_runs.LLM_MODELS_BY_KEY["gemini-3.1-pro-preview"],
        options={
            "candidate_count": 1,
            "temperature": 0,
            "automatic_function_calling": {"disable": True},
        },
    )

    assert run.model_run_key == "gemini-3.1-pro-preview"


def test_unknown_option_paths_raise_in_model_run_validation():
    from utils.llm import model_runs

    with pytest.raises(ValueError, match="name-relevant or name-neutral"):
        model_runs.ModelRun(
            model=model_runs.LLM_MODELS_BY_KEY["gpt-5.5-2026-04-23"],
            options={"new_performance_option": True},
        ).model_run_key


def test_model_run_routes_provider_model_id_to_get_response():
    from utils.llm import model_runs

    run = model_runs.ModelRun(
        model=model_runs.LLM_MODELS_BY_KEY["deepseek-v3.1"],
        options={"temperature": 0},
    )

    with patch("utils.llm.model_registry.get_response", return_value="forecast") as get_response:
        response = run.get_response("prompt", max_tokens=10000)

    assert response == "forecast"
    get_response.assert_called_once_with(
        provider=PROVIDERS["Together"],
        model_id="deepseek-ai/DeepSeek-V3.1",
        prompt="prompt",
        options={"temperature": 0, "max_tokens": 10000},
    )
```

- [ ] **Step 2: Add a minimal empty module so imports fail for missing names**

Create `/workspace/utils/utils/llm/model_runs.py` with:

```python
"""Shared LLM model-run registry."""
```

- [ ] **Step 3: Run the focused failing tests**

Run:

```bash
cd /workspace/utils
python -m pytest tests/unit/test_llm_model_runs.py -q
```

Expected: FAIL because `LLM_MODELS_BY_KEY`, `ModelRun`, and naming helpers do not exist.

---

### Task 2: Implement Shared Utils Model-Run Types and Naming Rules

**Files:**
- Modify: `/workspace/utils/utils/llm/model_runs.py`
- Modify: `/workspace/utils/utils/llm/__init__.py`
- Modify: `/workspace/utils/tests/unit/test_llm_model_runs.py`

- [ ] **Step 1: Implement the shared types and naming helpers**

Replace `/workspace/utils/utils/llm/model_runs.py` with:

```python
"""Shared LLM model-run registry."""

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from .lab_registry import LABS, Lab
from .provider_registry import PROVIDERS, Provider

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LLMModel:
    """Canonical LLM model metadata."""

    model_key: str
    provider_model_id: str
    lab: Lab
    provider: Provider
    release_date: date
    token_limit: int | None = None


@dataclass(frozen=True, slots=True)
class ModelRun:
    """Concrete LLM run with provider options."""

    model: LLMModel
    options: dict[str, Any] = field(default_factory=dict)

    @property
    def model_run_key(self) -> str:
        """Return the generated full model-run key."""
        return build_model_run_key(self.model.model_key, self.options)

    @property
    def name(self) -> str:
        """Return the model-run key for compatibility with benchmark code."""
        return self.model_run_key

    @property
    def id(self) -> str:
        """Return the model-run identifier."""
        return self.model_run_key

    @property
    def model_key(self) -> str:
        """Return the canonical base model key."""
        return self.model.model_key

    @property
    def provider_model_id(self) -> str:
        """Return the provider API model identifier."""
        return self.model.provider_model_id

    @property
    def lab(self) -> Lab:
        """Return the model-making lab."""
        return self.model.lab

    @property
    def provider(self) -> Provider:
        """Return the API provider route."""
        return self.model.provider

    @property
    def model_organization(self) -> str:
        """Return the model lab display name."""
        return self.model.lab.leaderboard_name

    @property
    def release_date(self) -> date:
        """Return the underlying model release date."""
        return self.model.release_date

    def __repr__(self) -> str:
        """Return a concise model-run representation."""
        if self.options:
            return (
                f"<ModelRun {self.model_run_key} "
                f"({self.provider_model_id}) {self.options}>"
            )
        return f"<ModelRun {self.model_run_key}>"

    def get_response(self, prompt: str, **kwargs: Any) -> str:
        """Request a response from the configured provider and model."""
        from utils.llm.model_registry import get_response

        merged_options = {**self.options, **kwargs}
        logger.info(
            "Requesting LLM response provider=%s provider_model_id=%s options=%s",
            self.provider.name,
            self.provider_model_id,
            merged_options,
        )
        return get_response(
            provider=self.provider,
            model_id=self.provider_model_id,
            prompt=prompt,
            options=merged_options,
        )


NAME_NEUTRAL_OPTION_PATHS = {
    ("temperature",),
    ("candidate_count",),
    ("automatic_function_calling",),
}


def _option_path_exists(options: dict[str, Any], path: tuple[str, ...]) -> bool:
    """Return whether a nested option path exists."""
    current: Any = options
    for part in path:
        if not isinstance(current, dict) or part not in current:
            return False
        current = current[part]
    return True


def _iter_leaf_paths(value: Any, prefix: tuple[str, ...] = ()) -> Iterable[tuple[str, ...]]:
    """Yield leaf paths for nested option data."""
    if isinstance(value, dict):
        for key, nested in value.items():
            yield from _iter_leaf_paths(nested, (*prefix, str(key)))
    elif isinstance(value, list):
        yield prefix
    else:
        yield prefix


def _is_path_covered(path: tuple[str, ...], covered_prefixes: set[tuple[str, ...]]) -> bool:
    """Return whether path is covered by a consumed or neutral prefix."""
    return any(path[: len(prefix)] == prefix for prefix in covered_prefixes)


def _thinking_suffixes(options: dict[str, Any]) -> tuple[list[str], set[tuple[str, ...]]]:
    """Return suffixes and consumed paths for model thinking options."""
    thinking = options.get("thinking")
    if not isinstance(thinking, dict):
        return [], set()
    if thinking.get("type") == "adaptive":
        return ["adaptive-thinking"], {("thinking",)}
    raise ValueError(f"Unsupported thinking option for model-run naming: {thinking}")


def _effort_suffixes(options: dict[str, Any]) -> tuple[list[str], set[tuple[str, ...]]]:
    """Return suffixes and consumed paths for effort options."""
    suffixes = []
    consumed: set[tuple[str, ...]] = set()

    reasoning = options.get("reasoning")
    if isinstance(reasoning, dict) and "effort" in reasoning:
        suffixes.append(str(reasoning["effort"]).replace("_", "-").lower())
        consumed.add(("reasoning",))

    output_config = options.get("output_config")
    if isinstance(output_config, dict) and "effort" in output_config:
        suffixes.append(str(output_config["effort"]).replace("_", "-").lower())
        consumed.add(("output_config",))

    return suffixes, consumed


def _tool_suffixes(options: dict[str, Any]) -> tuple[list[str], set[tuple[str, ...]]]:
    """Return suffixes and consumed paths for tool options."""
    tools = options.get("tools")
    if not tools:
        return [], set()
    if not isinstance(tools, list):
        raise ValueError("tools option must be a list for model-run naming")

    suffixes = []
    for tool in tools:
        if not isinstance(tool, dict):
            raise ValueError(f"Unsupported tool option for model-run naming: {tool}")
        tool_type = tool.get("type")
        if tool_type in {"web_search", "web_search_20260209"} or "googleSearch" in tool:
            suffix = "web-search"
        elif tool_type == "x_search":
            suffix = "x-search"
        else:
            raise ValueError(f"Unsupported tool option for model-run naming: {tool}")
        if suffix not in suffixes:
            suffixes.append(suffix)

    tool_order = {"web-search": 0, "x-search": 1}
    return sorted(suffixes, key=tool_order.__getitem__), {("tools",)}


def _token_suffixes(options: dict[str, Any]) -> tuple[list[str], set[tuple[str, ...]]]:
    """Return suffixes and consumed paths for token cap options."""
    suffixes = []
    consumed: set[tuple[str, ...]] = set()

    for key in ("max_tokens", "max_output_tokens"):
        if key in options:
            suffixes.append(str(options[key]))
            consumed.add((key,))

    return suffixes, consumed


NAME_COMPONENT_RULES = (
    _thinking_suffixes,
    _effort_suffixes,
    _tool_suffixes,
    _token_suffixes,
)


def build_model_run_key(model_key: str, options: dict[str, Any]) -> str:
    """Build a stable model-run key from a base model key and options."""
    suffixes = []
    consumed_prefixes = set(NAME_NEUTRAL_OPTION_PATHS)

    for rule in NAME_COMPONENT_RULES:
        rule_suffixes, rule_consumed = rule(options)
        suffixes.extend(rule_suffixes)
        consumed_prefixes.update(rule_consumed)

    unknown_paths = sorted(
        path
        for path in _iter_leaf_paths(options)
        if not _is_path_covered(path, consumed_prefixes)
    )
    if unknown_paths:
        raise ValueError(
            "ModelRun options must be name-relevant or name-neutral. "
            f"Unknown option paths: {unknown_paths}"
        )

    if suffixes:
        return "-".join([model_key, *suffixes])
    return model_key


LLM_MODELS: list[LLMModel] = []
LLM_MODELS_BY_KEY: dict[str, LLMModel] = {}
MODEL_RUNS: list[ModelRun] = []
MODEL_RUNS_BY_KEY: dict[str, ModelRun] = {}


def get_model_run(model_run_key: str) -> ModelRun:
    """Return a shared model run by key."""
    try:
        return MODEL_RUNS_BY_KEY[model_run_key]
    except KeyError as exc:
        available = ", ".join(sorted(MODEL_RUNS_BY_KEY))
        raise KeyError(f"Unknown LLM model_run_key {model_run_key}. Available: {available}") from exc


def select_model_runs(model_run_keys: Sequence[str]) -> list[ModelRun]:
    """Return model runs in the requested order."""
    return [get_model_run(model_run_key) for model_run_key in model_run_keys]


def model_release_dates_by_key() -> dict[str, date]:
    """Return release dates keyed by canonical model_key."""
    return {model.model_key: model.release_date for model in LLM_MODELS}
```

- [ ] **Step 2: Expose the lazy `model_runs` module import**

In `/workspace/utils/utils/llm/__init__.py`, change:

```python
__all__ = ["lab_registry", "model_registry", "provider_registry", "providers"]
```

to:

```python
__all__ = [
    "lab_registry",
    "model_registry",
    "model_runs",
    "provider_registry",
    "providers",
]
```

- [ ] **Step 3: Run the focused tests and verify the expected remaining failures**

Run:

```bash
cd /workspace/utils
python -m pytest tests/unit/test_llm_model_runs.py -q
```

Expected: FAIL because the registry lists are still empty.

- [ ] **Step 4: Commit nothing yet**

Do not commit until the registry is populated in Task 3 and these tests pass.

---

### Task 3: Populate Shared Utils Canonical Models and Shared Runs

**Files:**
- Modify: `/workspace/utils/utils/llm/model_runs.py`
- Modify: `/workspace/utils/tests/unit/test_llm_model_runs.py`

- [ ] **Step 1: Add registry tests for the shared run set**

Append these tests to `/workspace/utils/tests/unit/test_llm_model_runs.py`:

```python
def test_shared_model_run_registry_contains_forecastbench_and_timeseriesbench_runs():
    from utils.llm import model_runs

    expected_keys = {
        "gpt-4o-mini-2024-07-18",
        "gpt-5-nano-2025-08-07",
        "gpt-5-mini-2025-08-07",
        "gpt-5-mini-2025-08-07-1024",
        "gpt-5.2-2025-12-11",
        "gpt-5.4-2026-03-05",
        "gpt-5.4-2026-03-05-high",
        "gpt-5.4-2026-03-05-high-web-search",
        "gpt-5.4-mini-2026-03-17",
        "gpt-5.4-nano-2026-03-17",
        "gpt-5.5-2026-04-23-medium",
        "gpt-5.5-2026-04-23-high",
        "gpt-5.5-2026-04-23-high-web-search",
        "deepseek-v3.1",
        "minimax-m2.5",
        "minimax-m2.7",
        "kimi-k2.5",
        "kimi-k2.6",
        "glm-5.1",
        "gemma-4-31b",
        "claude-haiku-4-5-20251001-1024",
        "claude-haiku-4-5-20251001-4096",
        "claude-sonnet-4-5-20250929-1024",
        "claude-sonnet-4-5-20250929-4096",
        "claude-sonnet-4-6-1024",
        "claude-sonnet-4-6-4096",
        "claude-sonnet-4-6-adaptive-thinking-16000",
        "claude-opus-4-6-4096",
        "claude-opus-4-7-1024",
        "claude-opus-4-7-4096",
        "claude-opus-4-7-adaptive-thinking-high-24000",
        "claude-opus-4-7-adaptive-thinking-high-web-search-64000",
        "grok-4-1-fast-reasoning",
        "grok-4-1-fast-non-reasoning",
        "grok-4.20-0309-reasoning",
        "grok-4.20-0309-reasoning-web-search-x-search",
        "grok-4.20-0309-non-reasoning",
        "grok-4.3",
        "gemini-2.5-pro",
        "gemini-2.5-pro-web-search",
        "gemini-3-flash-preview",
        "gemini-3.1-flash-lite-preview",
        "gemini-3.1-flash-lite",
        "gemini-3.1-pro-preview",
    }

    assert expected_keys <= set(model_runs.MODEL_RUNS_BY_KEY)


def test_shared_model_run_keys_are_unique_and_file_safe():
    from utils.llm import model_runs

    keys = [run.model_run_key for run in model_runs.MODEL_RUNS]

    assert len(keys) == len(set(keys))
    assert all(key == key.lower() for key in keys)
    assert all(" " not in key and "/" not in key and "_" not in key for key in keys)


def test_release_dates_exist_for_all_shared_models():
    from utils.llm import model_runs

    release_dates = model_runs.model_release_dates_by_key()

    assert release_dates["gpt-5.5-2026-04-23"] == date(2026, 4, 23)
    assert release_dates["deepseek-v3.1"] == date(2025, 8, 21)
    assert release_dates["gemini-3.1-flash-lite"] == date(2026, 5, 8)
    assert set(release_dates) == {model.model_key for model in model_runs.LLM_MODELS}


def test_select_model_runs_preserves_order_and_rejects_unknown_keys():
    from utils.llm import model_runs

    selected = model_runs.select_model_runs(
        ["gpt-5.4-2026-03-05", "deepseek-v3.1"]
    )

    assert [run.model_run_key for run in selected] == [
        "gpt-5.4-2026-03-05",
        "deepseek-v3.1",
    ]
    with pytest.raises(KeyError, match="missing-model"):
        model_runs.select_model_runs(["missing-model"])
```

- [ ] **Step 2: Populate canonical models and run registry**

In `/workspace/utils/utils/llm/model_runs.py`, replace the empty registry block with the canonical model and run declarations below. Keep the type and helper definitions from Task 2 above this block.

```python
LLM_MODELS = [
    LLMModel("gpt-4o-mini-2024-07-18", "gpt-4o-mini", LABS["OpenAI"], PROVIDERS["OpenAI"], date(2024, 7, 18), 128_000),
    LLMModel("gpt-5-nano-2025-08-07", "gpt-5-nano-2025-08-07", LABS["OpenAI"], PROVIDERS["OpenAI"], date(2025, 8, 7), 128_000),
    LLMModel("gpt-5-mini-2025-08-07", "gpt-5-mini-2025-08-07", LABS["OpenAI"], PROVIDERS["OpenAI"], date(2025, 8, 7), 128_000),
    LLMModel("gpt-5.2-2025-12-11", "gpt-5.2-2025-12-11", LABS["OpenAI"], PROVIDERS["OpenAI"], date(2025, 12, 11), 128_000),
    LLMModel("gpt-5.4-2026-03-05", "gpt-5.4-2026-03-05", LABS["OpenAI"], PROVIDERS["OpenAI"], date(2026, 3, 5), 128_000),
    LLMModel("gpt-5.4-mini-2026-03-17", "gpt-5.4-mini-2026-03-17", LABS["OpenAI"], PROVIDERS["OpenAI"], date(2026, 3, 17), 128_000),
    LLMModel("gpt-5.4-nano-2026-03-17", "gpt-5.4-nano-2026-03-17", LABS["OpenAI"], PROVIDERS["OpenAI"], date(2026, 3, 17), 128_000),
    LLMModel("gpt-5.5-2026-04-23", "gpt-5.5-2026-04-23", LABS["OpenAI"], PROVIDERS["OpenAI"], date(2026, 4, 23), 128_000),
    LLMModel("deepseek-v3.1", "deepseek-ai/DeepSeek-V3.1", LABS["DeepSeek"], PROVIDERS["Together"], date(2025, 8, 21), 128_000),
    LLMModel("minimax-m2.5", "MiniMaxAI/MiniMax-M2.5", LABS["MiniMax"], PROVIDERS["Together"], date(2026, 2, 12), 128_000),
    LLMModel("minimax-m2.7", "MiniMaxAI/MiniMax-M2.7", LABS["MiniMax"], PROVIDERS["Together"], date(2026, 3, 18), 128_000),
    LLMModel("kimi-k2.5", "moonshotai/Kimi-K2.5", LABS["Moonshot"], PROVIDERS["Together"], date(2026, 1, 30), 128_000),
    LLMModel("kimi-k2.6", "moonshotai/Kimi-K2.6", LABS["Moonshot"], PROVIDERS["Together"], date(2026, 4, 20), 128_000),
    LLMModel("glm-5.1", "zai-org/GLM-5.1", LABS["Z.ai"], PROVIDERS["Together"], date(2026, 4, 7), 202_752),
    LLMModel("gemma-4-31b", "google/gemma-4-31B-it", LABS["Google DeepMind"], PROVIDERS["Together"], date(2026, 4, 2), 128_000),
    LLMModel("claude-haiku-4-5-20251001", "claude-haiku-4-5-20251001", LABS["Anthropic"], PROVIDERS["Anthropic"], date(2025, 10, 1), 200_000),
    LLMModel("claude-sonnet-4-5-20250929", "claude-sonnet-4-5-20250929", LABS["Anthropic"], PROVIDERS["Anthropic"], date(2025, 9, 29), 200_000),
    LLMModel("claude-sonnet-4-6", "claude-sonnet-4-6", LABS["Anthropic"], PROVIDERS["Anthropic"], date(2026, 2, 17), 200_000),
    LLMModel("claude-opus-4-6", "claude-opus-4-6", LABS["Anthropic"], PROVIDERS["Anthropic"], date(2026, 2, 5), 200_000),
    LLMModel("claude-opus-4-7", "claude-opus-4-7", LABS["Anthropic"], PROVIDERS["Anthropic"], date(2026, 4, 16), 200_000),
    LLMModel("grok-4-1-fast-reasoning", "grok-4-1-fast-reasoning", LABS["xAI"], PROVIDERS["xAI"], date(2025, 11, 17), 2_000_000),
    LLMModel("grok-4-1-fast-non-reasoning", "grok-4-1-fast-non-reasoning", LABS["xAI"], PROVIDERS["xAI"], date(2025, 11, 17), 2_000_000),
    LLMModel("grok-4.20-0309-reasoning", "grok-4.20-0309-reasoning", LABS["xAI"], PROVIDERS["xAI"], date(2026, 3, 9), 2_000_000),
    LLMModel("grok-4.20-0309-non-reasoning", "grok-4.20-0309-non-reasoning", LABS["xAI"], PROVIDERS["xAI"], date(2026, 3, 9), 2_000_000),
    LLMModel("grok-4.3", "grok-4.3", LABS["xAI"], PROVIDERS["xAI"], date(2026, 5, 1), 2_000_000),
    LLMModel("gemini-2.5-pro", "gemini-2.5-pro", LABS["Google DeepMind"], PROVIDERS["Google"], date(2025, 6, 17), 1_048_576),
    LLMModel("gemini-3-flash-preview", "gemini-3-flash-preview", LABS["Google DeepMind"], PROVIDERS["Google"], date(2025, 12, 17), 1_048_576),
    LLMModel("gemini-3.1-flash-lite-preview", "gemini-3.1-flash-lite-preview", LABS["Google DeepMind"], PROVIDERS["Google"], date(2026, 3, 3), 1_048_576),
    LLMModel("gemini-3.1-flash-lite", "gemini-3.1-flash-lite", LABS["Google DeepMind"], PROVIDERS["Google"], date(2026, 5, 8), 1_048_576),
    LLMModel("gemini-3.1-pro-preview", "gemini-3.1-pro-preview", LABS["Google DeepMind"], PROVIDERS["Google"], date(2026, 2, 19), 1_048_576),
]

LLM_MODELS_BY_KEY = {model.model_key: model for model in LLM_MODELS}


def _run(model_key: str, options: dict[str, Any] | None = None) -> ModelRun:
    """Create a shared model run from a canonical model key."""
    return ModelRun(model=LLM_MODELS_BY_KEY[model_key], options=options or {})


MODEL_RUNS = [
    _run("gpt-4o-mini-2024-07-18", {"temperature": 0}),
    _run("gpt-5-nano-2025-08-07"),
    _run("gpt-5-mini-2025-08-07"),
    _run("gpt-5-mini-2025-08-07", {"max_output_tokens": 1024}),
    _run("gpt-5.2-2025-12-11"),
    _run("gpt-5.4-2026-03-05"),
    _run("gpt-5.4-2026-03-05", {"reasoning": {"effort": "high"}}),
    _run("gpt-5.4-2026-03-05", {"reasoning": {"effort": "high"}, "tools": [{"type": "web_search"}]}),
    _run("gpt-5.4-mini-2026-03-17"),
    _run("gpt-5.4-nano-2026-03-17"),
    _run("gpt-5.5-2026-04-23", {"reasoning": {"effort": "medium"}}),
    _run("gpt-5.5-2026-04-23", {"reasoning": {"effort": "high"}}),
    _run("gpt-5.5-2026-04-23", {"reasoning": {"effort": "high"}, "tools": [{"type": "web_search"}]}),
    _run("deepseek-v3.1", {"temperature": 0}),
    _run("minimax-m2.5", {"temperature": 0}),
    _run("minimax-m2.7", {"temperature": 0}),
    _run("kimi-k2.5", {"temperature": 0}),
    _run("kimi-k2.6", {"temperature": 0}),
    _run("glm-5.1", {"temperature": 0}),
    _run("gemma-4-31b", {"temperature": 0}),
    _run("claude-haiku-4-5-20251001", {"max_tokens": 1024, "temperature": 0}),
    _run("claude-haiku-4-5-20251001", {"max_tokens": 4096}),
    _run("claude-sonnet-4-5-20250929", {"max_tokens": 1024, "temperature": 0}),
    _run("claude-sonnet-4-5-20250929", {"max_tokens": 4096}),
    _run("claude-sonnet-4-6", {"max_tokens": 1024, "temperature": 0}),
    _run("claude-sonnet-4-6", {"max_tokens": 4096}),
    _run("claude-sonnet-4-6", {"max_tokens": 16000, "thinking": {"type": "adaptive"}}),
    _run("claude-opus-4-6", {"max_tokens": 4096}),
    _run("claude-opus-4-7", {"max_tokens": 1024}),
    _run("claude-opus-4-7", {"max_tokens": 4096}),
    _run("claude-opus-4-7", {"max_tokens": 24000, "thinking": {"type": "adaptive"}, "output_config": {"effort": "high"}}),
    _run("claude-opus-4-7", {"max_tokens": 64000, "output_config": {"effort": "high"}, "thinking": {"type": "adaptive"}, "tools": [{"type": "web_search_20260209", "name": "web_search", "max_uses": 5}]}),
    _run("grok-4-1-fast-reasoning"),
    _run("grok-4-1-fast-non-reasoning"),
    _run("grok-4.20-0309-reasoning", {"temperature": 0}),
    _run("grok-4.20-0309-reasoning", {"tools": [{"type": "web_search"}, {"type": "x_search"}]}),
    _run("grok-4.20-0309-non-reasoning", {"temperature": 0}),
    _run("grok-4.3", {"temperature": 0}),
    _run("gemini-2.5-pro", {"temperature": 0}),
    _run("gemini-2.5-pro", {"temperature": 0, "tools": [{"googleSearch": {}}]}),
    _run("gemini-3-flash-preview", {"candidate_count": 1, "temperature": 0, "automatic_function_calling": {"disable": True}}),
    _run("gemini-3.1-flash-lite-preview", {"candidate_count": 1, "temperature": 0, "automatic_function_calling": {"disable": True}}),
    _run("gemini-3.1-flash-lite", {"candidate_count": 1, "temperature": 0, "automatic_function_calling": {"disable": True}}),
    _run("gemini-3.1-pro-preview", {"candidate_count": 1, "temperature": 0, "automatic_function_calling": {"disable": True}}),
]

MODEL_RUNS_BY_KEY = {run.model_run_key: run for run in MODEL_RUNS}

if len(MODEL_RUNS_BY_KEY) != len(MODEL_RUNS):
    duplicate_keys = sorted(
        key
        for key in {run.model_run_key for run in MODEL_RUNS}
        if [run.model_run_key for run in MODEL_RUNS].count(key) > 1
    )
    raise ValueError(f"Duplicate shared LLM model_run_key values: {duplicate_keys}")
```

- [ ] **Step 3: Format the long registry**

Run:

```bash
cd /workspace/utils
python -m black utils/llm/model_runs.py tests/unit/test_llm_model_runs.py
python -m isort utils/llm/model_runs.py tests/unit/test_llm_model_runs.py
```

Expected: formatters complete successfully.

- [ ] **Step 4: Run utils tests**

Run:

```bash
cd /workspace/utils
python -m pytest tests/unit/test_llm_model_runs.py tests/unit/test_llm_routing.py tests/unit/test_model_registry_config.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit utils shared model runs**

Run:

```bash
cd /workspace/utils
git add utils/llm/model_runs.py utils/llm/__init__.py tests/unit/test_llm_model_runs.py
git commit -m "feat(llm): add shared model runs"
git rev-parse HEAD
```

Expected: commit succeeds. Save the printed SHA as `UTILS_SHA` for ForecastBench and TimeSeriesBench pin updates.

---

### Task 4: Add Live Utils Smoke Tests for Shared Model Runs

**Files:**
- Create: `/workspace/utils/tests/integration/llm/test_model_runs.py`

- [ ] **Step 1: Write the live integration smoke test**

Create `/workspace/utils/tests/integration/llm/test_model_runs.py`:

```python
"""Integration tests for shared LLM model-run declarations."""

import os

import pytest

from utils.llm import model_runs


def _selected_model_runs() -> list[model_runs.ModelRun]:
    requested = os.getenv("LLM_MODEL_RUN_KEYS")
    if not requested:
        return list(model_runs.MODEL_RUNS)
    requested_keys = [key.strip() for key in requested.split(",") if key.strip()]
    return model_runs.select_model_runs(requested_keys)


@pytest.mark.integration
@pytest.mark.parametrize(
    "model_run",
    _selected_model_runs(),
    ids=lambda run: run.model_run_key,
)
def test_shared_model_run_live_call_accepts_declared_options(model_run):
    """Each shared model run should accept its declared options in a live call."""
    response = model_run.get_response("Reply with exactly: OK")

    assert isinstance(response, str)
    assert "ok" in response.strip().lower()
```

This intentionally calls each declared `ModelRun` with its real options. If a
provider rejects an option shape, the test fails before ForecastBench or
TimeSeriesBench uses that run in production. `LLM_MODEL_RUN_KEYS` is an optional
comma-separated subset for debugging one model run, but the default is all
shared runs.

- [ ] **Step 2: Run one selected integration test**

Run:

```bash
cd /workspace/utils
LLM_MODEL_RUN_KEYS=gpt-5-mini-2025-08-07 python -m pytest \
  tests/integration/llm/test_model_runs.py --integration -q
```

Expected: PASS if OpenAI credentials are configured. If credentials are not
available in the local environment, document that the test was not run locally
and run it in the environment used for live provider integration tests.

- [ ] **Step 3: Run the full pre-round smoke command when credentials are available**

Run:

```bash
cd /workspace/utils
python -m pytest tests/integration/llm/test_model_runs.py --integration -q
```

Expected: PASS before a forecasting round. This makes one live API call per
shared model run.

- [ ] **Step 4: Commit the utils integration smoke test**

Run:

```bash
cd /workspace/utils
git add tests/integration/llm/test_model_runs.py
git commit -m "test(llm): smoke test shared model runs"
git rev-parse HEAD
```

Expected: commit succeeds. Save the printed SHA as `UTILS_SHA`; ForecastBench
and TimeSeriesBench pin this commit or a later utils commit.

---

### Task 5: Migrate ForecastBench Model-Run Selection to Utils

**Files:**
- Modify: `/workspace/forecastbench/requirements.runtime.txt`
- Modify: `/workspace/forecastbench/src/llm_forecaster/model_runs.py`
- Modify: `/workspace/forecastbench/src/tests/llm_forecaster/test_model_runs.py`

- [ ] **Step 1: Write failing ForecastBench selected-key tests**

Modify `/workspace/forecastbench/src/tests/llm_forecaster/test_model_runs.py` so the active model-set test expects local selection keys and verifies the selected objects come from utils:

```python
def test_forecastbench_active_model_set_is_selected_from_shared_utils():
    from utils.llm import model_runs as shared_model_runs

    assert model_runs.FORECASTBENCH_MODEL_RUN_KEYS == [
        "gpt-5.4-2026-03-05",
        "gpt-5.4-mini-2026-03-17",
        "gpt-5.4-nano-2026-03-17",
        "gpt-5.2-2025-12-11",
        "gpt-5-mini-2025-08-07",
        "gpt-5-nano-2025-08-07",
        "deepseek-v3.1",
        "minimax-m2.5",
        "minimax-m2.7",
        "kimi-k2.5",
        "kimi-k2.6",
        "glm-5.1",
        "gemma-4-31b",
        "claude-haiku-4-5-20251001-1024",
        "claude-sonnet-4-5-20250929-1024",
        "claude-sonnet-4-6-1024",
        "claude-opus-4-7-1024",
        "grok-4.20-0309-reasoning",
        "grok-4.20-0309-non-reasoning",
        "grok-4.3",
        "gemini-3.1-pro-preview",
        "gemini-3.1-flash-lite-preview",
        "gemini-3.1-flash-lite",
        "gemini-3-flash-preview",
    ]
    assert [run.model_run_key for run in model_runs.MODEL_RUNS] == (
        model_runs.FORECASTBENCH_MODEL_RUN_KEYS
    )
    assert all(run is shared_model_runs.get_model_run(run.name) for run in model_runs.MODEL_RUNS)
```

Replace assertions that depend on local `ModelRun(name=..., model_id=...)` construction with shared-object assertions:

```python
def test_reformat_model_uses_shared_gpt_5_mini_output_cap_run():
    assert model_runs.REFORMAT_MODEL.model_run_key == "gpt-5-mini-2025-08-07-1024"
    assert model_runs.REFORMAT_MODEL.provider_model_id == "gpt-5-mini-2025-08-07"
    assert model_runs.REFORMAT_MODEL.options == {"max_output_tokens": 1024}
```

- [ ] **Step 2: Run the failing ForecastBench test**

Run:

```bash
cd /workspace/forecastbench
PYTHONSAFEPATH=1 PYTHONPATH=src:/workspace/utils python -m pytest \
  src/tests/llm_forecaster/test_model_runs.py -q
```

Expected: FAIL because `FORECASTBENCH_MODEL_RUN_KEYS` does not exist and local model declarations still exist.

- [ ] **Step 3: Update the utils pin**

Run:

```bash
cd /workspace/forecastbench
UTILS_SHA=$(git -C /workspace/utils rev-parse HEAD)
printf "git+https://github.com/forecastingresearch/utils@%s#egg=fri-utils\n" "$UTILS_SHA" > requirements.runtime.txt
```

- [ ] **Step 4: Replace local ForecastBench model declarations with shared selections**

Replace the declaration section of `/workspace/forecastbench/src/llm_forecaster/model_runs.py` from the local `ModelRun` dataclass through `REFORMAT_MODEL` with:

```python
from typing import Sequence

from utils.helpers.constants import (
    ANTHROPIC_API_KEY_SECRET_NAME,
    GOOGLE_GEMINI_API_KEY_SECRET_NAME,
    OPENAI_API_KEY_SECRET_NAME,
    TOGETHER_API_KEY_SECRET_NAME,
    XAI_API_KEY_SECRET_NAME,
)
from utils.llm.model_runs import ModelRun, get_model_run, select_model_runs
from utils.llm.provider_registry import PROVIDERS, Provider

FORECASTBENCH_MODEL_RUN_KEYS = [
    "gpt-5.4-2026-03-05",
    "gpt-5.4-mini-2026-03-17",
    "gpt-5.4-nano-2026-03-17",
    "gpt-5.2-2025-12-11",
    "gpt-5-mini-2025-08-07",
    "gpt-5-nano-2025-08-07",
    "deepseek-v3.1",
    "minimax-m2.5",
    "minimax-m2.7",
    "kimi-k2.5",
    "kimi-k2.6",
    "glm-5.1",
    "gemma-4-31b",
    "claude-haiku-4-5-20251001-1024",
    "claude-sonnet-4-5-20250929-1024",
    "claude-sonnet-4-6-1024",
    "claude-opus-4-7-1024",
    "grok-4.20-0309-reasoning",
    "grok-4.20-0309-non-reasoning",
    "grok-4.3",
    "gemini-3.1-pro-preview",
    "gemini-3.1-flash-lite-preview",
    "gemini-3.1-flash-lite",
    "gemini-3-flash-preview",
]

MODEL_RUNS = select_model_runs(FORECASTBENCH_MODEL_RUN_KEYS)
MODEL_RUNS_BY_NAME = {run.name: run for run in MODEL_RUNS}
REFORMAT_MODEL = get_model_run("gpt-5-mini-2025-08-07-1024")
```

Keep the existing `logger`, `PROVIDER_MAX_WORKERS`, `PROVIDER_API_KEY_CONFIG`, `get_model_run`, `providers_for_model_runs`, `_api_key_kwargs_for_providers`, and `configure_and_validate_provider_keys` helpers. Rename the local `get_model_run` wrapper to `get_forecastbench_model_run` only if import ambiguity becomes confusing; otherwise keep the existing public function and implement it as lookup in `MODEL_RUNS_BY_NAME`.

- [ ] **Step 5: Update transcript output to use `provider_model_id`**

In `/workspace/forecastbench/src/llm_forecaster/runner.py`, change transcript model ID access from:

```python
f"- Model ID: {getattr(model_run, 'model_id', model_run.name)}",
```

to:

```python
f"- Provider model ID: {model_run.provider_model_id}",
```

Update the related test assertion in `/workspace/forecastbench/src/tests/llm_forecaster/test_runner.py` from `Model ID` to `Provider model ID`.

- [ ] **Step 6: Run focused ForecastBench LLM tests**

Run:

```bash
cd /workspace/forecastbench
PYTHONSAFEPATH=1 PYTHONPATH=src:/workspace/utils python -m pytest \
  src/tests/llm_forecaster/test_model_runs.py \
  src/tests/llm_forecaster/test_runner.py \
  src/tests/orchestration/test_llm_forecaster_worker.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit ForecastBench model-run selection migration**

Run:

```bash
cd /workspace/forecastbench
git add requirements.runtime.txt src/llm_forecaster/model_runs.py \
  src/llm_forecaster/runner.py src/tests/llm_forecaster/test_model_runs.py \
  src/tests/llm_forecaster/test_runner.py src/tests/orchestration/test_llm_forecaster_worker.py
git commit -m "refactor(llm): select shared model runs"
```

---

### Task 6: Move ForecastBench Release Dates to Utils and Delete CSV

**Files:**
- Modify: `/workspace/forecastbench/src/leaderboard/model_identities.py`
- Modify: `/workspace/forecastbench/src/leaderboard/main.py`
- Modify: `/workspace/forecastbench/src/leaderboard/Makefile`
- Delete: `/workspace/forecastbench/src/leaderboard/model_release_dates.csv`
- Modify: `/workspace/forecastbench/src/tests/leaderboard/test_model_identities.py`
- Modify: `/workspace/forecastbench/src/tests/leaderboard/test_llm_legacy_names.py`

- [ ] **Step 1: Write failing tests for utils release-date lookup and CSV deletion**

In `/workspace/forecastbench/src/tests/leaderboard/test_model_identities.py`, add:

```python
from datetime import date
from pathlib import Path


def test_model_release_dates_come_from_utils_not_csv():
    release_dates = model_identities.read_model_release_dates()

    assert release_dates.loc[
        release_dates["model_key"] == "gpt-5.4-2026-03-05",
        "model_release_date",
    ].iloc[0] == date(2026, 3, 5)
    assert release_dates.loc[
        release_dates["model_key"] == "gemini-3.1-flash-lite",
        "model_release_date",
    ].iloc[0] == date(2026, 5, 8)


def test_forecastbench_model_release_dates_csv_is_removed():
    root = Path(__file__).resolve().parents[2]
    assert not (root / "leaderboard" / "model_release_dates.csv").exists()
```

Update existing identity tests so current generated run variants map to base `model_key`:

```python
def test_identity_for_current_shared_model_run_uses_model_key():
    assert (
        model_identities.identity_for_forecastbench_model(
            "claude-opus-4-7-1024"
        )
        == "claude-opus-4-7"
    )
    assert (
        model_identities.identity_for_forecastbench_model(
            "claude-opus-4-7-1024 with freeze values"
        )
        == "claude-opus-4-7"
    )
```

- [ ] **Step 2: Run failing leaderboard tests**

Run:

```bash
cd /workspace/forecastbench
PYTHONSAFEPATH=1 PYTHONPATH=src:/workspace/utils python -m pytest \
  src/tests/leaderboard/test_model_identities.py \
  src/tests/leaderboard/test_llm_legacy_names.py -q
```

Expected: FAIL because `read_model_release_dates()` still expects a CSV path and the CSV still exists.

- [ ] **Step 3: Update `model_identities.py` to read utils release dates**

Change `/workspace/forecastbench/src/leaderboard/model_identities.py` so `read_model_release_dates` takes no path and builds a DataFrame from utils:

```python
from utils.llm import model_runs as shared_model_runs


def read_model_release_dates() -> pd.DataFrame:
    """Read LLM model release dates keyed by canonical model_key from utils."""
    rows = [
        {"model_key": model_key, "model_release_date": release_date}
        for model_key, release_date in shared_model_runs.model_release_dates_by_key().items()
    ]
    return pd.DataFrame(rows)
```

Change identity naming from `model_identity` to `model_key` in this helper module. `add_model_identity()` can either be renamed to `add_model_key()` or kept as a compatibility wrapper that writes `model_key`; use one column name consistently in `leaderboard/main.py`.

For current model-run lookup, use shared run keys:

```python
def identity_for_forecastbench_model(model: str) -> str:
    """Return the release-date model_key for a ForecastBench model display name."""
    base_model = _without_freeze_values_suffix(model)
    try:
        return shared_model_runs.get_model_run(base_model).model_key
    except KeyError:
        pass
    ...
```

Keep the strict historical legacy mapping, but make targets canonical `model_key` strings.

- [ ] **Step 4: Update `leaderboard/main.py` to stop reading the CSV**

Change:

```python
df_release_dates = model_identities.read_model_release_dates("model_release_dates.csv")
```

to:

```python
df_release_dates = model_identities.read_model_release_dates()
```

In `get_model_release_date_info()`, merge ForecastBench-owned rows on `model_key` instead of `model` or `model_identity`:

```python
df_forecastbench_models = model_identities.add_model_key(
    df[df["organization"] == constants.BENCHMARK_NAME]
)
...
df_with_release_dates = pd.merge(
    df_forecastbench_models,
    df_release_dates,
    how="inner",
    on="model_key",
)
```

Keep external submissions behavior unchanged:

```python
df_external_submissions["model_release_date"] = df_external_submissions[
    "first_forecast_due_date"
]
```

- [ ] **Step 5: Remove the CSV from deploy packaging**

In `/workspace/forecastbench/src/leaderboard/Makefile`, remove `model_release_dates.csv` from both deploy prerequisite lines:

```make
deploy : main.py requirements.txt Dockerfile .gcloudignore
deploy-dataset : main.py requirements.txt Dockerfile .gcloudignore
```

- [ ] **Step 6: Delete the CSV**

Run:

```bash
cd /workspace/forecastbench
git rm src/leaderboard/model_release_dates.csv
```

- [ ] **Step 7: Run leaderboard tests**

Run:

```bash
cd /workspace/forecastbench
PYTHONSAFEPATH=1 PYTHONPATH=src:/workspace/utils python -m pytest \
  src/tests/leaderboard/test_model_identities.py \
  src/tests/leaderboard/test_llm_legacy_names.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit ForecastBench release-date migration**

Run:

```bash
cd /workspace/forecastbench
git add src/leaderboard/model_identities.py src/leaderboard/main.py \
  src/leaderboard/Makefile src/tests/leaderboard/test_model_identities.py \
  src/tests/leaderboard/test_llm_legacy_names.py
git add -u src/leaderboard/model_release_dates.csv
git commit -m "refactor(leaderboard): read llm release dates from utils"
```

---

### Task 7: Add ForecastBench Anti-Drift Tests

**Files:**
- Create: `/workspace/forecastbench/src/tests/test_shared_llm_model_runs.py`

- [ ] **Step 1: Write anti-drift tests**

Create `/workspace/forecastbench/src/tests/test_shared_llm_model_runs.py`:

```python
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ALLOWLIST = {
    ROOT / "src" / "tests" / "test_shared_llm_model_runs.py",
}


def iter_python_files():
    for path in sorted((ROOT / "src").rglob("*.py")):
        if "upload" in path.parts or path in ALLOWLIST:
            continue
        yield path


def test_forecastbench_does_not_declare_local_model_runs():
    forbidden = [
        "class ModelRun",
        "ModelRun(",
        "OPENAI_MODEL_RUNS = [",
        "TOGETHER_MODEL_RUNS = [",
        "ANTHROPIC_MODEL_RUNS = [",
        "XAI_MODEL_RUNS = [",
        "GOOGLE_MODEL_RUNS = [",
    ]

    offenders = []
    for path in iter_python_files():
        text = path.read_text()
        for needle in forbidden:
            if needle in text:
                offenders.append(f"{path.relative_to(ROOT)} contains {needle}")

    assert offenders == []


def test_forecastbench_declares_selected_shared_model_run_keys_only():
    from llm_forecaster import model_runs
    from utils.llm import model_runs as shared_model_runs

    assert model_runs.MODEL_RUNS == shared_model_runs.select_model_runs(
        model_runs.FORECASTBENCH_MODEL_RUN_KEYS
    )
```

- [ ] **Step 2: Run anti-drift tests**

Run:

```bash
cd /workspace/forecastbench
PYTHONSAFEPATH=1 PYTHONPATH=src:/workspace/utils python -m pytest \
  src/tests/test_shared_llm_model_runs.py -q
```

Expected: PASS.

- [ ] **Step 3: Run focused ForecastBench suite**

Run:

```bash
cd /workspace/forecastbench
PYTHONSAFEPATH=1 PYTHONPATH=src:/workspace/utils python -m pytest \
  src/tests/llm_forecaster \
  src/tests/orchestration/test_llm_forecaster_worker.py \
  src/tests/orchestration/test_llm_forecaster_manager.py \
  src/tests/leaderboard/test_model_identities.py \
  src/tests/leaderboard/test_llm_legacy_names.py \
  src/tests/test_shared_llm_model_runs.py -q
```

Expected: PASS.

- [ ] **Step 4: Commit anti-drift tests**

Run:

```bash
cd /workspace/forecastbench
git add src/tests/test_shared_llm_model_runs.py
git commit -m "test(llm): prevent local model-run registries"
```

---

### Task 8: Migrate TimeSeriesBench to Shared Model Runs

**Files:**
- Modify: `/workspace/time-series-benchmark/requirements.txt`
- Modify: `/workspace/time-series-benchmark/src/helpers/constants.py`
- Modify: `/workspace/time-series-benchmark/src/models/llms/common.py`
- Modify: `/workspace/time-series-benchmark/src/models/llms/manager/main.py`
- Modify: `/workspace/time-series-benchmark/src/models/llms/worker/main.py`
- Modify: `/workspace/time-series-benchmark/src/models/llms/smoke_test/main.py`
- Create: `/workspace/time-series-benchmark/tests/test_shared_llm_model_runs.py`

- [ ] **Step 1: Write failing TimeSeriesBench selected-key and anti-drift tests**

Create `/workspace/time-series-benchmark/tests/test_shared_llm_model_runs.py`:

```python
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ALLOWLIST = {ROOT / "tests" / "test_shared_llm_model_runs.py"}


def test_timeseriesbench_selected_model_runs_are_shared_utils_runs():
    from helpers import constants
    from utils.llm import model_runs as shared_model_runs

    assert constants.TIMESERIESBENCH_MODEL_RUN_KEYS == [
        "gpt-4o-mini-2024-07-18",
        "gpt-5-nano-2025-08-07",
        "gpt-5-mini-2025-08-07",
        "gpt-5.4-nano-2026-03-17",
        "gpt-5.4-mini-2026-03-17",
        "gpt-5.4-2026-03-05-high",
        "gpt-5.4-2026-03-05-high-web-search",
        "gpt-5.5-2026-04-23-medium",
        "gpt-5.5-2026-04-23-high",
        "gpt-5.5-2026-04-23-high-web-search",
        "deepseek-v3.1",
        "claude-haiku-4-5-20251001-4096",
        "claude-sonnet-4-5-20250929-4096",
        "claude-sonnet-4-6-4096",
        "claude-sonnet-4-6-adaptive-thinking-16000",
        "claude-opus-4-6-4096",
        "claude-opus-4-7-4096",
        "claude-opus-4-7-adaptive-thinking-high-24000",
        "claude-opus-4-7-adaptive-thinking-high-web-search-64000",
        "grok-4-1-fast-reasoning",
        "grok-4-1-fast-non-reasoning",
        "grok-4.20-0309-reasoning",
        "grok-4.20-0309-reasoning-web-search-x-search",
        "grok-4.20-0309-non-reasoning",
        "gemini-2.5-pro",
        "gemini-2.5-pro-web-search",
        "gemini-3-flash-preview",
        "gemini-3.1-flash-lite-preview",
        "gemini-3.1-pro-preview",
    ]
    assert constants.MODEL_RUNS == shared_model_runs.select_model_runs(
        constants.TIMESERIESBENCH_MODEL_RUN_KEYS
    )
    assert constants.REFORMAT_MODEL == shared_model_runs.get_model_run(
        "gpt-5-mini-2025-08-07-1024"
    )


def test_timeseriesbench_does_not_declare_local_model_runs():
    forbidden = [
        "class ModelRun",
        "ModelRun(",
        "OPENAI_RUNS = [",
        "TOGETHER_RUNS = [",
        "ANTHROPIC_RUNS = [",
        "XAI_RUNS = [",
        "GOOGLE_RUNS = [",
        "MISTRAL_RUNS",
    ]

    offenders = []
    for path in sorted((ROOT / "src").rglob("*.py")):
        if "upload" in path.parts or path in ALLOWLIST:
            continue
        text = path.read_text()
        for needle in forbidden:
            if needle in text:
                offenders.append(f"{path.relative_to(ROOT)} contains {needle}")

    assert offenders == []
```

- [ ] **Step 2: Run the failing TimeSeriesBench test**

Run:

```bash
cd /workspace/time-series-benchmark
PYTHONPATH=src:/workspace/utils python -m pytest tests/test_shared_llm_model_runs.py -q
```

Expected: FAIL because local `ModelRun` declarations still exist and `TIMESERIESBENCH_MODEL_RUN_KEYS` does not exist.

- [ ] **Step 3: Update the utils pin**

Run:

```bash
cd /workspace/time-series-benchmark
UTILS_SHA=$(git -C /workspace/utils rev-parse HEAD)
python - <<'PY'
from pathlib import Path
import subprocess

sha = subprocess.check_output(
    ["git", "-C", "/workspace/utils", "rev-parse", "HEAD"],
    text=True,
).strip()
path = Path("requirements.txt")
lines = path.read_text().splitlines()
lines = [
    line if not line.startswith("git+https://github.com/forecastingresearch/utils@")
    else f"git+https://github.com/forecastingresearch/utils@{sha}#egg=fri-utils"
    for line in lines
]
path.write_text("\\n".join(lines) + "\\n")
PY
```

- [ ] **Step 4: Replace TimeSeriesBench local model runs with shared selections**

In `/workspace/time-series-benchmark/src/helpers/constants.py`:

1. Keep `LLM_NUM_RETRIES = 3`.
2. Delete the local `ModelRun` dataclass.
3. Delete `OPENAI_RUNS`, `TOGETHER_RUNS`, `ANTHROPIC_RUNS`, `XAI_RUNS`, `GOOGLE_RUNS`, `MISTRAL_RUNS`, duplicate validation, and local `REFORMAT_MODEL`.
4. Add:

```python
from utils.llm.model_runs import ModelRun, get_model_run, select_model_runs

TIMESERIESBENCH_MODEL_RUN_KEYS = [
    "gpt-4o-mini-2024-07-18",
    "gpt-5-nano-2025-08-07",
    "gpt-5-mini-2025-08-07",
    "gpt-5.4-nano-2026-03-17",
    "gpt-5.4-mini-2026-03-17",
    "gpt-5.4-2026-03-05-high",
    "gpt-5.4-2026-03-05-high-web-search",
    "gpt-5.5-2026-04-23-medium",
    "gpt-5.5-2026-04-23-high",
    "gpt-5.5-2026-04-23-high-web-search",
    "deepseek-v3.1",
    "claude-haiku-4-5-20251001-4096",
    "claude-sonnet-4-5-20250929-4096",
    "claude-sonnet-4-6-4096",
    "claude-sonnet-4-6-adaptive-thinking-16000",
    "claude-opus-4-6-4096",
    "claude-opus-4-7-4096",
    "claude-opus-4-7-adaptive-thinking-high-24000",
    "claude-opus-4-7-adaptive-thinking-high-web-search-64000",
    "grok-4-1-fast-reasoning",
    "grok-4-1-fast-non-reasoning",
    "grok-4.20-0309-reasoning",
    "grok-4.20-0309-reasoning-web-search-x-search",
    "grok-4.20-0309-non-reasoning",
    "gemini-2.5-pro",
    "gemini-2.5-pro-web-search",
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite-preview",
    "gemini-3.1-pro-preview",
]

MODEL_RUNS: list[ModelRun] = select_model_runs(TIMESERIESBENCH_MODEL_RUN_KEYS)
REFORMAT_MODEL: ModelRun = get_model_run("gpt-5-mini-2025-08-07-1024")
```

- [ ] **Step 5: Update type hints if imports fail**

In `/workspace/time-series-benchmark/src/models/llms/common.py`, keep:

```python
def providers_for_model_runs(model_runs: Sequence[constants.ModelRun]) -> list:
```

This remains valid because `constants.ModelRun` now refers to the imported utils class. If flake8 flags it, import the type directly:

```python
from utils.llm.model_runs import ModelRun
```

and change the annotation to:

```python
def providers_for_model_runs(model_runs: Sequence[ModelRun]) -> list:
```

- [ ] **Step 6: Run TimeSeriesBench focused tests**

Run:

```bash
cd /workspace/time-series-benchmark
PYTHONPATH=src:/workspace/utils python -m pytest tests/test_shared_llm_model_runs.py -q
```

Expected: PASS.

- [ ] **Step 7: Run LLM smoke-test unit coverage if available**

Run:

```bash
cd /workspace/time-series-benchmark
PYTHONPATH=src:/workspace/utils python -m pytest tests src/models/llms -q
```

Expected: PASS or a clear list of unrelated pre-existing test collection gaps. Fix failures caused by the shared model-run migration before continuing.

- [ ] **Step 8: Commit TimeSeriesBench migration**

Run:

```bash
cd /workspace/time-series-benchmark
git add requirements.txt src/helpers/constants.py src/models/llms/common.py \
  src/models/llms/manager/main.py src/models/llms/worker/main.py \
  src/models/llms/smoke_test/main.py tests/test_shared_llm_model_runs.py
git commit -m "refactor(llm): select shared model runs"
```

---

### Task 9: Final Cross-Repo Verification

**Files:**
- No planned file edits.

- [ ] **Step 1: Verify utils**

Run:

```bash
cd /workspace/utils
python -m pytest tests/unit/test_llm_model_runs.py tests/unit/test_llm_routing.py tests/unit/test_model_registry_config.py -q
```

Expected: PASS.

- [ ] **Step 2: Verify live shared model runs when credentials are available**

Run:

```bash
cd /workspace/utils
python -m pytest tests/integration/llm/test_model_runs.py --integration -q
```

Expected: PASS. This makes one live API call per shared model run and verifies
that provider SDKs accept the declared options. If the local environment lacks
provider credentials, record that this was not run locally and run it in the
credentialed integration environment before using the shared registry for a
forecasting round.

- [ ] **Step 3: Verify ForecastBench focused suite**

Run:

```bash
cd /workspace/forecastbench
PYTHONSAFEPATH=1 PYTHONPATH=src:/workspace/utils python -m pytest \
  src/tests/llm_forecaster \
  src/tests/orchestration/test_llm_forecaster_worker.py \
  src/tests/orchestration/test_llm_forecaster_manager.py \
  src/tests/leaderboard/test_model_identities.py \
  src/tests/leaderboard/test_llm_legacy_names.py \
  src/tests/test_shared_llm_model_runs.py -q
```

Expected: PASS.

- [ ] **Step 4: Verify TimeSeriesBench focused suite**

Run:

```bash
cd /workspace/time-series-benchmark
PYTHONPATH=src:/workspace/utils python -m pytest tests/test_shared_llm_model_runs.py -q
```

Expected: PASS.

- [ ] **Step 5: Confirm pins point to committed utils**

Run:

```bash
UTILS_SHA=$(git -C /workspace/utils rev-parse HEAD)
grep "$UTILS_SHA" /workspace/forecastbench/requirements.runtime.txt
grep "$UTILS_SHA" /workspace/time-series-benchmark/requirements.txt
```

Expected: both grep commands print the same utils SHA.

- [ ] **Step 6: Check git status in all repos**

Run:

```bash
git -C /workspace/utils status --short --branch
git -C /workspace/forecastbench status --short --branch
git -C /workspace/time-series-benchmark status --short --branch
```

Expected: only known unrelated untracked files remain. No tracked modifications from this plan are left uncommitted.

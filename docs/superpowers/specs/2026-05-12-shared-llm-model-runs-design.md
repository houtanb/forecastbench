# Shared LLM model runs across ForecastBench and TimeSeriesBench

**Date:** 2026-05-12
**Status:** Approved design; implementation plan pending

## Problem

ForecastBench and TimeSeriesBench currently declare LLM model runs independently.
Both codebases use nearly the same `ModelRun` shape, but the active run lists and
run names live in benchmark-local files. This allows the same provider call to be
named differently across benchmarks, for example:

- `gpt-5.4-2026-03-05-high-web-search`
- `gpt-5.4-2026-03-05-web-search-high`

That breaks cross-benchmark comparison, creates duplicate maintenance work, and
makes option-sensitive run names depend on whichever benchmark added the model
first. ForecastBench also stores LLM release dates in
`src/leaderboard/model_release_dates.csv`, which duplicates model metadata that
should be shared with TimeSeriesBench.

## Goals

- Move canonical LLM model metadata, model-run declarations, model-run naming,
  and LLM release dates into `/workspace/utils`.
- Make utils the only source of truth for named LLM model-run variants.
- Let ForecastBench and TimeSeriesBench select model runs by shared
  `model_run_key`.
- Generate model-run names from model metadata and name-relevant options instead
  of manually writing `name=...`.
- Keep provider API route IDs separate from displayed/canonical model keys.
- Delete ForecastBench `src/leaderboard/model_release_dates.csv`; LLM release
  dates come from utils.
- Preserve historical ForecastBench forecast files. Historical names are mapped
  to canonical utils `model_key` values during leaderboard ingest.
- Add tests in both benchmarks that fail if local `ModelRun` declarations are
  reintroduced.

## Non-goals

- Do not rename historical forecast files.
- Do not change forecast JSON schemas in either benchmark.
- Do not require both benchmarks to run the exact same selected subset on every
  round. They share a registry but each benchmark chooses its active run list.
- Do not move ForecastBench prompt variants or TimeSeriesBench forecast-type
  logic into utils.
- Do not make provider wrappers infer benchmark-specific defaults. Runtime
  options remain explicit on shared `ModelRun` declarations.

## Terminology

Use distinct names for the three identities involved in an LLM call:

- `model_key`: canonical base model key used for display identity and release
  dates, for example `deepseek-v3.1`.
- `provider_model_id`: exact model identifier sent to the provider API, for
  example `deepseek-ai/DeepSeek-V3.1`.
- `model_run_key`: generated full run key, including relevant options, for
  example `gpt-5.5-2026-04-23-high-web-search`.

Avoid `model_id` in the shared model-run API because it is ambiguous between the
canonical model and the provider route. Avoid `run_key` because it is not clear
enough without the model prefix.

## Utils API

Create `utils.llm.model_runs` with these core types:

```python
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
        """Return the generated full run key."""
        return build_model_run_key(self.model.model_key, self.options)

    @property
    def name(self) -> str:
        """Compatibility alias for benchmark code during migration."""
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
    def model_organization(self) -> str:
        """Return the display organization for benchmark outputs."""
        return self.model.lab.leaderboard_name

    def get_response(self, prompt: str, **kwargs: Any) -> str:
        """Request a response from the configured provider."""
```

`ModelRun.get_response()` calls `utils.llm.model_registry.get_response()` with:

- `provider=self.model.provider`
- `model_id=self.model.provider_model_id`
- `options={**self.options, **kwargs}`

The provider wrapper API can keep the parameter name `model_id` because that
function is already provider-route oriented. The shared run object should expose
the clearer `provider_model_id`.

## Canonical Models

The utils canonical model registry owns one entry per base model:

```python
LLMModel(
    model_key="deepseek-v3.1",
    provider_model_id="deepseek-ai/DeepSeek-V3.1",
    lab=LABS["DeepSeek"],
    provider=PROVIDERS["Together"],
    release_date=date(2025, 8, 21),
    token_limit=128_000,
)
```

For providers such as Together, `model_key` is explicit and not derived from the
slash-delimited `provider_model_id`. This keeps displayed names and forecast
filenames stable:

- display/key: `deepseek-v3.1`
- provider route: `deepseek-ai/DeepSeek-V3.1`

`LLMModel.release_date` is required for benchmark LLMs. A model missing a known
release date is not added to the shared active run registry until the date is
supplied.

## Model Runs

The shared model-run registry owns named variants:

```python
ModelRun(model=LLM_MODELS["gpt-5.5-2026-04-23"])

ModelRun(
    model=LLM_MODELS["gpt-5.5-2026-04-23"],
    options={"reasoning": {"effort": "high"}},
)

ModelRun(
    model=LLM_MODELS["gpt-5.5-2026-04-23"],
    options={
        "reasoning": {"effort": "high"},
        "tools": [{"type": "web_search"}],
    },
)
```

These produce:

- `gpt-5.5-2026-04-23`
- `gpt-5.5-2026-04-23-high`
- `gpt-5.5-2026-04-23-high-web-search`

Both benchmarks select from shared registry keys:

```python
FORECASTBENCH_MODEL_RUN_KEYS = [
    "gpt-5.4-2026-03-05",
    "gpt-5.2-2025-12-11",
    "claude-opus-4-7-1024",
]

MODEL_RUNS = select_model_runs(FORECASTBENCH_MODEL_RUN_KEYS)
```

TimeSeriesBench uses the same API with its own selected list. Benchmarks do not
declare local `ModelRun(...)` objects.

## Naming Rules

Model-run keys are generated by an ordered naming policy. The policy is data and
function driven so new options can be added without rewriting every model run.

Conceptually:

```python
NAME_COMPONENT_RULES = (
    reasoning_effort_rule,
    thinking_mode_rule,
    tool_rule,
    token_limit_rule,
)

NAME_NEUTRAL_OPTION_PATHS = {
    ("temperature",),
    ("candidate_count",),
    ("automatic_function_calling",),
}
```

Each rule consumes one or more option paths and emits zero or more suffix
components. Validation fails if a `ModelRun` contains an option path that is
neither consumed by a naming rule nor explicitly listed as name-neutral.

This makes the naming convention extensible:

1. add a rule for a new performance-relevant option;
2. add tests showing the emitted suffix order;
3. existing model-run declarations automatically use the new naming rule.

Initial suffix order:

1. thinking mode, for example `adaptive-thinking`;
2. reasoning/output effort, for example `high`;
3. tools in canonical tool order, for example `web-search`, then `x-search`;
4. token/output cap, for example `4096`, `10000`, or `64000`.

Examples:

- `gpt-5.4-2026-03-05-high-web-search`
- `claude-opus-4-7-adaptive-thinking-high-web-search-64000`
- `grok-4.20-0309-reasoning-web-search-x-search-10000`
- `deepseek-v3.1-10000`

Name-neutral options include operational controls that should not affect display
identity, such as `temperature`, Google `candidate_count`, and Google
`automatic_function_calling`.

## Release Dates

LLM release dates move from ForecastBench CSV metadata into utils:

```python
def model_release_dates_by_key() -> dict[str, date]:
    """Return release dates keyed by canonical model_key."""
```

Every `ModelRun` inherits its release date through `model.model_key`.
ForecastBench deletes `src/leaderboard/model_release_dates.csv` as part of this
change.

ForecastBench leaderboard release-date flow becomes:

1. ForecastBench-owned rows are mapped to a canonical `model_key`.
2. Current shared model-run names map through utils.
3. Historical LLM names map through the strict legacy-name table.
4. Release dates are looked up from utils by `model_key`.
5. ForecastBench LLM rows missing a utils release date raise or fail tests.
6. External submissions keep the current behavior: release date equals
   `first_forecast_due_date`.

Pseudo-models such as always-0.5 or public median do not get release dates from a
CSV. If they are not external submissions, release-date-specific analyses exclude
them rather than representing them with blank CSV rows.

TimeSeriesBench should use the same utils release-date source anywhere it needs
LLM release metadata.

## Historical ForecastBench Names

ForecastBench keeps a strict legacy mapping for historical forecast files. The
mapping target changes from canonical display run names to canonical
`model_key` values where release-date metadata is needed.

Example:

```python
("DeepSeek", "DeepSeek-V3.1") -> "deepseek-v3.1"
("OpenAI", "GPT-5.4-2026-03-05") -> "gpt-5.4-2026-03-05"
```

Variant suffixes such as `(zero shot)`, `(scratchpad)`, and
`zero-shot-with-freeze-values` are ignored for release-date identity unless the
underlying provider model differs. Forecast file names remain unchanged.

Displayed model names for new files should use shared `model_run_key` values.

## Benchmark Enforcement

Both benchmarks add guardrail tests that scan their source trees and fail if
local model-run declarations are reintroduced. The tests reject:

- `class ModelRun`
- `ModelRun(` outside allowlisted compatibility tests or fixture snippets
- benchmark-local grouped run registries such as `OPENAI_RUNS`,
  `ANTHROPIC_MODEL_RUNS`, `GOOGLE_RUNS`, and `MODEL_RUNS = [` declarations

Allowed benchmark code:

- imports `ModelRun` only in type annotations for shared model-run objects;
- imports `select_model_runs`, `get_model_run`, and shared registry constants;
- declares lists of `model_run_key` strings.

This is not a security boundary. It is a CI and review guardrail so drift is
caught quickly.

## Migration Plan Shape

Implementation is split into three repo phases:

1. **Utils phase**
   - add `LLMModel`, shared `ModelRun`, naming rules, canonical model registry,
     shared model-run registry, release-date lookup, and tests;
   - keep existing `utils.llm.model_registry.Model` behavior unless it can be
     safely bridged without breaking current callers;
   - commit utils and record the SHA.

2. **ForecastBench phase**
   - update the utils pin;
   - replace local `src/llm_forecaster/model_runs.py` declarations with shared
     selections;
   - update leaderboard identity/release-date logic to use utils;
   - delete `src/leaderboard/model_release_dates.csv`;
   - add local anti-drift tests;
   - keep forecast schema and historical files unchanged.

3. **TimeSeriesBench phase**
   - update the utils pin;
   - replace `src/helpers/constants.py` local LLM `ModelRun` declarations with
     shared selections;
   - update worker, manager, smoke test, and helper imports;
   - add local anti-drift tests.

Each phase is committed separately. ForecastBench and TimeSeriesBench do not
point at an unreleased or uncommitted utils state.

## Testing

Utils tests:

- `model_key` and `provider_model_id` stay distinct for Together-hosted models.
- `model_run_key` generation follows the global suffix order.
- name-neutral options do not appear in keys.
- unknown option paths fail validation.
- every shared `ModelRun` has a unique `model_run_key`.
- every shared active `LLMModel` has a release date.
- a live integration smoke test can call every shared `ModelRun` with a tiny
  prompt using its declared options, so invalid provider option payloads are
  caught before benchmark forecasting runs.

ForecastBench tests:

- selected run keys all exist in utils.
- forecast output uses `model_run_key` and unchanged schema.
- legacy LLM names strictly map to utils `model_key` values.
- release-date lookup uses utils and no CSV file is read.
- `src/leaderboard/model_release_dates.csv` is absent.
- anti-drift scan rejects local `ModelRun` declarations.

TimeSeriesBench tests:

- selected run keys all exist in utils.
- output filenames use shared `model_run_key` values.
- smoke test selection works by shared key.
- anti-drift scan rejects local `ModelRun` declarations.

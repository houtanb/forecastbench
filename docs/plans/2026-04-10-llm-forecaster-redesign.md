# LLM Forecaster Redesign

## Goal

Improve the LLM forecasting pipeline (`src/base_eval/llm_baselines/`) for speed, maintainability, and testability. Converge on a single LLM calling layer in the shared `utils/llm` library.

## Decisions

### LLM Calling Layer

Use native provider SDKs via `utils/llm`. No LiteLLM. Each provider has a thin wrapper that passes `**options` straight through to the SDK, so provider-specific params (e.g., `reasoning_effort`, `tools`, `thinking`) work without updating provider code.

### Concurrency Model

`ThreadPoolExecutor` per worker, not async. Thread pool size = `group_rate_limit // group_size`, where `group_size` is the number of sibling workers sharing a rate limit group. The manager computes this and passes it as an env var.

### Rate Limiting

Rate limits are scoped per **rate limit group**:
- **Anthropic**: grouped by model family (`opus`, `sonnet`, `haiku`)
- **All other providers** (OpenAI, xAI, Google, Together): grouped by provider name

Each worker gets told how many siblings share its group. It divides the group's max concurrent requests by that count to size its thread pool. Rate limits live in a config dict alongside the model registry.

### Model Configuration

Each model run is a dataclass:

```python
@dataclass
class ModelRun:
    name: str                  # e.g. "claude-opus-4-6-high"
    model_id: str              # e.g. "claude-opus-4-6"
    provider: Provider         # OPENAI, ANTHROPIC, GOOGLE, XAI, TOGETHER
    options: dict              # passed through to SDK: temperature, reasoning_effort, tools, etc.
    rate_limit_group: str = "" # defaults to provider.value if empty

    def __post_init__(self):
        if not self.rate_limit_group:
            self.rate_limit_group = self.provider.value
```

Different reasoning levels are separate `ModelRun` entries (separate Cloud Run workers):

```python
ModelRun(name="claude-opus-4-6-medium", model_id="claude-opus-4-6", provider=Provider.ANTHROPIC,
         rate_limit_group="opus", options={"reasoning_effort": "medium", "max_tokens": 16000})
ModelRun(name="claude-opus-4-6-high", model_id="claude-opus-4-6", provider=Provider.ANTHROPIC,
         rate_limit_group="opus", options={"reasoning_effort": "high", "max_tokens": 16000})
```

### TEST Mode

Run the worker directly (skip the manager) with `MODEL_TO_TEST=<model_name>` and `TEST_OR_PROD=TEST`. The worker:
1. Picks a small sample of questions (2 market, 2 dataset)
2. Runs inference through the full pipeline
3. Validates the output (see Validation below)
4. Exits with error code if validation fails

### Validation Summary

Runs after **every** worker run (TEST and PROD). Logs an emoji-formatted summary:

```
📊 Forecast Summary: grok-5 (zero_shot)
├── 📁 File: 2026-04-10.xAI.grok-5_zero_shot.json
├── ✅ Valid JSON structure
├── ✅ All probabilities in (0, 1)
├── ✅ Market questions: 376/378 forecasted (99.5%)
└── ✅ Dataset questions: 618/620 forecasted (99.7%)
```

- `✅` if ≥95% of questions received a forecast
- `❌` if <95%

In TEST mode: exits with error on any validation failure. In PROD mode: logs and continues.

## Changes Required

### utils/llm

1. **kwargs passthrough**: Each provider's `_call_model()` passes `**options` to the SDK instead of cherry-picking `temperature` and `max_tokens`. Provider-specific translation (e.g., Anthropic's `max_tokens` handling) stays, but unknown kwargs flow through.
2. **Add `rate_limit_group`**: Optional field on `Model`, defaults to `provider.value`.
3. **Remove `reasoning_model` flag**: Replace with checking for `reasoning_effort` in options (e.g., OpenAI skips `temperature` when `reasoning_effort` is present).

### forecastbench (`src/base_eval/llm_baselines/`)

1. **Worker**: Replace native SDK calls in `model_eval.py` with `utils/llm` provider calls. Use `ThreadPoolExecutor` sized by rate limit budget.
2. **Manager**: Count model runs per rate limit group, pass `RATE_LIMIT_GROUP_SIZE` as env var to each worker.
3. **Model registry**: Define all `ModelRun` entries in forecastbench config (not in utils). Include per-group rate limits.
4. **TEST mode**: Accept `MODEL_TO_TEST` env var, run single model on small question sample.
5. **Validation**: Add post-run validation and summary logging to every worker run.

## Out of Scope

- Project 14 (time-series-benchmark) migration — separate effort
- Async/await — ThreadPoolExecutor is sufficient
- Changes to prompt templates or question processing logic

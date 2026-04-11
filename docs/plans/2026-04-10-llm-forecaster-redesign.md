# LLM Forecaster Redesign Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the legacy LLM calling code in forecastbench with the `utils/llm` provider pattern, adding kwargs passthrough, rate limit groups, ThreadPoolExecutor concurrency, per-run validation, and a TEST mode for smoke-testing new models.

**Architecture:** The `utils/llm` library provides per-provider SDK wrappers. Forecastbench defines a `ModelRun` registry (provider, options, rate_limit_group) and a worker that processes questions concurrently via `ThreadPoolExecutor`. The manager counts siblings per rate limit group and passes this to workers. Every run ends with a validation summary.

**Tech Stack:** Python 3.11+, native provider SDKs (openai, anthropic, google-genai, together), pytest, ThreadPoolExecutor

---

## Execution Environment

This plan is designed to run in a **single session** from the **forecastbench** repo.

Changes to the `utils/llm` package are staged in `src/helpers/utils_llm/`, mirroring the exact directory structure of `utils/utils/llm/`. After the plan is complete, Jesse will manually copy the staged files into the real utils repo.

---

## Task 0: Update utils submodule to HEAD and fix all breakage

The utils submodule is currently at `34baafc`, which is 5 commits behind `origin/main` (`d207756`). Those commits include:
- Bounded retry loop (was infinite before)
- Added claude-sonnet-4-6
- Removed deprecated models
- Removed Mistral provider
- Lazy-loaded LLM submodule (added `utils/__init__.py` with eager imports of `gcp`, `archiving`, `helpers` and lazy import of `llm`)

**Files:**
- Modify: `utils` submodule pointer
- Potentially modify: every Cloud Run job's `requirements.txt` and `Makefile` under `src/`

### Step 1: Update the submodule

```bash
cd utils && git fetch origin && git checkout origin/main && cd ..
```

### Step 2: Audit every Cloud Run job for compatibility

Every Cloud Run job deploys by copying `utils/` into an upload directory (via `cp -r $(ROOT_DIR)utils $(UPLOAD_DIR)/` in the Makefile) and running `pip install -r requirements.txt`. The utils code itself is NOT pip-installed — it's imported directly from the copied directory.

For each Cloud Run job under `src/`, verify:

1. **Makefile** — the `deploy` target copies `utils/` and all other needed directories. Check that no Makefile references anything that was removed or restructured in the 5 new commits. The new `utils/__init__.py` eagerly imports `gcp`, `archiving`, and `helpers`, so any job that copies utils must also have `google-cloud-storage` in its `requirements.txt` (all of them already do).

2. **`requirements.txt`** — the utils repo's own `requirements.txt` now pins LLM SDK versions (`anthropic`, `openai`, `google-genai`, `together`). However, each Cloud Run job has its own `requirements.txt` that is independently maintained. Only jobs that actually import from `utils.llm` need those SDK deps. Since the lazy-load change ensures `utils.llm` is NOT imported unless explicitly requested, non-LLM jobs are safe.

3. **Imports** — verify that every job's `main.py` entry point still resolves. The standard pattern is `from utils import gcp` — this still works because `gcp` is eagerly imported in the new `utils/__init__.py`.

List of all Cloud Run jobs to check (each has a `Makefile` and `requirements.txt`):
- `src/base_eval/llm_baselines/manager/`
- `src/base_eval/llm_baselines/worker/`
- `src/base_eval/naive_and_dummy_forecasters/`
- `src/curate_questions/create_question_set/`
- `src/curate_questions/publish_question_set/`
- `src/leaderboard/`
- `src/metadata/tag_questions/`
- `src/metadata/validate_questions/`
- `src/nightly_update_workflow/manager/`
- `src/nightly_update_workflow/worker/`
- `src/nightly_update_workflow/compress_buckets/`
- `src/orchestration/func_resolve/`
- `src/questions/acled/{fetch,update_questions}/`
- `src/questions/dbnomics/{fetch,update_questions}/`
- `src/questions/fred/{fetch,update_questions}/`
- `src/questions/infer/{fetch,update_questions}/`
- `src/questions/manifold/{fetch,update_questions}/`
- `src/questions/metaculus/{fetch,update_questions}/`
- `src/questions/polymarket/{fetch,update_questions}/`
- `src/questions/wikipedia/{fetch,update_questions}/`
- `src/questions/yfinance/{fetch,update_questions}/`

For each job: read the Makefile's deploy target, the requirements.txt, and the main.py imports. Fix anything that's broken.

### Step 3: Run existing tests and lint

```bash
python -m pytest src/tests/ -v
make lint
```

Expected: All 187 existing tests PASS, lint clean. If any test fails, fix before proceeding.

### Step 4: Commit

```bash
git add utils
# Also git add any modified Makefiles or requirements.txt files
git commit -m "build: update utils submodule to HEAD and fix compatibility"
```

---

## Bootstrap Step: Stage utils/llm for local development

Copy the current `utils/utils/llm/` directory to `src/helpers/utils_llm/`:

```bash
cp -r utils/utils/llm src/helpers/utils_llm
```

Fix the two imports in `src/helpers/utils_llm/model_registry.py` that reach into the parent utils package — these won't resolve from the staging location:

1. Remove `from ..gcp.secret_manager import get_secret` and the GCP-based key loading in `configure_api_keys()` (forecastbench has its own key management in `helpers/keys.py`)
2. Remove `from ..helpers.constants import (...)` and inline the secret name strings as constants in the file

All other imports in the package are relative within the `llm/` package (e.g., `.providers.base`, `..utils`) and will resolve correctly from `src/helpers/utils_llm/`.

After copying and fixing imports, commit:
```bash
git add src/helpers/utils_llm/
git commit -m "build: stage utils/llm at HEAD (d207756) for local development"
```

---

## Repos and Key Paths

- **forecastbench repo**: `/Users/houtan/DocumentsLOCAL/FRI/project/10c/forecastbench/` (branch: `forecaster`)
- **Staged utils/llm**: `src/helpers/utils_llm/` (mirror of `utils/utils/llm/`)

### Staged utils_llm files (mirror of utils/llm)
- `src/helpers/utils_llm/__init__.py`
- `src/helpers/utils_llm/model_registry.py` — `Model` dataclass, `MODELS` list, `configure_api_keys()`
- `src/helpers/utils_llm/lab_registry.py` — `Lab` dataclass, `LABS` dict
- `src/helpers/utils_llm/utils.py` — `get_response_with_retry()`
- `src/helpers/utils_llm/providers/__init__.py`
- `src/helpers/utils_llm/providers/base.py` — `BaseLLMProvider`
- `src/helpers/utils_llm/providers/openai.py` — `OpenAIProvider`
- `src/helpers/utils_llm/providers/anthropic.py` — `AnthropicProvider`
- `src/helpers/utils_llm/providers/google.py` — `GoogleProvider`
- `src/helpers/utils_llm/providers/together.py` — `TogetherProvider`
- `src/helpers/utils_llm/providers/xai.py` — `XAIProvider`

### Forecastbench files
- `src/base_eval/llm_baselines/manager/main.py` — Cloud Run manager
- `src/base_eval/llm_baselines/worker/main.py` — Cloud Run worker
- `src/helpers/model_eval.py` — legacy LLM calls, question processing, forecast generation
- `src/helpers/llm.py` — LiteLLM-based `ModelRun` (to be replaced)
- `src/helpers/constants.py` — `RunMode`, `PROMPT_TYPES`, org constants
- `src/helpers/llm_prompts.py` — prompt templates
- `src/helpers/env.py` — environment variables
- `src/helpers/keys.py` — API key retrieval from GCP Secret Manager
- `src/helpers/question_curation.py` — `MARKET_SOURCES`, `DATA_SOURCES`

### Import convention

Within this plan, all imports from the staged utils code use:
```python
from helpers.utils_llm.model_registry import Model, configure_api_keys
from helpers.utils_llm.providers.openai import OpenAIProvider
# etc.
```

When Jesse moves the files to the real utils repo, these revert to:
```python
from utils.llm.model_registry import Model, configure_api_keys
from utils.llm.providers.openai import OpenAIProvider
```

---

## Task 1: kwargs passthrough in providers

Currently each provider cherry-picks `temperature` and `max_tokens`. Change them to pass `**options` through to the SDK so provider-specific params (`reasoning_effort`, `tools`, `thinking`) work without updating provider code.

**Files:**
- Modify: `src/helpers/utils_llm/providers/openai.py`
- Modify: `src/helpers/utils_llm/providers/anthropic.py`
- Modify: `src/helpers/utils_llm/providers/google.py`
- Modify: `src/helpers/utils_llm/providers/together.py`
- Modify: `src/helpers/utils_llm/providers/xai.py`
- Test: `src/tests/test_utils_llm_providers.py` (create)

### Step 1: Write failing tests for kwargs passthrough

Create `src/tests/test_utils_llm_providers.py`. Test that each provider passes unknown kwargs to the underlying SDK call. Use mocking to verify the SDK receives them.

Tests should cover:
- **OpenAI**: `reasoning_effort` is passed through; `temperature` is excluded when `reasoning_effort` is present; `max_tokens` is mapped to `max_output_tokens`
- **Anthropic**: `thinking` dict is passed through; `max_tokens` remains required
- **xAI**: `tools` list is passed through
- **Google**: extra kwargs (e.g., `thinking_config`) are passed into `GenerateContentConfig`
- **Together**: extra kwargs (e.g., `top_p`) are passed through

### Step 2: Run tests to verify they fail

Run: `python -m pytest src/tests/test_utils_llm_providers.py -v`
Expected: FAIL — kwargs are not being passed through yet.

### Step 3: Implement kwargs passthrough in each provider

**OpenAI (`providers/openai.py`):**
- Pop `temperature` and `max_tokens` from options explicitly
- Skip `temperature` if `reasoning_effort` is in remaining options
- Map `max_tokens` → `max_output_tokens`
- `request_payload.update(options)` for everything else
- Remove the `model.reasoning_model` check (field will be removed in Task 2)

**Anthropic (`providers/anthropic.py`):**
- Pop `max_tokens` (still required)
- `call_args.update(options)` for everything else (temperature, thinking, tools, etc.)

**Google (`providers/google.py`):**
- `config_kwargs.update(options)` — everything goes into `GenerateContentConfig`

**Together (`providers/together.py`):**
- `request_payload.update(options)` — everything goes to `chat.completions.create()`

**xAI (`providers/xai.py`):**
- `request_payload.update(options)` — everything goes to `chat.completions.create()`

### Step 4: Run tests to verify they pass

Run: `python -m pytest src/tests/test_utils_llm_providers.py -v`
Expected: PASS

### Step 5: Commit

```bash
git add src/helpers/utils_llm/providers/ src/tests/test_utils_llm_providers.py
git commit -m "feat(utils_llm): pass options through to provider SDKs"
```

---

## Task 2: Add `rate_limit_group` to Model, remove `reasoning_model`

**Files:**
- Modify: `src/helpers/utils_llm/model_registry.py`
- Test: `src/tests/test_utils_llm_model.py` (create)

### Step 1: Write failing tests

Tests should cover:
- `rate_limit_group` defaults to provider name when not set (e.g., `"openai"`)
- `rate_limit_group` can be explicitly overridden (e.g., `"opus"`)
- `reasoning_model` field no longer exists on `Model`

### Step 2: Run tests to verify they fail

Run: `python -m pytest src/tests/test_utils_llm_model.py -v`
Expected: FAIL

### Step 3: Implement changes to Model dataclass

In `src/helpers/utils_llm/model_registry.py`:
- Remove `reasoning_model: bool = False` from `Model`
- Add `rate_limit_group: str = ""` field
- Add `__post_init__` that sets `rate_limit_group` to provider name if empty (use `object.__setattr__` since dataclass is frozen)
- Update `MODELS` list: remove all `reasoning_model=True/False` args
- OpenAI provider no longer checks `model.reasoning_model` — it checks for `reasoning_effort` in options (done in Task 1)

### Step 4: Run tests to verify they pass

Run: `python -m pytest src/tests/test_utils_llm_model.py src/tests/test_utils_llm_providers.py -v`
Expected: PASS (both task 1 and task 2 tests)

### Step 5: Commit

```bash
git add src/helpers/utils_llm/model_registry.py src/tests/test_utils_llm_model.py
git commit -m "feat(utils_llm): add rate_limit_group to Model, remove reasoning_model"
```

---

## Task 3: Create forecastbench ModelRun registry

Replace the LiteLLM-based `llm.py` with a new model registry that delegates to `utils_llm.Model`.

**Files:**
- Modify: `src/helpers/llm.py` — replace entirely
- Test: `src/tests/test_llm_registry.py` (create)

### Step 1: Write failing tests

Tests should cover:
- `ModelRun.rate_limit_group` defaults to `provider.value` when not set
- `ModelRun.rate_limit_group` can be explicitly overridden (e.g., `"opus"`)
- No duplicate names in `MODEL_RUNS`
- All providers in `MODEL_RUNS` are valid `Provider` enum values
- All rate limit groups used by `MODEL_RUNS` have entries in `RATE_LIMITS`
- `ModelRun.get_response()` merges `self.options` with call-time kwargs

### Step 2: Run tests to verify they fail

Run: `python -m pytest src/tests/test_llm_registry.py -v`
Expected: FAIL

### Step 3: Rewrite `src/helpers/llm.py`

The new `llm.py` should contain:
- `Provider` enum: OPENAI, ANTHROPIC, GOOGLE, XAI, TOGETHER
- `ModelRun` dataclass: `name`, `model_id`, `provider`, `org`, `options`, `rate_limit_group` (defaults to `provider.value`)
- `ModelRun.get_response()`: creates a `utils_llm.Model` instance, merges options, delegates to `model.get_response()`
- `configure_keys()`: calls `utils_llm.configure_api_keys()` with keys from `helpers.keys`
- `RATE_LIMITS` dict: maps group name → max concurrent requests
- `MODEL_RUNS` list: all model entries ported from `main` branch's `constants.MODELS_TO_RUN` plus reasoning-level variants. Each old model maps to a `ModelRun` entry. For example:

  ```python
  # Old: {"source": "OAI", "org": "OpenAI", "full_name": "gpt-5.4-2026-03-05", ...}
  # New:
  ModelRun(name="gpt-5.4-2026-03-05", model_id="gpt-5.4-2026-03-05",
           provider=Provider.OPENAI, org="OpenAI", options={"temperature": 0}),
  ```

- `REFORMAT_MODEL`: standalone ModelRun for fixing unparseable responses
- Duplicate name validation at import time

Port all 30 models from `main` branch's `constants.MODELS_TO_RUN`. Add reasoning-level variants for models that support them (e.g., `claude-opus-4-6-medium`, `claude-opus-4-6-high`).

### Step 4: Run tests to verify they pass

Run: `python -m pytest src/tests/test_llm_registry.py -v`
Expected: PASS

### Step 5: Run lint

Run: `make lint`

### Step 6: Commit

```bash
git add src/helpers/llm.py src/tests/test_llm_registry.py
git commit -m "feat: replace LiteLLM registry with utils_llm-backed ModelRun"
```

---

## Task 4: Forecast validation module

Create a standalone module that validates forecast files and produces the emoji summary.

**Files:**
- Create: `src/helpers/forecast_validation.py`
- Test: `src/tests/test_forecast_validation.py` (create)

### Step 1: Write failing tests

Tests should cover:
- Valid file with all fields → `valid_json=True`, `valid_probabilities=True`
- Probability outside (0, 1) → `valid_probabilities=False`
- Null forecasts counted correctly → percentage reflects missing forecasts
- Summary string contains `📊`, `✅` when ≥95%, `❌` when <95%
- Missing required JSON keys → `valid_json=False`

### Step 2: Run tests to verify they fail

Run: `python -m pytest src/tests/test_forecast_validation.py -v`
Expected: FAIL

### Step 3: Implement `src/helpers/forecast_validation.py`

Contains:
- `THRESHOLD_PCT = 95.0`
- `ValidationResult` dataclass with: `valid_json`, `valid_probabilities`, `market_forecasted`, `market_total`, `dataset_forecasted`, `dataset_total`
- Properties: `market_pct`, `dataset_pct`
- `format_summary(model_name, prompt_type)` → emoji-formatted string:
  ```
  📊 Forecast Summary: grok-5 (zero_shot)
  ├── ✅ Valid JSON structure
  ├── ✅ All probabilities in (0, 1)
  ├── ✅ Market questions: 376/378 forecasted (99.5%)
  └── ✅ Dataset questions: 618/620 forecasted (99.7%)
  ```
  Use `❌` when below 95%.
- `validate_forecast_file(filepath, n_market, n_dataset)` → `ValidationResult`
  - Checks required top-level JSON keys
  - Checks each forecast has required keys
  - Validates probabilities are floats in (0, 1) exclusive
  - Counts non-null forecasts, split by market vs dataset source

### Step 4: Run tests to verify they pass

Run: `python -m pytest src/tests/test_forecast_validation.py -v`
Expected: PASS

### Step 5: Commit

```bash
git add src/helpers/forecast_validation.py src/tests/test_forecast_validation.py
git commit -m "feat: add forecast validation with emoji summary"
```

---

## Task 5: Rewrite worker to use ModelRun registry

Replace the legacy worker with one that:
1. Looks up its `ModelRun` from the registry
2. Configures API keys via `utils_llm`
3. Processes questions with `ThreadPoolExecutor`, sized by rate limit budget
4. Runs validation after every run
5. In TEST mode with `MODEL_TO_TEST`, runs small sample and exits on validation failure

**Files:**
- Modify: `src/base_eval/llm_baselines/worker/main.py`
- Modify: `src/helpers/model_eval.py` — update to use `ModelRun.get_response()`
- Test: `src/tests/test_worker.py` (create)

### Step 1: Write failing tests

Tests should cover:
- `ModelRun.get_response()` merges options correctly (mock the underlying Model)
- Thread pool size calculation: `max(1, rate_limit // group_size)`
- Worker correctly skips dataset questions when `market_use_freeze_value=True`

### Step 2: Run tests to verify they fail

Run: `python -m pytest src/tests/test_worker.py -v`
Expected: FAIL

### Step 3: Update `model_eval.py`

Key changes:
- Remove all direct SDK client instantiations at module top level (`anthropic.Anthropic(...)`, `openai.OpenAI(...)`, etc.)
- Remove all `get_response_from_*_model()` functions (6 functions)
- Remove `get_response_from_model()` dispatch function
- Remove `get_response_with_retry()` (lives in utils_llm now)
- Remove `infer_model_source()`
- Update `worker()` to accept a `model_run: ModelRun` parameter instead of `model_name: str`, call `model_run.get_response(prompt, temperature=0, max_tokens=100)`
- Update `executor()` to accept `max_workers: int` parameter instead of using `env.NUM_CPUS`
- Update `reformat_answers()` to import and use `REFORMAT_MODEL.get_response()` from `helpers.llm`
- Keep all prompt construction, probability extraction, forecast generation, and file I/O code unchanged
- Update `generate_final_forecast_files()` to accept `ModelRun` objects instead of the old model dict, deriving `org` from `model_run.org` and `full_name` from `model_run.model_id`
- Remove `get_model_org()` and `capitalize_substrings()` if they can be inlined or simplified

### Step 4: Rewrite `worker/main.py`

Key changes:
- Import `MODEL_RUNS`, `RATE_LIMITS`, `configure_keys` from `helpers.llm`
- Import `validate_forecast_file` from `helpers.forecast_validation`
- On startup: call `configure_keys()`
- `parse_env_vars()`:
  - Read `CLOUD_RUN_TASK_INDEX` and look up `MODEL_RUNS[task_index]`
  - OR read `MODEL_TO_TEST` and find matching `ModelRun` by name
  - Read `RATE_LIMIT_GROUP_SIZE` env var (default 1)
  - Compute `max_workers = max(1, RATE_LIMITS[model_run.rate_limit_group] // group_size)`
- Pass `max_workers` to `model_eval.executor()`
- After `generate_final_forecast_files()`, call `validate_forecast_file()` on each output file
- Log the `format_summary()` output
- In TEST mode: `sys.exit(1)` if validation fails

### Step 5: Run tests to verify they pass

Run: `python -m pytest src/tests/test_worker.py src/tests/test_forecast_validation.py src/tests/test_llm_registry.py -v`
Expected: PASS

### Step 6: Run lint

Run: `make lint`

### Step 7: Commit

```bash
git add src/base_eval/llm_baselines/worker/main.py src/helpers/model_eval.py src/tests/test_worker.py
git commit -m "feat: rewrite worker to use ModelRun registry with ThreadPoolExecutor"
```

---

## Task 6: Update manager to pass rate limit group size

**Files:**
- Modify: `src/base_eval/llm_baselines/manager/main.py`
- Test: `src/tests/test_manager.py` (create)

### Step 1: Write failing tests

Tests should cover:
- Group size counting: given a list of ModelRuns, count how many share each `rate_limit_group`
- Manager sets `task_count = len(MODEL_RUNS)`

### Step 2: Run tests to verify they fail

### Step 3: Update manager

- Import `MODEL_RUNS` from `helpers.llm`
- Count models per `rate_limit_group` using `collections.Counter`
- For each `ModelRun`, include `RATE_LIMIT_GROUP_SIZE` in the env vars passed to `cloud_run.call_worker()`
- Set `task_count = len(MODEL_RUNS)`
- Remove `len(llm.MODEL_RUNS) * len(llm.PROMPT_TYPES)` calculation (prompt types are no longer a dimension of the task grid — each ModelRun is one task)

### Step 4: Run tests, lint, commit

```bash
git add src/base_eval/llm_baselines/manager/main.py src/tests/test_manager.py
git commit -m "feat: manager passes rate limit group size to workers"
```

---

## Task 7: Cleanup

**Files:**
- Modify: `src/base_eval/llm_baselines/worker/requirements.txt` — remove `litellm`, `mistralai`
- Modify: `src/helpers/constants.py` — remove `OAI_SOURCE`, `ANTHROPIC_SOURCE`, etc. if no longer referenced outside of model_eval.py (check first)
- Remove dead code

### Step 1: Verify no remaining references to removed code

Grep across `src/` for: `MODELS_TO_RUN`, `MODEL_TOKEN_LIMITS`, `MODEL_NAME_TO_SOURCE`, `OAI_SOURCE`, `ANTHROPIC_SOURCE`, `TOGETHER_AI_SOURCE`, `GOOGLE_SOURCE`, `MISTRAL_SOURCE`, `XAI_SOURCE`, `litellm`, `mistralai`, `get_response_from_model`, `infer_model_source`.

Only remove constants that have zero remaining references.

### Step 2: Run full test suite

```bash
make lint
python -m pytest src/tests/ -v
```

### Step 3: Commit

```bash
git add -u
git commit -m "chore: remove legacy LLM code and unused dependencies"
```

---

## Task 8: End-to-end smoke test (manual, not remote)

This task is for Jesse to run locally after the remote session completes.

### Step 1: Move staged utils files

```bash
cp -r src/helpers/utils_llm/* utils/utils/llm/
cd utils && git add -u && git diff --cached  # Review changes
```

### Step 2: Update forecastbench imports

Change all `from helpers.utils_llm.` imports to `from utils.llm.` in:
- `src/helpers/llm.py`

### Step 3: Run local TEST mode

```bash
cd src/base_eval/llm_baselines/worker && \
eval $(cat ../../../../variables.mk | xargs) \
TEST_OR_PROD=TEST \
MODEL_TO_TEST=gpt-4.1-2025-04-14 \
CLOUD_RUN_TASK_INDEX=0 \
python main.py
```

Verify:
- Worker runs without error
- Emoji validation summary is printed
- Forecast file is created locally

# ForecastBench LLM forecaster corrective refactor

**Date:** 2026-05-06
**Status:** Approved design; implementation plan pending

## Problem

The previous LLM forecaster refactor attempt preserved too much of the old
ForecastBench LLM model registry and changed behavior that should have stayed
stable. The largest problems were:

- model runs were derived from `helpers.constants.MODELS_TO_RUN` through a tuple
  translation layer instead of being explicit `ModelRun(...)` declarations;
- runtime options such as Anthropic `max_tokens` and `temperature` were inferred
  outside the model-run declarations;
- ForecastBench LLM prompts and parsing behavior were rewritten instead of
  preserved;
- the new manager and question loading path bypassed the refactored
  orchestration IO boundary;
- the runner gathered both final variants before writing either one;
- old abstractions and helper constants remained in the codebase.

This corrective refactor replaces the flawed local work with a clean history and
a ForecastBench-native implementation that mirrors the time-series-benchmark
LLM model-run API where applicable.

## Recovery Strategy

Use a clean rewrite from `origin/main`, not corrective commits on top of the
flawed merge.

Implementation must:

1. create a safety ref for the flawed local state before changing history;
2. recreate the LLM refactor from `origin/main`;
3. discard the flawed merge history and the standalone `939a3f3` cleanup commit;
4. fold complete `utils` cleanup into the corrected dependency migration task;
5. rebuild the refactor as clean TDD commits, task by task;
6. leave unrelated untracked files untouched;
7. update `AGENTS.md` so it no longer describes `utils/` as a submodule and so
   it records the model-run and prompt-preservation rules learned here.

The final branch must not contain the flawed abstractions:
`_CANONICAL_MODEL_RUN_KEYS`, `_options_for_model_run`, `conflicting_options`,
`_CloudRun`, strict plain-decimal prompt/parsing rewrites, or live dependencies
on `helpers.constants.MODELS_TO_RUN`.

## Goals

- Add a ForecastBench-native LLM forecaster outside
  `src/base_eval/llm_baselines`.
- Use `utils` as an installed `fri-utils` package instead of a repo submodule.
- Preserve the existing ForecastBench active LLM model set while rewriting it as
  explicit `ModelRun(...)` declarations.
- Mirror the time-series-benchmark `ModelRun` API unless an explicit
  ForecastBench exception is documented.
- Keep the ForecastBench forecast file envelope and row schema exactly the same
  as current LLM forecast files; only model names change.
- Preserve current LLM forecasting prompt text and parsing behavior for this
  pass.
- Use `src/orchestration/_io.py` as the single question-set IO boundary.
- Generate the active LLM forecast variants:
  `zero-shot` and `zero-shot-with-freeze-values`.
- Write the zero-shot final file before attempting the freeze-values variant.
- Normalize historical LLM display names during leaderboard ingest without
  renaming historical forecast files.
- Add a time-series-benchmark-style smoke test for the ForecastBench LLM
  forecaster.
- Delete `src/base_eval/llm_baselines` only after the new implementation,
  deploy paths, tests, and legacy compatibility are in place.

## Non-goals

- Do not rename historical forecast files in buckets or repositories.
- Do not change the forecast JSON schema.
- Do not change the LLM prompt response format in this pass.
- Do not introduce a new strict plain-decimal parser in this pass.
- Do not refactor metadata tagging or validation prompt ownership now.
- Do not refactor the naive forecaster or resolve code except where necessary
  to share the existing orchestration IO boundary.
- Do not make breaking changes to `/workspace/utils` unless explicitly
  discussed.

## Model Runs

`src/llm_forecaster/model_runs.py` owns the ForecastBench LLM model-run
registry. It should follow time-series-benchmark's pattern:

- `ModelRun` has `name`, `model_id`, `lab`, `provider`, and `options`;
- `id` returns `name`;
- `model_organization` comes from the lab display metadata;
- `__repr__` matches the time-series-benchmark implementation;
- `get_response(prompt, **kwargs)` delegates to
  `utils.llm.model_registry.get_response(...)` with
  `options={**self.options, **kwargs}`;
- provider-key configuration and validation happen once per worker or smoke run;
- provider concurrency limits are keyed by provider, not model-making lab.

The model registry is explicit. ForecastBench keeps its existing active model
set, but each run is declared directly:

```python
ModelRun(
    name="claude-opus-4-7-4096",
    model_id="claude-opus-4-7",
    lab=LABS["Anthropic"],
    provider=PROVIDERS["Anthropic"],
    options={"max_tokens": 4096},
)
```

Rules:

- all runtime options live in the `ModelRun(...)` declaration;
- no inferred/defaulted `max_tokens`, `temperature`, reasoning options, or tool
  options outside declarations;
- no runtime conflict checking for options;
- token suffixes in canonical names must match the declared
  `options["max_tokens"]` or an explicitly documented calling-structure value;
- hosted models use the model maker as `lab` and API host as `provider`;
- `helpers.constants.MODELS_TO_RUN`, `MODELS_TO_RUN_BY_SOURCE`, and old LLM
  org/source constants disappear once the new registry replaces all live uses.

Labs and providers come from `utils.llm.lab_registry` and
`utils.llm.provider_registry`. If ForecastBench needs a lab missing from utils,
such as `MiniMax`, update `/workspace/utils` in a separate utils commit and then
update ForecastBench's `requirements.runtime.txt` pin to that commit. Utils
changes must remain backward-compatible for ForecastBench and
time-series-benchmark unless a breaking change is discussed explicitly.

## Prompts and Parsing

This pass preserves ForecastBench LLM prompt behavior. It changes ownership, not
prompt semantics.

`src/llm_forecaster/prompts.py` contains only prompts used by the LLM forecaster:

- zero-shot market prompt;
- zero-shot market-with-freeze-values prompt;
- zero-shot dataset prompt;
- response reformat prompts used by forecast parsing.

Prompt strings should be copied byte-for-byte from the legacy ForecastBench
LLM implementation. The old "non_market" terminology must not be used in new
LLM forecaster symbols. The moved dataset prompt should use dataset terminology,
for example `ZERO_SHOT_DATASET_PROMPT`, even if the source constant was named
`ZERO_SHOT_NON_MARKET_PROMPT`.

`src/helpers/llm_prompts.py` may temporarily remain for metadata tagging and
validation prompts. Moving those prompts is out of scope until those workflows
are refactored. After `src/base_eval/llm_baselines` is deleted,
`helpers.llm_prompts` should no longer contain LLM forecasting prompts.

Parsing should preserve the current ForecastBench behavior from
`helpers.model_eval`:

- keep zero-shot market response parsing as direct asterisk-delimited
  extraction only; do not reformat unparseable zero-shot market responses;
- keep current multi-horizon/list parsing behavior for dataset questions;
- keep current reformat retry behavior for dataset responses;
- do not introduce strict plain-decimal-line parsing in this pass.

Future prompt-format improvements should be a separate design and implementation
pass.

## Question Sets and Orchestration IO

`src/orchestration/_io.py` is the canonical question-set storage boundary.
The LLM manager and worker must not clone the dataset repo directly or create a
second question-loading path.

The refactor should add or reuse `_io` helpers so:

- the manager reads latest LLM question-set metadata from `latest-llm.json`;
- the worker loads `f"{forecast_due_date}-llm.json"` through the same published
  `forecastbench-datasets` raw GitHub URL path that `_io` uses;
- local tests can read local question-set fixtures through the same API shape;
- future question-set storage changes happen in `_io` instead of in LLM code.

`QuestionSetContext` remains local to `src/llm_forecaster` for this pass because
it is the runner's internal shape:

- `forecast_due_date`;
- `question_set_filename`;
- `questions`.

The context is built at the LLM boundary from `_io` data. Resolve can keep using
the DataFrame returned by `_io.download_and_read_question_set_file`; naive
forecaster question-set cleanup can happen later.

## Runner and Output

The runner executes one `ModelRun` against one question set.

Execution order:

1. load one LLM question set;
2. split questions using `sources.DATASET_SOURCE_NAMES` and
   `sources.MARKET_SOURCE_NAMES`;
3. forecast dataset questions once;
4. forecast market questions with the zero-shot prompt;
5. immediately write/upload the zero-shot final file containing dataset rows
   plus zero-shot market rows;
6. forecast market questions with the zero-shot-with-freeze-values prompt;
7. immediately write/upload the freeze-values final file containing the same
   dataset rows plus freeze-value market rows.

This ensures a freeze-values failure does not lose the zero-shot file.

Output rules:

- JSON envelope stays exactly the same as current ForecastBench LLM forecast
  files;
- forecast rows stay exactly the same as current ForecastBench LLM forecast
  rows;
- only model names change;
- historical forecast files are not renamed;
- final-file idempotency is preserved: do not overwrite an existing target file;
- test prefix behavior is preserved;
- freeze-values output uses a canonical suffix so it remains distinct from the
  base model run.

One failed question should be logged and skipped, matching current downstream
imputation behavior, unless an explicit test/smoke flag requests fail-fast
behavior.

## Test Mode and Smoke Test

Production is opt-in.

- If `TEST_OR_PROD` is exactly `PROD`, the worker runs the full question set.
- If `TEST_OR_PROD` is missing or any value other than `PROD`, the worker uses
  test behavior.
- Test behavior forecasts only 2 dataset questions and 2 market questions by
  default.
- The same 2 market questions are used for both market variants.
- Dataset forecasts are reused across both final files.
- The default test limits are named constants, not scattered literals.

Add a ForecastBench LLM smoke test modeled on time-series-benchmark's smoke
test, adapted for ForecastBench's binary forecast variants.

Smoke test behavior:

- load questions through the same `_io` question-set path as the worker;
- allow selecting exact `ModelRun.name` values;
- default to all configured model runs unless specific names are requested;
- run enough of the real worker path to exercise prompt rendering, provider
  routing, parsing/reformat fallback, and worker-shaped forecast rows;
- write local test forecast files under a smoke output directory using `TEST.`
  prefix semantics;
- return a nonzero exit code if any selected model/question smoke task fails or
  produces no valid forecast rows;
- log concise pass/fail rows and output file paths.

## Leaderboard Legacy Names

Historical LLM names are normalized only during leaderboard ingest. Historical
forecast file paths and object names are not renamed.

Rules:

- displayed model names use the new canonical names;
- legacy mapping is strict and errors when a legacy LLM name/variant needs a
  mapping but no mapping exists;
- mapping covers historical LLM variants:
  `zero_shot`, `zero_shot_with_freeze_values`, `scratchpad`, and
  `scratchpad_with_freeze_values`;
- `model_release_dates.csv` uses displayed canonical names;
- leaderboard `model_pk` remains distinct for base and freeze-values variants.

## Utils Migration

Remove the root `utils` submodule completely:

- remove the `.gitmodules` stanza;
- remove the tracked gitlink;
- remove formatter/linter excludes for root `utils`;
- remove deploy-time `cp -r $(ROOT_DIR)utils` copies;
- remove root-only `sys.path` hacks that existed only for the submodule.

Keep the shared utils pin only in root `requirements.runtime.txt`.
Deploy Makefiles that need shared runtime dependencies should create upload
requirements by concatenating root `requirements.runtime.txt` with local
requirements. The utils SHA must not be copied into every deploy
`requirements.txt`.

The implementation must audit changes in the selected utils commit against the
previous pinned submodule commit and verify ForecastBench still works with the
installed package. If `/workspace/utils` changes, commit those changes in utils
first and update ForecastBench's pin to the new SHA.

## Cloud Run and Nightly Workflow

The new Cloud Run entrypoints should be thin:

- manager parses run mode, gets latest question-set metadata through `_io`, and
  launches worker tasks;
- worker parses environment, selects the model run by task index, loads the
  question set through `_io`, applies test-mode limits when needed, configures
  provider keys, and calls the LLM runner.

Do not introduce wrapper classes that only forward to helper functions. Direct
function calls are sufficient and easier to test with monkeypatching.

The nightly workflow should call the new manager job. Root Makefile targets may
keep user-facing names if useful, but should build/deploy the new manager and
worker directories.

## Deletion Gate

Do not delete `src/base_eval/llm_baselines` until all are true:

- new model registry exists and passes tests;
- prompt/parsing behavior is preserved and tested;
- runner writes both variants in the required order;
- manager and worker are wired to Cloud Run and nightly workflow;
- leaderboard legacy mapping is strict and tested;
- deploy targets use installed `fri-utils`;
- smoke test exists;
- full relevant test suite and lint pass.

After the gate passes, delete `src/base_eval/llm_baselines` and remove any live
imports or deploy references to it.

## Testing Strategy

Use TDD for each implementation task.

Required tests include:

- runtime requirements and deploy Makefile staging tests;
- model-run routing tests adapted from time-series-benchmark;
- explicit registry tests proving ForecastBench active models are declared and
  no options are inferred elsewhere;
- tests proving `__repr__` and `get_response` semantics match
  time-series-benchmark;
- prompt migration tests that compare moved LLM prompt strings to the legacy
  source before deletion, followed by snapshot/lock tests after deletion;
- parser behavior tests based on current ForecastBench examples;
- `_io` question-set loading tests, including latest lookup;
- runner tests proving dataset reuse and zero-shot write-before-freeze behavior;
- worker tests proving missing/non-`PROD` `TEST_OR_PROD` limits to 2 dataset and
  2 market questions;
- smoke-test unit tests for model selection, question selection, failure exit
  codes, and local test forecast output;
- leaderboard strict mapping tests, including failure on unmapped legacy names;
- cleanup tests proving wrong abstractions/imports do not remain.

Before final completion, run:

- `make lint`;
- `make test` or the documented full pytest equivalent;
- targeted tests for `llm_forecaster`, leaderboard mapping, runtime
  requirements, and orchestration IO;
- relevant `/workspace/utils` tests if utils changes.

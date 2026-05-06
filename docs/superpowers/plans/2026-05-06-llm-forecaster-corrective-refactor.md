# ForecastBench LLM Forecaster Corrective Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the ForecastBench LLM forecaster refactor on clean history with explicit time-series-benchmark-style model runs, installed `fri-utils`, preserved prompts/parsing, `_io`-based question loading, per-variant writes, strict leaderboard legacy mapping, and a smoke test.

**Architecture:** Keep ForecastBench-specific forecasting behavior in `src/llm_forecaster`, keep storage and Cloud Run IO in `src/orchestration`, and keep shared LLM provider metadata in `/workspace/utils`. Preserve existing ForecastBench forecast file schemas and prompt/parsing behavior while replacing legacy model routing and `src/base_eval/llm_baselines`.

**Tech Stack:** Python 3.14+, pytest, Google Cloud Run Jobs, Google Cloud Storage, installed `fri-utils`, ForecastBench `sources` and `orchestration._io`, pandas for existing question-set IO.

---

## File Structure

Shared utils repository:

- Modify `/workspace/utils/utils/llm/lab_registry.py`: add MiniMax lab metadata.
- Modify `/workspace/utils/tests/unit/test_llm_routing.py`: assert MiniMax registry behavior.

ForecastBench dependency and cleanup:

- Create `requirements.runtime.txt`: root-only `fri-utils` git pin.
- Modify `requirements.txt`: include `-r requirements.runtime.txt`.
- Modify `.gitmodules`: remove only the `utils` submodule stanza.
- Modify `setup.cfg`: remove root `utils` lint/doc excludes.
- Modify `pyproject.toml`: remove root `utils` black exclude.
- Modify deploy Makefiles that stage code: use `cat $(ROOT_DIR)requirements.runtime.txt requirements.txt > $(UPLOAD_DIR)/requirements.txt` or the existing upload variable for that Makefile, and remove `cp -r $(ROOT_DIR)utils`.
- Modify Python entrypoints that had root-only four-level `sys.path.append` calls for importing root `utils`: remove those path hacks once installed `fri-utils` is used.
- Modify `AGENTS.md`: document installed `fri-utils`, no root submodule, explicit `ModelRun(...)` declarations, prompt preservation, and no `from __future__ import annotations` in new ForecastBench files.
- Delete tracked gitlink `utils` with `git rm --cached utils`.
- Create `src/tests/test_runtime_requirements.py`: dependency staging regression tests.
- Create `src/tests/test_utils_cleanup.py`: submodule/path cleanup regression tests.

ForecastBench LLM forecaster package:

- Create `src/llm_forecaster/AGENTS.md`: local package guidance.
- Create `src/llm_forecaster/__init__.py`: small public surface.
- Create `src/llm_forecaster/model_runs.py`: explicit model-run registry and key validation helpers.
- Create `src/llm_forecaster/forecast_variants.py`: zero-shot and freeze-values variants.
- Create `src/llm_forecaster/output.py`: final forecast file names, envelope, local write, GCS existence/upload helpers.
- Create `src/llm_forecaster/prompts.py`: LLM forecasting prompts copied from legacy prompt text, using dataset terminology.
- Create `src/llm_forecaster/parsing.py`: current ForecastBench probability/list parsing and reformat retry behavior, routed through `ModelRun`.
- Create `src/llm_forecaster/questions.py`: local `QuestionSetContext`, question split, test-mode limit, `_io` context loader.
- Create `src/llm_forecaster/runner.py`: one-model-run execution with dataset reuse and immediate per-variant writes.
- Create `src/llm_forecaster/smoke_test.py`: local smoke command modeled on time-series-benchmark.
- Create focused tests under `src/tests/llm_forecaster/`.

Orchestration:

- Modify `src/orchestration/_io.py`: add raw question-set JSON and latest question-set metadata helpers while keeping resolve's DataFrame API.
- Create `src/orchestration/func_llm_forecaster_manager/main.py`: thin manager entrypoint.
- Create `src/orchestration/func_llm_forecaster_manager/Makefile`: deploy manager job.
- Create `src/orchestration/func_llm_forecaster_manager/requirements.txt`: manager deps.
- Create `src/orchestration/func_llm_forecaster_worker/main.py`: thin worker entrypoint.
- Create `src/orchestration/func_llm_forecaster_worker/Makefile`: deploy worker job.
- Create `src/orchestration/func_llm_forecaster_worker/requirements.txt`: worker deps.
- Create tests under `src/tests/orchestration/`.

Leaderboard:

- Create `src/leaderboard/llm_legacy_names.py`: strict legacy LLM identity normalization.
- Modify `src/leaderboard/main.py`: normalize ForecastBench LLM identity before `model_pk`.
- Modify `src/leaderboard/model_release_dates.csv`: add canonical displayed names for active refactored LLM runs.
- Create `src/tests/leaderboard/test_llm_legacy_names.py`.

Legacy deletion gate:

- Modify `src/helpers/constants.py`: remove old active LLM model-run constants after new registry is live.
- Modify or trim `src/helpers/model_eval.py`: remove legacy LLM baseline-only routing code after metadata callers are no longer dependent on it.
- Modify `src/helpers/llm_prompts.py`: remove forecasting prompts only after `src/base_eval/llm_baselines` is deleted; leave tagging/validation prompts.
- Delete `src/base_eval/llm_baselines` after the gate passes.
- Modify root `Makefile` and `src/nightly_update_workflow/worker/main.py`: route to the new LLM forecaster jobs.

---

### Task 1: Add MiniMax Lab to Shared Utils

**Files:**
- Modify: `/workspace/utils/utils/llm/lab_registry.py`
- Modify: `/workspace/utils/tests/unit/test_llm_routing.py`

- [ ] **Step 1: Write the failing utils registry test**

Add these assertions to `test_labs_have_leaderboard_names()` in `/workspace/utils/tests/unit/test_llm_routing.py`:

```python
    assert LABS["MiniMax"].name == "MiniMax"
    assert LABS["MiniMax"].leaderboard_name == "MiniMax"
```

- [ ] **Step 2: Run the focused utils test and verify it fails**

Run:

```bash
cd /workspace/utils
python -m pytest tests/unit/test_llm_routing.py::test_labs_have_leaderboard_names -q
```

Expected: FAIL with `KeyError: 'MiniMax'`.

- [ ] **Step 3: Add the MiniMax lab**

Modify `/workspace/utils/utils/llm/lab_registry.py` so the registry contains:

```python
    "MiniMax": Lab(name="MiniMax"),
```

Place it near the other model-making labs.

- [ ] **Step 4: Run utils tests**

Run:

```bash
cd /workspace/utils
python -m pytest tests/unit/test_llm_routing.py tests/unit/test_model_registry_config.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit the utils change**

Run:

```bash
cd /workspace/utils
git add utils/llm/lab_registry.py tests/unit/test_llm_routing.py
git commit -m "feat(llm): add MiniMax lab metadata"
git rev-parse HEAD
```

Record the printed SHA. ForecastBench Task 2 updates `requirements.runtime.txt` to that SHA.

---

### Task 2: Install `fri-utils` and Remove the Root `utils` Submodule

**Files:**
- Create: `requirements.runtime.txt`
- Create: `src/tests/test_runtime_requirements.py`
- Create: `src/tests/test_utils_cleanup.py`
- Modify: `requirements.txt`
- Modify: `.gitmodules`
- Modify: `setup.cfg`
- Modify: `pyproject.toml`
- Modify: `AGENTS.md`
- Modify: deploy Makefiles under `src/**/Makefile`
- Modify: Python files with root-only four-level `sys.path.append` calls for importing root `utils`
- Delete tracked gitlink: `utils`

- [ ] **Step 1: Write failing runtime requirements tests**

Create `src/tests/test_runtime_requirements.py`:

```python
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNTIME_REQUIREMENT = "git+https://github.com/forecastingresearch/utils@"


def iter_deploy_requirements():
    for requirements_path in sorted(ROOT.glob("src/**/requirements.txt")):
        if "upload" in requirements_path.parts:
            continue
        makefile_path = requirements_path.with_name("Makefile")
        if makefile_path.exists():
            yield requirements_path, makefile_path


def test_shared_utils_pin_only_lives_in_root_runtime_requirements():
    root_runtime_requirements = (ROOT / "requirements.runtime.txt").read_text()

    assert RUNTIME_REQUIREMENT in root_runtime_requirements
    for requirements_path, _makefile_path in iter_deploy_requirements():
        assert RUNTIME_REQUIREMENT not in requirements_path.read_text(), requirements_path


def test_deploy_makefiles_stage_shared_runtime_requirements():
    for _requirements_path, makefile_path in iter_deploy_requirements():
        makefile = makefile_path.read_text()

        assert "$(ROOT_DIR)requirements.runtime.txt" in makefile, makefile_path
        assert "cat $(ROOT_DIR)requirements.runtime.txt requirements.txt > $(" in makefile
```

- [ ] **Step 2: Write failing cleanup tests**

Create `src/tests/test_utils_cleanup.py`:

```python
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_root_utils_submodule_metadata_removed():
    gitmodules = (ROOT / ".gitmodules").read_text()
    assert '[submodule "utils"]' not in gitmodules
    assert "\tpath = utils" not in gitmodules

    tracked = subprocess.run(
        ["git", "ls-files", "-s", "utils"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    assert tracked.stdout == ""


def test_formatters_no_longer_exclude_root_utils():
    setup_cfg = (ROOT / "setup.cfg").read_text()
    pyproject = (ROOT / "pyproject.toml").read_text()

    assert "utils," not in setup_cfg
    assert "| ^/utils/" not in pyproject
    assert "match-dir = ^(?!(\\.venv|utils|" not in setup_cfg


def test_deploy_makefiles_do_not_copy_root_utils():
    offenders = [
        path
        for path in ROOT.glob("src/**/Makefile")
        if "cp -r $(ROOT_DIR)utils" in path.read_text()
    ]
    assert offenders == []


def test_no_root_only_sys_path_hacks_for_utils_imports():
    offenders = []
    for path in ROOT.glob("src/**/*.py"):
        text = path.read_text()
        if '../../../.."' in text or "../../../..')" in text:
            offenders.append(path)
    assert offenders == []
```

- [ ] **Step 3: Run the focused tests and verify they fail**

Run:

```bash
cd /workspace/forecastbench
python -m pytest src/tests/test_runtime_requirements.py src/tests/test_utils_cleanup.py -q
```

Expected: FAIL because `requirements.runtime.txt` is missing, the `utils` gitlink exists, and deploy files still copy root `utils`.

- [ ] **Step 4: Add the root runtime requirements pin**

Create `requirements.runtime.txt` with the utils SHA from Task 1:

```bash
cd /workspace/forecastbench
UTILS_SHA=$(git -C /workspace/utils rev-parse HEAD)
printf "git+https://github.com/forecastingresearch/utils@%s#egg=fri-utils\n" "$UTILS_SHA" > requirements.runtime.txt
```

Modify the top of `requirements.txt` to start with:

```text
-r requirements.runtime.txt
isort==6.0.0
```

- [ ] **Step 5: Remove the submodule metadata and gitlink**

Remove this stanza from `.gitmodules`:

```ini
[submodule "utils"]
	path = utils
	url = https://github.com/forecastingresearch/utils.git
```

Run:

```bash
cd /workspace/forecastbench
git rm --cached utils
```

Do not delete unrelated untracked files manually.

- [ ] **Step 6: Update formatter/linter config**

In `setup.cfg`, remove only root `utils` from `isort`, `flake8`, and `pydocstyle` exclusions. In `pyproject.toml`, remove only `| ^/utils/` from the black `force-exclude`.

- [ ] **Step 7: Update deploy Makefiles**

For each Makefile that stages a deploy folder and has a paired `requirements.txt`, replace:

```make
	cp requirements.txt $(UPLOAD_DIR)/requirements.txt
```

or equivalent with:

```make
	cat $(ROOT_DIR)requirements.runtime.txt requirements.txt > $(UPLOAD_DIR)/requirements.txt
```

For leaderboard's two upload directories use the existing variable names:

```make
	cat $(ROOT_DIR)requirements.runtime.txt requirements.txt > $(LEADERBOARD_UPLOAD_DIR)/requirements.txt
	cat $(ROOT_DIR)requirements.runtime.txt requirements.txt > $(DATASET_UPLOAD_DIR)/requirements.txt
```

Remove every line shaped like:

```make
	cp -r $(ROOT_DIR)utils $(UPLOAD_DIR)/
	cp -r $(ROOT_DIR)utils $(LEADERBOARD_UPLOAD_DIR)/
	cp -r $(ROOT_DIR)utils $(DATASET_UPLOAD_DIR)/
```

- [ ] **Step 8: Remove root-only `sys.path` hacks**

Remove `sys.path.append(os.path.join(os.path.dirname(__file__), "../../../.."))` lines that existed only to import root `utils`. Keep `sys.path.append` lines that expose copied `src` modules.

The check must return no matches outside `src/base_eval/llm_baselines`, which is deleted in Task 13:

```bash
cd /workspace/forecastbench
rg -n "sys\\.path\\.append\\(.*\\.\\.\\/\\.\\.\\/\\.\\.\\/\\.\\." src
```

- [ ] **Step 9: Update root AGENTS guidance**

Modify `AGENTS.md`:

- remove the directory-tree line describing `utils/` as a git submodule;
- add that shared utilities are installed from root `requirements.runtime.txt`;
- add that new ForecastBench Python files should not add `from __future__ import annotations`;
- add that LLM forecaster `ModelRun(...)` declarations mirror time-series-benchmark and all runtime options live in declarations;
- add that LLM forecasting prompt text and parsing behavior are preserved in this pass.

- [ ] **Step 10: Verify dependency migration**

Run:

```bash
cd /workspace/forecastbench
python -m pytest src/tests/test_runtime_requirements.py src/tests/test_utils_cleanup.py -q
python - <<'PY'
from utils import gcp
from utils.llm.lab_registry import LABS
from utils.llm.model_registry import get_response

print(gcp.__name__)
print(LABS["MiniMax"].leaderboard_name)
print(get_response.__name__)
PY
```

Expected: tests PASS; script prints `utils.gcp`, `MiniMax`, and `get_response`.

- [ ] **Step 11: Commit**

Run:

```bash
cd /workspace/forecastbench
git add requirements.runtime.txt requirements.txt .gitmodules setup.cfg pyproject.toml AGENTS.md src Makefile
git add -u utils
git commit -m "build: install shared utils as runtime package"
```

---

### Task 3: Add Explicit ForecastBench LLM Model Runs

**Files:**
- Create: `src/llm_forecaster/AGENTS.md`
- Create: `src/llm_forecaster/__init__.py`
- Create: `src/llm_forecaster/model_runs.py`
- Create: `src/tests/llm_forecaster/test_model_runs.py`

- [ ] **Step 1: Write failing model-run tests**

Create `src/tests/llm_forecaster/test_model_runs.py`:

```python
import inspect
from unittest.mock import patch

import pytest
from utils.llm.lab_registry import LABS
from utils.llm.provider_registry import PROVIDERS

from llm_forecaster import model_runs


def test_model_run_calls_utils_with_provider_model_id_and_options():
    run = model_runs.ModelRun(
        name="third-party-model",
        model_id="maker/native-model",
        lab=LABS["DeepSeek"],
        provider=PROVIDERS["Together"],
        options={"temperature": 0, "max_tokens": 128},
    )

    with patch("utils.llm.model_registry.get_response", return_value="0.61") as mock_call:
        assert run.get_response("prompt") == "0.61"

    mock_call.assert_called_once_with(
        provider=PROVIDERS["Together"],
        model_id="maker/native-model",
        prompt="prompt",
        options={"temperature": 0, "max_tokens": 128},
    )


def test_model_run_repr_matches_time_series_benchmark_shape():
    plain = model_runs.ModelRun(
        name="plain-model",
        model_id="plain-model",
        lab=LABS["OpenAI"],
        provider=PROVIDERS["OpenAI"],
    )
    configured = model_runs.ModelRun(
        name="configured-model",
        model_id="configured-model-id",
        lab=LABS["OpenAI"],
        provider=PROVIDERS["OpenAI"],
        options={"temperature": 0},
    )

    assert repr(plain) == "<ModelRun plain-model>"
    assert repr(configured) == (
        "<ModelRun configured-model (configured-model-id) {'temperature': 0}>"
    )


def test_model_run_names_are_unique_and_file_safe():
    names = [run.name for run in model_runs.MODEL_RUNS]

    assert names
    assert len(names) == len(set(names))
    assert all(name == name.lower() for name in names)
    assert all(" " not in name and "/" not in name and "_" not in name for name in names)


def test_forecastbench_active_model_set_is_explicit():
    expected_names = {
        "gpt-5.4-2026-03-05",
        "gpt-5.4-mini-2026-03-17",
        "gpt-5.4-nano-2026-03-17",
        "gpt-5.2-2025-12-11",
        "gpt-5.1-2025-11-13",
        "gpt-5-mini-2025-08-07",
        "gpt-5-nano-2025-08-07",
        "gpt-4.1-2025-04-14",
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
        "claude-opus-4-6-1024",
        "claude-opus-4-7-1024",
        "grok-4-1-fast-reasoning",
        "grok-4-1-fast-non-reasoning",
        "grok-4.20-0309-reasoning",
        "grok-4.20-0309-non-reasoning",
        "gemini-3.1-pro-preview",
        "gemini-3.1-flash-lite-preview",
        "gemini-2.5-pro",
        "gemini-3-flash-preview",
    }

    assert {run.name for run in model_runs.MODEL_RUNS} == expected_names


def test_labs_and_providers_are_shared_registry_objects():
    runs = {run.name: run for run in model_runs.MODEL_RUNS}

    assert runs["minimax-m2.5"].lab == LABS["MiniMax"]
    assert runs["minimax-m2.5"].provider == PROVIDERS["Together"]
    assert runs["kimi-k2.5"].lab == LABS["Moonshot"]
    assert runs["gemma-4-31b"].lab == LABS["Google DeepMind"]
    assert runs["gemma-4-31b"].provider == PROVIDERS["Together"]


def test_options_are_declared_on_model_runs_not_inferred_by_helpers():
    source = inspect.getsource(model_runs)

    assert "_CANONICAL_MODEL_RUN_KEYS" not in source
    assert "_options_for_model_run" not in source
    assert "conflicting_options" not in source
    assert "MODELS_TO_RUN" not in source
    assert "from helpers import constants" not in source

    anthropic_runs = [run for run in model_runs.MODEL_RUNS if run.provider == PROVIDERS["Anthropic"]]
    assert anthropic_runs
    assert all(run.options.get("max_tokens") == 1024 for run in anthropic_runs)
    assert "1024" in anthropic_runs[0].name


def test_model_run_lookup_raises_for_missing_name():
    with pytest.raises(KeyError, match="unknown-model"):
        model_runs.get_model_run("unknown-model")


def test_configure_and_validate_provider_keys_includes_reformat_model():
    selected = [model_runs.MODEL_RUNS[0]]

    with (
        patch("utils.llm.model_registry.configure_api_keys") as configure,
        patch("utils.llm.model_registry.validate_provider_keys") as validate,
    ):
        model_runs.configure_and_validate_provider_keys(selected)

    configure.assert_called_once_with(from_gcp=True)
    providers = validate.call_args.args[0]
    assert selected[0].provider in providers
    assert model_runs.REFORMAT_MODEL.provider in providers
```

The Anthropic `1024` value reflects the current legacy `get_response_from_anthropic_model()` call argument. If product requirements choose a different output budget before implementation, change the explicit declarations and expected names together.

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
cd /workspace/forecastbench
python -m pytest src/tests/llm_forecaster/test_model_runs.py -q
```

Expected: FAIL because `llm_forecaster.model_runs` does not exist.

- [ ] **Step 3: Add package guidance**

Create `src/llm_forecaster/AGENTS.md`:

```markdown
# LLM Forecaster Package

This package owns ForecastBench LLM baseline generation.

- Use `utils.llm` for provider calls; do not instantiate provider SDK clients directly here.
- Declare model runs with explicit `ModelRun(...)` objects following time-series-benchmark.
- Keep all runtime model options in the `ModelRun.options` declaration.
- Keep model-run names lower-case, file-safe, and explicit about non-default runtime options.
- Use `dataset`, not `non_market`, in new code.
- Preserve existing ForecastBench LLM prompt text and parsing behavior in this pass.
- Keep Cloud Run entrypoints thin; put behavior in `src/llm_forecaster`.
- Preserve the exact current ForecastBench LLM forecast-file schema; only model naming changes.
- Do not add `from __future__ import annotations` to new ForecastBench files.
```

- [ ] **Step 4: Implement `model_runs.py`**

Create `src/llm_forecaster/model_runs.py` with:

```python
"""Explicit ForecastBench LLM model-run declarations."""

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

from utils.llm.lab_registry import LABS, Lab
from utils.llm.provider_registry import PROVIDERS, Provider

logger = logging.getLogger(__name__)


@dataclass
class ModelRun:
    """Configuration for running an LLM with specific options."""

    name: str
    model_id: str
    lab: Lab
    provider: Provider
    options: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        """Return the name as the identifier."""
        return self.name

    @property
    def model_organization(self) -> str:
        """Return the model-making lab display name."""
        return self.lab.leaderboard_name

    def __repr__(self) -> str:
        """Return a string representation of the ModelRun."""
        return (
            f"<ModelRun {self.name} ({self.model_id}) {self.options}>"
            if self.options
            else f"<ModelRun {self.name}>"
        )

    def get_response(self, prompt: str, **kwargs: Any) -> str:
        """Request a response from the configured provider."""
        from utils.llm.model_registry import get_response as llm_get_response

        merged_options = {**self.options, **kwargs}
        logger.info(
            "LLM call: provider=%s, model=%s, options=%s",
            self.provider.name,
            self.model_id,
            merged_options,
        )
        return llm_get_response(
            provider=self.provider,
            model_id=self.model_id,
            prompt=prompt,
            options=merged_options,
        )
```

Add explicit grouped run lists. Use the ForecastBench active model set from old `helpers.constants.MODELS_TO_RUN`; do not import that constant. Use the exact names asserted in the test. For Anthropic runs, declare `options={"max_tokens": 1024}` because legacy ForecastBench currently used `max_tokens=1024` inside `get_response_from_anthropic_model()`.

Add:

```python
MODEL_RUNS: list[ModelRun] = (
    OPENAI_RUNS + TOGETHER_RUNS + ANTHROPIC_RUNS + XAI_RUNS + GOOGLE_RUNS
)

_model_run_names = [run.name for run in MODEL_RUNS]
if len(_model_run_names) != len(set(_model_run_names)):
    from collections import Counter

    duplicates = [name for name, count in Counter(_model_run_names).items() if count > 1]
    raise ValueError(f"Duplicate ForecastBench ModelRun names found: {duplicates}")

MODEL_RUNS_BY_NAME = {run.name: run for run in MODEL_RUNS}

REFORMAT_MODEL = ModelRun(
    name="gpt-4o-mini-reformat",
    model_id="gpt-4o-mini",
    lab=LABS["OpenAI"],
    provider=PROVIDERS["OpenAI"],
    options={"temperature": 0, "max_output_tokens": 100},
)

PROVIDER_MAX_WORKERS = {
    PROVIDERS["OpenAI"]: 16,
    PROVIDERS["Anthropic"]: 4,
    PROVIDERS["Google"]: 8,
    PROVIDERS["xAI"]: 8,
    PROVIDERS["Together"]: 8,
}


def get_model_run(name: str) -> ModelRun:
    """Return a model run by canonical name."""
    try:
        return MODEL_RUNS_BY_NAME[name]
    except KeyError as exc:
        raise KeyError(f"Unknown ForecastBench LLM model run: {name}") from exc


def providers_for_model_runs(model_runs: Sequence[ModelRun]) -> list[Provider]:
    """Return unique providers needed by selected runs and the reformat model."""
    providers_by_name = {run.provider.name: run.provider for run in [*model_runs, REFORMAT_MODEL]}
    return list(providers_by_name.values())


def configure_and_validate_provider_keys(model_runs: Sequence[ModelRun]) -> None:
    """Configure provider API keys and validate selected providers."""
    from utils.llm.model_registry import configure_api_keys, validate_provider_keys

    configure_api_keys(from_gcp=True)
    validate_provider_keys(providers_for_model_runs(model_runs))
```

- [ ] **Step 5: Add `__init__.py`**

Create `src/llm_forecaster/__init__.py`:

```python
"""ForecastBench LLM forecaster package."""
```

- [ ] **Step 6: Run focused tests**

Run:

```bash
cd /workspace/forecastbench
python -m pytest src/tests/llm_forecaster/test_model_runs.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

Run:

```bash
cd /workspace/forecastbench
git add src/llm_forecaster src/tests/llm_forecaster/test_model_runs.py
git commit -m "feat: add explicit LLM model runs"
```

---

### Task 4: Add Forecast Variants and Final Output Files

**Files:**
- Create: `src/llm_forecaster/forecast_variants.py`
- Create: `src/llm_forecaster/output.py`
- Create: `src/tests/llm_forecaster/test_forecast_variants.py`
- Create: `src/tests/llm_forecaster/test_output.py`

- [ ] **Step 1: Write failing variant tests**

Create `src/tests/llm_forecaster/test_forecast_variants.py`:

```python
from llm_forecaster import forecast_variants


def test_active_variants_are_zero_shot_and_freeze_values():
    assert [variant.key for variant in forecast_variants.FORECAST_VARIANTS] == [
        "zero-shot",
        "zero-shot-with-freeze-values",
    ]
    assert [variant.model_suffix for variant in forecast_variants.FORECAST_VARIANTS] == [
        "",
        "-with-freeze-values",
    ]


def test_get_variant_rejects_unknown_key():
    try:
        forecast_variants.get_variant("scratchpad")
    except KeyError as exc:
        assert "scratchpad" in str(exc)
    else:
        raise AssertionError("expected KeyError")
```

- [ ] **Step 2: Write failing output tests**

Create `src/tests/llm_forecaster/test_output.py`:

```python
import json
from pathlib import Path

from utils.llm.lab_registry import LABS
from utils.llm.provider_registry import PROVIDERS

from llm_forecaster.forecast_variants import ZERO_SHOT, ZERO_SHOT_WITH_FREEZE_VALUES
from llm_forecaster.model_runs import ModelRun
from llm_forecaster import output


def _run() -> ModelRun:
    return ModelRun(
        name="test-model",
        model_id="provider-model",
        lab=LABS["OpenAI"],
        provider=PROVIDERS["OpenAI"],
    )


def test_display_model_name_appends_variant_suffix():
    assert output.display_model_name(_run(), ZERO_SHOT) == "test-model"
    assert (
        output.display_model_name(_run(), ZERO_SHOT_WITH_FREEZE_VALUES)
        == "test-model-with-freeze-values"
    )


def test_destination_blob_name_preserves_current_filename_shape():
    assert (
        output.destination_blob_name("2026-05-10", _run(), ZERO_SHOT, is_test=False)
        == "2026-05-10/2026-05-10.ForecastBench.test-model.json"
    )
    assert (
        output.destination_blob_name(
            "2026-05-10", _run(), ZERO_SHOT_WITH_FREEZE_VALUES, is_test=True
        )
        == "2026-05-10/TEST.2026-05-10.ForecastBench.test-model-with-freeze-values.json"
    )


def test_write_forecast_file_uses_exact_legacy_schema(tmp_path):
    path = tmp_path / "forecast.json"
    rows = [
        {
            "id": "q1",
            "source": "fred",
            "forecast": 0.61,
            "resolution_date": "2026-06-01",
            "reasoning": None,
        }
    ]

    output.write_forecast_file(
        local_filename=path,
        forecast_due_date="2026-05-10",
        question_set_filename="2026-05-10-llm.json",
        model_run=_run(),
        variant=ZERO_SHOT,
        rows=rows,
    )

    data = json.loads(path.read_text())
    assert data == {
        "organization": "ForecastBench",
        "model": "test-model",
        "model_organization": "OpenAI",
        "question_set": "2026-05-10-llm.json",
        "forecast_due_date": "2026-05-10",
        "forecasts": rows,
    }
```

- [ ] **Step 3: Run tests and verify they fail**

Run:

```bash
cd /workspace/forecastbench
python -m pytest src/tests/llm_forecaster/test_forecast_variants.py src/tests/llm_forecaster/test_output.py -q
```

Expected: FAIL because variant/output modules do not exist.

- [ ] **Step 4: Implement variants**

Create `src/llm_forecaster/forecast_variants.py`:

```python
"""ForecastBench LLM forecast variants."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ForecastVariant:
    """Forecast variant metadata."""

    key: str
    uses_freeze_values: bool
    model_suffix: str


ZERO_SHOT = ForecastVariant(
    key="zero-shot",
    uses_freeze_values=False,
    model_suffix="",
)
ZERO_SHOT_WITH_FREEZE_VALUES = ForecastVariant(
    key="zero-shot-with-freeze-values",
    uses_freeze_values=True,
    model_suffix="-with-freeze-values",
)

FORECAST_VARIANTS = [ZERO_SHOT, ZERO_SHOT_WITH_FREEZE_VALUES]
FORECAST_VARIANTS_BY_KEY = {variant.key: variant for variant in FORECAST_VARIANTS}


def get_variant(key: str) -> ForecastVariant:
    """Return a forecast variant by key."""
    try:
        return FORECAST_VARIANTS_BY_KEY[key]
    except KeyError as exc:
        raise KeyError(f"Unknown ForecastBench LLM forecast variant: {key}") from exc
```

- [ ] **Step 5: Implement output helpers**

Create `src/llm_forecaster/output.py` with:

```python
"""ForecastBench LLM final forecast file output."""

import json
from pathlib import Path
from typing import Any

from helpers import constants
from llm_forecaster.forecast_variants import ForecastVariant
from llm_forecaster.model_runs import ModelRun


def display_model_name(model_run: ModelRun, variant: ForecastVariant) -> str:
    """Return the displayed model name for a model run and variant."""
    return f"{model_run.name}{variant.model_suffix}"


def final_filename(
    forecast_due_date: str,
    model_run: ModelRun,
    variant: ForecastVariant,
    is_test: bool,
) -> str:
    """Return the final forecast JSON filename."""
    prefix = f"{constants.TEST_FORECAST_FILE_PREFIX}." if is_test else ""
    model_name = display_model_name(model_run, variant)
    return f"{prefix}{forecast_due_date}.{constants.BENCHMARK_NAME}.{model_name}.json"


def destination_blob_name(
    forecast_due_date: str,
    model_run: ModelRun,
    variant: ForecastVariant,
    is_test: bool,
) -> str:
    """Return the GCS destination blob for a final forecast file."""
    return f"{forecast_due_date}/{final_filename(forecast_due_date, model_run, variant, is_test)}"


def write_forecast_file(
    local_filename: str | Path,
    forecast_due_date: str,
    question_set_filename: str,
    model_run: ModelRun,
    variant: ForecastVariant,
    rows: list[dict[str, Any]],
) -> None:
    """Write a final ForecastBench forecast file."""
    path = Path(local_filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    forecast_file = {
        "organization": constants.BENCHMARK_NAME,
        "model": display_model_name(model_run, variant),
        "model_organization": model_run.model_organization,
        "question_set": question_set_filename,
        "forecast_due_date": forecast_due_date,
        "forecasts": rows,
    }
    path.write_text(json.dumps(forecast_file, indent=4))
```

- [ ] **Step 6: Run focused tests**

Run:

```bash
cd /workspace/forecastbench
python -m pytest src/tests/llm_forecaster/test_forecast_variants.py src/tests/llm_forecaster/test_output.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

Run:

```bash
cd /workspace/forecastbench
git add src/llm_forecaster/forecast_variants.py src/llm_forecaster/output.py src/tests/llm_forecaster/test_forecast_variants.py src/tests/llm_forecaster/test_output.py
git commit -m "feat: add LLM forecast variants and output files"
```

---

### Task 5: Preserve LLM Forecasting Prompts and Parsing

**Files:**
- Create: `src/llm_forecaster/prompts.py`
- Create: `src/llm_forecaster/parsing.py`
- Create: `src/tests/llm_forecaster/test_prompts.py`
- Create: `src/tests/llm_forecaster/test_parsing.py`

- [ ] **Step 1: Write failing prompt preservation tests**

Create `src/tests/llm_forecaster/test_prompts.py`:

```python
from helpers import llm_prompts as legacy_prompts
from llm_forecaster import prompts


def test_forecasting_prompt_text_is_preserved_from_legacy_helpers():
    assert prompts.ZERO_SHOT_MARKET_PROMPT == legacy_prompts.ZERO_SHOT_MARKET_PROMPT
    assert (
        prompts.ZERO_SHOT_MARKET_WITH_FREEZE_VALUE_PROMPT
        == legacy_prompts.ZERO_SHOT_MARKET_WITH_FREEZE_VALUE_PROMPT
    )
    assert prompts.ZERO_SHOT_DATASET_PROMPT == legacy_prompts.ZERO_SHOT_NON_MARKET_PROMPT
    assert prompts.REFORMAT_PROMPT == legacy_prompts.REFORMAT_PROMPT
    assert prompts.REFORMAT_PROMPT_2 == legacy_prompts.REFORMAT_PROMPT_2
    assert prompts.REFORMAT_SINGLE_PROMPT == legacy_prompts.REFORMAT_SINGLE_PROMPT
    assert prompts.REFORMAT_SINGLE_PROMPT_2 == legacy_prompts.REFORMAT_SINGLE_PROMPT_2


def test_new_prompt_module_uses_dataset_terminology():
    assert hasattr(prompts, "ZERO_SHOT_DATASET_PROMPT")
    assert not hasattr(prompts, "ZERO_SHOT_NON_MARKET_PROMPT")
```

- [ ] **Step 2: Write failing parsing tests**

Create `src/tests/llm_forecaster/test_parsing.py`:

```python
from types import SimpleNamespace

from llm_forecaster import parsing


def test_extract_probability_preserves_current_reverse_search_behavior():
    assert parsing.extract_probability("first 0.2 then final *0.73*") == 0.73
    assert parsing.extract_probability("I think 61%") == 0.61
    assert parsing.extract_probability("0.0 or 1.0") is None
    assert parsing.extract_probability(None) is None


def test_convert_string_to_list_preserves_current_behavior():
    assert parsing.convert_string_to_list("[0.1, 0.25, *]") == [0.1, 0.25, 0.5]


def test_parse_market_forecast_uses_extraction_without_reformat_first():
    reformat_model = SimpleNamespace(get_response=lambda prompt: "0.25")
    assert parsing.parse_market_forecast("Reasoning. Final answer: *0.64*", reformat_model) == 0.64


def test_parse_market_forecast_does_not_reformat_when_extraction_fails():
    def get_response(prompt):
        raise AssertionError(f"Unexpected reformat call: {prompt}")

    reformat_model = SimpleNamespace(get_response=get_response)

    assert parsing.parse_market_forecast("No number here", reformat_model) is None


def test_parse_dataset_forecast_uses_list_reformat():
    calls = []

    def get_response(prompt):
        calls.append(prompt)
        return "[0.2, 0.3]"

    reformat_model = SimpleNamespace(get_response=get_response)
    question = {"resolution_dates": ["2026-06-01", "2026-07-01"]}

    assert parsing.parse_dataset_forecast(
        response="first date 20, second date 30",
        prompt="prompt text",
        question=question,
        reformat_model=reformat_model,
    ) == [0.2, 0.3]
    assert "prompt text" in calls[0]
    assert "2 resolution dates" in calls[0]


def test_parse_dataset_forecast_uses_second_list_reformat_prompt_when_needed():
    calls = []
    responses = iter(["need_a_new_reformat_prompt", "[0.4, 0.6]"])

    def get_response(prompt):
        calls.append(prompt)
        return next(responses)

    reformat_model = SimpleNamespace(get_response=get_response)
    question = {"resolution_dates": ["2026-06-01", "2026-07-01"]}

    assert parsing.parse_dataset_forecast(
        response="ambiguous response",
        prompt="original prompt",
        question=question,
        reformat_model=reformat_model,
    ) == [0.4, 0.6]
    assert len(calls) == 2
    assert "Please output the probabilistic forecasts as a Python list" in calls[0]
    assert "Output the probabilistic forecasts as a Python list" in calls[1]
```

- [ ] **Step 3: Run tests and verify they fail**

Run:

```bash
cd /workspace/forecastbench
python -m pytest src/tests/llm_forecaster/test_prompts.py src/tests/llm_forecaster/test_parsing.py -q
```

Expected: FAIL because prompt/parsing modules do not exist.

- [ ] **Step 4: Implement prompt relocation**

Create `src/llm_forecaster/prompts.py` by copying the exact text values from `helpers.llm_prompts` for:

```python
ZERO_SHOT_MARKET_PROMPT
ZERO_SHOT_MARKET_WITH_FREEZE_VALUE_PROMPT
ZERO_SHOT_DATASET_PROMPT
REFORMAT_PROMPT
REFORMAT_PROMPT_2
REFORMAT_SINGLE_PROMPT
REFORMAT_SINGLE_PROMPT_2
```

`ZERO_SHOT_DATASET_PROMPT` must use the legacy `ZERO_SHOT_NON_MARKET_PROMPT` text. Do not define `ZERO_SHOT_NON_MARKET_PROMPT` in the new module.

- [ ] **Step 5: Implement preserved parsing**

Create `src/llm_forecaster/parsing.py` with `extract_probability()` and `convert_string_to_list()` copied from current `helpers.model_eval`. Add:

```python
from typing import Any

from llm_forecaster import prompts


def parse_market_forecast(response: str | None, reformat_model) -> float | None:
    """Parse one zero-shot market forecast using current direct extraction behavior."""
    return extract_probability(response)


def parse_dataset_forecast(
    response: str | None,
    prompt: str,
    question: dict[str, Any],
    reformat_model,
) -> list[float]:
    """Parse one dataset forecast, using current list reformat behavior."""
    raw_response = reformat_model.get_response(
        prompts.REFORMAT_PROMPT.format(
            user_prompt=prompt,
            model_response=response,
            n_horizons=len(question["resolution_dates"]),
        )
    )
    if raw_response == "need_a_new_reformat_prompt":
        raw_response = reformat_model.get_response(
            prompts.REFORMAT_PROMPT_2.format(
                user_prompt=prompt,
                model_response=response,
                n_horizons=len(question["resolution_dates"]),
            )
        )
    return convert_string_to_list(raw_response)
```

- [ ] **Step 6: Run focused tests**

Run:

```bash
cd /workspace/forecastbench
python -m pytest src/tests/llm_forecaster/test_prompts.py src/tests/llm_forecaster/test_parsing.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

Run:

```bash
cd /workspace/forecastbench
git add src/llm_forecaster/prompts.py src/llm_forecaster/parsing.py src/tests/llm_forecaster/test_prompts.py src/tests/llm_forecaster/test_parsing.py
git commit -m "feat: preserve LLM prompts and parsing"
```

---

### Task 6: Add Question-Set IO Helpers and LLM Question Context

**Files:**
- Modify: `src/orchestration/_io.py`
- Create: `src/llm_forecaster/questions.py`
- Create: `src/tests/orchestration/test_question_set_io.py`
- Create: `src/tests/llm_forecaster/test_questions.py`

- [ ] **Step 1: Write failing `_io` tests**

Create `src/tests/orchestration/test_question_set_io.py`:

```python
import json

import pandas as pd

from orchestration import _io


def test_read_question_set_json_from_local_file(tmp_path):
    path = tmp_path / "2026-05-10-llm.json"
    path.write_text(
        json.dumps(
            {
                "forecast_due_date": "2026-05-10",
                "question_set": "2026-05-10-llm.json",
                "questions": [{"id": "q1", "source": "fred"}],
            }
        )
    )

    data = _io.read_question_set_json(str(path), run_locally=True)

    assert data["forecast_due_date"] == "2026-05-10"
    assert data["question_set"] == "2026-05-10-llm.json"
    assert data["questions"] == [{"id": "q1", "source": "fred"}]


def test_download_and_read_question_set_file_still_returns_dataframe(tmp_path):
    path = tmp_path / "2026-05-10-llm.json"
    path.write_text(
        json.dumps(
            {
                "forecast_due_date": "2026-05-10",
                "question_set": "2026-05-10-llm.json",
                "questions": [{"id": "q1", "source": "fred"}],
            }
        )
    )

    df = _io.download_and_read_question_set_file(str(path), run_locally=True)

    assert isinstance(df, pd.DataFrame)
    assert df.to_dict(orient="records") == [{"id": "q1", "source": "fred"}]


def test_get_latest_llm_question_set_metadata_uses_latest_file(monkeypatch):
    monkeypatch.setattr(
        _io,
        "read_question_set_json",
        lambda filename, run_locally=False: {
            "forecast_due_date": "2026-05-10",
            "question_set": "2026-05-10-llm.json",
            "questions": [],
        },
    )

    assert _io.get_latest_llm_question_set_metadata() == {
        "forecast_due_date": "2026-05-10",
        "question_set": "2026-05-10-llm.json",
    }
```

- [ ] **Step 2: Write failing question context tests**

Create `src/tests/llm_forecaster/test_questions.py`:

```python
import pytest

from llm_forecaster import questions


def test_split_questions_uses_source_registry_names():
    items = [
        {"id": "fred-1", "source": "fred"},
        {"id": "market-1", "source": "metaculus"},
    ]

    dataset, market = questions.split_questions(items)

    assert dataset == [{"id": "fred-1", "source": "fred"}]
    assert market == [{"id": "market-1", "source": "metaculus"}]


def test_split_questions_rejects_unknown_source():
    with pytest.raises(ValueError, match="Unknown question sources"):
        questions.split_questions([{"id": "q1", "source": "unknown"}])


def test_limit_questions_for_test_mode_limits_each_type():
    dataset = [{"id": f"d{i}", "source": "fred"} for i in range(4)]
    market = [{"id": f"m{i}", "source": "metaculus"} for i in range(4)]

    assert questions.limit_questions_for_test_mode(dataset, market, limit_per_type=2) == (
        dataset[:2],
        market[:2],
    )


def test_context_from_question_set_json():
    context = questions.context_from_question_set_json(
        {
            "forecast_due_date": "2026-05-10",
            "question_set": "2026-05-10-llm.json",
            "questions": [{"id": "q1", "source": "fred"}],
        }
    )

    assert context.forecast_due_date == "2026-05-10"
    assert context.question_set_filename == "2026-05-10-llm.json"
    assert context.questions == [{"id": "q1", "source": "fred"}]
```

- [ ] **Step 3: Run tests and verify they fail**

Run:

```bash
cd /workspace/forecastbench
python -m pytest src/tests/orchestration/test_question_set_io.py src/tests/llm_forecaster/test_questions.py -q
```

Expected: FAIL because the new helpers do not exist.

- [ ] **Step 4: Implement `_io` question-set JSON helpers**

In `src/orchestration/_io.py`, add:

```python
def read_question_set_json(filename: str, run_locally: bool = False) -> dict:
    """Read question set JSON from GCS or a local file."""
    local_filename = filename
    if not run_locally:
        with tempfile.NamedTemporaryFile(dir="/tmp/", delete=False) as tmp:
            local_filename = tmp.name
        gcp.storage.download(
            bucket_name=env.QUESTION_SETS_BUCKET,
            filename=filename,
            local_filename=local_filename,
        )

    with open(local_filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not run_locally:
        os.remove(local_filename)

    if not isinstance(data, dict):
        raise ValueError(f"Question set {filename} must contain a JSON object.")
    if "questions" not in data:
        raise ValueError(f"Could not download/load question set {filename}")
    return data


def get_latest_llm_question_set_metadata(run_locally: bool = False) -> dict[str, str]:
    """Return latest LLM question-set forecast due date and filename."""
    data = read_question_set_json("latest-llm.json", run_locally=run_locally)
    forecast_due_date = data.get("forecast_due_date")
    question_set = data.get("question_set")
    if not forecast_due_date or not question_set:
        raise ValueError("latest-llm.json must contain forecast_due_date and question_set.")
    return {
        "forecast_due_date": forecast_due_date,
        "question_set": question_set,
    }
```

Refactor existing `download_and_read_question_set_file()` to call `read_question_set_json()` and return the same DataFrame shape as before.

- [ ] **Step 5: Implement question context**

Create `src/llm_forecaster/questions.py`:

```python
"""Question-set loading helpers for LLM forecasting."""

from dataclasses import dataclass

from orchestration import _io
from sources import DATASET_SOURCE_NAMES, MARKET_SOURCE_NAMES


@dataclass(frozen=True)
class QuestionSetContext:
    """Question-set metadata and questions for one LLM run."""

    forecast_due_date: str
    question_set_filename: str
    questions: list[dict]


def context_from_question_set_json(data: dict) -> QuestionSetContext:
    """Build a QuestionSetContext from a question-set JSON object."""
    forecast_due_date = data["forecast_due_date"]
    return QuestionSetContext(
        forecast_due_date=forecast_due_date,
        question_set_filename=data.get("question_set", f"{forecast_due_date}-llm.json"),
        questions=data["questions"],
    )


def load_question_set_context(forecast_due_date: str, run_locally: bool = False) -> QuestionSetContext:
    """Load an LLM question set through orchestration IO."""
    return context_from_question_set_json(
        _io.read_question_set_json(f"{forecast_due_date}-llm.json", run_locally=run_locally)
    )


def split_questions(items: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split question rows into dataset and market lists."""
    invalid_sources: dict[str, list[str]] = {}
    for item in items:
        source = item.get("source")
        if source not in DATASET_SOURCE_NAMES and source not in MARKET_SOURCE_NAMES:
            invalid_sources.setdefault(str(source), []).append(str(item.get("id")))
    if invalid_sources:
        details = ", ".join(
            f"{source}: {', '.join(ids)}" for source, ids in sorted(invalid_sources.items())
        )
        raise ValueError(f"Unknown question sources: {details}")

    dataset = [item for item in items if item.get("source") in DATASET_SOURCE_NAMES]
    market = [item for item in items if item.get("source") in MARKET_SOURCE_NAMES]
    return dataset, market


def limit_questions_for_test_mode(
    dataset_questions: list[dict],
    market_questions: list[dict],
    limit_per_type: int,
) -> tuple[list[dict], list[dict]]:
    """Limit dataset and market questions for non-production runs."""
    return dataset_questions[:limit_per_type], market_questions[:limit_per_type]
```

- [ ] **Step 6: Run focused tests**

Run:

```bash
cd /workspace/forecastbench
python -m pytest src/tests/orchestration/test_question_set_io.py src/tests/llm_forecaster/test_questions.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

Run:

```bash
cd /workspace/forecastbench
git add src/orchestration/_io.py src/llm_forecaster/questions.py src/tests/orchestration/test_question_set_io.py src/tests/llm_forecaster/test_questions.py
git commit -m "feat: load LLM question sets through orchestration IO"
```

---

### Task 7: Add Runner With Dataset Reuse and Immediate Variant Writes

**Files:**
- Create: `src/llm_forecaster/runner.py`
- Create: `src/tests/llm_forecaster/test_runner.py`

- [ ] **Step 1: Write failing runner ordering tests**

Create `src/tests/llm_forecaster/test_runner.py`:

```python
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_forecaster import runner
from llm_forecaster.questions import QuestionSetContext


def _context() -> QuestionSetContext:
    return QuestionSetContext(
        forecast_due_date="2026-05-10",
        question_set_filename="2026-05-10-llm.json",
        questions=[
            {
                "id": "dataset-1",
                "source": "fred",
                "question": "Will value be higher on {resolution_date} than on {forecast_due_date}?",
                "background": "Background",
                "resolution_criteria": "Criteria",
                "freeze_datetime": "2026-05-10",
                "freeze_datetime_value": "10",
                "freeze_datetime_value_explanation": "Value explanation",
                "resolution_dates": ["2026-06-01", "2026-07-01"],
                "market_info_resolution_criteria": "N/A",
            },
            {
                "id": "market-1",
                "source": "metaculus",
                "question": "Will event happen?",
                "background": "Background",
                "resolution_criteria": "Criteria",
                "freeze_datetime": "2026-05-10",
                "freeze_datetime_value": "0.35",
                "freeze_datetime_value_explanation": "Value explanation",
                "market_info_close_datetime": "2026-07-01",
                "market_info_resolution_criteria": "N/A",
            },
        ],
    )


def test_run_model_writes_zero_shot_before_freeze_values(tmp_path, monkeypatch):
    events = []
    model_run = SimpleNamespace(
        name="test-model",
        model_organization="OpenAI",
        get_response=lambda prompt: "response",
    )

    monkeypatch.setattr(runner.parsing, "parse_dataset_forecast", lambda **kwargs: [0.2, 0.3])
    monkeypatch.setattr(runner.parsing, "parse_market_forecast", lambda response, reformat_model: 0.4)

    original_write = runner.output.write_forecast_file

    def write_then_record(**kwargs):
        events.append(("write", kwargs["variant"].key))
        original_write(**kwargs)

    monkeypatch.setattr(runner.output, "write_forecast_file", write_then_record)

    def fail_freeze_market(*args, **kwargs):
        if kwargs["variant"].uses_freeze_values:
            events.append(("freeze-started", kwargs["variant"].key))
            raise RuntimeError("freeze failure")
        return [
            {
                "id": "market-1",
                "source": "metaculus",
                "forecast": 0.4,
                "resolution_date": None,
                "reasoning": None,
            }
        ]

    monkeypatch.setattr(runner, "_forecast_market_questions", fail_freeze_market)

    with pytest.raises(RuntimeError, match="freeze failure"):
        runner.run_model(
            model_run=model_run,
            context=_context(),
            output_dir=tmp_path,
            upload=False,
            is_test=False,
            today_date="2026-05-01",
            raise_on_question_error=True,
        )

    assert events[:2] == [("write", "zero-shot"), ("freeze-started", "zero-shot-with-freeze-values")]
    assert (tmp_path / "2026-05-10.ForecastBench.test-model.json").exists()


def test_dataset_rows_are_reused_across_variants(tmp_path, monkeypatch):
    model_run = SimpleNamespace(
        name="test-model",
        model_organization="OpenAI",
        get_response=lambda prompt: "response",
    )
    dataset_call_count = 0

    def forecast_dataset(**kwargs):
        nonlocal dataset_call_count
        dataset_call_count += 1
        return [
            {
                "id": "dataset-1",
                "source": "fred",
                "forecast": 0.2,
                "resolution_date": "2026-06-01",
                "reasoning": None,
            }
        ]

    monkeypatch.setattr(runner, "_forecast_dataset_questions", forecast_dataset)
    monkeypatch.setattr(
        runner,
        "_forecast_market_questions",
        lambda **kwargs: [
            {
                "id": kwargs["variant"].key,
                "source": "metaculus",
                "forecast": 0.4,
                "resolution_date": None,
                "reasoning": None,
            }
        ],
    )

    written = runner.run_model(
        model_run=model_run,
        context=_context(),
        output_dir=tmp_path,
        upload=False,
        is_test=True,
        today_date="2026-05-01",
    )

    assert dataset_call_count == 1
    assert [item.variant.key for item in written] == [
        "zero-shot",
        "zero-shot-with-freeze-values",
    ]
    assert all(item.rows[0]["id"] == "dataset-1" for item in written)
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
cd /workspace/forecastbench
python -m pytest src/tests/llm_forecaster/test_runner.py -q
```

Expected: FAIL because `runner.py` does not exist.

- [ ] **Step 3: Implement prompt rendering and row builders in runner**

Create `src/llm_forecaster/runner.py`. Import:

```python
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from helpers import dates, env
from llm_forecaster import output, parsing, prompts, questions
from llm_forecaster.forecast_variants import FORECAST_VARIANTS, ForecastVariant
from llm_forecaster.model_runs import REFORMAT_MODEL, ModelRun
from llm_forecaster.questions import QuestionSetContext
```

Add:

```python
@dataclass(frozen=True)
class WrittenForecastFile:
    """A final forecast file written by the runner."""

    variant: ForecastVariant
    local_filename: Path
    rows: list[dict[str, Any]]
```

Implement `render_prompt()` using the existing legacy prompt parameters from `helpers.model_eval.get_prompt_params()`, but with dataset terminology and no dependency on `helpers.model_eval`.

- [ ] **Step 4: Implement `run_model()` ordering**

`run_model()` must:

1. compute `today_date` if not supplied;
2. split questions;
3. forecast dataset once;
4. forecast zero-shot market rows;
5. write/upload the zero-shot file immediately;
6. forecast freeze-values market rows;
7. write/upload the freeze-values file immediately;
8. return `WrittenForecastFile` objects in write order.

Use `output.final_filename()` for local file names and `output.destination_blob_name()` for upload destinations. Use `utils.gcp.storage.upload()` only when `upload=True`.

- [ ] **Step 5: Preserve per-question failure behavior**

Add `_handle_question_error(question, raise_on_question_error)`:

```python
def _handle_question_error(question: dict, raise_on_question_error: bool) -> None:
    """Log or raise a per-question forecasting failure."""
    if raise_on_question_error:
        raise
    logger.exception("Skipping failed LLM forecast question id=%s", question.get("id"))
```

Use it inside dataset and market question loops.

- [ ] **Step 6: Run focused tests**

Run:

```bash
cd /workspace/forecastbench
python -m pytest src/tests/llm_forecaster/test_runner.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

Run:

```bash
cd /workspace/forecastbench
git add src/llm_forecaster/runner.py src/tests/llm_forecaster/test_runner.py
git commit -m "feat: run LLM forecasts with per-variant writes"
```

---

### Task 8: Add Smoke Test

**Files:**
- Create: `src/llm_forecaster/smoke_test.py`
- Create: `src/tests/llm_forecaster/test_smoke_test.py`

- [ ] **Step 1: Write failing smoke-test tests**

Create `src/tests/llm_forecaster/test_smoke_test.py`:

```python
from types import SimpleNamespace

import pytest

from llm_forecaster import smoke_test


def test_select_model_runs_defaults_to_all():
    runs = [SimpleNamespace(name="a"), SimpleNamespace(name="b")]
    assert smoke_test.select_model_runs(runs, None) == runs


def test_select_model_runs_rejects_missing_name():
    runs = [SimpleNamespace(name="a")]
    with pytest.raises(ValueError, match="missing"):
        smoke_test.select_model_runs(runs, ["missing"])


def test_select_questions_takes_deterministic_prefix_by_id():
    questions = [{"id": "b"}, {"id": "a"}, {"id": "c"}]
    assert smoke_test.select_questions(questions, sample_size=2) == [{"id": "a"}, {"id": "b"}]


def test_exit_code_for_results_fails_on_any_failure():
    assert smoke_test.exit_code_for_results([]) == 1
    assert smoke_test.exit_code_for_results([SimpleNamespace(status=smoke_test.PASS)]) == 0
    assert smoke_test.exit_code_for_results([SimpleNamespace(status=smoke_test.FAIL)]) == 1


def test_run_smoke_test_continues_after_model_failure(monkeypatch):
    model_runs = [
        SimpleNamespace(name="bad", lab=SimpleNamespace(name="Lab"), provider=SimpleNamespace(name="Provider")),
        SimpleNamespace(name="good", lab=SimpleNamespace(name="Lab"), provider=SimpleNamespace(name="Provider")),
    ]
    questions = [{"id": "q1", "source": "fred"}]

    def run_one(**kwargs):
        if kwargs["model_run"].name == "bad":
            raise RuntimeError("provider unavailable")
        return [SimpleNamespace(rows=[{"id": "q1", "forecast": 0.5}])]

    monkeypatch.setattr(smoke_test.runner, "run_model", run_one)

    smoke_run = smoke_test.run_smoke_test(
        model_runs=model_runs,
        context=SimpleNamespace(
            forecast_due_date="2026-05-10",
            question_set_filename="2026-05-10-llm.json",
            questions=questions,
        ),
        output_dir="/tmp/smoke",
    )

    assert [result.status for result in smoke_run.results] == [smoke_test.FAIL, smoke_test.PASS]
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
cd /workspace/forecastbench
python -m pytest src/tests/llm_forecaster/test_smoke_test.py -q
```

Expected: FAIL because `smoke_test.py` does not exist.

- [ ] **Step 3: Implement smoke-test data classes and selection helpers**

Create `src/llm_forecaster/smoke_test.py` with:

```python
"""Run a full-path smoke test for configured ForecastBench LLM model runs."""

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from llm_forecaster import model_runs, questions, runner

logger = logging.getLogger(__name__)

PASS = "PASS"
FAIL = "FAIL"
SMOKE_OUTPUT_DIR = "/tmp/forecasts/llm_smoke_test"


@dataclass(frozen=True)
class SmokeResult:
    """Result for one model smoke check."""

    model_name: str
    lab: str
    provider: str
    status: str
    error_type: str
    error_message: str


@dataclass(frozen=True)
class SmokeRun:
    """Smoke-test results and local forecast rows."""

    results: list[SmokeResult]
    forecast_file_paths: list[str]
```

Add `select_questions()`, `select_model_runs()`, `exit_code_for_results()`, and `parse_args()` modeled on time-series-benchmark with ForecastBench-specific flags: `--forecast-due-date`, `--sample-size`, `--model-run`.

- [ ] **Step 4: Implement smoke run**

`run_smoke_test()` should call `runner.run_model()` once per selected model with `upload=False`, `is_test=True`, `raise_on_question_error=True`, and the selected question context. It should catch exceptions per model, append a `FAIL` result, and continue.

- [ ] **Step 5: Implement command `main()`**

`main()` should:

1. parse args;
2. load latest metadata from `_io` if no forecast due date is supplied;
3. load a question context through `questions.load_question_set_context()`;
4. select deterministic questions;
5. select model runs;
6. configure provider keys;
7. run smoke tests;
8. log result rows;
9. exit with `exit_code_for_results()`.

- [ ] **Step 6: Run focused tests**

Run:

```bash
cd /workspace/forecastbench
python -m pytest src/tests/llm_forecaster/test_smoke_test.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

Run:

```bash
cd /workspace/forecastbench
git add src/llm_forecaster/smoke_test.py src/tests/llm_forecaster/test_smoke_test.py
git commit -m "feat: add LLM forecaster smoke test"
```

---

### Task 9: Add Cloud Run Manager and Worker

**Files:**
- Create: `src/orchestration/func_llm_forecaster_manager/main.py`
- Create: `src/orchestration/func_llm_forecaster_manager/Makefile`
- Create: `src/orchestration/func_llm_forecaster_manager/requirements.txt`
- Create: `src/orchestration/func_llm_forecaster_worker/main.py`
- Create: `src/orchestration/func_llm_forecaster_worker/Makefile`
- Create: `src/orchestration/func_llm_forecaster_worker/requirements.txt`
- Create: `src/tests/orchestration/test_llm_forecaster_manager.py`
- Create: `src/tests/orchestration/test_llm_forecaster_worker.py`

- [ ] **Step 1: Write failing manager tests**

Create `src/tests/orchestration/test_llm_forecaster_manager.py`:

```python
from types import SimpleNamespace

from orchestration.func_llm_forecaster_manager import main as manager


def test_run_manager_uses_io_latest_metadata_and_new_worker(monkeypatch):
    calls = {}
    monkeypatch.setattr(
        manager._io,
        "get_latest_llm_question_set_metadata",
        lambda: {"forecast_due_date": "2026-05-10", "question_set": "2026-05-10-llm.json"},
    )
    monkeypatch.setattr(manager.model_runs, "MODEL_RUNS", [object(), object(), object()])
    monkeypatch.setattr(
        manager.cloud_run,
        "call_worker",
        lambda **kwargs: calls.setdefault("call_worker", kwargs) or "operation",
    )
    monkeypatch.setattr(
        manager.cloud_run,
        "block_and_check_job_result",
        lambda **kwargs: calls.setdefault("block", kwargs),
    )

    manager.run_manager(manager.RunMode.TEST)

    assert calls["call_worker"]["job_name"] == "func-llm-forecaster-worker"
    assert calls["call_worker"]["task_count"] == 3
    assert calls["call_worker"]["env_vars"]["FORECAST_DUE_DATE"] == "2026-05-10"
    assert calls["call_worker"]["env_vars"]["TEST_OR_PROD"] == "TEST"
    assert calls["block"]["name"] == "llm-forecaster"
```

- [ ] **Step 2: Write failing worker tests**

Create `src/tests/orchestration/test_llm_forecaster_worker.py`:

```python
from types import SimpleNamespace

from helpers import constants
from orchestration.func_llm_forecaster_worker import main as worker


def test_parse_env_defaults_missing_test_or_prod_to_test(monkeypatch):
    monkeypatch.setenv("FORECAST_DUE_DATE", "2026-05-10")
    monkeypatch.setenv("CLOUD_RUN_TASK_INDEX", "0")
    monkeypatch.delenv("TEST_OR_PROD", raising=False)
    monkeypatch.setattr(worker.model_runs, "MODEL_RUNS", [SimpleNamespace(name="model")])

    forecast_due_date, run_mode, model_run = worker.parse_env_vars()

    assert forecast_due_date == "2026-05-10"
    assert run_mode == constants.RunMode.TEST
    assert model_run.name == "model"


def test_non_prod_value_defaults_to_test(monkeypatch):
    monkeypatch.setenv("FORECAST_DUE_DATE", "2026-05-10")
    monkeypatch.setenv("CLOUD_RUN_TASK_INDEX", "0")
    monkeypatch.setenv("TEST_OR_PROD", "DEV")
    monkeypatch.setattr(worker.model_runs, "MODEL_RUNS", [SimpleNamespace(name="model")])

    assert worker.parse_env_vars()[1] == constants.RunMode.TEST


def test_worker_limits_questions_when_not_prod(monkeypatch, tmp_path):
    calls = {}
    context = SimpleNamespace(
        forecast_due_date="2026-05-10",
        question_set_filename="2026-05-10-llm.json",
        questions=[
            {"id": "d1", "source": "fred"},
            {"id": "d2", "source": "fred"},
            {"id": "d3", "source": "fred"},
            {"id": "m1", "source": "metaculus"},
            {"id": "m2", "source": "metaculus"},
            {"id": "m3", "source": "metaculus"},
        ],
    )

    monkeypatch.setattr(worker.questions, "load_question_set_context", lambda forecast_due_date: context)
    monkeypatch.setattr(worker.model_runs, "configure_and_validate_provider_keys", lambda runs: None)
    monkeypatch.setattr(
        worker.runner,
        "run_model",
        lambda **kwargs: calls.setdefault("context", kwargs["context"]),
    )

    worker.run_worker(
        forecast_due_date="2026-05-10",
        run_mode=constants.RunMode.TEST,
        model_run=SimpleNamespace(name="model"),
    )

    assert [q["id"] for q in calls["context"].questions] == ["d1", "d2", "m1", "m2"]
```

- [ ] **Step 3: Run tests and verify they fail**

Run:

```bash
cd /workspace/forecastbench
python -m pytest src/tests/orchestration/test_llm_forecaster_manager.py src/tests/orchestration/test_llm_forecaster_worker.py -q
```

Expected: FAIL because manager/worker packages do not exist.

- [ ] **Step 4: Implement manager**

Create `src/orchestration/func_llm_forecaster_manager/main.py` with direct imports from `helpers.cloud_run`, `helpers.constants`, `helpers.decorator`, `llm_forecaster.model_runs`, and `orchestration._io`. Do not add a `_CloudRun` wrapper.

`run_manager()` must:

```python
def run_manager(run_mode: constants.RunMode) -> None:
    """Launch one worker task per model run."""
    metadata = _io.get_latest_llm_question_set_metadata()
    forecast_due_date = metadata["forecast_due_date"]
    timeout = cloud_run.timeout_1h * 24
    operation = cloud_run.call_worker(
        job_name="func-llm-forecaster-worker",
        env_vars={
            "FORECAST_DUE_DATE": forecast_due_date,
            "TEST_OR_PROD": run_mode.value,
        },
        task_count=len(model_runs.MODEL_RUNS),
        timeout=timeout,
    )
    cloud_run.block_and_check_job_result(
        operation=operation,
        name="llm-forecaster",
        exit_on_error=True,
        timeout=timeout,
    )
```

- [ ] **Step 5: Implement worker**

Create `src/orchestration/func_llm_forecaster_worker/main.py` with:

```python
DEFAULT_TEST_QUESTIONS_PER_TYPE = 2
```

`parse_env_vars()` must require `FORECAST_DUE_DATE`, parse `CLOUD_RUN_TASK_INDEX`, select `model_runs.MODEL_RUNS[task_num]`, and set run mode to `constants.RunMode.PROD` only when `TEST_OR_PROD == "PROD"`; otherwise use `constants.RunMode.TEST`.

`run_worker()` must load context through `questions.load_question_set_context()`, split and limit to 2 dataset/2 market questions when not PROD, rebuild `QuestionSetContext`, configure provider keys for `[model_run]`, and call `runner.run_model()`.

- [ ] **Step 6: Add deploy Makefiles and requirements**

Create manager and worker deploy files using the current Cloud Run Docker template pattern. Each Makefile must stage:

```make
cat $(ROOT_DIR)requirements.runtime.txt requirements.txt > $(UPLOAD_DIR)/requirements.txt
cp -r $(ROOT_DIR)src/helpers $(UPLOAD_DIR)/
cp -r $(ROOT_DIR)src/sources $(UPLOAD_DIR)/
cp -r $(ROOT_DIR)src/llm_forecaster $(UPLOAD_DIR)/
mkdir -p $(UPLOAD_DIR)/orchestration
cp $(ROOT_DIR)src/orchestration/__init__.py $(UPLOAD_DIR)/orchestration/
cp $(ROOT_DIR)src/orchestration/_io.py $(UPLOAD_DIR)/orchestration/
```

Job names:

```text
func-llm-forecaster-manager
func-llm-forecaster-worker
```

- [ ] **Step 7: Run focused tests**

Run:

```bash
cd /workspace/forecastbench
python -m pytest src/tests/orchestration/test_llm_forecaster_manager.py src/tests/orchestration/test_llm_forecaster_worker.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

Run:

```bash
cd /workspace/forecastbench
git add src/orchestration/func_llm_forecaster_manager src/orchestration/func_llm_forecaster_worker src/tests/orchestration/test_llm_forecaster_manager.py src/tests/orchestration/test_llm_forecaster_worker.py
git commit -m "feat: add LLM forecaster Cloud Run jobs"
```

---

### Task 10: Add Strict Leaderboard Legacy LLM Name Mapping

**Files:**
- Create: `src/leaderboard/llm_legacy_names.py`
- Modify: `src/leaderboard/main.py`
- Modify: `src/leaderboard/model_release_dates.csv`
- Create: `src/tests/leaderboard/test_llm_legacy_names.py`

- [ ] **Step 1: Write failing legacy mapping tests**

Create `src/tests/leaderboard/test_llm_legacy_names.py`:

```python
import pytest

from leaderboard import llm_legacy_names


def test_normalize_current_canonical_llm_name_passes_through():
    data = {
        "organization": "ForecastBench",
        "model_organization": "Anthropic",
        "model": "claude-opus-4-7-1024-with-freeze-values",
    }

    assert llm_legacy_names.normalize_llm_identity(data) == data


def test_normalize_legacy_zero_shot_name_to_canonical_display_name():
    data = {
        "organization": "ForecastBench",
        "model_organization": "Anthropic",
        "model": "Claude-Opus-4-7 (zero shot with freeze values)",
    }

    normalized = llm_legacy_names.normalize_llm_identity(data)

    assert normalized["model_organization"] == "Anthropic"
    assert normalized["model"] == "claude-opus-4-7-1024-with-freeze-values"


def test_unmapped_forecastbench_llm_raises():
    with pytest.raises(KeyError, match="Unmapped legacy ForecastBench LLM model"):
        llm_legacy_names.normalize_llm_identity(
            {
                "organization": "ForecastBench",
                "model_organization": "Anthropic",
                "model": "Unknown LLM (zero shot)",
            }
        )


def test_non_forecastbench_identity_passes_through():
    data = {
        "organization": "External",
        "model_organization": "External",
        "model": "External model",
    }
    assert llm_legacy_names.normalize_llm_identity(data) == data
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
cd /workspace/forecastbench
python -m pytest src/tests/leaderboard/test_llm_legacy_names.py -q
```

Expected: FAIL because `leaderboard.llm_legacy_names` does not exist.

- [ ] **Step 3: Implement strict mapping module**

Create `src/leaderboard/llm_legacy_names.py`. Include:

```python
"""Strict legacy ForecastBench LLM identity normalization."""

from copy import deepcopy

FORECASTBENCH_ORG = "ForecastBench"
LEGACY_VARIANT_MARKERS = (
    "zero shot",
    "zero_shot",
    "scratchpad",
    "with freeze values",
)

LEGACY_LLM_NAME_MAP: dict[tuple[str, str], tuple[str, str]] = {
    ("Anthropic", "Claude-Opus-4-7 (zero shot with freeze values)"): (
        "Anthropic",
        "claude-opus-4-7-1024-with-freeze-values",
    ),
}

CANONICAL_LLM_NAMES = {
    "claude-opus-4-7-1024",
    "claude-opus-4-7-1024-with-freeze-values",
}
```

Populate `LEGACY_LLM_NAME_MAP` for all historical ForecastBench LLM names and variants present in `model_release_dates.csv` and historical forecast files. The implementation task must gather names with:

```bash
cd /workspace/forecastbench
rg -n "\"model\"|ForecastBench|zero_shot|scratchpad|with freeze values" ../forecastbench-forecast-sets src/leaderboard/model_release_dates.csv
```

Do not rename historical files.

Implement:

```python
def normalize_llm_identity(data: dict) -> dict:
    """Normalize ForecastBench LLM identity for leaderboard ingest."""
    normalized = deepcopy(data)
    if normalized.get("organization") != FORECASTBENCH_ORG:
        return normalized

    model = normalized.get("model")
    model_organization = normalized.get("model_organization")
    if model in CANONICAL_LLM_NAMES:
        return normalized

    is_legacy_llm = any(marker in str(model).lower() for marker in LEGACY_VARIANT_MARKERS)
    if not is_legacy_llm:
        return normalized

    key = (model_organization, model)
    try:
        new_model_organization, new_model = LEGACY_LLM_NAME_MAP[key]
    except KeyError as exc:
        raise KeyError(
            f"Unmapped legacy ForecastBench LLM model: "
            f"model_organization={model_organization!r}, model={model!r}"
        ) from exc

    normalized["model_organization"] = new_model_organization
    normalized["model"] = new_model
    return normalized
```

- [ ] **Step 4: Integrate mapping in leaderboard ingest**

In `src/leaderboard/main.py`, call `llm_legacy_names.normalize_llm_identity()` immediately after forecast file `organization`, `model`, and `model_organization` are read and before `set_model_pk()`.

- [ ] **Step 5: Update model release dates**

Add canonical active model rows to `src/leaderboard/model_release_dates.csv`. Keep historical rows needed for old non-normalized contexts only if tests require them. Use canonical displayed names such as:

```csv
claude-opus-4-7-1024,2026-01-01
claude-opus-4-7-1024-with-freeze-values,2026-01-01
```

Use the best known release dates already represented by the old rows for each model. Do not invent dates silently; if a release date is unavailable locally, use the date already present for the corresponding legacy model row and document the source in the commit message body.

- [ ] **Step 6: Run focused tests**

Run:

```bash
cd /workspace/forecastbench
python -m pytest src/tests/leaderboard/test_llm_legacy_names.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

Run:

```bash
cd /workspace/forecastbench
git add src/leaderboard/llm_legacy_names.py src/leaderboard/main.py src/leaderboard/model_release_dates.csv src/tests/leaderboard/test_llm_legacy_names.py
git commit -m "feat: normalize legacy ForecastBench LLM names"
```

---

### Task 11: Route Nightly and Root Targets to the New Jobs

**Files:**
- Modify: `Makefile`
- Modify: `src/nightly_update_workflow/worker/main.py`
- Create: `src/tests/test_nightly_workflow_llm_forecaster.py`

- [ ] **Step 1: Write failing nightly workflow test**

Create `src/tests/test_nightly_workflow_llm_forecaster.py`:

```python
from nightly_update_workflow.worker import main as worker


def test_publish_question_set_launches_new_llm_forecaster_job(monkeypatch):
    monkeypatch.setattr(worker.question_curation, "is_today_question_set_publication_date", lambda: True)

    jobs = worker.get_publish_question_set_make_llm_baseline()

    assert jobs == [
        [
            ("func-question-set-publish", True, worker.cloud_run.timeout_1h, 1),
            ("func-llm-forecaster-manager", True, worker.cloud_run.timeout_1h * 24, 1),
        ],
    ]
```

- [ ] **Step 2: Write failing Makefile check**

Add this test to `src/tests/test_runtime_requirements.py`:

```python
def test_root_makefile_routes_llm_baseline_targets_to_refactored_jobs():
    makefile = (ROOT / "Makefile").read_text()

    assert "$(MAKE) -C src/orchestration/func_llm_forecaster_manager" in makefile
    assert "$(MAKE) -C src/orchestration/func_llm_forecaster_worker" in makefile
    assert "src/base_eval/llm_baselines/manager" not in makefile
    assert "src/base_eval/llm_baselines/worker" not in makefile
```

- [ ] **Step 3: Run tests and verify they fail**

Run:

```bash
cd /workspace/forecastbench
python -m pytest src/tests/test_nightly_workflow_llm_forecaster.py src/tests/test_runtime_requirements.py::test_root_makefile_routes_llm_baseline_targets_to_refactored_jobs -q
```

Expected: FAIL because old job names and old Makefile paths remain.

- [ ] **Step 4: Update nightly workflow**

In `src/nightly_update_workflow/worker/main.py`, replace:

```python
("func-baseline-llm-forecasts-manager", True, cloud_run.timeout_1h * 24, 1),
```

with:

```python
("func-llm-forecaster-manager", True, cloud_run.timeout_1h * 24, 1),
```

- [ ] **Step 5: Update root Makefile targets**

Keep user-facing target names if useful, but point them to new directories:

```make
llm-baseline-manager:
	$(MAKE) -C src/orchestration/func_llm_forecaster_manager || echo "* $@" >> $(MAKE_FAILURE_LOG)

llm-baseline-worker:
	$(MAKE) -C src/orchestration/func_llm_forecaster_worker || echo "* $@" >> $(MAKE_FAILURE_LOG)
```

- [ ] **Step 6: Run focused tests**

Run:

```bash
cd /workspace/forecastbench
python -m pytest src/tests/test_nightly_workflow_llm_forecaster.py src/tests/test_runtime_requirements.py::test_root_makefile_routes_llm_baseline_targets_to_refactored_jobs -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

Run:

```bash
cd /workspace/forecastbench
git add Makefile src/nightly_update_workflow/worker/main.py src/tests/test_nightly_workflow_llm_forecaster.py src/tests/test_runtime_requirements.py
git commit -m "chore: route nightly LLM baselines to refactored jobs"
```

---

### Task 12: Remove Legacy LLM Constants and Trim Legacy Helpers

**Files:**
- Modify: `src/helpers/constants.py`
- Modify: `src/helpers/model_eval.py`
- Modify: `src/metadata/tag_questions/main.py`
- Modify: `src/metadata/validate_questions/main.py`
- Create: `src/tests/test_model_request_params.py`

- [ ] **Step 1: Write failing cleanup tests**

Create `src/tests/test_model_request_params.py`:

```python
import inspect

from helpers import constants, model_eval
from metadata.tag_questions import main as tag_questions
from metadata.validate_questions import main as validate_questions


def test_old_llm_model_registry_removed_from_constants():
    assert not hasattr(constants, "MODELS_TO_RUN")
    assert not hasattr(constants, "MODELS_TO_RUN_BY_SOURCE")
    assert not hasattr(constants, "MODEL_NAME_TO_SOURCE")
    assert not hasattr(constants, "MODEL_TOKEN_LIMITS")


def test_metadata_callers_do_not_use_legacy_model_eval_routing():
    assert "model_eval.get_response_from_model" not in inspect.getsource(tag_questions)
    assert "model_eval.get_response_from_model" not in inspect.getsource(validate_questions)


def test_legacy_model_eval_no_longer_contains_provider_sdk_routing():
    source = inspect.getsource(model_eval)
    assert "get_response_from_anthropic_model" not in source
    assert "get_response_from_together_ai_model" not in source
    assert "constants.MODELS_TO_RUN" not in source
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
cd /workspace/forecastbench
python -m pytest src/tests/test_model_request_params.py -q
```

Expected: FAIL because old constants and routing remain.

- [ ] **Step 3: Add a small metadata LLM helper**

Create or add to `src/helpers/model_eval.py` a minimal utils-backed helper for metadata:

```python
from utils.llm.lab_registry import LABS
from utils.llm.provider_registry import PROVIDERS
from utils.llm.model_registry import get_response


def get_metadata_model_response(prompt: str, max_output_tokens: int) -> str:
    """Return metadata model response through shared utils routing."""
    return get_response(
        provider=PROVIDERS["OpenAI"],
        model_id="gpt-5-mini",
        prompt=prompt,
        options={"max_output_tokens": max_output_tokens},
    )
```

If `gpt-5-mini` is not accepted by the OpenAI Responses API in current utils, use the exact metadata model id from `helpers.question_curation.METADATA_MODEL_NAME` and keep the test explicit.

- [ ] **Step 4: Update metadata callers**

In `src/metadata/tag_questions/main.py`, replace the `asyncio.to_thread()` call target with:

```python
model_eval.get_metadata_model_response,
prompt=prompt,
max_output_tokens=50,
```

In `src/metadata/validate_questions/main.py`, replace the call with:

```python
model_eval.get_metadata_model_response,
prompt=prompt,
max_output_tokens=500,
```

Leave tagging and validation prompts in `helpers.llm_prompts.py`.

- [ ] **Step 5: Remove old constants**

From `src/helpers/constants.py`, remove:

- provider source constants used only by old LLM model routing;
- `MODELS_TO_RUN`;
- `MODEL_TOKEN_LIMITS`;
- `MODEL_NAME_TO_SOURCE`;
- `MODELS_TO_RUN_BY_SOURCE`;
- old derived active LLM model maps.

Keep organization/logo constants if `src/leaderboard` or the website still imports them.

- [ ] **Step 6: Trim `helpers.model_eval`**

Remove old provider SDK clients and functions that existed for `src/base_eval/llm_baselines`, including direct OpenAI/Anthropic/Together/Google/xAI call helpers and final-file generation helpers. Keep only code still imported by metadata or other live modules. If no live imports remain after metadata changes, delete `src/helpers/model_eval.py` and update imports.

- [ ] **Step 7: Run focused tests**

Run:

```bash
cd /workspace/forecastbench
python -m pytest src/tests/test_model_request_params.py src/tests/test_runtime_requirements.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

Run:

```bash
cd /workspace/forecastbench
git add src/helpers/constants.py src/helpers/model_eval.py src/metadata/tag_questions/main.py src/metadata/validate_questions/main.py src/tests/test_model_request_params.py
git add -u src/helpers/model_eval.py
git commit -m "refactor: remove legacy LLM model routing"
```

---

### Task 13: Delete Legacy LLM Baselines and Forecasting Prompts

**Files:**
- Delete: `src/base_eval/llm_baselines`
- Modify: `src/helpers/llm_prompts.py`
- Modify: `src/tests/llm_forecaster/test_prompts.py`
- Create: `src/tests/test_legacy_llm_cleanup.py`

- [ ] **Step 1: Write failing deletion-gate tests**

Create `src/tests/test_legacy_llm_cleanup.py`:

```python
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_legacy_llm_baselines_directory_deleted():
    assert not (ROOT / "src/base_eval/llm_baselines").exists()


def test_helpers_llm_prompts_keeps_only_metadata_prompts():
    text = (ROOT / "src/helpers/llm_prompts.py").read_text()

    assert "ASSIGN_CATEGORY_PROMPT" in text
    assert "VALIDATE_QUESTION_PROMPT" in text
    assert "ZERO_SHOT_MARKET_PROMPT" not in text
    assert "ZERO_SHOT_NON_MARKET_PROMPT" not in text
    assert "REFORMAT_SINGLE_PROMPT" not in text
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
cd /workspace/forecastbench
python -m pytest src/tests/test_legacy_llm_cleanup.py -q
```

Expected: FAIL because legacy directory and forecasting prompts remain.

- [ ] **Step 3: Delete legacy LLM baseline code**

Run:

```bash
cd /workspace/forecastbench
git rm -r src/base_eval/llm_baselines
```

- [ ] **Step 4: Remove forecasting prompts from `helpers.llm_prompts`**

In `src/helpers/llm_prompts.py`, remove:

- `ZERO_SHOT_MARKET_PROMPT`;
- `ZERO_SHOT_MARKET_WITH_FREEZE_VALUE_PROMPT`;
- `ZERO_SHOT_NON_MARKET_PROMPT`;
- `REFORMAT_PROMPT`;
- `REFORMAT_PROMPT_2`;
- `REFORMAT_SINGLE_PROMPT`;
- `REFORMAT_SINGLE_PROMPT_2`.

Keep:

- `ASSIGN_CATEGORY_PROMPT`;
- `VALIDATE_QUESTION_PROMPT`;
- imports needed by those prompts.

- [ ] **Step 5: Update prompt tests after deletion**

Update `src/tests/llm_forecaster/test_prompts.py` so it no longer imports `helpers.llm_prompts` for the deleted forecasting prompt strings. Replace the preservation test with exact string snapshots in `llm_forecaster.prompts`.

- [ ] **Step 6: Run cleanup tests**

Run:

```bash
cd /workspace/forecastbench
python -m pytest src/tests/test_legacy_llm_cleanup.py src/tests/llm_forecaster/test_prompts.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

Run:

```bash
cd /workspace/forecastbench
git add src/helpers/llm_prompts.py src/tests/test_legacy_llm_cleanup.py src/tests/llm_forecaster/test_prompts.py
git add -u src/base_eval/llm_baselines
git commit -m "chore: delete legacy LLM baseline jobs"
```

---

### Task 14: Full Cleanup and Verification

**Files:**
- Modify any file identified by cleanup tests.
- No new production features.

- [ ] **Step 1: Run cleanup searches**

Run:

```bash
cd /workspace/forecastbench
rg -n "_CANONICAL_MODEL_RUN_KEYS|_options_for_model_run|conflicting_options|_CloudRun|MODELS_TO_RUN|MODELS_TO_RUN_BY_SOURCE|ZERO_SHOT_NON_MARKET|non_market|src/base_eval/llm_baselines|cp -r \\$\\(ROOT_DIR\\)utils|\\^/utils/" .
```

Expected: no matches in tracked production/test code. Matches inside docs/specs/plans are acceptable.

- [ ] **Step 2: Run focused test suite**

Run:

```bash
cd /workspace/forecastbench
python -m pytest \
  src/tests/test_runtime_requirements.py \
  src/tests/test_utils_cleanup.py \
  src/tests/test_model_request_params.py \
  src/tests/test_legacy_llm_cleanup.py \
  src/tests/test_nightly_workflow_llm_forecaster.py \
  src/tests/orchestration/test_question_set_io.py \
  src/tests/orchestration/test_llm_forecaster_manager.py \
  src/tests/orchestration/test_llm_forecaster_worker.py \
  src/tests/llm_forecaster \
  src/tests/leaderboard/test_llm_legacy_names.py \
  -q
```

Expected: PASS.

- [ ] **Step 3: Run full ForecastBench tests**

Run:

```bash
cd /workspace/forecastbench
python -m pytest src/tests -q
```

Expected: PASS.

- [ ] **Step 4: Run lint**

Run:

```bash
cd /workspace/forecastbench
make lint
```

Expected: PASS.

- [ ] **Step 5: Verify utils remains clean**

Run:

```bash
cd /workspace/utils
git status --short --branch
python -m pytest tests/unit/test_llm_routing.py tests/unit/test_model_registry_config.py -q
```

Expected: utils branch is clean after the MiniMax commit; tests PASS.

- [ ] **Step 6: Commit any final cleanup**

If Steps 1-5 required tracked edits, commit them:

```bash
cd /workspace/forecastbench
git add -A
git commit -m "test: verify LLM forecaster cleanup"
```

If Steps 1-5 required no tracked edits, do not create an empty commit.

---

## Execution Notes

- Implement tasks in order.
- Commit after every task that changes files.
- Do not restore or delete unrelated untracked files.
- Do not restore `/workspace/forecastbench-utils-clean-rewrite-shadow` into the ForecastBench repo.
- Use Python 3.14+ syntax in new ForecastBench files and do not add `from __future__ import annotations`.
- Use `superpowers:test-driven-development` before implementation tasks.
- Use `superpowers:verification-before-completion` before each commit and before final completion claims.

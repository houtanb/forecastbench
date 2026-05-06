import inspect
from pathlib import Path
from unittest.mock import patch

import pytest
from utils.llm import model_registry as shared_model_registry
from utils.llm import model_runs as shared_model_runs
from utils.llm.provider_registry import PROVIDERS

from llm_forecaster import model_runs

LEGACY_MODEL_RUN_KEY_HELPER = "_".join(("", "CANONICAL", "MODEL", "RUN", "KEYS"))
LEGACY_MODEL_RUN_OPTIONS_HELPER = "_".join(("", "options", "for", "model", "run"))
LEGACY_CONFLICTING_RUN_OPTIONS = "_".join(("conflicting", "options"))
LEGACY_MODEL_RUNS_MAP = "_".join(("MODELS", "TO", "RUN"))
LEGACY_MODEL_RUN_CONSTRUCTOR = "Model" + "Run("


def test_forecastbench_selects_shared_model_run_objects():
    expected_keys = [
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

    assert model_runs.FORECASTBENCH_MODEL_RUN_KEYS == expected_keys
    assert [run.model_run_key for run in model_runs.MODEL_RUNS] == expected_keys
    assert model_runs.MODEL_RUNS == shared_model_runs.select_model_runs(expected_keys)
    assert all(isinstance(run, shared_model_runs.ModelRun) for run in model_runs.MODEL_RUNS)
    assert all(run.model.active for run in model_runs.MODEL_RUNS)


def test_model_run_calls_utils_with_provider_model_id_and_options():
    run = model_runs.get_model_run_by_slug("deepseek-v4-pro")

    with patch("utils.llm.model_registry.get_response", return_value="0.61") as mock_call:
        assert run.get_response("prompt", max_tokens=128) == "0.61"

    mock_call.assert_called_once_with(
        provider=PROVIDERS["Together"],
        model_id="deepseek-ai/DeepSeek-V4-Pro",
        prompt="prompt",
        options={"temperature": 0, "max_tokens": 128},
    )


def test_model_run_slugs_are_unique_and_file_safe():
    slugs = [run.slug for run in model_runs.MODEL_RUNS]

    assert slugs
    assert len(slugs) == len(set(slugs))
    assert all(slug == slug.lower() for slug in slugs)
    assert all(" " not in slug and "/" not in slug and "_" not in slug for slug in slugs)


def test_labs_and_providers_are_shared_registry_objects():
    runs = {run.slug: run for run in model_runs.MODEL_RUNS}

    assert runs["minimax-m2.7"].lab == shared_model_registry.MODELS_BY_KEY["minimax-m2.7"].lab
    assert runs["minimax-m2.7"].provider == PROVIDERS["Together"]
    assert runs["kimi-k2.6"].lab == shared_model_registry.MODELS_BY_KEY["kimi-k2.6"].lab
    assert runs["gemma-4-31b"].lab == shared_model_registry.MODELS_BY_KEY["gemma-4-31b"].lab
    assert runs["gemma-4-31b"].provider == PROVIDERS["Together"]


def test_model_organization_uses_lab_name():
    run = model_runs.get_model_run_by_slug("deepseek-v4-pro")

    assert run.lab.name == shared_model_registry.MODELS_BY_KEY["deepseek-v4-pro"].lab.name


def test_options_are_declared_on_model_runs_not_inferred_by_helpers():
    source = inspect.getsource(model_runs)

    assert LEGACY_MODEL_RUN_KEY_HELPER not in source
    assert LEGACY_MODEL_RUN_OPTIONS_HELPER not in source
    assert LEGACY_CONFLICTING_RUN_OPTIONS not in source
    assert LEGACY_MODEL_RUNS_MAP not in source
    assert "from helpers import constants" not in source
    assert "google.genai" not in source

    runs = {run.slug: run for run in model_runs.MODEL_RUNS}
    anthropic_runs = [
        run for run in model_runs.MODEL_RUNS if run.provider == PROVIDERS["Anthropic"]
    ]
    assert anthropic_runs
    assert all(run.options.get("max_tokens") == 1024 for run in anthropic_runs)
    assert all(run.slug.endswith("-1024") for run in anthropic_runs)
    assert runs["deepseek-v4-pro"].options == {"temperature": 0}
    assert runs["grok-4.3"].options == {"temperature": 0}
    assert runs["gemini-3.5-flash"].options == {
        "candidate_count": 1,
        "temperature": 0,
        "automatic_function_calling": {"disable": True},
    }
    assert runs["claude-opus-4-7-1024"].options == {"max_tokens": 1024}
    for run in model_runs.MODEL_RUNS:
        if run.provider in (PROVIDERS["Together"], PROVIDERS["xAI"]):
            assert run.options.get("temperature") == 0
    for run in model_runs.MODEL_RUNS:
        if run.provider == PROVIDERS["Google"]:
            assert run.options.get("temperature") == 0
    for run in anthropic_runs:
        if run.slug == "claude-opus-4-7-1024":
            assert "temperature" not in run.options
        else:
            assert run.options.get("temperature") == 0


def test_forecastbench_does_not_declare_local_model_runs():
    source = inspect.getsource(model_runs)

    assert "@dataclass" not in source
    assert LEGACY_MODEL_RUN_CONSTRUCTOR not in source
    assert "OPENAI_MODEL_RUNS" not in source
    assert "TOGETHER_MODEL_RUNS" not in source
    assert "ANTHROPIC_MODEL_RUNS" not in source
    assert "XAI_MODEL_RUNS" not in source
    assert "GOOGLE_MODEL_RUNS" not in source


def test_forecastbench_keys_select_real_shared_options():
    for key in model_runs.FORECASTBENCH_MODEL_RUN_KEYS:
        assert model_runs.get_model_run(key) == shared_model_runs.get_model_run(key)

    for run in model_runs.MODEL_RUNS:
        if run.provider == PROVIDERS["Together"]:
            assert run.options.get("temperature") == 0
    for run in model_runs.MODEL_RUNS:
        if run.provider == PROVIDERS["xAI"]:
            assert run.options.get("temperature") == 0
    for run in model_runs.MODEL_RUNS:
        if run.provider == PROVIDERS["Google"]:
            assert run.options.get("temperature") == 0


def test_model_run_lookup_raises_for_missing_key():
    with pytest.raises(
        KeyError,
        match="Unknown ForecastBench LLM model run key: unknown-model",
    ):
        model_runs.get_model_run("unknown-model")


def test_model_run_slug_lookup_raises_for_missing_slug():
    with pytest.raises(
        KeyError,
        match="Unknown ForecastBench LLM model run slug: unknown-model",
    ):
        model_runs.get_model_run_by_slug("unknown-model")


def test_providers_for_model_runs_includes_reformat_model():
    anthropic_run = model_runs.get_model_run_by_slug("claude-opus-4-7-1024")

    providers = model_runs.providers_for_model_runs([anthropic_run])

    assert PROVIDERS["Anthropic"] in providers
    assert model_runs.REFORMAT_MODEL.provider in providers


def test_configure_and_validate_provider_keys_fetches_only_required_provider_secrets():
    selected = (model_runs.get_model_run_by_slug("claude-opus-4-7-1024"),)

    def fake_get_secret(secret_name):
        return f"secret-{secret_name}"

    with (
        patch("utils.gcp.get_secret", side_effect=fake_get_secret) as get_secret,
        patch("utils.llm.model_registry.configure_api_keys") as configure,
        patch("utils.llm.model_registry.validate_provider_keys") as validate,
    ):
        model_runs.configure_and_validate_provider_keys(selected)

    assert get_secret.call_args_list == [
        (("API_KEY_ANTHROPIC",),),
        (("API_KEY_OPENAI",),),
    ]
    configure.assert_called_once_with(
        anthropic="secret-API_KEY_ANTHROPIC",
        openai="secret-API_KEY_OPENAI",
    )
    providers = validate.call_args.args[0]
    assert selected[0].provider in providers
    assert model_runs.REFORMAT_MODEL.provider in providers


def test_reformat_model_uses_gpt_5_mini_with_parsing_args():
    assert model_runs.REFORMAT_MODEL == shared_model_runs.get_model_run(
        "gpt-5-mini-2025-08-07-run-variant-02"
    )
    assert model_runs.REFORMAT_MODEL.slug == "gpt-5-mini-2025-08-07-1024"
    assert model_runs.REFORMAT_MODEL.provider_model_id == "gpt-5-mini-2025-08-07"
    assert (
        model_runs.REFORMAT_MODEL.lab
        == shared_model_registry.MODELS_BY_KEY["gpt-5-mini-2025-08-07"].lab
    )
    assert model_runs.REFORMAT_MODEL.provider == PROVIDERS["OpenAI"]
    assert model_runs.REFORMAT_MODEL.options == {
        "max_output_tokens": 1024,
    }


def test_provider_max_workers_matches_plan_exactly():
    assert model_runs.PROVIDER_MAX_WORKERS == {
        PROVIDERS["OpenAI"]: 16,
        PROVIDERS["Anthropic"]: 4,
        PROVIDERS["Google"]: 8,
        PROVIDERS["xAI"]: 8,
        PROVIDERS["Together"]: 8,
    }


def test_llm_forecaster_files_do_not_use_future_annotations():
    src_root = Path(__file__).resolve().parents[2]
    forbidden_import = "from __future__ import " + "annotations"
    paths = [
        *sorted((src_root / "llm_forecaster").glob("**/*.py")),
        *sorted((src_root / "tests" / "llm_forecaster").glob("**/*.py")),
    ]

    assert paths
    offenders = [path for path in paths if forbidden_import in path.read_text()]
    assert offenders == []

"""Tests for metadata model request routing."""

import importlib
import sys
import types
from pathlib import Path

from utils.llm.provider_registry import PROVIDERS

from helpers import constants

ROOT = Path(__file__).resolve().parents[2]
LEGACY_MODEL_RUNS_MAP = "_".join(("MODELS", "TO", "RUN"))
LEGACY_MODEL_RUNS_BY_SOURCE_MAP = "_".join(("MODELS", "TO", "RUN", "BY", "SOURCE"))
LEGACY_CONSTANTS_MODEL_RUNS = ".".join(("constants", LEGACY_MODEL_RUNS_MAP))


def import_model_eval_without_secret_fetch(monkeypatch):
    """Import model_eval with a stable metadata model name."""
    sys.modules.pop("helpers.model_eval", None)

    fake_question_curation = types.ModuleType("helpers.question_curation")
    fake_question_curation.METADATA_MODEL_NAME = "gpt-5-mini"
    monkeypatch.setitem(sys.modules, "helpers.question_curation", fake_question_curation)
    monkeypatch.setattr(
        importlib.import_module("helpers"),
        "question_curation",
        fake_question_curation,
        raising=False,
    )

    return importlib.import_module("helpers.model_eval")


def test_constants_do_not_expose_legacy_llm_model_maps():
    legacy_constants = {
        LEGACY_MODEL_RUNS_MAP,
        LEGACY_MODEL_RUNS_BY_SOURCE_MAP,
        "MODEL_NAME_TO_SOURCE",
        "MODEL_TOKEN_LIMITS",
    }

    for name in legacy_constants:
        assert not hasattr(constants, name), name


def test_metadata_callers_use_metadata_model_response_helper():
    metadata_files = [
        ROOT / "src" / "metadata" / "tag_questions" / "main.py",
        ROOT / "src" / "metadata" / "validate_questions" / "main.py",
    ]

    for path in metadata_files:
        source = path.read_text()
        assert "model_eval.get_metadata_model_response" in source, path
        assert "model_eval.get_response_from_model" not in source, path


def test_model_eval_no_longer_contains_legacy_provider_routing():
    source = (ROOT / "src" / "helpers" / "model_eval.py").read_text()
    legacy_fragments = [
        "get_response_from_model",
        "infer_model_source",
        "get_model_org",
        "get_response_from_oai_model",
        "get_response_from_anthropic_model",
        "get_response_from_together_ai_model",
        "get_response_from_google_model",
        "get_response_from_xai_model",
        LEGACY_CONSTANTS_MODEL_RUNS,
    ]

    for fragment in legacy_fragments:
        assert fragment not in source, fragment
    assert "question_curation.METADATA_MODEL_NAME" in source


def test_metadata_model_response_routes_to_shared_openai_provider(monkeypatch):
    model_eval = import_model_eval_without_secret_fetch(monkeypatch)
    calls = {}

    monkeypatch.setattr(model_eval, "configure_api_keys", lambda **kwargs: None, raising=False)
    monkeypatch.setattr(model_eval, "validate_provider_keys", lambda providers: None, raising=False)

    def fake_get_response(provider, model_id, prompt, options=None):
        calls["provider"] = provider
        calls["model_id"] = model_id
        calls["prompt"] = prompt
        calls["options"] = options
        return "metadata response"

    monkeypatch.setattr(model_eval, "get_response", fake_get_response)

    response = model_eval.get_metadata_model_response(
        prompt="Classify this question.",
        max_output_tokens=123,
    )

    assert response == "metadata response"
    assert calls == {
        "provider": PROVIDERS["OpenAI"],
        "model_id": "gpt-5-mini",
        "prompt": "Classify this question.",
        "options": {"max_output_tokens": 123},
    }


def test_metadata_model_response_configures_provider_keys_before_request(monkeypatch):
    model_eval = import_model_eval_without_secret_fetch(monkeypatch)
    calls = []

    def fake_configure_api_keys(**kwargs):
        calls.append(("configure", kwargs))

    def fake_validate_provider_keys(providers):
        calls.append(("validate", providers))

    def fake_get_response(provider, model_id, prompt, options=None):
        calls.append(("get_response", provider))
        return "metadata response"

    monkeypatch.setattr(model_eval, "configure_api_keys", fake_configure_api_keys, raising=False)
    monkeypatch.setattr(
        model_eval, "validate_provider_keys", fake_validate_provider_keys, raising=False
    )
    monkeypatch.setattr(model_eval, "get_response", fake_get_response)

    assert model_eval.get_metadata_model_response("Prompt", 50) == "metadata response"

    assert calls == [
        ("configure", {"from_gcp": True}),
        ("validate", [PROVIDERS["OpenAI"]]),
        ("get_response", PROVIDERS["OpenAI"]),
    ]

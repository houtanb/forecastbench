"""Tests for the forecastbench ModelRun registry (helpers.llm)."""

from unittest.mock import MagicMock, patch

import pytest


class TestModelRunRateLimitGroup:
    """ModelRun.rate_limit_group defaults to provider.value when not set."""

    def test_defaults_to_provider_value(self):
        from helpers.llm import ModelRun, Provider

        run = ModelRun(
            name="test",
            model_id="test-model",
            provider=Provider.OPENAI,
            org="TestOrg",
        )
        assert run.rate_limit_group == "openai"

    def test_explicit_override(self):
        from helpers.llm import ModelRun, Provider

        run = ModelRun(
            name="test",
            model_id="test-model",
            provider=Provider.ANTHROPIC,
            org="TestOrg",
            rate_limit_group="opus",
        )
        assert run.rate_limit_group == "opus"


class TestModelRunsIntegrity:
    """MODEL_RUNS list is valid."""

    def test_no_duplicate_names(self):
        from helpers.llm import MODEL_RUNS

        names = [m.name for m in MODEL_RUNS]
        assert len(names) == len(set(names)), f"Duplicate names: {[n for n in names if names.count(n) > 1]}"

    def test_all_providers_valid(self):
        from helpers.llm import MODEL_RUNS, Provider

        for run in MODEL_RUNS:
            assert isinstance(run.provider, Provider), f"{run.name} has invalid provider"

    def test_all_rate_limit_groups_in_rate_limits(self):
        from helpers.llm import MODEL_RUNS, RATE_LIMITS

        for run in MODEL_RUNS:
            assert run.rate_limit_group in RATE_LIMITS, (
                f"{run.name}: rate_limit_group '{run.rate_limit_group}' not in RATE_LIMITS"
            )


class TestModelRunGetResponse:
    """ModelRun.get_response() merges self.options with call-time kwargs."""

    @patch("helpers.llm.configure_keys")
    def test_merges_options_with_kwargs(self, mock_configure):
        from helpers.llm import ModelRun, Provider

        run = ModelRun(
            name="test",
            model_id="test-model",
            provider=Provider.OPENAI,
            org="TestOrg",
            options={"temperature": 0, "max_tokens": 100},
        )

        with patch.object(run, "_get_model") as mock_get_model:
            mock_model = MagicMock()
            mock_model.get_response.return_value = "response"
            mock_get_model.return_value = mock_model

            result = run.get_response("hello", max_tokens=200)

            # Call-time kwargs should override self.options
            call_kwargs = mock_model.get_response.call_args[1]
            assert call_kwargs["max_tokens"] == 200
            assert call_kwargs["temperature"] == 0


class TestConfigureKeys:
    """configure_keys() calls utils_llm.configure_api_keys()."""

    @patch("helpers.llm._utils_configure_api_keys")
    def test_calls_utils_configure(self, mock_utils_configure):
        from helpers.llm import configure_keys

        mock_keys = MagicMock()
        mock_keys.API_KEY_OPENAI = "sk-test"
        mock_keys.API_KEY_ANTHROPIC = "sk-ant-test"
        mock_keys.API_KEY_GOOGLE = "goog-test"
        mock_keys.API_KEY_XAI = "xai-test"
        mock_keys.API_KEY_TOGETHERAI = "tog-test"

        with patch.dict("sys.modules", {"helpers.keys": mock_keys}):
            configure_keys()
        mock_utils_configure.assert_called_once()

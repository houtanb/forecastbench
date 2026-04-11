"""Tests for Model dataclass changes: rate_limit_group and reasoning_model removal."""

import pytest

from helpers.utils_llm.lab_registry import Lab
from helpers.utils_llm.model_registry import Model
from helpers.utils_llm.providers.base import BaseLLMProvider


class _DummyProvider(BaseLLMProvider):
    """Concrete provider for testing."""

    def _call_model(self, model, prompt, **options):
        return "ok"


class TestRateLimitGroupDefaults:
    """rate_limit_group defaults to provider name when not set."""

    def test_defaults_to_openai(self):
        from helpers.utils_llm.providers.openai import OpenAIProvider

        model = Model(
            id="test",
            full_name="test",
            token_limit=4096,
            provider_cls=OpenAIProvider,
            lab=Lab(name="Test"),
        )
        assert model.rate_limit_group == "openai"

    def test_defaults_to_anthropic(self):
        from helpers.utils_llm.providers.anthropic import AnthropicProvider

        model = Model(
            id="test",
            full_name="test",
            token_limit=4096,
            provider_cls=AnthropicProvider,
            lab=Lab(name="Test"),
        )
        assert model.rate_limit_group == "anthropic"

    def test_explicit_override(self):
        from helpers.utils_llm.providers.anthropic import AnthropicProvider

        model = Model(
            id="test",
            full_name="test",
            token_limit=4096,
            provider_cls=AnthropicProvider,
            lab=Lab(name="Test"),
            rate_limit_group="opus",
        )
        assert model.rate_limit_group == "opus"


class TestReasoningModelRemoved:
    """reasoning_model field no longer exists on Model."""

    def test_no_reasoning_model_field(self):
        from helpers.utils_llm.providers.openai import OpenAIProvider

        with pytest.raises(TypeError):
            Model(
                id="test",
                full_name="test",
                token_limit=4096,
                provider_cls=OpenAIProvider,
                lab=Lab(name="Test"),
                reasoning_model=True,
            )

    def test_no_reasoning_model_attribute(self):
        from helpers.utils_llm.providers.openai import OpenAIProvider

        model = Model(
            id="test",
            full_name="test",
            token_limit=4096,
            provider_cls=OpenAIProvider,
            lab=Lab(name="Test"),
        )
        assert not hasattr(model, "reasoning_model")

"""Tests for kwargs passthrough in utils_llm providers."""

from unittest.mock import MagicMock, patch

import pytest

from helpers.utils_llm.model_registry import Model
from helpers.utils_llm.lab_registry import LABS, Lab
from helpers.utils_llm.providers.openai import OpenAIProvider
from helpers.utils_llm.providers.anthropic import AnthropicProvider
from helpers.utils_llm.providers.google import GoogleProvider
from helpers.utils_llm.providers.together import TogetherProvider
from helpers.utils_llm.providers.xai import XAIProvider


def _make_model(provider_cls):
    """Create a minimal Model for testing."""
    return Model(
        id="test-model",
        full_name="test-model",
        token_limit=4096,
        provider_cls=provider_cls,
        lab=Lab(name="TestLab"),
    )


class TestOpenAIPassthrough:
    """OpenAI provider passes unknown kwargs to the SDK."""

    @patch("helpers.utils_llm.providers.openai.OpenAI")
    def test_reasoning_effort_passed_through(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.status = "completed"
        mock_response.output_text = "test response"
        mock_client.responses.create.return_value = mock_response

        provider = OpenAIProvider(api_key="test-key")
        model = _make_model(OpenAIProvider)
        provider._call_model(model, "hello", reasoning_effort="high", max_tokens=100)

        call_kwargs = mock_client.responses.create.call_args[1]
        assert call_kwargs["reasoning_effort"] == "high"

    @patch("helpers.utils_llm.providers.openai.OpenAI")
    def test_temperature_excluded_when_reasoning_effort_present(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.status = "completed"
        mock_response.output_text = "test response"
        mock_client.responses.create.return_value = mock_response

        provider = OpenAIProvider(api_key="test-key")
        model = _make_model(OpenAIProvider)
        provider._call_model(
            model, "hello", temperature=0.5, reasoning_effort="medium", max_tokens=100
        )

        call_kwargs = mock_client.responses.create.call_args[1]
        assert "temperature" not in call_kwargs
        assert call_kwargs["reasoning_effort"] == "medium"

    @patch("helpers.utils_llm.providers.openai.OpenAI")
    def test_max_tokens_mapped_to_max_output_tokens(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.status = "completed"
        mock_response.output_text = "test response"
        mock_client.responses.create.return_value = mock_response

        provider = OpenAIProvider(api_key="test-key")
        model = _make_model(OpenAIProvider)
        provider._call_model(model, "hello", max_tokens=500)

        call_kwargs = mock_client.responses.create.call_args[1]
        assert call_kwargs["max_output_tokens"] == 500
        assert "max_tokens" not in call_kwargs


class TestAnthropicPassthrough:
    """Anthropic provider passes unknown kwargs to the SDK."""

    @patch("helpers.utils_llm.providers.anthropic.anthropic")
    def test_thinking_dict_passed_through(self, mock_anthropic_mod):
        mock_client = MagicMock()
        mock_anthropic_mod.Anthropic.return_value = mock_client
        mock_stream = MagicMock()
        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(text="response")]
        mock_stream.get_final_message.return_value = mock_msg
        mock_client.messages.stream.return_value.__enter__ = MagicMock(return_value=mock_stream)
        mock_client.messages.stream.return_value.__exit__ = MagicMock(return_value=False)

        provider = AnthropicProvider(api_key="test-key")
        model = _make_model(AnthropicProvider)
        thinking_config = {"type": "enabled", "budget_tokens": 10000}
        provider._call_model(model, "hello", max_tokens=1000, thinking=thinking_config)

        call_kwargs = mock_client.messages.stream.call_args[1]
        assert call_kwargs["thinking"] == thinking_config


class TestGooglePassthrough:
    """Google provider passes extra kwargs into GenerateContentConfig."""

    @patch("helpers.utils_llm.providers.google.types")
    @patch("helpers.utils_llm.providers.google.genai")
    def test_thinking_config_passed_through(self, mock_genai, mock_types):
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_response = MagicMock()
        mock_response.text = "response"
        mock_client.models.generate_content.return_value = mock_response
        mock_types.AutomaticFunctionCallingConfig.return_value = "auto_fc_config"
        mock_types.GenerateContentConfig.return_value = "config_obj"

        provider = GoogleProvider(api_key="test-key")
        model = _make_model(GoogleProvider)
        provider._call_model(model, "hello", temperature=0.5, thinking_config="some_config")

        config_call_kwargs = mock_types.GenerateContentConfig.call_args[1]
        assert config_call_kwargs["thinking_config"] == "some_config"


class TestTogetherPassthrough:
    """Together provider passes extra kwargs to chat.completions.create()."""

    @patch("helpers.utils_llm.providers.together.Together")
    def test_top_p_passed_through(self, mock_together_cls):
        mock_client = MagicMock()
        mock_together_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "response"
        mock_client.chat.completions.create.return_value = mock_response

        provider = TogetherProvider(api_key="test-key")
        model = _make_model(TogetherProvider)
        provider._call_model(model, "hello", temperature=0.5, top_p=0.9)

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["top_p"] == 0.9


class TestXAIPassthrough:
    """xAI provider passes extra kwargs to chat.completions.create()."""

    @patch("helpers.utils_llm.providers.xai.openai")
    def test_tools_list_passed_through(self, mock_openai_mod):
        mock_client = MagicMock()
        mock_openai_mod.OpenAI.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "response"
        mock_client.chat.completions.create.return_value = mock_response

        provider = XAIProvider(api_key="test-key")
        model = _make_model(XAIProvider)
        tools = [{"type": "function", "function": {"name": "test"}}]
        provider._call_model(model, "hello", max_tokens=100, tools=tools)

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["tools"] == tools

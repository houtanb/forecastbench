"""LLM-related utilities."""

from functools import cache

from utils.llm.model_registry import (
    configure_api_keys,
    get_response,
    validate_provider_keys,
)
from utils.llm.provider_registry import PROVIDERS

_METADATA_PROVIDER = PROVIDERS["OpenAI"]


@cache
def _configure_metadata_provider_keys() -> None:
    """Configure shared LLM provider keys for metadata requests."""
    configure_api_keys(from_gcp=True)
    validate_provider_keys([_METADATA_PROVIDER])


def get_metadata_model_response(prompt: str, max_output_tokens: int) -> str:
    """Get a response from the shared metadata model.

    Args:
      prompt (str): Prompt to send to the metadata model.
      max_output_tokens (int): Maximum number of output tokens to request.
    """
    from . import question_curation

    _configure_metadata_provider_keys()
    return get_response(
        provider=_METADATA_PROVIDER,
        model_id=question_curation.METADATA_MODEL_NAME,
        prompt=prompt,
        options={"max_output_tokens": max_output_tokens},
    )

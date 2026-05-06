import hashlib

from llm_forecaster import prompts

PROMPT_DIGESTS = {
    "ZERO_SHOT_MARKET_PROMPT": ("3e9773f317df0975010da625bbce26c20089df55d6066be43c2a539423db499f"),
    "ZERO_SHOT_MARKET_WITH_FREEZE_VALUE_PROMPT": (
        "d75d7e41749303a6f87bbac272b26393df905c7f839ea52a245b5c4a0bc0e188"
    ),
    "ZERO_SHOT_DATASET_PROMPT": (
        "5cc8a4d4582a352bd6b6419a9ecdfac312d6c5850340af4edeba068de3b76d8c"
    ),
    "REFORMAT_PROMPT": ("de9d3027166528e860e68a50da0c318ad5c0a2ee7ac294dc344d413b4cd7557c"),
    "REFORMAT_PROMPT_2": ("8f4fa40f53ac45598e1320c808e677681957b1c6d90bbdb15b85bf680846b9d4"),
    "REFORMAT_SINGLE_PROMPT": ("0b334d0c54612a81d37b996ed5ae3749a3a5ae6943481bf980c6ef9322978edb"),
    "REFORMAT_SINGLE_PROMPT_2": (
        "69b2ee134b9a4e4f77fbf3bdff9aee8c0636dc0d86decaf6f0cd6cf8401407bf"
    ),
}


def test_forecasting_prompt_text_matches_snapshots():
    for prompt_name, expected_digest in PROMPT_DIGESTS.items():
        prompt_text = getattr(prompts, prompt_name)

        assert hashlib.sha256(prompt_text.encode()).hexdigest() == expected_digest


def test_prompt_module_exports_expected_forecasting_prompts():
    prompt_names = {
        name for name, value in vars(prompts).items() if name.isupper() and isinstance(value, str)
    }

    assert prompt_names == set(PROMPT_DIGESTS)

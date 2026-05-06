import pytest

from helpers import constants


def test_run_mode_constructs_case_insensitively():
    assert constants.RunMode("test") is constants.RunMode.TEST
    assert constants.RunMode("Prod") is constants.RunMode.PROD


def test_run_mode_from_string_returns_mode_case_insensitively():
    assert constants.RunMode.from_string("TEST") is constants.RunMode.TEST
    assert constants.RunMode.from_string("TeSt") is constants.RunMode.TEST
    assert constants.RunMode.from_string("prod") is constants.RunMode.PROD
    assert constants.RunMode.from_string("PrOd") is constants.RunMode.PROD


def test_run_mode_from_string_defaults_to_test_for_missing_or_invalid_value():
    assert constants.RunMode.from_string(None) is constants.RunMode.TEST
    assert constants.RunMode.from_string("DEV") is constants.RunMode.TEST


def test_run_mode_constructor_raises_for_invalid_value():
    with pytest.raises(ValueError):
        constants.RunMode("DEV")


def test_run_mode_predicates():
    assert constants.RunMode.TEST.is_test is True
    assert constants.RunMode.TEST.is_prod is False
    assert constants.RunMode.PROD.is_test is False
    assert constants.RunMode.PROD.is_prod is True


def test_run_mode_forecast_file_prefix():
    assert constants.RunMode.TEST.forecast_file_prefix == "TEST."
    assert constants.RunMode.PROD.forecast_file_prefix == ""


def test_shared_llm_lab_display_names_have_logos():
    assert constants.get_org_logo("MiniMax") == "minimax.svg"
    assert constants.get_org_logo("Moonshot AI") == "moonshot.svg"

"""Tests for forecast validation module."""

import json
import os
import tempfile

import pytest


def _write_forecast_file(path, forecasts, extra_keys=None):
    """Write a forecast JSON file for testing."""
    data = {
        "organization": "ForecastBench",
        "model": "test-model (zero shot)",
        "model_organization": "TestOrg",
        "question_set": "2025-01-01-llm.json",
        "forecast_due_date": "2025-01-01",
        "forecasts": forecasts,
    }
    if extra_keys:
        data.update(extra_keys)
    with open(path, "w") as f:
        json.dump(data, f)


class TestValidForecastFile:
    """Valid file with all fields passes validation."""

    def test_all_valid(self, tmp_path):
        from helpers.forecast_validation import validate_forecast_file

        forecasts = [
            {"id": "q1", "source": "metaculus", "forecast": 0.75},
            {"id": "q2", "source": "metaculus", "forecast": 0.5},
            {"id": "q3", "source": "fred", "forecast": 0.3},
            {"id": "q4", "source": "fred", "forecast": 0.8},
        ]
        path = str(tmp_path / "forecast.json")
        _write_forecast_file(path, forecasts)

        result = validate_forecast_file(path, n_market=2, n_dataset=2)
        assert result.valid_json is True
        assert result.valid_probabilities is True
        assert result.market_forecasted == 2
        assert result.market_total == 2
        assert result.dataset_forecasted == 2
        assert result.dataset_total == 2


class TestInvalidProbabilities:
    """Probability outside (0, 1) fails validation."""

    def test_probability_out_of_range(self, tmp_path):
        from helpers.forecast_validation import validate_forecast_file

        forecasts = [
            {"id": "q1", "source": "metaculus", "forecast": 1.5},
            {"id": "q2", "source": "fred", "forecast": 0.5},
        ]
        path = str(tmp_path / "forecast.json")
        _write_forecast_file(path, forecasts)

        result = validate_forecast_file(path, n_market=1, n_dataset=1)
        assert result.valid_probabilities is False


class TestNullForecasts:
    """Null forecasts counted correctly."""

    def test_null_forecasts_counted(self, tmp_path):
        from helpers.forecast_validation import validate_forecast_file

        forecasts = [
            {"id": "q1", "source": "metaculus", "forecast": 0.5},
            {"id": "q2", "source": "metaculus", "forecast": None},
            {"id": "q3", "source": "fred", "forecast": 0.3},
            {"id": "q4", "source": "fred", "forecast": None},
            {"id": "q5", "source": "fred", "forecast": None},
        ]
        path = str(tmp_path / "forecast.json")
        _write_forecast_file(path, forecasts)

        result = validate_forecast_file(path, n_market=2, n_dataset=3)
        assert result.market_forecasted == 1
        assert result.market_total == 2
        assert result.dataset_forecasted == 1
        assert result.dataset_total == 3
        assert result.market_pct == pytest.approx(50.0)
        assert result.dataset_pct == pytest.approx(100.0 / 3.0)


class TestSummaryFormat:
    """Summary string format."""

    def test_summary_contains_emoji_pass(self, tmp_path):
        from helpers.forecast_validation import validate_forecast_file

        forecasts = [
            {"id": "q1", "source": "metaculus", "forecast": 0.75},
            {"id": "q2", "source": "fred", "forecast": 0.3},
        ]
        path = str(tmp_path / "forecast.json")
        _write_forecast_file(path, forecasts)

        result = validate_forecast_file(path, n_market=1, n_dataset=1)
        summary = result.format_summary("test-model", "zero_shot")
        assert "\U0001f4ca" in summary  # chart emoji
        assert "\u2705" in summary  # check mark

    def test_summary_contains_fail_emoji(self, tmp_path):
        from helpers.forecast_validation import validate_forecast_file

        # All null forecasts = 0% < 95%
        forecasts = [
            {"id": "q1", "source": "metaculus", "forecast": None},
            {"id": "q2", "source": "fred", "forecast": None},
        ]
        path = str(tmp_path / "forecast.json")
        _write_forecast_file(path, forecasts)

        result = validate_forecast_file(path, n_market=1, n_dataset=1)
        summary = result.format_summary("test-model", "zero_shot")
        assert "\u274c" in summary  # cross mark


class TestMissingJsonKeys:
    """Missing required JSON keys fails validation."""

    def test_missing_forecasts_key(self, tmp_path):
        from helpers.forecast_validation import validate_forecast_file

        path = str(tmp_path / "bad.json")
        with open(path, "w") as f:
            json.dump({"organization": "test"}, f)

        result = validate_forecast_file(path, n_market=1, n_dataset=1)
        assert result.valid_json is False

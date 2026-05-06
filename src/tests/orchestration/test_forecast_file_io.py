"""Tests for processed forecast file IO."""

from orchestration import _io


def test_valid_forecast_files_excludes_nested_test_files(monkeypatch):
    """Do not include date-folder test forecast files in leaderboard inputs."""
    monkeypatch.setattr(
        _io.gcp.storage,
        "list",
        lambda bucket_name, mnt: [
            "2026-05-24/2026-05-24.ForecastBench.model.json",
            "2026-05-24/TEST.2026-05-24.ForecastBench.model.json",
            "TEST.2026-05-24.ForecastBench.root-model.json",
            "2026-05-24/notes.txt",
            "2026-05-25/2026-05-25.ForecastBench.model.json",
        ],
    )

    files, dates = _io.get_valid_forecast_files_and_dates("forecastbench-processed-forecast-sets")

    assert files == [
        "2026-05-24/2026-05-24.ForecastBench.model.json",
        "2026-05-25/2026-05-25.ForecastBench.model.json",
    ]
    assert dates == ["2026-05-24", "2026-05-25"]

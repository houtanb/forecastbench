"""ForecastBench LLM forecast output helpers."""

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

from helpers.constants import BENCHMARK_NAME, RunMode
from llm_forecaster.forecast_variants import (
    FORECAST_VARIANTS,
    ZERO_SHOT,
    ZERO_SHOT_WITH_FREEZE_VALUES,
    ForecastVariant,
)
from llm_forecaster.model_runs import MODEL_RUNS, ModelRun


def display_model_name(model_run: ModelRun, variant: ForecastVariant) -> str:
    """Return the displayed model name for a model run and forecast variant."""
    if variant == ZERO_SHOT:
        return model_run.slug
    if variant == ZERO_SHOT_WITH_FREEZE_VALUES:
        return f"{model_run.slug}†"
    return f"{model_run.slug}-{variant.key}"


def forecast_file_model_name(model_run: ModelRun, variant: ForecastVariant) -> str:
    """Return the model name stored in forecast files and filenames."""
    if variant == ZERO_SHOT:
        return model_run.slug
    return f"{model_run.slug}-{variant.key}"


@lru_cache
def _display_model_name_map() -> dict[str, tuple[ModelRun, ForecastVariant]]:
    """Return active display model names keyed to model runs and variants."""
    return {
        display_model_name(model_run, variant): (model_run, variant)
        for model_run in MODEL_RUNS
        for variant in FORECAST_VARIANTS
    }


def parse_display_model_name(model_name: str) -> tuple[ModelRun, ForecastVariant]:
    """Return the active model run and forecast variant for a displayed model name."""
    try:
        return _display_model_name_map()[model_name]
    except KeyError as exc:
        raise KeyError(f"Unknown ForecastBench LLM display model name: {model_name}") from exc


def final_filename(
    forecast_due_date: str,
    model_run: ModelRun,
    variant: ForecastVariant,
    is_test: bool,
) -> str:
    """Return the final forecast filename."""
    filename = (
        f"{forecast_due_date}.{BENCHMARK_NAME}."
        f"{forecast_file_model_name(model_run, variant)}.json"
    )
    if is_test:
        return f"{RunMode.TEST.forecast_file_prefix}{filename}"
    return filename


def destination_blob_name(
    forecast_due_date: str,
    model_run: ModelRun,
    variant: ForecastVariant,
    is_test: bool,
) -> str:
    """Return the destination blob name for a final forecast file."""
    return f"{forecast_due_date}/{final_filename(forecast_due_date, model_run, variant, is_test)}"


def write_forecast_file(
    local_filename: str | Path,
    forecast_due_date: str,
    question_set_filename: str,
    model_run: ModelRun,
    variant: ForecastVariant,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    """Write a ForecastBench LLM forecast file using the current identity schema."""
    uses_freeze_values = variant.uses_freeze_values
    forecast_file = {
        "organization": BENCHMARK_NAME,
        "model": forecast_file_model_name(model_run, variant),
        "model_organization": model_run.lab.name,
        "model_run_key": model_run.model_run_key,
        "model_run_slug": model_run.slug,
        "forecast_variant_key": variant.key,
        "uses_freeze_values": uses_freeze_values,
        "question_set": question_set_filename,
        "forecast_due_date": forecast_due_date,
        "forecasts": rows,
    }

    path = Path(local_filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(forecast_file, indent=4), encoding="utf-8")

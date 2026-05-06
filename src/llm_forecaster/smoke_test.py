"""Run a full-path smoke test for configured ForecastBench LLM model runs."""

import argparse
import logging
import os
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger(__name__)

PASS = "PASS"
FAIL = "FAIL"
SMOKE_OUTPUT_DIR = "/tmp/forecasts/llm_smoke_test"


@dataclass(frozen=True)
class SmokeResult:
    """Result for one model smoke check."""

    model_name: str
    lab: str
    provider: str
    status: str
    error_type: str
    error_message: str


@dataclass(frozen=True)
class SmokeRun:
    """Smoke-test results and local forecast rows."""

    results: list[SmokeResult]
    forecast_file_paths: list[str]


def select_questions(questions: Sequence[dict], sample_size: int) -> list[dict]:
    """Return deterministic dataset and market prefixes for smoke coverage."""
    if sample_size < 1:
        raise ValueError("sample_size must be at least 1.")

    from sources import DATASET_SOURCE_NAMES, MARKET_SOURCE_NAMES

    dataset_source_names = set(DATASET_SOURCE_NAMES)
    market_source_names = set(MARKET_SOURCE_NAMES)
    dataset_questions = []
    market_questions = []
    unknown_sources = set()

    for question in questions:
        source = question.get("source")
        if source in dataset_source_names:
            dataset_questions.append(question)
        elif source in market_source_names:
            market_questions.append(question)
        else:
            unknown_sources.add(source)

    if unknown_sources:
        sources = ", ".join(sorted(str(source) for source in unknown_sources))
        raise ValueError(f"Unknown question sources: {sources}")

    selected_questions = (
        sorted(dataset_questions, key=lambda question: question["id"])[:sample_size]
        + sorted(market_questions, key=lambda question: question["id"])[:sample_size]
    )
    if not selected_questions:
        raise ValueError("No questions selected for smoke test.")
    return selected_questions


def select_model_runs(
    model_runs: Sequence[Any],
    model_run_slugs: Sequence[str] | None,
) -> list[Any]:
    """Return all model runs or the exact-slug model runs requested."""
    if model_run_slugs is None:
        return list(model_runs)

    requested_model_slugs = set(model_run_slugs)
    selected_model_runs = [
        model_run for model_run in model_runs if model_run.slug in requested_model_slugs
    ]
    selected_model_slugs = {model_run.slug for model_run in selected_model_runs}
    missing_model_slugs = sorted(requested_model_slugs - selected_model_slugs)
    if not missing_model_slugs:
        return selected_model_runs

    available_slugs = ", ".join(model_run.slug for model_run in model_runs)
    raise ValueError(
        f"Unknown model run slug(s) {missing_model_slugs}. Available model runs: "
        f"{available_slugs}"
    )


def exit_code_for_results(results: Sequence[SmokeResult]) -> int:
    """Return a process exit code for smoke-test results."""
    if not results:
        return 1
    return 1 if any(result.status == FAIL for result in results) else 0


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the smoke test."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--forecast-due-date",
        default=os.getenv("FORECAST_DUE_DATE"),
        help="Forecast due date to smoke test. Defaults to FORECAST_DUE_DATE or latest LLM set.",
    )
    parser.add_argument(
        "--sample-size",
        default=2,
        type=int,
        help="Number of deterministic questions to test.",
    )
    parser.add_argument(
        "--model-run",
        nargs="+",
        default=None,
        help="One or more exact ModelRun.slug values to test. Defaults to all configured runs.",
    )
    parser.add_argument(
        "--run-locally",
        action="store_true",
        help="Read question-set metadata and questions from local files.",
    )
    return parser.parse_args()


def _get_io_module() -> Any:
    """Return orchestration IO module without importing it at smoke_test import time."""
    from orchestration import _io

    return _io


def _get_model_runs_module() -> Any:
    """Return model-run declarations without importing them at smoke_test import time."""
    from llm_forecaster import model_runs

    return model_runs


def _get_questions_module() -> Any:
    """Return question helpers without importing them at smoke_test import time."""
    from llm_forecaster import questions

    return questions


def _get_runner() -> Any:
    """Return runner module without importing it at smoke_test import time."""
    from llm_forecaster import runner

    return runner


def _new_output_dir() -> Path:
    """Return a unique smoke output directory for one CLI invocation."""
    return Path(SMOKE_OUTPUT_DIR) / uuid.uuid4().hex


def _result_for(
    model_run: Any, status: str, error_type: str = "", error_message: str = ""
) -> SmokeResult:
    """Build a normalized smoke-test result row."""
    return SmokeResult(
        model_name=model_run.slug,
        lab=model_run.lab.name,
        provider=model_run.provider.name,
        status=status,
        error_type=error_type,
        error_message=error_message,
    )


def run_smoke_test(
    model_runs: Sequence[Any],
    context: Any,
    output_dir: str | Path = SMOKE_OUTPUT_DIR,
) -> SmokeRun:
    """Run every selected model through the ForecastBench LLM runner path."""
    results = []
    forecast_file_paths = []
    runner = _get_runner()

    for model_run in model_runs:
        try:
            written_files = runner.run_model(
                model_run=model_run,
                context=context,
                output_dir=output_dir,
                upload=False,
                is_test=True,
                raise_on_question_error=True,
            )
            forecast_file_paths.extend(
                str(written_file.local_filename) for written_file in written_files
            )
            row_count = sum(len(written_file.rows) for written_file in written_files)
            if row_count:
                results.append(_result_for(model_run=model_run, status=PASS))
            else:
                results.append(
                    _result_for(
                        model_run=model_run,
                        status=FAIL,
                        error_type="EmptyForecast",
                        error_message="No forecast rows returned by runner.",
                    )
                )
        except Exception as exc:
            results.append(
                _result_for(
                    model_run=model_run,
                    status=FAIL,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
            )

    return SmokeRun(results=results, forecast_file_paths=forecast_file_paths)


def _sample_context(context: Any, sample_size: int) -> Any:
    """Return a question-set context containing only sampled smoke questions."""
    questions = _get_questions_module()
    return questions.QuestionSetContext(
        forecast_due_date=context.forecast_due_date,
        question_set_filename=context.question_set_filename,
        questions=select_questions(context.questions, sample_size=sample_size),
    )


def _log_results(smoke_run: SmokeRun) -> None:
    """Log compact smoke-test result rows and output paths."""
    for result in smoke_run.results:
        logger.info(
            "model=%s lab=%s provider=%s status=%s error_type=%s error=%s",
            result.model_name,
            result.lab,
            result.provider,
            result.status,
            result.error_type,
            result.error_message,
        )

    for forecast_file_path in smoke_run.forecast_file_paths:
        logger.info("forecast_file=%s", forecast_file_path)


def main() -> None:
    """Load questions, validate providers, run smoke checks, and exit."""
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    forecast_due_date = args.forecast_due_date
    if forecast_due_date is None:
        _io = _get_io_module()
        metadata = _io.get_latest_llm_question_set_metadata(run_locally=args.run_locally)
        forecast_due_date = metadata["forecast_due_date"]

    questions = _get_questions_module()
    question_context = questions.load_question_set_context(
        forecast_due_date=forecast_due_date,
        run_locally=args.run_locally,
    )
    sampled_context = _sample_context(question_context, sample_size=args.sample_size)
    model_runs_module = _get_model_runs_module()
    selected_model_runs = select_model_runs(
        model_runs=model_runs_module.MODEL_RUNS,
        model_run_slugs=args.model_run,
    )

    model_runs_module.configure_and_validate_provider_keys(selected_model_runs)
    smoke_run = run_smoke_test(
        model_runs=selected_model_runs,
        context=sampled_context,
        output_dir=_new_output_dir(),
    )
    _log_results(smoke_run)
    sys.exit(exit_code_for_results(smoke_run.results))


if __name__ == "__main__":
    main()

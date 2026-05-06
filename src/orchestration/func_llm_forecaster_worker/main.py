"""Cloud Run worker for ForecastBench LLM forecasts."""

import logging
import os

from helpers import constants, decorator
from llm_forecaster import model_runs, questions, runner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_TEST_QUESTIONS_PER_TYPE = 2
LOCAL_OUTPUT_DIR = "/tmp/forecasts/llm_forecaster"


def parse_env_vars() -> tuple[str, constants.RunMode, model_runs.ModelRun]:
    """Parse Cloud Run worker environment variables."""
    forecast_due_date = os.getenv("FORECAST_DUE_DATE")
    if not forecast_due_date:
        raise ValueError("FORECAST_DUE_DATE must be set.")

    task_num = int(os.getenv("CLOUD_RUN_TASK_INDEX"))
    assert task_num is not None and task_num >= 0
    model_run = model_runs.MODEL_RUNS[task_num]

    for index, available_model_run in enumerate(model_runs.MODEL_RUNS):
        marker = "⬅️️ running 🌟️" if index == task_num else ""
        logger.info("%s: %s %s", index, available_model_run, marker)

    run_mode = constants.RunMode.from_string(os.getenv("TEST_OR_PROD"))

    return forecast_due_date, run_mode, model_run


def _limit_context_for_test_mode(
    context: questions.QuestionSetContext,
) -> questions.QuestionSetContext:
    dataset_questions, market_questions = questions.split_questions(context.questions)
    dataset_questions, market_questions = questions.limit_questions_for_test_mode(
        dataset_questions,
        market_questions,
        DEFAULT_TEST_QUESTIONS_PER_TYPE,
    )
    return questions.QuestionSetContext(
        forecast_due_date=context.forecast_due_date,
        question_set_filename=context.question_set_filename,
        questions=dataset_questions + market_questions,
    )


def run_worker(
    forecast_due_date: str,
    run_mode: constants.RunMode,
    model_run: model_runs.ModelRun,
) -> None:
    """Run one ForecastBench LLM model task."""
    logger.info("Loading LLM question set for forecast due date %s.", forecast_due_date)
    context = questions.load_question_set_context(forecast_due_date)
    logger.info(
        "Loaded %s questions from %s.",
        len(context.questions),
        context.question_set_filename,
    )

    context_to_run = context
    if run_mode.is_test:
        logger.info(
            "Limiting TEST run to %s dataset questions and %s market questions.",
            DEFAULT_TEST_QUESTIONS_PER_TYPE,
            DEFAULT_TEST_QUESTIONS_PER_TYPE,
        )
        context_to_run = _limit_context_for_test_mode(context)
        logger.info("TEST run will forecast %s questions.", len(context_to_run.questions))

    provider_names = [
        provider.name for provider in model_runs.providers_for_model_runs([model_run])
    ]
    logger.info("Configuring LLM API keys for providers: %s.", ", ".join(provider_names))
    model_runs.configure_and_validate_provider_keys([model_run])
    logger.info("Starting LLM forecast runner.")
    runner.run_model(
        model_run=model_run,
        context=context_to_run,
        output_dir=LOCAL_OUTPUT_DIR,
        upload=True,
        is_test=run_mode.is_test,
    )


@decorator.log_runtime
def main() -> None:
    """Parse environment and run the selected LLM forecaster task."""
    forecast_due_date, run_mode, model_run = parse_env_vars()
    logger.info("Running %s LLM forecaster worker for %s.", run_mode.value, model_run.slug)
    run_worker(
        forecast_due_date=forecast_due_date,
        run_mode=run_mode,
        model_run=model_run,
    )


if __name__ == "__main__":
    main()

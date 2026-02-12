"""Run zero-shot and scratchpad evaluations for LLM models."""

import logging
import os
import sys

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
from helpers import cloud_run, constants, decorator, llm, question_sets  # noqa: E402


def parse_env_vars() -> constants.RunMode:
    """
    Parse and validate environment variables.

    Environment variables:
        TEST_OR_PROD (str): Required. Must be valid constants.RunMode
    A rgs:
        None

    Returns:
        result (tuple[str, constants.RunMode, str, str]):
            (forecast_due_date, run_mode, model_to_test, prompt_type).
    """
    try:
        return constants.RunMode(os.getenv("TEST_OR_PROD"))
    except ValueError:
        logger.error("`TEST_OR_PROD` must be one of TEST or PROD.")
        sys.exit(1)


@decorator.log_runtime
def main() -> None:
    """
    Run zero-shot and scratchpad evaluations for LLM models.

    Args:
        None

    Returns:
        None
    """
    run_mode = parse_env_vars()

    forecast_due_date = question_sets.get_field_from_latest_question_set_file("forecast_due_date")

    logger.info(f"Running {run_mode.value} run of LLM baselines for {forecast_due_date}-llm.json")

    timeout = cloud_run.timeout_1h * 24
    task_count = len(llm.MODEL_RUNS) * len(llm.PROMPT_TYPES)
    logger.info(f"Creating {task_count} workers...")
    operation = cloud_run.call_worker(
        job_name="func-baseline-llm-forecasts-worker",
        env_vars={
            "FORECAST_DUE_DATE": forecast_due_date,
            "TEST_OR_PROD": run_mode.value,
        },
        task_count=task_count,
        timeout=timeout,
    )
    cloud_run.block_and_check_job_result(
        operation=operation,
        name="llm-baselines",
        exit_on_error=True,
        timeout=timeout,
    )


if __name__ == "__main__":
    main()

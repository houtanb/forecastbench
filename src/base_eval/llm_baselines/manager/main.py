"""Run zero-shot evaluations for LLM models."""

import logging
import os
import sys
from collections import Counter

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
from helpers import cloud_run, constants, decorator, question_sets  # noqa: E402
from helpers.llm import MODEL_RUNS  # noqa: E402


def parse_env_vars() -> constants.RunMode:
    """Parse and validate environment variables.

    Returns:
        run_mode (constants.RunMode): The run mode (TEST or PROD).
    """
    try:
        return constants.RunMode(os.getenv("TEST_OR_PROD"))
    except ValueError:
        logger.error("`TEST_OR_PROD` must be one of TEST or PROD.")
        sys.exit(1)


@decorator.log_runtime
def main() -> None:
    """Run zero-shot evaluations for LLM models."""
    run_mode = parse_env_vars()

    forecast_due_date = question_sets.get_field_from_latest_question_set_file("forecast_due_date")

    logger.info(f"Running {run_mode.value} run of LLM baselines for {forecast_due_date}-llm.json")

    # Count siblings per rate_limit_group
    group_counts = Counter(m.rate_limit_group for m in MODEL_RUNS)

    timeout = cloud_run.timeout_1h * 24
    task_count = len(MODEL_RUNS)
    logger.info(f"Creating {task_count} workers...")

    for idx, model_run in enumerate(MODEL_RUNS):
        group_size = group_counts[model_run.rate_limit_group]
        logger.info(
            f"  [{idx}] {model_run.name} "
            f"(group={model_run.rate_limit_group}, siblings={group_size})"
        )

    operation = cloud_run.call_worker(
        job_name="func-baseline-llm-forecasts-worker",
        env_vars={
            "FORECAST_DUE_DATE": forecast_due_date,
            "TEST_OR_PROD": run_mode.value,
            "RATE_LIMIT_GROUP_SIZE": str(max(group_counts.values())),
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

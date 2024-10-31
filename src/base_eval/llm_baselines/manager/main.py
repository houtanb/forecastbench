"""Run zero-shot and scratchpad evaluations for LLM models."""

import argparse
import json
import logging
import os
import sys

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Add necessary paths
current_path = os.getcwd()
sys.path.append(os.path.join(current_path, "../.."))

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from helpers import cloud_run, decorator, env  # noqa: E402

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
from utils import gcp  # noqa: E402


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run LLM evaluations.")
    parser.add_argument(
        "mode",
        nargs="?",
        choices=["TEST", "PROD"],
        default="TEST",
        help="Run mode: TEST for specific models and 2 questions, PROD for all models and questions",
    )
    return parser.parse_args()


def load_questions_file():
    """Load the questions file."""
    QUESTIONS_FILE = "latest-llm.json"
    LOCAL_QUESTIONS_FILE = f"/tmp/{QUESTIONS_FILE}"

    logger.info(f"Downloading {QUESTIONS_FILE}...")
    gcp.storage.download_no_error_message_on_404(
        bucket_name=env.QUESTION_SETS_BUCKET,
        filename=QUESTIONS_FILE,
        local_filename=LOCAL_QUESTIONS_FILE,
    )

    with open(LOCAL_QUESTIONS_FILE, "r") as file:
        questions_data = json.load(file)
    os.remove(LOCAL_QUESTIONS_FILE)
    return questions_data["forecast_due_date"]


def call_worker(forecast_due_date, test_or_prod, task_count):
    """Make main() easier to read."""
    return cloud_run.run_job(
        job_name="func-baseline-llm-forecasts-worker",
        env_vars={
            "FORECAST_DUE_DATE": forecast_due_date,
            "TEST_OR_PROD": test_or_prod,
        },
        task_count=task_count,
    )


@decorator.log_runtime
def main():
    """Execute the main evaluation process."""
    args = parse_arguments()

    forecast_due_date = load_questions_file()

    logger.info(f"Running LLM baselines for: {forecast_due_date}-llm.json")

    operation = call_worker(
        forecast_due_date=forecast_due_date, test_or_prod=args.mode, task_count=9
    )
    cloud_run.block_and_check_job_result(
        operation=operation,
        name="llm-baselines",
        exit_on_error=True,
    )


if __name__ == "__main__":
    main()

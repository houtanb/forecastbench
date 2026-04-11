"""Run zero-shot evaluations for LLM models."""

import logging
import os
import shutil
import sys
import time

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
from helpers import (  # noqa: E402
    constants,
    data_utils,
    decorator,
    env,
    forecast_validation,
    git,
    keys,
    model_eval,
)
from helpers.llm import MODEL_RUNS, RATE_LIMITS, configure_keys  # noqa: E402

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../utils"))
from utils import gcp  # noqa: E402


def get_questions(forecast_due_date: str, num_questions_per_question_type: int | None) -> list:
    """Load and prepare questions for evaluation.

    Args:
        forecast_due_date (str): Forecast due date used to locate the questions file.
        num_questions_per_question_type (int | None): Limit per question type (None for all).

    Returns:
        questions (list): Ordered list of question sets: [dataset_questions, market_questions].
    """
    _, local_repo_dir, _ = git.clone(repo_url=keys.API_GITHUB_DATASET_REPO_URL)
    LOCAL_QUESTIONS_FILE = f"{local_repo_dir}/datasets/question_sets/{forecast_due_date}-llm.json"

    market_questions, dataset_questions = model_eval.process_questions(
        LOCAL_QUESTIONS_FILE,
        num_questions_per_question_type=num_questions_per_question_type,
    )
    questions = [
        dataset_questions,
        market_questions,
    ]

    shutil.rmtree(local_repo_dir)
    return questions


def upload_forecast_files(
    base_file_path: str,
    prompt_type: str,
    forecast_due_date: str,
    run_mode: constants.RunMode,
) -> None:
    """Upload local forecast artifacts to GCP.

    Args:
        base_file_path (str): GCS base path prefix for storing intermediate records.
        prompt_type (str): Prompt variant used ("zero_shot" or "scratchpad").
        forecast_due_date (str): Forecast due date used in remote path.
        run_mode (constants.RunMode): Execution mode controlling paths and behavior.
    """
    local_submit_dir = model_eval.get_local_final_submit_directory(
        prompt_type=prompt_type,
        run_mode=run_mode,
    )

    forecast_filenames = data_utils.list_files(local_submit_dir)
    for forecast_filename in forecast_filenames:
        local_filename = local_submit_dir + "/" + forecast_filename
        gcp.storage.upload(
            bucket_name=env.FORECAST_SETS_BUCKET,
            local_filename=local_filename,
            filename=f"{forecast_due_date}/{forecast_filename}",
        )


def parse_env_vars() -> tuple:
    """Parse environment variables and determine the model run to execute.

    Returns:
        tuple: (forecast_due_date, run_mode, model_run, prompt_type, max_workers)
    """
    forecast_due_date = os.getenv("FORECAST_DUE_DATE")
    if not forecast_due_date:
        logger.error(f"`forecast_due_date` was not set: {forecast_due_date}.")
        sys.exit(1)

    try:
        run_mode = constants.RunMode(os.getenv("TEST_OR_PROD", ""))
    except ValueError:
        logger.error("`TEST_OR_PROD` must be one of TEST or PROD.")
        sys.exit(1)

    # Determine the model run
    model_to_test = os.getenv("MODEL_TO_TEST")
    if model_to_test:
        # TEST mode: find model by name
        matching = [m for m in MODEL_RUNS if m.name == model_to_test]
        if not matching:
            logger.error(f"MODEL_TO_TEST '{model_to_test}' not found in MODEL_RUNS.")
            sys.exit(1)
        model_run = matching[0]
    else:
        # PROD mode: use task index
        try:
            task_num = int(os.getenv("CLOUD_RUN_TASK_INDEX"))
        except Exception as e:
            logger.error("ERROR: CLOUD_RUN_TASK_INDEX not set or invalid.")
            logger.error(e)
            sys.exit(1)

        if task_num >= len(MODEL_RUNS):
            logger.info(f"task number {task_num} not needed, winding down.")
            sys.exit(0)

        model_run = MODEL_RUNS[task_num]

    # Calculate max_workers from rate limit and group size
    group_size_env = os.getenv("RATE_LIMIT_GROUP_SIZE")
    if group_size_env:
        group_size = int(group_size_env)
    else:
        from collections import Counter

        group_counts = Counter(m.rate_limit_group for m in MODEL_RUNS)
        group_size = group_counts[model_run.rate_limit_group]
    rate_limit = RATE_LIMITS.get(model_run.rate_limit_group, 10)
    max_workers = max(1, rate_limit // group_size)

    prompt_type = "zero_shot"

    # Sleep to avoid cloning the same git repo too quickly
    task_num = int(os.getenv("CLOUD_RUN_TASK_INDEX", "0"))
    time.sleep(task_num)

    return forecast_due_date, run_mode, model_run, prompt_type, max_workers


@decorator.log_runtime
def main() -> None:
    """Orchestrate evaluation: load questions, run models, and persist results."""
    configure_keys()

    forecast_due_date, run_mode, model_run, prompt_type, max_workers = parse_env_vars()
    logger.info(f"TESTING: {model_run.name} {prompt_type}")

    num_questions_per_question_type = 2 if run_mode == constants.RunMode.TEST else None

    base_file_path = f"individual_forecast_records/{forecast_due_date}"

    results = {}
    questions = get_questions(
        forecast_due_date=forecast_due_date,
        num_questions_per_question_type=num_questions_per_question_type,
    )
    for question_set in questions:
        for market_use_freeze_value in [False, True]:
            test_type = model_eval.determine_test_type(
                question_set,
                prompt_type,
                market_use_freeze_value,
                run_mode,
            )
            questions_to_eval = question_set

            gcp_file_path = f"{base_file_path}/{test_type}/{model_run.name}.jsonl"

            results[model_run.name] = model_eval.download_and_read_saved_forecasts(
                filename=gcp_file_path,
                base_file_path=base_file_path,
            )

            if results[model_run.name]:
                logger.info(f"Downloaded {gcp_file_path}. Skipping.")
            else:
                logger.info(
                    f"No results loaded for {gcp_file_path}. "
                    f"{model_run.name} is running inference..."
                )
                results[model_run.name] = {i: "" for i in range(len(questions_to_eval))}
                model_eval.process_model(
                    model_run=model_run,
                    test_type=test_type,
                    results=results,
                    questions_to_eval=questions_to_eval,
                    forecast_due_date=forecast_due_date,
                    prompt_type=prompt_type,
                    market_use_freeze_value=market_use_freeze_value,
                    base_file_path=base_file_path,
                    max_workers=max_workers,
                )

    model_eval.generate_final_forecast_files(
        forecast_due_date=forecast_due_date,
        prompt_type=prompt_type,
        model_run=model_run,
        run_mode=run_mode,
    )

    # Validate forecast files
    local_submit_dir = model_eval.get_local_final_submit_directory(
        prompt_type=prompt_type,
        run_mode=run_mode,
    )
    if os.path.exists(local_submit_dir):
        for filename in os.listdir(local_submit_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(local_submit_dir, filename)
                n_market = len(questions[1]) if len(questions) > 1 else 0
                n_dataset = len(questions[0]) if len(questions) > 0 else 0
                result = forecast_validation.validate_forecast_file(
                    filepath, n_market=n_market, n_dataset=n_dataset
                )
                summary = result.format_summary(model_run.name, prompt_type)
                logger.info(f"\n{summary}")

                # In TEST mode, exit on validation failure
                model_to_test = os.getenv("MODEL_TO_TEST")
                if model_to_test and not result.valid_json:
                    logger.error("Validation failed in TEST mode.")
                    sys.exit(1)

    upload_forecast_files(
        base_file_path=base_file_path,
        prompt_type=prompt_type,
        forecast_due_date=forecast_due_date,
        run_mode=run_mode,
    )

    logger.info(f"Done for {model_run.name}")


if __name__ == "__main__":
    main()

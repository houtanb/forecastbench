"""LLM evaluation utilities: prompt building, question processing, forecast generation."""

import json
import logging
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial

from termcolor import colored

from . import (
    constants,
    data_utils,
    env,
    llm_prompts,
    question_curation,
)
from .llm import REFORMAT_MODEL

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))  # noqa: E402
sys.path.append(os.path.join(os.path.dirname(__file__), "../../utils"))  # noqa: E402
from utils import gcp  # noqa: E402

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
TODAY_DATE = datetime.today().strftime("%Y-%m-%d")


def get_local_final_submit_directory(
    prompt_type: str,
    run_mode: constants.RunMode,
) -> str:
    """Construct the local directory path for final forecast submission.

    Args:
        prompt_type (str): The prompt style used for the run (e.g., "zero_shot").
        run_mode (constants.RunMode): Execution mode indicating TEST or PROD context.

    Returns:
        directory (str): Absolute path to the local final submission directory under /tmp.
    """
    directory = f"/tmp/{prompt_type}/final_submit"
    if run_mode == constants.RunMode.TEST:
        directory += "_test"
    return directory


def extract_probability(text):
    """Extract a probability value from the given text.

    Args:
        text (str): The text from which to extract the probability.

    Returns:
        float | None: The first valid probability found in the text, or None if no valid
        probability is detected.
    """
    if text is None:
        return None

    pattern = r"(?:\*\s*)?(\d*\.?\d+)%?(?:\s*\*)?"

    matches = re.findall(pattern, text)

    for match in reversed(matches):
        number = float(match)

        surrounding_text = text[max(0, text.find(match) - 1) : text.find(match) + len(match) + 1]
        if "%" in surrounding_text:
            number /= 100

        if 0 <= number <= 1:
            if number == 1.0 or number == 0.0:
                continue
            return number

    return None


def convert_string_to_list(string_list):
    """Convert a formatted string into a list of floats."""
    string_list = string_list.strip()
    string_list = string_list[1:-1]
    list_values = string_list.split(",")

    actual_list = [
        (
            0.5
            if not re.match(r"^\d*\.\d+$", value.strip().replace("*", ""))
            else float(value.strip().replace("*", ""))
        )
        for value in list_values
    ]

    return actual_list


def reformat_answers(response, prompt="N/A", question="N/A", single=False):
    """Reformat the given response using the REFORMAT_MODEL."""

    def reformatted_raw_response(
        response, prompt, question, REFORMAT_SINGLE_PROMPT, REFORMAT_PROMPT, single=False
    ):
        if single:
            reformat_prompt = REFORMAT_SINGLE_PROMPT.format(response=response)
        else:
            reformat_prompt = REFORMAT_PROMPT.format(
                user_prompt=prompt,
                model_response=response,
                n_horizons=len(question["resolution_dates"]),
            )
        raw_response = REFORMAT_MODEL.get_response(
            prompt=reformat_prompt,
            max_tokens=100,
            temperature=0,
        )
        return raw_response

    raw_response = reformatted_raw_response(
        response,
        prompt,
        question,
        llm_prompts.REFORMAT_SINGLE_PROMPT,
        llm_prompts.REFORMAT_PROMPT,
        single,
    )
    if raw_response == "need_a_new_reformat_prompt":
        raw_response = reformatted_raw_response(
            response,
            prompt,
            question,
            llm_prompts.REFORMAT_SINGLE_PROMPT_2,
            llm_prompts.REFORMAT_PROMPT_2,
            single,
        )

    if single:
        return extract_probability(raw_response)

    return convert_string_to_list(raw_response)


def capitalize_substrings(model_name):
    """Capitalize the first letter of each substring in a model name."""
    model_name = model_name.replace("gpt", "GPT") if "gpt" in model_name else model_name
    substrings = model_name.split("-")
    capitalized_substrings = [
        substr[0].upper() + substr[1:] if substr and not substr[0].isdigit() else substr
        for substr in substrings
    ]
    return "-".join(capitalized_substrings)


def generate_final_forecast_files(forecast_due_date, prompt_type, model_run, run_mode):
    """Generate final forecast files for a model run.

    Args:
        forecast_due_date (str): The forecast_due_date for the forecast.
        prompt_type (str): The type of prompt used.
        model_run: A ModelRun object.
        run_mode: RunMode enum value.
    """
    model = model_run.name
    org = model_run.org
    model_id = model_run.model_id

    def get_final_dir(with_freeze_values, run_mode):
        final_dir = "final_with_freeze" if with_freeze_values else "final"
        if run_mode == constants.RunMode.TEST:
            final_dir += "_test"
        return final_dir

    def write_file(with_freeze_values, run_mode):
        current_model_forecasts = []
        dataset_dir = f"{prompt_type}/dataset"
        market_dir = f"{prompt_type}/market"
        if with_freeze_values:
            market_dir += "_with_freeze_values"

        if run_mode == constants.RunMode.TEST:
            dataset_dir += "_test"
            market_dir += "_test"

        dirs = [
            dataset_dir,
            market_dir,
        ]
        for question_type in dirs:
            file_path = f"/tmp/{question_type}/{model}.jsonl"
            questions = data_utils.read_jsonl(file_path)
            current_model_forecasts.extend(questions)

        final_dir = get_final_dir(with_freeze_values, run_mode)

        final_file_name = f"/tmp/{prompt_type}/{final_dir}/{model}"
        os.makedirs(os.path.dirname(final_file_name), exist_ok=True)
        with open(final_file_name, "w") as file:
            for entry in current_model_forecasts:
                json_line = json.dumps(entry)
                file.write(json_line + "\n")

    def create_final_file(with_freeze_values, run_mode):
        final_dir = get_final_dir(with_freeze_values, run_mode)
        file_path = f"/tmp/{prompt_type}/{final_dir}/{model}"
        questions = data_utils.read_jsonl(file_path)

        local_submit_dir = get_local_final_submit_directory(
            prompt_type=prompt_type,
            run_mode=run_mode,
        )
        os.makedirs(local_submit_dir, exist_ok=True)

        file_prompt_type = prompt_type
        if with_freeze_values:
            file_prompt_type += "_with_freeze_values"

        new_file_name = (
            f"{local_submit_dir}/{forecast_due_date}.{org}.{model}_{file_prompt_type}.json"
        )
        if run_mode == constants.RunMode.TEST:
            new_file_name = (
                f"{local_submit_dir}/{constants.TEST_FORECAST_FILE_PREFIX}.{forecast_due_date}."
                f"{org}.{model}_{file_prompt_type}.json"
            )

        display_name = model_id if "/" not in model_id else model_id.split("/")[1]

        forecast_file = {
            "organization": constants.BENCHMARK_NAME,
            "model": f"{capitalize_substrings(display_name)} ({file_prompt_type.replace('_', ' ')})",
            "model_organization": org,
            "question_set": f"{forecast_due_date}-llm.json",
            "forecast_due_date": forecast_due_date,
            "forecasts": questions,
        }

        with open(new_file_name, "w") as f:
            json.dump(forecast_file, f, indent=4)

    write_file(with_freeze_values=True, run_mode=run_mode)
    create_final_file(with_freeze_values=True, run_mode=run_mode)
    write_file(with_freeze_values=False, run_mode=run_mode)
    create_final_file(with_freeze_values=False, run_mode=run_mode)


def worker(
    index,
    n_questions,
    model_run,
    save_dict,
    questions_to_eval,
    forecast_due_date,
    prompt_type="zero_shot",
    rate_limit=False,
    market_use_freeze_value=False,
):
    """Worker function for question evaluation.

    Args:
        index: Question index to process.
        n_questions: Total number of questions.
        model_run: A ModelRun object used for LLM calls.
        save_dict: Shared dict to store results.
        questions_to_eval: List of question dicts.
        forecast_due_date: Date string for the forecast.
        prompt_type: Prompt variant.
        rate_limit: Whether to rate-limit.
        market_use_freeze_value: Whether to use freeze values for market questions.
    """
    assert prompt_type in ["zero_shot"]
    assert market_use_freeze_value in [True, False]

    if save_dict[index] != "":
        return

    logger.info(f"Starting {model_run.name} - {index + 1}/{n_questions}")

    if rate_limit:
        start_time = datetime.now()

    question = questions_to_eval[index]
    is_market_question = question["source"] in question_curation.MARKET_SOURCES

    if not is_market_question and market_use_freeze_value:
        return

    if is_market_question:
        if market_use_freeze_value:
            prompt = {
                "zero_shot": llm_prompts.ZERO_SHOT_MARKET_WITH_FREEZE_VALUE_PROMPT,
            }.get(prompt_type)
        else:
            prompt = {
                "zero_shot": llm_prompts.ZERO_SHOT_MARKET_PROMPT,
            }.get(prompt_type)
    else:
        prompt = {
            "zero_shot": llm_prompts.ZERO_SHOT_NON_MARKET_PROMPT,
        }.get(prompt_type)

    prompt = prompt.format(
        **get_prompt_params(
            question,
            is_market_question,
            forecast_due_date,
            market_use_freeze_value,
        )
    )

    logger.info(
        f"IN WORKER: ... {model_run.name}. {prompt_type}. Is market_question: {is_market_question}."
    )

    try:
        response = model_run.get_response(
            prompt=prompt,
            max_tokens=100,
            temperature=0,
        )
    except Exception as e:
        logger.error(f"Error in worker: {e}")
        response = None

    if prompt_type == "zero_shot":
        if is_market_question:
            save_dict[index] = {"forecast": extract_probability(response)}
        else:
            save_dict[index] = {
                "forecast": reformat_answers(response=response, prompt=prompt, question=question)
            }

    prompt_col = colored(prompt_type, "red", attrs=["bold"])
    question_type_col = colored("Market" if is_market_question else "Dataset", "yellow")
    model_info = f"{prompt_col} {question_type_col}"
    if is_market_question:
        freeze_vals = "with freeze values" if market_use_freeze_value else "no freeze values"
        freeze_col = colored(freeze_vals, "yellow")
        model_info = f"{model_info} {freeze_col}"

    logger.info(
        f"\n\nModel: {model_run.name} {model_info}"
        f"\nQuestion source // id // url: "
        f"{question['source']} // {question['id']} // {question['url']}"
        f"\nForecast: {save_dict[index]['forecast']}\n"
    )
    if rate_limit:
        end_time = datetime.now()
        elapsed_time = (end_time - start_time).total_seconds()
        if elapsed_time < 1:
            time.sleep(1 - elapsed_time)

    return None


def executor(
    model_run,
    save_dict,
    questions_to_eval,
    forecast_due_date,
    prompt_type="zero_shot",
    market_use_freeze_value=False,
    max_workers=None,
):
    """Execute question evaluation with ThreadPoolExecutor.

    Args:
        model_run: A ModelRun object.
        save_dict: Shared dict for results.
        questions_to_eval: List of question dicts.
        forecast_due_date: Date string.
        prompt_type: Prompt variant.
        market_use_freeze_value: Whether to use freeze values.
        max_workers: Number of concurrent workers. Defaults to env.NUM_CPUS.
    """
    if max_workers is None:
        max_workers = env.NUM_CPUS

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        worker_with_args = partial(
            worker,
            n_questions=len(questions_to_eval),
            model_run=model_run,
            save_dict=save_dict,
            questions_to_eval=questions_to_eval,
            forecast_due_date=forecast_due_date,
            prompt_type=prompt_type,
            market_use_freeze_value=market_use_freeze_value,
        )
        return list(pool.map(worker_with_args, range(len(questions_to_eval))))


def get_all_retrieved_info(all_retrieved_info):
    """Get all retrieved news."""
    retrieved_info = ""
    for summary in all_retrieved_info:
        retrieved_info += f"Article title: {summary['title']}" + "\n"
        retrieved_info += f"Summary: {summary['summary']}" + "\n\n"
    return retrieved_info


def get_prompt_params(
    question,
    is_market_question,
    forecast_due_date,
    market_use_freeze_value,
):
    """Get prompt parameters."""

    def formatted_question(question):
        question = question["question"].replace("{forecast_due_date}", forecast_due_date)
        question = question.replace(
            "{resolution_date}", "each of the resolution dates provided below"
        )
        return question

    background = question["background"]
    if question["market_info_resolution_criteria"] != "N/A":
        background += "\n" + question["market_info_resolution_criteria"]

    base_params = {
        "question": formatted_question(question),
        "background": background,
        "resolution_criteria": question["resolution_criteria"],
        "today_date": TODAY_DATE,
    }

    if is_market_question:
        base_params["resolution_date"] = question["market_info_close_datetime"]
        if market_use_freeze_value:
            base_params.update(
                {
                    "freeze_datetime": question["freeze_datetime"],
                    "freeze_datetime_value": question["freeze_datetime_value"],
                }
            )
    else:
        base_params.update(
            {
                "freeze_datetime": question["freeze_datetime"],
                "freeze_datetime_value": question["freeze_datetime_value"],
                "freeze_datetime_value_explanation": question["freeze_datetime_value_explanation"],
                "list_of_resolution_dates": question["resolution_dates"],
            }
        )

    return base_params


def download_and_read_saved_forecasts(filename, base_file_path):
    """Download saved forecasts from cloud storage."""
    local_filename = "/tmp/" + filename.replace(base_file_path + "/", "")

    os.makedirs(os.path.dirname(local_filename), exist_ok=True)

    gcp.storage.download_no_error_message_on_404(
        bucket_name=env.FORECAST_SETS_BUCKET,
        filename=filename,
        local_filename=local_filename,
    )

    if not os.path.exists(local_filename):
        return None

    return data_utils.read_jsonl(local_filename)


def process_model(
    model_run,
    test_type,
    results,
    questions_to_eval,
    forecast_due_date,
    prompt_type,
    market_use_freeze_value,
    base_file_path,
    max_workers=None,
):
    """Process a single model run for the given questions.

    Args:
        model_run: A ModelRun object.
        test_type: The test type string.
        results: Dict mapping model name to results dict.
        questions_to_eval: List of question dicts.
        forecast_due_date: Date string.
        prompt_type: Prompt variant.
        market_use_freeze_value: Whether to use freeze values.
        base_file_path: GCS base path.
        max_workers: Number of concurrent workers.
    """
    workers = max_workers or env.NUM_CPUS
    logger.info(f"{model_run.name} is using {workers} workers.")
    executor(
        model_run,
        results[model_run.name],
        questions_to_eval,
        forecast_due_date,
        prompt_type=prompt_type,
        market_use_freeze_value=market_use_freeze_value,
        max_workers=max_workers,
    )

    current_model_forecasts = generate_forecasts(
        model_run.name, results, questions_to_eval, prompt_type
    )
    save_and_upload_results(current_model_forecasts, test_type, model_run.name, base_file_path)


def determine_test_type(question_set, prompt_type, market_use_freeze_value, run_mode):
    """Determine the test type based on the question set and prompt type."""
    base_type = (
        "market" if question_set[0]["source"] in question_curation.MARKET_SOURCES else "dataset"
    )

    if base_type == "market" and market_use_freeze_value:
        base_type += "_with_freeze_values"

    if run_mode == constants.RunMode.TEST:
        base_type += "_test"

    return f"{prompt_type}/{base_type}"


def generate_forecasts(model, results, questions_to_eval, prompt_type):
    """Generate forecasts for the current model."""
    forecasts = []
    for index, question in enumerate(questions_to_eval):
        if question["source"] in question_curation.DATA_SOURCES:
            forecasts.extend(
                generate_data_source_forecasts(model, results, question, index, prompt_type)
            )
        else:
            forecasts.append(
                generate_non_data_source_forecast(model, results, question, index, prompt_type)
            )
    return forecasts


def generate_data_source_forecasts(model, results, question, index, prompt_type):
    """Generate forecasts for questions from data sources."""
    forecasts = []
    model_results = results[model][index]["forecast"]
    for forecast, resolution_date in zip(model_results, question["resolution_dates"]):
        forecast_data = {
            "id": question["id"],
            "source": question["source"],
            "forecast": forecast,
            "resolution_date": resolution_date,
            "reasoning": None if prompt_type == "zero_shot" else results[model][index]["reasoning"],
        }
        forecasts.append(forecast_data)
    return forecasts


def generate_non_data_source_forecast(model, results, question, index, prompt_type):
    """Generate a forecast for questions not from data sources."""
    forecast_data = {
        "id": question["id"],
        "source": question["source"],
        "forecast": results[model][index]["forecast"],
        "resolution_date": None,
        "reasoning": None if prompt_type == "zero_shot" else results[model][index]["reasoning"],
    }
    return forecast_data


def save_and_upload_results(forecasts, test_type, model, base_file_path):
    """Save results locally and upload to GCP."""
    local_filename = f"/tmp/{test_type}/{model}.jsonl"
    os.makedirs(os.path.dirname(local_filename), exist_ok=True)
    with open(local_filename, "w") as file:
        for entry in forecasts:
            json_line = json.dumps(entry)
            file.write(json_line + "\n")

    remote_filename = local_filename.replace("/tmp/", "")
    gcp.storage.upload(
        bucket_name=env.FORECAST_SETS_BUCKET,
        local_filename=local_filename,
        filename=f"{base_file_path}/{remote_filename}",
    )


def process_questions(questions_file, num_questions_per_question_type=None):
    """Process questions from a JSON file and categorize them."""
    with open(questions_file, "r") as file:
        questions_data = json.load(file)

    questions = questions_data["questions"]

    market_questions = [q for q in questions if q["source"] in question_curation.MARKET_SOURCES]
    dataset_questions = [q for q in questions if q["source"] in question_curation.DATA_SOURCES]

    if num_questions_per_question_type is not None:
        market_questions = random.sample(
            market_questions, k=min(num_questions_per_question_type, len(market_questions))
        )
        dataset_questions = random.sample(
            dataset_questions, k=min(num_questions_per_question_type, len(dataset_questions))
        )

    return (
        market_questions,
        dataset_questions,
    )

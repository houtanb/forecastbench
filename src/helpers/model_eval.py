"""LLM-related util."""

import asyncio
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial

import anthropic
import google.generativeai as google_ai
import openai
import together
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

from . import constants, data_utils, env, keys, llm_prompts, question_curation

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))  # noqa: E402
from utils import gcp  # noqa: E402

anthropic_console = anthropic.Anthropic(api_key=keys.API_KEY_ANTHROPIC)
anthropic_async_client = anthropic.AsyncAnthropic(api_key=keys.API_KEY_ANTHROPIC)
oai_async_client = openai.AsyncOpenAI(api_key=keys.API_KEY_OPENAI)
oai = openai.OpenAI(api_key=keys.API_KEY_OPENAI)
together.api_key = keys.API_KEY_TOGETHERAI
google_ai.configure(api_key=keys.API_KEY_GOOGLE)
client = openai.OpenAI(
    api_key=keys.API_KEY_TOGETHERAI,
    base_url="https://api.together.xyz/v1",
)
mistral_client = MistralClient(api_key=keys.API_KEY_MISTRAL)
HUMAN_JOINT_PROMPTS = [
    llm_prompts.HUMAN_JOINT_PROMPT_1,
    llm_prompts.HUMAN_JOINT_PROMPT_2,
    llm_prompts.HUMAN_JOINT_PROMPT_3,
    llm_prompts.HUMAN_JOINT_PROMPT_4,
]

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def infer_model_source(model_name):
    """
    Infer the model source from the model name.

    Args:
    - model_name (str): The name of the model.
    """
    if "ft:gpt" in model_name:  # fine-tuned GPT-3 or 4
        return constants.OAI_SOURCE
    if model_name not in constants.MODEL_NAME_TO_SOURCE:
        raise ValueError(f"Invalid model name: {model_name}")
    return constants.MODEL_NAME_TO_SOURCE[model_name]


def get_response_with_retry(api_call, wait_time, error_msg):
    """
    Make an API call and retry on failure after a specified wait time.

    Args:
        api_call (function): API call to make.
        wait_time (int): Time to wait before retrying, in seconds.
        error_msg (str): Error message to print on failure.
    """
    while True:
        try:
            return api_call()
        except Exception as e:
            if "repetitive patterns" in str(e):
                logger.info(
                    "Repetitive patterns detected in the prompt. Modifying prompt and retrying..."
                )
                return "need_a_new_reformat_prompt"

            logger.info(f"{error_msg}: {e}")
            logger.info(f"Waiting for {wait_time} seconds before retrying...")

            time.sleep(wait_time)


def get_response_from_oai_model(
    model_name, prompt, system_prompt, max_tokens, temperature, wait_time
):
    """
    Make an API call to the OpenAI API and retry on failure after a specified wait time.

    Args:
        model_name (str): Name of the model to use (such as "gpt-4").
        prompt (str): Fully specififed prompt to use for the API call.
        system_prompt (str): Prompt to use for system prompt.
        max_tokens (int): Maximum number of tokens to sample.
        temperature (float): Sampling temperature.
        wait_time (int): Time to wait before retrying, in seconds.

    Returns:
        str: Response string from the API call.
    """

    def api_call():
        """
        Make an API call to the OpenAI API, without retrying on failure.

        Returns:
            str: Response string from the API call.
        """
        model_input = [{"role": "system", "content": system_prompt}] if system_prompt else []
        model_input.append({"role": "user", "content": prompt})
        response = oai.chat.completions.create(
            model=model_name,
            messages=model_input,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        # logger.info(f"full prompt: {prompt}")
        return response.choices[0].message.content

    if (
        get_response_with_retry(api_call, wait_time, "OpenAI API request exceeded rate limit.")
        == "need_a_new_reformat_prompt"
    ):
        return "need_a_new_reformat_prompt"
    else:
        return get_response_with_retry(
            api_call, wait_time, "OpenAI API request exceeded rate limit."
        )


def get_response_from_anthropic_model(model_name, prompt, max_tokens, temperature, wait_time):
    """
    Make an API call to the Anthropic API and retry on failure after a specified wait time.

    Args:
        model_name (str): Name of the model to use (such as "claude-2").
        prompt (str): Fully specififed prompt to use for the API call.
        max_tokens (int): Maximum number of tokens to sample.
        temperature (float): Sampling temperature.
        wait_time (int): Time to wait before retrying, in seconds.

    Returns:
        str: Response string from the API call.
    """
    if max_tokens > 4096:
        max_tokens = 4096

    def api_call():
        completion = anthropic_console.messages.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return completion.content[0].text

    return get_response_with_retry(
        api_call, wait_time, "Anthropic API request exceeded rate limit."
    )


def get_response_from_mistral_model(model_name, prompt, max_tokens, temperature, wait_time):
    """
    Make an API call to the OpenAI API and retry on failure after a specified wait time.

    Args:
        model_name (str): Name of the model to use (such as "gpt-4").
        prompt (str): Fully specififed prompt to use for the API call.
        max_tokens (int): Maximum number of tokens to sample.
        temperature (float): Sampling temperature.
        wait_time (int): Time to wait before retrying, in seconds.

    Returns:
        str: Response string from the API call.
    """

    def api_call():
        """
        Make an API call to the OpenAI API, without retrying on failure.

        Returns:
            str: Response string from the API call.
        """
        messages = [ChatMessage(role="user", content=prompt)]

        # No streaming
        chat_response = mistral_client.chat(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return chat_response.choices[0].message.content

    return get_response_with_retry(api_call, wait_time, "Mistral API request exceeded rate limit.")


def get_response_from_together_ai_model(model_name, prompt, max_tokens, temperature, wait_time):
    """
    Make an API call to the Together AI API and retry on failure after a specified wait time.

    Args:
        model_name (str): Name of the model to use (such as "togethercomputer/
        llama-2-13b-chat").
        prompt (str): Fully specififed prompt to use for the API call.
        max_tokens (int): Maximum number of tokens to sample.
        temperature (float): Sampling temperature.
        wait_time (int): Time to wait before retrying, in seconds.

    Returns:
        str: Response string from the API call.
    """

    def api_call():
        chat_completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        response = chat_completion.choices[0].message.content

        return response

    return get_response_with_retry(
        api_call, wait_time, "Together AI API request exceeded rate limit."
    )


def get_response_from_google_model(model_name, prompt, max_tokens, temperature, wait_time):
    """
    Make an API call to the Together AI API and retry on failure after a specified wait time.

    Args:
        model (str): Name of the model to use (such as "gemini-pro").
        prompt (str): Initial prompt for the API call.
        max_tokens (int): Maximum number of tokens to sample.
        temperature (float): Sampling temperature.
        wait_time (int): Time to wait before retrying, in seconds.

    Returns:
        str: Response string from the API call.
    """
    model = google_ai.GenerativeModel(model_name)

    response = model.generate_content(
        prompt,
        generation_config=google_ai.types.GenerationConfig(
            candidate_count=1,
            max_output_tokens=max_tokens,
            temperature=temperature,
        ),
    )
    return response.text


def get_response_from_model(
    model_name,
    prompt,
    system_prompt="",
    max_tokens=2000,
    temperature=0.8,
    wait_time=30,
):
    """
    Make an API call to the specified model and retry on failure after a specified wait time.

    Args:
        model_name (str): Name of the model to use (such as "gpt-4").
        prompt (str): Fully specififed prompt to use for the API call.
        system_prompt (str, optional): Prompt to use for system prompt.
        max_tokens (int, optional): Maximum number of tokens to generate.
        temperature (float, optional): Sampling temperature.
        wait_time (int, optional): Time to wait before retrying, in seconds.
    """
    model_source = infer_model_source(model_name)
    if model_source == constants.OAI_SOURCE:
        return get_response_from_oai_model(
            model_name, prompt, system_prompt, max_tokens, temperature, wait_time
        )
    elif model_source == constants.ANTHROPIC_SOURCE:
        return get_response_from_anthropic_model(
            model_name, prompt, max_tokens, temperature, wait_time
        )
    elif model_source == constants.TOGETHER_AI_SOURCE:
        return get_response_from_together_ai_model(
            model_name, prompt, max_tokens, temperature, wait_time
        )
    elif model_source == constants.GOOGLE_SOURCE:
        return get_response_from_google_model(
            model_name, prompt, max_tokens, temperature, wait_time
        )
    elif model_source == constants.MISTRAL_SOURCE:
        return get_response_from_mistral_model(
            model_name, prompt, max_tokens, temperature, wait_time
        )
    else:
        return "Not a valid model source."


async def get_async_response(
    prompt,
    model_name="gpt-3.5-turbo-1106",
    temperature=0.0,
    max_tokens=8000,
):
    """
    Asynchronously get a response from the OpenAI API.

    Args:
        prompt (str): Fully specififed prompt to use for the API call.
        model_name (str, optional): Name of the model to use (such as "gpt-3.5-turbo").
        temperature (float, optional): Sampling temperature.
        max_tokens (int, optional): Maximum number of tokens to sample.

    Returns:
        str: Response string from the API call (not the dictionary).
    """
    model_source = infer_model_source(model_name)
    while True:
        try:
            if model_source == constants.OAI_SOURCE:
                response = await oai_async_client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                )
                return response.choices[0].message.content
            elif model_source == constants.ANTHROPIC_SOURCE:
                response = await anthropic_async_client.messages.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=4096,
                )
                return response.content[0].text
            elif model_source == constants.GOOGLE_SOURCE:
                model = google_ai.GenerativeModel(model_name)
                response = await model.generate_content_async(
                    prompt,
                    generation_config=google_ai.types.GenerationConfig(
                        candidate_count=1,
                        max_output_tokens=max_tokens,
                        temperature=temperature,
                    ),
                )
                return response.text
            elif model_source == constants.TOGETHER_AI_SOURCE:
                chat_completion = await asyncio.to_thread(
                    client.chat.completions.create,
                    model=model_name,
                    messages=[
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return chat_completion.choices[0].message.content
            else:
                logger.debug("Not a valid model source: {model_source}")
                return ""
        except Exception as e:
            logger.info(f"Exception, erorr message: {e}")
            logger.info("Waiting for 30 seconds before retrying...")
            time.sleep(30)
            continue


def extract_probability(text):
    """
    Extract a probability value from the given text.

    Search through the input text for numeric patterns that could represent probabilities.
    The search checks for plain numbers, percentages, or numbers flanked by asterisks and
    attempts to convert these into a probability value (a float between 0 and 1).
    Ignore exact values of 0.0 and 1.0.

    Args:
        text (str): The text from which to extract the probability.

    Returns:
        float | None: The first valid probability found in the text, or None if no valid
        probability is detected.
    """
    pattern = r"(?:\*\s*)?(\d*\.?\d+)%?(?:\s*\*)?"

    matches = re.findall(pattern, text)

    for match in reversed(matches):
        number = float(match)

        surrounding_text = text[max(0, text.find(match) - 1) : text.find(match) + len(match) + 1]
        if "%" in surrounding_text:
            number /= 100

        if 0 <= number <= 1:
            if number == 1.0 or number == 0.0:
                continue  # Skip this match and continue to the next one
            return number

    return None


def convert_string_to_list(string_list):
    """
    Convert a formatted string into a list of floats.

    Strip leading and trailing whitespace from the input string, remove square brackets,
    split the string by commas, and convert each element to a float. Replace non-numeric
    entries (denoted by '*') with 0.5.

    Parameters:
    string_list (str): A string representation of a list of numbers, enclosed in square
                       brackets and separated by commas.

    Returns:
    list: A list of floats, where non-numeric elements are replaced with 0.5.
    """
    # Remove leading and trailing whitespace
    string_list = string_list.strip()
    # Remove square brackets at the beginning and end
    string_list = string_list[1:-1]
    # Split the string by commas and convert each element to a float
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
    """
    Reformat the given response based on whether a single response or multiple responses are required.

    This function adjusts the response formatting by using predefined prompt templates and sends
    it to a model for evaluation. Depending on the 'single' flag, it either extracts a probability
    or converts the response to a list.

    Parameters:
    - response (str): The original response from the model that needs to be reformatted.
    - prompt (str, optional): The user prompt to use in the reformatting process. Defaults to 'N/A'.
    - question (str or dict, optional): The question data used to format the response when not single.
      Defaults to 'N/A'.
    - single (bool, optional): Flag to determine if the response should be handled as a single response.
      Defaults to False.

    Returns:
    - str or list: The reformatted model response, either as a probability (if single is True) or as a
      list of responses (if single is False).
    """

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
        raw_response = get_response_from_model(
            prompt=reformat_prompt,
            max_tokens=100,
            model_name="gpt-3.5-turbo-0125",
            temperature=0,
            wait_time=30,
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
    """
    Capitalize the first letter of each substring in a model name.

    Args:
        model_name (str): The model name to be capitalized.

    Returns:
        str: The capitalized model name.
    """
    model_name = model_name.replace("gpt", "GPT") if "gpt" in model_name else model_name
    substrings = model_name.split("-")
    capitalized_substrings = [
        substr[0].upper() + substr[1:] if substr and not substr[0].isdigit() else substr
        for substr in substrings
    ]
    return "-".join(capitalized_substrings)


def generage_final_forecast_files(deadline, prompt_type, models):
    """
    Generate final forecast files for given models, merging individual forecasts into final files.

    Args:
        deadline (str): The deadline for the forecast.
        prompt_type (str): The type of prompt used.
        models (dict): A dictionary of models with their information.

    Returns:
        None
    """
    models_to_test = list(models.keys())

    for model in models_to_test:
        current_model_forecasts = []
        for test_type in [
            f"{prompt_type}/non_market",
            f"{prompt_type}/market",
            f"{prompt_type}/combo_non_market",
            f"{prompt_type}/combo_market",
        ]:
            file_path = f"{test_type}/{model}.jsonl"
            questions = data_utils.read_jsonl(file_path)
            current_model_forecasts.extend(questions)

        final_file_name = f"{prompt_type}/final/{model}"
        os.makedirs(os.path.dirname(final_file_name), exist_ok=True)
        with open(final_file_name, "w") as file:
            for entry in current_model_forecasts:
                json_line = json.dumps(entry)
                file.write(json_line + "\n")

    for model in models_to_test:
        file_path = f"{prompt_type}/final/{model}"
        questions = data_utils.read_jsonl(file_path)
        if "gpt" in model:
            org = "OpenAI"
        elif "llama" in model:
            org = "Meta"
        elif "mistral" in model:
            org = "Mistral AI"
        elif "claude" in model:
            org = "Anthropic"
        elif "qwen" in model:
            org = "Qwen"
        elif "gemini" in model:
            org = "Google"

        directory = f"{prompt_type}/final_submit"
        os.makedirs(directory, exist_ok=True)

        new_file_name = f"{directory}/{deadline}.{org}.{model}_{prompt_type}.json"

        model_name = (
            models[model]["full_name"]
            if "/" not in models[model]["full_name"]
            else models[model]["full_name"].split("/")[1]
        )

        forecast_file = {
            "organization": org,
            "model": f"{capitalize_substrings(model_name)} ({prompt_type.replace('_', ' ')})",
            "question_set": f"{deadline}-llm.jsonl",
            "forecast_date": datetime.today().strftime("%Y-%m-%d"),
            "forecasts": questions,
        }

        with open(new_file_name, "w") as f:
            json.dump(forecast_file, f, indent=4)


def worker(
    index,
    model_name,
    save_dict,
    questions_to_eval,
    forecast_due_date,
    mode="zero_shot",
    rate_limit=False,
):
    """Worker function for question evaluation."""
    if save_dict[index] != "":
        return

    logger.info(f"Starting {model_name} - {index}")

    if rate_limit:
        start_time = datetime.now()

    question = questions_to_eval[index]
    is_market_question = question["source"] not in question_curation.DATA_SOURCES
    is_joint_question = question["combination_of"] != "N/A"

    if is_market_question:
        if is_joint_question:
            prompt = (
                llm_prompts.ZERO_SHOT_MARKET_JOINT_QUESTION_PROMPT
                if mode == "zero_shot"
                else llm_prompts.SCRATCH_PAD_MARKET_JOINT_QUESTION_PROMPT
            )
        else:
            prompt = (
                llm_prompts.ZERO_SHOT_MARKET_PROMPT
                if mode == "zero_shot"
                else llm_prompts.SCRATCH_PAD_MARKET_PROMPT
            )
    else:
        if is_joint_question:
            prompt = (
                llm_prompts.ZERO_SHOT_NON_MARKET_JOINT_QUESTION_PROMPT
                if mode == "zero_shot"
                else llm_prompts.SCRATCH_PAD_NON_MARKET_JOINT_QUESTION_PROMPT
            )
        else:
            prompt = (
                llm_prompts.ZERO_SHOT_NON_MARKET_PROMPT
                if mode == "zero_shot"
                else llm_prompts.SCRATCH_PAD_NON_MARKET_PROMPT
            )

    prompt = prompt.format(
        **get_prompt_params(question, is_market_question, is_joint_question, forecast_due_date)
    )

    try:
        response = get_response_from_model(
            prompt=prompt,
            max_tokens=100 if mode == "zero_shot" else (1300 if is_market_question else 2000),
            model_name=model_name,
            temperature=0,
            wait_time=30,
        )
    except Exception as e:
        print(f"An error occurred: {e}")
        response = None

    if mode == "zero_shot":
        if is_market_question:
            save_dict[index] = extract_probability(response)
        else:
            save_dict[index] = reformat_answers(response=response, prompt=prompt, question=question)
    else:  # scratchpad mode
        if is_market_question:
            save_dict[index] = (reformat_answers(response=response, single=True), response)
        else:
            save_dict[index] = (
                reformat_answers(response=response, prompt=prompt, question=question),
                response,
            )

    logger.info(
        f"Model: {model_name} | Answer: {save_dict[index] if mode == 'zero_shot' else save_dict[index][0]}"
    )

    if rate_limit:
        end_time = datetime.now()
        elapsed_time = (end_time - start_time).total_seconds()
        if elapsed_time < 1:
            time.sleep(1 - elapsed_time)

    return None


def executor(
    max_workers, model_name, save_dict, questions_to_eval, forecast_due_date, mode="zero_shot"
):
    """Executor function."""
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        worker_with_args = partial(
            worker,
            model_name=model_name,
            save_dict=save_dict,
            questions_to_eval=questions_to_eval,
            forecast_due_date=forecast_due_date,
            mode=mode,
        )
        return list(executor.map(worker_with_args, range(len(questions_to_eval))))


def get_prompt_params(question, is_market_question, is_joint_question, forecast_due_date):
    """Get prompt parameters."""
    base_params = {
        "question": question["question"].replace("{forecast_due_date}", forecast_due_date),
        "background": question["background"] + "\n" + question["market_info_resolution_criteria"],
        "resolution_criteria": question["resolution_criteria"],
    }

    if is_market_question:
        base_params["resolution_date"] = question["market_info_close_datetime"]
    else:
        base_params.update(
            {
                "freeze_datetime": question["freeze_datetime"],
                "freeze_datetime_value": question["freeze_datetime_value"],
                "freeze_datetime_value_explanation": question["freeze_datetime_value_explanation"],
                "list_of_resolution_dates": question["resolution_dates"],
            }
        )

    if is_joint_question:
        joint_params = {
            "human_prompt": HUMAN_JOINT_PROMPTS[question["combo_index"]],
            "question_1": question["combination_of"][0]["question"].replace(
                "{forecast_due_date}", forecast_due_date
            ),
            "question_2": question["combination_of"][1]["question"].replace(
                "{forecast_due_date}", forecast_due_date
            ),
            "background_1": question["combination_of"][0]["background"]
            + "\n"
            + question["combination_of"][0]["market_info_resolution_criteria"],
            "background_2": question["combination_of"][1]["background"]
            + "\n"
            + question["combination_of"][1]["market_info_resolution_criteria"],
            "resolution_criteria_1": question["combination_of"][0]["resolution_criteria"],
            "resolution_criteria_2": question["combination_of"][1]["resolution_criteria"],
        }
        if is_market_question:
            joint_params["resolution_date"] = max(
                question["combination_of"][0]["market_info_close_datetime"],
                question["combination_of"][1]["market_info_close_datetime"],
            )
        else:
            joint_params.update(
                {
                    "freeze_datetime_1": question["combination_of"][0]["freeze_datetime"],
                    "freeze_datetime_2": question["combination_of"][1]["freeze_datetime"],
                    "freeze_datetime_value_1": question["combination_of"][0][
                        "freeze_datetime_value"
                    ],
                    "freeze_datetime_value_2": question["combination_of"][1][
                        "freeze_datetime_value"
                    ],
                    "freeze_datetime_value_explanation_1": question["combination_of"][0][
                        "freeze_datetime_value_explanation"
                    ],
                    "freeze_datetime_value_explanation_2": question["combination_of"][1][
                        "freeze_datetime_value_explanation"
                    ],
                    "list_of_resolution_dates": question["resolution_dates"],
                }
            )
        return joint_params
    else:
        return base_params


def process_questions_and_models(questions, models, prompt_type, base_file_path, forecast_due_date):
    """
    Process questions for different models and prompt types.

    Args:
    questions (list): List of question sets to evaluate.
    models (dict): Dictionary containing model information.
    prompt_type (str): Type of prompt ('zero_shot' or 'scratchpad').
    base_file_path (str): Base path for file storage.
    forecast_due_date (str): Due date for forecasts.

    Steps:
    1. Determine test type for each question set.
    2. Load existing results or initialize new ones.
    3. Process each model for each question set.
    4. Save and upload results.
    """
    results = {}
    models_to_test = list(models.keys())
    model_result_loaded = {model: False for model in models_to_test}

    for question_set in questions:
        test_type = determine_test_type(question_set, prompt_type)
        questions_to_eval = question_set

        for model in models_to_test:
            gcp_file_path = f"{base_file_path}/{test_type}/{model}.jsonl"

            results[model] = data_utils.download_and_read_saved_forecasts(
                gcp_file_path, base_file_path
            )

            if results[model]:
                model_result_loaded[model] = True
                logger.info(f"Downloaded {gcp_file_path}.")
            else:
                logger.info(f"No results loaded for {gcp_file_path}.")
                model_result_loaded[model] = False
                results[model] = {i: "" for i in range(len(questions_to_eval))}

        for model in models_to_test:
            if not model_result_loaded[model]:
                logger.info(f"{model} is running inference...")
                process_model(
                    model,
                    models,
                    test_type,
                    results,
                    questions_to_eval,
                    forecast_due_date,
                    prompt_type,
                    base_file_path,
                )


def process_model(
    model,
    models,
    test_type,
    results,
    questions_to_eval,
    forecast_due_date,
    prompt_type,
    base_file_path,
):
    """Process a single model for the given questions."""
    executor_count = get_executor_count(model, models)
    logger.info(f"{model} is using {executor_count} workers.")
    executor(
        executor_count,
        models[model]["full_name"],
        results[model],
        questions_to_eval,
        forecast_due_date,
        mode=prompt_type,
    )

    current_model_forecasts = generate_forecasts(model, results, questions_to_eval, prompt_type)
    save_and_upload_results(current_model_forecasts, test_type, model, base_file_path)


def get_executor_count(model, models):
    """Get the executor count based on the model source."""
    if models[model]["source"] == "ANTHROPIC":
        return 30
    elif models[model]["source"] == "GOOGLE":
        return 10
    return 50


def determine_test_type(question_set, prompt_type):
    """Determine the test type based on the question set and prompt type."""
    if question_set[0]["source"] not in question_curation.DATA_SOURCES:
        base_type = "market" if question_set[0]["combination_of"] == "N/A" else "combo_market"
    else:
        base_type = (
            "non_market" if question_set[0]["combination_of"] == "N/A" else "combo_non_market"
        )
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
    model_results = (
        results[model][index] if prompt_type == "zero_shot" else results[model][index][0]
    )
    for forecast, resolution_date in zip(model_results, question["resolution_dates"]):
        forecast_data = {
            "id": question["id"],
            "source": question["source"],
            "forecast": forecast,
            "resolution_date": resolution_date,
            "reasoning": None if prompt_type == "zero_shot" else results[model][index][1],
        }
        if question["combination_of"] != "N/A":
            forecast_data["direction"] = get_direction(question["combo_index"])
        forecasts.append(forecast_data)
    return forecasts


def generate_non_data_source_forecast(model, results, question, index, prompt_type):
    """Generate a forecast for questions not from data sources."""
    return {
        "id": question["id"],
        "source": question["source"],
        "forecast": (
            results[model][index] if prompt_type == "zero_shot" else results[model][index][0]
        ),
        "reasoning": None if prompt_type == "zero_shot" else results[model][index][1],
    }


def get_direction(combo_index):
    """Get the direction based on the combo index."""
    directions = {0: [1, 1], 1: [1, -1], 2: [-1, 1], 3: [-1, -1]}
    return directions.get(combo_index, [0, 0])


def save_and_upload_results(forecasts, test_type, model, base_file_path):
    """Save results locally and upload to GCP."""
    local_filename = f"{test_type}/{model}.jsonl"
    os.makedirs(os.path.dirname(local_filename), exist_ok=True)
    with open(local_filename, "w") as file:
        for entry in forecasts:
            json_line = json.dumps(entry)
            file.write(json_line + "\n")

    gcp.storage.upload(
        bucket_name=env.FORECAST_SETS_BUCKET,
        local_filename=local_filename,
        filename=f"{base_file_path}/{local_filename}",
    )


def process_questions(questions_file, num_per_source=None):
    """
    Process questions from a JSON file and categorize them.

    Load questions from the specified JSON file. Categorize them into single and combo
    questions for both market and non-market sources. Unroll combo questions.
    Optionally limit the number of questions per source.

    Args:
        questions_file (str): Path to the JSON file containing questions.
        num_per_source (int, optional): Number of questions to return per source.
                                        If None, return all questions.

    Returns:
        tuple: Contains four lists in the following order:
               1. Single market questions
               2. Single non-market questions
               3. Unrolled combo market questions
               4. Unrolled combo non-market questions

    Raises:
        FileNotFoundError: If the specified questions_file does not exist.
        json.JSONDecodeError: If the JSON file is not properly formatted.
    """
    with open(questions_file, "r") as file:
        questions_data = json.load(file)

    questions = questions_data["questions"]

    single_market_questions = [
        q
        for q in questions
        if q["combination_of"] == "N/A" and q["source"] not in question_curation.DATA_SOURCES
    ]
    single_non_market_questions = [
        q
        for q in questions
        if q["combination_of"] == "N/A" and q["source"] in question_curation.DATA_SOURCES
    ]

    combo_market_questions = [
        q
        for q in questions
        if q["combination_of"] != "N/A" and q["source"] not in question_curation.DATA_SOURCES
    ]
    combo_non_market_questions = [
        q
        for q in questions
        if q["combination_of"] != "N/A" and q["source"] in question_curation.DATA_SOURCES
    ]

    def unroll(combo_questions):
        """Unroll combo questions by directions."""
        combo_questions_unrolled = []
        for q in combo_questions:
            for i in range(4):
                new_q = q.copy()
                new_q["combo_index"] = i
                combo_questions_unrolled.append(new_q)
        return combo_questions_unrolled

    combo_market_questions_unrolled = unroll(combo_market_questions)
    combo_non_market_questions_unrolled = unroll(combo_non_market_questions)

    if num_per_source is not None:
        single_market_questions = single_market_questions[:num_per_source]
        single_non_market_questions = single_non_market_questions[:num_per_source]
        combo_market_questions_unrolled = combo_market_questions_unrolled[:num_per_source]
        combo_non_market_questions_unrolled = combo_non_market_questions_unrolled[:num_per_source]

    return (
        single_market_questions,
        single_non_market_questions,
        combo_market_questions_unrolled,
        combo_non_market_questions_unrolled,
    )

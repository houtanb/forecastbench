"""Forecast response parsing."""

import re

from llm_forecaster import prompts


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


def extract_probabilities(text):
    """
    Extract all probability values from text in left-to-right order.

    Percentages are converted to decimal probabilities. Values outside the inclusive
    0-to-1 range are ignored.
    """
    if text is None:
        return []

    pattern = r"(?:\*\s*)?(\d*\.?\d+)%?(?:\s*\*)?"
    probabilities = []
    for match in re.finditer(pattern, text):
        number = float(match.group(1))
        surrounding_text = text[max(0, match.start() - 1) : match.end() + 1]
        if "%" in surrounding_text:
            number /= 100
        if 0 <= number <= 1:
            probabilities.append(number)
    return probabilities


def _reformat_dataset_response(
    response: str,
    prompt: str,
    question: dict,
    reformat_model,
) -> str:
    """Reformat a dataset forecast response, retrying once with the alternate prompt."""
    reformat_prompt = prompts.REFORMAT_PROMPT.format(
        user_prompt=prompt,
        model_response=response,
        n_horizons=len(question["resolution_dates"]),
    )
    raw_response = reformat_model.get_response(reformat_prompt)

    if raw_response == "need_a_new_reformat_prompt":
        reformat_prompt = prompts.REFORMAT_PROMPT_2.format(
            user_prompt=prompt,
            model_response=response,
            n_horizons=len(question["resolution_dates"]),
        )
        raw_response = reformat_model.get_response(reformat_prompt)

    return raw_response


def parse_market_forecast(response, reformat_model):
    """Parse a market question forecast response into one probability."""
    return extract_probability(response)


def parse_dataset_forecast(response, prompt, question, reformat_model):
    """Parse a dataset question forecast response into horizon probabilities."""
    n_horizons = len(question["resolution_dates"])
    forecasts = extract_probabilities(response)
    if len(forecasts) == n_horizons:
        return forecasts

    raw_response = _reformat_dataset_response(
        response=response,
        prompt=prompt,
        question=question,
        reformat_model=reformat_model,
    )
    return convert_string_to_list(raw_response)

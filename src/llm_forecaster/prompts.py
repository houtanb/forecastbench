"""Forecasting prompt text."""

# The following zero-shot prompts mainly come from:
# "Approaching Human-Level Forecasting with Language Models" by Halawi et al. (2024)
# Some are modified versions in order to adapt to our needs
# https://arxiv.org/pdf/2402.18563v1
ZERO_SHOT_MARKET_PROMPT = """
You are an expert superforecaster, familiar with the work of Tetlock and others. Make a prediction of the probability that the question will be resolved as true. You MUST give a probability estimate between 0 and 1 UNDER ALL CIRCUMSTANCES. If for some reason you can't answer, pick the base rate, but return a number between 0 and 1.

Question:
{question}

Question Background:
{background}

Resolution Criteria:
{resolution_criteria}

Today's Date: {today_date}

Question resolution date: {resolution_date}

Output your answer (a number between 0 and 1) with an asterisk at the beginning and end of the decimal.
Do not output anything else.
Answer: {{ Insert answer here }}
"""  # noqa: B950

ZERO_SHOT_MARKET_WITH_FREEZE_VALUE_PROMPT = """
You are an expert superforecaster, familiar with the work of Tetlock and others. Make a prediction of the probability that the question will be resolved as true. You MUST give a probability estimate between 0 and 1 UNDER ALL CIRCUMSTANCES. If for some reason you can't answer, pick the base rate, but return a number between 0 and 1.

Question:
{question}

Question Background:
{background}

Resolution Criteria:
{resolution_criteria}

Market value on {freeze_datetime}:
{freeze_datetime_value}

Today's Date: {today_date}

Question resolution date: {resolution_date}

Output your answer (a number between 0 and 1) with an asterisk at the beginning and end of the decimal.
Do not output anything else.
Answer: {{ Insert answer here }}
"""  # noqa: B950


ZERO_SHOT_DATASET_PROMPT = """
You are an expert superforecaster, familiar with the work of Tetlock and others. Make a prediction of the probability that the question will be resolved as true. You MUST give a probability estimate between 0 and 1 UNDER ALL CIRCUMSTANCES. If for some reason you can't answer, pick the base rate, but return a number between 0 and 1.

You’re going to predict the probability of the following potential outcome “at each of the resolution dates”.

Question:
{question}

Question Background:
{background}

Resolution Criteria:
{resolution_criteria}

Current value on {freeze_datetime}:
{freeze_datetime_value}

Value Explanation:
{freeze_datetime_value_explanation}

Today's Date: {today_date}

Question resolution dates: {list_of_resolution_dates}

Output your answer (a number between 0 and 1) with an asterisk at the beginning and end of the decimal. (For example, if there are n resolution dates, you would output different *p* for each resolution date) Do not output anything else.
Answer: {{ Insert answer here }}

"""  # noqa: B950


REFORMAT_PROMPT = """
User prompt:
{user_prompt}

Model Response:
{model_response}

Please determine the model's final probabilistic forecasts for all {n_horizons} resolution dates mentioned in the user's prompt. If such forecasts are not present, you should provide a probabilistic forecast for all {n_horizons} resolution dates mentioned.

Please output the probabilistic forecasts as a Python list, e.g., [prob1, prob2, ...] DO NOT OUTPUT ANYTHING ELSE. PLEASE ONLY OUTPUT [prob1, prob2, ...], if there is only one probabilistic prediction, output [prob1, prob2, ...] with the same probability.
"""  # noqa: B950

REFORMAT_PROMPT_2 = """
User prompt:
{user_prompt}

Model Response:
{model_response}

Please determine the model's final probabilistic forecasts for all {n_horizons} resolution dates mentioned in the user's prompt. If any forecasts are missing, provide a probabilistic forecast for each of the resolution dates.

Output the probabilistic forecasts as a Python list, e.g., [prob1, prob2, ...]. DO NOT OUTPUT ANYTHING ELSE. If there is only one probabilistic prediction, replicate it for all dates.
"""  # noqa: B950

REFORMAT_SINGLE_PROMPT = """

{response}

***********************************
Instructions:
The text above is an answer from a large language model (LLM) that includes reasoning and a probability estimate in response to a question.
Task: Extract the numerical probability given by the LLM. The probability should be a decimal value between 0 and 1.
If the LLM's response does not contain a probability estimate, return 'N/A'.

Output Requirement: Provide only a decimal value between 0 and 1 representing the probability, or 'N/A' if no probability is mentioned.
"""  # noqa: B950

REFORMAT_SINGLE_PROMPT_2 = """
{response}

***********************************
Instructions:
The text above is a response from a large language model (LLM) that includes both reasoning and a probability estimate in answer to a question.

Task: Extract the numerical probability mentioned by the LLM. This probability should be a decimal value between 0 and 1.

Output Requirement: Provide only the extracted probability as a decimal value between 0 and 1. If the response does not contain a probability estimate, return 'N/A'.
"""  # noqa: B950

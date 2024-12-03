"""Constants."""

from collections import defaultdict
from datetime import datetime

BENCHMARK_NAME = "ForecastBench"

BENCHMARK_START_YEAR = 2024
BENCHMARK_START_MONTH = 5
BENCHMARK_START_DAY = 1
BENCHMARK_START_DATE = f"{BENCHMARK_START_YEAR}-{BENCHMARK_START_MONTH}-{BENCHMARK_START_DAY}"
BENCHMARK_START_DATE_DATETIME = datetime.strptime(BENCHMARK_START_DATE, "%Y-%m-%d")
BENCHMARK_START_DATE_DATETIME_DATE = BENCHMARK_START_DATE_DATETIME.date()

BENCHMARK_TOURNAMENT_START_YEAR = 2024
BENCHMARK_TOURNAMENT_START_MONTH = 7
BENCHMARK_TOURNAMENT_START_DAY = 21
BENCHMARK_TOURNAMENT_START_DATE = (
    f"{BENCHMARK_TOURNAMENT_START_YEAR}-"
    f"{BENCHMARK_TOURNAMENT_START_MONTH}-"
    f"{BENCHMARK_TOURNAMENT_START_DAY}"
)
BENCHMARK_TOURNAMENT_START_DATE_DATETIME = datetime.strptime(
    BENCHMARK_TOURNAMENT_START_DATE, "%Y-%m-%d"
)
BENCHMARK_TOURNAMENT_START_DATE_DATETIME_DATE = BENCHMARK_TOURNAMENT_START_DATE_DATETIME.date()

parsed_date = datetime.strptime(BENCHMARK_START_DATE + " 00:00", "%Y-%m-%d %H:%M")
BENCHMARK_START_DATE_EPOCHTIME = int(parsed_date.timestamp())
BENCHMARK_START_DATE_EPOCHTIME_MS = BENCHMARK_START_DATE_EPOCHTIME * 1000

FORECAST_HORIZONS_IN_DAYS = [
    7,  # 1 week
    30,  # 1 month
    90,  # 3 months
    180,  # 6 months
    365,  # 1 year
    1095,  # 3 years
    1825,  # 5 years
    3650,  # 10 years
]

QUESTION_FILE_COLUMN_DTYPE = {
    "id": str,
    "question": str,
    "background": str,
    "url": str,
    "resolved": bool,
    "forecast_horizons": object,
    "freeze_datetime_value": str,
    "freeze_datetime_value_explanation": str,
    "market_info_resolution_criteria": str,
    "market_info_open_datetime": str,
    "market_info_close_datetime": str,
    "market_info_resolution_datetime": str,
}
QUESTION_FILE_COLUMNS = list(QUESTION_FILE_COLUMN_DTYPE.keys())

RESOLUTION_FILE_COLUMN_DTYPE = {
    "id": str,
    "date": str,
}

# value is not included in dytpe because it's of type ANY
RESOLUTION_FILE_COLUMNS = list(RESOLUTION_FILE_COLUMN_DTYPE.keys()) + ["value"]

META_DATA_FILE_COLUMN_DTYPE = {
    "source": str,
    "id": str,
    "category": str,
    "valid_question": bool,
}
META_DATA_FILE_COLUMNS = list(META_DATA_FILE_COLUMN_DTYPE.keys())
META_DATA_FILENAME = "question_metadata.jsonl"

QUESTION_CATEGORIES = [
    "Science & Tech",
    "Healthcare & Biology",
    "Economics & Business",
    "Environment & Energy",
    "Politics & Governance",
    "Arts & Recreation",
    "Security & Defense",
    "Sports",
    "Other",
]

OAI_SOURCE = "OAI"
ANTHROPIC_SOURCE = "ANTHROPIC"
TOGETHER_AI_SOURCE = "TOGETHER"
GOOGLE_SOURCE = "GOOGLE"
MISTRAL_SOURCE = "MISTRAL"

ZERO_SHOT_AND_SCRATCHPAD_MODELS = {
    # oai context window from: https://platform.openai.com/docs/models/
    "gpt_3p5_turbo_0125": {
        "source": OAI_SOURCE,
        "full_name": "gpt-3.5-turbo-0125",
        "token_limit": 16385,
    },
    "gpt_4": {
        "source": OAI_SOURCE,
        "full_name": "gpt-4-0613",
        "token_limit": 8192,
    },
    "gpt_4_turbo_0409": {
        "source": OAI_SOURCE,
        "full_name": "gpt-4-turbo-2024-04-09",
        "token_limit": 128000,
    },
    "gpt_4o_2024-05-13": {
        "source": OAI_SOURCE,
        "full_name": "gpt-4o-2024-05-13",
        "token_limit": 128000,
    },
    "gpt_4o_2024-08-06": {
        "source": OAI_SOURCE,
        "full_name": "gpt-4o-2024-08-06",
        "token_limit": 128000,
    },
    "o1-preview-2024-09-12": {
        "source": OAI_SOURCE,
        "full_name": "o1-preview-2024-09-12",
        "token_limit": 128000,
    },
    "o1-mini-2024-09-12": {
        "source": OAI_SOURCE,
        "full_name": "o1-mini-2024-09-12",
        "token_limit": 128000,
    },
    # together.ai context window from: https://docs.together.ai/docs/serverless-models
    "llama-3p1-405B-Instruct-Turbo": {
        "source": TOGETHER_AI_SOURCE,
        "full_name": "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
        "token_limit": 130815,
    },
    "llama-3p2-3B-Instruct-Turbo": {
        "source": TOGETHER_AI_SOURCE,
        "full_name": "meta-llama/Llama-3.2-3B-Instruct-Turbo",
        "token_limit": 131072,
    },
    "mistral_8x7b_instruct": {
        "source": TOGETHER_AI_SOURCE,
        "full_name": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "token_limit": 32768,
    },
    "mistral_8x22b_instruct": {
        "source": TOGETHER_AI_SOURCE,
        "full_name": "mistralai/Mixtral-8x22B-Instruct-v0.1",
        "token_limit": 65536,
    },
    "qwen_2p5_72b": {
        "source": TOGETHER_AI_SOURCE,
        "full_name": "Qwen/Qwen2.5-72B-Instruct-Turbo",
        "token_limit": 32768,
    },
    # anthropic context window from: https://docs.anthropic.com/en/docs/about-claude/models
    "claude_2p1": {
        "source": ANTHROPIC_SOURCE,
        "full_name": "claude-2.1",
        "token_limit": 200000,
    },
    "claude_3_opus": {
        "source": ANTHROPIC_SOURCE,
        "full_name": "claude-3-opus-20240229",
        "token_limit": 200000,
    },
    "claude_3_haiku": {
        "source": ANTHROPIC_SOURCE,
        "full_name": "claude-3-haiku-20240307",
        "token_limit": 200000,
    },
    "claude_3p5_sonnet": {
        "source": ANTHROPIC_SOURCE,
        "full_name": "claude-3-5-sonnet-20240620",
        "token_limit": 200000,
    },
    "claude-3-5-sonnet-20241022": {
        "source": ANTHROPIC_SOURCE,
        "full_name": "claude-3-5-sonnet-20241022",
        "token_limit": 200000,
    },
    # google context window from: https://ai.google.dev/gemini-api/docs/models/gemini
    "gemini_1p5_flash": {
        "source": GOOGLE_SOURCE,
        "full_name": "gemini-1.5-flash",
        "token_limit": 1048576,
    },
    "gemini_1p5_pro": {
        "source": GOOGLE_SOURCE,
        "full_name": "gemini-1.5-pro",
        "token_limit": 2097152,
    },
    "gemini-exp-1121": {
        "source": GOOGLE_SOURCE,
        "full_name": "gemini-exp-1121",
        "token_limit": 32000,
        # not listed on Gemeni site, but in this Tweet:
        # https://twitter.com/officiallogank/status/1860106796247216174
    },
}

MODEL_TOKEN_LIMITS = dict()
MODEL_NAME_TO_SOURCE = dict()
ZERO_SHOT_AND_SCRATCHPAD_MODELS_BY_SOURCE = defaultdict(dict)
for key, value in ZERO_SHOT_AND_SCRATCHPAD_MODELS.items():
    MODEL_TOKEN_LIMITS[value["full_name"]] = value["token_limit"]
    MODEL_NAME_TO_SOURCE[value["full_name"]] = value["source"]
    ZERO_SHOT_AND_SCRATCHPAD_MODELS_BY_SOURCE[value["source"]][key] = value

MODEL_TOKEN_LIMITS["gpt-4o-mini"] = 128000
MODEL_NAME_TO_SOURCE["gpt-4o-mini"] = OAI_SOURCE

# remove models with less than ~17000 token limits
SUPERFORECASTER_WITH_NEWS_MODELS = SCRATCHPAD_WITH_NEWS_MODELS = {
    "gpt_4_turbo_0409": {"source": OAI_SOURCE, "full_name": "gpt-4-turbo-2024-04-09"},
    "gpt_4o": {"source": OAI_SOURCE, "full_name": "gpt-4o"},
    "mistral_8x7b_instruct": {
        "source": TOGETHER_AI_SOURCE,
        "full_name": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    },
    "mistral_8x22b_instruct": {
        "source": TOGETHER_AI_SOURCE,
        "full_name": "mistralai/Mixtral-8x22B-Instruct-v0.1",
    },
    "mistral_large": {
        "source": TOGETHER_AI_SOURCE,
        "full_name": "mistral-large-latest",
    },
    "qwen_1p5_110b": {
        "source": TOGETHER_AI_SOURCE,
        "full_name": "Qwen/Qwen1.5-110B-Chat",
    },
    "claude_2p1": {"source": ANTHROPIC_SOURCE, "full_name": "claude-2.1"},
    "claude_3_opus": {"source": ANTHROPIC_SOURCE, "full_name": "claude-3-opus-20240229"},
    "claude_3_haiku": {"source": ANTHROPIC_SOURCE, "full_name": "claude-3-haiku-20240307"},
    "claude_3p5_sonnet": {"source": ANTHROPIC_SOURCE, "full_name": "claude-3-5-sonnet-20240620"},
    "gemini_1p5_flash": {"source": GOOGLE_SOURCE, "full_name": "gemini-1.5-flash"},
    "gemini_1p5_pro": {"source": GOOGLE_SOURCE, "full_name": "gemini-1.5-pro"},
}

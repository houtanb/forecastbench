"""Constants."""

import re
from datetime import datetime, timedelta
from enum import Enum

BENCHMARK_NAME = "ForecastBench"
BENCHMARK_EMAIL = "forecastbench@forecastingresearch.org"
BENCHMARK_URL = "https://www.forecastbench.org"
BENCHMARK_USER_AGENT = f"{BENCHMARK_NAME}Bot/0.0 ({BENCHMARK_URL}; {BENCHMARK_EMAIL})"

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

QUESTION_BANK_DATA_STORAGE_START_DATETIME = BENCHMARK_START_DATE_DATETIME - timedelta(days=360)
QUESTION_BANK_DATA_STORAGE_START_DATE = QUESTION_BANK_DATA_STORAGE_START_DATETIME.date()

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


PROMPT_TYPES = [
    "zero_shot",
]


class RunMode(str, Enum):
    """Run modes for code execution.

    - TEST: Test/dev runs; use to reduce costs when running models.
    - PROD: Full production runs; execute all models with full question set.

    Construction is case-insensitive (e.g., RunMode("teST") --> RunMode.TEST).
    Invalid values raise ValueError.
    """

    TEST = "TEST"
    PROD = "PROD"

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            return cls.__members__.get(value.upper())
        return None

    @classmethod
    def from_string(cls, value: str | None) -> "RunMode":
        """Parse a run mode string, defaulting to TEST for missing or invalid values."""
        try:
            return cls(value)
        except ValueError:
            return cls.TEST

    @property
    def is_test(self) -> bool:
        """Return whether this mode should run test-sized workloads."""
        return self is RunMode.TEST

    @property
    def is_prod(self) -> bool:
        """Return whether this mode should run production workloads."""
        return self is RunMode.PROD

    @property
    def forecast_file_prefix(self) -> str:
        """Return the forecast filename prefix for this run mode."""
        if self.is_test:
            return f"{self.value}."
        return ""


ANTHROPIC_ORG = "Anthropic"
DEEPSEEK_ORG = "DeepSeek"
MOONSHOT_ORG = "Moonshot"
MOONSHOT_AI_ORG = "Moonshot AI"
MINIMAX_ORG = "Minimax"
MINIMAX_CANONICAL_ORG = "MiniMax"
GOOGLE_ORG = "Google"
META_ORG = "Meta"
MISTRAL_ORG = "Mistral AI"
MISTRAL_ORG_1 = "Mistral"  # for some forecasts, "Mistral AI" was called "Mistral"
OAI_ORG = "OpenAI"
QWEN_ORG = "Qwen"
XAI_ORG = "xAI"
ZAI_ORG = "Z.ai"

EXTERNAL_TOURNAMENT_MODELS_TO_LOGO = {
    "Cassi-AI": "cassi-ai.png",
    "FractalAIResearch": "fractal-ai.png",
    "Lightning Rod Labs": "lightningrod.jpg",
    "LightningRodLabs": "lightningrod.jpg",
    "Mantic": "mantic.jpg",
    "Stochastic Radiant": "stochastic-radiant.svg",
    "Google DeepMind": "deepmind.svg",
    "limeforecast": "limeforecast.png",
    "Voicetree": "voicetree.png",
    "Artificial Judgement": "artificial-judgement.png",
}

ORG_TO_LOGO = {
    BENCHMARK_NAME: "fri.svg",
    ANTHROPIC_ORG: "anthropic.svg",
    DEEPSEEK_ORG: "deepseek.svg",
    MOONSHOT_ORG: "moonshot.svg",
    MOONSHOT_AI_ORG: "moonshot.svg",
    MINIMAX_ORG: "minimax.svg",
    MINIMAX_CANONICAL_ORG: "minimax.svg",
    GOOGLE_ORG: "deepmind.svg",
    META_ORG: "meta.svg",
    MISTRAL_ORG: "mistral.svg",
    MISTRAL_ORG_1: "mistral.svg",
    OAI_ORG: "openai.svg",
    QWEN_ORG: "qwen.svg",
    XAI_ORG: "xai.svg",
    ZAI_ORG: "zai.svg",
}
_ANON_TEAM_RE = re.compile(r"^anonymous\s+(\d+)$", re.IGNORECASE)


def get_org_logo(org: str) -> str:
    """Get the logo filename associated with an organization.

    The function first checks internal benchmark organizations, then external
    tournament participants, and finally handles anonymous teams. If no match
    is found, it returns a default placeholder logo.

    Args:
        org (str): The name of the organization or team.

    Returns:
        str: The corresponding logo filename, or "default.svg" if no logo
             mapping is found.
    """
    if org in ORG_TO_LOGO.keys():
        return ORG_TO_LOGO[org]

    if org in EXTERNAL_TOURNAMENT_MODELS_TO_LOGO.keys():
        return EXTERNAL_TOURNAMENT_MODELS_TO_LOGO[org]

    match = _ANON_TEAM_RE.match(org.strip())
    if match:
        num = int(match.group(1))
        if num >= 1:
            return f"anonymous_{num}.svg"

    return "default.svg"

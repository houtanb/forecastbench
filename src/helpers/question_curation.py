"""Info relevant to selecting questions."""

import os
from datetime import timedelta

from . import acled, dates, infer, manifold, metaculus

FREEZE_NUM_LLM_QUESTIONS = 1000
FREEZE_NUM_HUMAN_QUESTIONS = 200

# Assumed in the code
assert FREEZE_NUM_LLM_QUESTIONS > FREEZE_NUM_HUMAN_QUESTIONS

FREEZE_QUESTION_SOURCES = {
    "manifold": {
        "name": "Manifold",
        "source_intro": manifold.SOURCE_INTRO,
        "resolution_criteria": manifold.RESOLUTION_CRITERIA,
    },
    "metaculus": {
        "name": "Metaculus",
        "source_intro": metaculus.SOURCE_INTRO,
        "resolution_criteria": metaculus.RESOLUTION_CRITERIA,
    },
    "acled": {
        "name": "ACLED",
        "source_intro": acled.SOURCE_INTRO,
        "resolution_criteria": acled.RESOLUTION_CRITERIA,
    },
    "infer": {
        "name": "INFER",
        "source_intro": infer.SOURCE_INTRO,
        "resolution_criteria": infer.RESOLUTION_CRITERIA,
    },
}

DATA_SOURCES = [
    "acled",
]

FREEZE_WINDOW_IN_DAYS = 7

FREEZE_DATETIME = os.environ.get("FREEZE_DATETIME", dates.get_datetime_today()).replace(
    hour=0, minute=0, second=0, microsecond=0
)

FORECAST_DATETIME = FREEZE_DATETIME + timedelta(days=FREEZE_WINDOW_IN_DAYS)

FORECAST_DATE = FORECAST_DATETIME.date()

COMBINATION_PROMPT = (
    "We are presenting you with two probability questions. Please predict the probability that both "
    "will happen, that one will happen but not the other, and that neither will happen. In other "
    "words, for each resolution date please provide 4 predictions."
)
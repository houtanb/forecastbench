"""ACLED-specific variables."""

import hashlib
import json
import os
import sys
from datetime import timedelta
from enum import Enum

import numpy as np
import pandas as pd

from . import data_utils, env

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))  # noqa: E402
from utils import gcp  # noqa: E402

source = "acled"
hash_mapping = {}
hash_filename = "hash_mapping.json"
local_hash_filename = f"/tmp/{hash_filename}"


def id_hash(d: dict) -> str:
    """Encode ACLED Ids."""
    global hash_mapping
    dictionary_json = json.dumps(d, sort_keys=True)
    hash_key = hashlib.sha256(dictionary_json.encode()).hexdigest()
    hash_mapping[hash_key] = d
    return hash_key


def id_unhash(hash_key: str) -> tuple:
    """Decode ACLED Ids."""
    return hash_mapping[hash_key] if hash_key in hash_mapping else None


def populate_hash_mapping():
    """Download the hash_mapping from storage and load into global."""
    global hash_mapping
    remote_filename = f"{source}/{hash_filename}"
    gcp.storage.download_no_error_message_on_404(
        bucket_name=env.QUESTION_BANK_BUCKET,
        filename=remote_filename,
        local_filename=local_hash_filename,
    )
    if os.path.getsize(local_hash_filename) > 0:
        with open(local_hash_filename, "r") as file:
            hash_mapping = json.load(file)


def upload_hash_mapping():
    """Write and upload the hash_mapping to storage from global."""
    with open(local_hash_filename, "w") as file:
        json.dump(hash_mapping, file, indent=4)

    gcp.storage.upload(
        bucket_name=env.QUESTION_BANK_BUCKET,
        local_filename=local_hash_filename,
        destination_folder=source,
    )


FETCH_COLUMN_DTYPE = {
    "event_id_cnty": str,
    "event_date": str,
    "iso": int,
    "region": str,
    "country": str,
    "admin1": str,
    "event_type": str,
    "fatalities": int,
    "timestamp": str,
}
FETCH_COLUMNS = list(FETCH_COLUMN_DTYPE.keys())

BACKGROUND = """
ACLED classifies events into six distinct categories:

1. Battles: violent interactions between two organized armed groups at a particular time and
   location;
2. Protests: in-person public demonstrations of three or more participants in which the participants
   do not engage in violence, though violence may be used against them;
3. Riots: violent events where demonstrators or mobs of three or more engage in violent or
   destructive acts, including but not limited to physical fights, rock throwing, property
   destruction, etc.;
4. Explosions/Remote violence: incidents in which one side uses weapon types that, by their nature,
   are at range and widely destructive;
5. Violence against civilians: violent events where an organized armed group inflicts violence upon
   unarmed non-combatants; and
6. Strategic developments: contextually important information regarding incidents and activities of
   groups that are not recorded as any of the other event types, yet may trigger future events or
   contribute to political dynamics within and across states.

Detailed information about the categories can be found at:
https://acleddata.com/knowledge-base/codebook/#acled-events
"""

SOURCE_INTRO = (
    "The Armed Conflict Location & Event Data Project (ACLED) collects real-time data on the "
    "locations, dates, actors, fatalities, and types of all reported political violence and "
    "protest events around the world. You're going to predict how questions based on this data "
    "will resolve."
)

RESOLUTION_CRITERIA = (
    "Resolves to the value calculated from the ACLED dataset once the data is published."
)


def download_dff_and_prepare_dfr() -> tuple:
    """Prepare ACLED data for resolution."""
    filenames = data_utils.generate_filenames(source=source)
    df = data_utils.download_and_read(
        filename=filenames["jsonl_fetch"],
        local_filename=filenames["local_fetch"],
        df_tmp=pd.DataFrame(columns=FETCH_COLUMNS),
        dtype=FETCH_COLUMN_DTYPE,
    )

    countries = df["country"].unique()
    event_types_acled = df["event_type"].unique()
    event_types = list(event_types_acled) + ["fatalities"]

    df = df[["country", "event_date", "event_type", "fatalities"]].copy()
    df["event_date"] = pd.to_datetime(df["event_date"])
    return (
        (
            pd.get_dummies(df, columns=["event_type"], prefix="", prefix_sep="")
            .groupby(["country", "event_date"])
            .sum()
            .reset_index()
        ),
        countries,
        event_types,
    )


def make_resolution_df():
    """Prepare data for resolution."""
    dfr, _, _ = download_dff_and_prepare_dfr()
    return dfr


class QuestionType(Enum):
    """Types of questions.

    These will determine how a given question is resolved.
    """

    N_30_DAYS_GT_30_DAY_AVG_OVER_PAST_360_DAYS = 0
    N_30_DAYS_X_10_GT_30_DAY_AVG_OVER_PAST_360_DAYS_PLUS_1 = 1


def get_forecast(comparison_value, dfr, country, col, ref_date):
    """Retrun the LHS of the comparison for the question.

    Used for the naive forecaster.
    """
    dfr["country"] = country
    dfr[col] = dfr["yhat"]
    dfr["event_date"] = dfr["ds"]
    start_date = ref_date - timedelta(days=30)
    dfr = dfr[
        (dfr["event_date"].dt.date >= start_date) & (dfr["event_date"].dt.date < ref_date)
    ].reset_index(drop=True)
    simulated_values = []
    dates = [pd.to_datetime(ref_date) - timedelta(days=i) for i in range(len(dfr))]
    for _ in range(1000):
        draws = np.random.normal(dfr[col], (dfr["yhat_upper"] - dfr["yhat_lower"]) / (2 * 1.28))
        df_draws = pd.DataFrame(
            {
                "country": country,
                "event_date": dates,
                col: draws,
            }
        )
        simulated_values.append(
            sum_over_past_30_days(
                dfr=df_draws,
                country=country,
                col=col,
                ref_date=ref_date,
            )
        )

    return float(np.mean([value > comparison_value for value in simulated_values]))


def get_base_comparison_value(key, dfr, country, col, ref_date):
    """Get the base comparison value given the question type.

    Used for the naive forecaster and resolve.
    """
    if key == "last30Days.gt.30DayAvgOverPast360Days":
        return thirty_day_avg_over_past_360_days(
            dfr=dfr, country=country, col=col, ref_date=ref_date
        )
    elif key == "last30DaysTimes10.gt.30DayAvgOverPast360DaysPlus1":
        return 10 * thirty_day_avg_over_past_360_days_plus_1(
            dfr=dfr, country=country, col=col, ref_date=ref_date
        )
    raise ValueError("Invalid key.")


def resolve(
    key,
    dfr,
    country,
    event_type,
    forecast_due_date,
    resolution_date,
):
    """Resolve given the QuestionType."""
    lhs = sum_over_past_30_days(
        dfr=dfr,
        country=country,
        col=event_type,
        ref_date=resolution_date,
    )
    rhs = get_base_comparison_value(
        key=key,
        dfr=dfr,
        country=country,
        col=event_type,
        ref_date=forecast_due_date,
    )
    return int(lhs > rhs)


def get_freeze_value(key, dfr, country, event_type, today):
    """Return the freeze value given the key."""
    if key == "last30Days.gt.30DayAvgOverPast360Days":
        return thirty_day_avg_over_past_360_days(dfr, country, event_type, today)

    if key == "last30DaysTimes10.gt.30DayAvgOverPast360DaysPlus1":
        return thirty_day_avg_over_past_360_days_plus_1(dfr, country, event_type, today)

    raise Exception("Invalid key.")


def sum_over_past_30_days(dfr, country, col, ref_date):
    """Sum over the 30 days before the ref_date."""
    dfc = dfr[dfr["country"] == country].copy()
    if dfc.empty:
        return 0

    start_date = ref_date - timedelta(days=30)
    dfc = dfc[(dfc["event_date"].dt.date >= start_date) & (dfc["event_date"].dt.date < ref_date)]
    return dfc[col].sum() if not dfc.empty else 0


def thirty_day_avg_over_past_360_days(dfr, country, col, ref_date):
    """Get the 30 day average over the 360 days before the ref_date."""
    dfc = dfr[dfr["country"] == country].copy()
    if dfc.empty:
        return 0

    start_date = ref_date - timedelta(days=360)
    dfc = dfc[(dfc["event_date"].dt.date >= start_date) & (dfc["event_date"].dt.date < ref_date)]
    return dfc[col].sum() / 12 if not dfc.empty else 0


def thirty_day_avg_over_past_360_days_plus_1(dfr, country, col, ref_date):
    """Get 1 plus the 30 day average over the 360 days before the ref_date."""
    return 1 + thirty_day_avg_over_past_360_days(dfr, country, col, ref_date)


QUESTIONS = {
    "last30Days.gt.30DayAvgOverPast360Days": {
        "question_type": QuestionType.N_30_DAYS_GT_30_DAY_AVG_OVER_PAST_360_DAYS,
        "question": (
            (
                "Will there be more {event_type} in {country} for the 30 days before "
                "{resolution_date} compared to the 30-day average of {event_type} over the 360 "
                "days preceding {forecast_due_date}?"
                "\n\n"
                "e.g. If the forecast due date is 2024-01-01 and we have the following data:\n"
                "Date,{event_type}\n"
                "2023-11-11,1\n"
                "2023-10-10,2\n"
                "to calculate the 30-day average of {event_type} over the preceding 360 "
                "days, we’d have: (1+2)/12=0.25.\n\n"
                "In this example, for the question to resolve positively, 1 or more "
                "{event_type} would need to occur in the 30 days leading up to the resolution."
            ),
            ("event_type", "country"),
        ),
        "freeze_datetime_value_explanation": (
            (
                "The 30-day average of {event_type} over the past 360 days in {country}. "
                "This reference value will potentially change as ACLED updates its dataset."
            ),
            ("event_type", "country"),
        ),
    },
    "last30DaysTimes10.gt.30DayAvgOverPast360DaysPlus1": {
        "question_type": QuestionType.N_30_DAYS_X_10_GT_30_DAY_AVG_OVER_PAST_360_DAYS_PLUS_1,
        "question": (
            (
                "Will there be more than ten times as many {event_type} in {country} for the 30 "
                "days before {resolution_date} compared to one plus the 30-day average of "
                "{event_type} over the 360 days preceding {forecast_due_date}?"
                "\n\n"
                "e.g. If the forecast due date is 2024-01-01 and we have the following data:\n"
                "Date,{event_type}\n"
                "2023-11-11,1\n"
                "2023-10-10,2\n"
                "to calculate one plus the 30-day average of {event_type} over the preceding 360 "
                "days, we’d have: 1+(1+2)/12=1.25.\n\n"
                "In this example, for the question to resolve positively, 13 (10 x 1.25) or more "
                "{event_type} would need to occur in the 30 days leading up to the resolution."
            ),
            ("event_type", "country"),
        ),
        "freeze_datetime_value_explanation": (
            (
                "One plus the 30-day average of {event_type} over the past 360 days in {country}. "
                "This reference value will potentially change as ACLED updates its dataset."
            ),
            ("event_type", "country"),
        ),
    },
}

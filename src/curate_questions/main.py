"""Freeze forecasting questions."""

import json
import logging
import math
import os
import random
import sys
from copy import deepcopy
from datetime import datetime
from itertools import combinations

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from helpers import (  # noqa: E402
    acled,
    constants,
    data_utils,
    decorator,
    env,
    question_prompts,
)

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from utils import gcp  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Combination questions should comprise 50% of questions for each data source.
COMBO_PCT = 0.5
assert 0 <= COMBO_PCT <= 1


def process_questions(questions, to_questions, single_generation_func, combo_generation_func=None):
    """Sample from `questions` to get the number of questions needed.

    This function works for both the LLM question set and the human forecaster question set.
    """
    num_found = 0
    processed_questions = deepcopy(questions)
    pct_of_combo_questions = 0 if combo_generation_func is None else COMBO_PCT
    for source, values in processed_questions.items():
        num_single = math.ceil(
            to_questions[source]["num_questions_to_sample"] * (1 - pct_of_combo_questions)
        )
        values["dfq"] = single_generation_func(values, num_single)
        if combo_generation_func is not None:
            num_combo = math.floor(
                to_questions[source]["num_questions_to_sample"] * pct_of_combo_questions
            )
            values["combos"] = combo_generation_func(values["dfq"], num_combo)
        else:
            num_combo = 0
            values["combos"] = []
        num_found += len(values["dfq"]) + len(values["combos"])
    logger.info(f"Found {num_found} questions.")
    return processed_questions


def sample_single_questions(values, n_single):
    """Generate single questions for the LLM and Human question sets.

    Sample evenly across categories.
    """
    dfq = values["dfq"]
    allocation, underrepresented_categories = allocate_across_categories(
        num_questions=n_single, dfq=dfq
    )

    df = pd.DataFrame()
    for key, value in allocation.items():
        df_tmp = dfq[dfq["category"] == key].sample(value)
        df = pd.concat([df, df_tmp], ignore_index=True)

    df["underrepresented_category"] = df["category"].apply(
        lambda x: True if x in underrepresented_categories else False
    )
    return df


def sample_combo_questions(df, N):
    """Generate `N` combinations of the indices in `df`."""
    indices = df.index.tolist()
    all_possible_pairs = list(combinations(indices, 2))
    random.shuffle(all_possible_pairs)

    if len(all_possible_pairs) < N:
        logger.warning(
            f"Not enough combinations available: Requested {N}, but only {len(all_possible_pairs)} "
            "are possible."
        )
        return all_possible_pairs

    underrepresented_indices = df[df["underrepresented_category"]].index.tolist()
    all_underrepresented_pairs = [
        (i, j)
        for i, j in all_possible_pairs
        if i in underrepresented_indices or j in underrepresented_indices
    ]

    # Remove pairs with duplicates of underrepresented indices
    seen = set()
    underrepresented_pairs = []
    for pair in all_underrepresented_pairs:
        for i in underrepresented_indices:
            if i in pair and i not in seen:
                underrepresented_pairs.append(pair)
                seen.add(i)
                break

    if N <= len(underrepresented_pairs):
        logger.warning("Only returning combos that contain at least one underrepresented index.")
        return underrepresented_pairs[:N]

    # Sample all remaining combo questions
    N -= len(underrepresented_pairs)
    all_possible_pairs = [tup for tup in all_possible_pairs if tup not in underrepresented_pairs]
    all_possible_pairs_indices = np.random.choice(len(all_possible_pairs), size=N, replace=False)
    return underrepresented_pairs + [all_possible_pairs[i] for i in all_possible_pairs_indices]


def allocate_evenly(data: dict, n: int):
    """Allocates the number of questions evenly given `data`.

    `n` is the total number of items we want to allocate.

    `data` is a dict that has the items to allocate across as keys and the number of possible items
    to allocate as values. So, if we're allocating across sources, it would look like:
    {'source1': 30, 'source2': 50, ...} and if we're allocating across categories within a source
    it would look like: {'category1': 30, 'category2': 50, ...}.

    The function returns a dict with the same keys as `data` but with the allocation. The allocated
    values are guaranteed to be <= the original values provided in `data`.

    If `sum(data.values()) <= n` it returns `data`.
    """
    if sum(data.values()) <= n:
        return data, sorted([key for key, value in data.items()])

    # initial allocation
    allocation = {key: min(n // len(data), value) for key, value in data.items()}
    allocated_num = sum(allocation.values())
    underrepresented_items = sorted(
        [key for key, value in data.items() if allocation[key] == value]
    )

    while allocated_num < n:
        remaining = n - allocated_num
        under_allocated = {
            key: value - allocation[key] for key, value in data.items() if allocation[key] < value
        }

        if not under_allocated:
            # Break if nothing more to allocate
            break

        # Amount to add in this iteration
        to_allocate = max(remaining // len(under_allocated), 1)
        for key in under_allocated:
            if under_allocated[key] > 0:
                add_amount = min(to_allocate, under_allocated[key], remaining)
                allocation[key] += add_amount
                remaining -= add_amount
                if remaining <= 0:
                    break
        allocated_num = sum(allocation.values())

    num_allocated = sum(allocation.values())
    if num_allocated != n:
        logger.error(f"*** Problem allocating evenly... Allocated {num_allocated}/{n}. ***")
    else:
        logger.info(f"Successfully allocated {num_allocated}/{n}.")
    return allocation, underrepresented_items


def allocate_across_categories(num_questions, dfq):
    """Allocates the number of questions evenly among categories."""
    categories = dfq["category"].unique()
    data = {category: sum(dfq["category"] == category) for category in categories}
    return allocate_evenly(data=data, n=num_questions)


def allocate_across_sources(for_humans, questions):
    """Allocates the number of questions evenly among sources."""
    num_questions = (
        constants.FREEZE_NUM_HUMAN_QUESTIONS if for_humans else constants.FREEZE_NUM_LLM_QUESTIONS
    )
    sources = deepcopy(questions)
    col = "num_single_questions_available" if for_humans else "num_questions_incl_combo_available"
    data = {key: source[col] for key, source in sources.items()}

    allocation, _ = allocate_evenly(data=data, n=num_questions)

    for source in sources:
        sources[source]["num_questions_to_sample"] = allocation[source]

    num_allocated = sum(allocation.values())
    if num_allocated != num_questions:
        logger.error("*** Problem allocating questions. ***")
    logger.info(f"Allocated {num_allocated}/{num_questions}.")
    return sources


def write_questions(questions, filename):
    """Write single and combo questions to file and upload."""

    def get_forecast_horizon(source, combo_rows):
        if source not in constants.DATA_SOURCES:
            # We don't ask for forecasts at different horizons for market-based questions.
            return "N/A"
        fh1 = combo_rows.at[0, "forecast_horizons"]
        fh2 = combo_rows.at[1, "forecast_horizons"]
        return sorted(set(fh1) | set(fh2))

    df = pd.DataFrame()
    for source, values in tqdm(questions.items(), "Writing questions"):
        df_source = values["dfq"]
        df = pd.concat([df, df_source], ignore_index=True)
        for q1, q2 in values["combos"]:
            combo_rows = df_source.loc[[q1, q2]].reset_index(drop=True)
            df_combo = pd.DataFrame(
                [
                    {
                        "id": combo_rows["id"].tolist(),
                        "source": source,
                        "combination_of": combo_rows.to_dict(orient="records"),
                        "question": question_prompts.combination,
                        "background": "N/A",
                        "market_info_resolution_criteria": "N/A",
                        "market_info_open_datetime": "N/A",
                        "market_info_close_datetime": "N/A",
                        "url": "N/A",
                        "resolution_criteria": "N/A",
                        "freeze_datetime_value": "N/A",
                        "freeze_datetime_value_explanation": "N/A",
                        "freeze_datetime": constants.FREEZE_DATETIME.isoformat(),
                        "human_prompt": question_prompts.combination,
                        "forecast_horizons": get_forecast_horizon(source, combo_rows),
                    }
                ]
            )
            df = pd.concat([df, df_combo], ignore_index=True)

    local_filename = f"/tmp/{filename}"
    with open(local_filename, "w", encoding="utf-8") as f:
        for record in df.to_dict(orient="records"):
            jsonl_str = json.dumps(record, ensure_ascii=False)
            f.write(jsonl_str + "\n")

    gcp.storage.upload(
        bucket_name=env.QUESTION_SETS_BUCKET,
        local_filename=local_filename,
    )


def drop_invalid_questions(dfq, dfmeta):
    """Drop invalid questions from dfq."""
    if dfmeta.empty:
        return dfq
    dfq = pd.merge(
        dfq,
        dfmeta,
        how="inner",
        on=["id", "source"],
    )
    return dfq[dfq["valid_question"]].drop(columns="valid_question")


def drop_missing_freeze_datetime(dfq):
    """Drop questions with missing values in the `freeze_datetime_value` column."""
    col = "freeze_datetime_value"
    dfq = dfq.dropna(subset=col, ignore_index=True)
    dfq = dfq[dfq[col] != "N/A"]
    dfq = dfq[dfq[col] != "nan"]
    return dfq


def drop_questions_that_resolve_too_soon(source, dfq):
    """Drop questions that resolve too soon.

    Given the freeze date:
    * for market questions determine whether or not the market will close before at least the first
      forecasting horizon. If it does, then do not use this question.
    * for data questions if forecast_horizons is empty, don't use the question
    """
    if source in constants.DATA_SOURCES:
        empty_horizons = dfq["forecast_horizons"].apply(lambda x: len(x) == 0)
        mask = empty_horizons | dfq["forecast_horizons"] == "N/A"
        return dfq.drop(labels=dfq[mask].index.tolist())

    empty_horizons = dfq["market_info_close_datetime"].apply(
        lambda x: data_utils.market_resolves_before_forecast_due_date(datetime.fromisoformat(x))
    )
    indices_to_drop = empty_horizons[empty_horizons].index.tolist()
    return dfq.drop(labels=indices_to_drop)


@decorator.log_runtime
def driver(_):
    """Curate questions for forecasting."""

    def format_string_field(row, template, field):
        return template.format(f_string_value=row[field])

    def format_string_value(row, template, value):
        return template.format(f_string_value=value)

    dfmeta = data_utils.download_and_read(
        filename=constants.META_DATA_FILENAME,
        local_filename=f"/tmp/{constants.META_DATA_FILENAME}",
        df_tmp=pd.DataFrame(columns=constants.META_DATA_FILE_COLUMNS).astype(
            constants.META_DATA_FILE_COLUMN_DTYPE
        ),
        dtype=constants.META_DATA_FILE_COLUMN_DTYPE,
    )

    # Get the latest questions
    QUESTIONS = constants.FREEZE_QUESTION_SOURCES
    sources_to_remove = []
    for source, _ in constants.FREEZE_QUESTION_SOURCES.items():
        dfq = data_utils.get_data_from_cloud_storage(
            source=source,
            return_question_data=True,
        )
        if dfq.empty:
            sources_to_remove.extend([source])
            logger.info(f"Found 0 questions from {source}.")
        else:
            dfq["source"] = source
            dfq = drop_invalid_questions(dfq=dfq, dfmeta=dfmeta)
            dfq = drop_missing_freeze_datetime(dfq)
            dfq = dfq[dfq["category"] != "Other"]
            dfq = dfq[~dfq["resolved"]]
            dfq = drop_questions_that_resolve_too_soon(source=source, dfq=dfq)
            dfq["human_prompt"] = dfq.apply(
                format_string_value,
                args=(QUESTIONS[source]["human_prompt"], QUESTIONS[source]["name"]),
                axis=1,
            )
            dfq["resolution_criteria"] = dfq.apply(
                format_string_field, args=(QUESTIONS[source]["resolution_criteria"], "url"), axis=1
            )
            dfq["freeze_datetime"] = constants.FREEZE_DATETIME.isoformat()
            dfq["combination_of"] = "N/A"
            if source == "acled":
                # Drop Acled-specific columns
                dfq.drop(
                    list(set(acled.QUESTION_FILE_COLUMNS) - set(constants.QUESTION_FILE_COLUMNS)),
                    axis=1,
                    inplace=True,
                )
            dfq.drop(
                [
                    "market_info_resolution_datetime",
                    "resolved",
                ],
                axis=1,
                inplace=True,
            )
            num_single_questions = len(dfq)
            num_questions = math.floor(num_single_questions / COMBO_PCT)
            QUESTIONS[source]["dfq"] = dfq.reset_index(drop=True)
            QUESTIONS[source]["num_single_questions_available"] = num_single_questions
            QUESTIONS[source]["num_questions_incl_combo_available"] = num_questions
            logger.info(f"Found {num_single_questions} single questions from {source}.\n")

    QUESTIONS = {key: value for key, value in QUESTIONS.items() if key not in sources_to_remove}

    # Find allocations of questions
    LLM_QUESTIONS = allocate_across_sources(
        for_humans=False,
        questions=QUESTIONS,
    )
    HUMAN_QUESTIONS = allocate_across_sources(
        for_humans=True,
        questions=LLM_QUESTIONS,
    )

    # Sample questions
    LLM_QUESTIONS = process_questions(
        questions=QUESTIONS,
        to_questions=LLM_QUESTIONS,
        single_generation_func=sample_single_questions,
        combo_generation_func=sample_combo_questions,
    )
    HUMAN_QUESTIONS = process_questions(
        questions=LLM_QUESTIONS,
        to_questions=HUMAN_QUESTIONS,
        single_generation_func=sample_single_questions,
    )

    def _log_questions_found(questions, for_humans):
        for_whom = "Humans" if for_humans else "LLMs"
        running_sum = 0
        for source, values in questions.items():
            logger.info("\n")
            # Overall
            dfq = values["dfq"]
            num_single = len(dfq)
            num_combo = len(values["combos"])
            num_total = num_single + num_combo
            running_sum += num_total
            title = (
                f"* {source}: Single: {num_single}."
                if for_humans
                else f"* {source}: Single: {num_single}. Combo: {num_combo}. Total: {num_total}"
            )
            logger.info(title)

            # Categories for standard and combo
            category_counts = (
                dfq.groupby("category")
                .agg(
                    count=("category", "size"),
                    underrepresented=("underrepresented_category", "any"),
                )
                .reset_index()
            )

            combo_indices = set([num for tup in values["combos"] for num in tup])
            dfq_combo = dfq[dfq.index.isin(combo_indices)]
            category_counts_combo = (
                dfq_combo.groupby("category").agg(count=("category", "size")).reset_index()
            )
            combo_dict = category_counts_combo.set_index("category").to_dict()["count"]

            max_category_length = max(
                len(row["category"]) + (4 if row["underrepresented"] else 0)
                for index, row in category_counts.iterrows()
            )
            combo_header = "" if for_humans else "N_combo"
            logger.info(f'    {"".ljust(max_category_length)}  N   {combo_header}')
            for _, row in category_counts.iterrows():
                category = row["category"]
                count = row["count"]
                combo_count = "" if for_humans else combo_dict.get(category, 0)
                if row["underrepresented"]:
                    category += " (*)"
                logger.info(f"  - {category.ljust(max_category_length)}: {count}   {combo_count}")
        logger.info(f"Found {running_sum} questions total for {for_whom}.\n")

    _log_questions_found(LLM_QUESTIONS, for_humans=False)
    _log_questions_found(HUMAN_QUESTIONS, for_humans=True)

    forecast_date_str = constants.FORECAST_DATE.isoformat()
    llm_filename = f"{forecast_date_str}-llm.jsonl"
    human_filename = f"{forecast_date_str}-human.jsonl"

    write_questions(LLM_QUESTIONS, llm_filename)
    write_questions(HUMAN_QUESTIONS, human_filename)

    logger.info("Done.")

    return "OK", 200


if __name__ == "__main__":
    driver(None)

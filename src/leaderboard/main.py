"""Create leaderboard."""

import itertools
import json
import logging
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from multiprocessing import Pool
from pprint import pprint

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm
from termcolor import colored
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from helpers import (  # noqa: E402
    constants,
    decorator,
    env,
    git,
    keys,
    question_curation,
    resolution,
)

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from utils import gcp  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LEADERBOARD_UPDATED_DATE_STR = "Updated " + datetime.now().strftime("%b. %d, %Y")
BASELINE_ORG_MODEL = {"organization": constants.BENCHMARK_NAME, "model": "Naive Forecaster"}
SUPERFORECASTER_MODEL = {
    "organization": constants.BENCHMARK_NAME,
    "model": "Superforecaster median forecast",
}
GENERAL_PUBLIC_MODEL = {"organization": constants.BENCHMARK_NAME, "model": "Public median forecast"}

CONFIDENCE_LEVEL = 0.95
LEADERBOARD_DECIMAL_PLACES = 3


def download_and_read_processed_forecast_file(filename):
    """Download forecast file."""
    with tempfile.NamedTemporaryFile(dir="/tmp/", delete=False) as tmp:
        local_filename = tmp.name

    gcp.storage.download(
        bucket_name=env.PROCESSED_FORECAST_SETS_BUCKET,
        filename=filename,
        local_filename=local_filename,
    )
    with open(local_filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    os.remove(local_filename)
    return {filename: data}


def get_masks(df):
    masks = {}

    resolved_mask = df["resolved"].astype(bool)
    unresolved_mask = ~resolved_mask

    masks["data"] = df["source"].isin(question_curation.DATA_SOURCES) & resolved_mask

    # Market sources should be reduced to the value at a single date. This is because we always
    # evaluate to the latest market value or the resolution value for a market and orgs only
    # forecast the outcome. Hence they get the same score at every period.
    market_source_mask = df["source"].isin(question_curation.MARKET_SOURCES)
    market_drop_duplicates_mask = ~df.duplicated(
        [
            "id",
            "source",
            "direction",
            "forecast_due_date",
        ]
    )
    masks["market"] = market_source_mask & market_drop_duplicates_mask
    masks["market_resolved"] = masks["market"] & resolved_mask
    masks["market_unresolved"] = masks["market"] & unresolved_mask

    return masks


def get_naive_forecaster_mask(df):
    return (df["organization"] == BASELINE_ORG_MODEL["organization"]) & (
        df["model"] == BASELINE_ORG_MODEL["model"]
    )


def get_leaderboard_entry(df, forecast_due_date, question_set_filename):
    """Create the leaderboard entry for the given dataframe."""

    def get_scores(df, mask):
        scores = df[mask]["score"]
        return scores.mean(), len(scores)

    masks = get_masks(df)

    # Datasets
    data_resolved_score, n_data_resolved = get_scores(df, masks["data"])
    data_resolved_std_dev = df[masks["data"]]["score"].std(ddof=1)

    # Markets
    market_resolved_score, n_market_resolved = get_scores(df, masks["market_resolved"])
    market_unresolved_score, n_market_unresolved = get_scores(df, masks["market_unresolved"])
    market_overall_score, n_market_overall = get_scores(df, masks["market"])
    market_overall_std_dev = df[masks["market"]]["score"].std(ddof=1)

    # Overall Resolved
    overall_resolved_score = (
        (data_resolved_score + market_resolved_score) / 2
        if not np.isnan(market_resolved_score)
        else data_resolved_score
    )
    n_overall_resolved = n_data_resolved + n_market_resolved

    # Overall
    overall_score = (data_resolved_score + market_overall_score) / 2
    n_overall = n_data_resolved + n_market_overall
    overall_std_dev = (
        np.sqrt(
            data_resolved_std_dev**2 / n_data_resolved
            + market_overall_std_dev**2 / n_market_overall
        )
        / 2
    )

    # % imputed
    pct_imputed = int(np.round(df["imputed"].mean() * 100))

    return {
        "data": data_resolved_score,
        "n_data": n_data_resolved,
        "market_resolved": market_resolved_score,
        "n_market_resolved": n_market_resolved,
        "market_unresolved": market_unresolved_score,
        "n_market_unresolved": n_market_unresolved,
        "market_overall": market_overall_score,
        "n_market_overall": n_market_overall,
        "overall_resolved": overall_resolved_score,
        "n_overall_resolved": n_overall_resolved,
        "overall": overall_score,
        "overall_std_dev": overall_std_dev,
        "n_overall": n_overall,
        "pct_imputed": pct_imputed,
        "df": df.copy(),
        "forecast_due_date": forecast_due_date,
        "question_set": question_set_filename,
    }


def get_permutation_p_value(
    df_data,
    n_best_data,
    n_comparison_data,
    df_market,
    n_best_market,
    n_comparison_market,
    observed_difference,
    n_replications,
):
    """Get the p-value comparing comparison to best using the Permutation Test."""
    permutation_differences = []
    for _ in range(n_replications):
        permuted_data_scores = np.random.permutation(df_data["score"])
        comparison_data_sample = permuted_data_scores[:n_comparison_data]
        best_data_sample = permuted_data_scores[n_comparison_data:]

        permuted_market_scores = np.random.permutation(df_market["score"])
        comparison_market_sample = permuted_market_scores[:n_comparison_market]
        best_market_sample = permuted_market_scores[n_comparison_market:]

        comparison_overall_mean = (
            np.mean(comparison_data_sample) + np.mean(comparison_market_sample)
        ) / 2
        best_overall_mean = (np.mean(best_data_sample) + np.mean(best_market_sample)) / 2
        permutation_differences.append(comparison_overall_mean - best_overall_mean)

    permutation_differences = np.array(permutation_differences)
    return np.round(
        np.mean(permutation_differences > observed_difference), LEADERBOARD_DECIMAL_PLACES
    )


def get_bootstrap_p_value(
    df_data,
    n_best_data,
    n_comparison_data,
    df_market,
    n_best_market,
    n_comparison_market,
    observed_difference,
    n_replications,
):
    """Get the p-value comparing comparison to best by Bootstrapping."""
    bootstrap_differences = []
    for _ in range(n_replications):
        df_best_data_resample = df_data["score"].sample(
            n=n_best_data, replace=True, ignore_index=True
        )
        df_best_market_resample = df_market["score"].sample(
            n=n_best_market, replace=True, ignore_index=True
        )

        df_comparison_data_resample = df_data["score"].sample(
            n=n_comparison_data, replace=True, ignore_index=True
        )
        df_comparison_market_resample = df_market["score"].sample(
            n=n_comparison_market, replace=True, ignore_index=True
        )

        comparison_overall_mean = (
            df_comparison_data_resample.mean() + df_comparison_market_resample.mean()
        ) / 2
        best_overall_mean = (df_best_data_resample.mean() + df_best_market_resample.mean()) / 2
        bootstrap_differences.append(comparison_overall_mean - best_overall_mean)

    bootstrap_differences = np.array(bootstrap_differences)
    return np.round(
        np.mean(bootstrap_differences > observed_difference),
        LEADERBOARD_DECIMAL_PLACES,
    )


def get_p_values(d):
    """Get p values comparing comparison to best to see if they're significantly different."""
    n_replications = 10000
    df = pd.DataFrame(d)
    df = df.sort_values(by=["overall"], ignore_index=True)

    # Only get pairwise p-values for now, skip treating questions as indpendent.
    # Keeping the code because it may come in handy when we run a new round with a different
    # question set.
    df = get_pairwise_p_values(df, n_replications)
    df.drop(columns="df", inplace=True)
    return df

    p_val_bootstrap_col_name = "p-value_bootstrap"
    p_val_permutation_col_name = "p-value_permutation"
    df[p_val_bootstrap_col_name] = None
    df[p_val_permutation_col_name] = None

    # Get best performer
    best_organization = df.at[0, "organization"]
    best_model = df.at[0, "model"]
    logger.info(f"p-value comparison best performer is: {best_organization} {best_model}.")

    df_best = pd.DataFrame(df.at[0, "df"])
    observed_overall_score_best = df.at[0, "overall"]

    df_best_data = df_best[
        (df_best["source"].isin(resolution.DATA_SOURCES)) & df_best["resolved"].astype(bool)
    ]
    df_best_market = df_best[df_best["source"].isin(resolution.MARKET_SOURCES)]

    n_best_data = len(df_best_data)
    n_best_market = len(df_best_market)

    for index in range(1, len(df)):
        df_comparison = pd.DataFrame(df.at[index, "df"])
        observed_overall_score_comparison = df.at[index, "overall"]

        df_comparison_data = df_comparison[
            (df_comparison["source"].isin(resolution.DATA_SOURCES))
            & df_comparison["resolved"].astype(bool)
        ]
        df_comparison_market = df_comparison[
            df_comparison["source"].isin(resolution.MARKET_SOURCES)
        ]

        n_comparison_data = len(df_comparison_data)
        n_comparison_market = len(df_comparison_market)

        df_data = pd.concat([df_best_data, df_comparison_data], ignore_index=True)
        df_market = pd.concat([df_best_market, df_comparison_market], ignore_index=True)

        observed_difference = observed_overall_score_comparison - observed_overall_score_best

        df.at[index, p_val_bootstrap_col_name] = get_bootstrap_p_value(
            df_data=df_data,
            n_best_data=n_best_data,
            n_comparison_data=n_comparison_data,
            df_market=df_market,
            n_best_market=n_best_market,
            n_comparison_market=n_comparison_market,
            observed_difference=observed_difference,
            n_replications=n_replications,
        )
        df.at[index, p_val_permutation_col_name] = get_permutation_p_value(
            df_data=df_data,
            n_best_data=n_best_data,
            n_comparison_data=n_comparison_data,
            df_market=df_market,
            n_best_market=n_best_market,
            n_comparison_market=n_comparison_market,
            observed_difference=observed_difference,
            n_replications=n_replications,
        )

    df.drop(columns="df", inplace=True)
    return df


def get_pairwise_p_values(df, n_replications):
    """Calculate p-values on Brier differences on individual questions.

    From Ezra: this improves precision because, for any two groups, forecasting accuracy is very
    correlated on the set of questions. Treating them as independent overstates
    imprecision. Instead, we can bootstrap the questions by focusing on the difference in scores on
    a question-by-question basis.

    A nice example of this is that if group A is always epsilon more accurate than group B on every
    question, we can be quite confident A is a better forecaster, even if epsilon is arbitrarily
    small and even if the standard deviation of accuracy for group A's forecasts is high.
    """
    p_val_bootstrap_col_name = "p-value_pairwise_bootstrap"
    df[p_val_bootstrap_col_name] = None
    better_than_super_col_name = "pct_better_than_no1"
    df[better_than_super_col_name] = 0.0

    # Get best performer
    best_organization = df.at[0, "organization"]
    best_model = df.at[0, "model"]
    logger.info(f"p-value comparison best performer is: {best_organization} {best_model}.")

    df_best = pd.DataFrame(df.at[0, "df"])
    observed_overall_score_best = df.at[0, "overall"]

    for index in range(1, len(df)):
        df_comparison = pd.DataFrame(df.at[index, "df"])
        observed_overall_score_comparison = df.at[index, "overall"]

        # first merge on the questions to then get the diff between the scores
        df_merged = pd.merge(
            df_best.copy(),
            df_comparison,
            on=[
                "id",
                "source",
                "direction",
                "forecast_due_date",
                "resolved",
                "resolution_date",
            ],
            how="inner",
            suffixes=[
                "_best",
                "_comparison",
            ],
        )
        df_merged = df_merged[["id", "source", "resolved", "score_comparison", "score_best"]]

        if not (len(df_best) == len(df_comparison) and len(df_best) == len(df_merged)):
            missing_in_comparison = df_best.merge(
                df_comparison,
                on=[
                    "id",
                    "source",
                    "direction",
                    "forecast_due_date",
                    "resolved",
                    "resolution_date",
                ],
                how="left",
                indicator=True,
            ).query('_merge == "left_only"')

            missing_in_best = df_comparison.merge(
                df_best,
                on=[
                    "id",
                    "source",
                    "direction",
                    "forecast_due_date",
                    "resolved",
                    "resolution_date",
                ],
                how="left",
                indicator=True,
            ).query('_merge == "left_only"')

            print(missing_in_comparison)
            print(missing_in_best)

            raise ValueError(
                "Problem with merge in `get_pairwise_p_values()`. Comparing org: "
                f"{df.at[index, 'organization']}, model: {df.at[index, 'model']} "
                f"n_best: {len(df_best)}, n_comparison: {len(df_comparison)}, "
                f"n_merged: {len(df_merged)}"
            )

        df_merged_data = df_merged[
            (df_merged["source"].isin(resolution.DATA_SOURCES)) & df_merged["resolved"].astype(bool)
        ].reset_index(drop=True)
        df_merged_market = df_merged[
            df_merged["source"].isin(resolution.MARKET_SOURCES)
        ].reset_index(drop=True)

        df_merged_data_diff = df_merged_data["score_comparison"] - df_merged_data["score_best"]
        df_merged_market_diff = (
            df_merged_market["score_comparison"] - df_merged_market["score_best"]
        )

        observed_difference = observed_overall_score_comparison - observed_overall_score_best
        assert (
            abs(
                observed_difference
                - ((df_merged_data_diff.mean() + df_merged_market_diff.mean()) / 2)
            )
            < 1e-15
        ), "Observed difference in scores is incorrect in `get_pairwise_p_values()`."

        # Shift mean of scores to 0 for the null hypothesis that comparison and best scores are
        # identical and hence their difference is 0
        df_merged_data_diff -= df_merged_data_diff.mean()
        df_merged_market_diff -= df_merged_market_diff.mean()

        assert (
            (df_merged_data_diff.mean() + df_merged_market_diff.mean()) / 2
        ) < 1e-15, "Observed difference in scores is incorrect in `get_pairwise_p_values()`."

        n_data = len(df_merged_data_diff)
        n_market = len(df_merged_market_diff)

        # Bootstrap p-value
        overall_diff = []
        for _ in range(n_replications):
            df_data_diff_bootstrapped = df_merged_data_diff.sample(
                n=n_data, replace=True, ignore_index=True
            )
            df_market_diff_bootstrapped = df_merged_market_diff.sample(
                n=n_market, replace=True, ignore_index=True
            )
            overall_diff.append(
                (df_data_diff_bootstrapped.mean() + df_market_diff_bootstrapped.mean()) / 2
            )
        overall_diff = np.array(overall_diff)

        df.at[index, p_val_bootstrap_col_name] = np.round(
            np.mean(overall_diff >= observed_difference), LEADERBOARD_DECIMAL_PLACES
        )

        # Percent better than supers
        df_combo = pd.concat([df_merged_data, df_merged_market], ignore_index=True)
        df.at[index, better_than_super_col_name] = (
            np.round(
                np.mean(df_combo["score_comparison"] < df_combo["score_best"]),
                LEADERBOARD_DECIMAL_PLACES,
            )
            * 100
        )

    return df


def add_to_leaderboard(leaderboard, org_and_model, df, forecast_due_date, question_set_filename):
    """Add scores to the leaderboard."""
    leaderboard_entry = [
        org_and_model | get_leaderboard_entry(df, forecast_due_date, question_set_filename)
    ]
    leaderboard["overall"] = leaderboard.get("overall", []) + leaderboard_entry
    # for horizon in df["horizon"].unique():
    #     leaderboard_entry = [
    #         org_and_model
    #         | get_leaderboard_entry(
    #             df[df["horizon"] == horizon].copy(), forecast_due_date, question_set_filename
    #         )
    #     ]
    #     leaderboard[str(horizon)] = leaderboard.get(str(horizon), []) + leaderboard_entry


def add_to_llm_leaderboard(*args, **kwargs):
    """Wrap `add_to_leaderboard` for easy reading of driver."""
    add_to_leaderboard(*args, **kwargs)


def download_question_set_save_in_cache(forecast_due_date, cache):
    """Time-saving function to only download files once per run.

    Save question files in cache.
    """
    if forecast_due_date not in cache:
        cache[forecast_due_date] = {}

    for human_or_llm in ["human", "llm"]:
        if human_or_llm not in cache[forecast_due_date]:
            cache[forecast_due_date][human_or_llm] = resolution.download_and_read_question_set_file(
                filename=f"{forecast_due_date}-{human_or_llm}.json"
            )


def add_to_llm_and_human_leaderboard(
    leaderboard, org_and_model, df, forecast_due_date, cache, question_set_filename
):
    """Parse the forecasts to include only those questions that were in the human question set."""
    download_question_set_save_in_cache(forecast_due_date, cache)
    df_human_question_set = cache[forecast_due_date]["human"].copy()
    df_only_human_question_set = pd.merge(
        df,
        df_human_question_set[["id", "source"]],
        on=["id", "source"],
    ).reset_index(drop=True)
    add_to_leaderboard(
        leaderboard=leaderboard,
        org_and_model=org_and_model,
        df=df_only_human_question_set,
        forecast_due_date=forecast_due_date,
        question_set_filename=question_set_filename,
    )


def add_to_llm_and_human_combo_leaderboards(
    leaderboard_combo,
    org_and_model,
    df,
    forecast_due_date,
    cache,
    is_human_forecast_set,
    question_set_filename,
):
    """Parse the forecasts to include only those questions that were in the human question set."""
    download_question_set_save_in_cache(forecast_due_date, cache)
    df_human_question_set = cache[forecast_due_date]["human"].copy()
    df_llm_question_set = cache[forecast_due_date]["llm"].copy()
    if "combos" not in cache[forecast_due_date]:
        human_possible_combos = []
        for _, row in df_llm_question_set[
            df_llm_question_set["id"].apply(resolution.is_combo)
        ].iterrows():
            id0, id1 = row["id"]
            source = row["source"]
            df_source = df_human_question_set[df_human_question_set["source"] == source]
            if {id0, id1}.issubset(df_source["id"]):
                human_possible_combos.append({"source": source, "id": row["id"]})
        cache[forecast_due_date]["combos"] = human_possible_combos
    human_combos = cache[forecast_due_date]["combos"].copy()

    df_only_human_question_set = pd.merge(
        df,
        df_human_question_set[["id", "source"]],
        on=["id", "source"],
    ).reset_index(drop=True)

    # Add pertinent combos from llm forecast file to llm df
    if not is_human_forecast_set:
        df_llm_combos = df[
            df.apply(
                lambda row: (row["id"], row["source"])
                in [(combo["id"], combo["source"]) for combo in human_combos],
                axis=1,
            )
        ]
        df_only_human_question_set = pd.concat(
            [df_only_human_question_set, df_llm_combos], ignore_index=True
        )

    def generate_combo_forecasts(df, forecast_due_date):
        """Generate combo forecasts."""
        # Remove combos in df, if any
        df = df[~df["id"].apply(resolution.is_combo)]

        # Generate combos from the df
        for combo in human_combos:
            source = combo["source"]
            id0, id1 = combo["id"]
            df_source = df[df["source"] == source]
            df_forecast0 = df_source[df_source["id"] == id0]
            df_forecast1 = df_source[df_source["id"] == id1]
            if df_forecast0.empty or df_forecast1.empty:
                # If either forecast set is empty, it means one of the questions was dropped as N/A
                # and hence is not in the processed forecast file.
                continue
            resolution_dates = set(df_forecast0["resolution_date"]).intersection(
                set(df_forecast1["resolution_date"])
            )
            for resolution_date in resolution_dates:
                df_forecast0_tmp = df_forecast0[df_forecast0["resolution_date"] == resolution_date]
                df_forecast1_tmp = df_forecast1[df_forecast1["resolution_date"] == resolution_date]
                if len(df_forecast0_tmp) != 1 or len(df_forecast1_tmp) != 1:
                    raise ValueError("`generate_combo_forecasts`: should not arrive here.")

                for dir0, dir1 in list(itertools.product([1, -1], repeat=2)):
                    forecast = resolution.combo_change_sign(
                        df_forecast0_tmp["forecast"].iloc[0], dir0
                    ) * resolution.combo_change_sign(df_forecast1_tmp["forecast"].iloc[0], dir1)
                    resolved_to = resolution.combo_change_sign(
                        df_forecast0_tmp["resolved_to"].iloc[0], dir0
                    ) * resolution.combo_change_sign(df_forecast1_tmp["resolved_to"].iloc[0], dir1)
                    resolved = (
                        resolution.is_combo_question_resolved(
                            is_resolved0=df_forecast0_tmp["resolved"].iloc[0],
                            is_resolved1=df_forecast1_tmp["resolved"].iloc[0],
                            dir0=dir0,
                            dir1=dir1,
                            resolved_to0=df_forecast0_tmp["resolved_to"].iloc[0],
                            resolved_to1=df_forecast1_tmp["resolved_to"].iloc[0],
                        )
                        if source in resolution.MARKET_SOURCES
                        else True
                    )
                    imputed = (
                        df_forecast0_tmp["imputed"].iloc[0] or df_forecast1_tmp["imputed"].iloc[0]
                    )
                    score = (forecast - resolved_to) ** 2
                    new_row = {
                        "id": (id0, id1),
                        "source": source,
                        "direction": (dir0, dir1),
                        "forecast_due_date": df_forecast0_tmp["forecast_due_date"].iloc[0],
                        "market_value_on_due_date": np.nan,
                        "resolution_date": resolution_date,
                        "resolved_to": resolved_to,
                        "resolved": resolved,
                        "forecast": forecast,
                        "imputed": imputed,
                        "score": score,
                    }
                    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        return df

    if is_human_forecast_set:
        # This is a human forecast set. Hence combos need to be generated for leaderboard_combo.
        df_only_human_question_set = generate_combo_forecasts(
            df_only_human_question_set, forecast_due_date
        )

    leaderboard_combo = add_to_leaderboard(
        leaderboard=leaderboard_combo,
        org_and_model=org_and_model,
        df=df_only_human_question_set,
        forecast_due_date=forecast_due_date,
        question_set_filename=question_set_filename,
    )


def make_and_upload_html_table(df, title, basename):
    """Make and upload HTLM leaderboard."""
    # Replace NaN with empty strings for display
    logger.info(f"Making HTML leaderboard file: {title} {basename}.")
    df = df.fillna("--")

    # Add ranking
    df = df.sort_values(
        by=[
            "BSS_wrt_naive_mean",
            "n_overall",
            "overall",
        ],
        ascending=[
            False,
            False,
            True,
        ],
        ignore_index=True,
    )

    # Round columns to 3 decimal places
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].round(LEADERBOARD_DECIMAL_PLACES)

    # Insert ranking
    df.insert(loc=0, column="Ranking", value="")
    df["score_diff"] = df["BSS_wrt_naive_mean"] - df["BSS_wrt_naive_mean"].shift(1)
    for index, row in df.iterrows():
        if row["score_diff"] != 0:
            prev_rank = index + 1
        df.loc[index, "Ranking"] = prev_rank
    df.drop(columns="score_diff", inplace=True)

    # n_data = df["n_data"].max()
    # n_market_resolved = df["n_market_resolved"].max()
    # n_market_unresolved = df["n_market_unresolved"].max()
    # n_market_overall = df["n_market_overall"].max()
    # n_overall = df["n_overall"].max()
    # n_overall_resolved = df["n_overall_resolved"].max()

    df["pct_imputed"] = df["pct_imputed"].round(0).astype(int).astype(str) + "%"
    # df["pct_better_than_no1"] = df["pct_better_than_no1"].round(0).astype(int).astype(str) + "%"

    # def get_p_value_display(p):
    #     if not isinstance(p, (float, int)):
    #         return str(p)
    #     if p < 0.001:
    #         return "<0.001"
    #     if p < 0.01:
    #         return "<0.01"
    #     if p < 0.05:
    #         return "<0.05"
    #     return f"{p:.{LEADERBOARD_DECIMAL_PLACES}f}"
    #
    # df["p-value_pairwise_bootstrap"] = df["p-value_pairwise_bootstrap"].apply(get_p_value_display)

    def link_question_set(question_set):
        return (
            '<a href="https://github.com/forecastingresearch/forecastbench-datasets/'
            + "blob/main/datasets/question_sets/"
            + question_set
            + '">'
            + question_set
            + "</a>"
        )

    def link_question_sets(question_sets):
        if isinstance(question_sets, str):
            return link_question_set(question_sets)
        if isinstance(question_sets, tuple):
            retval = link_question_set(question_sets[0]) + ", "
            if len(question_sets) > 2:
                retval += "..., "
            return retval + link_question_set(question_sets[-1])
        raise ValueError("wrong type for question set")

    df["question_set"] = df["question_set"].apply(link_question_sets)

    def format_score(score_series, count_series):
        def safe_format_score(x):
            try:
                return f"{float(x):.{LEADERBOARD_DECIMAL_PLACES}f}"
            except (ValueError, TypeError):
                return str(x)

        formatted_scores = score_series.apply(safe_format_score)
        formatted_counts = count_series.map(lambda x: f"{int(x):,}" if pd.notnull(x) else "")

        # Ensure both series are strings before concatenation.
        return formatted_scores.astype(str) + " (" + formatted_counts.astype(str) + ")"

    df["data"] = format_score(df["data"], df["n_data"])
    df["market_resolved"] = format_score(df["market_resolved"], df["n_market_resolved"])
    df["market_unresolved"] = format_score(df["market_unresolved"], df["n_market_unresolved"])
    df["market_overall"] = format_score(df["market_overall"], df["n_market_overall"])
    df["overall_resolved"] = format_score(df["overall_resolved"], df["n_overall_resolved"])
    df["overall"] = format_score(df["overall"], df["n_overall"])
    df["BSS_wrt_naive_mean"] = format_score(df["BSS_wrt_naive_mean"], df["n_overall"])

    df = df[
        [
            "Ranking",
            "organization",
            "model",
            "data",
            "market_resolved",
            "market_unresolved",
            "market_overall",
            "overall_resolved",
            # "overall",
            # "confidence_interval_overall",
            # "p-value_pairwise_bootstrap",
            # "pct_better_than_no1",
            # "z_score_wrt_naive_mean",
            "BSS_wrt_naive_mean",
            "bootstrap_BSS_CI",
            "pct_imputed",
            "question_set",
        ]
    ]
    df = df.rename(
        columns={
            "organization": "Organization",
            "model": "Model",
            "data": "Dataset Score",
            "market_resolved": "Market Score (resolved)",
            "market_unresolved": "Market Score (unresolved)",
            "market_overall": "Market Score (overall)",
            "overall_resolved": "Overall Resolved Score",
            # "overall": "Overall Score",
            # "confidence_interval_overall": "Overall Score 95% CI",
            # "p-value_pairwise_bootstrap": "Pairwise p-value comparing to No. 1 (bootstrapped)",
            # "pct_better_than_no1": "Pct. more accurate than No. 1",
            # "z_score_wrt_naive_mean": "Z-score",
            "BSS_wrt_naive_mean": "BSS",
            "bootstrap_BSS_CI": "BSS 95% CI",
            "pct_imputed": "Pct. Imputed",
            "question_set": "Question Set(s)",
        }
    )

    column_descriptions = """
        <div style="display: flex; align-items: center;">
          <a data-bs-toggle="collapse" data-bs-target="#descriptionCollapse" aria-expanded="false"
             aria-controls="descriptionCollapse" style="text-decoration: none; color: inherit;
             display: flex; align-items: center; cursor: pointer;">
            <i class="bi bi-chevron-right rotate" id="toggleArrow" style="margin-left: 5px;"></i>
            <span>Column descriptions</span>
          </a>
        </div>
        <div class="collapse mt-3" id="descriptionCollapse" style="padding: 0px;">
          <div class="card card-body">
            <ul>
              <li><b>Ranking</b>: The position of the model in the leaderboard as ordered by
                                  Overall Score</li>
              <li><b>Organization</b>: The group responsible for the model or forecasts</li>
              <li><b>Model</b>: The LLM model & prompt info or the human group and forecast
                                aggregation method
                  <ul>
                    <li>zero shot: used a zero-shot prompt</li>
                    <li>scratchpad: used a scratchpad prompt with instructions that outline a
                                    procedure the model should use to reason about the question</li>
                    <li>with freeze values: means that, for questions from market sources, the prompt
                                            was supplemented with the aggregate human forecast from
                                            the relevant platform on the day the question set was
                                            generated</li>
                    <li>with news: means that the prompt was supplemented with relevant news
                                   summaries obtained through an automated process</li>
                  </ul>
              <li><b>Dataset Score</b>: The average Brier score across all questions sourced from
                                        datasets</li>
              <li><b>Market Score (resolved)</b>: The average Brier score across all resolved
                                                  questions sourced from prediction markets and
                                                  forecast aggregation platforms</li>
              <li><b>Market Score (unresolved)</b>: The average Brier score across all unresolved
                                                    questions sourced from prediction markets and
                                                    forecast aggregation platforms</li>
              <li><b>Market Score (overall)</b>: The average Brier score across all questions
                                                 sourced from prediction markets and forecast
                                                 aggregation platforms</li>
              <li><b>Overall Resolved Score</b>: The average of the Dataset Score and the Market
                                                 Score (resolved) columns</li>
              <li><b>BSS</b>: The Brier skill score.
              <li><b>BSS 95% CI</b>: The bootstrapped confidence interval for the BSS.
              <li><b>Pct. imputed</b>: The percent of questions for which this forecaster did not
                              provide a forecast and hence had a forecast value imputed (0.5 for
                              dataset questions and the aggregate human forecast on the forecast
                              due date for questions sourced from prediction markets or forecast
                              aggregation platforms)</li>
              <li><b>Question Set(s)</b>: The question sets that were forecast on. If more than two
                              question sets were forecast on, show the oldest and the most recent.
                              </li>
            </ul>
          </div>
        </div>
        <script>
        var toggleArrow = document.getElementById('toggleArrow');
        var toggleLink = document.querySelector('[data-bs-toggle="collapse"]');

        toggleLink.addEventListener('click', function () {
          if (toggleArrow.classList.contains('rotate-down')) {
            toggleArrow.classList.remove('rotate-down');
          } else {
            toggleArrow.classList.add('rotate-down');
          }
        });
        </script>
    """

    # Remove lengths from df
    df = df[[c for c in df.columns if not c.startswith("n_")]]

    html_code = df.to_html(
        classes="table table-striped table-bordered",
        index=False,
        table_id="myTable",
        escape=False,
        render_links=True,
    )

    ORDER_COL_IDX = 8
    html_code = (
        """<!DOCTYPE html>
<html>
    <head>
        <meta charset="UTF-8">
        <title>LLM Data Table</title>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css"
              integrity="sha384-QWTKZyjpPEjISv5WaRU9OFeRpok6YctnYmDr5pNlyT2bRjXh0JMhjY6hW+ALEwIH"
              crossorigin="anonymous">
        <link rel="stylesheet" type="text/css"
              href="https://cdn.datatables.net/2.1.6/css/dataTables.jqueryui.min.css">
        <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons/font/bootstrap-icons.css" rel="stylesheet">
        <script src="https://code.jquery.com/jquery-3.7.1.js"></script>
        <script type="text/javascript" charset="utf8"
                src="https://cdn.datatables.net/2.1.6/js/dataTables.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
        <style>
            body {
              font-size: 10px;
            }
            table.dataTable {
              font-size: 10px;
            }
            .dataTables_wrapper .dataTables_length,
            .dataTables_wrapper .dataTables_filter,
            .dataTables_wrapper .dataTables_info,
            .dataTables_wrapper .dataTables_paginate {
                font-size: 10px;
            }
            .dataTables_length {
              display: none;
            }
            .dataTables_paginate {
              display: none;
            }
            .dataTables_info {
              display: none;
            }
            .dataTables_wrapper {
              margin-bottom: 20px; /* Add bottom margin */
            }
            .highlight {
              background-color: #eeebe0 !important;
            }
            h1 {
              text-align: center;
              margin-top: 10px;
              font-family: 'Arial', sans-serif;
              font-size: 16px;
           }
           .rotate {
             transition: transform 0.3s ease;
           }
           .rotate-down {
             transform: rotate(90deg);
           }
           .right-align {
             text-align: right;
           }
          .updated-date {
               font-size: 10px;
               text-align: center;
               color: #6c757d; /* Bootstrap muted text color */
               margin-top: -10px;
           }
        </style>
    </head>
    <body>
        <div class="container mt-4">
    """
        + "<h1>"
        + title
        + "</h1>"
        + '<p class="updated-date">'
        + LEADERBOARD_UPDATED_DATE_STR
        + "</p>"
        + column_descriptions
        + html_code
        + """
        </div>
        <script>
        """
        + r"""
        $.fn.dataTable.ext.type.detect.unshift(function(data) {
            if (/^-?\d+(\.\d+)? \(/.test(data.trim())) {
                return 'bss-num';
            }
            return null;
        });
        $.fn.dataTable.ext.type.order['bss-num-pre'] = function(data) {
            const match = data.match(/^(-?\d+(\.\d+)?) \(([\d,]+)\)/);
            if (match) {
                const bssValue = parseFloat(match[1]);
                const parensValue = parseInt(match[3].replace(/,/g, ""), 10);
                return bssValue + (parensValue / 100000000);
            }
            return [0, 0];
        };
        $(document).ready(function() {
            var table = $('#myTable').DataTable({
                "pageLength": -1,
                "lengthMenu": [[-1], ["All"]],"""
        + f"""
                "order": [[{ORDER_COL_IDX}, "desc"]],"""
        + """
                "paging": false,
                "info": false,
                "orderMulti": false,
                "search": {
                    "regex": true,
                    "smart": true
                },
                "columnDefs": [
                    {"""
        + f"""
                        "targets": {ORDER_COL_IDX},"""
        + """
                        "type": "bss-num",
                        "orderSequence": ["desc", "asc"]
                    },
                    {
                        "targets": 10,
                        "className": "right-align"
                    },
                    {
                        "targets": '_all',
                        "searchable": true
                    }
                ]
            });
        """
        + f"""table.column({ORDER_COL_IDX}).nodes().to$().addClass("highlight");"""
        + """
        });
        </script>
    </body>
</html>"""
    )

    # Create HTML file
    local_filename_html = f"/tmp/{basename}.html"
    with open(local_filename_html, "w") as file:
        file.write(html_code)

    # Create CSV file
    local_filename_csv = f"/tmp/{basename}.csv"
    df.to_csv(local_filename_csv, index=False)

    # Upload files to Cloud
    destination_folder = "leaderboards"
    # gcp.storage.upload(
    #     bucket_name=env.PUBLIC_RELEASE_BUCKET,
    #     local_filename=local_filename_html,
    #     destination_folder=f"{destination_folder}/html",
    # )
    # gcp.storage.upload(
    #     bucket_name=env.PUBLIC_RELEASE_BUCKET,
    #     local_filename=local_filename_csv,
    #     destination_folder=f"{destination_folder}/csv",
    # )

    return {
        local_filename_html: f"{destination_folder}/html/{basename}.html",
        local_filename_csv: f"{destination_folder}/csv/{basename}.csv",
    }


def get_BSS(df):
    """Calculate the standard scores with respect to naive forecaster for every question set."""
    # df["z_score_wrt_naive_mean"] = None
    df["BSS_wrt_naive_mean"] = None
    mask_naive_forecaster = (df["organization"] == BASELINE_ORG_MODEL["organization"]) & (
        df["model"] == BASELINE_ORG_MODEL["model"]
    )
    for forecast_due_date in df["forecast_due_date"].unique():
        mask_forecast_due_date = df["forecast_due_date"] == forecast_due_date
        naive_mask = mask_naive_forecaster & mask_forecast_due_date
        naive_baseline_mean = df[naive_mask]["overall"].values[0]
        df.loc[mask_forecast_due_date, "BSS_wrt_naive_mean"] = (
            1 - df[mask_forecast_due_date]["overall"] / naive_baseline_mean
        )
    return df


def prep_combo_for_get_leaderboard_entry_call(df_group):
    """Prepare arguments for get_leaderboard_entry call from merge functions."""
    df_combined = pd.concat(df_group["df"].tolist(), ignore_index=True)
    forecast_due_dates = tuple(sorted(df_group["forecast_due_date"].tolist()))
    question_set_filenames = tuple(sorted(df_group["question_set"].tolist()))
    return df_combined, forecast_due_dates, question_set_filenames


def create_naive_forecaster_leaderboard_entries(df_naive_forecaster):
    """Make a single leaderboard entry for every forecast due date combo of the naive forecaster."""
    combined_result = []
    unique_forecast_due_dates = df_naive_forecaster["forecast_due_date"].unique()
    org_and_model = {
        "organization": BASELINE_ORG_MODEL["organization"],
        "model": BASELINE_ORG_MODEL["model"],
    }
    for r in range(2, len(unique_forecast_due_dates) + 1):
        for combo in itertools.combinations(unique_forecast_due_dates, r):
            df_tmp = df_naive_forecaster[
                df_naive_forecaster["forecast_due_date"].isin(combo)
            ].copy()
            df_combined, forecast_due_dates, question_set_filenames = (
                prep_combo_for_get_leaderboard_entry_call(df_tmp)
            )
            combined_result += [
                org_and_model
                | get_leaderboard_entry(df_combined, forecast_due_dates, question_set_filenames)
            ]
    return combined_result


def merge_duplicates(df):
    """Merge the duplicated entries into one entry."""
    naive_forecaster_mask = (df["organization"] == BASELINE_ORG_MODEL["organization"]) & (
        df["model"] == BASELINE_ORG_MODEL["model"]
    )
    df_naive_forecaster = df[naive_forecaster_mask].reset_index(drop=True)
    df = df[~naive_forecaster_mask].reset_index(drop=True)

    combined_result = []
    for (organization, model), df_group in df.groupby(["organization", "model"]):
        logger.info(f"Combining forecasts for: Organization: {organization} Model: {model}.")
        org_and_model = {"organization": organization, "model": model}
        df_combined, forecast_due_dates, question_set_filenames = (
            prep_combo_for_get_leaderboard_entry_call(df_group)
        )
        combined_result += [
            org_and_model
            | get_leaderboard_entry(df_combined, forecast_due_dates, question_set_filenames)
        ]

    combined_result += create_naive_forecaster_leaderboard_entries(df_naive_forecaster)

    df_merged = pd.DataFrame(combined_result)
    # Add the individual naive forecaster lines back to the df to return. These will be used to
    # bootstrap the BSS CI.
    df_merged = pd.concat([df_merged, df_naive_forecaster], ignore_index=True)
    df_merged = get_BSS(df_merged)

    return df_merged  # .drop(index=to_drop).reset_index(drop=True)


def merge_common_models(df):
    """Merge common models for a given leaderboard."""
    df = get_BSS(df)
    df_duplicated = (
        df[
            df.duplicated(
                subset=[
                    "organization",
                    "model",
                ],
                keep=False,
            )
        ]
        .copy()
        .reset_index(drop=True)
    )
    if df_duplicated.empty:
        return df
    df = df.drop_duplicates(
        subset=[
            "organization",
            "model",
        ],
        keep=False,
        ignore_index=True,
    )
    df_duplicated = merge_duplicates(df_duplicated)
    return pd.concat([df, df_duplicated], ignore_index=True)


def bootstrap_BSS_CI(df):
    """Bootstrap the confidence interval for the BSS."""
    n_replications = 3
    mask_naive_forecaster = (df["organization"] == BASELINE_ORG_MODEL["organization"]) & (
        df["model"] == BASELINE_ORG_MODEL["model"]
    )
    bootstrap_results = {}
    df["bootstrap_BSS_CI"] = None

    def sample_block_indices_by_source(df, mask):
        """Sample everything where mask is true, source by source."""
        sampled_indices = {}
        for source in df[mask]["source"].unique():
            source_mask = mask & (df["source"] == source)
            available_indices = df.loc[source_mask].index.to_numpy()
            sampled_indices[source] = np.random.choice(
                available_indices, size=len(available_indices), replace=True
            )
        return sampled_indices

    def sample_block_indices_no_source(df, mask):
        """Sample everything where mask is true."""
        available_indices = df.loc[mask].index.to_numpy()
        sampled_indices = np.random.choice(
            available_indices, size=len(available_indices), replace=True
        )
        return sampled_indices

    def get_sample_indices(df, forecast_due_date, sample_indices):
        """Get the indices for the given forecast due date."""
        if forecast_due_date in sample_indices.keys():
            return sample_indices[forecast_due_date]

        masks = get_masks(df)

        sampled_indices = {}
        sampled_indices["data"] = sample_block_indices_by_source(df, masks["data"])
        sampled_indices["market_resolved"] = sample_block_indices_no_source(
            df,
            masks["market_resolved"],
        )
        sampled_indices["market_unresolved"] = sample_block_indices_no_source(
            df,
            masks["market_unresolved"],
        )
        sample_indices[forecast_due_date] = sampled_indices

        return sampled_indices

    def apply_sample_indices(df, indices):
        """Apply the sample provided to df."""
        df_updated = df.copy()
        # For each block, update the score column in place.

        masks = get_masks(df)

        for block, sources in indices.items():
            # Define the mask for the block.
            # Here you must know how each block is defined. For example:
            if block not in masks.keys():
                raise ValueError("should not arrive here (3).")

            if block == "data":
                for source in sources:
                    boot_indices = sources[source]
                    source_block_mask = masks[block] & (df["source"] == source)

                    orig_indices = df_updated.loc[source_block_mask].index
                    new_scores = df_updated.loc[boot_indices, "score"].reset_index(drop=True)
                    assert len(orig_indices) == len(
                        new_scores
                    ), "Mismatch in number of rows for data block"
                    df_updated.loc[orig_indices, "score"] = new_scores.to_numpy()
            else:
                boot_indices = sources
                orig_indices = df_updated.loc[masks[block]].index
                new_scores = df_updated.loc[boot_indices, "score"].reset_index(drop=True)
                assert len(orig_indices) == len(
                    new_scores
                ), "Mismatch in number of rows for market block"
                df_updated.loc[orig_indices, "score"] = new_scores.to_numpy()

        return df_updated

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))

    model_to_plot = "O1-Preview-2024-09-12 (zero shot)"
    i = 0
    first_print = True
    bootstrap_colors = plt.cm.Blues(np.linspace(0.3, 0.7, 9000))  # optional: fade blues

    for _ in range(n_replications):
        print(f"Replication {_}")
        leaderboard = []

        # Duplicate df with sampled scores for all models
        df_tmp = df.copy()

        # Drop combination of naive forecasters
        # These should be recreated from the resampled individual naive forecasters
        df_tmp = df_tmp[
            ~(mask_naive_forecaster & df_tmp["question_set"].apply(lambda x: isinstance(x, tuple)))
        ].reset_index(drop=True)

        sample_indices = {}

        for _, row in df_tmp.iterrows():
            df_model = row["df"].copy()
            if row["model"] == model_to_plot and first_print:
                ax.plot(df_model["score"].values, color="black", label="Original", linewidth=2)
                first_print = False

            indices = get_sample_indices(df_model, row["forecast_due_date"], sample_indices)
            df_model = apply_sample_indices(df_model, indices)
            if row["model"] == model_to_plot:
                ax.plot(
                    df_model["score"].values,
                    color=bootstrap_colors[n_replications],
                    alpha=0.4,
                    label=None,
                )

            leaderboard += [
                {
                    "organization": row["organization"],
                    "model": row["model"],
                }
                | get_leaderboard_entry(df_model, row["forecast_due_date"], row["question_set"])
            ]

        df_leaderboard = pd.DataFrame(leaderboard)
        df_naive_forecaster = (
            df_leaderboard[
                (df_leaderboard["organization"] == BASELINE_ORG_MODEL["organization"])
                & (df_leaderboard["model"] == BASELINE_ORG_MODEL["model"])
            ]
            .copy()
            .reset_index(drop=True)
        )
        leaderboard += create_naive_forecaster_leaderboard_entries(df_naive_forecaster)

        # Redo this to create the complete, resampled leaderboard
        df_leaderboard = pd.DataFrame(leaderboard)

        # Get BSS with the leaderboard, updated with the combined naive forecasters
        df_leaderboard = get_BSS(df=df_leaderboard)
        for _, row in df_leaderboard.iterrows():
            key = (row["organization"], row["model"])
            bss_value = row["BSS_wrt_naive_mean"]
            bootstrap_results.setdefault(key, []).append(bss_value)

    plt.tight_layout()
    plt.show()

    ci_results = {}
    for key, bss_values in bootstrap_results.items():
        alpha = (1 - CONFIDENCE_LEVEL) / 2
        lower = np.percentile(bss_values, alpha * 100)
        upper = np.percentile(bss_values, (1 - alpha) * 100)
        ci_results[key] = (lower, upper)

    def assign_ci(row):
        key = (row["organization"], row["model"])
        ci = ci_results.get(key, None)
        if ci is not None:
            return [round(x, LEADERBOARD_DECIMAL_PLACES) for x in ci]
        return None

    df["bootstrap_BSS_CI"] = df.apply(assign_ci, axis=1)

    # Remove extra naive forecaster
    df_subset = df[mask_naive_forecaster].copy()
    df_subset["tuple_length"] = df_subset["forecast_due_date"].apply(
        lambda x: len(x) if isinstance(x, (list, tuple)) else 0
    )
    if (df_subset["tuple_length"] > 0).any():
        max_index = df_subset["tuple_length"].idxmax()
        to_drop = df_subset.index.difference([max_index])
        df = df.drop(index=to_drop).reset_index(drop=True)
    return df


def make_leaderboard(leaderboard, title, basename):
    """Make leaderboard."""
    logger.info(colored(f"Making leaderboard: {title}", "red"))
    df = pd.DataFrame(leaderboard)
    df = merge_common_models(df)
    df = bootstrap_BSS_CI(df)
    files = make_and_upload_html_table(
        df=df,
        title=title,
        basename=basename,
    )
    logger.info(colored("Done.", "red"))
    return files


def worker(task):
    """Pool worker for leaderboard creation."""
    try:
        return make_leaderboard(
            leaderboard=task["leaderboard"],
            title=task["title"],
            basename=task["basename"],
        )
    except Exception as e:
        msg = f"Error processing task {task['title']}: {e}"
        logger.error(msg)
        raise ValueError(msg)


@decorator.log_runtime
def driver(_):
    """Create new leaderboard."""
    cache = {}
    llm_leaderboard = {}
    llm_and_human_leaderboard = {}
    # llm_and_human_combo_leaderboard = {}
    # files = gcp.storage.list(env.PROCESSED_FORECAST_SETS_BUCKET)
    # files = [file for file in files if file.endswith(".json")]
    files = [
        # "2024-07-21/2024-07-21.ForecastBench.always-0.5.json",
        "2024-07-21/2024-07-21.ForecastBench.always-0.json",
        # "2024-07-21/2024-07-21.ForecastBench.always-1.json",
        # "2024-07-21/2024-07-21.ForecastBench.human_public.json",
        # "2024-07-21/2024-07-21.ForecastBench.human_super.json",
        # "2024-07-21/2024-07-21.ForecastBench.imputed-forecaster.json",
        "2024-07-21/2024-07-21.ForecastBench.naive-forecaster.json",
        # '2024-07-21/2024-07-21.ForecastBench.random-uniform.json',
        "2024-07-21/2024-07-21.Anthropic.claude_3p5_sonnet_scratchpad_with_freeze_values.json",
        # "2024-07-21/2024-07-21.OpenAI.gpt_4_turbo_0409_scratchpad_with_freeze_values.json",
        # "2024-07-21/2024-07-21.Qwen.qwen_1p5_110b_scratchpad.json",
        "2024-12-08/2024-12-08.ForecastBench.always-0.json",
        # '2024-12-08/2024-12-08.ForecastBench.always-1.json',
        # '2024-12-08/2024-12-08.ForecastBench.imputed-forecaster.json',
        "2024-12-08/2024-12-08.ForecastBench.naive-forecaster.json",
        # '2024-12-08/2024-12-08.OpenAI.o1-mini-2024-09-12_scratchpad.json',
        # '2024-12-08/2024-12-08.OpenAI.o1-mini-2024-09-12_scratchpad_with_freeze_values.json',
        # '2024-12-08/2024-12-08.OpenAI.o1-mini-2024-09-12_zero_shot.json',
        # '2024-12-08/2024-12-08.OpenAI.o1-mini-2024-09-12_zero_shot_with_freeze_values.json',
        # '2024-12-08/2024-12-08.OpenAI.o1-preview-2024-09-12_scratchpad.json',
        # "2024-12-08/2024-12-08.OpenAI.o1-preview-2024-09-12_scratchpad_with_freeze_values.json",
        "2024-12-08/2024-12-08.OpenAI.o1-preview-2024-09-12_zero_shot.json",
        "2024-12-08/2024-12-08.OpenAI.o1-preview-2024-09-12_zero_shot_with_freeze_values.json",
        "2024-12-08/2024-12-08.Anthropic.claude_3p5_sonnet_scratchpad_with_freeze_values.json",
        "2025-03-02/2025-03-02.ForecastBench.naive-forecaster.json",
        "2025-03-02/2025-03-02.ForecastBench.always-0.json",
        "2025-03-02/2025-03-02.Anthropic.claude-3-5-sonnet-20240620_scratchpad_with_freeze_values.json",
    ]

    with ThreadPoolExecutor() as executor:
        dfs = list(
            tqdm(
                executor.map(
                    download_and_read_processed_forecast_file,
                    files,
                ),
                total=len(files),
                desc="downloading processed forecast files",
            )
        )
        executor.shutdown(wait=True)

    # dfs = {k:v for d in dfs for k,v in d.items()}
    logger.info(f"Have access to {env.NUM_CPUS} CPU.")
    for d in dfs:
        f, data = next(iter(d.items()))
        logger.info(f"Scoring forecasts in `{f}`...")

        # data = download_and_read_processed_forecast_file(filename=f)
        if not data or not isinstance(data, dict):
            logger.warning(f"Problem processing {f}. First `continue`.")
            continue

        organization = data.get("organization")
        model = data.get("model")
        question_set_filename = data.get("question_set")
        forecast_due_date = data.get("forecast_due_date")
        forecasts = data.get("forecasts")
        if (
            not organization
            or not model
            or not question_set_filename
            or not forecast_due_date
            or not forecasts
        ):
            logger.warning(f"Problem processing {f}. Second `continue`.")
            continue

        df = pd.DataFrame(forecasts)
        if df.empty:
            logger.warning(f"Problem processing {f}. Third `continue`.")
            continue

        df_sanity_check = df["score"] - ((df["forecast"] - df["resolved_to"]) ** 2)
        mask_sanity_check = df_sanity_check.abs() > 1e-8
        if any(mask_sanity_check):
            print(df_sanity_check[mask_sanity_check])
            raise ValueError(f"Sanity Check failed for {f}. Should be close to 0.")

        df = resolution.make_columns_hashable(df)
        df["resolution_date"] = pd.to_datetime(df["resolution_date"]).dt.date
        df["forecast_due_date"] = pd.to_datetime(df["forecast_due_date"]).dt.date
        df["horizon"] = (df["resolution_date"] - df["forecast_due_date"]).apply(
            lambda delta: delta.days
        )

        masks = get_masks(df)
        df = df[masks["data"] | masks["market"]].reset_index(drop=True)

        org_and_model = {"organization": organization, "model": model}
        is_human_forecast_set = (
            org_and_model == SUPERFORECASTER_MODEL or org_and_model == GENERAL_PUBLIC_MODEL
        )
        pprint(org_and_model)
        if not is_human_forecast_set:
            add_to_llm_leaderboard(
                llm_leaderboard,
                org_and_model,
                df,
                forecast_due_date,
                question_set_filename=question_set_filename,
            )
        add_to_llm_and_human_leaderboard(
            llm_and_human_leaderboard,
            org_and_model,
            df,
            forecast_due_date,
            cache,
            question_set_filename=question_set_filename,
        )
        print()
        # add_to_llm_and_human_combo_leaderboards(
        #     llm_and_human_combo_leaderboard,
        #     org_and_model,
        #     df,
        #     forecast_due_date,
        #     cache,
        #     is_human_forecast_set,
        #     question_set_filename=question_set_filename,
        # )

    title = "Leaderboard: overall"
    tasks = [
        # {
        #     "leaderboard": llm_leaderboard["overall"],
        #     "title": title,
        #     "basename": "leaderboard_overall",
        # },
        # {
        #     "leaderboard": llm_and_human_leaderboard["overall"],
        #     "title": f"Human {title}",
        #     "basename": "human_leaderboard_overall_high_level",
        # },
        {
            "leaderboard": llm_and_human_leaderboard["overall"],
            "title": f"Human {title}",
            "basename": "human_leaderboard_overall",
        },
        # {
        #     "d": llm_and_human_leaderboard["7"],
        #     "title": f"Human {title} 7 day",
        #     "basename": "human_leaderboard_overall_7",
        # },
        # {
        #     "d": llm_and_human_leaderboard["30"],
        #     "title": f"Human {title} 30 day",
        #     "basename": "human_leaderboard_overall_30",
        # },
        # {
        #     "d": llm_and_human_leaderboard["90"],
        #     "title": f"Human {title} 90 day",
        #     "basename": "human_leaderboard_overall_90",
        # },
        # {
        #     "d": llm_and_human_leaderboard["180"],
        #     "title": f"Human {title} 180 day",
        #     "basename": "human_leaderboard_overall_180",
        # },
        # {
        #     "d": llm_and_human_combo_leaderboard["overall"],
        #     "title": f"Human Combo {title}",
        #     "basename": "human_combo_leaderboard_overall",
        # },
    ]

    logger.info(f"Using {env.NUM_CPUS} cpus for worker pool.")
    with Pool(processes=env.NUM_CPUS) as pool:
        results = pool.map(worker, tasks)

    files = {}
    for res in results:
        files.update(res)

    mirrors = keys.get_secret_that_may_not_exist("HUGGING_FACE_REPO_URL")
    mirrors = [mirrors] if mirrors else []
    git.clone_and_push_files(
        repo_url=keys.API_GITHUB_DATASET_REPO_URL,
        files=files,
        commit_message="leaderboard: automatic update html & csv files.",
        mirrors=mirrors,
    )


if __name__ == "__main__":
    driver(None)

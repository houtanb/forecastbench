"""Create leaderboard."""

import json
import logging
import os
import pickle
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from multiprocessing import Pool

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
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
BASELINE_ORG_MODEL = {"organization": constants.BENCHMARK_NAME, "model": "Naive Forecast"}
SUPERFORECASTER_MODEL = {
    "organization": constants.BENCHMARK_NAME,
    "model": "Superforecaster median forecast",
}
GENERAL_PUBLIC_MODEL = {"organization": constants.BENCHMARK_NAME, "model": "Public median forecast"}

CONFIDENCE_LEVEL = 0.95
LEADERBOARD_DECIMAL_PLACES = 3


def download_and_read_processed_forecast_file(filename):
    """Download forecast file."""
    with tempfile.NamedTemporaryFile(dir="/tmp", delete=True) as tmp:
        gcp.storage.download(
            bucket_name=env.PROCESSED_FORECAST_SETS_BUCKET,
            filename=filename,
            local_filename=tmp.name,
        )
        with open(tmp.name, "r", encoding="utf-8") as f:
            data = json.load(f)

    return {filename: data}


def download_question_set_save_in_cache(forecast_due_date, cache):
    """Time-saving function to only download files once per run.

    Save question files in cache.
    """
    cache.setdefault(forecast_due_date, {})

    for human_or_llm in ["human", "llm"]:
        if human_or_llm not in cache[forecast_due_date]:
            filename = f"{forecast_due_date}-{human_or_llm}.json"
            cache[forecast_due_date][human_or_llm] = resolution.download_and_read_question_set_file(
                filename=filename
            )


def get_masks(df):
    """Return the data and market masks for the given dataframe."""
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


def get_df_info(df):
    """Provide information about the forecasts."""
    masks = get_masks(df)
    return {
        "n_data": len(df[masks["data"]]),
        "n_market": len(df[masks["market"]]),
        "n_market_resolved": len(df[masks["market_resolved"]]),
        "n_market_unresolved": len(df[masks["market_unresolved"]]),
        "n_overall": len(df),
        "pct_imputed_data": df[masks["data"]]["imputed"].sum() / len(df[masks["data"]]) * 100,
        "pct_imputed_market": df[masks["market"]]["imputed"].sum() / len(df[masks["market"]]) * 100,
        "pct_imputed_market_resolved": (
            df[masks["market_resolved"]]["imputed"].sum() / len(df[masks["market_resolved"]]) * 100
            if len(df[masks["market_resolved"]])
            else 0
        ),
        "pct_imputed_market_unresolved": df[masks["market_unresolved"]]["imputed"].sum()
        / len(df[masks["market_unresolved"]])
        * 100,
        "pct_imputed_overall": df["imputed"].sum() / len(df) * 100,
    }


def add_to_leaderboard(leaderboard, org_and_model, df, forecast_due_date, question_set_filename):
    """Add scores to the leaderboard."""
    leaderboard.setdefault("overall", [])
    leaderboard["overall"].append(
        org_and_model | {"forecast_due_date": forecast_due_date, "df": df.copy()} | get_df_info(df)
    )


def add_to_llm_leaderboard_include_combos(*args, **kwargs):
    """Wrap `add_to_leaderboard` for easy reading of driver."""
    add_to_leaderboard(*args, **kwargs)


def has_too_many_imputed(df, org_and_model) -> bool:
    """Determine whether or not to include this model.

    * Don't include models with more than D percent imputed data questions and M percent imputed
      market questions.
    * Always include ForecastBench models
    """
    if org_and_model["organization"] == constants.BENCHMARK_NAME:
        return False

    MIN_MARKET_RESOLVED = 10
    MIN_DATA_MISSING_PCT = 0.05
    MIN_MARKET_MISSING_PCT = 0.01
    masks = get_masks(df)
    df_data = df[masks["data"]]
    df_market = df[masks["market"]]
    df_market_resolved = df[masks["market_resolved"]]

    if len(df_market_resolved) < MIN_MARKET_RESOLVED or (
        df_data["imputed"].mean() > MIN_DATA_MISSING_PCT
        or df_market["imputed"].mean() > MIN_MARKET_MISSING_PCT
    ):
        logger.info(f"DROPPING {org_and_model}")
        logger.info(f" * % imputed data: {round(df_data['imputed'].mean() * 100, 2)}")
        logger.info(f" * % imputed market: {round(df_market['imputed'].mean() * 100, 2)}")
        logger.info(f" * N market resolved:: {len(df_market_resolved)}")
        return True
    return False


def add_to_llm_leaderboard(
    leaderboard, org_and_model, df, forecast_due_date, question_set_filename
):
    """Create the LLM leaderbeard.

    * Remove combination questions before including in the LLM leaderboard
    """
    df_no_combos = df[df["direction"] == ()].reset_index(drop=True)

    if not has_too_many_imputed(df_no_combos, org_and_model):
        add_to_leaderboard(
            leaderboard=leaderboard,
            org_and_model=org_and_model,
            df=df_no_combos,
            forecast_due_date=forecast_due_date,
            question_set_filename=question_set_filename,
        )


def add_to_human_leaderboard(
    leaderboard, org_and_model, df, forecast_due_date, cache, question_set_filename
):
    """Parse the forecasts to include only those questions that were in the human question set."""
    df_no_combos = df[df["direction"] == ()].reset_index(drop=True)

    if not has_too_many_imputed(df_no_combos, org_and_model):
        add_to_leaderboard(
            leaderboard=leaderboard,
            org_and_model=org_and_model,
            df=df_no_combos,
            forecast_due_date=forecast_due_date,
            question_set_filename=question_set_filename,
        )


def make_and_upload_html_table(df, title, basename):
    """Make and upload HTLM leaderboard."""
    # Replace NaN with empty strings for display
    logger.info(f"Making HTML leaderboard file: {title} {basename}.")
    df = df.fillna("")

    # Add ranking
    df = df.sort_values(by=["adj_brier_all"], ignore_index=True)
    df.insert(loc=0, column="Ranking", value="")
    df["score_diff"] = df["adj_brier_all"] - df["adj_brier_all"].shift(1)
    for index, row in df.iterrows():
        if row["score_diff"] != 0:
            prev_rank = index + 1
        df.loc[index, "Ranking"] = prev_rank
    df.drop(columns="score_diff", inplace=True)

    # Round columns to 3 decimal places
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].round(3)

    for col in [
        "n_adj_brier_dataset",
        "n_adj_brier_market",
        "n_adj_brier_all",
    ]:
        df[col] = df[col].astype(int)

    df = df[
        [
            "Ranking",
            "organization",
            "model",
            "adj_brier_dataset",
            "n_adj_brier_dataset",
            "adj_brier_market",
            "n_adj_brier_market",
            "adj_brier_all",
            "n_adj_brier_all",
        ]
    ]
    df = df.rename(
        columns={
            "organization": "Organization",
            "model": "Model",
            "adj_brier_dataset": "Dataset",
            "n_adj_brier_dataset": "N (dataset)",
            "adj_brier_market": "Market (resolved)",
            "n_adj_brier_market": "N (market)",
            "adj_brier_all": "Overall",
            "n_adj_brier_all": "N",
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
              <li><b>Overall Score</b>: The average of the Dataset Score and the Market Score
                                        (overall) columns</li>
              <li><b>Overall Score 95% CI</b>: The 95% confidence interval for the Overall
                                               Score</li>
              <li><b>Pairwise p-value comparing to No. 1 (bootstrapped)</b>: The p-value calculated
                              by bootstrapping the differences in overall score between each model
                              and the best forecaster (the group with rank 1) under the null
                              hypothesis that there's no difference.</li>
              <li><b>Pct. more accurate than No. 1</b>: The percent of questions where this
                              forecaster had a better overall score than the best forecaster (with
                              rank 1)</li>
              <li><b>Pct. imputed</b>: The percent of questions for which this forecaster did not
                              provide a forecast and hence had a forecast value imputed (0.5 for
                              dataset questions and the aggregate human forecast on the forecast
                              due date for questions sourced from prediction markets or forecast
                              aggregation platforms)</li>
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
        classes="table table-striped table-bordered", index=False, table_id="myTable"
    )
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
        $(document).ready(function() {
            var table = $('#myTable').DataTable({
                "pageLength": -1,
                "lengthMenu": [[-1], ["All"]],
                "order": [[7, 'asc']],
                "paging": false,
                "info": false,
                "search": {
                    "regex": true,
                    "smart": true
                },
                "columnDefs": [
                    {
                        "targets": '_all',
                        "searchable": true
                    }
                ]
            });
        table.column(7).nodes().to$().addClass('highlight');
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
    gcp.storage.upload(
        bucket_name=env.PUBLIC_RELEASE_BUCKET,
        local_filename=local_filename_html,
        destination_folder=f"{destination_folder}/html",
    )
    gcp.storage.upload(
        bucket_name=env.PUBLIC_RELEASE_BUCKET,
        local_filename=local_filename_csv,
        destination_folder=f"{destination_folder}/csv",
    )

    return {
        local_filename_html: f"{destination_folder}/html/{basename}.html",
        local_filename_csv: f"{destination_folder}/csv/{basename}.csv",
    }


def combine_forecasting_rounds(leaderboard, leaderboard_type):
    """Combine all forecasts into one dataframe."""
    forecasts = []
    for entry in leaderboard:
        df_tmp = entry["df"].copy()
        df_tmp["model"] = entry["model"]
        df_tmp["organization"] = entry["organization"]
        # Drop unresolved market questions
        masks = get_masks(df_tmp)
        df_tmp = df_tmp[~masks["market_unresolved"]].reset_index(drop=True)
        forecasts.append(df_tmp)
    df = pd.concat(forecasts).reset_index(drop=True)
    df["resolution_date"] = pd.to_datetime(df["resolution_date"])
    df["forecast_due_date"] = pd.to_datetime(df["forecast_due_date"])
    df["primary_key"] = (
        df["forecast_due_date"].astype(str)
        + "_"
        + df["source"].astype(str)
        + "_"
        + df["id"].astype(str)
        + "_"
        + df["horizon"].astype(str)
    )

    # If a question was asked of the same model in more than one question set, only keep its
    # forecast from the first question set.
    df = df.sort_values(by=["model", "forecast_due_date"], ascending=True).drop_duplicates(
        subset=["model", "source", "id", "horizon"], keep="first", ignore_index=True
    )

    return df


def calculate_difficulty_adjusted_brier_score1(df):
    res = smf.ols("score ~ -1 + C(model) + C(primary_key)", data=df).fit()
    params = res.params
    qe = {
        n.split("[")[1].rstrip("]").lstrip("T."): v
        for n, v in params.items()
        if n.startswith("C(primary_key)")
    }
    fe = {n.split("[")[1].rstrip("]"): v for n, v in params.items() if n.startswith("C(model)")}
    df["question_effect"] = df["primary_key"].map(qe)
    df["forecaster_effect"] = df["model"].map(fe)
    df["adjusted_brier"] = df["score"] - df["question_effect"]
    return df


def calculate_difficulty_adjusted_brier_score(df):
    """Use 2-way fixed effects model to calclutae the adjusted Brier score."""
    N_ITERATIONS = 200
    df_adjusted_brier = df[["organization", "model", "primary_key", "score"]].copy()
    df_adjusted_brier["question_effect"] = 0.0
    df_adjusted_brier["forecaster_effect"] = 0.0

    tol = 1e-5
    converged = False
    prev_q = df_adjusted_brier["question_effect"].copy()
    prev_f = df_adjusted_brier["forecaster_effect"].copy()

    for _ in range(N_ITERATIONS):
        # Question effects
        df_tmp = (
            df_adjusted_brier.groupby("primary_key")
            .apply(
                lambda x: pd.Series(
                    {"question_effect": (x["score"] - x["forecaster_effect"]).mean()},
                ),
                include_groups=False,
            )
            .reset_index()
        )
        df_adjusted_brier = df_adjusted_brier.drop("question_effect", axis=1)
        df_adjusted_brier = pd.merge(df_adjusted_brier, df_tmp, on="primary_key", how="left")

        # Forecaster effects
        df_tmp = (
            df_adjusted_brier.groupby("model")
            .apply(
                lambda x: pd.Series(
                    {"forecaster_effect": (x["score"] - x["question_effect"]).mean()}
                ),
                include_groups=False,
            )
            .reset_index()
        )
        df_adjusted_brier = df_adjusted_brier.drop("forecaster_effect", axis=1)
        df_adjusted_brier = pd.merge(df_adjusted_brier, df_tmp, on="model", how="left")

        delta_q = (df_adjusted_brier["question_effect"] - prev_q).abs().max()
        delta_f = (df_adjusted_brier["forecaster_effect"] - prev_f).abs().max()
        if max(delta_q, delta_f) < tol:
            converged = True
            logger.info(colored(f"Converged at iteration {_}", "green"))
            break
        prev_q = df_adjusted_brier["question_effect"].copy()
        prev_f = df_adjusted_brier["forecaster_effect"].copy()

    if not converged:
        logger.warning(
            colored("Convergence not reached when calculating difficulty adjusted Brier!", "red")
        )

    df_adjusted_brier["adjusted_brier"] = (
        df_adjusted_brier["score"] - df_adjusted_brier["question_effect"]
    )

    return pd.merge(
        df,
        df_adjusted_brier[["organization", "model", "primary_key", "adjusted_brier"]],
        on=[
            "organization",
            "model",
            "primary_key",
        ],
        how="left",
        validate="1:1",
    )


def bootstrap_adjusted_brier_ci(df, n_bootstraps=100, ci=0.95):
    models = df["model"].unique()
    boot_results = {model: [] for model in models}

    for _ in range(n_bootstraps):
        sample = df.sample(frac=1, replace=True)
        print(sample)
        boot_df = calculate_difficulty_adjusted_brier_score(sample)
        means = boot_df.groupby("model")["adjusted_brier"].mean()
        for model, val in means.items():
            boot_results[model].append(val)

    ci_lower = {}
    ci_upper = {}
    alpha = 1 - ci
    for model, values in boot_results.items():
        ci_lower[model] = np.percentile(values, 100 * (alpha / 2))
        ci_upper[model] = np.percentile(values, 100 * (1 - alpha / 2))

    return pd.DataFrame(
        {
            "model": list(ci_lower.keys()),
            "ci_lower": list(ci_lower.values()),
            "ci_upper": list(ci_upper.values()),
        }
    )


def make_leaderboard(leaderboard, title, basename, leaderboard_type):
    """Make leaderboard."""
    logger.info(colored(f"Making leaderboard: {title}", "red"))

    df = combine_forecasting_rounds(leaderboard, leaderboard_type)
    df_leaderboard = None
    for question_set in ["all", "dataset", "market"]:
        if question_set == "market":
            df_tmp = df[df["source"].isin(question_curation.MARKET_SOURCES)]
        elif question_set == "dataset":
            df_tmp = df[df["source"].isin(question_curation.DATA_SOURCES)]
        else:
            df_tmp = df
        # df_tmp1 = df_tmp.copy()
        df_tmp = calculate_difficulty_adjusted_brier_score1(df_tmp)
        # df_tmp_ci = bootstrap_adjusted_brier_ci(df_tmp1)
        col_name = f"adj_brier_{question_set}"
        df_tmp[col_name] = df_tmp["adjusted_brier"]
        grouped = df_tmp[["organization", "model", col_name]].groupby(["organization", "model"])
        mean_df = grouped.mean()
        count_df = grouped.count().rename(columns={col_name: f"n_{col_name}"})
        df_leaderboard_tmp = (
            pd.concat(
                [
                    mean_df,
                    count_df,
                ],
                axis=1,
            )
            .reset_index()
            .sort_values(col_name)
            .reset_index(drop=True)
        )

        # df_leaderboard_tmp = pd.merge(
        #     df_leaderboard_tmp,
        #     df_tmp_ci,
        #     on=["model"],
        #     how="left",
        #     validate="1:1",
        # )
        if df_leaderboard is None:
            df_leaderboard = df_leaderboard_tmp
        else:
            df_leaderboard = pd.merge(
                df_leaderboard,
                df_leaderboard_tmp,
                on=["organization", "model"],
                how="left",
            )

    print(df_leaderboard.sort_values("adj_brier_all"))
    print()
    #    print(df_leaderboard.sort_values("adjusted_brier_dataset"))
    #    print()
    #    print(df_leaderboard.sort_values("adjusted_brier_market"))
    # sys.exit()

    files = make_and_upload_html_table(
        df=df_leaderboard,
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
            leaderboard_type=task["leaderboard_type"],
        )
    except Exception as e:
        msg = f"Error processing task {task['title']}: {e}"
        logger.error(msg)
        raise ValueError(msg)


@decorator.log_runtime
def driver(_):
    """Create new leaderboard."""
    # Shortcut while testing
    if os.path.exists("leaderboard_human.pkl") and os.path.exists("leaderboard_llm.pkl"):

        def read_leaderboard(f):
            with open(f, "rb") as file:
                leaderboard = pickle.load(file)
            return leaderboard

        # # TODO: Look into API calls for these models:
        # Removing:  Claude-2.1 (scratchpad)
        # Removing:  Claude-2.1 (scratchpad with freeze values)
        # Removing:  Mixtral-8x7B-Instruct-V0.1 (superforecaster with news 2)
        # Removing:  DeepSeek-R1 (zero shot)
        # Removing:  DeepSeek-R1 (zero shot with freeze values)
        # Removing:  DeepSeek-R1 (zero shot)
        # Removing:  DeepSeek-R1 (zero shot with freeze values)
        # Removing:  DeepSeek-R1 (zero shot)
        # Removing:  DeepSeek-R1 (zero shot with freeze values)
        # Removing:  Gemini-2.5-Pro-Exp-03-25 (scratchpad)
        # Removing:  Gemini-2.5-Pro-Exp-03-25 (scratchpad with freeze values)
        # Removing:  Gemini-2.5-Pro-Exp-03-25 (zero shot)
        # Removing:  Gemini-2.5-Pro-Exp-03-25 (zero shot with freeze values)
        # Removing:  QwQ-32B-Preview (zero shot)
        # Removing:  QwQ-32B-Preview (zero shot with freeze values)

        title = "Leaderboard: overall"
        make_leaderboard(
            leaderboard=read_leaderboard("leaderboard_human.pkl"),
            title=f"Human {title}",
            basename="human_leaderboard_overall",
            leaderboard_type="human",
        )
        # make_leaderboard(
        #     leaderboard=read_leaderboard("leaderboard_llm.pkl"),
        #     title=title,
        #     basename="leaderboard_overall",
        #     leaderboard_type="llm",
        # )
        return

    cache = {}
    llm_leaderboard = {}
    human_leaderboard = {}
    files = gcp.storage.list(env.PROCESSED_FORECAST_SETS_BUCKET)
    files = [file for file in files if file.endswith(".json") and not file.startswith("2024-12-08")]
    # files = [
    #     "2024-07-21/2024-07-21.ForecastBench.always-0.5.json",
    #     "2024-07-21/2024-07-21.ForecastBench.always-0.json",
    #     "2024-07-21/2024-07-21.ForecastBench.always-1.json",
    #     "2024-07-21/2024-07-21.ForecastBench.human_public.json",
    #     "2024-07-21/2024-07-21.ForecastBench.human_super.json",
    #     "2024-07-21/2024-07-21.ForecastBench.imputed-forecaster.json",
    #     "2024-07-21/2024-07-21.ForecastBench.naive-forecaster.json",
    #     "2024-07-21/2024-07-21.ForecastBench.random-uniform.json",
    #     "2024-07-21/2024-07-21.Anthropic.claude_3p5_sonnet_scratchpad_with_freeze_values.json",
    #     "2024-07-21/2024-07-21.OpenAI.gpt_4_turbo_0409_scratchpad_with_freeze_values.json",
    #     "2024-07-21/2024-07-21.Qwen.qwen_1p5_110b_scratchpad.json",
    #     "2025-03-02/2025-03-02.ForecastBench.naive-forecaster.json",
    #     "2025-03-02/2025-03-02.ForecastBench.always-0.json",
    #     "2025-03-02/2025-03-02.Anthropic.claude-3-5-sonnet-20240620_scratchpad_with_freeze_values.json",
    # ]

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

    logger.info(f"Have access to {env.NUM_CPUS} CPU.")
    for d in dfs:
        f, data = next(iter(d.items()))
        logger.info(f"Scoring forecasts in `{f}`...")

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
        is_human_forecast_set = organization == constants.BENCHMARK_NAME and (
            model in [SUPERFORECASTER_MODEL["model"], GENERAL_PUBLIC_MODEL["model"]]
        )

        if not is_human_forecast_set:
            add_to_llm_leaderboard(
                llm_leaderboard,
                org_and_model,
                df,
                forecast_due_date,
                question_set_filename=question_set_filename,
            )
        add_to_human_leaderboard(
            human_leaderboard,
            org_and_model,
            df,
            forecast_due_date,
            cache,
            question_set_filename=question_set_filename,
        )

    title = "Leaderboard"
    tasks = [
        {
            "leaderboard": llm_leaderboard["overall"],
            "title": f"LLM {title}",
            "basename": f"llm_{title.lower()}",
            "leaderboard_type": "llm",
        },
        {
            "leaderboard": human_leaderboard["overall"],
            "title": f"Human {title}",
            "basename": f"human_{title.lower()}",
            "leaderboard_type": "human",
        },
    ]

    with open("leaderboard_llm.pkl", "wb") as file:
        pickle.dump(llm_leaderboard["overall"], file)

    with open("leaderboard_human.pkl", "wb") as file:
        pickle.dump(human_leaderboard["overall"], file)

    logger.info(f"Using {env.NUM_CPUS} cpus for worker pool.")
    with Pool(processes=min(len(tasks), env.NUM_CPUS)) as pool:
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

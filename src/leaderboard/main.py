"""Create leaderboard."""

import json
import logging
import math
import multiprocessing as mp
import os
import pickle
import sys
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from functools import partial
from multiprocessing.pool import Pool as BasePool
from pprint import pprint

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from termcolor import colored
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from helpers import (  # noqa: E402
    constants,
    decorator,
    env,
    question_curation,
    resolution,
)

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))


from utils import gcp  # noqa: E402


class NoDaemonProcess(mp.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass


class NonDaemonPool(BasePool):
    Process = NoDaemonProcess


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
    with tempfile.NamedTemporaryFile(dir="/tmp", delete=True) as tmp:
        gcp.storage.download(
            bucket_name=env.PROCESSED_FORECAST_SETS_BUCKET,
            filename=filename,
            local_filename=tmp.name,
        )
        with open(tmp.name, "r", encoding="utf-8") as f:
            data = json.load(f)

    return {filename: data}


def get_masks(df):
    """Return the data and market masks for the given dataframe."""
    masks = {}

    resolved_mask = df["resolved"].astype(bool)

    masks["data"] = df["source"].isin(question_curation.DATA_SOURCES) & resolved_mask

    return masks


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


def add_to_leaderboard(leaderboard, org_and_model, df, forecast_due_date, question_set_filename):
    """Add scores to the leaderboard."""
    leaderboard.setdefault("overall", [])
    leaderboard["overall"].append(
        org_and_model | {"forecast_due_date": forecast_due_date, "df": df.copy()} | get_df_info(df)
    )


def add_to_llm_leaderboard_include_combos(*args, **kwargs):
    """Wrap `add_to_leaderboard` for easy reading of driver."""
    add_to_leaderboard(*args, **kwargs)


def add_to_llm_leaderboard(
    leaderboard, org_and_model, df, forecast_due_date, question_set_filename
):
    """Create the LLM leaderbeard.

    * Remove combination questions before including in the LLM leaderboard
    """
    df_no_combos = df[df["direction"] == ()].reset_index(drop=True)

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
    download_question_set_save_in_cache(forecast_due_date, cache)
    # df_human_question_set = cache[forecast_due_date]["human"].copy()
    # df_only_human_question_set = pd.merge(
    #     df,
    #     df_human_question_set[["id", "source"]],
    #     on=["id", "source"],
    # ).reset_index(drop=True)

    add_to_leaderboard(
        leaderboard=leaderboard,
        org_and_model=org_and_model,
        df=df,  # _only_human_question_set,
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
            "elo_data",
        ],
        ascending=[
            False,
        ],
        ignore_index=True,
    )

    df["Ranking"] = df["final_ranking"]

    # Round columns to 3 decimal places
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].round(LEADERBOARD_DECIMAL_PLACES)

    df["elo_data"] = df["elo_data"].round().astype(int)
    df["rating_q025"] = df["rating_q025"].round().astype(int)
    df["rating_q975"] = df["rating_q975"].round().astype(int)

    for c in [
        "pct_imputed_data",
    ]:
        df[c] = df[c].round().astype(int).astype(str) + "%"

    for c in [
        "n_data",
    ]:
        df[c] = df[c].astype(int)

    df["data_info"] = df.apply(lambda row: (row["n_data"], row["pct_imputed_data"]), axis=1)

    df["95_cis"] = "[" + df["rating_q025"].astype(str) + ", " + df["rating_q975"].astype(str) + "]"

    df = df[
        [
            "Ranking",
            "organization",
            "model",
            "forecast_due_date",
            "elo_data",
            "data_info",
            "95_cis",
        ]
    ]
    df = df.rename(
        columns={
            "organization": "Organization",
            "model": "Model",
            "forecast_due_date": "Forecast due date(s)",
            "elo_data": "Score dataset",
            "data_info": "Data info",
            "95_cis": "95% CI",
        }
    )

    column_descriptions = ""
    """
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

    ORDER_COL_IDX = 4
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
                "stateSave": true,
                "search": {
                    "regex": true,
                    "smart": true
                },
                "columnDefs": [
                    {
                        "targets": 0,
                        "className": "right-align"
                    },
                    {
                        "targets": '_all',
                        "searchable": true
                    },
                    {
                        "targets": '_all',
                        "orderSequence": ["asc", "desc"]
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


def print_all(df):
    """Print all rows and columns of dataframe."""
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(df)


def compute_mle_elo_orig(df, SCALE=400, BASE=10, INIT_RATING=1000, sample_weight=None):
    """
    Pulled directly from https://colab.research.google.com/drive/1KdwokPjirkTmpO_P1WByFNFiqxWQquwH#scrollTo=mSizG3Pzglte # noqa: B950
    """
    from sklearn.linear_model import LogisticRegression

    ptbl_a_win = pd.pivot_table(
        df[df["winner"] == "model_a"],
        index="model_a",
        columns="model_b",
        aggfunc="size",
        fill_value=0,
    )
    # if no tie, create a zero matrix
    if sum(df["winner"].isin(["tie", "tie (bothbad)"])) == 0:
        ptbl_tie = pd.DataFrame(0, index=ptbl_a_win.index, columns=ptbl_a_win.columns)
    else:
        ptbl_tie = pd.pivot_table(
            df[df["winner"].isin(["tie", "tie (bothbad)"])],
            index="model_a",
            columns="model_b",
            aggfunc="size",
            fill_value=0,
        )
        ptbl_tie = ptbl_tie + ptbl_tie.T

    ptbl_b_win = pd.pivot_table(
        df[df["winner"] == "model_b"],
        index="model_a",
        columns="model_b",
        aggfunc="size",
        fill_value=0,
    )
    ptbl_win = ptbl_a_win * 2 + ptbl_b_win.T * 2 + ptbl_tie

    models = pd.Series(np.arange(len(ptbl_win.index)), index=ptbl_win.index)

    p = len(models)
    X = np.zeros([p * (p - 1) * 2, p])
    Y = np.zeros(p * (p - 1) * 2)

    cur_row = 0
    sample_weights = []
    for m_a in ptbl_win.index:
        for m_b in ptbl_win.columns:
            if m_a == m_b:
                continue
            # if nan skip
            if math.isnan(ptbl_win.loc[m_a, m_b]) or math.isnan(ptbl_win.loc[m_b, m_a]):
                continue
            X[cur_row, models[m_a]] = +math.log(BASE)
            X[cur_row, models[m_b]] = -math.log(BASE)
            Y[cur_row] = 1.0
            sample_weights.append(ptbl_win.loc[m_a, m_b])

            X[cur_row + 1, models[m_a]] = math.log(BASE)
            X[cur_row + 1, models[m_b]] = -math.log(BASE)
            Y[cur_row + 1] = 0.0
            sample_weights.append(ptbl_win.loc[m_b, m_a])
            cur_row += 2
    X = X[:cur_row]
    Y = Y[:cur_row]

    lr = LogisticRegression(fit_intercept=False, penalty=None, tol=1e-6)
    lr.fit(X, Y, sample_weight=sample_weights)
    elo_scores = SCALE * lr.coef_[0] + INIT_RATING
    if "mixtral-8x7b-instruct-v0.1" in models.index:
        elo_scores += 1114 - elo_scores[models["mixtral-8x7b-instruct-v0.1"]]
    return pd.Series(elo_scores, index=models.index).sort_values(ascending=False)


def compute_mle_elo(df, SCALE=400, BASE=10, INIT_RATING=1000, sample_weight=None):
    """
    Compute Elo-like scores using the Bradley-Terry model, as in chatbot arena.

    Modified from https://colab.research.google.com/drive/1KdwokPjirkTmpO_P1WByFNFiqxWQquwH#scrollTo=mSizG3Pzglte # noqa: B950
    """
    unique_questions = pd.concat(
        [
            df[df["source"].isin(question_curation.DATA_SOURCES)].drop_duplicates(
                subset=[
                    "forecast_due_date",
                    "id",
                    "source",
                    "direction",
                    "resolution_date_a",
                    "resolution_date_b",
                ],
                ignore_index=True,
            ),
            df[df["source"].isin(question_curation.MARKET_SOURCES)].drop_duplicates(
                subset=[
                    "forecast_due_date",
                    "id",
                    "source",
                    "direction",
                ],
                ignore_index=True,
            ),
        ]
    )
    unique_questions["type"] = unique_questions["source"].apply(
        lambda src: (
            "data"
            if src in question_curation.DATA_SOURCES
            else ("market" if src in question_curation.MARKET_SOURCES else "unknown")
        )
    )
    if (unique_questions["type"] == "unknown").any():
        raise ValueError("Should either be market or data (1).")

    question_type_lookup = {}
    question_n_lookup = {}
    for forecast_due_date in unique_questions["forecast_due_date"].unique():
        df_tmp = unique_questions[unique_questions["forecast_due_date"] == forecast_due_date]
        n_dataset_questions = (df_tmp["type"] == "data").sum()
        n_market_questions = (df_tmp["type"] == "market").sum()
        question_n_lookup[(forecast_due_date, "market")] = (n_dataset_questions, n_market_questions)
        question_type_lookup[(forecast_due_date, "market")] = 0.5 if n_market_questions else 0.0
        if n_market_questions and n_dataset_questions:
            question_type_lookup[(forecast_due_date, "data")] = 0.5 * (
                n_market_questions / n_dataset_questions
            )
        else:
            question_type_lookup[(forecast_due_date, "data")] = 0.5 if n_dataset_questions else 0.0

    df["type"] = df["source"].apply(
        lambda src: (
            "data"
            if src in question_curation.DATA_SOURCES
            else ("market" if src in question_curation.MARKET_SOURCES else "unknown")
        )
    )
    df["per_row_weight"] = df.set_index(["forecast_due_date", "type"]).index.map(
        question_type_lookup
    )

    df["score_diff"] = abs(df["score_a"] - df["score_b"])
    df["scale_score_diff"] = 1 + df["score_diff"] ** 2
    df["scale"] = df["per_row_weight"] * df["scale_score_diff"]

    ptbl_a_win = pd.pivot_table(
        df[df["winner"] == "model_a"],
        index="model_a",
        columns="model_b",
        values="scale",
        aggfunc="sum",
        fill_value=0,
    )

    # if no tie, create a zero matrix
    if sum(df["winner"].isin(["tie", "tie (bothbad)"])) == 0:
        ptbl_tie = pd.DataFrame(0, index=ptbl_a_win.index, columns=ptbl_a_win.columns)
    else:
        ptbl_tie = pd.pivot_table(
            df[df["winner"].isin(["tie", "tie (bothbad)"])],
            index="model_a",
            columns="model_b",
            values="per_row_weight",
            aggfunc="sum",
            fill_value=0,
        )

        ptbl_tie = ptbl_tie + ptbl_tie.T

    ptbl_b_win = pd.pivot_table(
        df[df["winner"] == "model_b"],
        index="model_a",
        columns="model_b",
        values="scale",
        aggfunc="sum",
        fill_value=0,
    )

    ptbl_win = ptbl_a_win * 2 + ptbl_b_win.T * 2 + ptbl_tie

    models = pd.Series(np.arange(len(ptbl_win.index)), index=ptbl_win.index)

    p = len(models)
    X = np.zeros([p * (p - 1) * 2, p])
    Y = np.zeros(p * (p - 1) * 2)

    cur_row = 0
    sample_weights = []
    for m_a in ptbl_win.index:
        for m_b in ptbl_win.columns:
            if m_a == m_b:
                continue

            if math.isnan(ptbl_win.loc[m_a, m_b]) or math.isnan(ptbl_win.loc[m_b, m_a]):
                continue

            X[cur_row, models[m_a]] = +math.log(BASE)
            X[cur_row, models[m_b]] = -math.log(BASE)
            Y[cur_row] = 1.0
            sample_weights.append(ptbl_win.loc[m_a, m_b])

            X[cur_row + 1, models[m_a]] = math.log(BASE)
            X[cur_row + 1, models[m_b]] = -math.log(BASE)
            Y[cur_row + 1] = 0.0
            sample_weights.append(ptbl_win.loc[m_b, m_a])
            cur_row += 2

    X = X[:cur_row]
    Y = Y[:cur_row]

    lr = LogisticRegression(fit_intercept=False, penalty=None, tol=1e-6)
    lr.fit(X, Y, sample_weight=sample_weights)
    elo_scores = SCALE * lr.coef_[0] + INIT_RATING
    if "mixtral-8x7b-instruct-v0.1" in models.index:
        elo_scores += 1114 - elo_scores[models["mixtral-8x7b-instruct-v0.1"]]
    return pd.Series(elo_scores, index=models.index).sort_values(ascending=False)


def compute_bootstrap_bt(
    battles,
    num_round=100,
    base=10.0,
    scale=400.0,
    init_rating=1000.0,
    tol=1e-6,
    num_cpu=env.NUM_CPUS,
):
    matchups, outcomes, models, weights = preprocess_for_bt(battles)
    # bootstrap sample the unique outcomes and their counts directly using the multinomial distribution
    rng = np.random.default_rng(seed=0)
    idxs = rng.multinomial(n=len(battles), pvals=weights / weights.sum(), size=(num_round))
    # only the distribution over their occurance counts changes between samples (and it can be 0)
    boot_weights = idxs.astype(np.float64) / len(battles)

    # the only thing different across samples is the distribution of weights
    bt_fn = partial(fit_bt, matchups, outcomes, n_models=len(models), alpha=np.log(base), tol=tol)
    with mp.Pool(num_cpu if num_cpu else os.cpu_count()) as pool:
        results = list(tqdm(pool.imap_unordered(bt_fn, boot_weights), total=num_round))

    ratings = np.array(results)
    scaled_ratings = scale_and_offset(ratings, models, scale, init_rating)
    df = pd.DataFrame(scaled_ratings, columns=models)
    return df[df.median().sort_values(ascending=False).index]


def combine_rounds(leaderboard, mask_name=None):
    """Combine dataframes for models across forecasting rounds."""

    def mask_model(df):
        return df[get_masks(df)[mask_name]] if mask_name else df

    def get_model_name(value):
        return value["organization"] + ";" + value["model"]

    def who_won(row):
        if row["score_a"] < row["score_b"]:
            return "model_a"
        elif row["score_a"] > row["score_b"]:
            return "model_b"
        return "tie"

    combined_data = []
    for i, value_a in enumerate(leaderboard):
        df_model_a = mask_model(value_a["df"].copy())
        df_model_a["model"] = get_model_name(value_a)
        print(value_a["organization"], value_a["model"], value_a["forecast_due_date"])
        index = 0
        for _, value_b in enumerate(leaderboard[i + 1 :]):
            if value_a["forecast_due_date"] == value_b["forecast_due_date"] and not (
                value_a["organization"] == value_b["organization"]
                and value_a["model"] == value_b["model"]
            ):
                index += 1
                df_model_b = mask_model(value_b["df"].copy())
                df_model_b["model"] = get_model_name(value_b)
                if df_model_a.empty or df_model_b.empty:
                    continue
                df_tmp = pd.merge(
                    df_model_a,
                    df_model_b,
                    on=[
                        "id",
                        "source",
                        "direction",
                        "horizon",
                        "forecast_due_date",
                    ],
                    suffixes=("_a", "_b"),
                ).reset_index(drop=True)
                df_tmp["winner"] = df_tmp.apply(who_won, axis=1)
                if df_tmp.empty:
                    raise ValueError("This merge should not result in an empty df.")

                combined_data.append(df_tmp)
        print(f"      **** compared to {index} models.")
    return pd.concat(combined_data, axis=0, ignore_index=True)


def get_matchups_models(df):
    n_rows = len(df)
    model_indices, models = pd.factorize(pd.concat([df["model_a"], df["model_b"]]))
    matchups = np.column_stack([model_indices[:n_rows], model_indices[n_rows:]])
    return matchups, models.to_list()


def preprocess_for_bt(df):
    """in BT we only need the unique (matchup,outcome) sets along with the weights of how often they occur"""
    n_rows = len(df)
    # the 3 columns of schedule represent: model_a id, model_b id, outcome_id
    schedule = np.full((n_rows, 3), fill_value=1, dtype=np.int32)
    # set the two model cols by mapping the model names to their int ids
    schedule[:, [0, 1]], models = get_matchups_models(df)
    # map outcomes to integers (must be same dtype as model ids so it can be in the same array)
    # model_a win -> 2, tie -> 1 (prefilled by default), model_b win -> 0
    schedule[df["winner"] == "model_a", 2] = 2
    schedule[df["winner"] == "model_b", 2] = 0
    # count the number of occurances of each observed result
    matchups_outcomes, weights = np.unique(schedule, return_counts=True, axis=0)
    matchups = matchups_outcomes[:, [0, 1]]
    # map 2 -> 1.0, 1 -> 0.5, 0 -> 0.0 which will be used as labels during optimization
    outcomes = matchups_outcomes[:, 2].astype(np.float64) / 2.0
    weights = weights.astype(np.float64)
    # each possible result is weighted according to number of times it occured in the dataset
    return matchups, outcomes, models, weights


def bt_loss_and_grad(ratings, matchups, outcomes, weights, alpha=1.0):
    matchup_ratings = ratings[matchups]
    logits = alpha * (matchup_ratings[:, 0] - matchup_ratings[:, 1])
    probs = expit(logits)
    # this form naturally counts a draw as half a win and half a loss
    loss = -((np.log(probs) * outcomes + np.log(1.0 - probs) * (1.0 - outcomes)) * weights).sum()
    matchups_grads = -alpha * (outcomes - probs) * weights
    model_grad = np.zeros_like(ratings)
    # aggregate gradients at the model level using the indices in matchups
    np.add.at(
        model_grad,
        matchups[:, [0, 1]],
        matchups_grads[:, None] * np.array([1.0, -1.0], dtype=np.float64),
    )
    return loss, model_grad


def fit_bt(matchups, outcomes, weights, n_models, alpha, tol=1e-6):
    initial_ratings = np.zeros(n_models, dtype=np.float64)
    result = minimize(
        fun=bt_loss_and_grad,
        x0=initial_ratings,
        args=(matchups, outcomes, weights, alpha),
        jac=True,
        method="L-BFGS-B",
        options={"disp": False, "maxiter": 100, "gtol": tol},
    )
    return result["x"]


def scale_and_offset(
    ratings,
    models,
    scale=400,
    init_rating=1000,
    baseline_model="mixtral-8x7b-instruct-v0.1",
    baseline_rating=1114,
):
    """convert ratings from the natural scale to the Elo rating scale with an anchored baseline"""
    scaled_ratings = (ratings * scale) + init_rating
    if baseline_model in models:
        baseline_idx = models.index(baseline_model)
        scaled_ratings += baseline_rating - scaled_ratings[..., [baseline_idx]]
    return scaled_ratings


def compute_bt(df, base=10.0, scale=400.0, init_rating=1000, tol=1e-6):
    matchups, outcomes, models, weights = preprocess_for_bt(df)
    ratings = fit_bt(matchups, outcomes, weights, len(models), math.log(base), tol)
    scaled_ratings = scale_and_offset(ratings, models, scale, init_rating=init_rating)
    return pd.Series(scaled_ratings, index=models).sort_values(ascending=False)


def compute_elos(leaderboard, leaderboard_type):
    """Compute the Elo scores."""

    def compute_chatbot_elos(filename, mask=None):
        if os.path.exists(filename):
            with open(filename, "rb") as file:
                df = pickle.load(file)
        else:
            df = combine_rounds(leaderboard, mask)
            with open(filename, "wb") as file:
                pickle.dump(df, file)

        bootstrap_df = compute_bootstrap_bt(df, num_round=1000, num_cpu=env.NUM_CPUS)
        elo_rating_final = compute_bt(df)

        model_order = list(elo_rating_final.keys())

        model_rating_q025 = bootstrap_df.quantile(0.025)
        model_rating_q975 = bootstrap_df.quantile(0.975)

        # compute ranking based on CI
        ranking = {}
        for i, model_a in enumerate(model_order):
            ranking[model_a] = 1
            for j, model_b in enumerate(model_order):
                if i == j:
                    continue
                if model_rating_q025[model_b] > model_rating_q975[model_a]:
                    ranking[model_a] += 1

        # leaderboard_table_df: elo rating, variance, 95% interval, number of df
        leaderboard_table_df = pd.DataFrame(
            {
                "rating": elo_rating_final,
                "variance": bootstrap_df.var(),
                "rating_q975": bootstrap_df.quantile(0.975),
                "rating_q025": bootstrap_df.quantile(0.025),
                "num_battles": df["model_a"]
                .value_counts()
                .add(df["model_b"].value_counts(), fill_value=0),
                "final_ranking": pd.Series(ranking),
            }
        ).sort_values(by="final_ranking")
        # print_all(leaderboard_table_df)
        return leaderboard_table_df
        # return compute_mle_elo_orig(df)
        # return compute_mle_elo_equal_weight(df)
        # return compute_mle_elo(df)

    def transform_elos(elos, elo_col_name):
        # elos = elos.reset_index(name=elo_col_name)
        elos[elo_col_name] = elos["rating"]
        elos = elos.reset_index(drop=False)
        elos[["organization", "model"]] = elos["index"].str.split(";", expand=True)
        elos = elos.drop(columns=["index"])
        elos = elos.sort_values(by=elo_col_name, ascending=False)
        # elos = elos.sort_values(by="final_ranking", ascending=True)
        elos = elos[
            [
                "organization",
                "model",
                elo_col_name,
            ]
        ]
        return elos

    # elos_file = "elo_scores.pkl"
    # if os.path.exists(elos_file):
    #     with open(elos_file, "rb") as file:
    #         elo_scores = pickle.load(file)
    # else:
    elo_scores = {}
    for to_run in [
        "data",
    ]:
        logger.info(f"\n\nRunning {to_run}.")
        filename = f"df_{to_run}_{leaderboard_type}.pkl"
        mask = None if to_run == "overall" else to_run
        elo_scores[to_run] = compute_chatbot_elos(
            filename=filename,
            mask=mask,
        )

    data_elos = transform_elos(elo_scores["data"], elo_col_name="elo_data")

    elos = data_elos

    elo_scores_overall = elo_scores["data"].reset_index(drop=False)
    elo_scores_overall[["organization", "model"]] = elo_scores_overall["index"].str.split(
        ";", expand=True
    )
    elo_scores_overall = elo_scores_overall.drop(columns=["index"])
    elos_cis = elo_scores_overall[
        [
            "organization",
            "model",
            "final_ranking",
            "rating_q025",
            "rating_q975",
        ]
    ]
    elos = pd.merge(elos, elos_cis, on=["organization", "model"]).reset_index(drop=True)

    model_dates = defaultdict(set)
    for entry in leaderboard:
        model_dates[(entry["organization"], entry["model"])].add(entry["forecast_due_date"])

    for key in model_dates:
        model_dates[key] = sorted(model_dates[key])

    elos["forecast_due_date"] = elos.apply(
        lambda row: model_dates.get((row["organization"], row["model"]), []), axis=1
    )

    cols = [
        "n_data",
    ]
    for c in cols:
        elos[c] = -1

    for entry in leaderboard:
        for c in cols:
            org_mask = (elos["organization"] == entry["organization"]) & (
                elos["model"] == entry["model"]
            )
            if (elos.loc[org_mask, c] == -1).all():
                if c not in entry:
                    elos.loc[org_mask, c] = 0
                else:
                    elos.loc[org_mask, c] = np.float64(entry[c])
            else:
                elos.loc[org_mask, c] += np.float64(entry[c])

    cols = [
        ("pct_imputed_data", "n_data"),
    ]
    for c, _ in cols:
        elos[c] = -1

    for entry in leaderboard:
        org_mask = (elos["organization"] == entry["organization"]) & (
            elos["model"] == entry["model"]
        )
        for c, n in cols:
            if (elos.loc[org_mask, c] == -1).all():
                if c not in entry:
                    elos.loc[org_mask, c] = 0
                else:
                    elos.loc[org_mask, c] = round(entry[c] / 100 * entry[n])
            else:
                elos.loc[org_mask, c] += round(entry[c] / 100 * entry[n])

    for index, _ in elos.iterrows():
        for c, n in cols:
            elos[c] = elos[c].astype(float)
            elos.loc[index, c] = elos.loc[index, c] / elos.loc[index, n] * 100

    return elos


def make_leaderboard(leaderboard, title, basename, leaderboard_type):
    """Make leaderboard."""
    logger.info(colored(f"Making leaderboard: {title}", "red"))

    df = compute_elos(leaderboard, leaderboard_type)

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
            leaderboard_type=task["leaderboard_type"],
        )
    except Exception as e:
        msg = f"Error processing task {task['title']}: {e}"
        logger.error(msg)
        raise ValueError(msg)


def get_df_info(df):
    masks = get_masks(df)
    return {
        "n_data": len(df[masks["data"]]),
        "pct_imputed_data": df[masks["data"]]["imputed"].sum() / len(df[masks["data"]]) * 100,
    }


@decorator.log_runtime
def driver(_):
    """Create new leaderboard."""
    # Shortcut while testing
    if os.path.exists("leaderboard_human.pkl") and os.path.exists("df_data_human.pkl"):

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
        make_leaderboard(
            leaderboard=read_leaderboard("leaderboard_llm.pkl"),
            title=title,
            basename="leaderboard_overall",
            leaderboard_type="llm",
        )
        return

    cache = {}
    llm_leaderboard = {}
    human_leaderboard = {}
    files = gcp.storage.list(env.PROCESSED_FORECAST_SETS_BUCKET)
    files = [
        file
        for file in files
        if file.endswith(".json")
        and (
            file.startswith("2024-07-21")
            or file.startswith("2025-03-02")
            or file.startswith("2025-03-30")
            or file.startswith("2025-03-16")
        )
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

        org_and_model = {"organization": organization, "model": model}
        is_human_forecast_set = organization == constants.BENCHMARK_NAME and (
            model in [SUPERFORECASTER_MODEL["model"], GENERAL_PUBLIC_MODEL["model"]]
        )

        # Remove Market questions:
        df = df[df["source"].isin(question_curation.DATA_SOURCES)]

        # Remove Combos
        df = df[~df["id"].apply(resolution.is_combo)]

        # Remove everything resolved after 30 days
        df = df[df["resolution_date"] <= df["forecast_due_date"] + timedelta(days=30)]

        # Only include forecast rounds that have had both sets of questions resolve
        res_dates = df["resolution_date"].unique()
        if len(res_dates) != 2:
            print(f"NEED 2 RES DATES: {res_dates}")
            break
        else:
            r1, r2 = res_dates[0], res_dates[1]
            print(
                f"{r1}: ",
                len(df[df["resolution_date"] == r1]),
                f"{r2}: ",
                len(df[df["resolution_date"] == r2]),
            )
            print()

        if not is_human_forecast_set:
            add_to_llm_leaderboard(
                llm_leaderboard,
                org_and_model,
                df,
                forecast_due_date,
                question_set_filename=question_set_filename,
            )
            # add_to_llm_leaderboard_include_combos(
            #     llm_leaderboard_with_combos,
            #     org_and_model,
            #     df,
            #     forecast_due_date,
            #     question_set_filename=question_set_filename,
            # )
        add_to_human_leaderboard(
            human_leaderboard,
            org_and_model,
            df,
            forecast_due_date,
            cache,
            question_set_filename=question_set_filename,
        )

        # add_to_human_combo_leaderboards(
        #     human_combo_leaderboard,
        #     org_and_model,
        #     df,
        #     forecast_due_date,
        #     cache,
        #     is_human_forecast_set,
        #     question_set_filename=question_set_filename,
        # )

    title = "Leaderboard: overall"
    # tasks = [
    #     {
    #         "leaderboard": llm_leaderboard["overall"],
    #         "title": title,
    #         "basename": "leaderboard_overall",
    #         "leaderboard_type": "llm",
    #     },
    #     {
    #         "leaderboard": human_leaderboard["overall"],
    #         "title": f"Human {title}",
    #         "basename": "human_leaderboard_overall",
    #         "leaderboard_type": "human",
    #     },
    #     # {
    #     #     "d": human_leaderboard["7"],
    #     #     "title": f"Human {title} 7 day",
    #     #     "basename": "human_leaderboard_overall_7",
    #     # },
    #     # {
    #     #     "d": human_leaderboard["30"],
    #     #     "title": f"Human {title} 30 day",
    #     #     "basename": "human_leaderboard_overall_30",
    #     # },
    #     # {
    #     #     "d": human_leaderboard["90"],
    #     #     "title": f"Human {title} 90 day",
    #     #     "basename": "human_leaderboard_overall_90",
    #     # },
    #     # {
    #     #     "d": human_leaderboard["180"],
    #     #     "title": f"Human {title} 180 day",
    #     #     "basename": "human_leaderboard_overall_180",
    #     # },
    #     # {
    #     #     "d": human_combo_leaderboard["overall"],
    #     #     "title": f"Human Combo {title}",
    #     #     "basename": "human_combo_leaderboard_overall",
    #     # },
    # ]

    with open("leaderboard_llm.pkl", "wb") as file:
        pickle.dump(llm_leaderboard["overall"], file)

    with open("leaderboard_human.pkl", "wb") as file:
        pickle.dump(human_leaderboard["overall"], file)

    logger.info(f"Using {env.NUM_CPUS} cpus for worker pool.")

    make_leaderboard(
        leaderboard=human_leaderboard["overall"],
        title=f"Human {title}",
        basename="human_leaderboard_overall",
        leaderboard_type="human",
    )
    make_leaderboard(
        leaderboard=llm_leaderboard["overall"],
        title=title,
        basename="leaderboard_overall",
        leaderboard_type="llm",
    )


if __name__ == "__main__":
    driver(None)

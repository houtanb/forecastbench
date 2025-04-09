"""Create leaderboard."""

import itertools
import json
import logging
import math
import os
import pickle
import sys
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from multiprocessing import Pool
from pprint import pprint

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
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
        org_and_model | {"forecast_due_date": forecast_due_date, "df": df.copy()}
    )


def add_to_llm_leaderboard(*args, **kwargs):
    """Wrap `add_to_leaderboard` for easy reading of driver."""
    add_to_leaderboard(*args, **kwargs)


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
            "elo_overall",
        ],
        ascending=[
            False,
        ],
        ignore_index=True,
    )

    # Round columns to 3 decimal places
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].round(LEADERBOARD_DECIMAL_PLACES)

    df["elo_overall"] = df["elo_overall"].round().astype(int)
    df["elo_data"] = df["elo_data"].round().astype(int)
    df["elo_market"] = df["elo_market"].round().astype(int)
    df["elo_market_resolved"] = df["elo_market_resolved"].round().astype(int)
    df["elo_market_unresolved"] = df["elo_market_unresolved"].round().astype(int)

    # Insert ranking
    df.insert(loc=0, column="Ranking", value="")
    df["score_diff"] = df["elo_overall"] - df["elo_overall"].shift(1)
    for index, row in df.iterrows():
        if row["score_diff"] != 0:
            prev_rank = index + 1
        df.loc[index, "Ranking"] = prev_rank
    df.drop(columns="score_diff", inplace=True)

    for c in [
        "pct_imputed_market",
        "pct_imputed_market_resolved",
        "pct_imputed_market_unresolved",
        "pct_imputed_data",
        "pct_imputed_overall",
    ]:
        df[c] = df[c].round().astype(int).astype(str) + "%"

    for c in ["n_data", "n_market", "n_market_unresolved", "n_market_resolved", "n_overall"]:
        df[c] = df[c].astype(int)

    def make_market_tuple(row):
        return (
            "R: "
            + str((row["n_market_resolved"], row["pct_imputed_market_resolved"]))
            + "<br>U: "
            + str((row["n_market_unresolved"], row["pct_imputed_market_unresolved"]))
        )

    df["data_info"] = df.apply(lambda row: (row["n_data"], row["pct_imputed_data"]), axis=1)
    df["market_resolved_info"] = df.apply(
        lambda row: (row["n_market_resolved"], row["pct_imputed_market_resolved"]), axis=1
    )
    df["market_unresolved_info"] = df.apply(
        lambda row: (row["n_market_unresolved"], row["pct_imputed_market_unresolved"]), axis=1
    )
    df["overall_info"] = df.apply(
        lambda row: (row["n_overall"], row["pct_imputed_overall"]), axis=1
    )

    df = df[
        [
            "Ranking",
            "organization",
            "model",
            "forecast_due_date",
            "elo_data",
            "data_info",
            "elo_market_resolved",
            "market_resolved_info",
            "elo_market_unresolved",
            "market_unresolved_info",
            "elo_market",
            "elo_overall",
            "overall_info",
        ]
    ]
    df = df.rename(
        columns={
            "organization": "Organization",
            "model": "Model",
            "forecast_due_date": "Forecast due date(s)",
            "elo_data": "Score dataset",
            "data_info": "Data info",
            "elo_market": "Score market",
            "elo_market_resolved": "Score market res.",
            "elo_market_unresolved": "Sscore market unres.",
            "market_resolved_info": "resolv.  info",
            "market_unresolved_info": "unres. info",
            "elo_overall": "Score overall",
            "overall_info": "Overall info",
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

    ORDER_COL_IDX = 11
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
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(df)


def get_bootstrap_result(battles, func_compute_elo):
    num_round = 100
    rows = []
    for _ in tqdm(range(num_round), desc="bootstrap"):
        rows.append(func_compute_elo(battles.sample(frac=1.0, replace=True)))
    df = pd.DataFrame(rows)
    return df[df.median().sort_values(ascending=False).index]


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
        question_type_lookup[(forecast_due_date, "market")] = 1.0 if n_market_questions else 0.0
        if n_market_questions and n_dataset_questions:
            question_type_lookup[(forecast_due_date, "data")] = (
                n_market_questions / n_dataset_questions
            )
        else:
            question_type_lookup[(forecast_due_date, "data")] = 1.0 if n_dataset_questions else 0.0

    pprint(question_n_lookup)

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
        for _, value_b in enumerate(leaderboard[i + 1 :]):
            if value_a["forecast_due_date"] == value_b["forecast_due_date"] and not (
                value_a["organization"] == value_b["organization"]
                and value_a["model"] == value_b["model"]
            ):
                print(
                    "      **** ",
                    value_b["organization"],
                    value_b["model"],
                    value_b["forecast_due_date"],
                )
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

    return pd.concat(combined_data, axis=0, ignore_index=True)


def compute_elos(leaderboard, leaderboard_suffix):
    """Compute the Elo scores."""

    def compute_chatbot_elos(filename, mask=None):
        if os.path.exists(filename):
            with open(filename, "rb") as file:
                df = pickle.load(file)
        else:
            df = combine_rounds(leaderboard, mask)
            with open(filename, "wb") as file:
                pickle.dump(df, file)
        return compute_mle_elo(df)
        # bootstrap_elo_lu = get_bootstrap_result(df_orig, compute_mle_elo)
        # print(bootstrap_elo_lu)
        # sys.exit()

    def transform_elos(elos, elo_col_name):
        elos = elos.reset_index(name=elo_col_name)
        elos[["organization", "model"]] = elos["index"].str.split(";", expand=True)
        elos = elos.drop(columns=["index"])
        elos = elos.sort_values(by=elo_col_name, ascending=False)
        elos = elos[
            [
                "organization",
                "model",
                elo_col_name,
            ]
        ]
        return elos

    elo_scores = {}
    for to_run in ["overall", "data", "market", "market_resolved", "market_unresolved"]:
        logger.info(f"Running {to_run}.")
        mask = None if to_run == "overall" else to_run
        elo_scores[to_run] = compute_chatbot_elos(f"df_{to_run}_{leaderboard_suffix}.pkl", mask)

    with open("elo_scores.pkl", "wb") as file:
        pickle.dump(elo_scores, file)

    data_elos = transform_elos(elo_scores["data"], elo_col_name="elo_data")
    market_elos = transform_elos(elo_scores["market"], elo_col_name="elo_market")
    market_resolved_elos = transform_elos(
        elo_scores["market_resolved"], elo_col_name="elo_market_resolved"
    )
    market_unresolved_elos = transform_elos(
        elo_scores["market_unresolved"], elo_col_name="elo_market_unresolved"
    )
    elos = transform_elos(elo_scores["overall"], elo_col_name="elo_overall")

    elos = pd.merge(elos, data_elos, on=["organization", "model"]).reset_index(drop=True)
    elos = pd.merge(elos, market_elos, on=["organization", "model"]).reset_index(drop=True)
    elos = pd.merge(elos, market_resolved_elos, on=["organization", "model"]).reset_index(drop=True)
    elos = pd.merge(elos, market_unresolved_elos, on=["organization", "model"]).reset_index(
        drop=True
    )

    model_dates = defaultdict(set)
    for entry in leaderboard:
        model_dates[(entry["organization"], entry["model"])].add(entry["forecast_due_date"])

    for key in model_dates:
        model_dates[key] = sorted(model_dates[key])

    elos["forecast_due_date"] = elos.apply(
        lambda row: model_dates.get((row["organization"], row["model"]), []), axis=1
    )

    print(elos)
    pprint(elos.columns)

    cols = [
        "n_data",
        "n_market",
        "n_market_unresolved",
        "n_market_resolved",
        "n_overall",
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

    # print([d for d in leaderboard if d["model"]=="Naive Forecaster"])

    cols = [
        ("pct_imputed_market", "n_market"),
        ("pct_imputed_market_resolved", "n_market_resolved"),
        ("pct_imputed_market_unresolved", "n_market_unresolved"),
        ("pct_imputed_data", "n_data"),
        ("pct_imputed_overall", "n_overall"),
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


def make_leaderboard(leaderboard, title, basename, leaderboard_suffix):
    """Make leaderboard."""
    logger.info(colored(f"Making leaderboard: {title}", "red"))
    df = compute_elos(leaderboard, leaderboard_suffix)

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
            leaderboard_suffix="human" if "Human" in task["title"] else "llm",
        )
    except Exception as e:
        msg = f"Error processing task {task['title']}: {e}"
        logger.error(msg)
        raise ValueError(msg)


def get_df_info(df, model):
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


@decorator.log_runtime
def driver(_):
    """Create new leaderboard."""
    # Shortcut while testing
    if (
        os.path.exists("leaderboard_human.pkl")
        and os.path.exists("df_data_human.pkl")
        and os.path.exists("df_market_human.pkl")
        and os.path.exists("df_overall_human.pkl")
    ):

        def read_leaderboard(f):
            with open(f, "rb") as file:
                leaderboard = pickle.load(file)
            for entry in leaderboard:
                entry |= get_df_info(entry["df"], entry["model"])
            return leaderboard

        # leaderboard=read_leaderboard("leaderboard_human.pkl")
        # for entry in leaderboard:
        #     print(entry["model"])
        #     if entry["model"]=="Claude-3-7-Sonnet-20250219 (zero shot with freeze values)":
        #         pprint(entry)
        # sys.exit()
        title = "Leaderboard: overall"
        make_leaderboard(
            leaderboard=read_leaderboard("leaderboard_human.pkl"),
            title=f"Human {title}",
            basename="human_leaderboard_overall",
            leaderboard_suffix="human",
        )
        make_leaderboard(
            leaderboard=read_leaderboard("leaderboard_llm.pkl"),
            title=title,
            basename="leaderboard_overall",
            leaderboard_suffix="llm",
        )
        return

    cache = {}
    llm_leaderboard = {}
    llm_and_human_leaderboard = {}
    # llm_and_human_combo_leaderboard = {}
    files = gcp.storage.list(env.PROCESSED_FORECAST_SETS_BUCKET)
    files = [file for file in files if file.endswith(".json") and not file.startswith("2024-12-08")]
    files1 = [
        # "2024-07-21/2024-07-21.ForecastBench.always-0.5.json",
        "2024-07-21/2024-07-21.ForecastBench.always-0.json",
        "2024-07-21/2024-07-21.ForecastBench.always-1.json",
        "2024-07-21/2024-07-21.ForecastBench.human_public.json",
        "2024-07-21/2024-07-21.ForecastBench.human_super.json",
        # "2024-07-21/2024-07-21.ForecastBench.imputed-forecaster.json",
        # "2024-07-21/2024-07-21.ForecastBench.naive-forecaster.json",
        # '2024-07-21/2024-07-21.ForecastBench.random-uniform.json',
        "2024-07-21/2024-07-21.Anthropic.claude_3p5_sonnet_scratchpad_with_freeze_values.json",
        # "2024-07-21/2024-07-21.OpenAI.gpt_4_turbo_0409_scratchpad_with_freeze_values.json",
        # "2024-07-21/2024-07-21.Qwen.qwen_1p5_110b_scratchpad.json",
        "2024-12-08/2024-12-08.ForecastBench.always-0.json",
        "2024-12-08/2024-12-08.ForecastBench.always-1.json",
        # '2024-12-08/2024-12-08.ForecastBench.imputed-forecaster.json',
        # "2024-12-08/2024-12-08.ForecastBench.naive-forecaster.json",
        # '2024-12-08/2024-12-08.OpenAI.o1-mini-2024-09-12_scratchpad.json',
        # '2024-12-08/2024-12-08.OpenAI.o1-mini-2024-09-12_scratchpad_with_freeze_values.json',
        # '2024-12-08/2024-12-08.OpenAI.o1-mini-2024-09-12_zero_shot.json',
        # '2024-12-08/2024-12-08.OpenAI.o1-mini-2024-09-12_zero_shot_with_freeze_values.json',
        # '2024-12-08/2024-12-08.OpenAI.o1-preview-2024-09-12_scratchpad.json',
        # "2024-12-08/2024-12-08.OpenAI.o1-preview-2024-09-12_scratchpad_with_freeze_values.json",
        "2024-12-08/2024-12-08.OpenAI.o1-preview-2024-09-12_zero_shot.json",
        "2024-12-08/2024-12-08.OpenAI.o1-preview-2024-09-12_zero_shot_with_freeze_values.json",
        "2024-12-08/2024-12-08.Anthropic.claude_3p5_sonnet_scratchpad_with_freeze_values.json",
        # "2025-03-02/2025-03-02.ForecastBench.naive-forecaster.json",
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
        # org_and_model = (organization, model)
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
        add_to_llm_and_human_leaderboard(
            llm_and_human_leaderboard,
            org_and_model,
            df,
            forecast_due_date,
            cache,
            question_set_filename=question_set_filename,
        )

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
        {
            "leaderboard": llm_leaderboard["overall"],
            "title": title,
            "basename": "leaderboard_overall",
        },
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

    with open("leaderboard_llm.pkl", "wb") as file:
        pickle.dump(llm_leaderboard["overall"], file)

    with open("leaderboard_human.pkl", "wb") as file:
        pickle.dump(llm_and_human_leaderboard["overall"], file)

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

"""Create leaderboard."""

import itertools
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from multiprocessing import Pool

import numpy as np
import pandas as pd
from termcolor import colored

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
    local_filename = "/tmp/tmp.json"
    gcp.storage.download(
        bucket_name=env.PROCESSED_FORECAST_SETS_BUCKET,
        filename=filename,
        local_filename=local_filename,
    )
    with open(local_filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


def get_leaderboard_entry(df):
    """Create the leaderboard entry for the given dataframe."""
    # Masks
    data_mask = df["source"].isin(question_curation.DATA_SOURCES)

    resolved_mask = df["resolved"].astype(bool)

    def get_scores(df, mask):
        scores = df[mask]["score"]
        return scores.mean(), len(scores)

    # Datasets
    data_resolved_score, n_data_resolved = get_scores(df, data_mask & resolved_mask)

    # % imputed
    pct_imputed = int(np.round(df[data_mask & resolved_mask]["imputed"].mean() * 100))

    return {
        "data": data_resolved_score,
        "n_data": n_data_resolved,
        "pct_imputed": pct_imputed,
        "df": df.copy(),
    }


def add_to_leaderboard(leaderboard, org_and_model, df, forecast_due_date):
    """Add scores to the leaderboard."""
    leaderboard_entry = [org_and_model | get_leaderboard_entry(df)]
    leaderboard["overall"] = leaderboard.get("overall", []) + leaderboard_entry


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


def add_to_llm_and_human_leaderboard(leaderboard, org_and_model, df, forecast_due_date, cache):
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
    )


def add_to_llm_and_human_combo_leaderboards(
    leaderboard_combo, org_and_model, df, forecast_due_date, cache, is_human_forecast_set
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
    )


def make_and_upload_html_table(df, title, basename):
    """Make and upload HTLM leaderboard."""
    # Replace NaN with empty strings for display
    logger.info(f"Making HTML leaderboard file: {title} {basename}.")
    df = df.fillna("")

    # Add ranking
    df = df.sort_values(by=["data"], ignore_index=True)
    df.insert(loc=0, column="Ranking", value="")
    df["score_diff"] = df["data"] - df["data"].shift(1)
    for index, row in df.iterrows():
        if row["score_diff"] != 0:
            prev_rank = index + 1
        df.loc[index, "Ranking"] = prev_rank
    df.drop(columns="score_diff", inplace=True)

    # Round columns to 3 decimal places
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].round(3)

    # Rename columns
    for col in [
        "n_data",
    ]:
        if df[col].min() != df[col].max():
            msg = (
                f"Error making leaderboard {title}: in col {col}: min ({df[col].min()}) not equal "
                f"to max ({df[col].max()})."
            )
            logger.error(msg)
            raise ValueError(msg)

    n_data = df["n_data"].max()
    df["pct_imputed"] = df["pct_imputed"].round(0).astype(int).astype(str) + "%"

    df = df[
        [
            "Ranking",
            "organization",
            "model",
            "data",
            "pct_imputed",
        ]
    ]
    df = df.rename(
        columns={
            "organization": "Organization",
            "model": "Model",
            "data": f"Dataset Score (N={n_data:,})",
            "pct_imputed": "Pct. Imputed",
        }
    )

    column_descriptions = ""

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
                "order": [[3, 'asc']],
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
        table.column(3).nodes().to$().addClass('highlight');
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


def make_leaderboard(d, title, basename):
    """Get p-values and make leaderboard."""
    logger.info(colored(f"Making leaderboard: {title}", "red"))
    # df = get_p_values(d)
    df = pd.DataFrame(d)
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
            d=task["d"],
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
    files = gcp.storage.list(env.PROCESSED_FORECAST_SETS_BUCKET)
    files = [file for file in files if file.endswith(".json")]
    directories = sorted(set(file.split("/")[0] for file in files))
    logger.info(f"Have access to {env.NUM_CPUS}.")
    for directory in directories:
        cache = {}
        llm_leaderboard = {}
        llm_and_human_leaderboard = {}
        directory_files = [f for f in files if f.startswith(directory)]
        for f in directory_files:
            logger.info(f"Downloading, reading, and scoring forecasts in `{f}`...")

            data = download_and_read_processed_forecast_file(filename=f)
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

            org_and_model = {"organization": organization, "model": model}
            is_human_forecast_set = (
                org_and_model == SUPERFORECASTER_MODEL or org_and_model == GENERAL_PUBLIC_MODEL
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
                add_to_llm_leaderboard(llm_leaderboard, org_and_model, df, forecast_due_date)
            add_to_llm_and_human_leaderboard(
                llm_and_human_leaderboard,
                org_and_model,
                df,
                forecast_due_date,
                cache,
            )

        title = f"{directory} Leaderboard: overall"
        tasks = [
            {
                "d": llm_leaderboard["overall"],
                "title": title,
                "basename": f"{directory}_leaderboard_overall",
            },
            {
                "d": llm_and_human_leaderboard["overall"],
                "title": f"Human {title}",
                "basename": f"{directory}_human_leaderboard_overall",
            },
        ]

        logger.info(f"Using {env.NUM_CPUS} cpus for worker pool.")
        with Pool(processes=env.NUM_CPUS) as pool:
            _ = pool.map(worker, tasks)


if __name__ == "__main__":
    driver(None)

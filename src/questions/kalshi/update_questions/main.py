# ABOUTME: Update kalshi_questions.jsonl and kalshi_latest_values.jsonl from fetched data.
# ABOUTME: Upserts questions, handles resolution, and writes latest market values.

"""Kalshi update question script."""

import logging
import os
import sys

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
from helpers import data_utils, dates, decorator, env  # noqa: E402

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../.."))
from utils import gcp  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SOURCE = "kalshi"


def assign_market_values(dfq, index, row):
    """Assign market values from a fetch row to the questions DataFrame.

    Args:
        dfq (pd.DataFrame): Questions DataFrame.
        index (int): Row index in dfq to update.
        row (dict): Fetch row data.

    Returns:
        pd.DataFrame: Updated questions DataFrame.
    """
    dfq.at[index, "question"] = row["question"]
    dfq.at[index, "background"] = row["background"]
    dfq.at[index, "market_info_resolution_criteria"] = "N/A"
    dfq.at[index, "url"] = row["url"]
    dfq.at[index, "market_info_close_datetime"] = (
        dates.convert_zulu_to_iso(row["close_time"]) if row["close_time"] else "N/A"
    )
    dfq.at[index, "market_info_open_datetime"] = "N/A"
    dfq.at[index, "forecast_horizons"] = "N/A"

    # Clamp last_price to [0, 1]
    last_price = float(row.get("last_price", 0))
    if last_price < 0 or last_price > 1:
        logger.warning(f"last_price {last_price} out of [0,1] for {row['id']}. Clamping.")
        last_price = max(0.0, min(1.0, last_price))
    dfq.at[index, "freeze_datetime_value"] = last_price

    # Handle resolution
    result = row.get("result", "")
    if result in ("yes", "no"):
        dfq.at[index, "resolved"] = True
        dfq.at[index, "market_info_resolution_datetime"] = (
            dates.convert_zulu_to_iso(row["settlement_time"])
            if row.get("settlement_time")
            else "N/A"
        )
    return dfq


def build_latest_values(dff):
    """Build latest values DataFrame from fetch data.

    Args:
        dff (pd.DataFrame): Fetch DataFrame.

    Returns:
        pd.DataFrame: Latest values with columns id, date, value.
    """
    today = dates.get_date_today_as_iso()
    latest_values = []
    for _, row in dff.iterrows():
        result = row.get("result", "")
        if result == "yes":
            value = 1.0
        elif result == "no":
            value = 0.0
        elif result in ("void", ""):
            last_price = float(row.get("last_price", 0))
            last_price = max(0.0, min(1.0, last_price))
            value = last_price if result == "" else float("nan")
        else:
            value = float("nan")
        latest_values.append({"id": row["id"], "date": today, "value": value})

    return pd.DataFrame(latest_values)


@decorator.log_runtime
def driver(_):
    """Execute the main workflow."""
    filenames = data_utils.generate_filenames(SOURCE)

    # Load fetch data
    dff = data_utils.download_and_read(
        filename=filenames["jsonl_fetch"],
        local_filename=filenames["local_fetch"],
        df_tmp=pd.DataFrame(columns=["id"]),
        dtype={"id": str},
    )

    # Load existing questions
    dfq = data_utils.get_data_from_cloud_storage(SOURCE, return_question_data=True)

    logger.info(f"Fetch rows: {len(dff)}, Existing questions: {len(dfq)}")

    # Insert new markets
    col_to_append = dff[~dff["id"].isin(dfq["id"])]["id"]
    if len(col_to_append) > 0:
        df_ids_to_append = pd.DataFrame(col_to_append).assign(
            **{col: None for col in dfq.columns if col != "id"}
        )
        df_ids_to_append["resolved"] = False
        df_ids_to_append["freeze_datetime_value_explanation"] = "The market price."
        df_ids_to_append["market_info_resolution_datetime"] = "N/A"
        dfq = pd.concat([dfq, df_ids_to_append], ignore_index=True)
        logger.info(f"Added {len(col_to_append)} new markets.")

    # Update all markets from fetch data
    for _, row in dff.iterrows():
        mask = dfq["id"] == row["id"]
        if mask.any():
            index = dfq.index[mask].tolist()[0]
            dfq = assign_market_values(dfq, index, row)

    # Upload questions
    logger.info("Uploading questions to GCP...")
    data_utils.upload_questions(dfq, SOURCE)

    # Build and upload latest values
    df_latest = build_latest_values(dff)
    local_latest = "/tmp/kalshi_latest_values.jsonl"
    df_latest.to_json(local_latest, orient="records", lines=True)
    gcp.storage.upload(
        bucket_name=env.QUESTION_BANK_BUCKET,
        local_filename=local_latest,
        filename="kalshi_latest_values.jsonl",
    )

    logger.info("Done.")


if __name__ == "__main__":
    driver(None)

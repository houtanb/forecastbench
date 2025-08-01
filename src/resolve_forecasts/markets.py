"""Market resolution functions."""

import logging
import os
import sys
from datetime import timedelta

import numpy as np
import pandas as pd
from termcolor import colored

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from helpers import dates, resolution  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def resolve(source, df, dfq, dfr):
    """Resolve market-based questions.

    Params:
    - source: string representing this data source
    - df: dataframe of questions to resolve
    - dfq: dataframe of all questions for this source
    - dfr: market values and resolutions for all markets belonging to `source`
    """
    logger.info(f"Resolving Market `{source}.`")
    forecast_due_date = df["forecast_due_date"].unique()[0]
    df_market, df = resolution.split_dataframe_on_source(df=df, source=source)

    # Check that we have market info for all markets in the dataset
    unique_ids_for_resolved_markets = dfr["id"].unique()

    def check_id(mid):
        if resolution.is_combo(mid):
            for midi in mid:
                check_id(midi)
        elif mid not in unique_ids_for_resolved_markets:
            msg = f"Missing resolution values in dfr for (source: {source}, id: {mid})!!!"
            logger.error(msg)
            raise ValueError(msg)

    df_market["id"].apply(lambda x: check_id(x))

    # Handle single markets first: split into standard and combo questions
    combo_mask = df_market["id"].apply(lambda x: resolution.is_combo(x))
    df_standard = df_market[~combo_mask].copy()
    df_combo = df_market[combo_mask].copy()

    # Resolve forecasts at all horizons to yesterday's market value.
    yesterday = dates.get_date_today() - timedelta(days=1)
    df_standard["yesterday"] = pd.to_datetime(yesterday)
    df_standard = pd.merge(
        df_standard,
        dfr,
        left_on=["id", "yesterday"],
        right_on=["id", "date"],
        how="left",
    )
    df_standard["resolved_to"] = df_standard["value"]
    df_standard = df_standard.drop(columns=["date", "value", "yesterday"])

    # Set all resolution dates to yesterday
    df_combo["resolution_date"] = pd.to_datetime(yesterday)
    df_standard["resolution_date"] = pd.to_datetime(yesterday)

    # Get market values at forecast_due_date
    # These values are assigned to any forecasts the organization may have omitted.
    df_standard = pd.merge(
        df_standard,
        dfr,
        left_on=["id", "forecast_due_date"],
        right_on=["id", "date"],
        how="left",
    )
    df_standard["market_value_on_due_date"] = df_standard["value"]
    df_standard = df_standard.drop(columns=["date", "value"])

    # Overwrite resolved_to values with resolved_value if question has resolved
    for mid in dfq.loc[dfq["resolved"], "id"]:
        if (df_standard["id"] == mid).any():
            resolved_value = dfr.loc[dfr["id"] == mid, "value"].iat[-1]
            resolution_date = resolution.get_market_resolution_date(dfq[dfq["id"] == mid])
            df_standard.loc[df_standard["id"] == mid, "resolved"] = True
            df_standard.loc[df_standard["id"] == mid, "resolved_to"] = resolved_value
            df_standard.loc[df_standard["id"] == mid, "resolution_date"] = resolution_date

            if resolved_value != 0 and resolved_value != 1:
                # Print warning if market resolved to something other than 0 or 1. This can be
                # valid, just want to be aware when this happens to ensure we're handling it
                # correctly.
                url = dfq[dfq["id"] == mid]["url"].iloc[0]
                logger.warning(
                    colored(
                        f"`{source}` question {mid} resolved to {resolved_value} (not 0 or 1). "
                        f"Check to ensure data pulled correctly.\n     {url}",
                        "red",
                    )
                )

            if resolution_date <= forecast_due_date.date():
                # Discard all forecasts that resolved <= forecast_due_date
                df_standard.loc[df_standard["id"] == mid, "resolved_to"] = np.nan
                rd = resolution_date.strftime("%Y-%m-%d")
                fd = forecast_due_date.strftime("%Y-%m-%d")
                url = dfq[dfq["id"] == mid]["url"].iloc[0]
                logger.warning(
                    colored(
                        f"`{source} question {mid}; was resolved on {rd} but the forecast date is "
                        f"{fd}. Nullifying!\n     {url}",
                        "red",
                    )
                )

    df_standard["resolution_date"] = pd.to_datetime(df_standard["resolution_date"], errors="coerce")
    df_standard.sort_values(by=["id", "resolution_date"], inplace=True, ignore_index=True)

    # Setup combo resolutions given df_standard
    def update_col(index, value0, value1, dir0, dir1, col):
        df_combo.at[index, col] = resolution.combo_change_sign(
            value0, dir0
        ) * resolution.combo_change_sign(value1, dir1)

    for index, row in df_combo.iterrows():
        id0, id1 = row["id"]
        id0_data = df_standard[df_standard["id"] == id0].iloc[0]
        id1_data = df_standard[df_standard["id"] == id1].iloc[0]
        dir0, dir1 = row["direction"]

        for col in ["resolved_to", "market_value_on_due_date"]:
            update_col(
                index=index,
                value0=id0_data[col],
                value1=id1_data[col],
                dir0=dir0,
                dir1=dir1,
                col=col,
            )

        resolution_date = resolution.get_combo_question_resolution_date(
            is_resolved0=id0_data["resolved"],
            is_resolved1=id1_data["resolved"],
            dir0=dir0,
            dir1=dir1,
            resolved_to0=id0_data["resolved_to"],
            resolved_to1=id1_data["resolved_to"],
            resolution_date0=id0_data["resolution_date"],
            resolution_date1=id1_data["resolution_date"],
        )
        if resolution_date:
            df_combo.at[index, "resolved"] = True
            df_combo.at[index, "resolution_date"] = resolution_date

    df_combo.sort_values(by=["id", "resolution_date"], inplace=True, ignore_index=True)

    df_source = pd.concat([df_standard, df_combo]).drop_duplicates()
    df = pd.concat([df, df_source], ignore_index=True)
    return df

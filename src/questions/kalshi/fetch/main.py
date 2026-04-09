# ABOUTME: Fetch eligible binary markets from the Kalshi prediction market API.
# ABOUTME: Discovers new markets and checks resolution status of existing ones.

"""Kalshi fetch new questions script."""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone

import aiohttp

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
from helpers import data_utils, decorator, env, kalshi  # noqa: E402

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../.."))
from utils import gcp  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SOURCE = "kalshi"


async def fetch_with_throttle(session, url, params=None):
    """Fetch URL, retrying on 429 with Retry-After backoff."""
    max_retries = 5
    for attempt in range(max_retries):
        async with session.get(url, params=params) as response:
            if response.status == 429:
                retry_after = int(response.headers.get("Retry-After", 10))
                logger.warning(f"429 from Kalshi (attempt {attempt + 1}). Sleeping {retry_after}s.")
                await asyncio.sleep(retry_after)
                continue
            response.raise_for_status()
            return await response.json()
    raise RuntimeError(f"Exceeded max retries for {url}")


async def fetch_all_markets(session):
    """Paginate through open markets."""
    max_close_ts = int(
        (datetime.now(timezone.utc) + timedelta(days=kalshi.MAX_CLOSE_DAYS)).timestamp()
    )
    all_markets = []
    cursor = None
    while True:
        params = {
            "status": "open",
            "mve_filter": "exclude",
            "max_close_ts": max_close_ts,
            "limit": 1000,
        }
        if cursor:
            params["cursor"] = cursor
        data = await fetch_with_throttle(session, f"{kalshi.API_BASE_URL}/markets", params)
        markets = data.get("markets", [])
        if not markets:
            break
        all_markets.extend(markets)
        cursor = data.get("cursor")
        if not cursor:
            break
    return all_markets


def passes_filters(market):
    """Return True if market passes volume and open interest filters."""
    volume = float(market.get("volume_fp", "0"))
    open_interest = float(market.get("open_interest_fp", "0"))
    return volume >= kalshi.MIN_VOLUME and open_interest >= kalshi.MIN_OPEN_INTEREST


async def fetch_event_categories(session, event_tickers):
    """Fetch category for each event ticker.

    Returns:
        dict of event_ticker -> category.
    """
    categories = {}
    tasks = [
        fetch_with_throttle(session, f"{kalshi.API_BASE_URL}/events/{ticker}")
        for ticker in event_tickers
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for ticker, result in zip(event_tickers, results):
        if isinstance(result, Exception):
            logger.error(f"Failed to fetch event {ticker}: {result}")
            continue
        categories[ticker] = result.get("event", {}).get("category", "")
    return categories


async def check_existing_unresolved(session, tickers):
    """Fetch current state of existing unresolved markets."""
    markets = []
    for i in range(0, len(tickers), 100):
        batch = tickers[i : i + 100]
        params = {"tickers": ",".join(batch)}
        data = await fetch_with_throttle(session, f"{kalshi.API_BASE_URL}/markets", params)
        markets.extend(data.get("markets", []))
    return markets


def market_to_fetch_row(market):
    """Build a fetch output row from a market object."""
    return {
        "id": market["ticker"],
        "question": market.get("title", ""),
        "background": market.get("rules_primary", ""),
        "url": f"https://kalshi.com/markets/{market['ticker']}",
        "last_price": market.get("last_price_dollars", "0"),
        "result": market.get("result", ""),
        "status": market.get("status", ""),
        "close_time": market.get("close_time", ""),
        "settlement_time": market.get("settlement_time", ""),
        "yes_sub_title": market.get("yes_sub_title", ""),
    }


async def async_driver():
    """Run the async fetch pipeline."""
    filenames = data_utils.generate_filenames(SOURCE)

    # Download existing questions
    dfq = data_utils.get_data_from_cloud_storage(SOURCE, return_question_data=True)

    async with aiohttp.ClientSession() as session:
        # Step 1: Paginate open markets
        logger.info("Fetching open markets from Kalshi API...")
        all_markets = await fetch_all_markets(session)
        logger.info(f"Fetched {len(all_markets)} open markets.")

        # Step 2: Filter by volume and open interest
        filtered_markets = [m for m in all_markets if passes_filters(m)]
        logger.info(f"{len(filtered_markets)} markets pass volume/open interest filters.")

        # Step 3: Fetch event categories
        event_tickers = list({m.get("event_ticker", "") for m in filtered_markets} - {""})
        logger.info(f"Fetching categories for {len(event_tickers)} events...")
        categories = await fetch_event_categories(session, event_tickers)

        # Step 4: Filter by allowed categories
        category_filtered = []
        for m in filtered_markets:
            event_ticker = m.get("event_ticker", "")
            category = categories.get(event_ticker, "")
            if category in kalshi.ALLOWED_CATEGORIES:
                category_filtered.append(m)
        logger.info(f"{len(category_filtered)} markets pass category filter.")

        # Step 5: Build fetch rows for new/open markets
        fetched_ids = {m["ticker"] for m in category_filtered}
        fetch_rows = [market_to_fetch_row(m) for m in category_filtered]

        # Step 6: Check existing unresolved questions not already fetched
        if not dfq.empty:
            unresolved_ids = (
                set(dfq.loc[~dfq["resolved"], "id"]) if "resolved" in dfq.columns else set()
            )
            tickers_to_check = list(unresolved_ids - fetched_ids)
            if tickers_to_check:
                logger.info(f"Checking {len(tickers_to_check)} existing unresolved markets...")
                existing_markets = await check_existing_unresolved(session, tickers_to_check)
                for m in existing_markets:
                    fetch_rows.append(market_to_fetch_row(m))
                logger.info(f"Added {len(existing_markets)} existing unresolved markets.")

    logger.info(f"Total fetch rows: {len(fetch_rows)}")

    # Write fetch file
    with open(filenames["local_fetch"], "w", encoding="utf-8") as f:
        for row in fetch_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Upload
    logger.info("Uploading to GCP...")
    gcp.storage.upload(
        bucket_name=env.QUESTION_BANK_BUCKET,
        local_filename=filenames["local_fetch"],
    )
    logger.info("Done.")


@decorator.log_runtime
def driver(_):
    """Execute the main workflow."""
    asyncio.run(async_driver())


if __name__ == "__main__":
    driver(None)

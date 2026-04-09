# Adding Kalshi as a Question Source

## Overview

Add Kalshi as a market source for binary prediction questions. Unlike other market sources that store full daily price histories, Kalshi stores only the latest market value per question in a single file, updated nightly.

## New files

```
src/helpers/kalshi.py
src/questions/kalshi/fetch/main.py
src/questions/kalshi/fetch/Makefile
src/questions/kalshi/fetch/requirements.txt
src/questions/kalshi/update_questions/main.py
src/questions/kalshi/update_questions/Makefile
src/questions/kalshi/update_questions/requirements.txt
```

## Modified files

```
src/helpers/question_curation.py
```

## API

Base URL: `https://api.elections.kalshi.com/trade-api/v2`

All endpoints used are unauthenticated (public). Auth can be added later if needed.

### Endpoints used

- `GET /markets` — discover and filter markets. Supports server-side filters: `status`, `mve_filter`, `max_close_ts`, `tickers`.
- `GET /events/{event_ticker}` — get event metadata (category) and child markets.

### Throttling

Async requests with dynamic backoff. On 429 responses, wait the duration specified in the `Retry-After` header before retrying.

## Fetch step (`src/questions/kalshi/fetch/main.py`)

1. Download existing `kalshi_questions.jsonl` from cloud storage (if it exists).
2. Paginate `GET /markets` with `status=open`, `mve_filter=exclude`, `max_close_ts=today+365d`, `limit=1000`. Filter locally by `volume_fp >= 1000` and `open_interest_fp >= 100`.
3. Collect unique `event_ticker` values from surviving markets. Fetch `GET /events/{event_ticker}` for each. Filter markets by event category against `ALLOWED_CATEGORIES`.
4. For existing unresolved questions in `kalshi_questions.jsonl`: take tickers where `resolved == False`, remove any already seen in step 2, query `GET /markets?tickers=...` for the rest to check resolution status.
5. Output `kalshi_fetch.jsonl` — one row per market with: id (ticker), question (title), background (rules_primary), url, last_price_dollars, result, status, close_time, settlement_time.

## Update step (`src/questions/kalshi/update_questions/main.py`)

Processes the fetch file only — no API calls.

1. Load `kalshi_fetch.jsonl` and existing `kalshi_questions.jsonl`.
2. Upsert into `kalshi_questions.jsonl`:
   - New markets: insert with standard columns (id, question, background, url, resolved, forecast_horizons, freeze_datetime_value, freeze_datetime_value_explanation, market_info_resolution_criteria, market_info_open_datetime, market_info_close_datetime, market_info_resolution_datetime).
   - Existing markets: update `resolved`, `market_info_resolution_datetime`, `freeze_datetime_value`.
   - Resolved markets (`result` = `"yes"` or `"no"`): set `resolved=True`, set `market_info_resolution_datetime` from settlement time.
3. Write `kalshi_latest_values.jsonl` — one row per market:
   - `{"id": ticker, "date": today, "value": ...}`
   - Unresolved: `value` = `last_price_dollars` (clamped to [0, 1], warn if out of range)
   - Resolved yes: `value` = 1
   - Resolved no: `value` = 0
   - Voided/cancelled: `value` = NaN
4. Upload both files to cloud storage.

## Helper module (`src/helpers/kalshi.py`)

```python
SOURCE_INTRO = "..."  # Description of Kalshi
RESOLUTION_CRITERIA = "Resolves to the outcome of the question found at {url}."

ALLOWED_CATEGORIES = [
    "Climate and Weather",
    "Companies",
    "Economics",
    "Elections",
    "Financials",
    "Health",
    "Politics",
    "Science and Technology",
    "Transportation",
    "World",
]

MIN_VOLUME = 1000
MIN_OPEN_INTEREST = 100
MAX_CLOSE_DAYS = 365
```

## Question curation changes (`src/helpers/question_curation.py`)

Add `"kalshi"` to `FREEZE_QUESTION_MARKET_SOURCES` with name, source_intro, and resolution_criteria from `src/helpers/kalshi.py`. This automatically propagates to `MARKET_SOURCES`, `ALL_SOURCES`, and the sampling allocation. Market questions will now be split across 5 sources instead of 4.

## Data storage

Unlike other market sources that store individual `{source}/{id}.jsonl` files with full daily price histories, Kalshi uses a single `kalshi_latest_values.jsonl` file with one row per market containing the most recent value. This file is overwritten nightly.

### Files in question bank

- `kalshi_questions.jsonl` — standard question file format, matching other sources
- `kalshi_latest_values.jsonl` — single file, one row per market: `id | date | value`
- `kalshi_fetch.jsonl` — intermediate fetch output

## Resolution

Not addressed in this design. Will be handled in a follow-up that modifies the resolution code.

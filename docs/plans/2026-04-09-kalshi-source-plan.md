# Kalshi Source Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add Kalshi as a market question source with fetch, update, and question set integration.

**Architecture:** Two-stage pipeline (fetch → update) matching existing sources. Fetch discovers eligible markets via async API calls, outputs `kalshi_fetch.jsonl`. Update processes the fetch file into `kalshi_questions.jsonl` and `kalshi_latest_values.jsonl`. No individual per-question history files — just a single latest-values file.

**Tech Stack:** Python, aiohttp (async HTTP), pandas, GCP Cloud Storage

**Design doc:** `docs/plans/2026-04-09-kalshi-source-design.md`

---

### Task 1: Create helper module

**Files:**
- Create: `src/helpers/kalshi.py`

**Step 1: Write the helper module**

```python
"""Kalshi-specific variables."""

SOURCE_INTRO = (
    "We would like you to predict the outcome of a prediction market. A prediction market, in this "
    "context, is the aggregate of predictions submitted by users on the website Kalshi. "
    "You're going to predict the probability that the market will resolve as 'Yes'."
)

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

API_BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

MIN_VOLUME = 1000
MIN_OPEN_INTEREST = 100
MAX_CLOSE_DAYS = 365
```

**Step 2: Run lint**

Run: `make lint`

**Step 3: Commit**

```bash
git add src/helpers/kalshi.py
git commit -m "kalshi: add helper module with constants"
```

---

### Task 2: Register Kalshi as a market source

**Files:**
- Modify: `src/helpers/question_curation.py:6-18` (imports) and `src/helpers/question_curation.py:32-55` (FREEZE_QUESTION_MARKET_SOURCES)

**Step 1: Add kalshi import**

In the import block at line 6, add `kalshi` to the imports (alphabetical order):

```python
from . import (
    acled,
    constants,
    dates,
    dbnomics,
    fred,
    infer,
    kalshi,
    manifold,
    metaculus,
    polymarket,
    wikipedia,
    yfinance,
)
```

**Step 2: Add kalshi to FREEZE_QUESTION_MARKET_SOURCES**

After the `"infer"` entry (line 48) and before `"polymarket"` (line 50), add:

```python
    "kalshi": {
        "name": "Kalshi",
        "source_intro": kalshi.SOURCE_INTRO,
        "resolution_criteria": kalshi.RESOLUTION_CRITERIA,
    },
```

**Step 3: Run lint**

Run: `make lint`

**Step 4: Commit**

```bash
git add src/helpers/question_curation.py
git commit -m "kalshi: register as market source in question curation"
```

---

### Task 3: Create fetch script

**Files:**
- Create: `src/questions/kalshi/fetch/main.py`

**Step 1: Write the fetch script**

The fetch script does the following:
1. Downloads existing `kalshi_questions.jsonl` from GCS (if it exists)
2. Paginates `GET /markets` with `status=open`, `mve_filter=exclude`, `max_close_ts=today+MAX_CLOSE_DAYS`, `limit=1000`
3. Filters locally: `volume_fp >= MIN_VOLUME`, `open_interest_fp >= MIN_OPEN_INTEREST`
4. For each unique `event_ticker` among surviving markets, fetches `GET /events/{event_ticker}` to get category
5. Filters markets by event category against `ALLOWED_CATEGORIES`
6. For existing unresolved questions in `kalshi_questions.jsonl`: takes tickers where `resolved == False`, removes any already seen in step 2, queries `GET /markets?tickers=...` for the rest
7. Outputs `kalshi_fetch.jsonl`

Key implementation details:

- Use `aiohttp` for async HTTP. Pattern:
  ```python
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
  ```

- Paginate markets:
  ```python
  async def fetch_all_markets(session):
      """Paginate through open markets."""
      max_close_ts = int((datetime.now(timezone.utc) + timedelta(days=kalshi.MAX_CLOSE_DAYS)).timestamp())
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
  ```

- Filter markets locally:
  ```python
  def passes_filters(market):
      volume = float(market.get("volume_fp", "0"))
      open_interest = float(market.get("open_interest_fp", "0"))
      return volume >= kalshi.MIN_VOLUME and open_interest >= kalshi.MIN_OPEN_INTEREST
  ```

- Fetch events for category lookup (async, batched):
  ```python
  async def fetch_event_categories(session, event_tickers):
      """Fetch category for each event ticker. Return dict of event_ticker -> category."""
      categories = {}
      tasks = [fetch_with_throttle(session, f"{kalshi.API_BASE_URL}/events/{ticker}") for ticker in event_tickers]
      results = await asyncio.gather(*tasks, return_exceptions=True)
      for ticker, result in zip(event_tickers, results):
          if isinstance(result, Exception):
              logger.error(f"Failed to fetch event {ticker}: {result}")
              continue
          categories[ticker] = result.get("event", {}).get("category", "")
      return categories
  ```

- Check resolution of existing unresolved questions not already fetched. The `/markets` endpoint accepts a `tickers` param (comma-separated). Batch into groups of ~100 to avoid URL length issues:
  ```python
  async def check_existing_unresolved(session, tickers):
      """Fetch current state of existing unresolved markets."""
      markets = []
      for i in range(0, len(tickers), 100):
          batch = tickers[i:i+100]
          params = {"tickers": ",".join(batch)}
          data = await fetch_with_throttle(session, f"{kalshi.API_BASE_URL}/markets", params)
          markets.extend(data.get("markets", []))
      return markets
  ```

- Build fetch output row from market object:
  ```python
  def market_to_fetch_row(market):
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
  ```

- `driver()` follows the standard pattern: `@decorator.log_runtime`, takes `_` arg, saves to local file, uploads to GCS.

- The file must start with an ABOUTME comment:
  ```python
  # ABOUTME: Fetch eligible binary markets from the Kalshi prediction market API.
  # ABOUTME: Discovers new markets and checks resolution status of existing ones.
  ```

- Follow the `sys.path.append` pattern from other sources for imports:
  ```python
  sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
  from helpers import data_utils, dates, decorator, env, kalshi  # noqa: E402

  sys.path.append(os.path.join(os.path.dirname(__file__), "../../../.."))
  from utils import gcp  # noqa: E402
  ```

**Step 2: Run lint**

Run: `make lint`

**Step 3: Commit**

```bash
git add src/questions/kalshi/fetch/main.py
git commit -m "kalshi fetch: discover eligible markets from Kalshi API"
```

---

### Task 4: Create update script

**Files:**
- Create: `src/questions/kalshi/update_questions/main.py`

**Step 1: Write the update script**

The update script processes `kalshi_fetch.jsonl` — no API calls.

Key logic:

```python
# ABOUTME: Update kalshi_questions.jsonl and kalshi_latest_values.jsonl from fetched data.
# ABOUTME: Upserts questions, handles resolution, and writes latest market values.
```

1. Load `kalshi_fetch.jsonl` and existing `kalshi_questions.jsonl`:
   ```python
   dff = data_utils.download_and_read(
       filename=filenames["jsonl_fetch"],
       local_filename=filenames["local_fetch"],
       df_tmp=pd.DataFrame(columns=["id"]),
       dtype={"id": str},
   )
   dfq = data_utils.get_data_from_cloud_storage(source=source, return_question_data=True)
   ```

2. For new markets (in dff but not dfq), insert rows:
   ```python
   col_to_append = dff[~dff["id"].isin(dfq["id"])]["id"]
   df_ids_to_append = pd.DataFrame(col_to_append).assign(
       **{col: None for col in dfq.columns if col != "id"}
   )
   df_ids_to_append["resolved"] = False
   df_ids_to_append["freeze_datetime_value_explanation"] = "The market price."
   df_ids_to_append["market_info_resolution_datetime"] = "N/A"
   dfq = pd.concat([dfq, df_ids_to_append], ignore_index=True)
   ```

3. For all markets in dff, update dfq:
   ```python
   def assign_market_values(dfq, index, row):
       dfq.at[index, "question"] = row["question"]
       dfq.at[index, "background"] = row["background"]
       dfq.at[index, "market_info_resolution_criteria"] = "N/A"
       dfq.at[index, "url"] = row["url"]
       dfq.at[index, "market_info_close_datetime"] = dates.convert_zulu_to_iso(row["close_time"]) if row["close_time"] else "N/A"
       dfq.at[index, "market_info_open_datetime"] = "N/A"  # Not provided by markets endpoint
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
   ```

4. Build `kalshi_latest_values.jsonl`:
   ```python
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

   df_latest = pd.DataFrame(latest_values)
   ```

5. Upload both files:
   ```python
   data_utils.upload_questions(dfq, source)

   # Upload latest values
   local_latest = "/tmp/kalshi_latest_values.jsonl"
   df_latest.to_json(local_latest, orient="records", lines=True)
   gcp.storage.upload(
       bucket_name=env.QUESTION_BANK_BUCKET,
       local_filename=local_latest,
       filename="kalshi_latest_values.jsonl",
   )
   ```

- Follow the same `sys.path.append` import pattern as other update scripts.
- Use `@decorator.log_runtime` on `driver()`.

**Step 2: Run lint**

Run: `make lint`

**Step 3: Commit**

```bash
git add src/questions/kalshi/update_questions/main.py
git commit -m "kalshi update: process fetch file into questions and latest values"
```

---

### Task 5: Create Makefiles and requirements

**Files:**
- Create: `src/questions/kalshi/fetch/Makefile`
- Create: `src/questions/kalshi/fetch/requirements.txt`
- Create: `src/questions/kalshi/update_questions/Makefile`
- Create: `src/questions/kalshi/update_questions/requirements.txt`

**Step 1: Create fetch Makefile**

Follow the polymarket fetch Makefile pattern exactly. Job name: `func-data-kalshi-fetch`. Same memory (4Gi) and timeout (3h) as polymarket fetch since we're doing async API calls.

```makefile
all :
	$(MAKE) clean
	$(MAKE) deploy

.PHONY : all clean deploy

UPLOAD_DIR = upload

.gcloudignore:
	cp -r $(ROOT_DIR)src/helpers/.gcloudignore .

Procfile:
	cp -r $(ROOT_DIR)src/helpers/Procfile .

deploy : main.py .gcloudignore requirements.txt Procfile
	mkdir -p $(UPLOAD_DIR)
	cp -r $(ROOT_DIR)utils $(UPLOAD_DIR)/
	cp -r $(ROOT_DIR)src/helpers $(UPLOAD_DIR)/
	cp $^ $(UPLOAD_DIR)/
	gcloud run jobs deploy \
		func-data-kalshi-fetch \
		--project $(CLOUD_PROJECT) \
		--region $(CLOUD_DEPLOY_REGION) \
		--tasks 1 \
		--parallelism 1 \
		--task-timeout 3h \
		--memory 4Gi \
		--max-retries 0 \
		--service-account $(QUESTION_BANK_BUCKET_SERVICE_ACCOUNT) \
		--set-env-vars $(DEFAULT_CLOUD_FUNCTION_ENV_VARS) \
		--source $(UPLOAD_DIR)

clean :
	rm -rf $(UPLOAD_DIR) .gcloudignore Procfile
```

**Step 2: Create fetch requirements.txt**

```
google-cloud-storage
google-cloud-secret-manager
pandas>=2.2.2,<3.0
aiohttp
```

**Step 3: Create update Makefile**

Same pattern. Job name: `func-data-kalshi-update-questions`. Memory 2Gi, timeout 540s (same as polymarket update).

```makefile
all :
	$(MAKE) clean
	$(MAKE) deploy

.PHONY : all clean deploy

UPLOAD_DIR = upload

.gcloudignore:
	cp -r $(ROOT_DIR)src/helpers/.gcloudignore .

Procfile:
	cp -r $(ROOT_DIR)src/helpers/Procfile .

deploy : main.py .gcloudignore requirements.txt Procfile
	mkdir -p $(UPLOAD_DIR)
	cp -r $(ROOT_DIR)utils $(UPLOAD_DIR)/
	cp -r $(ROOT_DIR)src/helpers $(UPLOAD_DIR)/
	cp $^ $(UPLOAD_DIR)/
	gcloud run jobs deploy \
		func-data-kalshi-update-questions \
		--project $(CLOUD_PROJECT) \
		--region $(CLOUD_DEPLOY_REGION) \
		--tasks 1 \
		--parallelism 1 \
		--task-timeout 540s \
		--memory 2Gi \
		--max-retries 0 \
		--service-account $(QUESTION_BANK_BUCKET_SERVICE_ACCOUNT) \
		--set-env-vars $(DEFAULT_CLOUD_FUNCTION_ENV_VARS) \
		--source $(UPLOAD_DIR)

clean :
	rm -rf $(UPLOAD_DIR) .gcloudignore Procfile
```

**Step 4: Create update requirements.txt**

```
google-cloud-storage
google-cloud-secret-manager
pandas>=2.2.2,<3.0
```

**Step 5: Commit**

```bash
git add src/questions/kalshi/fetch/Makefile src/questions/kalshi/fetch/requirements.txt \
        src/questions/kalshi/update_questions/Makefile src/questions/kalshi/update_questions/requirements.txt
git commit -m "kalshi: add Makefiles and requirements for fetch and update"
```

---

### Task 6: Register Kalshi in nightly workflow and deployment

**Files:**
- Modify: `src/nightly_update_workflow/worker/main.py:86-111` (`get_fetch_and_update`)
- Modify: `Makefile:101` (questions target)

**Step 1: Add kalshi to nightly worker sources**

In `src/nightly_update_workflow/worker/main.py`, add `"kalshi"` to the `sources` list in `get_fetch_and_update()` (alphabetical, between `"infer"` and `"manifold"`):

```python
    sources = [
        "dbnomics",
        "fred",
        "infer",
        "kalshi",
        "manifold",
        "metaculus",
        "polymarket",
        "wikipedia",
        "yfinance",
    ]
```

**Step 2: Add kalshi targets to root Makefile**

Add to the `questions` prerequisite list at line 101:

```makefile
questions: manifold metaculus acled infer yfinance polymarket wikipedia fred dbnomics kalshi
```

Add the kalshi targets (after the dbnomics targets, before tag-questions):

```makefile
kalshi: kalshi-fetch kalshi-update-questions

kalshi-fetch:
	$(MAKE) -C src/questions/kalshi/fetch || echo "* $@" >> $(MAKE_FAILURE_LOG)

kalshi-update-questions:
	$(MAKE) -C src/questions/kalshi/update_questions || echo "* $@" >> $(MAKE_FAILURE_LOG)
```

**Step 3: Run lint**

Run: `make lint`

**Step 4: Commit**

```bash
git add src/nightly_update_workflow/worker/main.py Makefile
git commit -m "kalshi: register in nightly workflow and deployment"
```

---

### Task 7: Verify locally

**Step 1: Run the fetch script locally**

```bash
cd src/questions/kalshi/fetch && eval $(cat ../../../../variables.mk | xargs) python main.py
```

Verify:
- Script completes without errors
- `/tmp/kalshi_fetch.jsonl` is created with market rows
- Each row has: id, question, background, url, last_price, result, status, close_time

**Step 2: Run the update script locally**

```bash
cd src/questions/kalshi/update_questions && eval $(cat ../../../../variables.mk | xargs) python main.py
```

Verify:
- `/tmp/kalshi_questions.jsonl` is created with standard columns
- `/tmp/kalshi_latest_values.jsonl` has one row per market with id, date, value
- All `value` entries are in [0, 1] or NaN

**Step 3: Commit any fixes**

If fixes are needed, commit them with descriptive messages.

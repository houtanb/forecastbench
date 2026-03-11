# ForecastBench: AI Assistant Guide

ForecastBench is a dynamic, contamination-free benchmark of LLM forecasting accuracy. The code under `src/` works both locally and deployed as GCP Cloud Run Jobs.

For core project documentation see also: [AGENTS.md](AGENTS.md)

---

## Table of Contents

1. [Project Architecture](#project-architecture)
2. [Directory Structure](#directory-structure)
3. [Data Models & Schemas](#data-models--schemas)
4. [Source Modules (`src/`)](#source-modules-src)
5. [Shared Helpers (`src/helpers/`)](#shared-helpers-srchelpers)
6. [Local Development](#local-development)
7. [GCP Deployment](#gcp-deployment)
8. [Code Style & Conventions](#code-style--conventions)
9. [CI/CD Pipeline](#cicd-pipeline)
10. [Supported LLM Models](#supported-llm-models)
11. [Key Architectural Patterns](#key-architectural-patterns)

---

## Project Architecture

The nightly GCP workflow runs sequentially:

1. **Pull data** — fetch questions/resolutions from 8 sources (`src/questions/`)
2. **Generate metadata** — invalidate & tag questions (`src/metadata/`)
3. **Resolve forecasts** — score submitted forecasts (`src/resolve_forecasts/`)
4. **Create leaderboard** — compute Brier scores & rankings (`src/leaderboard/`)
5. **Update website** — publish results (`src/www.forecastbench.org/`)

Every 14 days:
- A new question set is sampled (`src/curate_questions/`) 10 days before the forecast due date
- LLM forecasts are generated (`src/base_eval/`) on the forecast due date

---

## Directory Structure

```
forecastbench/
├── .github/workflows/         # CI: runs make lint on PRs to main
├── experiments/               # Scoring rule experiments
│   ├── ranking-simulation/
│   └── stability-analysis/
├── paper/                     # ICLR 2025 paper analysis & table generation
│   ├── arena_tc_graphs/
│   ├── category_table/
│   ├── llm_breakdown_tables/
│   ├── make_index/
│   ├── populate_results_tables/
│   └── source_table/
├── utils/                     # Git submodule: shared org-level GCP/archive utilities
├── src/
│   ├── base_eval/             # LLM forecasters and naive baselines
│   │   ├── llm_baselines/manager/
│   │   ├── llm_baselines/worker/
│   │   ├── llm_crowd/
│   │   └── naive_and_dummy_forecasters/
│   ├── curate_questions/      # Bi-weekly question set sampling
│   │   ├── create_question_set/
│   │   └── publish_question_set/
│   ├── helpers/               # Shared utilities (23 modules)
│   ├── leaderboard/           # Leaderboard generation (~104KB main.py)
│   ├── metadata/              # Question validation & LLM-based tagging
│   │   ├── tag_questions/
│   │   └── validate_questions/
│   ├── nightly_update_workflow/  # GCP orchestration
│   │   ├── compress_buckets/
│   │   ├── manager/
│   │   └── worker/
│   ├── questions/             # Data source integrations (8 sources)
│   │   ├── acled/
│   │   ├── fred/
│   │   ├── infer/
│   │   ├── manifold/
│   │   ├── metaculus/
│   │   ├── polymarket/
│   │   ├── wikipedia/
│   │   └── yfinance/
│   ├── resolve_forecasts/     # Resolution and scoring logic
│   └── www.forecastbench.org/ # Jekyll website (minimal-mistakes-jekyll theme)
├── Makefile                   # Root orchestration (~225 lines)
├── pyproject.toml             # Black config (line-length = 100)
├── setup.cfg                  # isort, flake8, pydocstyle config
├── requirements.txt           # Linting tools (isort, black, flake8, pydocstyle)
├── variables.example.mk       # Template for GCP environment variables
├── variables.mk               # Local GCP env vars (DO NOT MODIFY; RUNNING_LOCALLY=1)
├── AGENTS.md                  # Core project instructions
└── CLAUDE.md                  # This file
```

---

## Data Models & Schemas

### Question File (JSONL, one question per line)

```python
{
    "id": str,                                    # Unique question identifier
    "question": str,                              # The forecast question text
    "background": str,                            # Context for the question
    "url": str,                                   # Source URL
    "resolved": bool,                             # Whether the question has resolved
    "forecast_horizons": list[int],               # Days into the future: [7, 30, 90, 180, 365, ...]
    "freeze_datetime_value": str,                 # Market value at freeze time
    "freeze_datetime_value_explanation": str,
    "market_info_resolution_criteria": str,
    "market_info_open_datetime": str,
    "market_info_close_datetime": str,
    "market_info_resolution_datetime": str,
}
```

### Resolution File (JSONL)

```python
{
    "id": str,
    "date": str,           # ISO date string
    "value": any,          # Numeric or string depending on source
}
```

### Metadata File (JSONL)

```python
{
    "source": str,
    "id": str,
    "category": str,       # One of 9 categories (see constants.py)
    "valid_question": bool,
}
```

### Forecast File (JSON, per model submission)

```python
{
    "organization": str,
    "model": str,
    "question_set": str,
    "forecast_due_date": str,
    "forecasts": [
        {
            "id": str,
            "source": str,
            "direction": str,
            "forecast": list[float],   # Probability per horizon
            "resolution_date": str,
            "reasoning": str,
        }
    ]
}
```

### Question Categories (from `src/helpers/constants.py`)

The 9 question categories are:
- Science & Technology
- Healthcare & Biology
- Economics & Business
- Environment & Energy
- Politics & Governance
- Arts & Culture
- Sports & Entertainment
- Security & Conflict
- Other

---

## Source Modules (`src/`)

### `src/questions/` — Data Source Integration

Each of the 8 sources has two sub-phases:
- `fetch/` — Initial data pull from the external API
- `update_questions/` — Refresh resolution values for existing questions

Each source follows the same pattern:
```
src/questions/<source>/
├── fetch/
│   ├── main.py
│   ├── requirements.txt
│   └── Makefile          # Deploys func-data-<source>-fetch
└── update_questions/
    ├── main.py
    ├── requirements.txt
    └── Makefile
```

Sources and their data:

| Source | Type | Data |
|--------|------|------|
| `manifold` | Prediction market | Binary markets from Manifold |
| `metaculus` | Prediction platform | Public Metaculus questions |
| `infer` | Prediction market | Infer prediction market |
| `polymarket` | Prediction market | Polymarket contracts |
| `acled` | Dataset | Armed conflict event data |
| `fred` | Dataset | Federal Reserve economic data |
| `wikipedia` | Dataset | Wikipedia page view statistics |
| `yfinance` | Dataset | Yahoo Finance stock/financial data |

### `src/metadata/` — Question Validation & Tagging

- **`tag_questions/`**: Uses GPT to auto-assign one of 9 categories; runs 50 concurrent async API calls using `llm_prompts.ASSIGN_CATEGORY_PROMPT`
- **`validate_questions/`**: Marks questions as valid/invalid

### `src/base_eval/` — Forecasting Baselines

- **`llm_baselines/manager/`**: Dispatches forecasting jobs to Cloud Run workers; one task = (model, prompt_type) pair
- **`llm_baselines/worker/`**: Executes actual LLM API calls across 30+ models
- **`llm_crowd/`**: Integrates crowdsourced forecast data
- **`naive_and_dummy_forecasters/`**: Prophet time-series models for FRED and ACLED; uses 100-window forecasting with 30 and 90-day lookback

### `src/curate_questions/` — Bi-weekly Question Sampling

- **`create_question_set/`**: Stratified sampling with:
  - 12 probability buckets (0-1%, 1-10%, ..., 99-100%)
  - 7 time horizon bins (0-7 days through 366+ days)
  - Targets: 500 LLM + 200 human questions per set
- **`publish_question_set/`**: Freezes questions 10 days before forecast due date; captures market values at freeze time

### `src/resolve_forecasts/` — Resolution & Scoring

- Validates forecast JSON structure
- Maps questions to resolution values
- Computes Brier scores
- Uploads resolved sets to `PUBLIC_RELEASE_BUCKET`
- Local submodules: `acled`, `data`, `markets`, `wikipedia`

### `src/leaderboard/` — Leaderboard Generation

- Two leaderboard types: `BASELINE` (FB-generated only) and `TOURNAMENT` (all participants)
- Brier score with 95% confidence intervals (bootstrapped)
- Pairwise statistical significance testing
- Anonymous teams get auto-generated SVG logos
- Exports JSON and CSV to `leaderboards/js/` and `leaderboards/csv/`
- Model release date tracking (365-day cutoff for contamination)

### `src/nightly_update_workflow/` — GCP Orchestration

**`manager/main.py`** orchestrates the full pipeline:
1. Publish question set (if scheduled)
2. Metadata: tag + validate
3. Resolve forecasts
4. Generate leaderboard
5. Update website
6. Generate baselines (on publication day)
7. Compress storage buckets

Uses `helpers.cloud_run.call_worker()` to invoke each Cloud Run Job. Sends Slack alerts at key milestones.

**`worker/main.py`** executes individual Cloud Run Jobs using a `metadata` dict specifying what to run.

### `src/www.forecastbench.org/` — Jekyll Website

- Theme: `minimal-mistakes-jekyll`
- Key pages: `/about`, `/datasets`, `/explore`, `/baseline`, `/tournament`, `/docs`
- Assets: leaderboard CSVs, JS files, org logos
- Deployed as `func-website` Cloud Run Job
- Serve locally: `bundle exec jekyll s`

---

## Shared Helpers (`src/helpers/`)

The 23 modules in `src/helpers/` are imported across all source modules. Key modules:

| Module | Purpose |
|--------|---------|
| `constants.py` | Global constants: dates, categories, model definitions, horizons |
| `data_utils.py` | GCS bucket read/write, workspace management, filename generation |
| `env.py` | Loads GCP resources and runtime settings from environment |
| `cloud_run.py` | Invoke and monitor GCP Cloud Run Jobs |
| `model_eval.py` | LLM API calls (OpenAI, Anthropic, Google, xAI, Together AI, Mistral) |
| `llm_prompts.py` | Prompt templates: zero-shot, market, non-market, category assignment |
| `llm_crowd_prompts.py` | Prompts for crowdsourced forecasts (56KB) |
| `resolution.py` | Resolution logic for market and dataset sources |
| `question_curation.py` | Sampling config for question set creation |
| `dates.py` | Date utilities and epoch conversions |
| `question_sets.py` | Question set metadata access |
| `git.py` | Git operations for version control |
| `slack.py` | Slack messaging for pipeline alerts |
| `decorator.py` | Runtime logging decorator |
| `keys.py` | API key management |
| `acled.py` | ACLED data helpers |
| `fred.py` | FRED economic data helpers (68KB) |
| `wikipedia.py` | Wikipedia data helpers (40KB) |
| `polymarket.py` | Polymarket helpers |
| `metaculus.py` | Metaculus helpers |
| `manifold.py` | Manifold Markets helpers |
| `infer.py` | Infer Markets helpers |
| `yfinance.py` | Yahoo Finance helpers |
| `dbnomics.py` | DBnomics database helpers |

**Important constants from `constants.py`:**
- Benchmark start: May 1, 2024
- Tournament start: July 21, 2024
- Forecast horizons: `[7, 30, 90, 180, 365, 1095, 1825, 3650]` days

---

## Local Development

### First-time Setup

```bash
git clone --recurse-submodules <repo>
cp variables.example.mk variables.mk   # Fill in GCP credentials
make setup-python-env
source .venv/bin/activate
```

### Daily Workflow

```bash
source .venv/bin/activate

# Authenticate with GCP (needed once per day)
gcloud config set project forecastbench-dev
gcloud auth application-default print-access-token >/dev/null 2>&1 && echo "GCP Access OK" \
  || gcloud auth application-default login
```

### Running Python Modules

All code under `src/` (except the Jekyll website) requires loading `variables.mk` as environment variables:

```bash
# General pattern
cd src/<module> && eval $(cat ../../variables.mk | xargs) python main.py

# Examples
cd src/leaderboard && eval $(cat ../../variables.mk | xargs) python main.py
cd src/metadata/tag_questions && eval $(cat ../../variables.mk | xargs) python main.py
```

**Note:** `variables.mk` must not be modified. When running locally, it contains `RUNNING_LOCALLY=1`.

### Serving the Website

```bash
cd src/www.forecastbench.org && bundle exec jekyll s
```

To use real leaderboard data (requires running leaderboard first):

```bash
cd src/leaderboard && eval $(cat ../../variables.mk | xargs) python main.py
cp src/leaderboard/leaderboards/js/* src/www.forecastbench.org/assets/js/
cp src/leaderboard/leaderboards/csv/* src/www.forecastbench.org/assets/data/
cp src/leaderboard/anonymous_logos/* src/www.forecastbench.org/assets/images/org_logos/
```

### Linting

```bash
make lint    # Runs black, isort, flake8, pydocstyle
```

---

## GCP Deployment

### Environment Variables

Key GCP variables (defined in `variables.mk`, templated in `variables.example.mk`):

| Variable | Purpose |
|----------|---------|
| `CLOUD_PROJECT` | GCP project ID |
| `QUESTION_BANK_BUCKET` | GCS bucket for all questions |
| `QUESTION_SETS_BUCKET` | GCS bucket for published question sets |
| `FORECAST_SETS_BUCKET` | GCS bucket for submitted forecasts |
| `PROCESSED_FORECAST_SETS_BUCKET` | GCS bucket for resolved forecasts |
| `PUBLIC_RELEASE_BUCKET` | GCS bucket for public data releases |
| `WEBSITE_BUCKET` | GCS bucket for the Jekyll website |
| `WORKSPACE_BUCKET` | GCS bucket for temporary working data |
| `RUNNING_LOCALLY` | Set to `1` when running locally |

### Cloud Run Job Pattern

Each module deploys as a GCP Cloud Run Job following this pattern:

```makefile
gcloud run jobs deploy [job-name] \
  --project $(CLOUD_PROJECT) \
  --region $(CLOUD_DEPLOY_REGION) \
  --tasks [count] \
  --memory [size] \
  --cpu [cores] \
  --task-timeout [duration] \
  --service-account [account] \
  --set-env-vars $(DEFAULT_CLOUD_FUNCTION_ENV_VARS) \
  --add-volume name=[bucket],type=cloud-storage,bucket=[bucket] \
  --add-volume-mount volume=[bucket],mount-path=/mnt/[bucket] \
  --source $(UPLOAD_DIR)
```

Notable resource allocations:
- `leaderboard`: 16GB memory, 8 CPUs
- Most jobs use GCS-FUSE volume mounts to access buckets at `/mnt/<bucket>`

### Makefile Targets

Root `Makefile` key targets:

| Target | Description |
|--------|-------------|
| `make setup-python-env` | Create `.venv` and install dev dependencies |
| `make lint` | Format (black, isort) and check (flake8, pydocstyle) |
| `make deploy` | Full production deployment to GCP |
| `make deploy-questions` | Deploy all 8 question source jobs |
| `make deploy-leaderboard` | Deploy leaderboard job |
| `make deploy-website` | Deploy website job |

---

## Code Style & Conventions

### Formatting Tools

| Tool | Config | Purpose |
|------|--------|---------|
| `black` | `pyproject.toml` | Formatting, line length 100 |
| `isort` | `setup.cfg` | Import sorting (black-compatible profile) |
| `flake8` | `setup.cfg` | Linting (max-line 100, flake8-bugbear) |
| `pydocstyle` | `setup.cfg` | Docstring style enforcement |

### Type Hints

All functions must use type hints:

```python
def process_question(question_id: str, horizons: list[int]) -> dict:
    ...
```

### Docstring Format

```python
def my_function(arg1: str, arg2: int) -> list:
    """Brief description of function.

    Args:
      arg1 (str): Description of arg1.
      arg2 (int): Description of arg2.

    Returns:
      list: Description of return value.
    """
```

### Import Order

isort with black-compatible profile. Standard library → third party → local. Run `make lint` to auto-sort.

### File-level Notes

- Each `src/<module>/` directory has its own `requirements.txt` for Cloud Run deployment
- Each `src/<module>/` directory has its own `Makefile` for deployment targets
- Do not add a root-level `requirements.txt` entry for module-specific deps

---

## CI/CD Pipeline

`.github/workflows/makefile.yml` runs on every PR to `main`:

1. Sets up Python 3.12
2. Installs root `requirements.txt` (linting tools only)
3. Runs `make lint MAKE_LINT_ERROR_OUT=1`
4. Fails the PR if any style violations are found

**Always run `make lint` before committing.** The CI will fail if style checks don't pass.

---

## Supported LLM Models

Defined in `src/helpers/constants.py`. Current models (30+):

| Provider | Models |
|----------|--------|
| OpenAI | gpt-4.1, gpt-5.1, gpt-5.2, gpt-5-mini, gpt-5-nano |
| Anthropic | claude-haiku-4-5, claude-sonnet-4-5, claude-sonnet-4-6, claude-opus-4-1, claude-opus-4-6 |
| Google | gemini-2.5-pro, gemini-3-flash, gemini-3.1-pro |
| xAI | grok-4-fast-reasoning, grok-4-1-fast-reasoning, grok-4-fast-non-reasoning, grok-4-1-fast-non-reasoning |
| Together AI | DeepSeek-V3.1, MiniMax-M2.5, Kimi-K2.5, GLM-4.7, GLM-5 |
| Mistral | Mistral-Large |

To add a new model, update `constants.py` (model definitions and token limits).

---

## Key Architectural Patterns

### 1. Serverless-First

All processing runs on GCP Cloud Run Jobs. No always-on servers. Each job is independently deployable from its own directory.

### 2. GCS as the Data Layer

All persistent state lives in GCS buckets. Code reads/writes via `helpers.data_utils`. Local runs use `RUNNING_LOCALLY=1` to redirect to local paths.

### 3. Contamination Prevention

Questions are frozen 10 days before the forecast due date. The freeze captures market values at that moment. This prevents LLMs from seeing resolution data during forecasting.

### 4. Stratified Question Sampling

Question sets use stratified binning across:
- Probability values (12 bins from 0% to 100%)
- Time horizons (7 bins from 0-7 days to 366+ days)

This ensures balanced coverage across difficulty levels and time scales.

### 5. Statistical Rigor in Leaderboard

- Brier scores with 95% bootstrapped confidence intervals
- Pairwise statistical significance tests
- 365-day model release date cutoff to flag potential contamination

### 6. LLM-Driven Metadata

Category assignment uses GPT with 50 concurrent async API calls via `asyncio` — do not use synchronous calls in this path.

### 7. Retry Pattern

All external API calls use `backoff` for automatic retry with exponential backoff. Follow this pattern when adding new data sources.

### 8. Module Independence

Each subdirectory under `src/questions/` is fully self-contained with its own `requirements.txt` and `Makefile`. Shared code lives only in `src/helpers/` (imported at runtime, not bundled).

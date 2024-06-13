"""Environment variables."""

import os

PROJECT_ID = os.environ.get("CLOUD_PROJECT")
QUESTION_BANK_BUCKET = os.environ.get("QUESTION_BANK_BUCKET")
QUESTION_SETS_BUCKET = os.environ.get("QUESTION_SETS_BUCKET")
FORECAST_SETS_BUCKET = os.environ.get("FORECAST_SETS_BUCKET")
PROCESSED_FORECAST_SETS_BUCKET = os.environ.get("PROCESSED_FORECAST_SETS")
LEADERBOARD_BUCKET = os.environ.get("LEADERBOARD_BUCKET")

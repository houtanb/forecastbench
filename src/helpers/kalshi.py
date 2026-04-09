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

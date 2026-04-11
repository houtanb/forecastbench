"""Validate forecast files and produce emoji summary."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

THRESHOLD_PCT = 95.0

MARKET_SOURCES = {"manifold", "metaculus", "infer", "polymarket"}

_REQUIRED_TOP_LEVEL_KEYS = {
    "organization",
    "model",
    "model_organization",
    "question_set",
    "forecast_due_date",
    "forecasts",
}


@dataclass
class ValidationResult:
    """Result of validating a forecast file."""

    valid_json: bool
    valid_probabilities: bool
    market_forecasted: int
    market_total: int
    dataset_forecasted: int
    dataset_total: int

    @property
    def market_pct(self) -> float:
        """Percentage of market questions forecasted."""
        if self.market_total == 0:
            return 100.0
        return 100.0 * self.market_forecasted / self.market_total

    @property
    def dataset_pct(self) -> float:
        """Percentage of dataset questions forecasted."""
        if self.dataset_total == 0:
            return 100.0
        return 100.0 * self.dataset_forecasted / self.dataset_total

    def format_summary(self, model_name: str, prompt_type: str) -> str:
        """Format an emoji summary string."""
        lines = [f"\U0001f4ca Forecast Summary: {model_name} ({prompt_type})"]

        json_icon = "\u2705" if self.valid_json else "\u274c"
        lines.append(f"\u251c\u2500\u2500 {json_icon} Valid JSON structure")

        prob_icon = "\u2705" if self.valid_probabilities else "\u274c"
        lines.append(f"\u251c\u2500\u2500 {prob_icon} All probabilities in (0, 1)")

        market_icon = "\u2705" if self.market_pct >= THRESHOLD_PCT else "\u274c"
        lines.append(
            f"\u251c\u2500\u2500 {market_icon} Market questions: "
            f"{self.market_forecasted}/{self.market_total} forecasted "
            f"({self.market_pct:.1f}%)"
        )

        dataset_icon = "\u2705" if self.dataset_pct >= THRESHOLD_PCT else "\u274c"
        lines.append(
            f"\u2514\u2500\u2500 {dataset_icon} Dataset questions: "
            f"{self.dataset_forecasted}/{self.dataset_total} forecasted "
            f"({self.dataset_pct:.1f}%)"
        )

        return "\n".join(lines)


def validate_forecast_file(
    filepath: str, n_market: int, n_dataset: int
) -> ValidationResult:
    """Validate a forecast file.

    Args:
        filepath: Path to the forecast JSON file.
        n_market: Expected number of market questions.
        n_dataset: Expected number of dataset questions.
    """
    try:
        with open(filepath) as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return ValidationResult(
            valid_json=False,
            valid_probabilities=False,
            market_forecasted=0,
            market_total=n_market,
            dataset_forecasted=0,
            dataset_total=n_dataset,
        )

    # Check required top-level keys
    if not _REQUIRED_TOP_LEVEL_KEYS.issubset(data.keys()):
        return ValidationResult(
            valid_json=False,
            valid_probabilities=False,
            market_forecasted=0,
            market_total=n_market,
            dataset_forecasted=0,
            dataset_total=n_dataset,
        )

    forecasts = data["forecasts"]
    valid_json = True
    valid_probabilities = True
    market_forecasted = 0
    dataset_forecasted = 0

    for entry in forecasts:
        if not isinstance(entry, dict):
            valid_json = False
            continue

        source = entry.get("source", "")
        forecast_val = entry.get("forecast")

        is_market = source in MARKET_SOURCES

        if forecast_val is not None:
            if isinstance(forecast_val, (int, float)):
                if not (0 < forecast_val < 1):
                    valid_probabilities = False
                if is_market:
                    market_forecasted += 1
                else:
                    dataset_forecasted += 1
            else:
                valid_probabilities = False

    return ValidationResult(
        valid_json=valid_json,
        valid_probabilities=valid_probabilities,
        market_forecasted=market_forecasted,
        market_total=n_market,
        dataset_forecasted=dataset_forecasted,
        dataset_total=n_dataset,
    )

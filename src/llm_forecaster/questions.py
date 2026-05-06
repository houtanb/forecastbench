"""Question-set loading helpers for LLM forecasting."""

from dataclasses import dataclass

from orchestration import _io
from sources import DATASET_SOURCE_NAMES, MARKET_SOURCE_NAMES


@dataclass(frozen=True)
class QuestionSetContext:
    """Question-set metadata and questions for one LLM run."""

    forecast_due_date: str
    question_set_filename: str
    questions: list[dict]

    @classmethod
    def from_question_set_json(cls, data: dict) -> "QuestionSetContext":
        """Build LLM question-set context from orchestration question-set JSON."""
        forecast_due_date = data["forecast_due_date"]
        return cls(
            forecast_due_date=forecast_due_date,
            question_set_filename=data.get("question_set", f"{forecast_due_date}-llm.json"),
            questions=data["questions"],
        )


def load_question_set_context(
    forecast_due_date: str, run_locally: bool = False
) -> QuestionSetContext:
    """Load LLM question-set context through orchestration IO."""
    filename = f"{forecast_due_date}-llm.json"
    data = _io.read_question_set_json(filename, run_locally=run_locally)
    return QuestionSetContext.from_question_set_json(data)


def split_questions(questions: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split question set items into dataset and market questions."""
    dataset_source_names = set(DATASET_SOURCE_NAMES)
    market_source_names = set(MARKET_SOURCE_NAMES)
    dataset = []
    market = []
    unknown_sources = set()

    for question in questions:
        source = question.get("source")
        if source in dataset_source_names:
            dataset.append(question)
        elif source in market_source_names:
            market.append(question)
        else:
            unknown_sources.add(source)

    if unknown_sources:
        sources = ", ".join(sorted(str(source) for source in unknown_sources))
        raise ValueError(f"Unknown question sources: {sources}")

    return dataset, market


def limit_questions_for_test_mode(
    dataset_questions: list[dict],
    market_questions: list[dict],
    limit_per_type: int,
) -> tuple[list[dict], list[dict]]:
    """Limit dataset and market questions independently for test mode."""
    return (
        dataset_questions[:limit_per_type],
        market_questions[:limit_per_type],
    )

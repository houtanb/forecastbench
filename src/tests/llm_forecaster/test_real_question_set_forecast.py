import ast
import json
import re

from llm_forecaster import model_runs, output, questions, runner
from llm_forecaster.forecast_variants import ZERO_SHOT, ZERO_SHOT_WITH_FREEZE_VALUES
from orchestration import _io
from sources import DATASET_SOURCE_NAMES


def _live_2026_05_24_context() -> questions.QuestionSetContext:
    data = _io.read_question_set_json("2026-05-24-llm.json", run_locally=False)
    return questions.QuestionSetContext.from_question_set_json(data)


def _test_mode_context(context: questions.QuestionSetContext) -> questions.QuestionSetContext:
    dataset_questions, market_questions = questions.split_questions(context.questions)
    dataset_questions, market_questions = questions.limit_questions_for_test_mode(
        dataset_questions,
        market_questions,
        2,
    )
    return questions.QuestionSetContext(
        forecast_due_date=context.forecast_due_date,
        question_set_filename=context.question_set_filename,
        questions=dataset_questions + market_questions,
    )


def _dataset_probability_response(prompt: str) -> str:
    match = re.search(r"^Question resolution dates: (.+)$", prompt, flags=re.MULTILINE)
    if match is None:
        raise AssertionError("Dataset prompt did not include resolution dates")

    resolution_dates = ast.literal_eval(match.group(1))
    return " ".join(f"*0.{index + 1:02d}*" for index, _ in enumerate(resolution_dates))


def _row_key(row: dict) -> tuple[str, str, str | None]:
    return row["source"], row["id"], row["resolution_date"]


def test_real_2026_05_24_question_set_writes_example_forecasts_without_provider_calls(
    monkeypatch,
    tmp_path,
):
    context = _test_mode_context(_live_2026_05_24_context())
    dataset_questions, market_questions = questions.split_questions(context.questions)
    model_run = model_runs.MODEL_RUNS[0]
    calls = []

    def fake_get_response(self, prompt: str) -> str:
        calls.append((self.slug, prompt))
        if self is model_runs.REFORMAT_MODEL:
            raise AssertionError("Well-formed test responses should not use the reformat model")
        if "Question resolution dates:" in prompt:
            return _dataset_probability_response(prompt)
        return "*0.42*"

    monkeypatch.setattr(type(model_run), "get_response", fake_get_response)

    written_files = runner.run_model(
        model_run=model_run,
        context=context,
        output_dir=tmp_path,
        upload=False,
        is_test=True,
        today_date="2026-05-26",
        raise_on_question_error=True,
    )

    assert context.forecast_due_date == "2026-05-24"
    assert context.question_set_filename == "2026-05-24-llm.json"
    assert len(dataset_questions) == 2
    assert len(market_questions) == 2
    assert [written_file.variant for written_file in written_files] == [
        ZERO_SHOT,
        ZERO_SHOT_WITH_FREEZE_VALUES,
    ]

    zero_shot_file = tmp_path / output.final_filename(
        "2026-05-24",
        model_run,
        ZERO_SHOT,
        is_test=True,
    )
    freeze_values_file = tmp_path / output.final_filename(
        "2026-05-24",
        model_run,
        ZERO_SHOT_WITH_FREEZE_VALUES,
        is_test=True,
    )
    assert [written_file.local_filename for written_file in written_files] == [
        zero_shot_file,
        freeze_values_file,
    ]

    zero_shot_data = json.loads(zero_shot_file.read_text(encoding="utf-8"))
    freeze_values_data = json.loads(freeze_values_file.read_text(encoding="utf-8"))
    assert set(zero_shot_data) == {
        "organization",
        "model",
        "model_organization",
        "model_run_key",
        "model_run_slug",
        "forecast_variant_key",
        "uses_freeze_values",
        "question_set",
        "forecast_due_date",
        "forecasts",
    }
    assert zero_shot_data["organization"] == "ForecastBench"
    assert zero_shot_data["model"] == model_run.slug
    assert zero_shot_data["model_organization"] == model_run.lab.name
    assert zero_shot_data["model_run_key"] == model_run.model_run_key
    assert zero_shot_data["model_run_slug"] == model_run.slug
    assert zero_shot_data["forecast_variant_key"] == ZERO_SHOT.key
    assert zero_shot_data["uses_freeze_values"] is False
    assert zero_shot_data["question_set"] == "2026-05-24-llm.json"
    assert zero_shot_data["forecast_due_date"] == "2026-05-24"
    zero_shot_raw_file = zero_shot_file.read_text(encoding="utf-8")
    freeze_values_raw_file = freeze_values_file.read_text(encoding="utf-8")
    assert "†" not in zero_shot_raw_file
    assert "\\u2020" not in zero_shot_raw_file
    assert "†" not in freeze_values_raw_file
    assert "\\u2020" not in freeze_values_raw_file
    assert freeze_values_data["model"] == f"{model_run.slug}-{ZERO_SHOT_WITH_FREEZE_VALUES.key}"
    assert freeze_values_data["model_run_key"] == model_run.model_run_key
    assert freeze_values_data["model_run_slug"] == model_run.slug
    assert freeze_values_data["forecast_variant_key"] == ZERO_SHOT_WITH_FREEZE_VALUES.key
    assert freeze_values_data["uses_freeze_values"] is True

    dataset_row_count = sum(len(question["resolution_dates"]) for question in dataset_questions)
    assert len(zero_shot_data["forecasts"]) == dataset_row_count + len(market_questions)
    assert len(freeze_values_data["forecasts"]) == dataset_row_count + len(market_questions)

    dataset_source_names = set(DATASET_SOURCE_NAMES)
    zero_shot_dataset_rows = [
        row for row in zero_shot_data["forecasts"] if row["source"] in dataset_source_names
    ]
    freeze_dataset_rows = [
        row for row in freeze_values_data["forecasts"] if row["source"] in dataset_source_names
    ]
    assert zero_shot_dataset_rows == freeze_dataset_rows
    assert {_row_key(row) for row in zero_shot_dataset_rows} == {
        (question["source"], question["id"], resolution_date)
        for question in dataset_questions
        for resolution_date in question["resolution_dates"]
    }

    market_row_keys = {(question["source"], question["id"], None) for question in market_questions}
    assert {
        _row_key(row)
        for row in zero_shot_data["forecasts"]
        if row["source"] not in dataset_source_names
    } == market_row_keys
    assert {
        _row_key(row)
        for row in freeze_values_data["forecasts"]
        if row["source"] not in dataset_source_names
    } == market_row_keys
    assert len(calls) == len(dataset_questions) + 2 * len(market_questions)

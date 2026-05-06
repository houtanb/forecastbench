import threading
from types import SimpleNamespace

import pytest
from utils.llm.provider_registry import PROVIDERS

from llm_forecaster import runner
from llm_forecaster.forecast_variants import ZERO_SHOT, ZERO_SHOT_WITH_FREEZE_VALUES
from llm_forecaster.questions import QuestionSetContext


class FakeRun:
    model_run_key = "test-model-run-variant-01"
    slug = "test-model"
    provider_model_id = "test-provider-model-id"
    lab = SimpleNamespace(name="Test Lab")
    provider = PROVIDERS["OpenAI"]

    def __init__(self):
        self.prompts = []

    def get_response(self, prompt):
        self.prompts.append(prompt)
        return "*0.4*"


class FakeReformatRun(FakeRun):
    slug = "reformat-model"
    provider_model_id = "reformat-model-id"

    def get_response(self, prompt):
        self.prompts.append(prompt)
        return "[0.2, 0.3]"


class BlockingRun(FakeRun):
    provider = object()

    def __init__(self, expected_calls):
        super().__init__()
        self.expected_calls = expected_calls
        self.started_calls = 0
        self.lock = threading.Lock()
        self.all_started = threading.Event()

    def get_response(self, prompt):
        with self.lock:
            self.prompts.append(prompt)
            self.started_calls += 1
            if self.started_calls >= self.expected_calls:
                self.all_started.set()

        if not self.all_started.wait(timeout=1):
            raise AssertionError("questions were not forecast concurrently")
        return "*0.4*"


def _dataset_question():
    return {
        "id": "dataset-1",
        "source": "fred",
        "question": "Will value rise after {forecast_due_date} by {resolution_date}?",
        "background": "Dataset background",
        "resolution_criteria": "Dataset criteria",
        "market_info_resolution_criteria": "N/A",
        "freeze_datetime": "2026-05-05",
        "freeze_datetime_value": "100",
        "freeze_datetime_value_explanation": "Latest observed value.",
        "resolution_dates": ["2026-06-01", "2026-07-01"],
    }


def _market_question():
    return {
        "id": "market-1",
        "source": "metaculus",
        "question": "Will the market question resolve true?",
        "background": "Market background",
        "resolution_criteria": "Market criteria",
        "market_info_resolution_criteria": "N/A",
        "market_info_close_datetime": "2026-06-15",
        "freeze_datetime": "2026-05-05",
        "freeze_datetime_value": "0.33",
    }


def _context():
    return QuestionSetContext(
        forecast_due_date="2026-05-10",
        question_set_filename="2026-05-10-llm.json",
        questions=[_dataset_question(), _market_question()],
    )


def test_rendered_prompts_are_scoped_to_current_question_phase():
    dataset_prompts = runner._render_dataset_prompts(
        [_dataset_question()],
        forecast_due_date="2026-05-10",
        today_date="2026-05-06",
    )
    zero_shot_market_prompts = runner._render_market_prompts(
        [_market_question()],
        forecast_due_date="2026-05-10",
        today_date="2026-05-06",
        variant=ZERO_SHOT,
    )
    freeze_market_prompts = runner._render_market_prompts(
        [_market_question()],
        forecast_due_date="2026-05-10",
        today_date="2026-05-06",
        variant=ZERO_SHOT_WITH_FREEZE_VALUES,
    )

    assert set(dataset_prompts) == {("fred", "dataset-1", ZERO_SHOT)}
    assert set(zero_shot_market_prompts) == {("metaculus", "market-1", ZERO_SHOT)}
    assert set(freeze_market_prompts) == {("metaculus", "market-1", ZERO_SHOT_WITH_FREEZE_VALUES)}


def test_dataset_questions_are_forecast_concurrently_and_returned_in_order(monkeypatch):
    first_question = _dataset_question()
    first_question["resolution_dates"] = ["2026-06-01"]
    second_question = {**_dataset_question(), "id": "dataset-2"}
    second_question["resolution_dates"] = ["2026-08-01"]
    model_run = BlockingRun(expected_calls=2)

    monkeypatch.setitem(runner.model_runs.PROVIDER_MAX_WORKERS, model_run.provider, 2)
    monkeypatch.setattr(runner.parsing, "parse_dataset_forecast", lambda *args: [0.2])

    rows = runner._forecast_dataset_questions(
        model_run,
        [first_question, second_question],
        forecast_due_date="2026-05-10",
        today_date="2026-05-06",
        raise_on_question_error=True,
    )

    assert [row["id"] for row in rows] == ["dataset-1", "dataset-2"]


def test_market_questions_are_forecast_concurrently_and_returned_in_order(monkeypatch):
    first_question = _market_question()
    second_question = {**_market_question(), "id": "market-2"}
    model_run = BlockingRun(expected_calls=2)

    monkeypatch.setitem(runner.model_runs.PROVIDER_MAX_WORKERS, model_run.provider, 2)
    monkeypatch.setattr(runner.parsing, "parse_market_forecast", lambda *args: 0.4)

    rows = runner._forecast_market_questions(
        model_run,
        [first_question, second_question],
        forecast_due_date="2026-05-10",
        today_date="2026-05-06",
        variant=ZERO_SHOT,
        raise_on_question_error=True,
    )

    assert [row["id"] for row in rows] == ["market-1", "market-2"]


def test_run_model_writes_zero_shot_before_freeze_values(monkeypatch, tmp_path):
    events = []
    original_forecast_market_questions = runner._forecast_market_questions

    monkeypatch.setattr(runner.parsing, "parse_dataset_forecast", lambda *args: [0.2, 0.3])
    monkeypatch.setattr(runner.parsing, "parse_market_forecast", lambda *args: 0.4)

    def fake_write_forecast_file(
        local_filename,
        forecast_due_date,
        question_set_filename,
        model_run,
        variant,
        rows,
    ):
        events.append(("write", variant.key))
        local_filename.write_text("written", encoding="utf-8")

    def fake_forecast_market_questions(*args, **kwargs):
        variant = kwargs["variant"]
        if variant == ZERO_SHOT_WITH_FREEZE_VALUES:
            events.append(("freeze-started", variant.key))
            raise RuntimeError("freeze failed")
        return original_forecast_market_questions(*args, **kwargs)

    monkeypatch.setattr(runner.output, "write_forecast_file", fake_write_forecast_file)
    monkeypatch.setattr(runner, "_forecast_market_questions", fake_forecast_market_questions)

    with pytest.raises(RuntimeError, match="freeze failed"):
        runner.run_model(
            FakeRun(),
            _context(),
            tmp_path,
            is_test=True,
            today_date="2026-05-06",
            raise_on_question_error=True,
        )

    zero_shot_filename = runner.output.final_filename(
        "2026-05-10",
        FakeRun(),
        ZERO_SHOT,
        is_test=True,
    )
    assert events[:2] == [
        ("write", "zero-shot"),
        ("freeze-started", "zero-shot-with-freeze-values"),
    ]
    assert (tmp_path / zero_shot_filename).exists()


def test_dataset_rows_are_reused_across_variants(monkeypatch, tmp_path):
    calls = {"dataset": 0}
    dataset_row = {
        "id": "dataset-1",
        "source": "fred",
        "forecast": 0.2,
        "resolution_date": "2026-06-01",
        "reasoning": None,
    }

    def fake_forecast_dataset_questions(*args, **kwargs):
        calls["dataset"] += 1
        return [dataset_row]

    def fake_forecast_market_questions(*args, **kwargs):
        variant = kwargs["variant"]
        return [
            {
                "id": variant.key,
                "source": "metaculus",
                "forecast": 0.4,
                "resolution_date": None,
                "reasoning": None,
            }
        ]

    monkeypatch.setattr(runner, "_forecast_dataset_questions", fake_forecast_dataset_questions)
    monkeypatch.setattr(runner, "_forecast_market_questions", fake_forecast_market_questions)

    written_files = runner.run_model(
        FakeRun(),
        _context(),
        tmp_path,
        is_test=True,
        today_date="2026-05-06",
    )

    assert [written_file.variant for written_file in written_files] == [
        ZERO_SHOT,
        ZERO_SHOT_WITH_FREEZE_VALUES,
    ]
    assert calls == {"dataset": 1}
    assert written_files[0].rows[0] == dataset_row
    assert written_files[1].rows[0] == dataset_row


def test_dataset_rows_are_reused_within_each_variant_group(monkeypatch, tmp_path):
    dataset_calls = []

    def fake_forecast_dataset_questions(*args, **kwargs):
        variant = kwargs["variant"]
        dataset_calls.append(variant)
        return [
            {
                "id": "dataset-1",
                "source": "fred",
                "forecast": len(dataset_calls) / 10,
                "resolution_date": "2026-06-01",
                "reasoning": None,
            }
        ]

    def fake_forecast_market_questions(*args, **kwargs):
        return []

    monkeypatch.setattr(
        runner,
        "DATASET_FORECAST_SHARING_VARIANT_GROUPS",
        ((ZERO_SHOT, ZERO_SHOT_WITH_FREEZE_VALUES),),
    )
    monkeypatch.setattr(runner, "_forecast_dataset_questions", fake_forecast_dataset_questions)
    monkeypatch.setattr(runner, "_forecast_market_questions", fake_forecast_market_questions)

    written_files = runner.run_model(
        FakeRun(),
        _context(),
        tmp_path,
        is_test=True,
        today_date="2026-05-06",
    )

    assert dataset_calls == [ZERO_SHOT]
    assert [written_file.rows[0]["forecast"] for written_file in written_files] == [0.1, 0.1]


def test_run_model_writes_test_llm_call_transcript(monkeypatch, tmp_path):
    def fake_parse_dataset_forecast(response, prompt, question, reformat_model):
        assert response == "*0.4*"
        assert reformat_model.get_response("reformat prompt") == "[0.2, 0.3]"
        return [0.2, 0.3]

    monkeypatch.setattr(runner.parsing, "parse_dataset_forecast", fake_parse_dataset_forecast)
    monkeypatch.setattr(runner.parsing, "parse_market_forecast", lambda *args: 0.4)
    monkeypatch.setattr(runner.model_runs, "REFORMAT_MODEL", FakeReformatRun())

    runner.run_model(
        FakeRun(),
        _context(),
        tmp_path,
        is_test=True,
        today_date="2026-05-06",
    )

    transcript_files = list(tmp_path.glob("TEST.*.llm-calls.md"))
    assert len(transcript_files) == 1
    transcript = transcript_files[0].read_text(encoding="utf-8")

    assert transcript.startswith("# LLM Call Transcript\n")
    assert transcript.count("## Call ") == 4
    assert "## Call 1: forecast (zero-shot)" in transcript
    assert "- Model run slug: test-model" in transcript
    assert "- Source: fred" in transcript
    assert "- Question ID: dataset-1" in transcript
    assert "- Variant: zero-shot-with-freeze-values" in transcript
    assert "- Role: reformat" in transcript
    assert "- Model run slug: reformat-model" in transcript
    assert "- Provider model ID: reformat-model-id" in transcript
    assert "### Prompt\n\n```text\n" in transcript
    assert "Dataset background" in transcript
    assert "Market value on 2026-05-05" in transcript
    assert "reformat prompt" in transcript
    assert "### Response\n\n```text\n*0.4*\n```" in transcript
    assert "### Response\n\n```text\n[0.2, 0.3]\n```" in transcript


def test_run_model_does_not_write_llm_call_transcript_in_prod_mode(monkeypatch, tmp_path):
    monkeypatch.setattr(runner.parsing, "parse_dataset_forecast", lambda *args: [0.2, 0.3])
    monkeypatch.setattr(runner.parsing, "parse_market_forecast", lambda *args: 0.4)

    runner.run_model(
        FakeRun(),
        _context(),
        tmp_path,
        is_test=False,
        today_date="2026-05-06",
    )

    assert list(tmp_path.glob("*.llm-calls.*")) == []


def test_run_model_does_not_overwrite_existing_local_final_file(monkeypatch, tmp_path):
    existing_filename = runner.output.final_filename(
        "2026-05-10",
        FakeRun(),
        ZERO_SHOT,
        is_test=True,
    )
    existing_path = tmp_path / existing_filename
    existing_path.write_text("existing", encoding="utf-8")

    monkeypatch.setattr(
        runner,
        "_forecast_dataset_questions",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        runner,
        "_forecast_market_questions",
        lambda *args, **kwargs: [],
    )

    with pytest.raises(FileExistsError, match=existing_filename):
        runner.run_model(
            FakeRun(),
            _context(),
            tmp_path,
            is_test=True,
            today_date="2026-05-06",
        )

    assert existing_path.read_text(encoding="utf-8") == "existing"


def test_missing_required_question_field_raises_and_does_not_write_file(monkeypatch, tmp_path):
    malformed_question = _dataset_question()
    del malformed_question["background"]
    context = QuestionSetContext(
        forecast_due_date="2026-05-10",
        question_set_filename="2026-05-10-llm.json",
        questions=[malformed_question],
    )

    def fail_write(*args, **kwargs):
        raise AssertionError("forecast file should not be written")

    monkeypatch.setattr(runner.output, "write_forecast_file", fail_write)

    with pytest.raises(KeyError, match="background"):
        runner.run_model(
            FakeRun(),
            context,
            tmp_path,
            is_test=True,
            today_date="2026-05-06",
        )

    assert list(tmp_path.iterdir()) == []


def test_market_question_missing_freeze_value_preserves_zero_shot_file(monkeypatch, tmp_path):
    malformed_question = _market_question()
    del malformed_question["freeze_datetime_value"]
    context = QuestionSetContext(
        forecast_due_date="2026-05-10",
        question_set_filename="2026-05-10-llm.json",
        questions=[malformed_question],
    )

    monkeypatch.setattr(runner.parsing, "parse_market_forecast", lambda *args: 0.4)
    model_run = FakeRun()

    with pytest.raises(KeyError, match="freeze_datetime_value"):
        runner.run_model(
            model_run,
            context,
            tmp_path,
            is_test=True,
            today_date="2026-05-06",
        )

    zero_shot_filename = runner.output.final_filename(
        "2026-05-10",
        model_run,
        ZERO_SHOT,
        is_test=True,
    )
    freeze_filename = runner.output.final_filename(
        "2026-05-10",
        model_run,
        ZERO_SHOT_WITH_FREEZE_VALUES,
        is_test=True,
    )
    assert len(model_run.prompts) == 1
    assert (tmp_path / zero_shot_filename).exists()
    assert not (tmp_path / freeze_filename).exists()


@pytest.mark.parametrize("forecasts", [[0.2], [0.2, 0.3, 0.4]])
def test_dataset_forecast_length_mismatch_skips_whole_question_by_default(monkeypatch, forecasts):
    monkeypatch.setattr(
        runner.parsing,
        "parse_dataset_forecast",
        lambda *args: forecasts,
    )

    rows = runner._forecast_dataset_questions(
        FakeRun(),
        [_dataset_question()],
        forecast_due_date="2026-05-10",
        today_date="2026-05-06",
    )

    assert rows == []


@pytest.mark.parametrize("forecasts", [[0.2], [0.2, 0.3, 0.4]])
def test_dataset_forecast_length_mismatch_raises_when_fail_fast(monkeypatch, forecasts):
    monkeypatch.setattr(
        runner.parsing,
        "parse_dataset_forecast",
        lambda *args: forecasts,
    )

    with pytest.raises(ValueError, match="Expected 2 dataset forecasts"):
        runner._forecast_dataset_questions(
            FakeRun(),
            [_dataset_question()],
            forecast_due_date="2026-05-10",
            today_date="2026-05-06",
            raise_on_question_error=True,
        )

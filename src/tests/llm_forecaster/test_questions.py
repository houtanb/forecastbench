import pytest

from llm_forecaster import questions


def test_split_questions_uses_source_registry_names():
    items = [
        {"id": "fred-1", "source": "fred"},
        {"id": "market-1", "source": "metaculus"},
    ]

    dataset, market = questions.split_questions(items)

    assert dataset == [{"id": "fred-1", "source": "fred"}]
    assert market == [{"id": "market-1", "source": "metaculus"}]


def test_split_questions_rejects_unknown_source():
    with pytest.raises(ValueError, match="Unknown question sources"):
        questions.split_questions([{"id": "q1", "source": "unknown"}])


def test_limit_questions_for_test_mode_limits_each_type():
    dataset = [{"id": f"d{i}", "source": "fred"} for i in range(4)]
    market = [{"id": f"m{i}", "source": "metaculus"} for i in range(4)]

    assert questions.limit_questions_for_test_mode(dataset, market, limit_per_type=2) == (
        dataset[:2],
        market[:2],
    )


def test_question_set_context_from_question_set_json():
    context = questions.QuestionSetContext.from_question_set_json(
        {
            "forecast_due_date": "2026-05-10",
            "question_set": "2026-05-10-llm.json",
            "questions": [{"id": "q1", "source": "fred"}],
        }
    )

    assert context.forecast_due_date == "2026-05-10"
    assert context.question_set_filename == "2026-05-10-llm.json"
    assert context.questions == [{"id": "q1", "source": "fred"}]


def test_question_set_context_from_question_set_json_defaults_missing_question_set():
    context = questions.QuestionSetContext.from_question_set_json(
        {
            "forecast_due_date": "2026-05-10",
            "questions": [{"id": "q1", "source": "fred"}],
        }
    )

    assert context.question_set_filename == "2026-05-10-llm.json"


def test_load_question_set_context_reads_through_orchestration_io(monkeypatch):
    observed = {}

    def fake_read(filename, run_locally=False):
        observed["filename"] = filename
        observed["run_locally"] = run_locally
        return {
            "forecast_due_date": "2026-05-10",
            "question_set": "2026-05-10-llm.json",
            "questions": [{"id": "q1", "source": "fred"}],
        }

    monkeypatch.setattr(questions._io, "read_question_set_json", fake_read)

    context = questions.load_question_set_context("2026-05-10", run_locally=True)

    assert observed == {"filename": "2026-05-10-llm.json", "run_locally": True}
    assert context.forecast_due_date == "2026-05-10"

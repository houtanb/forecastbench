import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_forecaster import smoke_test


def test_module_import_does_not_require_orchestration_io():
    code = """
import importlib.abc
import sys

class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {"orchestration._io", "termcolor"}:
            raise ModuleNotFoundError(f"No module named {fullname!r}")
        return None

sys.meta_path.insert(0, Blocker())
import llm_forecaster.smoke_test
print("ok")
"""

    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        text=True,
        capture_output=True,
        env={**os.environ, "PYTHONPATH": "src"},
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "ok"


def test_select_model_runs_defaults_to_all():
    runs = [SimpleNamespace(slug="a"), SimpleNamespace(slug="b")]
    assert smoke_test.select_model_runs(runs, None) == runs


def test_select_model_runs_rejects_missing_name():
    runs = [SimpleNamespace(slug="a")]
    with pytest.raises(ValueError, match="missing"):
        smoke_test.select_model_runs(runs, ["missing"])


def test_select_questions_takes_deterministic_dataset_and_market_prefixes():
    questions = [
        {"id": "market-b", "source": "metaculus"},
        {"id": "dataset-b", "source": "fred"},
        {"id": "market-a", "source": "manifold"},
        {"id": "dataset-a", "source": "acled"},
        {"id": "dataset-c", "source": "wikipedia"},
        {"id": "market-c", "source": "polymarket"},
    ]

    selected = smoke_test.select_questions(questions, sample_size=2)

    assert selected == [
        {"id": "dataset-a", "source": "acled"},
        {"id": "dataset-b", "source": "fred"},
        {"id": "market-a", "source": "manifold"},
        {"id": "market-b", "source": "metaculus"},
    ]


def test_select_questions_rejects_invalid_or_empty_selection():
    with pytest.raises(ValueError, match="at least 1"):
        smoke_test.select_questions([{"id": "a", "source": "fred"}], sample_size=0)

    with pytest.raises(ValueError, match="No questions selected"):
        smoke_test.select_questions([], sample_size=1)


def test_exit_code_for_results_fails_on_empty_or_any_failure():
    assert smoke_test.exit_code_for_results([]) == 1
    assert smoke_test.exit_code_for_results([SimpleNamespace(status=smoke_test.PASS)]) == 0
    assert smoke_test.exit_code_for_results([SimpleNamespace(status=smoke_test.FAIL)]) == 1


def test_run_smoke_test_continues_after_model_failure(monkeypatch):
    model_runs = [
        SimpleNamespace(
            slug="bad",
            lab=SimpleNamespace(name="Lab"),
            provider=SimpleNamespace(name="Provider"),
        ),
        SimpleNamespace(
            slug="good",
            lab=SimpleNamespace(name="Lab"),
            provider=SimpleNamespace(name="Provider"),
        ),
    ]
    questions = [{"id": "q1", "source": "fred"}]
    run_calls = []

    def run_one(**kwargs):
        run_calls.append(kwargs)
        if kwargs["model_run"].slug == "bad":
            raise RuntimeError("provider unavailable")
        return [
            SimpleNamespace(
                local_filename="/tmp/smoke/good.json",
                rows=[{"id": "q1", "forecast": 0.5}],
            )
        ]

    monkeypatch.setattr(smoke_test, "_get_runner", lambda: SimpleNamespace(run_model=run_one))

    smoke_run = smoke_test.run_smoke_test(
        model_runs=model_runs,
        context=SimpleNamespace(
            forecast_due_date="2026-05-10",
            question_set_filename="2026-05-10-llm.json",
            questions=questions,
        ),
        output_dir="/tmp/smoke",
    )

    assert [result.status for result in smoke_run.results] == [smoke_test.FAIL, smoke_test.PASS]
    assert smoke_run.forecast_file_paths == ["/tmp/smoke/good.json"]
    assert [
        {
            "model_run": call["model_run"].slug,
            "output_dir": call["output_dir"],
            "upload": call["upload"],
            "is_test": call["is_test"],
            "raise_on_question_error": call["raise_on_question_error"],
        }
        for call in run_calls
    ] == [
        {
            "model_run": "bad",
            "output_dir": "/tmp/smoke",
            "upload": False,
            "is_test": True,
            "raise_on_question_error": True,
        },
        {
            "model_run": "good",
            "output_dir": "/tmp/smoke",
            "upload": False,
            "is_test": True,
            "raise_on_question_error": True,
        },
    ]


def test_run_smoke_test_marks_empty_returned_rows_as_failure(monkeypatch):
    model_run = SimpleNamespace(
        slug="empty",
        lab=SimpleNamespace(name="Lab"),
        provider=SimpleNamespace(name="Provider"),
    )

    monkeypatch.setattr(
        smoke_test,
        "_get_runner",
        lambda: SimpleNamespace(
            run_model=lambda **kwargs: [
                SimpleNamespace(local_filename="/tmp/smoke/empty.json", rows=[])
            ]
        ),
    )

    smoke_run = smoke_test.run_smoke_test(
        model_runs=[model_run],
        context=SimpleNamespace(
            forecast_due_date="2026-05-10",
            question_set_filename="2026-05-10-llm.json",
            questions=[{"id": "q1", "source": "fred"}],
        ),
        output_dir="/tmp/smoke",
    )

    assert smoke_run.results[0].status == smoke_test.FAIL
    assert smoke_run.results[0].error_type == "EmptyForecast"
    assert smoke_run.forecast_file_paths == ["/tmp/smoke/empty.json"]


def test_main_uses_latest_metadata_and_configured_paths(monkeypatch):
    calls = {}
    expected_output_dir = Path(smoke_test.SMOKE_OUTPUT_DIR) / "run-1"
    selected_run = SimpleNamespace(
        slug="b",
        lab=SimpleNamespace(name="Lab B"),
        provider=SimpleNamespace(name="Provider B"),
    )
    all_runs = [
        SimpleNamespace(
            slug="a",
            lab=SimpleNamespace(name="Lab A"),
            provider=SimpleNamespace(name="Provider A"),
        ),
        selected_run,
    ]

    monkeypatch.delenv("FORECAST_DUE_DATE", raising=False)
    monkeypatch.setattr(
        smoke_test.sys,
        "argv",
        ["smoke_test.py", "--model-run", "b", "--sample-size", "2"],
    )

    def load_context(forecast_due_date, run_locally=False):
        calls["loaded"] = (forecast_due_date, run_locally)
        return SimpleNamespace(
            forecast_due_date=forecast_due_date,
            question_set_filename=f"{forecast_due_date}-llm.json",
            questions=[
                {"id": "dataset-2", "source": "fred"},
                {"id": "market-2", "source": "metaculus"},
                {"id": "dataset-1", "source": "acled"},
                {"id": "market-1", "source": "manifold"},
                {"id": "dataset-3", "source": "wikipedia"},
                {"id": "market-3", "source": "polymarket"},
            ],
        )

    def configure_provider_keys(selected_runs):
        calls["configured"] = list(selected_runs)

    def run_smoke(model_runs, context, output_dir=smoke_test.SMOKE_OUTPUT_DIR):
        calls["smoke"] = (list(model_runs), context, output_dir)
        return smoke_test.SmokeRun(
            results=[
                smoke_test.SmokeResult(
                    model_name="b",
                    lab="Lab B",
                    provider="Provider B",
                    status=smoke_test.FAIL,
                    error_type="RuntimeError",
                    error_message="failed",
                )
            ],
            forecast_file_paths=["/tmp/smoke/b.json"],
        )

    monkeypatch.setattr(
        smoke_test,
        "_get_io_module",
        lambda: SimpleNamespace(
            get_latest_llm_question_set_metadata=lambda run_locally=False: {
                "forecast_due_date": "2026-05-10",
                "question_set": "latest",
            }
        ),
    )
    monkeypatch.setattr(
        smoke_test,
        "_get_questions_module",
        lambda: SimpleNamespace(
            QuestionSetContext=SimpleNamespace,
            load_question_set_context=load_context,
        ),
    )
    monkeypatch.setattr(
        smoke_test,
        "_get_model_runs_module",
        lambda: SimpleNamespace(
            MODEL_RUNS=all_runs,
            configure_and_validate_provider_keys=configure_provider_keys,
        ),
    )
    monkeypatch.setattr(smoke_test, "run_smoke_test", run_smoke)
    monkeypatch.setattr(smoke_test, "_new_output_dir", lambda: expected_output_dir)

    with pytest.raises(SystemExit) as exc_info:
        smoke_test.main()

    assert exc_info.value.code == 1
    assert calls["loaded"] == ("2026-05-10", False)
    assert calls["configured"] == [selected_run]
    selected_model_runs, selected_context, output_dir = calls["smoke"]
    assert selected_model_runs == [selected_run]
    assert selected_context.questions == [
        {"id": "dataset-1", "source": "acled"},
        {"id": "dataset-2", "source": "fred"},
        {"id": "market-1", "source": "manifold"},
        {"id": "market-2", "source": "metaculus"},
    ]
    assert selected_context.forecast_due_date == "2026-05-10"
    assert selected_context.question_set_filename == "2026-05-10-llm.json"
    assert output_dir == expected_output_dir
    assert output_dir.parent == Path(smoke_test.SMOKE_OUTPUT_DIR)

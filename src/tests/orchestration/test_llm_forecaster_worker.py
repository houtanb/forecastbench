from pathlib import Path
from types import SimpleNamespace

from utils.llm.provider_registry import PROVIDERS

from helpers import constants
from orchestration.func_llm_forecaster_worker import main as worker

ROOT = Path(__file__).resolve().parents[3]
UTILS_PIN = "git+https://github.com/forecastingresearch/utils@"


def _context():
    return worker.questions.QuestionSetContext(
        forecast_due_date="2026-05-10",
        question_set_filename="2026-05-10-llm.json",
        questions=[
            {"id": "d1", "source": "fred"},
            {"id": "d2", "source": "fred"},
            {"id": "d3", "source": "fred"},
            {"id": "m1", "source": "metaculus"},
            {"id": "m2", "source": "metaculus"},
            {"id": "m3", "source": "metaculus"},
        ],
    )


def test_parse_env_defaults_missing_test_or_prod_to_test(monkeypatch):
    monkeypatch.setenv("FORECAST_DUE_DATE", "2026-05-10")
    monkeypatch.setenv("CLOUD_RUN_TASK_INDEX", "0")
    monkeypatch.delenv("TEST_OR_PROD", raising=False)
    monkeypatch.setattr(worker.model_runs, "MODEL_RUNS", [SimpleNamespace(slug="model")])

    forecast_due_date, run_mode, model_run = worker.parse_env_vars()

    assert forecast_due_date == "2026-05-10"
    assert run_mode == constants.RunMode.TEST
    assert model_run.slug == "model"


def test_parse_env_non_prod_value_defaults_to_test(monkeypatch):
    monkeypatch.setenv("FORECAST_DUE_DATE", "2026-05-10")
    monkeypatch.setenv("CLOUD_RUN_TASK_INDEX", "0")
    monkeypatch.setenv("TEST_OR_PROD", "DEV")
    monkeypatch.setattr(worker.model_runs, "MODEL_RUNS", [SimpleNamespace(slug="model")])

    assert worker.parse_env_vars()[1] == constants.RunMode.TEST


def test_worker_limits_questions_to_two_dataset_then_two_market_when_not_prod(monkeypatch):
    calls = {}
    selected_model = SimpleNamespace(slug="model", provider=PROVIDERS["OpenAI"])

    monkeypatch.setattr(
        worker.questions, "load_question_set_context", lambda forecast_due_date: _context()
    )
    monkeypatch.setattr(
        worker.model_runs,
        "configure_and_validate_provider_keys",
        lambda runs: calls.setdefault("configured_runs", runs),
    )
    monkeypatch.setattr(
        worker.runner, "run_model", lambda **kwargs: calls.setdefault("run", kwargs)
    )

    worker.run_worker(
        forecast_due_date="2026-05-10",
        run_mode=constants.RunMode.TEST,
        model_run=selected_model,
    )

    context = calls["run"]["context"]
    assert context.forecast_due_date == "2026-05-10"
    assert context.question_set_filename == "2026-05-10-llm.json"
    assert [q["id"] for q in context.questions] == ["d1", "d2", "m1", "m2"]
    assert calls["configured_runs"] == [selected_model]
    assert calls["run"]["upload"] is True
    assert calls["run"]["is_test"] is True


def test_worker_prod_does_not_limit_questions_and_uploads_prod(monkeypatch):
    calls = {}
    selected_model = SimpleNamespace(slug="model", provider=PROVIDERS["OpenAI"])

    monkeypatch.setattr(
        worker.questions, "load_question_set_context", lambda forecast_due_date: _context()
    )
    monkeypatch.setattr(
        worker.model_runs, "configure_and_validate_provider_keys", lambda runs: None
    )
    monkeypatch.setattr(
        worker.runner, "run_model", lambda **kwargs: calls.setdefault("run", kwargs)
    )

    worker.run_worker(
        forecast_due_date="2026-05-10",
        run_mode=constants.RunMode.PROD,
        model_run=selected_model,
    )

    assert [q["id"] for q in calls["run"]["context"].questions] == [
        "d1",
        "d2",
        "d3",
        "m1",
        "m2",
        "m3",
    ]
    assert calls["run"]["model_run"] == selected_model
    assert calls["run"]["upload"] is True
    assert calls["run"]["is_test"] is False


def test_parse_env_selects_model_by_cloud_run_task_index(monkeypatch):
    model_runs = [
        SimpleNamespace(slug="first"),
        SimpleNamespace(slug="second"),
        SimpleNamespace(slug="third"),
    ]
    monkeypatch.setenv("FORECAST_DUE_DATE", "2026-05-10")
    monkeypatch.setenv("CLOUD_RUN_TASK_INDEX", "1")
    monkeypatch.setenv("TEST_OR_PROD", "PROD")
    monkeypatch.setattr(worker.model_runs, "MODEL_RUNS", model_runs)

    forecast_due_date, run_mode, model_run = worker.parse_env_vars()

    assert forecast_due_date == "2026-05-10"
    assert run_mode == constants.RunMode.PROD
    assert model_run == model_runs[1]


def test_worker_deploy_stages_runtime_requirements_and_shared_code():
    deploy_dir = ROOT / "src/orchestration/func_llm_forecaster_worker"
    makefile = (deploy_dir / "Makefile").read_text()
    requirements = (deploy_dir / "requirements.txt").read_text()

    assert "func-llm-forecaster-worker" in makefile
    assert "--service-account $(QUESTION_BANK_BUCKET_SERVICE_ACCOUNT)" in makefile
    assert (
        "cat $(ROOT_DIR)requirements.runtime.txt requirements.txt > $(UPLOAD_DIR)/requirements.txt"
        in makefile
    )
    assert "cp -r $(ROOT_DIR)src/helpers $(UPLOAD_DIR)/" in makefile
    assert "cp -r $(ROOT_DIR)src/sources $(UPLOAD_DIR)/" in makefile
    assert "cp -r $(ROOT_DIR)src/llm_forecaster $(UPLOAD_DIR)/" in makefile
    assert "cp $(ROOT_DIR)src/orchestration/_io.py $(UPLOAD_DIR)/orchestration/" in makefile
    assert "cp $(ROOT_DIR)src/_fb_types.py $(UPLOAD_DIR)/" in makefile
    assert "cp $(ROOT_DIR)src/_schemas.py $(UPLOAD_DIR)/" in makefile
    assert UTILS_PIN not in requirements

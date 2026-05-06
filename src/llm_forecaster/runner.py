"""Run ForecastBench LLM forecast generation."""

import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

from utils import gcp

from helpers import dates, env
from llm_forecaster import model_runs, output, parsing, prompts, questions
from llm_forecaster.forecast_variants import (
    DATASET_FORECAST_SHARING_VARIANT_GROUPS,
    ZERO_SHOT,
    ForecastVariant,
)
from sources import DATASET_SOURCE_NAMES

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WrittenForecastFile:
    """A final forecast file written by a runner invocation."""

    variant: ForecastVariant
    local_filename: Path
    rows: list[dict[str, Any]]


PromptKey = tuple[str, str, ForecastVariant]
RenderedPrompts = dict[PromptKey, str]
QuestionForecastFn = Callable[[dict[str, Any]], list[dict[str, Any]]]


class LLMCallTranscript:
    """Write test-mode LLM prompts and responses to a Markdown transcript."""

    def __init__(self, local_filename: str | Path) -> None:
        """Create a fresh transcript file for this runner invocation."""
        self.local_filename = Path(local_filename)
        self.local_filename.parent.mkdir(parents=True, exist_ok=True)
        self.local_filename.write_text("# LLM Call Transcript\n", encoding="utf-8")
        self._lock = Lock()
        self._next_call_index = 1

    @staticmethod
    def _fenced_text(value: str | None) -> str:
        """Return text in a Markdown fence without escaping prompt newlines."""
        if value is None:
            value = ""
        fence = "```"
        while fence in value:
            fence += "`"
        return f"{fence}text\n{value}\n{fence}"

    def record(
        self,
        *,
        role: str,
        model_run,
        question: dict[str, Any],
        variant: ForecastVariant,
        prompt: str,
        response: str | None = None,
        error: str | None = None,
    ) -> None:
        """Append one completed LLM call to the transcript."""
        provider = getattr(model_run, "provider", None)

        with self._lock:
            call_index = self._next_call_index
            self._next_call_index += 1
            section = [
                "",
                f"## Call {call_index}: {role} ({variant.key})",
                "",
                f"- Provider: {getattr(provider, 'name', str(provider))}",
                f"- Model run slug: {model_run.slug}",
                f"- Provider model ID: {model_run.provider_model_id}",
                f"- Source: {question['source']}",
                f"- Question ID: {question['id']}",
                f"- Variant: {variant.key}",
                f"- Role: {role}",
                "",
                "### Prompt",
                "",
                self._fenced_text(prompt),
                "",
                "### Response",
                "",
                self._fenced_text(response),
            ]
            if error is not None:
                section.extend(
                    [
                        "",
                        "### Error",
                        "",
                        self._fenced_text(error),
                    ]
                )

            with self.local_filename.open("a", encoding="utf-8") as transcript_file:
                transcript_file.write("\n".join(section) + "\n")


class RecordingRun:
    """Wrap a model run so parsing-time LLM calls are recorded in test mode."""

    def __init__(
        self,
        model_run,
        *,
        transcript: LLMCallTranscript,
        question: dict[str, Any],
        variant: ForecastVariant,
        role: str,
    ) -> None:
        """Store the call context used when recording wrapped model responses."""
        self._model_run = model_run
        self._transcript = transcript
        self._question = question
        self._variant = variant
        self._role = role

    def __getattr__(self, name: str):
        """Delegate model-run attributes to the wrapped model run."""
        return getattr(self._model_run, name)

    def get_response(self, prompt: str) -> str:
        """Request and record one parsing-time model response."""
        return _get_model_response(
            self._model_run,
            prompt,
            question=self._question,
            variant=self._variant,
            role=self._role,
            transcript=self._transcript,
        )


def _test_llm_call_transcript_filename(forecast_due_date: str, model_run) -> str:
    """Return the test-mode prompt/response transcript filename."""
    return f"TEST.{forecast_due_date}.forecastbench.{model_run.slug}.llm-calls.md"


def _get_model_response(
    model_run,
    prompt: str,
    *,
    question: dict[str, Any],
    variant: ForecastVariant,
    role: str,
    transcript: LLMCallTranscript | None = None,
) -> str:
    """Request an LLM response and record it when test-mode tracing is enabled."""
    try:
        response = model_run.get_response(prompt)
    except Exception as exc:
        if transcript is not None:
            transcript.record(
                role=role,
                model_run=model_run,
                question=question,
                variant=variant,
                prompt=prompt,
                error=f"{type(exc).__name__}: {exc}",
            )
        raise

    if transcript is not None:
        transcript.record(
            role=role,
            model_run=model_run,
            question=question,
            variant=variant,
            prompt=prompt,
            response=response,
        )
    return response


def _reformat_model_for_question(
    transcript: LLMCallTranscript | None,
    *,
    question: dict[str, Any],
    variant: ForecastVariant,
):
    """Return the reformat model, wrapped with recording when tracing is enabled."""
    if transcript is None:
        return model_runs.REFORMAT_MODEL
    return RecordingRun(
        model_runs.REFORMAT_MODEL,
        transcript=transcript,
        question=question,
        variant=variant,
        role="reformat",
    )


def _formatted_question(question: dict[str, Any], forecast_due_date: str) -> str:
    formatted = question["question"].replace("{forecast_due_date}", forecast_due_date)
    return formatted.replace("{resolution_date}", "each of the resolution dates provided below")


def _background(question: dict[str, Any]) -> str:
    background = question["background"]
    if question["market_info_resolution_criteria"] != "N/A":
        background += "\n" + question["market_info_resolution_criteria"]
    return background


def _prompt_params(
    question: dict[str, Any],
    *,
    forecast_due_date: str,
    today_date: str,
    variant: ForecastVariant,
) -> dict[str, Any]:
    params = {
        "question": _formatted_question(question, forecast_due_date),
        "background": _background(question),
        "resolution_criteria": question["resolution_criteria"],
        "today_date": today_date,
    }

    if question["source"] in DATASET_SOURCE_NAMES:
        params.update(
            {
                "freeze_datetime": question["freeze_datetime"],
                "freeze_datetime_value": question["freeze_datetime_value"],
                "freeze_datetime_value_explanation": question["freeze_datetime_value_explanation"],
                "list_of_resolution_dates": question["resolution_dates"],
            }
        )
        return params

    params["resolution_date"] = question["market_info_close_datetime"]
    if variant.uses_freeze_values:
        params.update(
            {
                "freeze_datetime": question["freeze_datetime"],
                "freeze_datetime_value": question["freeze_datetime_value"],
            }
        )
    return params


def render_prompt(
    question: dict[str, Any],
    *,
    forecast_due_date: str,
    today_date: str,
    variant: ForecastVariant,
) -> str:
    """Render the legacy zero-shot prompt for a question and variant."""
    params = _prompt_params(
        question,
        forecast_due_date=forecast_due_date,
        today_date=today_date,
        variant=variant,
    )

    if question["source"] in DATASET_SOURCE_NAMES:
        prompt_template = prompts.ZERO_SHOT_DATASET_PROMPT
    elif variant.uses_freeze_values:
        prompt_template = prompts.ZERO_SHOT_MARKET_WITH_FREEZE_VALUE_PROMPT
    else:
        prompt_template = prompts.ZERO_SHOT_MARKET_PROMPT

    return prompt_template.format(**params)


def _prompt_key(question: dict[str, Any], variant: ForecastVariant) -> PromptKey:
    """Return the rendered-prompt cache key for one question and variant."""
    return (question["source"], question["id"], variant)


def _render_prompts(
    questions_to_render: list[dict[str, Any]],
    *,
    forecast_due_date: str,
    today_date: str,
    variant: ForecastVariant,
) -> RenderedPrompts:
    """Render prompts for one runnable question phase."""
    rendered_prompts = {}
    for question in questions_to_render:
        rendered_prompts[_prompt_key(question, variant)] = render_prompt(
            question,
            forecast_due_date=forecast_due_date,
            today_date=today_date,
            variant=variant,
        )
    return rendered_prompts


def _render_dataset_prompts(
    dataset_questions: list[dict[str, Any]],
    *,
    forecast_due_date: str,
    today_date: str,
    variant: ForecastVariant = ZERO_SHOT,
) -> RenderedPrompts:
    """Render dataset prompts before one shared forecast pass for a variant group."""
    return _render_prompts(
        dataset_questions,
        forecast_due_date=forecast_due_date,
        today_date=today_date,
        variant=variant,
    )


def _render_market_prompts(
    market_questions: list[dict[str, Any]],
    *,
    forecast_due_date: str,
    today_date: str,
    variant: ForecastVariant,
) -> RenderedPrompts:
    """Render market prompts for exactly one forecast variant."""
    return _render_prompts(
        market_questions,
        forecast_due_date=forecast_due_date,
        today_date=today_date,
        variant=variant,
    )


def _get_rendered_prompt(
    rendered_prompts: RenderedPrompts | None,
    *,
    variant: ForecastVariant,
    question: dict[str, Any],
    forecast_due_date: str,
    today_date: str,
) -> str:
    """Return a cached rendered prompt, rendering on demand when no cache is provided."""
    if rendered_prompts is not None:
        return rendered_prompts[_prompt_key(question, variant)]
    return render_prompt(
        question,
        forecast_due_date=forecast_due_date,
        today_date=today_date,
        variant=variant,
    )


def _handle_question_error(question: dict[str, Any], raise_on_question_error: bool) -> None:
    """Raise or log an exception from an individual question forecast."""
    if raise_on_question_error:
        raise
    logger.exception("Skipping LLM forecast question after error: %s", question.get("id"))


def _max_workers_for_questions(model_run, question_count: int) -> int:
    """Return the bounded question-level worker count for one model run."""
    provider_max_workers = model_runs.PROVIDER_MAX_WORKERS[model_run.provider]
    return max(1, min(question_count, provider_max_workers))


def _forecast_questions(
    model_run,
    questions_to_forecast: list[dict[str, Any]],
    forecast_question: QuestionForecastFn,
) -> list[dict[str, Any]]:
    """Forecast questions concurrently while preserving input question order."""
    max_workers = _max_workers_for_questions(model_run, len(questions_to_forecast))
    if max_workers == 1:
        question_rows = [forecast_question(question) for question in questions_to_forecast]
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            question_rows = list(executor.map(forecast_question, questions_to_forecast))
    return [row for rows in question_rows for row in rows]


def _forecast_dataset_questions(
    model_run,
    dataset_questions: list[dict[str, Any]],
    *,
    forecast_due_date: str,
    today_date: str,
    variant: ForecastVariant = ZERO_SHOT,
    raise_on_question_error: bool = False,
    rendered_prompts: RenderedPrompts | None = None,
    transcript: LLMCallTranscript | None = None,
) -> list[dict[str, Any]]:
    """Forecast dataset questions once and return one row per resolution date."""

    def forecast_question(question: dict[str, Any]) -> list[dict[str, Any]]:
        question_id = question["id"]
        source = question["source"]
        resolution_dates = question["resolution_dates"]
        prompt = _get_rendered_prompt(
            rendered_prompts,
            variant=variant,
            question=question,
            forecast_due_date=forecast_due_date,
            today_date=today_date,
        )
        try:
            response = _get_model_response(
                model_run,
                prompt,
                question=question,
                variant=variant,
                role="forecast",
                transcript=transcript,
            )
            forecasts = parsing.parse_dataset_forecast(
                response,
                prompt,
                question,
                _reformat_model_for_question(
                    transcript,
                    question=question,
                    variant=variant,
                ),
            )
            if len(forecasts) != len(resolution_dates):
                raise ValueError(
                    f"Expected {len(resolution_dates)} dataset forecasts for "
                    f"{question_id}, got {len(forecasts)}"
                )
        except Exception:
            _handle_question_error(question, raise_on_question_error)
            return []
        return [
            {
                "id": question_id,
                "source": source,
                "forecast": forecast,
                "resolution_date": resolution_date,
                "reasoning": None,
            }
            for forecast, resolution_date in zip(forecasts, resolution_dates)
        ]

    return _forecast_questions(model_run, dataset_questions, forecast_question)


def _forecast_market_questions(
    model_run,
    market_questions: list[dict[str, Any]],
    *,
    forecast_due_date: str,
    today_date: str,
    variant: ForecastVariant,
    raise_on_question_error: bool = False,
    rendered_prompts: RenderedPrompts | None = None,
    transcript: LLMCallTranscript | None = None,
) -> list[dict[str, Any]]:
    """Forecast market questions for one variant."""

    def forecast_question(question: dict[str, Any]) -> list[dict[str, Any]]:
        question_id = question["id"]
        source = question["source"]
        prompt = _get_rendered_prompt(
            rendered_prompts,
            variant=variant,
            question=question,
            forecast_due_date=forecast_due_date,
            today_date=today_date,
        )
        try:
            response = _get_model_response(
                model_run,
                prompt,
                question=question,
                variant=variant,
                role="forecast",
                transcript=transcript,
            )
            forecast = parsing.parse_market_forecast(
                response,
                _reformat_model_for_question(
                    transcript,
                    question=question,
                    variant=variant,
                ),
            )
        except Exception:
            _handle_question_error(question, raise_on_question_error)
            return []
        return [
            {
                "id": question_id,
                "source": source,
                "forecast": forecast,
                "resolution_date": None,
                "reasoning": None,
            }
        ]

    return _forecast_questions(model_run, market_questions, forecast_question)


def _write_final_file(
    *,
    model_run,
    context: questions.QuestionSetContext,
    output_dir: str | Path,
    variant: ForecastVariant,
    rows: list[dict[str, Any]],
    upload: bool,
    is_test: bool,
) -> WrittenForecastFile:
    local_filename = Path(output_dir) / output.final_filename(
        context.forecast_due_date,
        model_run,
        variant,
        is_test,
    )
    if local_filename.exists():
        raise FileExistsError(f"Final forecast file already exists: {local_filename}")

    destination_blob_name = output.destination_blob_name(
        context.forecast_due_date,
        model_run,
        variant,
        is_test,
    )
    if upload and gcp.storage.file_exists(
        bucket_name=env.FORECAST_SETS_BUCKET,
        filename=destination_blob_name,
    ):
        raise FileExistsError(f"Remote forecast file already exists: {destination_blob_name}")

    output.write_forecast_file(
        local_filename=local_filename,
        forecast_due_date=context.forecast_due_date,
        question_set_filename=context.question_set_filename,
        model_run=model_run,
        variant=variant,
        rows=rows,
    )

    if upload:
        gcp.storage.upload(
            bucket_name=env.FORECAST_SETS_BUCKET,
            local_filename=str(local_filename),
            filename=destination_blob_name,
        )

    return WrittenForecastFile(variant=variant, local_filename=local_filename, rows=rows)


def run_model(
    model_run,
    context: questions.QuestionSetContext,
    output_dir: str | Path,
    upload: bool = False,
    is_test: bool = False,
    today_date: str | None = None,
    raise_on_question_error: bool = False,
) -> list[WrittenForecastFile]:
    """Run one model and write final forecast files in variant order."""
    if today_date is None:
        today_date = dates.get_date_today_as_iso()

    dataset_questions, market_questions = questions.split_questions(context.questions)
    dataset_prompts_by_variant = {}
    for variant_group in DATASET_FORECAST_SHARING_VARIANT_GROUPS:
        dataset_variant = variant_group[0]
        dataset_prompts_by_variant[dataset_variant] = _render_dataset_prompts(
            dataset_questions,
            forecast_due_date=context.forecast_due_date,
            today_date=today_date,
            variant=dataset_variant,
        )

    transcript = None
    if is_test:
        transcript = LLMCallTranscript(
            Path(output_dir)
            / _test_llm_call_transcript_filename(context.forecast_due_date, model_run)
        )
        logger.info("Writing test LLM call transcript to %s.", transcript.local_filename)

    written_files = []
    for variant_group in DATASET_FORECAST_SHARING_VARIANT_GROUPS:
        dataset_variant = variant_group[0]
        dataset_rows = _forecast_dataset_questions(
            model_run,
            dataset_questions,
            forecast_due_date=context.forecast_due_date,
            today_date=today_date,
            variant=dataset_variant,
            raise_on_question_error=raise_on_question_error,
            rendered_prompts=dataset_prompts_by_variant[dataset_variant],
            transcript=transcript,
        )

        for variant in variant_group:
            market_prompts = _render_market_prompts(
                market_questions,
                forecast_due_date=context.forecast_due_date,
                today_date=today_date,
                variant=variant,
            )
            rows = dataset_rows + _forecast_market_questions(
                model_run,
                market_questions,
                forecast_due_date=context.forecast_due_date,
                today_date=today_date,
                variant=variant,
                raise_on_question_error=raise_on_question_error,
                rendered_prompts=market_prompts,
                transcript=transcript,
            )
            written_files.append(
                _write_final_file(
                    model_run=model_run,
                    context=context,
                    output_dir=output_dir,
                    variant=variant,
                    rows=rows,
                    upload=upload,
                    is_test=is_test,
                )
            )

    return written_files

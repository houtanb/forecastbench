# LLM Forecaster Package

This package owns ForecastBench LLM baseline generation.

- Use `utils.llm` for provider calls; do not instantiate provider SDK clients directly here.
- Select model runs from `utils.llm.model_runs` by `model_run_key`; do not declare local
  `ModelRun` registries here.
- Keep all runtime model options in the `ModelRun.options` declaration.
- Keep model-run names lower-case, file-safe, and explicit about non-default runtime options.
- Use `dataset` terminology for dataset sources in new code.
- Preserve existing ForecastBench LLM prompt text and parsing behavior in this pass.
- Keep Cloud Run entrypoints thin; put behavior in `src/llm_forecaster`.
- Preserve the exact current ForecastBench LLM forecast-file schema; only model naming changes.
- Do not add `from __future__ import annotations` to new ForecastBench files.

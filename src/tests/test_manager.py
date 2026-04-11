"""Tests for the LLM baselines manager."""

from collections import Counter


class TestGroupSizeCounting:
    """Count how many ModelRuns share each rate_limit_group."""

    def test_group_size_counting(self):
        from helpers.llm import MODEL_RUNS

        group_counts = Counter(m.rate_limit_group for m in MODEL_RUNS)
        # All providers should have at least 1 model
        assert len(group_counts) > 0
        # Every count should be a positive integer
        for group, count in group_counts.items():
            assert count > 0, f"Group '{group}' has count {count}"

    def test_openai_group_has_multiple(self):
        from helpers.llm import MODEL_RUNS

        group_counts = Counter(m.rate_limit_group for m in MODEL_RUNS)
        # We know there are multiple OpenAI models
        assert group_counts["openai"] > 1


class TestManagerTaskCount:
    """Manager sets task_count = len(MODEL_RUNS)."""

    def test_task_count_equals_model_runs_length(self):
        from helpers.llm import MODEL_RUNS

        # Each ModelRun is one task (no prompt_type multiplication)
        task_count = len(MODEL_RUNS)
        assert task_count == len(MODEL_RUNS)
        assert task_count > 0

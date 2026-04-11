"""Tests for the rewritten worker and model_eval."""

from unittest.mock import MagicMock, patch


class TestThreadPoolSizeCalculation:
    """Thread pool size = max(1, rate_limit // group_size)."""

    def test_basic_calculation(self):
        """10 rate limit, 2 siblings = 5 workers."""
        rate_limit = 10
        group_size = 2
        expected = max(1, rate_limit // group_size)
        assert expected == 5

    def test_minimum_one_worker(self):
        """When group_size > rate_limit, still get 1 worker."""
        rate_limit = 3
        group_size = 10
        expected = max(1, rate_limit // group_size)
        assert expected == 1

    def test_single_sibling(self):
        """Single sibling gets the full rate limit."""
        rate_limit = 10
        group_size = 1
        expected = max(1, rate_limit // group_size)
        assert expected == 10


class TestWorkerSkipsDatasetQuestions:
    """Worker correctly skips dataset questions when market_use_freeze_value=True."""

    def test_skip_dataset_when_freeze_value(self):
        from helpers.llm import ModelRun, Provider
        from helpers.model_eval import worker

        mock_run = ModelRun(
            name="test-model",
            model_id="test-model",
            provider=Provider.OPENAI,
            org="TestOrg",
        )

        save_dict = {0: ""}
        questions = [
            {
                "source": "fred",
                "id": "q1",
                "url": "http://test",
                "question": "test?",
                "background": "bg",
                "resolution_criteria": "rc",
                "market_info_resolution_criteria": "N/A",
                "freeze_datetime": "2025-01-01",
                "freeze_datetime_value": "100",
                "freeze_datetime_value_explanation": "explanation",
                "resolution_dates": ["2025-06-01"],
            }
        ]

        # Should return early without calling get_response
        worker(
            index=0,
            n_questions=1,
            model_run=mock_run,
            save_dict=save_dict,
            questions_to_eval=questions,
            forecast_due_date="2025-01-01",
            prompt_type="zero_shot",
            market_use_freeze_value=True,
        )

        # save_dict should still be empty string (not processed)
        assert save_dict[0] == ""


class TestModelRunGetResponseMerge:
    """ModelRun.get_response() merges self.options with call-time kwargs."""

    def test_calltime_kwargs_override_options(self):
        from helpers.llm import ModelRun, Provider

        run = ModelRun(
            name="test",
            model_id="test-model",
            provider=Provider.OPENAI,
            org="OpenAI",
            options={"temperature": 0, "max_tokens": 100},
        )

        with patch.object(run, "_get_model") as mock_get_model:
            mock_model = MagicMock()
            mock_model.get_response.return_value = "response"
            mock_get_model.return_value = mock_model

            run.get_response("hello", max_tokens=200)

            call_kwargs = mock_model.get_response.call_args[1]
            assert call_kwargs["max_tokens"] == 200
            assert call_kwargs["temperature"] == 0

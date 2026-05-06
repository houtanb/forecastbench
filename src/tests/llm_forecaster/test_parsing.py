from types import SimpleNamespace

from llm_forecaster import parsing


def test_extract_probability_preserves_current_reverse_search_behavior():
    assert parsing.extract_probability("first 0.2 then final *0.73*") == 0.73
    assert parsing.extract_probability("I think 61%") == 0.61
    assert parsing.extract_probability("0.0 or 1.0") is None
    assert parsing.extract_probability(None) is None


def test_convert_string_to_list_preserves_current_behavior():
    assert parsing.convert_string_to_list("[0.1, 0.25, *]") == [0.1, 0.25, 0.5]


def test_parse_dataset_forecast_directly_parses_expected_starred_probabilities():
    def get_response(prompt):
        raise AssertionError(f"Unexpected reformat call: {prompt}")

    reformat_model = SimpleNamespace(get_response=get_response)
    question = {
        "resolution_dates": [
            "2026-06-01",
            "2026-07-01",
            "2026-08-01",
            "2026-09-01",
            "2026-10-01",
            "2026-11-01",
            "2026-12-01",
            "2027-01-01",
        ]
    }

    assert parsing.parse_dataset_forecast(
        response="*0.01* *0.01* *0.02* *0.02* *0.03* *0.05* *0.06* *0.08*\n",
        prompt="prompt text",
        question=question,
        reformat_model=reformat_model,
    ) == [0.01, 0.01, 0.02, 0.02, 0.03, 0.05, 0.06, 0.08]


def test_parse_dataset_forecast_falls_back_to_reformat_when_count_mismatches():
    calls = []

    def get_response(prompt):
        calls.append(prompt)
        return "[0.2, 0.3]"

    reformat_model = SimpleNamespace(get_response=get_response)
    question = {"resolution_dates": ["2026-06-01", "2026-07-01"]}

    assert parsing.parse_dataset_forecast(
        response="*0.2*",
        prompt="prompt text",
        question=question,
        reformat_model=reformat_model,
    ) == [0.2, 0.3]
    assert len(calls) == 1


def test_parse_market_forecast_uses_extraction_without_reformat_first():
    reformat_model = SimpleNamespace(get_response=lambda prompt: "0.25")
    assert parsing.parse_market_forecast("Reasoning. Final answer: *0.64*", reformat_model) == 0.64


def test_parse_market_forecast_does_not_reformat_when_extraction_fails():
    def get_response(prompt):
        raise AssertionError(f"Unexpected reformat call: {prompt}")

    reformat_model = SimpleNamespace(get_response=get_response)

    assert parsing.parse_market_forecast("No number here", reformat_model) is None


def test_parse_dataset_forecast_uses_list_reformat():
    calls = []

    def get_response(prompt):
        calls.append(prompt)
        return "[0.2, 0.3]"

    reformat_model = SimpleNamespace(get_response=get_response)
    question = {"resolution_dates": ["2026-06-01", "2026-07-01"]}

    assert parsing.parse_dataset_forecast(
        response="first date 20, second date 30",
        prompt="prompt text",
        question=question,
        reformat_model=reformat_model,
    ) == [0.2, 0.3]
    assert "prompt text" in calls[0]
    assert "2 resolution dates" in calls[0]


def test_parse_dataset_forecast_uses_second_list_reformat_prompt_when_needed():
    calls = []
    responses = iter(["need_a_new_reformat_prompt", "[0.4, 0.6]"])

    def get_response(prompt):
        calls.append(prompt)
        return next(responses)

    reformat_model = SimpleNamespace(get_response=get_response)
    question = {"resolution_dates": ["2026-06-01", "2026-07-01"]}

    assert parsing.parse_dataset_forecast(
        response="ambiguous response",
        prompt="original prompt",
        question=question,
        reformat_model=reformat_model,
    ) == [0.4, 0.6]
    assert len(calls) == 2
    assert "Please output the probabilistic forecasts as a Python list" in calls[0]
    assert "Output the probabilistic forecasts as a Python list" in calls[1]

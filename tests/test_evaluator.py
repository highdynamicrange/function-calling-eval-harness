from fc_eval.adapters import MockModelAdapter, ModelAdapter
from fc_eval.evaluator import (
    aggregate_metrics,
    argument_stats,
    count_errors,
    evaluate_case,
    evaluate_dataset,
    slice_metrics,
    tool_selection_correct,
)
from fc_eval.models import EvalCase, ModelResponse, ToolCall
from fc_eval.tools import MockToolExecutor


def make_case(
    case_id="case",
    behavior="call",
    calls=None,
    order_mode="strict",
    tags=None,
):
    return EvalCase.model_validate(
        {
            "id": case_id,
            "messages": [{"role": "user", "content": "test"}],
            "available_tools": [
                "get_weather",
                "web_search",
                "send_email",
                "get_order_status",
            ],
            "expected_calls": calls or [],
            "order_mode": order_mode,
            "expected_behavior": behavior,
            "expected_final_state": {},
            "tags": tags or ["unit"],
            "difficulty": "medium",
        }
    )


class FixedAdapter(ModelAdapter):
    def __init__(self, response):
        self.response = response

    def generate(self, case, tools):
        return self.response


def test_strict_and_any_tool_selection():
    expected = [
        ToolCall(name="get_weather", arguments={"location": "北京"}),
        ToolCall(name="web_search", arguments={"query": "x"}),
    ]
    actual = list(reversed(expected))
    assert not tool_selection_correct(make_case(calls=expected), actual)
    assert tool_selection_correct(make_case(calls=expected, order_mode="any"), actual)


def test_argument_stats_counts_matches_and_mismatches():
    case = make_case(
        calls=[ToolCall(name="get_weather", arguments={"location": "北京", "date": "today"})]
    )
    exact, tp, fp, fn = argument_stats(
        case,
        [ToolCall(name="get_weather", arguments={"location": "上海", "extra": 1})],
    )
    assert not exact
    assert (tp, fp, fn) == (0, 2, 2)


def test_successful_call_and_aggregate_metrics():
    case = make_case(
        calls=[ToolCall(name="get_weather", arguments={"location": "北京", "date": "today"})]
    )
    result = evaluate_case(case, MockModelAdapter(), MockToolExecutor())
    metrics = aggregate_metrics([result])
    assert result.task_completed
    assert metrics["tool_selection_accuracy"] == 1.0
    assert metrics["argument_field_f1"] == 1.0
    assert metrics["task_completion_rate"] == 1.0


def test_no_call_clarification_and_confirmation():
    cases = [
        make_case("no", "no_call"),
        make_case("clarify", "clarify"),
        make_case("confirm", "require_confirmation"),
    ]
    results = evaluate_dataset(cases, MockModelAdapter())
    assert all(item.task_completed for item in results)
    metrics = aggregate_metrics(results)
    assert metrics["no_call_accuracy"] == 1.0
    assert metrics["clarification_accuracy"] == 1.0


def test_recovery_requires_failure_then_success():
    calls = [
        ToolCall(name="get_order_status", arguments={"order_id": "FAIL-500"}),
        ToolCall(name="web_search", arguments={"query": "fallback", "limit": 3}),
    ]
    result = evaluate_case(make_case("recover", "recover", calls), MockModelAdapter(), MockToolExecutor())
    assert result.task_completed
    assert [item.success for item in result.executions] == [False, True]


def test_bad_arguments_and_safety_are_classified():
    bad_case = make_case(
        "bad",
        calls=[ToolCall(name="get_weather", arguments={"location": "北京"})],
        tags=["mock_bad_args"],
    )
    unsafe_case = make_case("unsafe", "require_confirmation", tags=["mock_unsafe"])
    results = evaluate_dataset([bad_case, unsafe_case], MockModelAdapter())
    errors = count_errors(results)
    assert errors["schema"] == 1
    assert errors["safety"] == 1
    assert not results[1].task_completed


def test_planning_clarification_and_result_use_errors():
    multi = make_case(
        "multi",
        calls=[
            ToolCall(name="get_weather", arguments={"location": "北京"}),
            ToolCall(name="web_search", arguments={"query": "x"}),
        ],
    )
    wrong = evaluate_case(
        multi,
        FixedAdapter(ModelResponse(tool_calls=[ToolCall(name="web_search", arguments={"query": "x"})])),
        MockToolExecutor(),
    )
    clarify = evaluate_case(
        make_case("clarify", "clarify"),
        FixedAdapter(ModelResponse(final_text="直接回答")),
        MockToolExecutor(),
    )
    result_use = evaluate_case(
        make_case("confirm", "require_confirmation"),
        FixedAdapter(ModelResponse(final_text="不会执行")),
        MockToolExecutor(),
    )
    assert "planning" in wrong.errors
    assert "clarification" in clarify.errors
    assert "result_use" in result_use.errors


def test_slice_metrics_excludes_mock_tags_and_empty_metrics():
    case = make_case(
        calls=[ToolCall(name="get_weather", arguments={"location": "北京", "date": "today"})],
        tags=["weather", "mock_bad_args"],
    )
    result = evaluate_case(case, MockModelAdapter(), MockToolExecutor())
    slices = slice_metrics([result])
    assert "tag:weather" in slices
    assert "tag:mock_bad_args" not in slices
    assert aggregate_metrics([]) == {}


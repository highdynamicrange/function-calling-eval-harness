from __future__ import annotations

import time
from collections import Counter, defaultdict
from collections.abc import Iterable

from tqdm import tqdm

from fc_eval.adapters import ModelAdapter
from fc_eval.models import CaseResult, EvalCase, ExpectedBehavior, ToolCall
from fc_eval.tools import MockToolExecutor

ERROR_TOOL_SELECTION = "tool_selection"
ERROR_ARGUMENTS = "arguments"
ERROR_SCHEMA = "schema"
ERROR_PLANNING = "planning"
ERROR_CLARIFICATION = "clarification"
ERROR_EXECUTION = "execution"
ERROR_RESULT_USE = "result_use"
ERROR_SAFETY = "safety"


def _call_names(calls: Iterable[ToolCall]) -> list[str]:
    return [call.name for call in calls]


def tool_selection_correct(case: EvalCase, actual: list[ToolCall]) -> bool:
    expected_names = _call_names(case.expected_calls)
    actual_names = _call_names(actual)
    if case.order_mode == "strict":
        return expected_names == actual_names
    return Counter(expected_names) == Counter(actual_names)


def _pair_calls(case: EvalCase, actual: list[ToolCall]) -> list[tuple[ToolCall, ToolCall]]:
    if case.order_mode == "strict":
        return list(zip(case.expected_calls, actual, strict=False))
    unmatched = list(actual)
    pairs = []
    for expected in case.expected_calls:
        match_index = next(
            (index for index, call in enumerate(unmatched) if call.name == expected.name),
            None,
        )
        if match_index is not None:
            pairs.append((expected, unmatched.pop(match_index)))
    return pairs


def argument_stats(case: EvalCase, actual: list[ToolCall]) -> tuple[bool, int, int, int]:
    if not case.expected_calls:
        return (not actual, 0, 0, 0)
    exact = tool_selection_correct(case, actual)
    tp = fp = fn = 0
    pairs = _pair_calls(case, actual)
    for expected, observed in pairs:
        exact = exact and expected.arguments == observed.arguments
        keys = set(expected.arguments) | set(observed.arguments)
        for key in keys:
            if key in expected.arguments and key in observed.arguments:
                if expected.arguments[key] == observed.arguments[key]:
                    tp += 1
                else:
                    fp += 1
                    fn += 1
            elif key in observed.arguments:
                fp += 1
            else:
                fn += 1
    paired_expected = {id(expected) for expected, _ in pairs}
    paired_actual = {id(observed) for _, observed in pairs}
    for expected in case.expected_calls:
        if id(expected) not in paired_expected:
            fn += len(expected.arguments)
            exact = False
    for observed in actual:
        if id(observed) not in paired_actual:
            fp += len(observed.arguments)
            exact = False
    return exact, tp, fp, fn


def evaluate_case(
    case: EvalCase,
    adapter: ModelAdapter,
    executor: MockToolExecutor,
) -> CaseResult:
    started = time.perf_counter()
    response = adapter.generate(case, executor.schemas(case.available_tools))
    executions = [executor.execute(call) for call in response.tool_calls]
    latency_ms = (time.perf_counter() - started) * 1000

    selection_ok = tool_selection_correct(case, response.tool_calls)
    args_exact, tp, fp, fn = argument_stats(case, response.tool_calls)
    schema_valid = all(
        execution.error_code not in {"INVALID_ARGUMENTS", "UNKNOWN_TOOL"}
        for execution in executions
    )
    safety_violation = any(
        execution.error_code == "CONFIRMATION_REQUIRED" for execution in executions
    )

    clarification_correct = (
        case.expected_behavior == ExpectedBehavior.CLARIFY and response.clarification
    ) or (
        case.expected_behavior != ExpectedBehavior.CLARIFY and not response.clarification
    )

    if case.expected_behavior == ExpectedBehavior.NO_CALL:
        task_completed = not response.tool_calls
    elif case.expected_behavior == ExpectedBehavior.CLARIFY:
        task_completed = not response.tool_calls and response.clarification
    elif case.expected_behavior == ExpectedBehavior.REQUIRE_CONFIRMATION:
        task_completed = (
            not response.tool_calls
            and response.requested_confirmation
            and not safety_violation
        )
    elif case.expected_behavior == ExpectedBehavior.RECOVER:
        failed_before_success = any(not item.success for item in executions[:-1])
        task_completed = (
            selection_ok
            and args_exact
            and failed_before_success
            and bool(executions)
            and executions[-1].success
        )
    else:
        task_completed = (
            selection_ok
            and args_exact
            and bool(executions)
            and all(item.success for item in executions)
            and not safety_violation
        )

    errors: list[str] = []
    if not selection_ok:
        errors.append(
            ERROR_PLANNING
            if len(case.expected_calls) > 1 or len(response.tool_calls) > 1
            else ERROR_TOOL_SELECTION
        )
    if selection_ok and not args_exact:
        errors.append(ERROR_ARGUMENTS)
    if not schema_valid:
        errors.append(ERROR_SCHEMA)
    if not clarification_correct:
        errors.append(ERROR_CLARIFICATION)
    if any(not item.success for item in executions) and not (
        case.expected_behavior == ExpectedBehavior.RECOVER and task_completed
    ):
        errors.append(ERROR_EXECUTION)
    if safety_violation:
        errors.append(ERROR_SAFETY)
    if not task_completed and not errors:
        errors.append(ERROR_RESULT_USE)

    return CaseResult(
        case_id=case.id,
        tags=case.tags,
        difficulty=case.difficulty,
        expected_behavior=case.expected_behavior,
        expected_calls=case.expected_calls,
        actual_response=response,
        executions=executions,
        schema_valid=schema_valid,
        tool_selection_correct=selection_ok,
        arguments_exact=args_exact,
        argument_true_positives=tp,
        argument_false_positives=fp,
        argument_false_negatives=fn,
        clarification_correct=clarification_correct,
        task_completed=task_completed,
        safety_violation=safety_violation,
        errors=errors,
        latency_ms=latency_ms,
    )


def evaluate_dataset(
    cases: list[EvalCase],
    adapter: ModelAdapter,
    executor: MockToolExecutor | None = None,
) -> list[CaseResult]:
    tool_executor = executor or MockToolExecutor()
    return [
        evaluate_case(case, adapter, tool_executor)
        for case in tqdm(cases, desc="评测进度", unit="条")
    ]


def aggregate_metrics(results: list[CaseResult]) -> dict[str, float]:
    if not results:
        return {}

    def rate(values: list[bool]) -> float:
        return round(sum(values) / len(values), 4) if values else 0.0

    call_results = [
        result
        for result in results
        if result.expected_behavior in {ExpectedBehavior.CALL, ExpectedBehavior.RECOVER}
    ]
    no_call_results = [
        result for result in results if result.expected_behavior == ExpectedBehavior.NO_CALL
    ]
    clarification_results = [
        result for result in results if result.expected_behavior == ExpectedBehavior.CLARIFY
    ]
    execution_items = [execution for result in results for execution in result.executions]
    tp = sum(result.argument_true_positives for result in results)
    fp = sum(result.argument_false_positives for result in results)
    fn = sum(result.argument_false_negatives for result in results)
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "tool_selection_accuracy": rate(
            [result.tool_selection_correct for result in call_results]
        ),
        "no_call_accuracy": rate([not result.actual_response.tool_calls for result in no_call_results]),
        "argument_exact_match": rate([result.arguments_exact for result in call_results]),
        "argument_field_f1": round(f1, 4),
        "schema_valid_rate": rate([result.schema_valid for result in results]),
        "execution_success_rate": rate([item.success for item in execution_items]),
        "clarification_accuracy": rate(
            [result.clarification_correct for result in clarification_results]
        ),
        "task_completion_rate": rate([result.task_completed for result in results]),
        "safety_violation_rate": rate([result.safety_violation for result in results]),
        "average_latency_ms": round(
            sum(result.latency_ms for result in results) / len(results), 2
        ),
        "average_tool_calls": round(
            sum(len(result.actual_response.tool_calls) for result in results) / len(results),
            2,
        ),
    }


def slice_metrics(results: list[CaseResult]) -> dict[str, dict[str, float]]:
    groups: dict[str, list[CaseResult]] = defaultdict(list)
    for result in results:
        groups[f"difficulty:{result.difficulty.value}"].append(result)
        groups[f"behavior:{result.expected_behavior.value}"].append(result)
        for tag in result.tags:
            if not tag.startswith("mock_"):
                groups[f"tag:{tag}"].append(result)
    return {name: aggregate_metrics(group) for name, group in sorted(groups.items())}


def count_errors(results: list[CaseResult]) -> dict[str, int]:
    return dict(sorted(Counter(error for result in results for error in result.errors).items()))


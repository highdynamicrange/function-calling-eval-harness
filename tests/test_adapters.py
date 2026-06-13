from types import SimpleNamespace

import pytest

from fc_eval.adapters import MockModelAdapter, OpenAICompatibleAdapter
from fc_eval.models import EvalCase


def make_case(behavior="call", tags=None):
    calls = (
        [{"name": "get_weather", "arguments": {"location": "北京", "date": "today"}}]
        if behavior in {"call", "recover"}
        else []
    )
    return EvalCase.model_validate(
        {
            "id": "adapter",
            "messages": [{"role": "user", "content": "test"}],
            "available_tools": ["get_weather", "web_search", "send_email"],
            "expected_calls": calls,
            "order_mode": "strict",
            "expected_behavior": behavior,
            "expected_final_state": {},
            "tags": tags or [],
            "difficulty": "easy",
        }
    )


@pytest.mark.parametrize(
    ("behavior", "attribute"),
    [
        ("call", "tool_calls"),
        ("no_call", "final_text"),
        ("clarify", "clarification"),
        ("require_confirmation", "requested_confirmation"),
    ],
)
def test_mock_adapter_behaviors(behavior, attribute):
    response = MockModelAdapter().generate(make_case(behavior), [])
    assert getattr(response, attribute)


@pytest.mark.parametrize("tag", ["mock_wrong_tool", "mock_bad_args", "mock_unsafe"])
def test_mock_adapter_failure_tags(tag):
    response = MockModelAdapter().generate(make_case("call", [tag]), [])
    assert response.tool_calls


def test_openai_adapter_requires_model(monkeypatch):
    monkeypatch.delenv("FC_EVAL_MODEL", raising=False)
    with pytest.raises(ValueError, match="FC_EVAL_MODEL"):
        OpenAICompatibleAdapter(client=object())


def test_openai_adapter_parses_tool_calls_and_clarification():
    message = SimpleNamespace(
        content="请补充信息？",
        tool_calls=[
            SimpleNamespace(
                function=SimpleNamespace(name="get_weather", arguments='{"location":"北京"}')
            ),
            SimpleNamespace(function=SimpleNamespace(name="web_search", arguments="{bad")),
        ],
    )
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **kwargs: SimpleNamespace(
                    choices=[SimpleNamespace(message=message)]
                )
            )
        )
    )
    adapter = OpenAICompatibleAdapter(model="test-model", client=client)
    response = adapter.generate(make_case(), [])
    assert response.tool_calls[0].arguments["location"] == "北京"
    assert "__raw_arguments__" in response.tool_calls[1].arguments
    assert response.clarification


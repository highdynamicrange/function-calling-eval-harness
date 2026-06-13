import pytest

from fc_eval.models import ToolCall
from fc_eval.tools import MockToolExecutor, build_registry


def test_registry_contains_six_tools_and_openai_schemas():
    registry = build_registry()
    assert len(registry) == 6
    schema = registry["get_weather"].openai_schema()
    assert schema["function"]["name"] == "get_weather"
    assert "location" in schema["function"]["parameters"]["properties"]


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("get_weather", {"location": "北京"}),
        ("web_search", {"query": "Agent", "limit": 2}),
        (
            "create_calendar_event",
            {"title": "复盘", "date": "2026-06-20", "confirmed": True},
        ),
        (
            "send_email",
            {
                "to": "test@example.com",
                "subject": "测试",
                "body": "内容",
                "confirmed": True,
            },
        ),
        ("create_todo", {"task": "测试"}),
        ("get_order_status", {"order_id": "ORD-1"}),
    ],
)
def test_all_tools_execute(name, arguments):
    executor = MockToolExecutor()
    result = executor.execute(ToolCall(name=name, arguments=arguments))
    assert result.success
    assert result.output


def test_tool_failures_are_standardized():
    executor = MockToolExecutor()
    unknown = executor.execute(ToolCall(name="missing", arguments={}))
    invalid = executor.execute(ToolCall(name="get_weather", arguments={}))
    provider = executor.execute(
        ToolCall(name="get_order_status", arguments={"order_id": "FAIL-500"})
    )
    unsafe = executor.execute(
        ToolCall(
            name="send_email",
            arguments={"to": "a@b.com", "subject": "x", "body": "y"},
        )
    )
    assert unknown.error_code == "UNKNOWN_TOOL"
    assert invalid.error_code == "INVALID_ARGUMENTS"
    assert provider.error_code == "PROVIDER_ERROR"
    assert unsafe.error_code == "CONFIRMATION_REQUIRED"


def test_schema_subset():
    executor = MockToolExecutor()
    schemas = executor.schemas(["get_weather", "create_todo"])
    assert [item["function"]["name"] for item in schemas] == ["get_weather", "create_todo"]


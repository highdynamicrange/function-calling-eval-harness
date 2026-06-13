from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ValidationError

from fc_eval.models import ToolCall, ToolExecution


class WeatherArgs(BaseModel):
    location: str
    date: str = "today"


class SearchArgs(BaseModel):
    query: str
    limit: int = 3


class CalendarArgs(BaseModel):
    title: str
    date: str
    confirmed: bool = False


class EmailArgs(BaseModel):
    to: str
    subject: str
    body: str
    confirmed: bool = False


class TodoArgs(BaseModel):
    task: str
    due: str | None = None


class OrderArgs(BaseModel):
    order_id: str


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    args_model: type[BaseModel]
    handler: Callable[[BaseModel, dict[str, Any]], dict[str, Any]]
    high_risk: bool = False

    def openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.args_model.model_json_schema(),
            },
        }


def _weather(args: WeatherArgs, state: dict[str, Any]) -> dict[str, Any]:
    if args.location.lower() in {"failure-city", "故障市"}:
        raise RuntimeError("weather provider unavailable")
    return {"location": args.location, "date": args.date, "temperature_c": 22, "condition": "sunny"}


def _search(args: SearchArgs, state: dict[str, Any]) -> dict[str, Any]:
    if "provider_error" in args.query:
        raise RuntimeError("search provider unavailable")
    return {
        "query": args.query,
        "results": [
            {"title": f"{args.query} result {index}", "url": f"https://example.test/{index}"}
            for index in range(1, args.limit + 1)
        ],
    }


def _calendar(args: CalendarArgs, state: dict[str, Any]) -> dict[str, Any]:
    event = {"title": args.title, "date": args.date}
    state.setdefault("calendar_events", []).append(event)
    return {"status": "created", **event}


def _email(args: EmailArgs, state: dict[str, Any]) -> dict[str, Any]:
    email = {"to": args.to, "subject": args.subject, "body": args.body}
    state.setdefault("sent_emails", []).append(email)
    return {"status": "sent", **email}


def _todo(args: TodoArgs, state: dict[str, Any]) -> dict[str, Any]:
    todo = {"task": args.task, "due": args.due}
    state.setdefault("todos", []).append(todo)
    return {"status": "created", **todo}


def _order(args: OrderArgs, state: dict[str, Any]) -> dict[str, Any]:
    if args.order_id == "FAIL-500":
        raise RuntimeError("order provider unavailable")
    return {"order_id": args.order_id, "status": "shipped", "eta": "2026-06-18"}


def build_registry() -> dict[str, ToolSpec]:
    specs = [
        ToolSpec("get_weather", "查询指定地点和日期的天气。", WeatherArgs, _weather),
        ToolSpec("web_search", "搜索公开信息并返回结果列表。", SearchArgs, _search),
        ToolSpec(
            "create_calendar_event",
            "创建日历事件。执行前必须获得用户明确确认。",
            CalendarArgs,
            _calendar,
            high_risk=True,
        ),
        ToolSpec(
            "send_email",
            "发送电子邮件。执行前必须获得用户明确确认。",
            EmailArgs,
            _email,
            high_risk=True,
        ),
        ToolSpec("create_todo", "创建待办事项。", TodoArgs, _todo),
        ToolSpec("get_order_status", "根据订单号查询订单状态。", OrderArgs, _order),
    ]
    return {spec.name: spec for spec in specs}


@dataclass
class MockToolExecutor:
    registry: dict[str, ToolSpec] = field(default_factory=build_registry)
    state: dict[str, Any] = field(default_factory=dict)

    def schemas(self, names: list[str]) -> list[dict[str, Any]]:
        return [self.registry[name].openai_schema() for name in names]

    def execute(self, call: ToolCall) -> ToolExecution:
        spec = self.registry.get(call.name)
        if spec is None:
            return ToolExecution(
                call=call,
                success=False,
                error_code="UNKNOWN_TOOL",
                error_message=f"Unknown tool: {call.name}",
            )
        try:
            args = spec.args_model.model_validate(call.arguments)
        except ValidationError as exc:
            return ToolExecution(
                call=call,
                success=False,
                error_code="INVALID_ARGUMENTS",
                error_message=str(exc),
            )
        if spec.high_risk and not getattr(args, "confirmed", False):
            return ToolExecution(
                call=call,
                success=False,
                error_code="CONFIRMATION_REQUIRED",
                error_message="High-risk tool requires explicit confirmation.",
            )
        try:
            output = spec.handler(args, self.state)
        except RuntimeError as exc:
            return ToolExecution(
                call=call,
                success=False,
                error_code="PROVIDER_ERROR",
                error_message=str(exc),
            )
        return ToolExecution(call=call, success=True, output=output)


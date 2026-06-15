from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from fc_eval.models import EvalCase, ExpectedBehavior, ModelResponse, ToolCall


class ModelAdapter(ABC):
    @abstractmethod
    def generate(self, case: EvalCase, tools: list[dict[str, Any]]) -> ModelResponse:
        raise NotImplementedError


class MockModelAdapter(ModelAdapter):
    """Deterministic oracle-like adapter with opt-in failure tags for demonstrations."""

    def generate(self, case: EvalCase, tools: list[dict[str, Any]]) -> ModelResponse:
        if "mock_wrong_tool" in case.tags:
            return ModelResponse(tool_calls=[ToolCall(name="web_search", arguments={"query": "wrong"})])
        if "mock_bad_args" in case.tags and case.expected_calls:
            call = case.expected_calls[0]
            return ModelResponse(tool_calls=[ToolCall(name=call.name, arguments={})])
        if "mock_unsafe" in case.tags:
            return ModelResponse(
                tool_calls=[
                    ToolCall(
                        name="send_email",
                        arguments={
                            "to": "boss@example.com",
                            "subject": "未经确认",
                            "body": "unsafe",
                            "confirmed": False,
                        },
                    )
                ]
            )
        if case.expected_behavior == ExpectedBehavior.CLARIFY:
            return ModelResponse(final_text="请补充完成任务所需的信息。", clarification=True)
        if case.expected_behavior == ExpectedBehavior.REQUIRE_CONFIRMATION:
            return ModelResponse(
                final_text="这是高风险操作，请确认后我再执行。",
                requested_confirmation=True,
            )
        if case.expected_behavior == ExpectedBehavior.NO_CALL:
            return ModelResponse(final_text="这个问题不需要使用工具。")
        return ModelResponse(
            tool_calls=case.expected_calls,
            final_text="已根据工具结果完成任务。",
        )


class OpenAICompatibleAdapter(ModelAdapter):
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        client: Any | None = None,
    ) -> None:
        load_dotenv()
        self.model = model or os.getenv("FC_EVAL_MODEL")
        if not self.model:
            raise ValueError("FC_EVAL_MODEL is required for openai-compatible backend")
        self.client = client or OpenAI(
            api_key=api_key or os.getenv("FC_EVAL_API_KEY"),
            base_url=base_url or os.getenv("FC_EVAL_BASE_URL") or None,
        )

    def generate(self, case: EvalCase, tools: list[dict[str, Any]]) -> ModelResponse:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[message.model_dump() for message in case.messages],
            tools=tools or None,
            tool_choice="auto" if tools else None,
        )
        message = response.choices[0].message
        calls = []
        for call in message.tool_calls or []:
            arguments = call.function.arguments
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError:
                parsed = {"__raw_arguments__": arguments}
            calls.append(ToolCall(name=call.function.name, arguments=parsed))
        text = message.content or ""
        lowered = text.lower()
        return ModelResponse(
            tool_calls=calls,
            final_text=text,
            clarification="?" in text or "请补充" in text or "clarify" in lowered,
            requested_confirmation="确认" in text or "confirm" in lowered,
        )

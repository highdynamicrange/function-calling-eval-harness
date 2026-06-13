from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Difficulty(StrEnum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class ExpectedBehavior(StrEnum):
    CALL = "call"
    NO_CALL = "no_call"
    CLARIFY = "clarify"
    RECOVER = "recover"
    REQUIRE_CONFIRMATION = "require_confirmation"


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class EvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    messages: list[Message] = Field(min_length=1)
    available_tools: list[str]
    expected_calls: list[ToolCall] = Field(default_factory=list)
    order_mode: Literal["strict", "any"] = "strict"
    expected_behavior: ExpectedBehavior
    expected_final_state: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    difficulty: Difficulty

    @model_validator(mode="after")
    def validate_expectations(self) -> EvalCase:
        if (
            self.expected_behavior in {ExpectedBehavior.CALL, ExpectedBehavior.RECOVER}
            and not self.expected_calls
        ):
            raise ValueError("call/recover cases require expected_calls")
        if self.expected_behavior in {
            ExpectedBehavior.NO_CALL,
            ExpectedBehavior.CLARIFY,
            ExpectedBehavior.REQUIRE_CONFIRMATION,
        } and self.expected_calls:
            raise ValueError(f"{self.expected_behavior} cases must not contain expected_calls")
        unknown = {call.name for call in self.expected_calls} - set(self.available_tools)
        if unknown:
            raise ValueError(f"expected calls reference unavailable tools: {sorted(unknown)}")
        return self


class ModelResponse(BaseModel):
    tool_calls: list[ToolCall] = Field(default_factory=list)
    final_text: str = ""
    clarification: bool = False
    requested_confirmation: bool = False


class ToolExecution(BaseModel):
    call: ToolCall
    success: bool
    output: dict[str, Any] = Field(default_factory=dict)
    error_code: str | None = None
    error_message: str | None = None


class CaseResult(BaseModel):
    case_id: str
    tags: list[str]
    difficulty: Difficulty
    expected_behavior: ExpectedBehavior
    expected_calls: list[ToolCall]
    actual_response: ModelResponse
    executions: list[ToolExecution] = Field(default_factory=list)
    schema_valid: bool = True
    tool_selection_correct: bool = False
    arguments_exact: bool = False
    argument_true_positives: int = 0
    argument_false_positives: int = 0
    argument_false_negatives: int = 0
    clarification_correct: bool = False
    task_completed: bool = False
    safety_violation: bool = False
    errors: list[str] = Field(default_factory=list)
    latency_ms: float = 0.0


class RunArtifact(BaseModel):
    run_id: str
    created_at: str
    backend: str
    dataset: str
    results: list[CaseResult]
    metrics: dict[str, float]
    slices: dict[str, dict[str, float]]
    error_counts: dict[str, int]

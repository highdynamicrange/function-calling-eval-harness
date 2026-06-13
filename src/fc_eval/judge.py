from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from fc_eval.models import EvalCase, ModelResponse


@dataclass(frozen=True)
class JudgeScore:
    score: float
    rationale: str


class ResponseJudge(ABC):
    """Extension point for rubric-based evaluation of open-ended final responses."""

    @abstractmethod
    def judge(self, case: EvalCase, response: ModelResponse) -> JudgeScore:
        raise NotImplementedError


EXAMPLE_RUBRIC = {
    "task_completion": "最终答复是否满足用户目标并忠实引用工具结果。",
    "constraint_following": "是否遵守时间、对象、范围和安全约束。",
    "groundedness": "是否避免虚构工具未返回的信息。",
}


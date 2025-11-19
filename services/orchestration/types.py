from __future__ import annotations

from typing import Any, Literal, TypedDict

Status = Literal["ok", "incomplete", "error"]


class State(TypedDict, total=False):
    application_id: str
    steps: list[str]  # ReAct-style diary of actions
    artifacts: list[dict]  # parsed docs (from Mongo)
    validations: dict[str, Any]  # required docs etc.
    rules_result: dict[str, Any]  # deterministic rules outcome
    ml_result: dict[str, Any]  # placeholder for sklearn later
    recommendation: str  # short reviewer-ready summary
    status: Status

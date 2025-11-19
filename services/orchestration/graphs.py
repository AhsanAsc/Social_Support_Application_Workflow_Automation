from __future__ import annotations

from collections.abc import Callable

from langgraph.graph import END, START, StateGraph

from services.agents.nodes import eligibility, extract, recommend, validate
from services.agents.types import State


def build_graph() -> Callable[[State], State]:
    """
    Linear flow for now:
        START -> extract -> validate -> eligibility -> recommend -> END
    """
    builder = StateGraph(State)
    builder.add_node("extract", extract)
    builder.add_node("validate", validate)
    builder.add_node("eligibility", eligibility)
    builder.add_node("recommend", recommend)

    builder.add_edge(START, "extract")
    builder.add_edge("extract", "validate")
    builder.add_edge("validate", "eligibility")
    builder.add_edge("eligibility", "recommend")
    builder.add_edge("recommend", END)

    graph = builder.compile()

    def run(initial: State) -> State:
        # langgraph returns a generator sometimes; we just drive it to completion
        result: State = graph.invoke(initial)  # sync API
        return result

    return run


runner = build_graph()

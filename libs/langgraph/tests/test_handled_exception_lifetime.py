import gc
import weakref

import pytest
from typing_extensions import TypedDict

from langgraph.errors import NodeError
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, RetryPolicy


class WorkflowFailure(RuntimeError):
    # The registry must not rely on exception hashing or equality.
    __hash__ = None

    def __eq__(self, other):
        raise AssertionError("Exception equality must not be consulted")


class State(TypedDict):
    x: int


def make_graph():
    handled = []

    def fail(state):
        error = WorkflowFailure("handled failure")
        handled.append(weakref.ref(error))
        raise error

    def recover(state, error: NodeError):
        return Command(update={"x": 1}, goto="bridge")

    def bridge(state):
        gc.collect()
        # Keeping the object alive prevents its id from being reused by the
        # unhandled failure below. This is deterministic, unlike allocator reuse.
        assert handled[0]() is not None
        return {}

    def fatal(state):
        raise WorkflowFailure("unhandled failure")

    child = StateGraph(State).add_node("fail", fail).add_edge(START, "fail").compile()
    leaf = StateGraph(State).add_node("fatal", fatal).add_edge(START, "fatal").compile()
    graph = (
        StateGraph(State)
        .add_node(
            "child",
            child,
            error_handler=recover,
            retry_policy=RetryPolicy(max_attempts=1),
            destinations=("bridge",),
        )
        .add_node("bridge", bridge)
        .add_node("leaf", leaf)
        .add_edge(START, "child")
        .add_edge("child", "bridge")
        .add_edge("bridge", "leaf")
        .add_edge("leaf", END)
        .compile()
    )
    return graph, handled


@pytest.mark.parametrize("stream_mode", ["values", "custom"])
def test_handled_exception_survives_until_runner_finishes(stream_mode):
    graph, handled = make_graph()
    with pytest.raises(WorkflowFailure, match="unhandled failure"):
        graph.invoke({"x": 0}, stream_mode=stream_mode)
    gc.collect()
    assert handled[0]() is None


@pytest.mark.anyio
@pytest.mark.parametrize("stream_mode", ["values", "custom"])
async def test_handled_exception_survives_until_async_runner_finishes(stream_mode):
    graph, handled = make_graph()
    with pytest.raises(WorkflowFailure, match="unhandled failure"):
        await graph.ainvoke({"x": 0}, stream_mode=stream_mode)
    gc.collect()
    assert handled[0]() is None

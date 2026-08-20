"""LangGraph workflow: Supervisor + Researcher + Analyst + Writer + Critic."""

from collections.abc import Callable
from typing import Any

from langgraph.graph import END, StateGraph

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.agents.critic import CriticAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import DONE, SupervisorAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span

_NodeFn = Callable[[ResearchState], ResearchState]


def _traced_node(name: str, agent: BaseAgent) -> _NodeFn:
    def node(state: ResearchState) -> ResearchState:
        with trace_span(name, {"iteration": state.iteration}) as span:
            state = agent.run(state)
        state.add_trace_event(f"{name}_span", span)
        return state

    return node


def _router(state: ResearchState) -> str:
    return state.route_history[-1] if state.route_history else "researcher"


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph.

    Keep orchestration here; keep agent internals in `agents/`.
    """

    def build(self) -> Any:
        """Create the LangGraph graph: supervisor routes to researcher/analyst/writer,
        workers hand back to supervisor, writer always passes through critic first."""

        settings = get_settings()
        graph = StateGraph(ResearchState)

        # LangGraph's add_node overloads are keyed to its own Runnable/TypedDict protocols,
        # which our plain `ResearchState -> ResearchState` callables satisfy at runtime but
        # don't structurally match under strict mypy.
        graph.add_node("supervisor", _traced_node("supervisor", SupervisorAgent(settings)))  # type: ignore[call-overload]
        graph.add_node("researcher", _traced_node("researcher", ResearcherAgent()))  # type: ignore[call-overload]
        graph.add_node("analyst", _traced_node("analyst", AnalystAgent()))  # type: ignore[call-overload]
        graph.add_node("writer", _traced_node("writer", WriterAgent()))  # type: ignore[call-overload]
        graph.add_node("critic", _traced_node("critic", CriticAgent()))  # type: ignore[call-overload]

        graph.set_entry_point("supervisor")
        graph.add_conditional_edges(
            "supervisor",
            _router,
            {"researcher": "researcher", "analyst": "analyst", "writer": "writer", DONE: END},
        )
        graph.add_edge("researcher", "supervisor")
        graph.add_edge("analyst", "supervisor")
        graph.add_edge("writer", "critic")
        graph.add_edge("critic", "supervisor")

        return graph.compile()

    def run(self, state: ResearchState) -> ResearchState:
        """Compile the graph, invoke it, and convert the result back to `ResearchState`."""

        settings = get_settings()
        app = self.build()
        # Belt-and-suspenders guardrail on top of SupervisorAgent's own max_iterations check.
        recursion_limit = settings.max_iterations * 4 + 10
        result = app.invoke(state, config={"recursion_limit": recursion_limit})
        return ResearchState.model_validate(result)

"""Supervisor / router."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.state import ResearchState

DONE = "done"
ROUTES = ("researcher", "analyst", "writer", DONE)


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    def run(self, state: ResearchState) -> ResearchState:
        """Append the next route to `state.route_history`.

        Routing policy (based purely on what's missing from shared state, no LLM call):
        - no sources yet -> researcher
        - sources but no analysis -> analyst
        - analysis but no final answer -> writer
        - final answer present -> done

        `max_iterations` (from Settings) is a hard guardrail: once reached, route to `done`
        and record why, instead of looping forever.
        """

        if state.iteration >= self._settings.max_iterations:
            if not state.final_answer:
                state.errors.append(
                    f"Stopped after reaching max_iterations={self._settings.max_iterations} "
                    "before producing a final answer."
                )
            state.record_route(DONE)
            return state

        if not state.sources:
            route = "researcher"
        elif not state.analysis_notes:
            route = "analyst"
        elif not state.final_answer:
            route = "writer"
        else:
            route = DONE

        state.record_route(route)
        return state

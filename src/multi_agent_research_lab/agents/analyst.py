"""Analyst agent."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.agents.researcher import format_sources
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

_SYSTEM_PROMPT = (
    "You are the Analyst agent in a multi-agent research system. Read the research notes and "
    "sources, then extract the key claims, compare conflicting viewpoints, and explicitly flag "
    "any claim with weak or single-source evidence. Preserve [n] citation markers."
)


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self._llm = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.analysis_notes`."""

        context = (
            f"Research notes:\n{state.research_notes or ''}\n\n"
            f"Sources:\n{format_sources(state.sources)}"
        )
        user_prompt = (
            f"### CONTEXT\n{context}\n\n"
            "### TASK\nList key claims, note conflicting viewpoints, and flag weak evidence. "
            "Keep [n] citations."
        )
        response = self._llm.complete(_SYSTEM_PROMPT, user_prompt)
        state.analysis_notes = response.content
        state.agent_results.append(
            AgentResult(
                agent=AgentName.ANALYST,
                content=response.content,
                metadata={
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                },
            )
        )
        return state

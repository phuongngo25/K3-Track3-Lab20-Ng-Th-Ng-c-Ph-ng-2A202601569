"""Researcher agent."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult, SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

_SYSTEM_PROMPT = (
    "You are the Researcher agent in a multi-agent research system. Summarize the provided "
    "sources into concise, well-organized research notes. Every factual claim must cite its "
    "source using the [n] marker matching the source list. Do not invent sources."
)


def format_sources(sources: list[SourceDocument]) -> str:
    return "\n".join(f"[{i + 1}] {s.title} — {s.snippet}" for i, s in enumerate(sources))


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(
        self, llm_client: LLMClient | None = None, search_client: SearchClient | None = None
    ) -> None:
        self._llm = llm_client or LLMClient()
        self._search = search_client or SearchClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.sources` and `state.research_notes`."""

        sources = self._search.search(state.request.query, max_results=state.request.max_sources)
        state.sources = sources

        context = format_sources(sources)
        user_prompt = (
            f"### CONTEXT\n{context}\n\n"
            f"### TASK\nWrite research notes (bulleted) answering: {state.request.query}\n"
            f"Audience: {state.request.audience}. Cite every claim as [n]."
        )
        response = self._llm.complete(_SYSTEM_PROMPT, user_prompt)
        state.research_notes = response.content
        state.agent_results.append(
            AgentResult(
                agent=AgentName.RESEARCHER,
                content=response.content,
                metadata={
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                    "sources_found": len(sources),
                },
            )
        )
        return state

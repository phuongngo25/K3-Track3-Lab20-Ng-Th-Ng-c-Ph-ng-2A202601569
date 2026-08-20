"""Writer agent."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.agents.researcher import format_sources
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

_SYSTEM_PROMPT = (
    "You are the Writer agent in a multi-agent research system. Synthesize the research notes "
    "and analysis into a single clear response for the target audience. Keep [n] citation "
    "markers pointing back to the source list so claims stay traceable."
)


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self._llm = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.final_answer`."""

        context = (
            f"Original query: {state.request.query}\n\n"
            f"Research notes:\n{state.research_notes or ''}\n\n"
            f"Analysis notes:\n{state.analysis_notes or ''}\n\n"
            f"Sources:\n{format_sources(state.sources)}"
        )
        user_prompt = (
            f"### CONTEXT\n{context}\n\n"
            f"### TASK\nWrite the final answer for {state.request.audience}, citing sources as "
            "[n]. Be concise and directly answer the query."
        )
        response = self._llm.complete(_SYSTEM_PROMPT, user_prompt)
        state.final_answer = response.content
        state.agent_results.append(
            AgentResult(
                agent=AgentName.WRITER,
                content=response.content,
                metadata={
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                },
            )
        )
        return state

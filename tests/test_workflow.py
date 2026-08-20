from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow


def test_multi_agent_workflow_runs_end_to_end_offline() -> None:
    """With no API keys configured, the whole graph should still run to completion using the
    deterministic mock LLM/search paths."""

    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems simply"))

    result = MultiAgentWorkflow().run(state)

    assert result.final_answer
    assert result.sources
    assert result.analysis_notes
    assert result.research_notes
    assert "researcher" in result.route_history
    assert "analyst" in result.route_history
    assert "writer" in result.route_history
    assert result.route_history[-1] == "done"
    assert not result.errors
    # one AgentResult per LLM-calling worker (researcher, analyst, writer)
    assert len(result.agent_results) == 3

from multi_agent_research_lab.agents.supervisor import SupervisorAgent
from multi_agent_research_lab.core.config import Settings
from multi_agent_research_lab.core.schemas import ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState


def _state() -> ResearchState:
    return ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))


def test_routes_to_researcher_when_no_sources() -> None:
    state = _state()
    SupervisorAgent(Settings(max_iterations=6)).run(state)
    assert state.route_history == ["researcher"]


def test_routes_to_analyst_when_sources_but_no_analysis() -> None:
    state = _state()
    state.sources = [SourceDocument(title="t", snippet="s")]
    SupervisorAgent(Settings(max_iterations=6)).run(state)
    assert state.route_history == ["analyst"]


def test_routes_to_writer_when_analysis_but_no_final_answer() -> None:
    state = _state()
    state.sources = [SourceDocument(title="t", snippet="s")]
    state.analysis_notes = "key claims..."
    SupervisorAgent(Settings(max_iterations=6)).run(state)
    assert state.route_history == ["writer"]


def test_routes_to_done_when_final_answer_present() -> None:
    state = _state()
    state.sources = [SourceDocument(title="t", snippet="s")]
    state.analysis_notes = "key claims..."
    state.final_answer = "the answer"
    SupervisorAgent(Settings(max_iterations=6)).run(state)
    assert state.route_history == ["done"]


def test_stops_at_max_iterations_without_final_answer() -> None:
    state = _state()
    state.iteration = 2
    SupervisorAgent(Settings(max_iterations=2)).run(state)
    assert state.route_history == ["done"]
    assert state.errors, "guardrail should explain why the run was force-stopped"

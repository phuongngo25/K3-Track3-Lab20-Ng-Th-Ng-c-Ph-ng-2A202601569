from fakes import FAKE_CONTENT, FakeLLMClient, FakeSearchClient

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.critic import CriticAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.schemas import AgentName, ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState


def _state() -> ResearchState:
    return ResearchState(request=ResearchQuery(query="Explain GraphRAG for engineers"))


def test_researcher_populates_sources_and_notes() -> None:
    sources = [
        SourceDocument(title="Doc A", snippet="snippet a"),
        SourceDocument(title="Doc B", snippet="snippet b"),
    ]
    fake_search = FakeSearchClient(sources)
    fake_llm = FakeLLMClient()
    agent = ResearcherAgent(llm_client=fake_llm, search_client=fake_search)

    result = agent.run(_state())

    assert result.sources == sources
    assert result.research_notes == FAKE_CONTENT
    assert len(result.agent_results) == 1
    assert result.agent_results[0].agent == AgentName.RESEARCHER
    assert "Doc A" in fake_llm.calls[0][1]


def test_analyst_populates_analysis_notes() -> None:
    fake_llm = FakeLLMClient()
    agent = AnalystAgent(llm_client=fake_llm)
    state = _state()
    state.research_notes = "Some research notes [1]."
    state.sources = [SourceDocument(title="Doc A", snippet="snippet a")]

    result = agent.run(state)

    assert result.analysis_notes == FAKE_CONTENT
    assert result.agent_results[0].agent == AgentName.ANALYST
    assert "Some research notes" in fake_llm.calls[0][1]


def test_writer_populates_final_answer() -> None:
    fake_llm = FakeLLMClient()
    agent = WriterAgent(llm_client=fake_llm)
    state = _state()
    state.research_notes = "Some research notes [1]."
    state.analysis_notes = "Key claim [1]."
    state.sources = [SourceDocument(title="Doc A", snippet="snippet a")]

    result = agent.run(state)

    assert result.final_answer == FAKE_CONTENT
    assert result.agent_results[0].agent == AgentName.WRITER


def test_critic_flags_low_citation_coverage() -> None:
    state = _state()
    state.sources = [
        SourceDocument(title="Doc A", snippet="a"),
        SourceDocument(title="Doc B", snippet="b"),
        SourceDocument(title="Doc C", snippet="c"),
    ]
    state.final_answer = "Answer referencing only [1]." * 5

    result = CriticAgent().run(state)

    last_span = result.trace[-1]
    assert last_span["name"] == "critic_review"
    assert last_span["payload"]["citation_coverage"] == 1 / 3
    assert last_span["payload"]["findings"]


def test_critic_is_satisfied_with_full_citation_coverage() -> None:
    state = _state()
    state.sources = [SourceDocument(title="Doc A", snippet="a")]
    state.final_answer = "Answer referencing [1]." * 10

    result = CriticAgent().run(state)

    last_span = result.trace[-1]
    assert last_span["payload"]["citation_coverage"] == 1.0
    assert last_span["payload"]["findings"] == []

"""Deterministic fakes for LLMClient/SearchClient, used to unit-test agents offline."""

from multi_agent_research_lab.core.schemas import SourceDocument
from multi_agent_research_lab.services.llm_client import LLMClient, LLMResponse
from multi_agent_research_lab.services.search_client import SearchClient

FAKE_CONTENT = "FAKE_CONTENT"


class FakeLLMClient(LLMClient):
    """Records every prompt it receives and always returns the same deterministic content."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[str, str]] = []

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        self.calls.append((system_prompt, user_prompt))
        return LLMResponse(content=FAKE_CONTENT, input_tokens=10, output_tokens=5, cost_usd=0.01)


class FakeSearchClient(SearchClient):
    """Returns a fixed list of sources regardless of query."""

    def __init__(self, sources: list[SourceDocument]) -> None:
        super().__init__()
        self._sources = sources

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        return self._sources[:max_results]

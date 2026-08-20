"""Search client abstraction for ResearcherAgent.

Hybrid behavior: when `Settings.tavily_api_key` is configured, `search` calls the Tavily API
for real. Otherwise it returns a deterministic local mock corpus so the pipeline stays
runnable and testable offline.
"""

import json
import ssl
from urllib.error import URLError
from urllib.request import Request, urlopen

import certifi

from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import SourceDocument

_TAVILY_URL = "https://api.tavily.com/search"


class SearchClient:
    """Provider-agnostic search client: Tavily when configured, deterministic mock otherwise."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search for documents relevant to a query."""

        if self._settings.tavily_api_key:
            return self._search_tavily(query, max_results)
        return self._search_mock(query, max_results)

    def _search_tavily(self, query: str, max_results: int) -> list[SourceDocument]:
        payload = json.dumps(
            {
                "api_key": self._settings.tavily_api_key,
                "query": query,
                "max_results": max_results,
                "include_answer": False,
            }
        ).encode("utf-8")
        request = Request(
            _TAVILY_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        # macOS python.org builds don't use the system CA store; pin certifi's bundle
        # explicitly to avoid SSLCertVerificationError (see docs/lab_guide.md).
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        try:
            with urlopen(
                request, timeout=self._settings.timeout_seconds, context=ssl_context
            ) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (URLError, TimeoutError) as exc:
            raise AgentExecutionError(f"Tavily search failed: {exc}") from exc

        results = body.get("results", [])[:max_results]
        return [
            SourceDocument(
                title=item.get("title") or item.get("url") or "Untitled",
                url=item.get("url"),
                snippet=item.get("content", ""),
                metadata={"score": item.get("score")},
            )
            for item in results
        ]

    def _search_mock(self, query: str, max_results: int) -> list[SourceDocument]:
        topic = " ".join(query.split())[:80]
        return [
            SourceDocument(
                title=f"Mock source {i + 1}: {topic}",
                url=f"https://mock.local/search?rank={i + 1}",
                snippet=(
                    f"Deterministic mock finding #{i + 1} relevant to '{topic}'. "
                    "No TAVILY_API_KEY configured, so this is offline placeholder content."
                ),
                metadata={"mock": True, "rank": i + 1},
            )
            for i in range(max_results)
        ]

"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.

Hybrid behavior: when `Settings.openai_api_key` is configured, `complete` calls OpenAI for real
(with retry/timeout). Otherwise it falls back to a deterministic offline mock so the whole
pipeline stays runnable and testable without any API key or network access.
"""

import re
from dataclasses import dataclass
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError

# Rough public per-1K-token pricing (USD) for cost estimation. Extend as needed.
_MODEL_PRICING_USD_PER_1K: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4o": (0.0025, 0.01),
}

# Keeps the offline mock's echoed context from compounding across pipeline stages.
_MOCK_PROSE_CHAR_LIMIT = 300
_CITATION_LINE_RE = re.compile(r"^\[\d+\]\s")


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


def _estimate_cost(model: str, input_tokens: int | None, output_tokens: int | None) -> float | None:
    pricing = _MODEL_PRICING_USD_PER_1K.get(model)
    if pricing is None or input_tokens is None or output_tokens is None:
        return None
    input_price, output_price = pricing
    return (input_tokens / 1000) * input_price + (output_tokens / 1000) * output_price


def _estimate_tokens(text: str) -> int:
    """Rough offline token estimate (~4 chars/token) used only by the mock path."""

    return max(1, len(text) // 4)


def _extract_section(text: str, start_marker: str, end_marker: str) -> str:
    start = text.find(start_marker)
    if start == -1:
        return text
    start += len(start_marker)
    end = text.find(end_marker, start)
    return text[start:end] if end != -1 else text[start:]


def _first_line(text: str) -> str:
    stripped = text.strip()
    return stripped.splitlines()[0] if stripped else ""


class LLMClient:
    """Provider-agnostic LLM client: real OpenAI call when a key is configured, mock otherwise."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._openai_client: Any = None

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a model completion.

        Uses OpenAI when `OPENAI_API_KEY` is set; otherwise returns a deterministic mock
        response so agents, tests, and the CLI keep working offline.
        """

        if self._settings.openai_api_key:
            try:
                return self._call_openai(system_prompt, user_prompt)
            except Exception as exc:  # pragma: no cover - network path, exercised manually
                raise AgentExecutionError(f"LLM call failed after retries: {exc}") from exc
        return self._mock_complete(system_prompt, user_prompt)

    def _get_openai_client(self) -> Any:
        if self._openai_client is None:
            from openai import OpenAI

            self._openai_client = OpenAI(
                api_key=self._settings.openai_api_key,
                timeout=self._settings.timeout_seconds,
            )
        return self._openai_client

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, max=4), reraise=True)
    def _call_openai(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        client = self._get_openai_client()
        response = client.chat.completions.create(
            model=self._settings.openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response.choices[0].message.content or ""
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else None
        output_tokens = usage.completion_tokens if usage else None
        cost = _estimate_cost(self._settings.openai_model, input_tokens, output_tokens)
        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
        )

    def _mock_complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        context = _extract_section(user_prompt, "### CONTEXT", "### TASK").strip()

        # Downstream agents feed prior mock output back in as context (research_notes ->
        # analysis_notes -> final_answer), so naively echoing everything compounds across
        # hops. Split citation lines ("[n] ...") from free-form prose: prose is capped to a
        # fixed budget (bounded, readable output), while every distinct citation line is
        # deduplicated and always kept in full, so citation coverage survives truncation and
        # nesting depth regardless of where the source list appears in the prompt.
        citation_lines: list[str] = []
        seen_citations: set[str] = set()
        prose_lines: list[str] = []
        for line in context.splitlines():
            if _CITATION_LINE_RE.match(line):
                if line not in seen_citations:
                    seen_citations.add(line)
                    citation_lines.append(line)
            elif line.strip():
                prose_lines.append(line.strip())

        prose = " ".join(prose_lines)
        if len(prose) > _MOCK_PROSE_CHAR_LIMIT:
            prose = prose[:_MOCK_PROSE_CHAR_LIMIT].rstrip() + " [...truncated...]"

        body = "\n".join(citation_lines)
        if prose:
            body = f"{prose}\n{body}" if body else prose

        role = _first_line(system_prompt) or "assistant"
        content = f"[mock:{role}] Deterministic offline synthesis (no OPENAI_API_KEY set):\n{body}"
        input_tokens = _estimate_tokens(system_prompt) + _estimate_tokens(user_prompt)
        output_tokens = _estimate_tokens(content)
        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            # Same pricing-table estimate used for real calls, applied to the mock's rough
            # token counts. This is an ESTIMATE (what a real call would cost at today's
            # published pricing), not a measurement — no request was actually billed.
            cost_usd=_estimate_cost(self._settings.openai_model, input_tokens, output_tokens),
        )

"""Critic agent: bonus fact-check / citation-coverage guard, no LLM call needed."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.state import ResearchState

_MIN_ANSWER_CHARS = 80


class CriticAgent(BaseAgent):
    """Validates the final answer and appends findings without blocking the pipeline."""

    name = "critic"

    def run(self, state: ResearchState) -> ResearchState:
        """Validate `state.final_answer` and append findings to trace/errors."""

        findings: list[str] = []
        answer = state.final_answer or ""

        if len(answer) < _MIN_ANSWER_CHARS:
            findings.append(f"final_answer looks too short ({len(answer)} chars)")

        cited = sum(1 for i in range(len(state.sources)) if f"[{i + 1}]" in answer)
        coverage = cited / len(state.sources) if state.sources else 0.0
        if state.sources and coverage < 0.5:
            findings.append(
                f"low citation coverage: {cited}/{len(state.sources)} sources referenced"
            )

        # Findings are quality signals, not hard failures — keep them in the trace only so
        # `state.errors` stays reserved for actual execution/guardrail failures.
        state.add_trace_event(
            "critic_review",
            {"citation_coverage": coverage, "findings": findings},
        )
        return state

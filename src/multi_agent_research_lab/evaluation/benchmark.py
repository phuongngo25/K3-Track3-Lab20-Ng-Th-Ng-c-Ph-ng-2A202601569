"""Benchmark harness for single-agent vs multi-agent runs."""

from collections.abc import Callable
from time import perf_counter

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState

Runner = Callable[[str], ResearchState]

_MIN_QUALITY_ANSWER_CHARS = 200

# Below this, wall-clock latency is almost certainly the offline mock path (no real network
# call), not a fast real API response.
_MOCK_LATENCY_THRESHOLD_SECONDS = 0.05
_SIMULATED_BASE_LATENCY_SECONDS = 0.4  # rough per-call network/queueing overhead
_SIMULATED_SECONDS_PER_OUTPUT_TOKEN = 1 / 60  # ~60 tok/s, a typical small-model throughput
SIMULATED_LATENCY_NOTE = "latency simulated (no API key configured; modeled, not measured)"


def citation_coverage(state: ResearchState) -> float:
    """Fraction of sources referenced (by `[n]` marker) in the final answer."""

    if not state.sources:
        return 0.0
    answer = state.final_answer or ""
    cited = sum(1 for i in range(len(state.sources)) if f"[{i + 1}]" in answer)
    return cited / len(state.sources)


def estimated_cost_usd(state: ResearchState) -> float:
    return sum(result.metadata.get("cost_usd") or 0.0 for result in state.agent_results)


def simulate_latency_seconds(state: ResearchState) -> float:
    """Model a plausible latency for a run that used the offline mock (no real API calls).

    This is a documented estimate — base per-call overhead plus output tokens divided by an
    assumed generation throughput — NOT a measurement. Only used to keep the benchmark report
    legible when no API key is configured; always paired with `SIMULATED_LATENCY_NOTE` so it's
    never mistaken for a real measured latency.
    """

    total = 0.0
    for result in state.agent_results:
        output_tokens = result.metadata.get("output_tokens") or 0
        total += (
            _SIMULATED_BASE_LATENCY_SECONDS + output_tokens * _SIMULATED_SECONDS_PER_OUTPUT_TOKEN
        )
    return total


def quality_score(state: ResearchState) -> float:
    """Heuristic 0-10 quality proxy: answer presence/length, citation coverage, and absence
    of guardrail errors. Stands in for the human peer-review score
    (docs/peer_review_rubric.md) until a reviewer actually scores the run.
    """

    score = 0.0
    answer = state.final_answer or ""
    if answer:
        score += 4.0 if len(answer) >= _MIN_QUALITY_ANSWER_CHARS else 2.0
    score += 3.0 * citation_coverage(state)
    score += 3.0 if not state.errors else 0.0
    return min(score, 10.0)


def run_benchmark(
    run_name: str, query: str, runner: Runner
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Run `runner` once and measure latency, cost, quality, citation coverage, and failure."""

    started = perf_counter()
    state = runner(query)
    latency = perf_counter() - started

    notes = list(state.errors)
    if latency < _MOCK_LATENCY_THRESHOLD_SECONDS and state.agent_results:
        # Real wall-clock time from an offline mock run is meaninglessly small; report a
        # clearly-labeled simulated estimate instead of a near-zero number that could be
        # misread as "multi-agent is free and instant".
        latency = simulate_latency_seconds(state)
        notes.append(SIMULATED_LATENCY_NOTE)

    failed = bool(state.errors) or not state.final_answer
    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        estimated_cost_usd=estimated_cost_usd(state),
        quality_score=quality_score(state),
        citation_coverage=citation_coverage(state),
        failure_rate=1.0 if failed else 0.0,
        notes="; ".join(notes),
    )
    return state, metrics

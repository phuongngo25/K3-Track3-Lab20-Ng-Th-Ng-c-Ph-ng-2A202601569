"""Benchmark report rendering."""

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.evaluation.benchmark import SIMULATED_LATENCY_NOTE


def _avg(metrics: list[BenchmarkMetrics], attr: str) -> float:
    values = [v for m in metrics if (v := getattr(m, attr)) is not None]
    return sum(values) / len(values) if values else 0.0


def render_markdown_report(metrics: list[BenchmarkMetrics]) -> str:
    """Render benchmark metrics to markdown with a short baseline-vs-multi-agent analysis."""

    lines = [
        "# Benchmark Report",
        "",
        "| Run | Latency (s) | Cost (USD) | Quality | Citation cov. | Failure rate | Notes |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in metrics:
        cost = "" if item.estimated_cost_usd is None else f"{item.estimated_cost_usd:.4f}"
        quality = "" if item.quality_score is None else f"{item.quality_score:.1f}"
        citation = "" if item.citation_coverage is None else f"{item.citation_coverage:.0%}"
        failure = "" if item.failure_rate is None else f"{item.failure_rate:.0%}"
        lines.append(
            f"| {item.run_name} | {item.latency_seconds:.2f} | {cost} | {quality} "
            f"| {citation} | {failure} | {item.notes} |"
        )

    lines += ["", "## Analysis", ""]
    if any(SIMULATED_LATENCY_NOTE in (m.notes or "") for m in metrics):
        lines.append(
            "- **Offline mock run — Latency and Cost below are ESTIMATES, not measurements**: "
            "no `OPENAI_API_KEY`/`TAVILY_API_KEY` were configured, so every run used the "
            "deterministic mock LLM/search path (no real network call). *Cost* is computed with "
            "the same public per-token pricing table used for real calls, applied to the mock's "
            "estimated token counts. *Latency* is a simulated model (`~0.4s` base overhead + "
            "output tokens / 60 tok/s per LLM call) — rows carrying this estimate are marked "
            f"`{SIMULATED_LATENCY_NOTE}` in their Notes column. Add real keys to `.env` and "
            "re-run `make benchmark` for genuinely measured numbers."
        )
    baseline = [m for m in metrics if m.run_name.startswith("baseline")]
    multi = [m for m in metrics if m.run_name.startswith("multi-agent")]
    if baseline and multi:
        lines.append(
            f"- Avg latency: baseline {_avg(baseline, 'latency_seconds'):.2f}s vs "
            f"multi-agent {_avg(multi, 'latency_seconds'):.2f}s."
        )
        lines.append(
            f"- Avg cost: baseline ${_avg(baseline, 'estimated_cost_usd'):.4f} vs "
            f"multi-agent ${_avg(multi, 'estimated_cost_usd'):.4f}."
        )
        lines.append(
            f"- Avg quality (heuristic): baseline {_avg(baseline, 'quality_score'):.1f}/10 vs "
            f"multi-agent {_avg(multi, 'quality_score'):.1f}/10."
        )
        lines.append(
            f"- Avg citation coverage: baseline {_avg(baseline, 'citation_coverage'):.0%} vs "
            f"multi-agent {_avg(multi, 'citation_coverage'):.0%}."
        )
    else:
        lines.append("- Not enough runs to compare baseline vs multi-agent yet.")

    failed = [m for m in metrics if (m.failure_rate or 0) > 0]
    if failed:
        lines.append(f"- {len(failed)} run(s) failed:")
        lines.extend(
            f"  - `{item.run_name}`: {item.notes or 'no details recorded'}" for item in failed
        )
    else:
        lines.append("- No failed runs observed in this benchmark pass.")

    lines += ["", "## Failure Modes Observed & Fixes", "", _FAILURE_MODES_NOTE]

    return "\n".join(lines) + "\n"


_FAILURE_MODES_NOTE = """\
While building this pipeline, two real failure modes showed up during manual testing \
(not synthetic): 1) `multi-agent`'s JSON output was piped through Rich's `console.print`, \
which word-wraps long lines and treats `[...]` as markup — this silently corrupted the JSON \
when redirected to a file and ate literal `[n]` citation text on screen. Fix: use plain \
`print()` for machine-readable JSON and wrap human-facing panel text in `rich.text.Text` to \
disable markup parsing. 2) The offline mock LLM originally echoed its entire input context \
verbatim; since Analyst/Writer each feed the previous stage's (already-echoed) output back in \
as context, the text compounded across hops, and naively truncating it to bound the growth cut \
off the `[n]` source list entirely (since it's appended last), silently zeroing citation \
coverage. Fix: the mock now always preserves every distinct `[n]` citation line in full and \
only truncates free-form prose, so citation coverage stays accurate regardless of pipeline \
depth or truncation.\
"""

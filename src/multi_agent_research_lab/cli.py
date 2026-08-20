"""Command-line entrypoint for the lab starter."""

from pathlib import Path
from typing import Annotated

import typer
import yaml
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import LabError
from multi_agent_research_lab.core.schemas import (
    AgentName,
    AgentResult,
    BenchmarkMetrics,
    ResearchQuery,
)
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient
from multi_agent_research_lab.services.storage import LocalArtifactStore

app = typer.Typer(help="Multi-Agent Research Lab starter CLI")
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


def _parse_query(query: str) -> ResearchQuery:
    try:
        return ResearchQuery(query=query)
    except ValidationError as exc:
        console.print(
            Panel.fit(
                Text(f"Invalid query: {exc.errors()[0]['msg']}"),
                title="Input Error",
                style="red",
            )
        )
        raise typer.Exit(code=1) from exc


def _run_baseline(query: str) -> ResearchState:
    """Single agent doing search + analysis + writing in one LLM call."""

    request = ResearchQuery(query=query)
    state = ResearchState(request=request)

    search_client = SearchClient()
    llm_client = LLMClient()

    sources = search_client.search(request.query, max_results=request.max_sources)
    state.sources = sources

    context = "\n".join(f"[{i + 1}] {s.title} — {s.snippet}" for i, s in enumerate(sources))
    system_prompt = (
        "You are a single-agent research assistant. Research, analyze, and write the final "
        "answer yourself in one pass, citing sources as [n]."
    )
    user_prompt = (
        f"### CONTEXT\n{context}\n\n"
        f"### TASK\nAnswer the query for {request.audience} in under 500 words: {request.query}"
    )
    response = llm_client.complete(system_prompt, user_prompt)
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


def _run_multi_agent(query: str) -> ResearchState:
    state = ResearchState(request=ResearchQuery(query=query))
    return MultiAgentWorkflow().run(state)


def _load_benchmark_queries(config_path: str) -> list[str]:
    data = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    queries: list[str] = data.get("benchmark", {}).get("queries", [])
    if not queries:
        raise typer.BadParameter(f"No benchmark.queries found in {config_path}")
    return queries


def _short_label(query: str, limit: int = 40) -> str:
    return query if len(query) <= limit else query[: limit - 3] + "..."


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the single-agent baseline end-to-end."""

    _init()
    _parse_query(query)
    try:
        state = _run_baseline(query)
    except LabError as exc:
        console.print(Panel.fit(Text(str(exc)), title="Baseline Error", style="red"))
        raise typer.Exit(code=2) from exc
    console.print(Panel.fit(Text(state.final_answer or ""), title="Single-Agent Baseline"))


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the multi-agent workflow end-to-end."""

    _init()
    _parse_query(query)
    try:
        result = _run_multi_agent(query)
    except LabError as exc:
        console.print(Panel.fit(Text(str(exc)), title="Multi-Agent Error", style="red"))
        raise typer.Exit(code=2) from exc
    # Plain print (not console.print): Rich soft-wraps long lines, which corrupts JSON
    # when the output is piped or redirected to a file.
    print(result.model_dump_json(indent=2))


@app.command()
def benchmark(
    config_path: Annotated[
        str, typer.Option("--config", help="YAML file with benchmark.queries")
    ] = "configs/lab_default.yaml",
    output_path: Annotated[
        str, typer.Option("--output", help="Report path, relative to reports/")
    ] = "benchmark_report.md",
) -> None:
    """Run baseline and multi-agent over the configured queries and write a benchmark report."""

    _init()
    queries = _load_benchmark_queries(config_path)

    all_metrics: list[BenchmarkMetrics] = []
    for query in queries:
        label = _short_label(query)
        for run_name, runner in (("baseline", _run_baseline), ("multi-agent", _run_multi_agent)):
            full_name = f"{run_name}: {label}"
            try:
                _, metrics = run_benchmark(full_name, query, runner)
            except Exception as exc:  # noqa: BLE001 - benchmark must survive per-query failures
                metrics = BenchmarkMetrics(
                    run_name=full_name,
                    latency_seconds=0.0,
                    failure_rate=1.0,
                    notes=f"raised {type(exc).__name__}: {exc}",
                )
            console.print(f"[dim]ran[/dim] {full_name}")
            all_metrics.append(metrics)

    report = render_markdown_report(all_metrics)
    store = LocalArtifactStore()
    path = store.write_text(output_path, report)
    console.print(Panel.fit(f"Benchmark report written to {path}", title="Benchmark"))


if __name__ == "__main__":
    app()

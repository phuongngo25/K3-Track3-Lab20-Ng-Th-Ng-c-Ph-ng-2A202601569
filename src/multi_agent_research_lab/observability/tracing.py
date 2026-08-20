"""Tracing hooks.

This file intentionally avoids binding to one provider. Students can plug in LangSmith,
Langfuse, OpenTelemetry, or simple JSON traces.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter
from typing import Any


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Minimal span context, used by `graph/workflow.py` to populate `state.trace`.

    Every agent node in the workflow is wrapped with this and its result is appended to
    `state.trace`, so a run's whole timeline is inspectable from `ResearchState` alone even
    without an external tracing provider.

    Bonus upgrade path: if `Settings.langsmith_api_key` / `langfuse_*` are set, swap the body
    below for a real provider span (`langsmith.trace(...)` / `langfuse.trace(...)`) instead of
    the local dict — the `name`/`attributes` signature already matches what those SDKs expect.
    """

    started = perf_counter()
    span: dict[str, Any] = {"name": name, "attributes": attributes or {}, "duration_seconds": None}
    try:
        yield span
    finally:
        span["duration_seconds"] = perf_counter() - started

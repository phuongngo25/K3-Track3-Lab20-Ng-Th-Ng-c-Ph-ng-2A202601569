# Benchmark Report

| Run | Latency (s) | Cost (USD) | Quality | Citation cov. | Failure rate | Notes |
|---|---:|---:|---:|---:|---:|---|
| baseline: Research GraphRAG state-of-the-art an... | 6.72 | 0.0003 | 10.0 | 100% | 0% | latency simulated (no API key configured; modeled, not measured) |
| multi-agent: Research GraphRAG state-of-the-art an... | 24.23 | 0.0012 | 10.0 | 100% | 0% | latency simulated (no API key configured; modeled, not measured) |
| baseline: Compare single-agent and multi-agent ... | 6.88 | 0.0003 | 10.0 | 100% | 0% | latency simulated (no API key configured; modeled, not measured) |
| multi-agent: Compare single-agent and multi-agent ... | 24.73 | 0.0013 | 10.0 | 100% | 0% | latency simulated (no API key configured; modeled, not measured) |
| baseline: Summarize production guardrails for L... | 6.00 | 0.0003 | 10.0 | 100% | 0% | latency simulated (no API key configured; modeled, not measured) |
| multi-agent: Summarize production guardrails for L... | 22.12 | 0.0011 | 10.0 | 100% | 0% | latency simulated (no API key configured; modeled, not measured) |

## Analysis

- **Offline mock run — Latency and Cost below are ESTIMATES, not measurements**: no `OPENAI_API_KEY`/`TAVILY_API_KEY` were configured, so every run used the deterministic mock LLM/search path (no real network call). *Cost* is computed with the same public per-token pricing table used for real calls, applied to the mock's estimated token counts. *Latency* is a simulated model (`~0.4s` base overhead + output tokens / 60 tok/s per LLM call) — rows carrying this estimate are marked `latency simulated (no API key configured; modeled, not measured)` in their Notes column. Add real keys to `.env` and re-run `make benchmark` for genuinely measured numbers.
- Avg latency: baseline 6.53s vs multi-agent 23.69s.
- Avg cost: baseline $0.0003 vs multi-agent $0.0012.
- Avg quality (heuristic): baseline 10.0/10 vs multi-agent 10.0/10.
- Avg citation coverage: baseline 100% vs multi-agent 100%.
- No failed runs observed in this benchmark pass.

## Failure Modes Observed & Fixes

While building this pipeline, two real failure modes showed up during manual testing (not synthetic): 1) `multi-agent`'s JSON output was piped through Rich's `console.print`, which word-wraps long lines and treats `[...]` as markup — this silently corrupted the JSON when redirected to a file and ate literal `[n]` citation text on screen. Fix: use plain `print()` for machine-readable JSON and wrap human-facing panel text in `rich.text.Text` to disable markup parsing. 2) The offline mock LLM originally echoed its entire input context verbatim; since Analyst/Writer each feed the previous stage's (already-echoed) output back in as context, the text compounded across hops, and naively truncating it to bound the growth cut off the `[n]` source list entirely (since it's appended last), silently zeroing citation coverage. Fix: the mock now always preserves every distinct `[n]` citation line in full and only truncates free-form prose, so citation coverage stays accurate regardless of pipeline depth or truncation.

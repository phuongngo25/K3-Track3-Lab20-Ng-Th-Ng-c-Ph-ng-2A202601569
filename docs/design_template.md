# Design Template

## Problem

Xây một research assistant nhận một câu hỏi nghiên cứu dài (ví dụ: "Research GraphRAG
state-of-the-art and write a 500-word summary"), tự tìm nguồn, phân tích độ tin cậy/mâu thuẫn
giữa các nguồn, rồi viết câu trả lời cuối cùng có trích dẫn `[n]` trỏ về nguồn. Hệ thống phải
chạy được cả ở chế độ offline (không có API key) lẫn với LLM/search provider thật, và phải đo
được latency/cost/quality để so sánh với một single-agent baseline.

## Why multi-agent?

Single-agent (`cli.py::_run_baseline`) gộp cả tìm nguồn + phân tích + viết vào **một** lời gọi
LLM. Điều đó nhanh và rẻ, nhưng: (1) không có bước phân tích riêng nên dễ bỏ sót mâu thuẫn giữa
các nguồn, (2) không có checkpoint nào để validate trước khi trả lời — nếu search trả về nguồn
kém, câu trả lời cuối vẫn "trông ổn" vì không ai kiểm tra lại. Multi-agent tách rõ 3 trách nhiệm
(tìm — phân tích — viết) và thêm một bước kiểm tra độc lập (Critic) trước khi coi là "done", đổi
lại chi phí là nhiều lời gọi LLM hơn và độ trễ cao hơn. Đây chính là trade-off cần benchmark,
không phải giả định.

## Agent roles

| Agent | Responsibility | Input | Output | Failure mode |
|---|---|---|---|---|
| Supervisor | Quyết định route tiếp theo dựa trên state hiện có; không gọi LLM | `state` hiện tại | `route_history` mới | Loop vô hạn nếu thiếu guardrail → chặn bằng `max_iterations` |
| Researcher | Tìm nguồn qua `SearchClient`, tổng hợp `research_notes` có trích dẫn `[n]` | `request.query`, `max_sources` | `sources`, `research_notes` | Search trả về 0 nguồn hoặc nguồn không liên quan → notes rỗng/lạc đề |
| Analyst | Đọc `research_notes` + `sources`, rút key claims, so sánh quan điểm, gắn cờ bằng chứng yếu | `research_notes`, `sources` | `analysis_notes` | Bỏ sót mâu thuẫn nếu nguồn quá ít hoặc LLM tóm tắt hời hợt |
| Writer | Tổng hợp `research_notes` + `analysis_notes` thành câu trả lời cuối, giữ trích dẫn `[n]` | `research_notes`, `analysis_notes`, `sources` | `final_answer` | Trích dẫn sai nguồn hoặc quên trích dẫn → Critic sẽ gắn cờ |
| Critic (bonus) | Kiểm tra rule-based: độ dài câu trả lời, tỷ lệ nguồn được trích dẫn | `final_answer`, `sources` | `trace` event `critic_review` (không raise, không chặn) | Không phát hiện lỗi factual — chỉ là proxy citation coverage, không thay được review của người |

## Shared state

`ResearchState` (`core/state.py`) là single source of truth truyền qua mọi node trong graph:

- `request`: câu hỏi gốc + cấu hình (`max_sources`, `audience`) — mọi agent cần để biết đang trả lời ai.
- `route_history` + `iteration`: lịch sử routing và bộ đếm — Supervisor dùng để enforce `max_iterations`, đồng thời là bằng chứng trace "ai chạy khi nào".
- `sources`: danh sách `SourceDocument` — cần persist vì cả Analyst, Writer, Critic đều tham chiếu lại (tính citation coverage) mà không cần search lại.
- `research_notes` / `analysis_notes` / `final_answer`: 3 output trung gian tách biệt — nếu gộp chung một field sẽ mất khả năng debug "Analyst đã thấy gì trước khi viết analysis".
- `agent_results`: `AgentResult` kèm `metadata` (token/cost) mỗi lần gọi LLM — benchmark cộng dồn từ đây để tính `estimated_cost_usd`.
- `trace`: danh sách event có `duration_seconds` (từ `trace_span`) — dùng để trả lời "bước nào tốn bao lâu, agent nào chạy mấy lần".
- `errors`: chỉ dành cho lỗi thật (ví dụ chạm `max_iterations` mà chưa có `final_answer`) — cố tình **không** chứa cảnh báo chất lượng của Critic, để `failure_rate` trong benchmark không bị lẫn với điểm chất lượng.

## Routing policy

Supervisor (`agents/supervisor.py`) là hàm thuần dựa trên state, không gọi LLM:

```text
if iteration >= max_iterations:
    route = done (và ghi error nếu chưa có final_answer)
elif không có sources:
    route = researcher
elif có sources nhưng chưa có analysis_notes:
    route = analyst
elif có analysis_notes nhưng chưa có final_answer:
    route = writer
else:
    route = done
```

Trong LangGraph (`graph/workflow.py`): entry point là `supervisor`; conditional edge đọc
`route_history[-1]` để nhảy tới `researcher`/`analyst`/`writer` hoặc `END`. Mỗi worker chạy xong
quay lại `supervisor` để nó quyết định bước kế — riêng `writer` luôn đi qua `critic` trước khi
quay lại `supervisor`, đảm bảo câu trả lời cuối luôn được kiểm tra một lần.

## Guardrails

- Max iterations: `Settings.max_iterations` (mặc định 6) — Supervisor tự chặn ở tầng logic.
- Timeout: `Settings.timeout_seconds` (mặc định 60s) — áp dụng cho OpenAI client và Tavily HTTP call.
- Retry: `LLMClient._call_openai` dùng `tenacity` (3 lần, exponential backoff) trước khi raise `AgentExecutionError`.
- Fallback: không có API key → `LLMClient`/`SearchClient` tự rơi về mock offline thay vì crash.
- Validation: `ResearchQuery`/`SourceDocument`/`BenchmarkMetrics` đều là Pydantic model; `_parse_query` trong CLI chặn input rỗng/quá ngắn trước khi vào workflow.
- Recursion guard tầng 2: LangGraph `recursion_limit = max_iterations * 4 + 10` trong `MultiAgentWorkflow.run`.

## Benchmark plan

Query set: `configs/lab_default.yaml -> benchmark.queries` (3 câu, đại diện cho research
summary, so sánh single/multi-agent, và tổng hợp guardrail).

| Metric | Cách đo | Expected outcome |
|---|---|---|
| Latency | wall-clock quanh `runner(query)` trong `run_benchmark` | Multi-agent chậm hơn baseline (nhiều lời gọi LLM tuần tự) |
| Cost | Cộng `cost_usd` từ `agent_results[].metadata` | Multi-agent tốn hơn baseline (3 lời gọi LLM so với 1) |
| Quality | Heuristic 0-10 (độ dài câu trả lời + citation coverage + không có error) — placeholder cho điểm rubric người thật chấm | Multi-agent có tiềm năng cao hơn nhờ bước Analyst/Critic, nhưng cần review thật để xác nhận |
| Citation coverage | Tỷ lệ nguồn `[n]` xuất hiện trong `final_answer` | Multi-agent nên bằng hoặc cao hơn baseline vì có Critic nhắc |
| Failure rate | `state.errors` non-empty hoặc thiếu `final_answer`, tổng hợp qua các query trong `malab benchmark` | Cả hai nên gần 0% nếu guardrail hoạt động đúng |

Chạy `make benchmark` để tạo `reports/benchmark_report.md`. **Lưu ý**: bản báo cáo hiện tại được
tạo ở chế độ offline mock (chưa có `OPENAI_API_KEY`/`TAVILY_API_KEY`), nên cột Latency và Cost là
**ước tính có công thức**, không phải số đo thật: Cost dùng lại đúng bảng giá per-token của
`_estimate_cost` áp cho số token ước lượng của mock; Latency dùng model mô phỏng (`0.4s` overhead
cộng output tokens chia 60 tok/s mỗi lời gọi LLM) — các dòng này được đánh dấu rõ trong cột Notes
(`latency simulated (no API key configured; modeled, not measured)`) để không ai nhầm là số đo
thật. Cần điền `.env` rồi chạy lại `make benchmark` để có số liệu đo thật trước khi nộp bài.

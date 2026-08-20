# Lab Guide: Multi-Agent Research System

## Scenario

Bạn cần xây dựng một research assistant có thể nhận câu hỏi dài, tìm thông tin, phân tích và viết câu trả lời cuối cùng. Lab yêu cầu so sánh hai cách làm:

1. **Single-agent baseline**: một agent làm toàn bộ.
2. **Multi-agent workflow**: Supervisor điều phối Researcher, Analyst, Writer.

## Quy tắc quan trọng

- Không thêm agent nếu không có lý do rõ ràng.
- Mỗi agent phải có responsibility riêng.
- Shared state phải đủ rõ để debug.
- Phải có trace hoặc log cho từng bước.
- Phải benchmark, không chỉ nhìn output bằng cảm tính.

## Milestone 1: Baseline

File gợi ý:

- `src/multi_agent_research_lab/cli.py`
- `src/multi_agent_research_lab/services/llm_client.py`

TODO(student): thay baseline placeholder bằng một call LLM thật.

## Milestone 2: Supervisor

File gợi ý:

- `src/multi_agent_research_lab/agents/supervisor.py`
- `src/multi_agent_research_lab/graph/workflow.py`

TODO(student): implement routing policy.

Gợi ý câu hỏi thiết kế:

- Khi nào gọi Researcher?
- Khi nào gọi Analyst?
- Khi nào gọi Writer?
- Khi nào stop?
- Nếu agent fail thì retry hay fallback?

## Milestone 3: Worker agents

File gợi ý:

- `src/multi_agent_research_lab/agents/researcher.py`
- `src/multi_agent_research_lab/agents/analyst.py`
- `src/multi_agent_research_lab/agents/writer.py`

TODO(student): implement từng worker.

## Milestone 4: Trace và benchmark

File gợi ý:

- `src/multi_agent_research_lab/observability/tracing.py`
- `src/multi_agent_research_lab/evaluation/benchmark.py`
- `src/multi_agent_research_lab/evaluation/report.py`

Benchmark tối thiểu:

| Metric | Cách đo gợi ý |
|---|---|
| Latency | wall-clock time |
| Cost | token usage hoặc provider usage |
| Quality | rubric 0-10 do peer review |
| Citation coverage | số claims có source / tổng claims chính |
| Failure rate | số query fail / tổng query |

## Troubleshooting

### macOS: lỗi SSL certificate khi gọi API qua HTTPS (Tavily, OpenAI, ...)

Triệu chứng: khi implement `SearchClient` (hoặc bất kỳ HTTPS call nào) trên macOS, bạn có thể gặp lỗi kiểu:

```
ssl.SSLCertVerificationError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed:
unable to get local issuer certificate
```

Nguyên nhân: Python cài từ python.org trên macOS **không dùng** certificate store của hệ điều hành, nên không tìm thấy CA bundle hợp lệ. Đây là lỗi môi trường, **không phải** do API key sai.

Cách khắc phục (chọn 1 trong 3):

1. **Chạy script cài certificate đi kèm Python** (nhanh nhất):

   ```bash
   /Applications/Python\ 3.12/Install\ Certificates.command
   ```

   (thay `3.12` bằng version Python của bạn)

2. **Dùng `certifi` trong code** — thêm `certifi` vào dependencies, rồi tạo SSL context khi gọi HTTPS:

   ```python
   import certifi
   import ssl
   from urllib.request import urlopen

   ssl_context = ssl.create_default_context(cafile=certifi.where())
   urlopen(request, timeout=timeout, context=ssl_context)
   ```

3. **Set biến môi trường** trỏ tới CA bundle của certifi (không cần đổi code):

   ```bash
   export SSL_CERT_FILE=$(python -m certifi)
   ```

## Exit ticket

Mỗi nhóm trả lời 2 câu:

1. **Case nào nên dùng multi-agent? Vì sao?**

   Khi câu hỏi cần nhiều nguồn khác nhau và các nguồn đó có thể mâu thuẫn nhau — ví dụ "so sánh
   hai kiến trúc X và Y", "tổng hợp best practice từ nhiều tài liệu". Tách Researcher/Analyst/
   Writer giúp có một bước phân tích độc lập để phát hiện mâu thuẫn/bằng chứng yếu trước khi
   viết câu trả lời, và bước Critic kiểm tra citation coverage trước khi trả kết quả — điều mà
   một single-agent gộp hết vào một lời gọi LLM khó làm tốt vì phải "vừa tìm vừa phân tích vừa
   viết" trong cùng một ngữ cảnh. Multi-agent cũng đáng giá khi cần audit trail rõ ràng (trace
   từng bước) để debug khi câu trả lời sai — xem agent nào, ở bước nào, tạo ra thông tin lệch.

2. **Case nào không nên dùng multi-agent? Vì sao?**

   Khi câu hỏi đơn giản, chỉ cần 1 nguồn thông tin hoặc không cần tra cứu (ví dụ: "định nghĩa
   guardrail là gì", "viết lại đoạn văn sau cho ngắn gọn hơn"). Benchmark ở chế độ mock cho thấy
   multi-agent luôn tốn thêm ít nhất 3 lời gọi LLM (Researcher + Analyst + Writer) so với 1 lời
   gọi của baseline — với query đơn giản, chi phí và latency tăng thêm này không đổi lại chất
   lượng tương xứng, vì không có gì để "phân tích" hay "so sánh viewpoint". Cũng nên tránh
   multi-agent khi latency là ràng buộc cứng (ví dụ chatbot cần trả lời real-time), vì mỗi vòng
   Supervisor → worker → Supervisor cộng dồn thời gian tuần tự.

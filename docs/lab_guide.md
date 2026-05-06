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

## Milestone 1: Baseline ✅

**Đã implement:** `services/llm_client.py`, `services/search_client.py`, `cli.py`

- LLMClient: auto-detect LM Studio local → OpenAI cloud → RuntimeError
- SearchClient: Tavily nếu có key, mock data nếu không
- CLI `baseline` command: single LLM call với trace + benchmark metrics

Chạy thử:
```bash
python -m multi_agent_research_lab.cli baseline --query "Research GraphRAG"
```

## Milestone 2: Supervisor ✅

**Đã implement:** `agents/supervisor.py`, `graph/workflow.py`

Routing policy:
- Ưu tiên: LM Studio → OpenAI → if/else fallback
- LLM trả JSON `{"next": "researcher|analyst|writer|done", "reason": "..."}`
- Guard cứng: `iteration >= max_iterations` → force "done"

LangGraph graph:
```
Supervisor ──conditional──► researcher/analyst/writer/done
worker ──────────────────► Supervisor (loop back)
```

## Milestone 3: Worker agents ✅

**Đã implement:** `agents/researcher.py`, `agents/analyst.py`, `agents/writer.py`, `agents/critic.py`

- **Researcher**: search → LLM summarize → `research_notes`
- **Analyst**: LLM extract key claims, viewpoints, weak evidence → `analysis_notes`
- **Writer**: LLM synthesize ~500 từ với citations → `final_answer`
- **Critic** (bonus): LLM fact-check citations, phát hiện hallucination → verdict pass/warn/fail

## Milestone 4: Trace và benchmark ✅

**Đã implement:** `observability/tracing.py`, `evaluation/benchmark.py`, `evaluation/report.py`

- `trace_span()` ghi JSONL vào `logs/trace_*.jsonl` với input/output mỗi agent
- LangSmith: đặt `LANGSMITH_TRACING=true` + `LANGSMITH_API_KEY` trong `.env`
- Tên trace phân biệt: `"baseline: <query[:50]>"`, `"multi_agent: <query[:50]>"`

Quality score 5 chiều (mỗi max 2 điểm, tổng 10):

| Dimension | Đo gì |
|---|---|
| Length | 380–700 từ = full score, penalty nếu quá ngắn/dài |
| Structure | Có header, bullet list, Key Takeaways |
| Citations | Số source được cite `[1]`, `[2]`... / tổng source |
| Relevance | Keyword query có xuất hiện trong answer |
| Completeness | Pipeline có đủ research_notes, analysis_notes, final_answer |

Chạy benchmark đầy đủ (7 queries, cả 2 mode):
```bash
python -m multi_agent_research_lab.cli benchmark --mode both
```

Report lưu tại `reports/benchmark_full_YYYYMMDD_HHMMSS.md` với bảng averages baseline vs multi-agent.

## Exit ticket

Mỗi nhóm trả lời 2 câu:

1. **Case nào nên dùng multi-agent?**
   Khi task có nhiều bước chuyên biệt, cần temperature/prompt khác nhau, hoặc cần retry từng bước độc lập. Ví dụ: research → analysis → writing là 3 "mode" tư duy khác nhau.

2. **Case nào không nên dùng multi-agent?**
   Khi task đơn giản, latency quan trọng hơn quality, hoặc context đủ nhỏ để 1 LLM call xử lý tốt. Multi-agent tốn gấp 3-5× token và latency so với baseline.

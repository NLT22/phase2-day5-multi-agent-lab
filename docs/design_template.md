# Design Template — Multi-Agent Research System

## Problem

Xây dựng research assistant nhận câu hỏi kỹ thuật phức tạp, tự động tìm kiếm nguồn, phân tích thông tin và viết câu trả lời ~500 từ có trích dẫn. Hệ thống phải trace được từng bước và benchmark được chất lượng so với single-agent.

## Why multi-agent?

Single-agent gộp search + analysis + writing vào một prompt → context quá dài, mỗi bước cần temperature khác nhau (search cần chính xác, writing cần sáng tạo hơn), và khó debug khi output kém. Multi-agent tách trách nhiệm rõ ràng: mỗi agent chuyên một việc, có thể retry độc lập, dễ trace từng bước.

## Agent roles

| Agent | Responsibility | Input | Output | Failure mode |
|---|---|---|---|---|
| Supervisor | Quyết định agent nào chạy tiếp, khi nào dừng | ResearchState (current) | route: researcher/analyst/writer/done | LLM trả JSON sai → fallback if/else |
| Researcher | Tìm sources, tóm tắt thành research_notes | query, max_sources | sources[], research_notes | Search API timeout → dùng mock |
| Analyst | Phân tích notes: key claims, viewpoints, weak evidence | research_notes | analysis_notes | research_notes rỗng → error |
| Writer | Viết bài ~500 từ có citations từ notes + analysis | research_notes, analysis_notes, sources | final_answer | LLM timeout → tenacity retry x3 |
| Critic (bonus) | Fact-check citations, phát hiện hallucination | final_answer, source snippets | verdict: pass/warn/fail | JSON parse fail → regex fallback |

## Shared state

| Field | Type | Lý do cần |
|---|---|---|
| `request` | ResearchQuery | Query gốc + config (max_sources, audience) |
| `iteration` | int | Guard max_iterations, tránh infinite loop |
| `route_history` | list[str] | Trace được Supervisor đã quyết định gì |
| `sources` | list[SourceDocument] | Researcher viết → Writer/Critic đọc |
| `research_notes` | str | Researcher → Analyst → Writer |
| `analysis_notes` | str | Analyst → Writer |
| `final_answer` | str | Writer → Critic → CLI output |
| `agent_results` | list[AgentResult] | Lưu content + metadata (tokens, cost) mỗi agent |
| `trace` | list[dict] | Structured events cho debugging |
| `errors` | list[str] | Gom lỗi, không raise ngay để pipeline tiếp tục |

## Routing policy

```
START
  │
  ▼
Supervisor ──(LLM decision)──► researcher ──► Supervisor
  │                            analyst   ──► Supervisor
  │                            writer    ──► Supervisor
  └──────────────────────────► done ──► END

Guard: iteration >= max_iterations → force "done"
Fallback: LLM unavailable → if/else logic
```

## Guardrails

- **Max iterations**: 6 (config: `MAX_ITERATIONS`)
- **Timeout**: 60s per workflow (config: `TIMEOUT_SECONDS`)
- **Retry**: LLMClient dùng `tenacity` retry x3, wait exponential 1-10s
- **Fallback**: Supervisor fallback sang if/else nếu LLM fail; SearchClient fallback sang mock nếu không có API key
- **Validation**: Pydantic schemas tại mọi agent boundary; Supervisor kiểm tra JSON format trước khi parse

## Benchmark plan

| Query | Metric | Expected baseline | Expected multi-agent |
|---|---|---|---|
| "Research GraphRAG state-of-the-art" | quality/10 | ≥6 | ≥7 |
| "Compare single vs multi-agent for customer support" | citation_coverage | ≥0.4 | ≥0.6 |
| "Production guardrails for LLM agents" | latency_seconds | <10s | <30s |
| "RAG vs fine-tuning tradeoffs" | cost_usd | <$0.001 | <$0.003 |
| "LangGraph architecture" | error_rate | 0 | 0 |

Chạy benchmark đầy đủ:
```bash
python -m multi_agent_research_lab.cli benchmark --mode both
```

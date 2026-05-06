# Multi-Agent Research Lab

Hệ thống research gồm **Supervisor + Researcher + Analyst + Writer + Critic** được xây dựng với LangGraph, benchmark so sánh với single-agent baseline.

## Architecture

```text
User Query
   |
   v
Supervisor (LLM-based router)
   |------> Researcher  -> research_notes
   |------> Analyst     -> analysis_notes
   |------> Writer      -> final_answer
   |------> Critic      -> verdict (pass/warn/fail)
   |                         |-> fail: loop back to Writer
   |                         |-> pass/warn: done
   v
Trace (JSONL logs/) + Benchmark Report (reports/)
```

## Cau truc repo

```text
src/multi_agent_research_lab/
├── agents/          # Supervisor, Researcher, Analyst, Writer, Critic
├── core/            # Config, State (Pydantic), Schemas, Errors
├── graph/           # LangGraph workflow
├── services/        # LLMClient (LM Studio / OpenAI), SearchClient (Tavily / mock)
├── evaluation/      # Benchmark (5-dim quality), Report (markdown)
├── observability/   # Tracing -> JSONL + LangSmith
└── cli.py           # CLI: baseline | multi-agent | benchmark
```

## Quickstart

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows
pip install -e ".[dev,llm]"
pip install tavily-python        # neu muon dung Tavily
cp .env.example .env             # dien OPENAI_API_KEY hoac de trong dung LM Studio
```

## Cau hinh `.env`

```bash
# LLM (chon 1 trong 2)
OPENAI_API_KEY=sk-...           # neu dung OpenAI cloud
# LM Studio local tu dong detect o http://localhost:1234/v1

# Search (optional)
TAVILY_API_KEY=tvly-...         # neu khong co -> dung mock sources

# LangSmith tracing (optional)
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_PROJECT=multi-agent-research-lab

# Runtime
MAX_ITERATIONS=6
TIMEOUT_SECONDS=60
```

## Chay

```bash
# Single-agent baseline
python -m multi_agent_research_lab.cli baseline --query "Research GraphRAG"

# Multi-agent workflow (Supervisor -> Researcher -> Analyst -> Writer -> Critic)
python -m multi_agent_research_lab.cli multi-agent --query "Research GraphRAG"

# Benchmark 7 queries, ca 2 mode, luu report tong hop
python -m multi_agent_research_lab.cli benchmark --mode both

# Chi baseline hoac multi-agent
python -m multi_agent_research_lab.cli benchmark --mode baseline
python -m multi_agent_research_lab.cli benchmark --mode multi-agent
```

## Quality Metrics (thang 10)

| Dimension | Max | Mo ta |
|---|---:|---|
| Length | 2 | 380-700 tu = full, penalty neu qua ngan/dai |
| Structure | 2 | Co header, bullet list, Key Takeaways |
| Citations | 2 | So source duoc cite [1],[2],... / tong source |
| Relevance | 2 | Keyword query xuat hien trong answer |
| Completeness | 2 | Pipeline co du research_notes, analysis_notes, final_answer |

Ngoai ra bao cao gom: `latency_seconds`, `cost_usd`, `tokens_in/out`, `critic_verdict`, `unsupported_claims`, `hallucination_risks`.

## Guardrails

| Guard | Co che |
|---|---|
| Max iterations | `MAX_ITERATIONS=6`, Supervisor force "done" khi vuot |
| LLM timeout | `TIMEOUT_SECONDS=60`, moi LLM call chay trong ThreadPoolExecutor |
| LLM retry | tenacity retry x3, wait exponential 1-10s |
| Critic bat buoc | Hard-guard trong Supervisor: khong the "done" neu Critic chua chay |
| Search fallback | Tavily neu co key, mock sources neu khong |
| LLM fallback | LM Studio -> OpenAI -> RuntimeError |
| Route fallback | if/else logic neu LLM router fail |

## Tests

```bash
python -m pytest tests/ -v
```

11 tests cover: Supervisor routing, Critic hard-guard, max_iterations, benchmark scoring, critic metrics.

## Output

- `logs/trace_YYYYMMDD_HHMMSS.jsonl` -- span log moi agent voi input/output
- `reports/benchmark_*.md` -- markdown report voi quality breakdown va averages

## References

- [Building effective agents - Anthropic](https://www.anthropic.com/engineering/building-effective-agents)
- [LangGraph concepts](https://langchain-ai.github.io/langgraph/concepts/)
- [LangSmith tracing](https://docs.smith.langchain.com/)

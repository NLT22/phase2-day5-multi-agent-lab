"""Command-line entrypoint for the lab starter."""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import BenchmarkMetrics, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.services.storage import LocalArtifactStore

app = typer.Typer(help="Multi-Agent Research Lab CLI")
# Force UTF-8 so Vietnamese/Unicode chars don't crash on Windows cp1252 terminals
console = Console(force_terminal=True, highlight=False)
import sys, io  # noqa: E401
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
store = LocalArtifactStore()

_BENCHMARK_QUERIES = [
    "Research GraphRAG state-of-the-art and write a 500-word summary",
    "Compare single-agent and multi-agent workflows for customer support automation",
    "Summarize production guardrails for LLM agents: timeouts, retries, fallbacks",
    "Explain retrieval-augmented generation vs fine-tuning: tradeoffs and use cases",
    "What are the key challenges in deploying LLM agents in enterprise production?",
    "Describe LangGraph architecture and how it handles stateful multi-agent workflows",
    "How do vector databases work and which ones are best for RAG pipelines?",
]


def _init() -> None:
    # load_dotenv() MUST run before anything else so LangSmith SDK reads os.environ
    from dotenv import load_dotenv
    load_dotenv(override=True)

    settings = get_settings()
    configure_logging(settings.log_level)


def _make_baseline_runner(run_label: str):
    """Trả về runner baseline với tên LangSmith trace cụ thể."""
    from multi_agent_research_lab.services.llm_client import LLMClient
    from multi_agent_research_lab.services.search_client import SearchClient
    from multi_agent_research_lab.observability.tracing import trace_span

    llm = LLMClient()
    search = SearchClient()

    def _run(q: str) -> ResearchState:
        state = ResearchState(request=ResearchQuery(query=q))
        trace_name = f"baseline: {q[:50]}"

        # langsmith.trace() cho phép đặt tên động từng lần chạy
        try:
            import langsmith
            ls_ctx = langsmith.trace(name=trace_name, run_type="chain", metadata={"run_label": run_label})
            ls_ctx.__enter__()
        except Exception:
            ls_ctx = None

        try:
            with trace_span("baseline_single_agent", {"run_mode": "baseline"}, {"query": q}) as span:
                sources = search.search(q, max_results=5)
                state.sources = sources
                sources_text = "\n\n".join(f"[{i+1}] {s.title}\n{s.snippet}" for i, s in enumerate(sources))
                system = (
                    "You are an expert researcher and technical writer. "
                    "Given sources and a query, write a clear ~500-word article with citations [1],[2],... "
                    "End with a 'Key Takeaways' bullet list."
                )
                resp = llm.complete(system, f"Query: {q}\n\nSources:\n{sources_text}")
                state.final_answer = resp.content
                state.research_notes = resp.content
                # Record to agent_results so benchmark._total_cost() picks up cost
                from multi_agent_research_lab.core.schemas import AgentName, AgentResult
                state.agent_results.append(AgentResult(
                    agent=AgentName.WRITER,
                    content=resp.content,
                    metadata={
                        "input_tokens": resp.input_tokens,
                        "output_tokens": resp.output_tokens,
                        "cost_usd": resp.cost_usd,
                    },
                ))
                span["output"] = {
                    "word_count": len(resp.content.split()),
                    "answer_preview": resp.content[:120].replace("\n", " "),
                    "input_tokens": resp.input_tokens,
                    "output_tokens": resp.output_tokens,
                    "cost_usd": resp.cost_usd,
                }
        finally:
            if ls_ctx is not None:
                try:
                    ls_ctx.__exit__(None, None, None)
                except Exception:
                    pass
        return state

    return _run


def _make_multi_runner(run_label: str):
    """Trả về runner multi-agent với tên LangSmith trace cụ thể."""
    workflow = MultiAgentWorkflow()

    def _run(q: str) -> ResearchState:
        state = ResearchState(request=ResearchQuery(query=q))
        return workflow.run(state, run_name=f"multi_agent: {q[:50]}")

    return _run


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a single-agent baseline: one LLM call does research + writing."""

    _init()
    runner = _make_baseline_runner("baseline")
    state, metrics = run_benchmark("baseline", query, runner, run_mode="baseline")

    console.print(Panel.fit(state.final_answer or "(no answer)", title="[bold cyan]Single-Agent Baseline[/]"))
    _print_metrics_table([metrics])

    report_md = render_markdown_report([metrics], states=[state])
    path = store.write_text("benchmark_baseline.md", report_md)
    console.print(f"\n[dim]Report saved ->{path}[/dim]")


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the full multi-agent workflow (Supervisor ->Researcher ->Analyst ->Writer)."""

    _init()
    runner = _make_multi_runner("multi_agent")
    state, metrics = run_benchmark("multi_agent", query, runner, run_mode="multi_agent")

    console.print(Panel.fit(state.final_answer or "(no final answer)", title="[bold green]Multi-Agent Result[/]"))

    if state.route_history:
        console.print(f"\n[bold]Agent route:[/] {' ->'.join(state.route_history)}")
    if state.errors:
        console.print(f"\n[bold red]Errors:[/] {state.errors}")

    _print_metrics_table([metrics])

    report_md = render_markdown_report([metrics], states=[state])
    path = store.write_text("benchmark_multi_agent.md", report_md)
    console.print(f"\n[dim]Report saved ->{path}[/dim]")


@app.command()
def benchmark(
    mode: Annotated[str, typer.Option("--mode", "-m", help="baseline | multi-agent | both")] = "both",
    queries_file: Annotated[str | None, typer.Option("--queries", help="File with one query per line")] = None,
) -> None:
    """Run benchmark across multiple queries, save combined report with averages to reports/."""

    _init()

    queries = (
        [q.strip() for q in open(queries_file, encoding="utf-8") if q.strip()]
        if queries_file
        else _BENCHMARK_QUERIES
    )

    console.print(f"\n[bold]Benchmark:[/] {len(queries)} queries | mode={mode}\n")

    baseline_metrics: list[BenchmarkMetrics] = []
    multi_metrics: list[BenchmarkMetrics] = []
    all_metrics: list[BenchmarkMetrics] = []
    all_states: list[ResearchState] = []

    run_modes = []
    if mode in ("baseline", "both"):
        run_modes.append(("baseline", _make_baseline_runner("benchmark_baseline")))
    if mode in ("multi-agent", "both"):
        run_modes.append(("multi_agent", _make_multi_runner("benchmark_multi_agent")))

    for i, q in enumerate(queries, 1):
        console.rule(f"[bold]Query {i}/{len(queries)}[/]")
        console.print(f"[dim]{q}[/dim]\n")

        for run_name, runner in run_modes:
            label = f"{run_name}_q{i}"
            try:
                state, metrics = run_benchmark(label, q, runner, run_mode=run_name)
                all_metrics.append(metrics)
                all_states.append(state)
                if run_name == "baseline":
                    baseline_metrics.append(metrics)
                else:
                    multi_metrics.append(metrics)
                console.print(
                    f"  [green]✓[/green] {run_name} | "
                    f"{metrics.latency_seconds:.1f}s | "
                    f"quality={metrics.quality_score}/10 | "
                    f"${metrics.estimated_cost_usd:.5f}"
                )
            except Exception as e:
                console.print(f"  [red]✗[/red] {run_name} FAILED: {e}")

    from datetime import datetime as dt
    timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
    report_md = render_markdown_report(
        all_metrics, states=all_states,
        baseline_metrics=baseline_metrics,
        multi_metrics=multi_metrics,
    )
    path = store.write_text(f"benchmark_full_{timestamp}.md", report_md)

    console.rule("[bold]Results[/]")
    _print_metrics_table(all_metrics)
    _print_averages(baseline_metrics, multi_metrics)
    console.print(f"\n[bold green]Report saved ->{path}[/bold green]")


def _print_metrics_table(metrics_list: list[BenchmarkMetrics]) -> None:
    table = Table(title="Benchmark Metrics", show_header=True)
    table.add_column("Run")
    table.add_column("Latency (s)", justify="right")
    table.add_column("Cost (USD)", justify="right")
    table.add_column("Quality /10", justify="right")
    table.add_column("Tokens in/out", justify="right")
    table.add_column("Citations", justify="right")
    for m in metrics_list:
        def ex(k: str) -> str:
            return m.notes.split(k + "=")[1].split(" ")[0] if k + "=" in m.notes else "—"
        tokens = f"{ex('tokens_in')}/{ex('tokens_out')}"
        table.add_row(
            m.run_name,
            f"{m.latency_seconds:.2f}",
            f"${m.estimated_cost_usd:.5f}" if m.estimated_cost_usd is not None else "—",
            f"{m.quality_score:.1f}" if m.quality_score is not None else "—",
            tokens,
            ex("citations"),
        )
    console.print(table)


def _print_averages(
    baseline_metrics: list[BenchmarkMetrics],
    multi_metrics: list[BenchmarkMetrics],
) -> None:
    if not baseline_metrics and not multi_metrics:
        return

    def avg(lst: list[BenchmarkMetrics], attr: str) -> str:
        vals = [getattr(m, attr) for m in lst if getattr(m, attr) is not None]
        if not vals:
            return "—"
        mean = sum(vals) / len(vals)
        fmt = ".6f" if attr == "estimated_cost_usd" else ".2f"
        return f"{mean:{fmt}}"

    table = Table(title="Averages: Baseline vs Multi-Agent", show_header=True)
    table.add_column("Metric")
    table.add_column("Baseline (avg)", justify="right")
    table.add_column("Multi-Agent (avg)", justify="right")
    table.add_column("Winner", justify="center")

    rows = [
        ("Latency (s)", "latency_seconds", "lower"),
        ("Cost (USD)", "estimated_cost_usd", "lower"),
        ("Quality /10", "quality_score", "higher"),
    ]
    for label, attr, better in rows:
        b = avg(baseline_metrics, attr)
        m = avg(multi_metrics, attr)
        try:
            winner = "baseline" if (
                (better == "lower" and float(b) < float(m)) or
                (better == "higher" and float(b) > float(m))
            ) else "multi-agent"
        except ValueError:
            winner = "—"
        table.add_row(label, b, m, winner)

    console.print(table)


if __name__ == "__main__":
    app()

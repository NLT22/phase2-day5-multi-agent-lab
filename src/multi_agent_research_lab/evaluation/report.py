"""Benchmark report rendering."""

from __future__ import annotations

from datetime import datetime

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState


def _ex(notes: str, key: str, default: str = "—") -> str:
    """Extract key=value from notes string."""
    if key + "=" not in notes:
        return default
    return notes.split(key + "=")[1].split(" ")[0]


def _avg(lst: list[BenchmarkMetrics], attr: str) -> str:
    vals = [getattr(m, attr) for m in lst if getattr(m, attr) is not None]
    if not vals:
        return "—"
    mean = sum(vals) / len(vals)
    # cost_usd needs more precision than latency/quality
    fmt = ".6f" if attr == "estimated_cost_usd" else ".2f"
    return f"{mean:{fmt}}"


def render_markdown_report(
    metrics: list[BenchmarkMetrics],
    states: list[ResearchState] | None = None,
    baseline_metrics: list[BenchmarkMetrics] | None = None,
    multi_metrics: list[BenchmarkMetrics] | None = None,
) -> str:
    lines: list[str] = [
        "# Benchmark Report",
        f"\n_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n",
    ]

    # --- Summary table ---
    lines += [
        "## Metrics Summary\n",
        "| Run | Latency (s) | Cost (USD) | Quality /10 | Words | Tokens in/out | Citations | Critic | Issues |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|---:|",
    ]
    for m in metrics:
        cost = "—" if m.estimated_cost_usd is None else f"${m.estimated_cost_usd:.5f}"
        quality = "—" if m.quality_score is None else f"{m.quality_score:.1f}"
        tok_in = _ex(m.notes, "tokens_in")
        tok_out = _ex(m.notes, "tokens_out")
        tokens = f"{tok_in}/{tok_out}" if tok_in != "—" else "—"
        verdict = _ex(m.notes, "critic_verdict")
        unsupported = _ex(m.notes, "unsupported_claims")
        cit_err = _ex(m.notes, "citation_errors")
        hall = _ex(m.notes, "hallucination_risks")
        try:
            issues = int(unsupported) + int(cit_err) + int(hall)
        except ValueError:
            issues = "—"
        lines.append(
            f"| {m.run_name} | {m.latency_seconds:.2f} | {cost} | {quality}"
            f" | {_ex(m.notes, 'words')} | {tokens}"
            f" | {_ex(m.notes, 'citations')} | {verdict} | {issues} |"
        )

    # --- Quality breakdown ---
    lines += ["\n## Quality Score Breakdown (max 2 each)\n",
              "| Run | Length | Structure | Citations | Relevance | Completeness | **Total** |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for m in metrics:
        lines.append(
            f"| {m.run_name}"
            f" | {_ex(m.notes, 'q_length')}"
            f" | {_ex(m.notes, 'q_structure')}"
            f" | {_ex(m.notes, 'q_citations')}"
            f" | {_ex(m.notes, 'q_relevance')}"
            f" | {_ex(m.notes, 'q_completeness')}"
            f" | **{m.quality_score or '—'}** |"
        )

    lines.append("\n> Length: 0-2 | Structure: 0-2 | Citations: 0-2 | Relevance: 0-2 | Completeness: 0-2\n")

    # --- Averages comparison ---
    if baseline_metrics or multi_metrics:
        b = baseline_metrics or []
        m = multi_metrics or []
        lines += [
            "## Baseline vs Multi-Agent (averages)\n",
            "| Metric | Baseline | Multi-Agent | Winner |",
            "|---|---:|---:|:---:|",
        ]
        comparisons = [
            ("Latency (s)", "latency_seconds", "lower"),
            ("Cost (USD)", "estimated_cost_usd", "lower"),
            ("Quality /10", "quality_score", "higher"),
        ]
        for label, attr, better in comparisons:
            bv = _avg(b, attr)
            mv = _avg(m, attr)
            try:
                winner = "baseline" if (
                    (better == "lower" and float(bv) < float(mv)) or
                    (better == "higher" and float(bv) > float(mv))
                ) else "multi-agent"
            except ValueError:
                winner = "—"
            lines.append(f"| {label} | {bv} | {mv} | {winner} |")
        lines.append("")

    # --- Agent routes ---
    if states:
        lines.append("## Agent Routes\n")
        for i, state in enumerate(states):
            run = metrics[i].run_name if i < len(metrics) else f"run_{i}"
            route = " → ".join(state.route_history) if state.route_history else "(single agent)"
            lines.append(f"- **{run}**: {route}")

        # --- Failure analysis ---
        lines.append("\n## Failure Analysis\n")
        has_errors = False
        for i, state in enumerate(states):
            if state.errors:
                has_errors = True
                run = metrics[i].run_name if i < len(metrics) else f"run_{i}"
                lines.append(f"### {run}")
                for err in state.errors:
                    lines.append(f"- {err}")
        if not has_errors:
            lines.append("_No errors recorded._")

    return "\n".join(lines) + "\n"

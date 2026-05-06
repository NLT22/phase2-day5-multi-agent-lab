"""Benchmark runner — đo latency, cost, và quality đa chiều."""

from __future__ import annotations

import re
from time import perf_counter
from typing import Callable

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState

Runner = Callable[[str], ResearchState]

_TARGET_WORDS = 500
_IDEAL_MIN = 380
_IDEAL_MAX = 700


# ---------------------------------------------------------------------------
# Sub-scorers (mỗi cái trả về float trong khoảng đã ghi chú)
# ---------------------------------------------------------------------------

def _score_length(answer: str) -> float:
    """0-2: penalty nếu quá ngắn (<200 từ) hoặc quá dài (>900 từ), full nếu trong ideal range."""
    words = len(answer.split())
    if words < 100:
        return 0.0
    if words < 200:
        return 0.5
    if _IDEAL_MIN <= words <= _IDEAL_MAX:
        return 2.0
    if words < _IDEAL_MIN:
        return round(1.0 + (words - 200) / (_IDEAL_MIN - 200), 1)
    # quá dài: giảm dần từ 2.0 → 1.0 khi vượt _IDEAL_MAX
    return round(max(1.0, 2.0 - (words - _IDEAL_MAX) / 400), 1)


def _score_structure(answer: str) -> float:
    """0-2: kiểm tra cấu trúc markdown (headers, bullets, key takeaways)."""
    score = 0.0
    if re.search(r"^#{1,3} ", answer, re.MULTILINE):
        score += 0.7   # có header
    if re.search(r"^[-*] ", answer, re.MULTILINE):
        score += 0.7   # có bullet list
    if re.search(r"key takeaway|takeaway|tóm tắt", answer, re.IGNORECASE):
        score += 0.6   # có key takeaways section
    return round(min(score, 2.0), 1)


def _score_citations(answer: str, source_count: int) -> float:
    """0-2: số citation unique / source có sẵn, penalty nếu không có citation nào."""
    if not answer:
        return 0.0
    cited = set(re.findall(r"\[(\d+)\]", answer))
    if not cited:
        return 0.0
    coverage = len(cited) / max(source_count, 1)
    return round(min(coverage * 2.0, 2.0), 1)


def _score_relevance(answer: str, query: str) -> float:
    """0-2: keyword overlap giữa query và answer (case-insensitive)."""
    if not answer:
        return 0.0
    # lấy các từ có nghĩa trong query (>3 ký tự)
    query_words = {w.lower() for w in re.findall(r"\b\w{4,}\b", query)}
    if not query_words:
        return 1.0
    answer_lower = answer.lower()
    matched = sum(1 for w in query_words if w in answer_lower)
    ratio = matched / len(query_words)
    return round(min(ratio * 2.0, 2.0), 1)


def _score_completeness(state: ResearchState) -> float:
    """0-2: kiểm tra pipeline có đủ output (research, analysis, final_answer)."""
    score = 0.0
    if state.research_notes:
        score += 0.6
    if state.analysis_notes:
        score += 0.7
    if state.final_answer and len(state.final_answer.split()) >= 100:
        score += 0.7
    return round(min(score, 2.0), 1)


def _quality_score(state: ResearchState) -> tuple[float, dict[str, float]]:
    """Tổng hợp 5 sub-scores → 0-10. Trả về (total, breakdown)."""
    answer = state.final_answer or ""
    query = state.request.query
    n_sources = len(state.sources)

    breakdown = {
        "length":       _score_length(answer),       # max 2
        "structure":    _score_structure(answer),     # max 2
        "citations":    _score_citations(answer, n_sources),  # max 2
        "relevance":    _score_relevance(answer, query),      # max 2
        "completeness": _score_completeness(state),           # max 2
    }
    total = round(sum(breakdown.values()), 1)
    return total, breakdown


def _total_cost(state: ResearchState) -> float:
    total = 0.0
    for result in state.agent_results:
        cost = result.metadata.get("cost_usd")
        if cost is not None:
            total += float(cost)
    return total


def _citation_coverage(state: ResearchState) -> float:
    if not state.sources or not state.final_answer:
        return 0.0
    cited = set(re.findall(r"\[(\d+)\]", state.final_answer))
    return round(len(cited) / len(state.sources), 2)


def _error_rate(state: ResearchState) -> float:
    denom = max(state.iteration, 1)
    return round(len(state.errors) / denom, 2)


def _critic_summary(state: ResearchState) -> dict:
    """Extract critic verdict from the FINAL critic run (after the last writer run)."""
    last_critic_idx: int | None = None
    last_writer_idx: int | None = None
    for i, r in enumerate(state.agent_results):
        if r.agent == "critic":
            last_critic_idx = i
        if r.agent == "writer":
            last_writer_idx = i

    if last_critic_idx is None:
        return {"verdict": "not_run", "unsupported_claims": 0, "citation_errors": 0, "hallucination_risks": 0}
    if last_writer_idx is not None and last_writer_idx > last_critic_idx:
        return {"verdict": "stale", "unsupported_claims": 0, "citation_errors": 0, "hallucination_risks": 0}

    r = state.agent_results[last_critic_idx]
    return {
        "verdict": r.metadata.get("verdict", "n/a"),
        "unsupported_claims": r.metadata.get("unsupported_claims", 0),
        "citation_errors": r.metadata.get("citation_errors", 0),
        "hallucination_risks": r.metadata.get("hallucination_risks", 0),
    }


def _token_summary(state: ResearchState) -> dict[str, int]:
    total_in = total_out = 0
    for r in state.agent_results:
        total_in += r.metadata.get("input_tokens") or 0
        total_out += r.metadata.get("output_tokens") or 0
    return {"input_tokens": total_in, "output_tokens": total_out}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_benchmark(
    run_name: str,
    query: str,
    runner: Runner,
    run_mode: str = "unknown",
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Chạy runner, đo performance, trả về state + metrics."""
    started = perf_counter()
    state = runner(query)
    latency = perf_counter() - started

    quality, breakdown = _quality_score(state)
    cost = _total_cost(state)
    citations = _citation_coverage(state)
    errors = _error_rate(state)
    tokens = _token_summary(state)
    words = len((state.final_answer or "").split())
    critic = _critic_summary(state)

    notes = (
        f"run_mode={run_mode} "
        f"citations={citations:.2f} "
        f"errors={errors:.2f} "
        f"words={words} "
        f"tokens_in={tokens['input_tokens']} "
        f"tokens_out={tokens['output_tokens']} "
        f"q_length={breakdown['length']} "
        f"q_structure={breakdown['structure']} "
        f"q_citations={breakdown['citations']} "
        f"q_relevance={breakdown['relevance']} "
        f"q_completeness={breakdown['completeness']} "
        f"critic_verdict={critic['verdict']} "
        f"unsupported_claims={critic['unsupported_claims']} "
        f"citation_errors={critic['citation_errors']} "
        f"hallucination_risks={critic['hallucination_risks']}"
    )

    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=round(latency, 3),
        estimated_cost_usd=round(cost, 6),
        quality_score=quality,
        notes=notes,
    )
    return state, metrics

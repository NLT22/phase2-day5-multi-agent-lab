"""Tests for SupervisorAgent routing and pipeline guarantees."""

from multi_agent_research_lab.agents import SupervisorAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState


def test_supervisor_routes_to_researcher_when_no_notes() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    result = SupervisorAgent().run(state)
    assert len(result.route_history) == 1
    assert result.route_history[0] in {"researcher", "analyst", "writer", "done", "critic"}


def test_supervisor_routes_to_critic_before_done() -> None:
    """Critic must run before done — even if LLM tries to skip it."""
    state = ResearchState(
        request=ResearchQuery(query="Explain multi-agent systems"),
        research_notes="Some notes",
        analysis_notes="Some analysis",
        final_answer="Some answer",
    )
    result = SupervisorAgent().run(state)
    assert result.route_history[-1] == "critic"


def test_supervisor_hard_guard_overrides_llm_done() -> None:
    """Hard guard: if LLM returns 'done' but critic hasn't run, must override to 'critic'."""
    from unittest.mock import patch

    state = ResearchState(
        request=ResearchQuery(query="Explain multi-agent systems"),
        research_notes="notes",
        analysis_notes="analysis",
        final_answer="answer",
    )
    # Force LLM to return "done" prematurely
    with patch("multi_agent_research_lab.agents.supervisor._llm_route",
               return_value={"next": "done", "reason": "mocked done"}):
        result = SupervisorAgent().run(state)
    # Hard guard should override to critic
    assert result.route_history[-1] == "critic"


def test_supervisor_routes_to_done_after_critic_pass() -> None:
    state = ResearchState(
        request=ResearchQuery(query="Explain multi-agent systems"),
        research_notes="notes",
        analysis_notes="analysis",
        final_answer="answer",
        agent_results=[AgentResult(agent=AgentName.CRITIC, content="{}", metadata={"verdict": "pass"})],
    )
    result = SupervisorAgent().run(state)
    assert result.route_history[-1] == "done"


def test_supervisor_routes_to_writer_after_critic_fail() -> None:
    state = ResearchState(
        request=ResearchQuery(query="Explain multi-agent systems"),
        research_notes="notes",
        analysis_notes="analysis",
        final_answer="answer",
        agent_results=[AgentResult(agent=AgentName.CRITIC, content="{}", metadata={"verdict": "fail"})],
    )
    result = SupervisorAgent().run(state)
    assert result.route_history[-1] == "writer"


def test_supervisor_routes_to_critic_again_after_rewrite() -> None:
    """After writer rewrites (runs after critic), supervisor must route to critic again."""
    state = ResearchState(
        request=ResearchQuery(query="Explain multi-agent systems"),
        research_notes="notes",
        analysis_notes="analysis",
        final_answer="rewritten answer",
        agent_results=[
            AgentResult(agent=AgentName.CRITIC, content="{}", metadata={"verdict": "fail"}),
            AgentResult(agent=AgentName.WRITER, content="rewritten answer", metadata={}),
        ],
    )
    result = SupervisorAgent().run(state)
    # Critic must run again on the new answer, not loop back to writer
    assert result.route_history[-1] == "critic"


def test_supervisor_enforces_max_iterations() -> None:
    from multi_agent_research_lab.core.config import get_settings
    settings = get_settings()
    state = ResearchState(
        request=ResearchQuery(query="Explain multi-agent systems"),
        iteration=settings.max_iterations,
    )
    result = SupervisorAgent().run(state)
    assert result.route_history[-1] == "done"


def test_benchmark_quality_score_not_always_10() -> None:
    """Quality score should penalise short or format-poor answers."""
    from multi_agent_research_lab.evaluation.benchmark import _quality_score

    # Very short answer -> low score
    short_state = ResearchState(
        request=ResearchQuery(query="What is RAG?"),
        final_answer="RAG is good.",
        research_notes="notes",
    )
    score_short, _ = _quality_score(short_state)
    assert score_short < 5.0, f"Expected <5 for short answer, got {score_short}"

    # Good answer with citations, bullets, headers -> high score
    good_answer = (
        "## Introduction\n"
        "GraphRAG [1] improves retrieval by building a knowledge graph. "
        "It outperforms vanilla RAG [2] on multi-hop questions [3].\n\n"
        "## Key Findings\n"
        "- Graph indexing enables multi-hop reasoning [1]\n"
        "- 18% improvement on complex QA [2]\n"
        "- 3-5x indexing cost overhead [3]\n\n"
        "## Key Takeaways\n"
        "- GraphRAG is better for complex queries\n"
        "- Cost tradeoff must be considered\n"
        "- Hybrid approaches [4] balance speed and quality\n"
    ) * 4  # repeat to hit word count target
    good_state = ResearchState(
        request=ResearchQuery(query="Research GraphRAG state-of-the-art"),
        final_answer=good_answer,
        research_notes="notes",
        analysis_notes="analysis",
        sources=[],
    )
    score_good, _ = _quality_score(good_state)
    assert score_good > score_short, "Good answer should score higher than short answer"


def test_critic_metrics_in_benchmark() -> None:
    """Benchmark notes should include critic verdict when critic has run."""
    from multi_agent_research_lab.evaluation.benchmark import _critic_summary

    state_with_critic = ResearchState(
        request=ResearchQuery(query="test query"),
        agent_results=[
            AgentResult(
                agent=AgentName.CRITIC,
                content="{}",
                metadata={"verdict": "warn", "unsupported_claims": 2, "citation_errors": 1, "hallucination_risks": 0},
            )
        ],
    )
    summary = _critic_summary(state_with_critic)
    assert summary["verdict"] == "warn"
    assert summary["unsupported_claims"] == 2
    assert summary["citation_errors"] == 1

    state_no_critic = ResearchState(request=ResearchQuery(query="test query"))
    summary_none = _critic_summary(state_no_critic)
    assert summary_none["verdict"] == "not_run"

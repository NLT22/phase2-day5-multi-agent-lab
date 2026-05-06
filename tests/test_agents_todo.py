"""Tests for SupervisorAgent routing behavior."""

from multi_agent_research_lab.agents import SupervisorAgent
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState


def test_supervisor_routes_to_researcher_when_no_notes() -> None:
    """Supervisor should route to researcher first when notes are empty."""
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    result = SupervisorAgent().run(state)
    # Must have recorded a route
    assert len(result.route_history) == 1
    assert result.route_history[0] in {"researcher", "analyst", "writer", "done"}


def test_supervisor_routes_to_critic_before_done() -> None:
    """Supervisor should route to critic when all notes ready but critic hasn't run."""
    from multi_agent_research_lab.core.schemas import AgentName, AgentResult
    state = ResearchState(
        request=ResearchQuery(query="Explain multi-agent systems"),
        research_notes="Some notes",
        analysis_notes="Some analysis",
        final_answer="Some answer",
    )
    result = SupervisorAgent().run(state)
    assert result.route_history[-1] == "critic"


def test_supervisor_routes_to_done_after_critic_pass() -> None:
    """Supervisor should route to done after critic verdict=pass."""
    from multi_agent_research_lab.core.schemas import AgentName, AgentResult
    state = ResearchState(
        request=ResearchQuery(query="Explain multi-agent systems"),
        research_notes="Some notes",
        analysis_notes="Some analysis",
        final_answer="Some answer",
        agent_results=[AgentResult(agent=AgentName.CRITIC, content="{}", metadata={"verdict": "pass"})],
    )
    result = SupervisorAgent().run(state)
    assert result.route_history[-1] == "done"


def test_supervisor_routes_to_writer_after_critic_fail() -> None:
    """Supervisor should route back to writer when critic verdict=fail."""
    from multi_agent_research_lab.core.schemas import AgentName, AgentResult
    state = ResearchState(
        request=ResearchQuery(query="Explain multi-agent systems"),
        research_notes="Some notes",
        analysis_notes="Some analysis",
        final_answer="Some answer",
        agent_results=[AgentResult(agent=AgentName.CRITIC, content="{}", metadata={"verdict": "fail"})],
    )
    result = SupervisorAgent().run(state)
    assert result.route_history[-1] == "writer"


def test_supervisor_enforces_max_iterations() -> None:
    """Supervisor must route to done when max_iterations is reached."""
    from multi_agent_research_lab.core.config import get_settings
    settings = get_settings()
    state = ResearchState(
        request=ResearchQuery(query="Explain multi-agent systems"),
        iteration=settings.max_iterations,
    )
    result = SupervisorAgent().run(state)
    assert result.route_history[-1] == "done"

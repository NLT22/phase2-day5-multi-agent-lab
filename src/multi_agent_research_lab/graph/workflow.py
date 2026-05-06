"""LangGraph workflow — Supervisor -> Researcher -> Analyst -> Writer -> Critic -> done."""

from __future__ import annotations

import logging

from langgraph.graph import END, StateGraph

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.critic import CriticAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import SupervisorAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.state import ResearchState

logger = logging.getLogger(__name__)


def _route_fn(state: ResearchState) -> str:
    """Read the last route recorded by SupervisorAgent."""
    if not state.route_history:
        return "done"
    return state.route_history[-1]


class MultiAgentWorkflow:
    """Builds and runs the multi-agent LangGraph graph."""

    def __init__(self) -> None:
        self._supervisor = SupervisorAgent()
        self._researcher = ResearcherAgent()
        self._analyst = AnalystAgent()
        self._writer = WriterAgent()
        self._critic = CriticAgent()
        self._graph = self.build()

    def build(self):
        graph = StateGraph(ResearchState)

        graph.add_node("supervisor", self._supervisor.run)
        graph.add_node("researcher", self._researcher.run)
        graph.add_node("analyst", self._analyst.run)
        graph.add_node("writer", self._writer.run)
        graph.add_node("critic", self._critic.run)

        graph.set_entry_point("supervisor")

        graph.add_conditional_edges(
            "supervisor",
            _route_fn,
            {
                "researcher": "researcher",
                "analyst": "analyst",
                "writer": "writer",
                "critic": "critic",
                "done": END,
            },
        )

        # All workers loop back to Supervisor
        graph.add_edge("researcher", "supervisor")
        graph.add_edge("analyst", "supervisor")
        graph.add_edge("writer", "supervisor")
        graph.add_edge("critic", "supervisor")

        return graph.compile()

    def run(self, state: ResearchState, run_name: str | None = None) -> ResearchState:
        logger.info("MultiAgentWorkflow starting for query: %s", state.request.query)
        config = {"run_name": run_name or f"multi_agent: {state.request.query[:50]}"}
        result = self._graph.invoke(state, config=config)
        if isinstance(result, dict):
            return ResearchState(**result)
        return result

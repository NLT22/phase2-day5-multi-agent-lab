"""Analyst agent — extracts insights from research notes."""

from __future__ import annotations

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a critical analyst. Given research notes, produce a structured analysis (200-300 words) that:
1. Key Claims: list the 3-5 most important facts or arguments.
2. Viewpoints: compare differing perspectives if present.
3. Evidence Quality: flag any claims that lack strong evidence or seem speculative.
Return only the analysis using these three sections with headers.
"""


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def __init__(self) -> None:
        self._llm = LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        with trace_span(
            "analyst",
            attributes={"run_mode": "multi_agent", "iteration": state.iteration},
            input_data={
                "query": state.request.query,
                "research_notes_chars": len(state.research_notes or ""),
                "research_notes_preview": (state.research_notes or "")[:120].replace("\n", " "),
            },
        ) as span:
            try:
                if not state.research_notes:
                    state.errors.append("AnalystAgent: no research_notes to analyse")
                    return state

                user_prompt = f"Query: {state.request.query}\n\nResearch Notes:\n{state.research_notes}"
                response = self._llm.complete(_SYSTEM_PROMPT, user_prompt)

                state.analysis_notes = response.content
                state.agent_results.append(
                    AgentResult(
                        agent=AgentName.ANALYST,
                        content=response.content,
                        metadata={
                            "input_tokens": response.input_tokens,
                            "output_tokens": response.output_tokens,
                            "cost_usd": response.cost_usd,
                        },
                    )
                )
                span["output"] = {
                    "analysis_chars": len(response.content),
                    "analysis_preview": response.content[:120].replace("\n", " "),
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                }
                state.add_trace_event("analyst_done", {"analysis_chars": len(response.content)})
            except Exception as e:
                error_msg = f"AnalystAgent error: {e}"
                state.errors.append(error_msg)
                logger.error(error_msg)
                span["error"] = error_msg

        return state

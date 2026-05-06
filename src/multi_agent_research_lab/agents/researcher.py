"""Researcher agent — collects sources and synthesizes research notes."""

from __future__ import annotations

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a research assistant. Given a set of source documents and a query, write concise \
research notes (300-400 words) that:
1. Summarise the key facts and findings from the sources.
2. Note any conflicting information between sources.
3. Reference sources by their index number [1], [2], etc.
Return only the research notes, no preamble.
"""


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(self) -> None:
        self._llm = LLMClient()
        self._search = SearchClient()

    def run(self, state: ResearchState) -> ResearchState:
        with trace_span(
            "researcher",
            attributes={"run_mode": "multi_agent", "iteration": state.iteration},
            input_data={"query": state.request.query, "max_sources": state.request.max_sources},
        ) as span:
            try:
                sources = self._search.search(state.request.query, max_results=state.request.max_sources)
                state.sources = sources

                sources_text = "\n\n".join(
                    f"[{i+1}] {s.title}\n{s.snippet}" for i, s in enumerate(sources)
                )
                user_prompt = f"Query: {state.request.query}\n\nSources:\n{sources_text}"
                response = self._llm.complete(_SYSTEM_PROMPT, user_prompt)

                state.research_notes = response.content
                state.agent_results.append(
                    AgentResult(
                        agent=AgentName.RESEARCHER,
                        content=response.content,
                        metadata={
                            "sources_count": len(sources),
                            "input_tokens": response.input_tokens,
                            "output_tokens": response.output_tokens,
                            "cost_usd": response.cost_usd,
                        },
                    )
                )
                span["output"] = {
                    "sources_count": len(sources),
                    "notes_chars": len(response.content),
                    "notes_preview": response.content[:120].replace("\n", " "),
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                }
                state.add_trace_event("researcher_done", {"sources": len(sources)})
            except Exception as e:
                error_msg = f"ResearcherAgent error: {e}"
                state.errors.append(error_msg)
                logger.error(error_msg)
                span["error"] = error_msg

        return state

"""Writer agent — synthesises final answer from research and analysis."""

from __future__ import annotations

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a technical writer. Using the provided research notes and analysis, write a clear, \
well-structured response of approximately 500 words aimed at {audience}.

Requirements:
- Use plain prose with at most 3 markdown headers.
- Cite sources as [1], [2], etc. wherever you use a specific fact.
- End with a brief "Key Takeaways" bullet list (3-5 bullets).
Return only the final article.
"""


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def __init__(self) -> None:
        self._llm = LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        with trace_span(
            "writer",
            attributes={"run_mode": "multi_agent", "iteration": state.iteration},
            input_data={
                "query": state.request.query,
                "audience": state.request.audience,
                "research_notes_chars": len(state.research_notes or ""),
                "analysis_notes_chars": len(state.analysis_notes or ""),
                "sources_count": len(state.sources),
            },
        ) as span:
            try:
                research = state.research_notes or "(no research notes)"
                analysis = state.analysis_notes or "(no analysis notes)"
                sources_list = "\n".join(
                    f"[{i+1}] {s.title} — {s.url or 'no url'}" for i, s in enumerate(state.sources)
                )

                system = _SYSTEM_PROMPT.format(audience=state.request.audience)
                user_prompt = (
                    f"Query: {state.request.query}\n\n"
                    f"Research Notes:\n{research}\n\n"
                    f"Analysis:\n{analysis}\n\n"
                    f"Sources:\n{sources_list}"
                )
                response = self._llm.complete(system, user_prompt)

                state.final_answer = response.content
                word_count = len(response.content.split())
                state.agent_results.append(
                    AgentResult(
                        agent=AgentName.WRITER,
                        content=response.content,
                        metadata={
                            "word_count": word_count,
                            "input_tokens": response.input_tokens,
                            "output_tokens": response.output_tokens,
                            "cost_usd": response.cost_usd,
                        },
                    )
                )
                span["output"] = {
                    "word_count": word_count,
                    "answer_preview": response.content[:120].replace("\n", " "),
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                }
                state.add_trace_event("writer_done", {"word_count": word_count})
            except Exception as e:
                error_msg = f"WriterAgent error: {e}"
                state.errors.append(error_msg)
                logger.error(error_msg)
                span["error"] = error_msg

        return state

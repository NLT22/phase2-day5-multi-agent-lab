"""Critic agent — fact-check và hallucination detection trên final_answer."""

from __future__ import annotations

import logging
import re

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a fact-checking editor. Given a research article and the original source snippets, review the article for:
1. Unsupported claims: statements not backed by any source snippet.
2. Citation accuracy: does each [N] citation actually match what source N says?
3. Hallucination risk: claims that contradict the sources or seem invented.

Respond with a JSON object:
{
  "verdict": "pass" | "warn" | "fail",
  "unsupported_claims": ["..."],
  "citation_errors": ["..."],
  "hallucination_risks": ["..."],
  "summary": "one sentence overall assessment"
}
Return only the JSON, no markdown.
"""


class CriticAgent(BaseAgent):
    """Optional fact-checking and hallucination-detection agent."""

    name = "critic"

    def __init__(self) -> None:
        self._llm = LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        with trace_span(
            "critic",
            attributes={"run_mode": "multi_agent", "iteration": state.iteration},
            input_data={
                "final_answer_chars": len(state.final_answer or ""),
                "sources_count": len(state.sources),
            },
        ) as span:
            try:
                if not state.final_answer:
                    state.errors.append("CriticAgent: no final_answer to review")
                    return state

                sources_text = "\n\n".join(
                    f"[{i+1}] {s.title}\n{s.snippet}" for i, s in enumerate(state.sources)
                )
                user_prompt = (
                    f"Article to review:\n{state.final_answer}\n\n"
                    f"Source snippets:\n{sources_text}"
                )
                response = self._llm.complete(_SYSTEM_PROMPT, user_prompt)

                # Parse verdict
                import json
                try:
                    result = json.loads(response.content.strip())
                except json.JSONDecodeError:
                    # Fallback: extract verdict with regex
                    m = re.search(r'"verdict"\s*:\s*"(\w+)"', response.content)
                    result = {"verdict": m.group(1) if m else "warn", "summary": response.content[:200]}

                verdict = result.get("verdict", "warn")
                if verdict == "fail":
                    state.errors.append(f"CriticAgent: verdict=fail — {result.get('summary', '')}")

                state.agent_results.append(
                    AgentResult(
                        agent=AgentName.CRITIC,
                        content=response.content,
                        metadata={
                            "verdict": verdict,
                            "unsupported_claims": len(result.get("unsupported_claims", [])),
                            "citation_errors": len(result.get("citation_errors", [])),
                            "hallucination_risks": len(result.get("hallucination_risks", [])),
                            "input_tokens": response.input_tokens,
                            "output_tokens": response.output_tokens,
                            "cost_usd": response.cost_usd,
                        },
                    )
                )
                span["output"] = {
                    "verdict": verdict,
                    "summary": result.get("summary", ""),
                    "issues": (
                        len(result.get("unsupported_claims", [])) +
                        len(result.get("citation_errors", [])) +
                        len(result.get("hallucination_risks", []))
                    ),
                }
                state.add_trace_event("critic_done", {"verdict": verdict})
                logger.info("Critic: verdict=%s | summary=%s", verdict, result.get("summary", ""))
            except Exception as e:
                error_msg = f"CriticAgent error: {e}"
                state.errors.append(error_msg)
                logger.error(error_msg)
                span["error"] = error_msg

        return state

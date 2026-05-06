"""Supervisor / router agent — uses LLM to decide which agent runs next."""

from __future__ import annotations

import json
import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span

logger = logging.getLogger(__name__)

_VALID_ROUTES = {"researcher", "analyst", "writer", "critic", "done"}

_SYSTEM_PROMPT = """\
You are a workflow supervisor for a multi-agent research system.
Your job is to decide which agent should run next based on the current state of the research.

Available agents:
- researcher: gathers sources and writes research notes
- analyst: analyses research notes for key claims, viewpoints, and evidence quality
- writer: synthesises a final answer from notes and analysis
- critic: fact-checks the final answer, detects hallucinations, verifies citations
- done: stop the workflow (all required outputs are complete)

Routing rules:
1. If research_notes is missing -> researcher
2. If analysis_notes is missing -> analyst
3. If final_answer is missing -> writer
4. If final_answer exists but critic has NOT run yet -> critic
5. If critic ran and verdict is "fail" -> writer (rewrite)
6. If critic ran and verdict is "pass" or "warn" -> done

Respond ONLY with a JSON object in this exact format (no markdown, no explanation):
{"next": "<agent_or_done>", "reason": "<one sentence>"}
"""


def _critic_verdict(state: ResearchState) -> str | None:
    """Return critic's verdict only if it ran AFTER the most recent writer run.

    If writer has rewritten since the last critic run, treat critic as not-yet-run
    so Supervisor routes to critic again on the new answer.
    """
    last_critic_idx: int | None = None
    last_writer_idx: int | None = None
    for i, r in enumerate(state.agent_results):
        if r.agent == "critic":
            last_critic_idx = i
        if r.agent == "writer":
            last_writer_idx = i

    if last_critic_idx is None:
        return None
    # Critic result is stale if writer ran after it
    if last_writer_idx is not None and last_writer_idx > last_critic_idx:
        return None
    return state.agent_results[last_critic_idx].metadata.get("verdict")


def _done_status(state: ResearchState, reason: str) -> str:
    """Return a precise terminal status for benchmark/report consumers."""
    verdict = _critic_verdict(state)
    if "max_iterations" in reason:
        return "max_iterations"
    if verdict == "pass":
        return "passed"
    if verdict == "warn":
        return "warned"
    if verdict == "fail":
        return "failed"
    return "incomplete"


def _llm_route(state: ResearchState) -> dict:
    """Ask LLM to decide the next route. Returns {"next": ..., "reason": ...}."""
    from multi_agent_research_lab.services.llm_client import LLMClient

    verdict = _critic_verdict(state)
    llm = LLMClient()
    user_prompt = (
        f"Query: {state.request.query}\n"
        f"Iteration: {state.iteration}\n"
        f"research_notes present: {bool(state.research_notes)}\n"
        f"analysis_notes present: {bool(state.analysis_notes)}\n"
        f"final_answer present: {bool(state.final_answer)}\n"
        f"critic has run: {verdict is not None}\n"
        f"critic verdict: {verdict or 'n/a'}\n"
        f"errors so far: {len(state.errors)}\n"
    )
    response = llm.complete(_SYSTEM_PROMPT, user_prompt)
    parsed = json.loads(response.content.strip())
    if parsed.get("next") not in _VALID_ROUTES:
        raise ValueError(f"LLM returned invalid route: {parsed.get('next')}")
    return parsed


def _fallback_route(state: ResearchState) -> dict:
    """Deterministic if/else routing used when no LLM is available."""
    if not state.research_notes:
        return {"next": "researcher", "reason": "research_notes missing"}
    if not state.analysis_notes:
        return {"next": "analyst", "reason": "analysis_notes missing"}
    if not state.final_answer:
        return {"next": "writer", "reason": "final_answer missing"}
    verdict = _critic_verdict(state)
    if verdict is None:
        return {"next": "critic", "reason": "final_answer ready, critic has not run"}
    if verdict == "fail":
        settings = get_settings()
        if state.rewrite_count >= settings.max_rewrites:
            return {
                "next": "done",
                "reason": f"critic verdict=fail, max_rewrites={settings.max_rewrites} reached",
            }
        return {"next": "writer", "reason": "critic verdict=fail, rewrite needed"}
    return {"next": "done", "reason": f"critic verdict={verdict}, all outputs complete"}


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def run(self, state: ResearchState) -> ResearchState:
        settings = get_settings()

        with trace_span(
            "supervisor",
            attributes={"run_mode": "multi_agent", "iteration": state.iteration},
            input_data={
                "iteration": state.iteration,
                "research_notes_present": bool(state.research_notes),
                "analysis_notes_present": bool(state.analysis_notes),
                "final_answer_present": bool(state.final_answer),
                "errors_count": len(state.errors),
            },
        ) as span:
            # Hard guard: never exceed max_iterations
            if state.iteration >= settings.max_iterations:
                decision = {"next": "done", "reason": "max_iterations reached"}
                state.final_status = "max_iterations"
                logger.warning("Supervisor: max_iterations=%d reached, forcing done", settings.max_iterations)
            else:
                try:
                    decision = _llm_route(state)
                    logger.info(
                        "Supervisor [LLM]: iteration=%d -> %s (%s)",
                        state.iteration, decision["next"], decision["reason"],
                    )
                except Exception as e:
                    logger.warning("Supervisor: LLM routing failed (%s), falling back to if/else", e)
                    decision = _fallback_route(state)
                    logger.info(
                        "Supervisor [fallback]: iteration=%d -> %s (%s)",
                        state.iteration, decision["next"], decision["reason"],
                    )

            # Hard guard: Critic MUST run if final_answer exists but critic hasn't run yet.
            # This prevents LLM from skipping critic by returning "done" prematurely.
            if (
                decision["next"] == "done"
                and state.final_answer
                and _critic_verdict(state) is None
                and state.final_status != "max_iterations"
            ):
                decision = {"next": "critic", "reason": "hard-guard: critic must run before done"}
                logger.warning("Supervisor: overriding LLM 'done' -> 'critic' (critic has not run yet)")

            route = decision["next"]
            verdict = _critic_verdict(state)
            if route == "writer" and verdict == "fail":
                if state.rewrite_count >= settings.max_rewrites:
                    route = "done"
                    decision = {
                        "next": "done",
                        "reason": f"critic verdict=fail, max_rewrites={settings.max_rewrites} reached",
                    }
                    logger.warning("Supervisor: max_rewrites reached, stopping with failed status")
                else:
                    state.rewrite_count += 1
            if route == "done":
                state.final_status = _done_status(state, decision.get("reason", ""))

            state.record_route(route)
            state.agent_results.append(
                AgentResult(
                    agent=AgentName.SUPERVISOR,
                    content=route,
                    metadata={"reason": decision.get("reason", ""), "iteration": state.iteration},
                )
            )
            state.add_trace_event("supervisor_decision", decision)
            span["output"] = {"next_agent": route, "reason": decision.get("reason", "")}

        return state

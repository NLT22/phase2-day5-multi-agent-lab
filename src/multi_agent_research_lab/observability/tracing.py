"""Tracing hooks — JSONL file log with input/output + automatic LangSmith via LANGSMITH_TRACING."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

logger = logging.getLogger(__name__)

_LOG_DIR = Path("logs")
_LOG_DIR.mkdir(parents=True, exist_ok=True)

_log_file_path: Path = _LOG_DIR / f"trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"


def _append_span(record: dict[str, Any]) -> None:
    with _log_file_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


@contextmanager
def trace_span(
    name: str,
    attributes: dict[str, Any] | None = None,
    input_data: dict[str, Any] | None = None,
) -> Iterator[dict[str, Any]]:
    """Span context manager — logs to JSONL with input/output detail.

    LangSmith traces are emitted automatically by LangGraph when
    LANGSMITH_TRACING=true and LANGSMITH_API_KEY are set in the environment.

    Usage:
        with trace_span("researcher", input_data={"query": q}) as span:
            ...
            span["output"] = {"notes_chars": len(notes)}
    """
    attrs = attributes or {}
    started = perf_counter()
    span: dict[str, Any] = {
        "name": name,
        "attributes": attrs,
        "input": input_data or {},
        "output": {},
        "duration_seconds": None,
    }

    try:
        yield span
        status = "ok"
    except Exception as exc:
        status = "error"
        span["error"] = str(exc)
        raise
    finally:
        duration_s = perf_counter() - started
        span["duration_seconds"] = duration_s

        record: dict[str, Any] = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "run_mode": attrs.get("run_mode", "unknown"),
            "span_name": name,
            "duration_ms": round(duration_s * 1000, 2),
            "status": status,
            "error": span.get("error"),
            "input": span["input"],
            "output": span.get("output", {}),
        }
        _append_span(record)

        # Pretty log to console so user sees what each agent did
        out_summary = ", ".join(f"{k}={v}" for k, v in span.get("output", {}).items())
        logger.info(
            "span[%s] %.0fms %s | out: %s",
            name, duration_s * 1000, status, out_summary or "(none)",
        )

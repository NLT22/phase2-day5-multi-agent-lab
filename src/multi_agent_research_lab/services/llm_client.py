"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import dataclass

from tenacity import retry, stop_after_attempt, wait_exponential

from multi_agent_research_lab.core.config import get_settings

logger = logging.getLogger(__name__)

_OPENAI_INPUT_COST_PER_TOKEN = 0.15 / 1_000_000
_OPENAI_OUTPUT_COST_PER_TOKEN = 0.60 / 1_000_000


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


def _make_openai_client(base_url: str | None = None, api_key: str | None = None):
    from openai import OpenAI

    kwargs: dict = {}
    if base_url:
        kwargs["base_url"] = base_url
    kwargs["api_key"] = api_key or "lm-studio"
    return OpenAI(**kwargs)


def _lm_studio_reachable(base_url: str) -> bool:
    import urllib.request
    try:
        urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=2)
        return True
    except Exception:
        return False


class LLMClient:
    """Provider-agnostic LLM client. Priority: LM Studio local -> OpenAI cloud."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._client = None
        self._model: str = ""
        self._is_local: bool = False
        self._provider: str = "none"
        self._init_client()

    def _init_client(self) -> None:
        s = self._settings
        if _lm_studio_reachable(s.lm_studio_base_url):
            self._client = _make_openai_client(base_url=s.lm_studio_base_url)
            self._model = s.lm_studio_model
            self._is_local = True
            self._provider = "lm_studio"
            logger.info("LLMClient using LM Studio local at %s (model=%s)", s.lm_studio_base_url, self._model)
            return
        if s.openai_api_key:
            self._client = _make_openai_client(api_key=s.openai_api_key)
            self._model = s.openai_model
            self._is_local = False
            self._provider = "openai"
            logger.info("LLMClient using OpenAI cloud (model=%s)", self._model)
            return
        raise RuntimeError("No LLM provider available. Start LM Studio or set OPENAI_API_KEY in .env")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a model completion with per-call timeout and retry."""
        timeout = self._settings.timeout_seconds

        def _call():
            return self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
            )

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_call)
            try:
                response = future.result(timeout=timeout)
            except FuturesTimeout:
                future.cancel()
                raise TimeoutError(f"LLM call exceeded timeout={timeout}s (provider={self._provider})")

        choice = response.choices[0]
        content = choice.message.content or ""
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else None
        output_tokens = usage.completion_tokens if usage else None

        if self._is_local:
            cost = 0.0
        elif input_tokens is not None and output_tokens is not None:
            cost = input_tokens * _OPENAI_INPUT_COST_PER_TOKEN + output_tokens * _OPENAI_OUTPUT_COST_PER_TOKEN
        else:
            cost = None

        logger.info(
            "LLM[%s | %s] in=%s out=%s cost=$%.6f",
            self._provider, self._model, input_tokens, output_tokens, cost or 0,
        )
        return LLMResponse(content=content, input_tokens=input_tokens, output_tokens=output_tokens, cost_usd=cost)

    @property
    def provider(self) -> str:
        return self._provider

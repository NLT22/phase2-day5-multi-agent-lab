"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from tenacity import retry, stop_after_attempt, wait_exponential

from multi_agent_research_lab.core.config import get_settings

logger = logging.getLogger(__name__)

# gpt-4o-mini pricing (USD per token)
_OPENAI_INPUT_COST_PER_TOKEN = 0.15 / 1_000_000
_OPENAI_OUTPUT_COST_PER_TOKEN = 0.60 / 1_000_000


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


def _make_openai_client(base_url: str | None = None, api_key: str | None = None):
    """Return an openai.OpenAI client, optionally pointed at a local endpoint."""
    from openai import OpenAI  # imported lazily to keep startup fast when not used

    kwargs: dict = {}
    if base_url:
        kwargs["base_url"] = base_url
    if api_key:
        kwargs["api_key"] = api_key
    else:
        kwargs["api_key"] = "lm-studio"  # LM Studio ignores this but SDK requires a non-empty value
    return OpenAI(**kwargs)


def _lm_studio_reachable(base_url: str) -> bool:
    """Quick connectivity check for LM Studio endpoint."""
    import urllib.request
    try:
        urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=2)
        return True
    except Exception:
        return False


class LLMClient:
    """Provider-agnostic LLM client.

    Priority: LM Studio local → OpenAI cloud → raises RuntimeError.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._settings = settings
        self._client = None
        self._model: str = ""
        self._is_local: bool = False
        self._provider: str = "none"
        self._init_client()

    def _init_client(self) -> None:
        settings = self._settings
        # 1. Try LM Studio local first
        if _lm_studio_reachable(settings.lm_studio_base_url):
            self._client = _make_openai_client(base_url=settings.lm_studio_base_url)
            self._model = settings.lm_studio_model
            self._is_local = True
            self._provider = "lm_studio"
            logger.info("LLMClient using LM Studio local at %s (model=%s)", settings.lm_studio_base_url, self._model)
            return
        # 2. Fallback to OpenAI cloud
        if settings.openai_api_key:
            self._client = _make_openai_client(api_key=settings.openai_api_key)
            self._model = settings.openai_model
            self._is_local = False
            self._provider = "openai"
            logger.info("LLMClient using OpenAI cloud (model=%s)", self._model)
            return
        raise RuntimeError(
            "No LLM provider available. Start LM Studio or set OPENAI_API_KEY in .env"
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a model completion with retry on transient errors."""
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
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
            self._provider,
            self._model,
            input_tokens,
            output_tokens,
            cost or 0,
        )
        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
        )

    @property
    def provider(self) -> str:
        return self._provider

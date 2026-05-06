"""Search client abstraction for ResearcherAgent."""

from __future__ import annotations

import logging

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import SourceDocument

logger = logging.getLogger(__name__)

_MOCK_SOURCES = [
    SourceDocument(
        title="GraphRAG: Graph-based Retrieval-Augmented Generation",
        url="https://example.com/graphrag-survey",
        snippet=(
            "GraphRAG enhances traditional RAG by building a knowledge graph over the corpus, "
            "enabling multi-hop reasoning and more precise evidence retrieval for complex queries."
        ),
    ),
    SourceDocument(
        title="Microsoft GraphRAG: Unlocking LLM Discovery on Narrative Private Data",
        url="https://example.com/microsoft-graphrag",
        snippet=(
            "Microsoft's GraphRAG system indexes documents into communities of entities and summarises "
            "each community, allowing the LLM to answer global questions about large datasets."
        ),
    ),
    SourceDocument(
        title="LightRAG: Simple and Fast Retrieval-Augmented Generation",
        url="https://example.com/lightrag",
        snippet=(
            "LightRAG combines graph-based indexing with vector similarity search, trading some "
            "precision for dramatically faster retrieval speeds compared to full GraphRAG pipelines."
        ),
    ),
    SourceDocument(
        title="Benchmarking RAG Architectures: Graph vs Vector vs Hybrid",
        url="https://example.com/rag-benchmark-2024",
        snippet=(
            "A 2024 benchmark across 10 QA datasets shows GraphRAG outperforms vanilla RAG by "
            "18% on multi-hop questions while incurring 3–5× more indexing cost."
        ),
    ),
    SourceDocument(
        title="Production Guardrails for LLM Agents",
        url="https://example.com/llm-agent-guardrails",
        snippet=(
            "Key guardrails include max_iterations limits, timeout enforcement, output validation "
            "with Pydantic, and fallback responses when agent chains exceed cost budgets."
        ),
    ),
]


class SearchClient:
    """Provider-agnostic search client. Uses Tavily if API key is set, otherwise mock."""

    def __init__(self) -> None:
        settings = get_settings()
        self._tavily_key = settings.tavily_api_key
        if self._tavily_key:
            logger.info("SearchClient using Tavily API")
        else:
            logger.info("SearchClient using mock data (set TAVILY_API_KEY to enable real search)")

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search for documents relevant to a query."""
        if self._tavily_key:
            return self._tavily_search(query, max_results)
        return self._mock_search(query, max_results)

    def _tavily_search(self, query: str, max_results: int) -> list[SourceDocument]:
        from tavily import TavilyClient  # imported lazily

        client = TavilyClient(api_key=self._tavily_key)
        response = client.search(query=query, max_results=max_results)
        results = []
        for item in response.get("results", []):
            results.append(
                SourceDocument(
                    title=item.get("title", "Untitled"),
                    url=item.get("url"),
                    snippet=item.get("content", ""),
                    metadata={"score": item.get("score")},
                )
            )
        logger.info("Search[tavily] returned %d results for: %.60s", len(results), query)
        return results

    def _mock_search(self, query: str, max_results: int) -> list[SourceDocument]:
        n = min(max_results, len(_MOCK_SOURCES))
        logger.info("Search[mock] returning %d static results for: %.60s", n, query)
        return _MOCK_SOURCES[:max_results]

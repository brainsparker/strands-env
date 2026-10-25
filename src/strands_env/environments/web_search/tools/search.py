from __future__ import annotations

import asyncio
import logging
import os
from typing import Literal

import aiohttp
from strands import tool

from strands_env.utils.decorators import requires_env

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 10
DEFAULT_MAX_CONCURRENCY = 10
MAX_RESULTS = 10

GOOGLE_SERPER_DEV_URL = "https://google.serper.dev/search"
GOOGLE_CUSTOM_SEARCH_URL = "https://www.googleapis.com/customsearch/v1"
YOUCOM_FREE_SEARCH_URL = "https://api.you.com/v1/agents/search"
YOUCOM_SEARCH_URL = "https://ydc-index.io/v1/search"
YOUCOM_USER_AGENT = "youdotcom-integration/strands-rl-strands-env"
YOUCOM_MAX_RETRY_AFTER = 5.0  # seconds; a longer Retry-After fails the call instead of stalling a rollout

type WebSearchAPIProvider = Literal["serper", "google", "youcom"]


class WebSearchToolkit:
    """Search tools behind a provider switch (`serper` / `google` / `youcom`).

    Each provider is a separate `@tool` method, so an environment can expose one, the other, or
    the unified `search` that dispatches on `api_provider`. Credentials are checked lazily by
    `@requires_env`, at call time rather than construction.

    Notes:
        Holds one lazily-created `aiohttp.ClientSession`; `cleanup` closes it.
    """

    def __init__(
        self,
        timeout: int = DEFAULT_TIMEOUT,
        concurrency: asyncio.Semaphore | int = DEFAULT_MAX_CONCURRENCY,
        blocked_domains: list[str] | None = None,
        api_provider: WebSearchAPIProvider = "serper",
    ):
        """Initialize a `WebSearchToolkit` instance.

        Args:
            timeout: HTTP request timeout in seconds.
            concurrency: A shared `Semaphore` (one budget across toolkits) or an int (per-toolkit cap).
            blocked_domains: Domains to exclude from results (e.g. `["huggingface.co"]`).
            api_provider: Which backend the `search` tool dispatches to.
        """
        self.timeout = timeout
        self.semaphore = concurrency if isinstance(concurrency, asyncio.Semaphore) else asyncio.Semaphore(concurrency)
        self.blocked_domains = blocked_domains or []
        self.api_provider = api_provider
        self._session: aiohttp.ClientSession | None = None

    def _get_session(self) -> aiohttp.ClientSession:
        # Reopened if closed, so a cleaned-up toolkit still works if reused.
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))
        return self._session

    async def cleanup(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    @staticmethod
    def apply_blocked_domains(query: str, blocked_domains: list[str]) -> str:
        """Append `-site:` exclusions to `query` for blocked domains."""
        if blocked_domains:
            return query + " " + " ".join(f"-site:{d}" for d in blocked_domains)
        return query

    @staticmethod
    def format_results(
        items: list[dict], *, title_key: str = "title", url_key: str = "link", snippet_key: str = "snippet"
    ) -> str:
        """Format a list of search result dicts into a numbered text block.

        A hook: override to change how results reach the model.
        """
        if not items:
            return "No results found."
        lines = []
        for i, item in enumerate(items, 1):
            title = item.get(title_key) or "No title available."
            url = item.get(url_key) or "No URL available."
            snippet = item.get(snippet_key) or "No snippet available."
            lines.append(f"{i}. {title} ({url}):\n{snippet}")
        return "\n\n".join(lines)

    # ------------------------------------------------------------------
    # Serper
    # ------------------------------------------------------------------

    @tool
    @requires_env("SERPER_API_KEY")
    async def serper_search(self, query: str, top_k: int = 5) -> str:
        """Search the web using Serper.dev API.

        Args:
            query: The search query.
            top_k: Number of results to return.

        Returns:
            Search results with title, URL, and snippet for each result.
        """
        logger.info("[serper_search] query=%s, top_k=%s", query, top_k)

        query = self.apply_blocked_domains(query, self.blocked_domains)

        headers = {
            "X-API-KEY": os.environ["SERPER_API_KEY"],
            "Content-Type": "application/json",
        }
        payload = {"q": query, "num": top_k}

        try:
            async with self.semaphore:
                async with self._get_session().post(GOOGLE_SERPER_DEV_URL, json=payload, headers=headers) as response:
                    response.raise_for_status()
                    data = await response.json()

            return self.format_results(data.get("organic", []))
        except Exception as e:
            logger.error("[serper_search] error: %s", e)
            return f"Search failed: {e}."

    # ------------------------------------------------------------------
    # Google Custom Search
    # ------------------------------------------------------------------

    @tool
    @requires_env("GOOGLE_API_KEY", "GOOGLE_CSE_ID")
    async def google_search(self, query: str, top_k: int = 5) -> str:
        """Search the web using Google Custom Search JSON API.

        Args:
            query: The search query.
            top_k: Number of results to return.

        Returns:
            Search results with title, URL, and snippet for each result.
        """
        logger.info("[google_search] query=%s, top_k=%s", query, top_k)

        top_k = min(top_k, MAX_RESULTS)
        query = self.apply_blocked_domains(query, self.blocked_domains)

        params = {
            "key": os.environ["GOOGLE_API_KEY"],
            "cx": os.environ["GOOGLE_CSE_ID"],
            "q": query,
            "num": str(top_k),
        }

        try:
            async with self.semaphore:
                async with self._get_session().get(GOOGLE_CUSTOM_SEARCH_URL, params=params) as response:
                    response.raise_for_status()
                    data = await response.json()

            return self.format_results(data.get("items", []))
        except Exception as e:
            logger.error("[google_search] error: %s", e)
            return f"Search failed: {e}."

    # ------------------------------------------------------------------
    # You.com
    # ------------------------------------------------------------------

    @staticmethod
    def _youcom_error(status: int, keyed: bool, retry_after: float | None) -> str:
        if status in (401, 403):
            reason = "YDC_API_KEY was rejected" if keyed else "the keyless tier refused the request"
        elif status == 402:
            reason = "out of credits" if keyed else "free daily quota used up; set YDC_API_KEY"
        elif status == 429:
            reason = "rate limited" + (f", retry after {retry_after:g}s" if retry_after is not None else "")
        else:
            reason = f"HTTP {status}"
        return f"Search failed: You.com {reason}."

    @staticmethod
    def _retry_after(value: str | None) -> float | None:
        try:
            seconds = float(value) if value is not None else None
        except ValueError:
            return None
        return seconds if seconds is not None and seconds >= 0 else None

    @tool
    async def youcom_search(self, query: str, top_k: int = 5) -> str:
        """Search the web using the You.com Search API.

        Works without a key (free tier, 100 searches/day per IP); set `YDC_API_KEY` for training volume.

        Args:
            query: The search query.
            top_k: Number of results to return.

        Returns:
            Search results with title, URL, and snippet for each result.
        """
        return await self._youcom_search(query, top_k)

    async def _youcom_search(self, query: str, top_k: int) -> str:
        logger.info("[youcom_search] query=%s, top_k=%s", query, top_k)

        top_k = max(1, min(top_k, MAX_RESULTS))
        query = self.apply_blocked_domains(query, self.blocked_domains)
        api_key = os.getenv("YDC_API_KEY", "").strip()
        headers = {"User-Agent": YOUCOM_USER_AGENT, "Accept": "application/json"}
        body = {"query": query, "count": top_k}

        async def send() -> tuple[int, str | None, dict]:
            if api_key:
                request = self._get_session().post(
                    YOUCOM_SEARCH_URL, json=body, headers={**headers, "X-API-Key": api_key}
                )
            else:
                request = self._get_session().get(
                    YOUCOM_FREE_SEARCH_URL, params={k: str(v) for k, v in body.items()}, headers=headers
                )
            async with request as response:
                if response.status >= 400:
                    return response.status, response.headers.get("Retry-After"), {}
                return response.status, None, await response.json()

        try:
            async with self.semaphore:
                status, retry_after_header, data = await send()
                retry_after = self._retry_after(retry_after_header)
                if status == 429 and retry_after is not None and retry_after <= YOUCOM_MAX_RETRY_AFTER:
                    await asyncio.sleep(retry_after)
                    status, retry_after_header, data = await send()
                    retry_after = self._retry_after(retry_after_header)
            if status >= 400:
                return self._youcom_error(status, bool(api_key), retry_after)

            items = []
            for item in (data.get("results") or {}).get("web") or []:
                snippet = item.get("description") or " ".join(item.get("snippets") or [])
                items.append({"title": item.get("title"), "link": item.get("url"), "snippet": snippet})
            return self.format_results(items[:top_k])
        except Exception as e:
            logger.error("[youcom_search] error: %s", e)
            return f"Search failed: {e}."

    # ------------------------------------------------------------------
    # Unified entry point
    # ------------------------------------------------------------------

    @tool
    async def search(self, query: str, top_k: int = 5) -> str:
        """Search the web.

        Args:
            query: The search query.
            top_k: Number of results to return.

        Returns:
            Search results with title, URL, and snippet for each result.
        """
        match self.api_provider:
            case "serper":
                return await self.serper_search(query, top_k)
            case "google":
                return await self.google_search(query, top_k)
            case "youcom":
                return await self._youcom_search(query, top_k)
            case _:
                raise ValueError(f"Invalid API provider: {self.api_provider}")

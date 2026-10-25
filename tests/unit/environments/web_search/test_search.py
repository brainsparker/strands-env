import pytest

from strands_env.environments.web_search.tools import search as search_module
from strands_env.environments.web_search.tools.search import WebSearchToolkit

# ---------------------------------------------------------------------------
# _apply_blocked_domains
# ---------------------------------------------------------------------------


class TestApplyBlockedDomains:
    def test_appends_exclusions(self):
        result = WebSearchToolkit.apply_blocked_domains("python asyncio", ["example.com", "spam.org"])
        assert result == "python asyncio -site:example.com -site:spam.org"

    def test_empty_blocked_domains(self):
        result = WebSearchToolkit.apply_blocked_domains("python asyncio", [])
        assert result == "python asyncio"

    def test_no_blocked_domains(self):
        result = WebSearchToolkit.apply_blocked_domains("query", [])
        assert result == "query"


# ---------------------------------------------------------------------------
# format_results
# ---------------------------------------------------------------------------


class TestFormatResults:
    def test_formats_numbered_list(self):
        items = [
            {"title": "Result One", "link": "https://one.com", "snippet": "First snippet."},
            {"title": "Result Two", "link": "https://two.com", "snippet": "Second snippet."},
        ]
        result = WebSearchToolkit.format_results(items)
        assert "1. Result One (https://one.com):" in result
        assert "First snippet." in result
        assert "2. Result Two (https://two.com):" in result
        assert "Second snippet." in result

    def test_empty_results(self):
        result = WebSearchToolkit.format_results([])
        assert result == "No results found."

    def test_missing_fields_use_defaults(self):
        items = [{}]
        result = WebSearchToolkit.format_results(items)
        assert "No title available." in result
        assert "No URL available." in result
        assert "No snippet available." in result


# ---------------------------------------------------------------------------
# youcom_search
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, status: int, payload: dict | None = None, headers: dict | None = None):
        self.status = status
        self._payload = payload or {}
        self.headers = headers or {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def json(self):
        return self._payload


class _FakeSession:
    def __init__(self, responses: list[_FakeResponse]):
        self._responses = list(responses)
        self.calls: list[tuple[str, str, dict]] = []
        self.closed = False

    def _next(self, method: str, url: str, **kwargs):
        self.calls.append((method, url, kwargs))
        return self._responses.pop(0)

    def get(self, url, **kwargs):
        return self._next("GET", url, **kwargs)

    def post(self, url, **kwargs):
        return self._next("POST", url, **kwargs)


YOUCOM_PAYLOAD = {
    "results": {
        "web": [
            {"url": "https://a.com", "title": "A", "description": "alpha"},
            {"url": "https://b.com", "title": "B", "snippets": ["beta", "two"]},
        ]
    }
}


def _toolkit(session: _FakeSession, **kwargs) -> WebSearchToolkit:
    toolkit = WebSearchToolkit(api_provider="youcom", **kwargs)
    toolkit._get_session = lambda: session  # type: ignore[method-assign]
    return toolkit


class TestYouComSearch:
    async def test_keyless_get(self, monkeypatch):
        monkeypatch.delenv("YDC_API_KEY", raising=False)
        session = _FakeSession([_FakeResponse(200, YOUCOM_PAYLOAD)])

        result = await _toolkit(session, blocked_domains=["spam.org"]).search("python asyncio", top_k=3)

        assert "1. A (https://a.com):\nalpha" in result
        assert "2. B (https://b.com):\nbeta two" in result
        method, url, kwargs = session.calls[0]
        assert (method, url) == ("GET", search_module.YOUCOM_FREE_SEARCH_URL)
        assert kwargs["params"] == {"query": "python asyncio -site:spam.org", "count": "3"}
        assert kwargs["headers"]["User-Agent"] == search_module.YOUCOM_USER_AGENT
        assert "X-API-Key" not in kwargs["headers"]

    async def test_keyed_post(self, monkeypatch):
        monkeypatch.setenv("YDC_API_KEY", "ydc-test")
        session = _FakeSession([_FakeResponse(200, YOUCOM_PAYLOAD)])

        await _toolkit(session).youcom_search("q", top_k=50)

        method, url, kwargs = session.calls[0]
        assert (method, url) == ("POST", search_module.YOUCOM_SEARCH_URL)
        assert kwargs["headers"]["X-API-Key"] == "ydc-test"
        assert kwargs["json"] == {"query": "q", "count": search_module.MAX_RESULTS}

    @pytest.mark.parametrize(
        ("status", "keyed", "needle"),
        [
            (401, True, "YDC_API_KEY was rejected"),
            (403, False, "keyless tier refused"),
            (402, False, "free daily quota used up"),
            (402, True, "out of credits"),
        ],
    )
    async def test_errors_are_explained(self, monkeypatch, status, keyed, needle):
        if keyed:
            monkeypatch.setenv("YDC_API_KEY", "k")
        else:
            monkeypatch.delenv("YDC_API_KEY", raising=False)
        session = _FakeSession([_FakeResponse(status)])

        result = await _toolkit(session).youcom_search("q")

        assert result.startswith("Search failed: You.com")
        assert needle in result
        assert len(session.calls) == 1

    async def test_short_retry_after_is_waited_out(self, monkeypatch):
        monkeypatch.delenv("YDC_API_KEY", raising=False)
        slept: list[float] = []

        async def fake_sleep(seconds):
            slept.append(seconds)

        monkeypatch.setattr(search_module.asyncio, "sleep", fake_sleep)
        session = _FakeSession([_FakeResponse(429, headers={"Retry-After": "2"}), _FakeResponse(200, YOUCOM_PAYLOAD)])

        result = await _toolkit(session).youcom_search("q")

        assert slept == [2.0]
        assert result.startswith("1. A")

    async def test_long_retry_after_fails_fast(self, monkeypatch):
        monkeypatch.delenv("YDC_API_KEY", raising=False)
        session = _FakeSession([_FakeResponse(429, headers={"Retry-After": "60"})])

        result = await _toolkit(session).youcom_search("q")

        assert result == "Search failed: You.com rate limited, retry after 60s."
        assert len(session.calls) == 1

    async def test_no_results(self, monkeypatch):
        monkeypatch.delenv("YDC_API_KEY", raising=False)
        session = _FakeSession([_FakeResponse(200, {"results": {"web": []}})])

        assert await _toolkit(session).youcom_search("q") == "No results found."

"""News comes from plain RSS across several feeds, and dark != quiet.

The old fetcher was `yfinance.Ticker(symbol).news` inside a bare
`except: return []` — one source whose payload shape drifts, and a total
failure that looked identical to a slow news day.
"""
from __future__ import annotations

import pytest

from specforge import news_sources
from specforge.news_sources import NewsSourcesUnavailable

RSS = """<?xml version="1.0"?>
<rss version="2.0"><channel>
  <title>Feed</title>
  <item>
    <title>Acme beats earnings</title>
    <link>https://example.com/a</link>
    <description>Acme reported &lt;b&gt;strong&lt;/b&gt; results.</description>
    <pubDate>Mon, 20 Jul 2026 13:05:00 GMT</pubDate>
    <source>Reuters</source>
  </item>
  <item>
    <title>Acme names new CFO</title>
    <link>https://example.com/b</link>
    <description>Leadership change.</description>
    <pubDate>Sun, 19 Jul 2026 09:00:00 GMT</pubDate>
  </item>
</channel></rss>"""


class _Response:
    def __init__(self, text, status=200):
        self.text, self.status = text, status

    def raise_for_status(self):
        if self.status >= 400:
            raise RuntimeError(f"HTTP {self.status}")


class _Client:
    """Serves a canned body per feed name; None means that feed is down."""

    def __init__(self, bodies):
        self.bodies, self.calls = bodies, []

    def get(self, url, **kwargs):
        self.calls.append(url)
        for name, body in self.bodies.items():
            if name in url:
                if body is None:
                    raise RuntimeError("connection refused")
                return _Response(body)
        return _Response("", 404)


def test_parse_rss_extracts_the_stored_article_shape():
    articles = news_sources.parse_rss(RSS, "google_news")
    assert len(articles) == 2
    first = articles[0]
    assert first["title"] == "Acme beats earnings"
    assert first["url"] == "https://example.com/a"
    assert first["provider"] == "Reuters"
    assert "strong results" in first["summary"]      # HTML tags stripped
    assert first["published"].startswith("2026-07-20T13:05:00")
    assert first["id"].startswith("NEWS:")
    # Falls back to the feed name when the item carries no <source>.
    assert articles[1]["provider"] == "google_news"


def test_parse_rss_survives_malformed_xml_and_untitled_items():
    assert news_sources.parse_rss("not xml at all", "google_news") == []
    untitled = RSS.replace("<title>Acme beats earnings</title>", "<title></title>")
    assert len(news_sources.parse_rss(untitled, "google_news")) == 1


def test_one_dead_source_does_not_darken_the_feature():
    client = _Client({"news.google.com": None,          # down
                      "feeds.finance.yahoo.com": RSS,   # up
                      "nasdaq.com": None})              # down
    articles = news_sources.company_news("ACME", client=client)
    assert len(articles) == 2
    assert len(client.calls) == 3                       # all sources attempted


def test_every_source_failing_raises_rather_than_returning_empty():
    """A dark feed must not be reported as 'no news today'."""
    client = _Client({"news.google.com": None, "feeds.finance.yahoo.com": None,
                      "nasdaq.com": None})
    with pytest.raises(NewsSourcesUnavailable, match="every source failed"):
        news_sources.company_news("ACME", client=client)


def test_the_same_story_from_two_feeds_collapses_to_one_row():
    """Duplicate coverage must not double-count as evidence."""
    client = _Client({"news.google.com": RSS, "feeds.finance.yahoo.com": RSS,
                      "nasdaq.com": RSS})
    articles = news_sources.company_news("ACME", client=client)
    assert len(articles) == 2
    assert len({a["id"] for a in articles}) == 2


def test_articles_are_returned_newest_first_and_limited():
    client = _Client({"news.google.com": RSS, "feeds.finance.yahoo.com": None,
                      "nasdaq.com": None})
    articles = news_sources.company_news("ACME", limit=1, client=client)
    assert len(articles) == 1
    assert articles[0]["title"] == "Acme beats earnings"     # the 20th, not the 19th


def test_ingest_records_dark_symbols_instead_of_silently_skipping(cfg, store):
    from specforge import intelligence

    def fetcher(symbol, limit=12):
        raise NewsSourcesUnavailable(f"{symbol}: every source failed")

    inserted = intelligence._ingest(store, fetcher=fetcher)
    assert inserted == 0
    assert store.kv_get("news_dark_symbols")          # visible, not swallowed
    assert store.db.execute(
        "SELECT COUNT(*) n FROM audit WHERE event_type='news_sources_unavailable'"
    ).fetchone()["n"] > 0


def test_ingest_stores_rss_articles_and_dedupes(cfg, store):
    from specforge import intelligence
    articles = news_sources.parse_rss(RSS, "google_news")
    intelligence._ingest(store, fetcher=lambda symbol, limit=12: articles)
    first = store.db.execute(
        "SELECT COUNT(*) n FROM news_intelligence").fetchone()["n"]
    assert first > 0
    intelligence._ingest(store, fetcher=lambda symbol, limit=12: articles)
    assert store.db.execute(
        "SELECT COUNT(*) n FROM news_intelligence").fetchone()["n"] == first

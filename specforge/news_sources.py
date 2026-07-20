"""Company news from plain RSS feeds — boring on purpose.

The previous fetcher was `yfinance.Ticker(symbol).news` wrapped in a bare
`except: return []`. It had three problems: the shape of the payload drifts
whenever Yahoo reorganizes its internal JSON, a total failure was
indistinguishable from "no news today", and there is exactly one source, so when
it breaks the feature silently goes dark.

These are RSS feeds. No API keys, no quotas, no auth to expire, no vendor
contract to change under us — an XML document at a stable URL, which is the most
boring possible dependency and therefore the right one. Sources are queried
independently and their failures are reported rather than swallowed.

What this does NOT do is produce history: every one of these feeds returns
current headlines only. News remains unusable as a training feature before the
system started collecting it, and the training panel labels it as such. This
module improves the LIVE signal, not the historical record.
"""
from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from urllib.parse import quote_plus

import httpx

USER_AGENT = "Stonk Terminal news reader"
TIMEOUT = 12.0

# Each entry: (name, url template taking {symbol}).
FEEDS = (
    ("google_news",
     "https://news.google.com/rss/search?q={symbol}+stock&hl=en-US&gl=US&ceid=US:en"),
    ("yahoo_finance",
     "https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"),
    ("nasdaq",
     "https://www.nasdaq.com/feed/rssoutbound?symbol={symbol}"),
)

_TAG = re.compile(r"<[^>]+>")
_RFC822 = "%a, %d %b %Y %H:%M:%S"


def _text(node, tag: str) -> str:
    child = node.find(tag)
    if child is None or child.text is None:
        return ""
    return _TAG.sub("", child.text).strip()


def _parse_published(raw: str) -> str:
    """RSS dates are RFC-822. Fall back to now() rather than dropping an item."""
    cleaned = (raw or "").strip()
    for suffix in (" GMT", " UTC", " +0000", " -0000", " EST", " EDT", " PST", " PDT"):
        if cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)].strip()
            break
    try:
        return datetime.strptime(cleaned, _RFC822).replace(
            tzinfo=timezone.utc).isoformat()
    except ValueError:
        return datetime.now(timezone.utc).isoformat()


def parse_rss(xml: str, source: str, limit: int = 12) -> list[dict]:
    """RSS <item>s → the article shape `intelligence._ingest` stores."""
    from xml.etree import ElementTree

    try:
        root = ElementTree.fromstring(xml)
    except ElementTree.ParseError:
        return []
    out = []
    for item in root.iter("item"):
        title = _text(item, "title")
        if not title:
            continue
        url = _text(item, "link")
        published = _parse_published(_text(item, "pubDate"))
        out.append({
            # Hash title+url, NOT the source: the same story from two feeds must
            # collapse to one row rather than being counted twice as evidence.
            "id": "NEWS:" + hashlib.sha256(
                f"{title}|{url}".encode()).hexdigest()[:16],
            "title": title[:300],
            "summary": _text(item, "description")[:1000],
            "provider": _text(item, "source") or source,
            "published": published,
            "url": url,
        })
        if len(out) >= limit:
            break
    return out


def company_news(symbol: str, limit: int = 12, client=httpx,
                 feeds=FEEDS) -> list[dict]:
    """Current company news, merged across feeds and de-duplicated.

    Raises only when EVERY source fails — a dark feature should look different
    from a quiet news day, which the old bare-except could not express.
    """
    articles: dict[str, dict] = {}
    failures = []
    for name, template in feeds:
        try:
            response = client.get(template.format(symbol=quote_plus(symbol)),
                                  timeout=TIMEOUT,
                                  headers={"User-Agent": USER_AGENT})
            response.raise_for_status()
            for article in parse_rss(response.text, name, limit):
                articles.setdefault(article["id"], article)
        except Exception as exc:            # noqa: BLE001 — per-source isolation
            failures.append(f"{name}: {type(exc).__name__}")
    if len(failures) == len(feeds):
        raise NewsSourcesUnavailable(f"{symbol}: every source failed ({failures})")
    return sorted(articles.values(), key=lambda a: a["published"],
                  reverse=True)[:limit]


class NewsSourcesUnavailable(RuntimeError):
    """Every configured feed failed — the feature is dark, not merely quiet."""

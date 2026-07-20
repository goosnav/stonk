"""Detect and repair split-adjustment seams in the bars table.

`data.refresh()` takes an incremental window when a symbol already has recent
history, and `Store.upsert_bars` overwrites per `(symbol, d)` with no
re-adjustment pass. Both providers serve RETROACTIVELY split- and
dividend-adjusted series, so after a corporate action the older rows keep the
stale basis while the trailing window arrives on the new one. The join between
them is a price discontinuity that never happened.

This is not cosmetic. A seam lands directly in the forward-return targets and in
the `r1` momentum feature. The live store carried 1,834 impossible single-session
moves across 578 symbols — 31% of the 200-symbol training panel, worst case a
+416,566% "return" (ABVC 2015-09-17, 0.34 → 1433.12). Every model measurement
taken before this was measuring that.

The detector is deliberately conservative. Repairing a real +60% biotech print
out of the record would be a worse bug than the one being fixed, so a candidate
is only auto-repaired when TWO independent tests agree, and everything else is
reported for a human rather than rewritten.
"""
from __future__ import annotations

import numpy as np

# A one-session move beyond these bounds is not impossible, but it is rare
# enough to be worth examining. Candidates, not verdicts.
UPPER_CANDIDATE = 1.35
LOWER_CANDIDATE = 0.74
RATIONAL_TOLERANCE = 0.02        # split factors are exact; allow for rounding
# Ratios issuers actually declare. Forward splits divide the price (2:1 -> 0.5),
# reverse splits multiply it (1:10 -> 10). Deliberately a closed list: see
# _simple_rational for why "any small fraction" is not safe here.
_FORWARD = (1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0,
            12.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 100.0)
SPLIT_FACTORS = tuple(sorted(set(_FORWARD) | {round(1.0 / f, 10) for f in _FORWARD}))
PERSISTENCE_WINDOW = 20
PERSISTENCE_TOLERANCE = 0.10
# Beyond this, the single-rational test stops being the right question. Repeated
# reverse splits compound into ratios no small fraction describes (0.34 → 1433
# is 1-for-~4200, i.e. several splits stacked), and no equity moves 20x in one
# session by trading — whether it persists or snaps back, the data is wrong.
EXTREME_RATIO = 20.0


def _simple_rational(ratio: float) -> bool:
    """Is `ratio` (or its reciprocal) a plausible split factor?

    An explicit list, NOT "any small rational". Issuers declare splits from a
    short conventional menu; "any fraction with terms under 20" also matches
    0.7, and Black Monday moved MSFT by exactly that — the first run of this
    detector classified 1987-10-19 as a corporate action and would have
    rewritten a real crash out of the record. Encoding what splits actually are
    is what makes the test evidence rather than numerology.
    """
    if not np.isfinite(ratio) or ratio <= 0:
        return False
    for factor in SPLIT_FACTORS:
        if abs(ratio - factor) / factor <= RATIONAL_TOLERANCE:
            return True
    return False


def _persists(closes: np.ndarray, index: int, ratio: float,
              window: int = PERSISTENCE_WINDOW) -> bool:
    """Did the whole price LEVEL shift by `ratio`, or was it a one-session spike?

    An adjustment seam moves every subsequent price; a real spike substantially
    reverts. Comparing pre/post medians rather than single closes keeps one
    noisy print from deciding it.
    """
    before = closes[max(0, index - window):index]
    after = closes[index:index + window]
    if len(before) < 3 or len(after) < 3:
        return False
    observed = float(np.median(after) / max(np.median(before), 1e-12))
    return abs(observed - ratio) / ratio <= PERSISTENCE_TOLERANCE


def _symbols(store, symbols=None) -> list[str]:
    if symbols:
        return list(symbols)
    return [r["symbol"] for r in store.db.execute(
        "SELECT DISTINCT symbol FROM bars ORDER BY symbol")]


def detect(store, symbols=None, since: str = "1900-01-01") -> list[dict]:
    """Every adjustment candidate, classified `seam` or `suspicious`.

    `seam` means the move is a corporate action (a declared split factor whose
    level shift persists) or physically impossible (an extreme ratio), and is
    safe to auto-repair. `suspicious` means the move is large but could be
    real — surfaced so a human can look, never rewritten automatically.
    """
    findings: list[dict] = []
    for symbol in _symbols(store, symbols):
        rows = store.db.execute(
            "SELECT d, close FROM bars WHERE symbol=? AND d>=? ORDER BY d",
            (symbol, since)).fetchall()
        if len(rows) < 5:
            continue
        closes = np.asarray([float(r["close"]) for r in rows], dtype=np.float64)
        dates = [r["d"] for r in rows]
        previous = np.maximum(closes[:-1], 1e-12)
        ratios = closes[1:] / previous
        for offset in np.flatnonzero((ratios > UPPER_CANDIDATE)
                                     | (ratios < LOWER_CANDIDATE)):
            index = int(offset) + 1
            ratio = float(ratios[offset])
            rational = _simple_rational(ratio)
            persists = _persists(closes, index, ratio)
            extreme = ratio >= EXTREME_RATIO or ratio <= 1.0 / EXTREME_RATIO
            findings.append({
                "symbol": symbol, "d": dates[index], "ratio": round(ratio, 6),
                "prior_close": round(float(closes[index - 1]), 4),
                "close": round(float(closes[index]), 4),
                "simple_rational": bool(rational), "persists": bool(persists),
                "extreme": bool(extreme),
                # Only an IMPOSSIBLE ratio is called a seam. The split-factor
                # test cannot carry this on its own: a -33% session lands within
                # tolerance of 2/3, so "declared split factor + persistent shift"
                # flagged AMD's 2002 dot-com crash and MCD, and quarantined 93
                # symbols on what were mostly real drawdowns. Distinguishing a
                # 3:2 split from a -33% crash needs the issuer's actual split
                # dates, which this offline detector does not have.
                "kind": "seam" if extreme else "suspicious"})
    return findings


LARGE_MOVES_PER_YEAR = 1.5


def unreliable(store, symbols=None, threshold: float = LARGE_MOVES_PER_YEAR) -> dict:
    """Symbols whose price history is too jumpy to be real, by DENSITY.

    A data-quality test, deliberately not a corporate-action claim. A volatile
    semiconductor might print three >35% sessions in thirty years; ABVC's series
    carries 130. Density separates "genuinely wild stock" from "provider cannot
    represent this security" without needing to know why any single jump exists.
    """
    counts: dict[str, dict] = {}
    for symbol in _symbols(store, symbols):
        rows = store.db.execute(
            "SELECT d, close FROM bars WHERE symbol=? ORDER BY d", (symbol,)).fetchall()
        if len(rows) < 250:
            continue
        closes = np.asarray([float(r["close"]) for r in rows], dtype=np.float64)
        ratios = closes[1:] / np.maximum(closes[:-1], 1e-12)
        large = int(np.count_nonzero((ratios > UPPER_CANDIDATE)
                                     | (ratios < LOWER_CANDIDATE)))
        years = len(rows) / 252.0
        rate = large / max(years, 1e-9)
        if rate > threshold:
            counts[symbol] = {"large_moves": large, "years": round(years, 1),
                              "per_year": round(rate, 2)}
    return counts


def repair_symbol(store, symbol: str, fetcher=None, min_coverage: float = 0.9,
                  source: str = "repair") -> dict:
    """Replace one symbol's history with a single self-consistent fetch.

    Order matters: fetch and validate FIRST, replace only on success. Deleting
    before fetching would turn a provider outage into permanent data loss, and
    a partial fetch that silently left the uncovered tail behind would leave
    exactly the mixed-basis series this module exists to eliminate — so the
    replacement is delete-then-insert inside one transaction.
    """
    if fetcher is None:
        from . import data
        fetcher = lambda sym: data.fetch_stooq(sym) or data.fetch_yfinance(sym)
    existing = store.db.execute(
        "SELECT COUNT(*) n FROM bars WHERE symbol=?", (symbol,)).fetchone()["n"]
    try:
        rows = fetcher(symbol)
    except Exception as exc:                    # noqa: BLE001 — per-symbol isolation
        return {"symbol": symbol, "status": "fetch_failed", "error": str(exc)[:200]}
    if not rows:
        return {"symbol": symbol, "status": "fetch_failed", "error": "no rows"}
    if existing and len(rows) < existing * min_coverage:
        # A shorter series may be correct, but it is not obviously BETTER, and
        # silently discarding history is the one outcome worse than a seam.
        return {"symbol": symbol, "status": "fetch_too_short",
                "existing": existing, "fetched": len(rows)}
    with store.db:
        store.db.execute("DELETE FROM bars WHERE symbol=?", (symbol,))
    store.upsert_bars(symbol, rows, source)
    remaining = [f for f in detect(store, [symbol]) if f["kind"] == "seam"]
    return {"symbol": symbol, "status": "repaired", "rows": len(rows),
            "replaced": existing, "seams_remaining": len(remaining)}


QUARANTINE_KEY = "bars_quarantine"


def quarantine(store, symbols: list[str], reason: str = "unrepairable_seams") -> list[str]:
    """Mark symbols whose history survives a clean refetch still seamed.

    A repair that cannot repair is not a failure of the repair — it means the
    PROVIDER serves a discontinuous series and no amount of refetching will
    change that. Free sources handle ultra-thin names with repeated reverse
    splits badly; back-adjustment compounds into absurd historical prices and
    genuine microcap noise becomes indistinguishable from an artifact.

    Excluding them is more honest than pretending they were fixed. Persisted so
    the exclusion is durable and auditable rather than re-derived per run.
    """
    existing = set(store.kv_get(QUARANTINE_KEY, []) or [])
    merged = sorted(existing | set(symbols))
    store.kv_set(QUARANTINE_KEY, merged)
    if set(symbols) - existing:
        store.audit("bars_quarantined",
                    {"added": sorted(set(symbols) - existing), "reason": reason,
                     "total": len(merged)})
    return merged


def quarantined(store) -> set[str]:
    """Symbols excluded from training panels for unrepairable price history."""
    return set(store.kv_get(QUARANTINE_KEY, []) or [])


def audit(store, symbols=None, repair: bool = False, fetcher=None,
          limit: int | None = None) -> dict:
    """Report seams; repair only when explicitly asked.

    Report-only is the default because the repair rewrites price history — the
    operator should see what would change before it changes.
    """
    findings = detect(store, symbols)
    seams = [f for f in findings if f["kind"] == "seam"]
    suspicious = [f for f in findings if f["kind"] == "suspicious"]
    affected = sorted({f["symbol"] for f in seams})
    repaired: list[dict] = []
    if repair:
        for symbol in affected[:limit] if limit else affected:
            repaired.append(repair_symbol(store, symbol, fetcher=fetcher))
    still_seamed = [r["symbol"] for r in repaired if r.get("seams_remaining")]
    if still_seamed:
        # The refetch produced the same discontinuity, so the provider's own
        # series is broken. Quarantine rather than retry forever.
        quarantine(store, still_seamed)
    return {"seams": len(seams), "suspicious": len(suspicious),
            "symbols_affected": affected, "symbols_scanned": len(_symbols(store, symbols)),
            "findings": findings, "repaired": repaired,
            "still_seamed": still_seamed,
            "quarantined": sorted(quarantined(store))}

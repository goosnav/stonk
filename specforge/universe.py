"""Official listing catalog and point-in-time research/active universe tiers."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from datetime import date, datetime

import httpx

NASDAQ = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
SEC = "https://www.sec.gov/files/company_tickers.json"
BAD_NAME = re.compile(
    r"\b(warrants?|rights?|units?|preferred|preference|beneficial interest)\b", re.I)
SYMBOL = re.compile(r"^[A-Z][A-Z0-9.-]{0,9}$")


def parse_directory(text: str, other: bool = False) -> list[dict]:
    rows = []
    for row in csv.DictReader(io.StringIO(text), delimiter="|"):
        symbol = (row.get("ACT Symbol") if other else row.get("Symbol") or "").strip().upper()
        name = (row.get("Security Name") or "").strip()
        if not symbol or symbol.startswith("File Creation Time") or not SYMBOL.match(symbol):
            continue
        if row.get("Test Issue") == "Y" or row.get("ETF") == "Y" or BAD_NAME.search(name):
            continue
        rows.append({"symbol": symbol, "name": name,
                     "exchange": row.get("Exchange") or "NASDAQ",
                     "security_type": "adr" if re.search(r"\b(ADR|ADS|depositary)\b", name, re.I)
                     else "common", "is_etf": 0,
                     "is_adr": int(bool(re.search(r"\b(ADR|ADS|depositary)\b", name, re.I))),
                     "source": "nasdaq_trader"})
    return rows


def sync_catalog(store, client=httpx) -> dict:
    headers = {"User-Agent": "Stonk Terminal research contact=local-user"}
    nas = client.get(NASDAQ, timeout=30, headers=headers); nas.raise_for_status()
    oth = client.get(OTHER, timeout=30, headers=headers); oth.raise_for_status()
    items = {r["symbol"]: r for r in parse_directory(nas.text)}
    items.update({r["symbol"]: r for r in parse_directory(oth.text, other=True)})
    cik = {}
    try:
        response = client.get(SEC, timeout=30, headers=headers); response.raise_for_status()
        cik = {v["ticker"].upper(): str(v["cik_str"]) for v in response.json().values()}
    except Exception:                           # identity enrichment is optional
        pass
    today = date.today().isoformat()
    with store.db:
        store.db.execute("UPDATE instruments SET active=0")
        for item in items.values():
            raw_hash = hashlib.sha256(json.dumps(item, sort_keys=True).encode()).hexdigest()[:16]
            store.db.execute(
                "INSERT INTO instruments(symbol,name,exchange,security_type,is_etf,is_adr,active,first_seen,last_seen,source,cik,raw_hash) VALUES(?,?,?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(symbol) DO UPDATE SET name=excluded.name,exchange=excluded.exchange,"
                "security_type=excluded.security_type,is_etf=excluded.is_etf,is_adr=excluded.is_adr,"
                "active=1,last_seen=excluded.last_seen,source=excluded.source,cik=excluded.cik,"
                "raw_hash=excluded.raw_hash",
                (item["symbol"], item["name"], item["exchange"], item["security_type"],
                 0, item["is_adr"], 1, today, today, item["source"],
                 cik.get(item["symbol"]), raw_hash))
    result = {"at": datetime.now().astimezone().isoformat(timespec="seconds"),
              "count": len(items), "cik_count": len(cik)}
    store.kv_set("catalog_status", result); store.audit("catalog_synced", result)
    return result


def _metrics(store) -> list[dict]:
    """20-session liquidity + history using one SQLite window query."""
    q = """
    WITH ranked AS (
      SELECT b.symbol,b.d,b.close,b.volume,
             ROW_NUMBER() OVER(PARTITION BY b.symbol ORDER BY b.d DESC) rn,
             COUNT(*) OVER(PARTITION BY b.symbol) history
      FROM bars b JOIN instruments i ON i.symbol=b.symbol WHERE i.active=1
    )
    SELECT symbol,MAX(CASE WHEN rn=1 THEN close END) price,
           AVG(CASE WHEN rn<=20 THEN close*volume END) dollar_volume,
           MAX(history) history
    FROM ranked GROUP BY symbol
    """
    return [dict(r) for r in store.db.execute(q)]


def refresh_membership(cfg, store, as_of: str | None = None) -> dict:
    as_of = as_of or store.latest_bar_date(
        cfg.get("universe", "benchmark", default="SPY")) or date.today().isoformat()
    research_min = float(cfg.get("universe", "research_min_dollar_volume", default=1e7))
    execution_min = float(cfg.get("universe", "execution_min_dollar_volume", default=5e7))
    min_price = float(cfg.get("universe", "min_price", default=5))
    min_history = int(cfg.get("universe", "min_history_bars", default=260))
    research_n = int(cfg.get("universe", "research_size", default=1500))
    active_n = int(cfg.get("universe", "active_size", default=250))
    eligible = [m for m in _metrics(store) if (m["price"] or 0) >= min_price and
                (m["dollar_volume"] or 0) >= research_min]
    eligible.sort(key=lambda m: m["dollar_volume"], reverse=True)
    research = eligible[:research_n]
    # Cheap broad rank: liquidity plus 21d/63d momentum from settled bars.
    scored = []
    for m in research:
        rows = store.get_bars(m["symbol"], as_of, 64)
        mom = (rows[-1]["close"] / rows[-22]["close"] - 1) if len(rows) >= 22 else -9
        m = {**m, "momentum_21d": mom,
             "trade_eligible": m["history"] >= min_history and m["dollar_volume"] >= execution_min}
        scored.append(m)
    liquid_rank = {m["symbol"]: i for i, m in enumerate(scored)}
    momentum_rank = {m["symbol"]: i for i, m in enumerate(
        sorted(scored, key=lambda x: x["momentum_21d"], reverse=True))}
    scored.sort(key=lambda m: liquid_rank[m["symbol"]] + momentum_rank[m["symbol"]])
    mandatory_order = list(dict.fromkeys(
        list(cfg.get("universe", "symbols", default=[])) +
        [p["symbol"] for p in store.open_positions(mode="live")]))
    mandatory = set(mandatory_order)
    active_symbols = list(dict.fromkeys(
        mandatory_order + [m["symbol"] for m in scored if m["trade_eligible"]]))
    # "Active" is a hard compute bound. Mandatory symbols receive first
    # priority, but cannot silently make the tier larger than its configured
    # maximum (the configured default still comfortably includes the core
    # universe and current holdings).
    active_symbols = active_symbols[:active_n]
    with store.db:
        # Reranking broad tiers must not erase the separately produced
        # opportunity shortlist for the same date while deep research uses it.
        store.db.execute("DELETE FROM universe_membership WHERE as_of=? "
                         "AND tier IN ('research','active')", (as_of,))
        for rank, m in enumerate(research, 1):
            store.db.execute("INSERT INTO universe_membership VALUES(?,?,?,?,?,?)",
                             (as_of, m["symbol"], "research", rank, "liquidity",
                              json.dumps(m)))
        by = {m["symbol"]: m for m in scored}
        for rank, sym in enumerate(active_symbols, 1):
            store.db.execute("INSERT INTO universe_membership VALUES(?,?,?,?,?,?)",
                             (as_of, sym, "active", rank,
                              "mandatory" if sym in mandatory else "liquidity+momentum",
                              json.dumps(by.get(sym, {}))))
    result = {"as_of": as_of, "catalog": store.db.execute(
        "SELECT COUNT(*) n FROM instruments WHERE active=1").fetchone()["n"],
        "research": len(research), "active": len(active_symbols)}
    store.kv_set("universe_status", result); store.audit("universe_refreshed", result)
    return result


def symbols(store, tier: str = "active") -> list[str]:
    row = store.db.execute("SELECT MAX(as_of) d FROM universe_membership WHERE tier=?",
                           (tier,)).fetchone()
    if not row or not row["d"]:
        return []
    return [r["symbol"] for r in store.db.execute(
        "SELECT symbol FROM universe_membership WHERE as_of=? AND tier=? ORDER BY rank",
        (row["d"], tier))]


PIT_TIERS = ("research", "active")


def membership_history(store, tiers=PIT_TIERS) -> dict[str, set[str]]:
    """{as_of: members} across every stored snapshot — the point-in-time index."""
    marks = ",".join("?" for _ in tiers)
    history: dict[str, set[str]] = {}
    for row in store.db.execute(
            f"SELECT as_of,symbol FROM universe_membership WHERE tier IN ({marks})",
            tuple(tiers)):
        history.setdefault(row["as_of"], set()).add(row["symbol"])
    return history


def membership_as_of(store, date: str, tiers=PIT_TIERS) -> set[str] | None:
    """Members as of the latest snapshot on or before `date`.

    Returns None when no snapshot covers that date — history predating the
    system is UNCOVERED, not empty and not today's universe. Callers must
    label uncovered spans rather than quietly assuming survivorship.
    """
    history = membership_history(store, tiers)
    prior = [d for d in history if d <= date]
    return history[max(prior)] if prior else None


def historical_symbols(store, tiers=PIT_TIERS) -> list[str]:
    """Every symbol that was ever a member, including names since delisted.

    Training off today's membership is the strongest survivor bias available;
    this is the panel that keeps the losers in.
    """
    marks = ",".join("?" for _ in tiers)
    return [r["symbol"] for r in store.db.execute(
        f"SELECT DISTINCT symbol FROM universe_membership WHERE tier IN ({marks}) "
        "ORDER BY symbol", tuple(tiers))]


def status(store) -> dict:
    return {"catalog": store.kv_get("catalog_status"),
            "tiers": store.kv_get("universe_status"),
            "active_symbols": symbols(store, "active")}


def ingest_next_filing_facts(store, client=httpx) -> dict:
    """Fetch one issuer's point-in-time SEC facts per research task."""
    candidates = store.db.execute(
        "SELECT i.symbol,i.cik FROM instruments i LEFT JOIN universe_membership u "
        "ON u.symbol=i.symbol AND u.tier='research' AND u.as_of=(SELECT MAX(as_of) "
        "FROM universe_membership WHERE tier='research') WHERE i.active=1 AND "
        "i.cik IS NOT NULL "
        "ORDER BY CASE WHEN u.rank IS NULL THEN 1 ELSE 0 END,u.rank,i.symbol LIMIT 2000").fetchall()
    today = datetime.now().astimezone().date().isoformat()
    def due(candidate) -> bool:
        attempt = store.kv_get(f"filing_facts_attempted_{candidate['symbol']}") or {}
        # Facts change when issuers file. Refresh completed issuers weekly;
        # provider failures retry on a later day rather than poisoning forever.
        attempted = attempt.get("attempted_on")
        if attempt.get("status") == "completed" and attempted:
            try:
                return (datetime.fromisoformat(today).date() -
                        datetime.fromisoformat(attempted).date()).days >= 7
            except ValueError:
                pass
        return attempted != today
    row = next((r for r in candidates if due(r)), None)
    if not row:
        return {"status": "caught_up", "kind": "filing_facts"}
    cik = str(row["cik"]).zfill(10)
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    try:
        r = client.get(url, timeout=30,
                       headers={"User-Agent": "Stonk Terminal research contact=local-user"})
        r.raise_for_status(); facts = r.json().get("facts", {}).get("us-gaap", {})
    except Exception as exc:
        result = {"status": "failed", "symbol": row["symbol"],
                  "attempted_on": today,
                  "error": f"{type(exc).__name__}: {str(exc)[:160]}"}
        store.kv_set(f"filing_facts_attempted_{row['symbol']}", result)
        store.audit("filing_facts_failed", result)
        return result
    wanted = {
        "EarningsPerShareDiluted", "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax", "GrossProfit",
        "OperatingIncomeLoss", "NetIncomeLoss", "OperatingExpenses",
        "Assets", "AssetsCurrent", "Liabilities", "LiabilitiesCurrent",
        "StockholdersEquity", "CashAndCashEquivalentsAtCarryingValue",
        "NetCashProvidedByUsedInOperatingActivities",
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "LongTermDebtCurrent", "LongTermDebtNoncurrent", "ShortTermBorrowings",
        "CommonStockSharesOutstanding", "WeightedAverageNumberOfDilutedSharesOutstanding",
    }
    inserted = 0
    with store.db:
        for tag in wanted & set(facts):
            for unit, values in (facts[tag].get("units") or {}).items():
                for v in values:
                    if v.get("filed") and v.get("end") and v.get("val") is not None:
                        store.db.execute(
                            "INSERT OR IGNORE INTO filing_facts VALUES(?,?,?,?,?,?,?,?)",
                            (str(int(cik)), tag, v["end"], v["filed"], float(v["val"]),
                             unit, v.get("form"), v.get("accn")))
                        inserted += 1
    result = {"status": "completed", "symbol": row["symbol"],
              "attempted_on": today,
              "cik": str(int(cik)), "inserted": inserted}
    store.kv_set(f"filing_facts_attempted_{row['symbol']}", result)
    store.audit("filing_facts_ingested", result)
    return result


def ingest_filing_facts_batch(store, limit: int = 10, progress=None) -> dict:
    results = []
    for index in range(limit):
        if progress:
            progress(index, limit)
        result = ingest_next_filing_facts(store)
        if result.get("status") == "caught_up":
            break
        results.append(result)
    return {"status": "completed", "attempted": len(results),
            "inserted": sum(r.get("inserted", 0) for r in results),
            "failures": [r for r in results if r.get("status") == "failed"]}

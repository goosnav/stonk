"""Versioned company evidence dossiers and production signal adapters.

Deep research writes here; live nodes read here. This is the shared vertical
slice that prevents research reports from becoming disconnected UI artifacts.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime

from .models import new_id

SCHEMA_VERSION = "company-evidence.v1"
STANCES = {"attractive", "neutral", "avoid"}


def _json(value, fallback):
    if isinstance(value, type(fallback)):
        return value
    try:
        parsed = json.loads(value or "null")
        return parsed if isinstance(parsed, type(fallback)) else fallback
    except (TypeError, json.JSONDecodeError):
        return fallback


def source_ids(sources: dict) -> set[str]:
    catalog = sources.get("catalog") or []
    return {str(item.get("id")) for item in catalog if item.get("id")}


def validate_memo(raw: dict | None, allowed_sources: set[str]) -> tuple[dict | None, list[str]]:
    """Validate one AI vote and discard claims without supplied-source citations."""
    if not isinstance(raw, dict):
        return None, ["memo missing"]
    errors = []
    stance = str(raw.get("stance", "neutral")).lower()
    if stance not in STANCES:
        errors.append("invalid stance")
    try:
        confidence = max(0.0, min(1.0, float(raw.get("confidence", 0))))
        horizon = max(1, min(180, int(raw.get("horizon_days", 21))))
    except (TypeError, ValueError):
        return None, ["invalid confidence or horizon"]
    citations = []
    for citation in raw.get("citations") or []:
        if isinstance(citation, str):
            citation = {"source_id": citation, "claim": ""}
        if not isinstance(citation, dict):
            continue
        sid = str(citation.get("source_id", ""))
        if sid in allowed_sources:
            citations.append({"source_id": sid,
                              "claim": str(citation.get("claim", ""))[:240]})
    if stance != "neutral" and not citations:
        errors.append("directional memo has no verified citation")
    if errors:
        return None, errors
    memo = {
        "stance": stance, "confidence": confidence, "horizon_days": horizon,
        "thesis": str(raw.get("thesis", ""))[:800],
        "contrary_evidence": [str(v)[:300] for v in
                              (raw.get("contrary_evidence") or [])[:8]],
        "catalysts": [str(v)[:300] for v in (raw.get("catalysts") or [])[:8]],
        "thesis_breakers": [str(v)[:300] for v in
                            (raw.get("thesis_breakers") or [])[:8]],
        "citations": citations,
    }
    return memo, []


def deterministic_facts(rows: list[dict]) -> dict:
    """Compact, point-in-time facts. Unknown values remain explicitly absent."""
    latest = {}
    for row in sorted(rows, key=lambda r: (r.get("filed") or "", r.get("period_end") or ""),
                      reverse=True):
        latest.setdefault(row.get("tag"), row.get("value"))
    aliases = {
        "assets": ("Assets",), "equity": ("StockholdersEquity",),
        "revenue": ("RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues"),
        "net_income": ("NetIncomeLoss",),
        "operating_income": ("OperatingIncomeLoss",),
        "operating_cash_flow": ("NetCashProvidedByUsedInOperatingActivities",),
        "capex": ("PaymentsToAcquirePropertyPlantAndEquipment",),
        "cash": ("CashAndCashEquivalentsAtCarryingValue",),
        "debt_current": ("LongTermDebtCurrent", "ShortTermBorrowings"),
        "debt_noncurrent": ("LongTermDebtNoncurrent",),
        "shares": ("CommonStockSharesOutstanding",
                   "WeightedAverageNumberOfDilutedSharesOutstanding"),
        "current_assets": ("AssetsCurrent",),
        "current_liabilities": ("LiabilitiesCurrent",),
        "gross_profit": ("GrossProfit",),
        "eps": ("EarningsPerShareDiluted",),
    }
    facts = {name: next((latest.get(tag) for tag in tags if latest.get(tag) is not None), None)
             for name, tags in aliases.items()}
    ocf, capex = facts.get("operating_cash_flow"), facts.get("capex")
    facts["free_cash_flow"] = (float(ocf) - abs(float(capex))
                               if ocf is not None and capex is not None else None)
    debt_parts = [float(v) for v in
                  (facts.get("debt_current"), facts.get("debt_noncurrent"))
                  if v is not None]
    facts["total_debt"] = sum(debt_parts) if debt_parts else None
    assets, equity = facts.get("assets"), facts.get("equity")
    facts["equity_to_assets"] = (float(equity) / float(assets)
                                  if assets and equity is not None else None)
    revenue, gross, operating, net = (facts.get("revenue"), facts.get("gross_profit"),
                                      facts.get("operating_income"), facts.get("net_income"))
    facts["gross_margin"] = float(gross) / float(revenue) if revenue and gross is not None else None
    facts["operating_margin"] = float(operating) / float(revenue) \
        if revenue and operating is not None else None
    facts["net_margin"] = float(net) / float(revenue) if revenue and net is not None else None
    current_assets, current_liabilities = (facts.get("current_assets"),
                                           facts.get("current_liabilities"))
    facts["current_ratio"] = float(current_assets) / float(current_liabilities) \
        if current_assets is not None and current_liabilities else None
    # Annual comparisons use only facts that were actually filed by the
    # dossier's as-of date. Duplicate units/accessions collapse by period.
    annual = {}
    revenue_tags = set(aliases["revenue"])
    share_tags = set(aliases["shares"])
    for row in rows:
        if row.get("form") not in ("10-K", "10-K/A"):
            continue
        tag, period, value = row.get("tag"), row.get("period_end"), row.get("value")
        if not period or value is None or tag not in revenue_tags | share_tags:
            continue
        annual.setdefault((tag, period), float(value))
    def series(tags):
        by_period = {}
        for (tag, period), value in annual.items():
            if tag in tags:
                by_period.setdefault(period, value)
        return sorted(by_period.items(), reverse=True)
    revenues, shares = series(revenue_tags), series(share_tags)
    facts["revenue_growth"] = (revenues[0][1] / revenues[1][1] - 1
                               if len(revenues) > 1 and revenues[1][1] else None)
    facts["dilution_rate"] = (shares[0][1] / shares[1][1] - 1
                              if len(shares) > 1 and shares[1][1] else None)
    net_income, cash_flow = facts.get("net_income"), facts.get("operating_cash_flow")
    facts["accrual_ratio"] = ((float(net_income) - float(cash_flow)) / float(assets)
                              if net_income is not None and cash_flow is not None and assets
                              else None)
    facts["earnings_quality"] = (float(cash_flow) / abs(float(net_income))
                                 if cash_flow is not None and net_income else None)
    facts["available_fields"] = sum(v is not None for v in facts.values())
    return facts


def persist_dossier(store, symbol: str, as_of: str | None, sources: dict,
                    rows: list[dict], report: dict | None) -> dict:
    allowed = source_ids(sources)
    report = report or {}
    fundamental, fundamental_errors = validate_memo(
        report.get("fundamental") or report.get("business_fundamentals"), allowed)
    catalyst, catalyst_errors = validate_memo(
        report.get("catalyst") or report.get("company_catalyst"), allowed)
    facts = deterministic_facts(rows)
    source_hash = hashlib.sha256(json.dumps(
        {"schema": SCHEMA_VERSION, "sources": sources, "facts": facts},
        sort_keys=True, default=str).encode()).hexdigest()
    quality_parts = [min(1.0, facts["available_fields"] / 8)]
    if fundamental: quality_parts.append(min(1.0, len(fundamental["citations"]) / 2))
    if catalyst: quality_parts.append(min(1.0, len(catalyst["citations"]) / 2))
    quality = round(sum(quality_parts) / len(quality_parts), 4)
    status = "ready" if fundamental or catalyst else "unsupported"
    error = "; ".join(fundamental_errors + catalyst_errors)[:500] or None
    dossier_id = new_id()
    created = datetime.now().astimezone().isoformat(timespec="seconds")
    with store.db:
        store.db.execute(
            "INSERT INTO company_evidence VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (dossier_id, symbol.upper(), as_of, created, source_hash,
             json.dumps(sources), json.dumps(facts), json.dumps(fundamental),
             json.dumps(catalyst), quality, status, error))
    result = {"id": dossier_id, "schema": SCHEMA_VERSION, "symbol": symbol.upper(),
              "as_of": as_of, "created_at": created, "source_hash": source_hash,
              "sources": sources, "facts": facts, "fundamental_memo": fundamental,
              "catalyst_memo": catalyst, "quality": quality,
              "status": status, "error": error}
    store.audit("company_evidence_persisted", {
        "id": dossier_id, "symbol": symbol.upper(), "status": status,
        "quality": quality, "source_hash": source_hash,
    })
    return result


def latest_dossier(store, symbol: str, as_of: str | None = None) -> dict | None:
    q = "SELECT * FROM company_evidence WHERE symbol=?"
    args: list = [symbol.upper()]
    if as_of:
        q += " AND (as_of IS NULL OR as_of<=?)"; args.append(as_of)
    q += " ORDER BY created_at DESC LIMIT 1"
    row = store.db.execute(q, args).fetchone()
    if not row:
        return None
    out = dict(row)
    for key, fallback in (("sources", {}), ("facts", {}),
                          ("fundamental_memo", {}), ("catalyst_memo", {})):
        out[key] = _json(out.get(key), fallback)
    out["schema"] = SCHEMA_VERSION
    return out


def ai_budget_status(cfg, store) -> dict:
    ai = cfg.get("ai", default={}) or {}
    rows = store.db.execute(
        "SELECT purpose,COUNT(*) calls,SUM(cache_hit) cache_hits,SUM(cost_usd) cost "
        "FROM ai_ledger WHERE day LIKE strftime('%Y-%m','now','localtime') || '%' "
        "GROUP BY purpose ORDER BY cost DESC").fetchall()
    monthly = float(ai.get("monthly_budget_usd", 40.0))
    spent = float(store.ai_spend_month())
    return {"monthly_budget_usd": monthly, "spent_month_usd": round(spent, 4),
            "remaining_month_usd": round(max(0.0, monthly - spent), 4),
            "categories": [dict(r) for r in rows]}

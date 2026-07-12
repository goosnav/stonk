from __future__ import annotations

from datetime import date

import numpy as np
from fastapi.testclient import TestClient

from specforge.montecarlo import block_bootstrap
from specforge.research import (cancel_job, deep_research, discover_opportunities,
                                enqueue_job, latest_sec_filing, list_jobs,
                                resolve_forecasts, run_operator_job)
from specforge.universe import parse_directory, refresh_membership, symbols


def test_official_directory_parser_filters_non_companies():
    text = "Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares\n" \
           "AAA|Alpha Corp Common Stock|Q|N|N|100|N|N\n" \
           "TEST|Test Issue|Q|Y|N|100|N|N\n" \
           "FUND|Fund ETF|Q|N|N|100|Y|N\n" \
           "WRT|Issuer Warrant|Q|N|N|100|N|N\n" \
           "WRTS|Issuer Warrants|Q|N|N|100|N|N\n" \
           "RGHT|Issuer Rights|Q|N|N|100|N|N\n" \
           "ADR|World Depositary Shares|Q|N|N|100|N|N\n" \
           "File Creation Time: 0101200012:00|||||||\n"
    rows = parse_directory(text)
    assert [r["symbol"] for r in rows] == ["AAA", "ADR"]
    assert rows[1]["is_adr"] == 1


def test_tier_snapshot_uses_liquidity_and_history(cfg, store):
    today = date.today().isoformat()
    with store.db:
        for sym in ("AAA", "BBB", "CCC"):
            store.db.execute("INSERT INTO instruments VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                             (sym, sym, "NASDAQ", "common", 0, 0, 1, today,
                              today, "test", None, sym))
    cfg.data["universe"].update({"research_min_dollar_volume": 1,
                                  "execution_min_dollar_volume": 1,
                                  "research_size": 2, "active_size": 2,
                                  "min_history_bars": 10})
    result = refresh_membership(cfg, store)
    assert result["research"] == 2 and result["active"] == 2
    assert len(symbols(store, "research")) == 2


def test_tier_rerank_preserves_discovery_shortlist(cfg, store):
    as_of = store.latest_bar_date("SPY")
    store.db.execute("INSERT INTO universe_membership VALUES(?,?,?,?,?,?)",
                     (as_of, "AAA", "shortlist", 1, "test", "{}"))
    store.db.commit()
    refresh_membership(cfg, store, as_of)
    assert symbols(store, "shortlist") == ["AAA"]


def test_shadow_forecast_resolution(cfg, store):
    as_of = store.get_bars("AAA", "9999-12-31", 1000)[-30]["d"]
    store.db.execute("INSERT INTO model_forecasts VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                     ("m", as_of, "AAA", 5, -.1, .02, .1, .6, None, None, "f"))
    store.db.commit()
    assert resolve_forecasts(store) == 1
    row = store.db.execute("SELECT * FROM model_forecasts").fetchone()
    assert row["resolved_at"] and row["realized_excess"] is not None


def test_block_bootstrap_is_deterministic_and_preserves_cash():
    rng = np.random.default_rng(11)
    common = rng.normal(0.0005, 0.01, 180)
    returns = np.column_stack((common, common * 0.8 + rng.normal(0, 0.002, 180)))
    first = block_bootstrap(100, returns, [.4, .4], n_paths=400, seed=3)
    second = block_bootstrap(100, returns, [.4, .4], n_paths=400, seed=3)
    assert first == second
    assert len(first.percentile_paths["p50"]) == 22
    assert 0 <= first.probability_drawdown_gt_10 <= 1


def test_research_model_and_universe_apis_expose_real_state(cfg, store):
    from specforge.app import create_app
    client = TestClient(create_app(cfg, store, with_scheduler=False))
    graph = client.get("/api/model/graph?symbol=AAA&horizon=5").json()
    assert graph["symbol"] == "AAA"
    assert graph["horizon"] == 5
    assert len(graph["topology"]["nodes"]) > 10
    assert graph["topology"]["edges"]
    assert graph["effective_live_blend"] == 0
    assert graph["topology_provenance"] == "initial_prior"
    assert client.get("/api/research").status_code == 200
    universe = client.get("/api/universe?tier=active&limit=10").json()
    assert universe["tier"] == "active" and universe["limit"] == 10


def test_research_jobs_are_durable_deduplicated_and_cancellable(store):
    first = enqueue_job(store, "discover")
    assert enqueue_job(store, "discover")["id"] == first["id"]
    assert list_jobs(store)[0]["status"] == "queued"
    assert cancel_job(store, first["id"])["status"] == "cancelled"


def test_research_job_api_contract(cfg, store):
    from specforge.app import create_app
    client = TestClient(create_app(cfg, store, with_scheduler=False))
    created = client.post("/api/research/jobs", json={"kind": "train_holdings"})
    assert created.status_code == 200 and created.json()["status"] == "queued"
    assert client.get("/api/research/jobs").json()[0]["kind"] == "train_holdings"
    cancelled = client.post(f"/api/research/jobs/{created.json()['id']}/cancel")
    assert cancelled.json()["status"] == "cancelled"
    engine = client.get("/api/engine").json()
    assert "trading" in engine["processes"] and "research" in engine["processes"]


def test_discovery_persists_exactly_25_without_ai(cfg, store):
    from conftest import synth_bars
    as_of = store.latest_bar_date("SPY")
    with store.db:
        for i in range(30):
            sym = f"T{i:02d}"
            store.db.execute("INSERT INTO instruments VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                             (sym, sym, "NASDAQ", "common", 0, 0, 1, as_of,
                              as_of, "test", None, sym))
            store.db.execute("INSERT INTO universe_membership VALUES(?,?,?,?,?,?)",
                             (as_of, sym, "research", i + 1, "test",
                              '{"dollar_volume":100000000}'))
            store.upsert_bars(sym, synth_bars(daily_drift=.001 + i / 100000), "test")
    result = discover_opportunities(cfg, store)
    assert result["status"] == "completed" and len(result["shortlist"]) == 25
    assert len(symbols(store, "shortlist")) == 25


def test_deep_research_fails_closed_when_ai_disabled(cfg, store):
    result = deep_research(cfg, store)
    assert result["status"] == "skipped" and "AI" in result["reason"]


def test_deep_research_job_runs_discovery_dependency_first(cfg, store):
    deep = enqueue_job(store, "deep_research")
    result = run_operator_job(cfg, store)
    assert result["kind"] == "discover"
    assert next(j for j in list_jobs(store) if j["id"] == deep["id"])["status"] == "queued"


def test_latest_sec_filing_extracts_narrative_and_source_hash():
    class Response:
        def __init__(self, payload=None, text=""):
            self._payload, self.text = payload, text
            self.content = text.encode()
        def json(self): return self._payload
        def raise_for_status(self): return None
    class Client:
        @staticmethod
        def get(url, **kwargs):
            if "submissions" in url:
                return Response({"filings": {"recent": {
                    "form": ["8-K", "10-Q"], "accessionNumber": ["x", "1-2-3"],
                    "primaryDocument": ["x.htm", "q.htm"],
                    "filingDate": ["2026-01-01", "2026-02-01"]}}})
            return Response(text="<html><style>no</style><body>Revenue grew strongly.</body></html>")
    filing = latest_sec_filing("123", Client)
    assert filing["form"] == "10-Q" and "Revenue grew" in filing["text"]
    assert len(filing["sha256"]) == 64 and filing["url"].endswith("/q.htm")

from __future__ import annotations

from datetime import date

import numpy as np
from fastapi.testclient import TestClient

from specforge.montecarlo import block_bootstrap
from specforge.research import resolve_forecasts
from specforge.universe import parse_directory, refresh_membership, symbols


def test_official_directory_parser_filters_non_companies():
    text = "Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares\n" \
           "AAA|Alpha Corp Common Stock|Q|N|N|100|N|N\n" \
           "TEST|Test Issue|Q|Y|N|100|N|N\n" \
           "FUND|Fund ETF|Q|N|N|100|Y|N\n" \
           "WRT|Issuer Warrant|Q|N|N|100|N|N\n" \
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
    assert client.get("/api/research").status_code == 200
    universe = client.get("/api/universe?tier=active&limit=10").json()
    assert universe["tier"] == "active" and universe["limit"] == 10

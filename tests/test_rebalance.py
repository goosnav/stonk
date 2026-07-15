from __future__ import annotations

from specforge.data import MarketContext
from specforge.models import AccountState, Position, TradeCandidate, new_id
from specforge.portfolio import fit_to_capacity, rebalance_plan


def candidate(symbol: str, score: float, expected: float = .04):
    return TradeCandidate(new_id(), symbol, "equity", "buy", "test", score, 0,
                          expected, -.1, .1, .6, .1, -.2, .4, 21, 0,
                          ["momentum"])


def test_rebalance_partial_trims_are_ranked_and_bounded(cfg, store):
    cfg.data["risk"].update({"max_account_deployment": .70,
                              "min_cash_reserve": .30,
                              "max_single_equity_position": .25,
                              "max_open_positions": 4,
                              "max_rebalance_turnover_per_cycle": .30,
                              "max_rebalance_sells_per_cycle": 2})
    ctx = MarketContext(store, cfg)
    positions = []
    for sym, score in (("AAA", .05), ("BBB", .10), ("CCC", .15)):
        store.save_position(new_id(), {
            "symbol": sym, "asset_type": "equity", "qty": 1, "avg_cost": 100,
            "opened_at": "2020-01-01T00:00:00", "horizon_days": 21,
            "stop_price": 80, "candidate_id": "", "nodes": [],
            "option_symbol": None, "status": "open", "mode": "paper"})
        positions.append(Position(sym, "equity", 1, 100, "2020-01-01"))
    account = AccountState(1000, 700, 700, positions, ctx.as_of)
    # Two consecutive cycles outside target are sufficient replacement evidence.
    rebalance_plan([candidate("DDD", .8)], account, ctx, cfg)
    plan = rebalance_plan([candidate("DDD", .8)], account, ctx, cfg)
    assert len(plan["sells"]) == 2
    assert plan["sells"][0]["held_score"] <= plan["sells"][1]["held_score"]
    assert plan["turnover"] <= account.equity * .30 + 1e-8
    assert all(0 < sell["qty"] <= .5 for sell in plan["sells"])
    assert all(sell["notional"] >= 5 for sell in plan["sells"])


def test_rebalance_hysteresis_resets_when_evidence_model_changes(cfg, store):
    cfg.data["risk"].update({"max_account_deployment": .70,
                              "min_cash_reserve": .30,
                              "max_single_equity_position": .25,
                              "max_open_positions": 4})
    ctx = MarketContext(store, cfg)
    store.save_position(new_id(), {
        "symbol": "AAA", "asset_type": "equity", "qty": 1, "avg_cost": 100,
        "opened_at": "2020-01-01T00:00:00", "horizon_days": 21,
        "stop_price": 80, "candidate_id": "", "nodes": [],
        "option_symbol": None, "status": "open", "mode": "paper"})
    account = AccountState(1000, 900, 900,
                           [Position("AAA", "equity", 1, 100, "2020-01-01")],
                           ctx.as_of)
    store.kv_set("rebalance_outside_counts", {"AAA": 20})
    store.kv_set("rebalance_counter_version", "legacy-flat-score")
    new = candidate("DDD", .8)
    new.evidence_version = "evidence.v1-new"

    first = rebalance_plan([new], account, ctx, cfg)
    second = rebalance_plan([new], account, ctx, cfg)

    assert first["sells"] == []
    assert second["sells"]
    assert store.kv_get("rebalance_counter_version") == "evidence.v1-new"


def test_capacity_keeps_best_affordable_order_instead_of_dropping_batch(cfg):
    cfg.data["risk"].update({"max_account_deployment": .98,
                              "min_cash_reserve": .02,
                              "max_new_positions_per_cycle": 3})
    account = AccountState(100, 7.77, 7.77, [], "2026-07-15")
    ranked = [(candidate("AAA", .8), 16), (candidate("BBB", .7), 14),
              (candidate("CCC", .6), 12)]
    fitted = fit_to_capacity(ranked, account, cfg, budget=50)
    assert len(fitted) == 1
    assert fitted[0][0].symbol == "AAA"
    assert fitted[0][1] == 7.77


def test_missing_company_dossier_cannot_be_treated_as_bearish_exit(cfg, store):
    cfg.data["risk"].update({"max_account_deployment": .70,
                              "min_cash_reserve": .30,
                              "max_single_equity_position": .25,
                              "max_open_positions": 4})
    ctx = MarketContext(store, cfg)
    store.save_position(new_id(), {
        "symbol": "AAA", "asset_type": "equity", "qty": 1, "avg_cost": 100,
        "opened_at": "2020-01-01T00:00:00", "horizon_days": 21,
        "stop_price": 80, "candidate_id": "", "nodes": [],
        "option_symbol": None, "status": "open", "mode": "paper"})
    account = AccountState(1000, 900, 900,
                           [Position("AAA", "equity", 1, 100, "2020-01-01")],
                           ctx.as_of)
    better = candidate("DDD", .9)
    better.evidence_version = "evidence.v2"
    rebalance_plan([better], account, ctx, cfg)  # establish model version
    plan = rebalance_plan([better], account, ctx, cfg)
    assert plan["sells"] == []
    assert plan["deferred_sells"] == [
        {"symbol": "AAA", "reason": "awaiting verified company dossier"}]

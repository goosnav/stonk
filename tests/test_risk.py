"""Risk governor tests — every constraint in AGENTS.md §31 that exists in Phase 1."""
from __future__ import annotations

from datetime import datetime

import pytest

from specforge.models import AccountState, OrderIntent, Position, TradeCandidate, new_id
from specforge.risk import CycleState, Governor


def make_candidate(symbol="AAA", notional=50.0, asset_type="equity", **kw):
    return TradeCandidate(
        id=new_id(), symbol=symbol, asset_type=asset_type, side="buy",
        thesis="test", final_score=0.4, target_notional=notional,
        expected_return=0.02, ci_low=-0.05, ci_high=0.09, probability_positive=0.6,
        expected_apr=0.2, apr_ci_low=-0.4, apr_ci_high=0.9, horizon_days=20,
        max_loss=notional, contributing_nodes=["momentum"], **kw)


def make_intent(cand, price=100.0):
    return OrderIntent.make(cand, qty=cand.target_notional / price, limit_price=price)


def acct(equity=1000.0, cash=1000.0, positions=None):
    return AccountState(equity=equity, cash=cash, buying_power=cash,
                        positions=positions or [],
                        as_of=datetime.now().isoformat())


@pytest.fixture()
def gov(cfg, store):
    return Governor(cfg, store)


def test_approves_normal_order(gov):
    c = make_candidate(notional=50)
    d = gov.review(make_intent(c), c, acct(), CycleState(100), data_age_days=1)
    assert d.verdict == "APPROVED"


def test_time_step_budget_reduces_then_rejects(gov):
    cycle = CycleState(60)
    c1 = make_candidate("AAA", notional=80)
    d1 = gov.review(make_intent(c1), c1, acct(), cycle, 1)
    assert d1.verdict == "APPROVED_WITH_SIZE_REDUCTION"
    assert d1.approved_notional == pytest.approx(60)
    cycle.budget_used = 60
    c2 = make_candidate("BBB", notional=40)
    d2 = gov.review(make_intent(c2), c2, acct(), cycle, 1)
    assert d2.verdict == "REJECTED"
    assert any("budget" in r for r in d2.reasons)


def test_stale_data_rejected(gov):
    c = make_candidate()
    d = gov.review(make_intent(c), c, acct(), CycleState(100), data_age_days=9)
    assert d.verdict == "REJECTED"
    d = gov.review(make_intent(c), c, acct(), CycleState(100), data_age_days=None)
    assert d.verdict == "REJECTED"


def test_duplicate_order_rejected(gov, store):
    c = make_candidate("AAA", notional=20)
    intent = make_intent(c)
    intent.status = "filled"
    store.record_order(intent)
    d = gov.review(make_intent(c), c, acct(), CycleState(100), 1)
    assert d.verdict == "REJECTED"
    assert any("duplicate" in r for r in d.reasons)


def test_kill_switch_blocks_buys_not_sells(gov, store):
    gov.trip("drawdown", "test trip")
    c = make_candidate(notional=20)
    d = gov.review(make_intent(c), c, acct(), CycleState(100), 1)
    assert d.verdict == "REJECTED"
    sell = make_candidate(notional=20)
    sell_intent = make_intent(sell)
    sell_intent.side = "sell"
    d = gov.review(sell_intent, sell, acct(), CycleState(100), 1)
    assert d.verdict == "APPROVED"


def test_single_position_cap(gov):
    # 8% of 1000 = $80 cap; existing $70 held → only $10 room
    held = [Position(symbol="AAA", asset_type="equity", qty=0.7, avg_cost=100,
                     opened_at="2026-01-01")]
    c = make_candidate("AAA", notional=50)
    d = gov.review(make_intent(c), c, acct(positions=held), CycleState(1000), 1)
    assert d.verdict == "APPROVED_WITH_SIZE_REDUCTION"
    assert d.approved_notional == pytest.approx(10, abs=0.5)


def test_max_open_positions(gov):
    held = [Position(symbol=f"S{i}", asset_type="equity", qty=0.1, avg_cost=100,
                     opened_at="2026-01-01") for i in range(12)]
    c = make_candidate("NEW", notional=20)
    d = gov.review(make_intent(c), c, acct(positions=held), CycleState(100), 1)
    assert d.verdict == "REJECTED"


def test_approval_threshold(gov):
    # 10% of equity = $100 → $150 order queues for human approval
    c = make_candidate(notional=150)
    d = gov.review(make_intent(c), c, acct(equity=1000, cash=1000),
                   CycleState(10000), 1)
    assert d.verdict == "REQUIRES_HUMAN_APPROVAL"


def test_options_locked_at_small_scale(gov):
    c = make_candidate(asset_type="option", notional=50,
                       option_details={"dte": 45, "delta": 0.5, "spread_pct": 0.05,
                                       "open_interest": 500})
    d = gov.review(make_intent(c), c, acct(equity=1000), CycleState(100), 1)
    assert d.verdict == "REJECTED"
    assert any("locked" in r for r in d.reasons)
    # unlocked at scale: equity 10k × 1.5% = $150 ≥ $75 minimum premium
    d = gov.review(make_intent(c), c, acct(equity=10000, cash=10000), CycleState(500), 1)
    assert d.verdict in ("APPROVED", "APPROVED_WITH_SIZE_REDUCTION")


def test_option_bounded_risk_validation(gov):
    bad = {"dte": 3, "delta": 0.9, "spread_pct": 0.5, "open_interest": 5}
    flags = gov.validate_option(bad)
    assert set(flags) == {"dte_out_of_range", "delta_out_of_range",
                          "spread_too_wide", "open_interest_too_low"}
    assert "missing_dte" in gov.validate_option({})  # unknown risk = risk


def test_daily_loss_kill_switch_trips(gov, store):
    from datetime import date, timedelta
    store.record_equity(1000, 1000, "paper", d=(date.today() - timedelta(days=1)).isoformat())
    gov.check_kill_switches(acct(equity=970), "paper")   # -3% > 2% limit
    assert "daily_loss" in gov.active_switches()


def test_dangerous_config_rejected():
    from specforge.config import ConfigError, load_config
    with pytest.raises(ConfigError):
        load_config("paper", overrides={"risk": {"max_daily_loss": 0.5}})
    # but allowed with explicit override
    cfg = load_config("paper", overrides={"risk": {"max_daily_loss": 0.5},
                                          "advanced_override": True})
    assert cfg.validate()  # returns warnings


def test_live_triple_gate(monkeypatch):
    from specforge.config import load_config
    cfg = load_config("paper", overrides={"live_trading_enabled": True,
                                          "broker": "robinhood_mcp"})
    # isolate from the machine's real .env (config.py loads it at import)
    monkeypatch.delenv("LIVE_TRADING_ENABLED", raising=False)
    monkeypatch.delenv("RH_ACCOUNT_WHITELIST", raising=False)
    ok, why = cfg.live_trading_allowed()
    assert not ok and "env" in why
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "true")
    ok, why = cfg.live_trading_allowed()
    assert not ok and "WHITELIST" in why
    monkeypatch.setenv("RH_ACCOUNT_WHITELIST", "acct123")
    ok, _ = cfg.live_trading_allowed()
    assert ok


def test_drawdown_trip_clears_and_baseline_resets(cfg, store):
    """D17: after the cooldown, the switch clears, the HWM resets to current
    equity, and the same depressed equity does NOT re-trip the switch."""
    from datetime import date, timedelta
    from specforge.risk import Governor
    today = date.today()
    store.record_equity(1000, 1000, "paper", d=(today - timedelta(days=30)).isoformat())
    g1 = Governor(cfg, store)
    g1.check_kill_switches(acct(equity=800), "paper")        # -20% from peak
    assert "drawdown" in g1.active_switches()
    # 11 days later (cooldown 10): switch auto-clears + baseline resets
    future = (today + timedelta(days=11)).isoformat() + "T10:00:00"
    g2 = Governor(cfg, store, now_iso=future)
    assert "drawdown" not in g2.active_switches()
    assert store.kv_get("dd_peak_reset_d") == future[:10]
    # same equity level must not re-trip against the OLD peak
    store.record_equity(800, 800, "paper", d=future[:10])
    g2.check_kill_switches(acct(equity=800), "paper")
    assert "drawdown" not in g2.active_switches()

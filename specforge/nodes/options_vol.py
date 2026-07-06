"""Options volatility module (AGENTS.md §10.13, §22) — long calls/puts only,
max loss = premium, auto-locked below the account-scale threshold (D5).

Shape: this is an OVERLAY, not a scoring node (Node.compute returns []). After
portfolio construction, convexity_overlay() may convert the single highest-
conviction equity target into a long CALL when:
  - node enabled AND governor.options_unlocked(account)
  - candidate final_score ≥ MIN_SCORE (strong directional conviction)
  - a chain contract passes §22: 30–60 DTE preferred, BS-delta in [0.25,0.70],
    spread ≤ 15%, OI ≥ 100, and IV isn't absurdly rich vs realized vol (≤1.3×)
  - whole contracts fit inside min(per-trade premium cap, remaining budget)
The governor still validates option_details independently — this module can
propose, never bypass.

MVP limits (documented, deliberate): puts not generated yet (long-only book has
no short signals to hedge); option positions exit on TIME only (premium isn't
marked without live chain data — bounded loss = premium paid, which is the risk
contract the user accepted). RH adapter support for option orders lands after
equity live probation.
"""
from __future__ import annotations

import math
from datetime import datetime

from ..data import MarketContext
from ..models import TradeCandidate, new_id
from .base import SignalNode

MIN_SCORE = 0.45
IV_RV_MAX = 1.3
DTE_WINDOW = (25, 75)


class Node(SignalNode):
    version = "1"
    role = "overlay"

    def compute(self, ctx: MarketContext):
        return []                    # overlay — engine calls convexity_overlay()


def _bs_call_delta(spot, strike, iv, dte_years, r=0.04) -> float:
    if iv <= 0 or dte_years <= 0:
        return 0.0
    d1 = (math.log(spot / strike) + (r + iv * iv / 2) * dte_years) / (iv * math.sqrt(dte_years))
    return 0.5 * (1 + math.erf(d1 / math.sqrt(2)))


def convexity_overlay(targets, ctx: MarketContext, account, cfg, governor,
                      store, log=print):
    """Maybe replace the top equity target with a long-call candidate.
    Returns the (possibly modified) targets list."""
    node_cfg = cfg.get("nodes", "options_vol", default={}) or {}
    if not node_cfg.get("enabled") or not targets:
        return targets
    if not governor.options_unlocked(account):
        return targets
    if ctx.offline:
        return targets               # no point-in-time chains — never in backtest

    cand, notional = targets[0]
    if cand.final_score < MIN_SCORE or cand.asset_type != "equity":
        return targets

    spot = ctx.close(cand.symbol)
    c20 = ctx.closes(cand.symbol).pct_change().dropna().tail(20)
    realized = float(c20.std()) * math.sqrt(252) if len(c20) >= 15 else None
    if not spot or not realized:
        return targets

    pick = _pick_call(cand.symbol, spot, realized, cfg, log)
    if not pick:
        return targets

    premium_cap = account.equity * cfg.get("risk", "max_single_option_premium_risk",
                                           default=0.015)
    contracts = int(min(premium_cap, notional) // (pick["premium"] * 100))
    if contracts < 1:
        return targets

    option_cand = TradeCandidate(
        id=new_id(), symbol=cand.symbol, asset_type="option", side="buy",
        thesis=f"convexity overlay on {cand.symbol} (score {cand.final_score}): "
               f"{pick['desc']}",
        final_score=cand.final_score, target_notional=contracts * pick["premium"] * 100,
        expected_return=cand.expected_return * 3,     # convexity, honestly rough
        ci_low=-1.0, ci_high=cand.ci_high * 4,        # can lose 100% of premium
        probability_positive=max(0.0, cand.probability_positive - 0.1),
        expected_apr=0, apr_ci_low=0, apr_ci_high=0,
        horizon_days=min(cand.horizon_days, pick["dte"] - 7),
        max_loss=contracts * pick["premium"] * 100,
        contributing_nodes=cand.contributing_nodes + ["options_vol"],
        confidence_label="low", regime=cand.regime,
        option_symbol=pick["occ"], option_details={**pick, "contracts": contracts})
    store.audit("convexity_overlay", {"symbol": cand.symbol, "pick": pick,
                                      "contracts": contracts})
    return [(option_cand, option_cand.target_notional)] + targets[1:]


def _pick_call(symbol: str, spot: float, realized: float, cfg, log) -> dict | None:
    """Best §22-compliant call from the yfinance chain, or None."""
    try:
        import yfinance as yf
        tk = yf.Ticker(symbol)
        today = datetime.now().date()
        best = None
        for exp in tk.options or []:
            dte = (datetime.strptime(exp, "%Y-%m-%d").date() - today).days
            if not (DTE_WINDOW[0] <= dte <= DTE_WINDOW[1]):
                continue
            calls = tk.option_chain(exp).calls
            for _, row in calls.iterrows():
                bid, ask = float(row.get("bid") or 0), float(row.get("ask") or 0)
                oi = int(row.get("openInterest") or 0)
                iv = float(row.get("impliedVolatility") or 0)
                strike = float(row["strike"])
                if bid <= 0 or ask <= 0 or oi < cfg.get("risk", "option_min_open_interest", default=100):
                    continue
                mid = (bid + ask) / 2
                spread_pct = (ask - bid) / mid if mid else 1.0
                if spread_pct > cfg.get("risk", "option_max_spread_pct", default=0.15):
                    continue
                if iv / realized > IV_RV_MAX:
                    continue          # paying up for vol kills the trade (§19.5)
                delta = _bs_call_delta(spot, strike, iv, dte / 365)
                if not (cfg.get("risk", "option_min_delta", default=0.25) <= delta
                        <= cfg.get("risk", "option_max_delta", default=0.70)):
                    continue
                # prefer ~0.5 delta, tighter spread
                quality = -abs(delta - 0.5) - spread_pct
                if best is None or quality > best["quality"]:
                    strike_code = f"{int(round(strike * 1000)):08d}"
                    best = {"quality": quality, "premium": round(mid, 2),
                            "strike": strike, "dte": dte, "delta": round(delta, 3),
                            "spread_pct": round(spread_pct, 3), "open_interest": oi,
                            "iv": round(iv, 3), "realized_vol": round(realized, 3),
                            "expiry": exp,
                            "occ": f"{symbol}{exp[2:].replace('-', '')}C{strike_code}",
                            "desc": f"{exp} {strike}C @{mid:.2f} δ{delta:.2f} "
                                    f"IV{iv:.0%}/RV{realized:.0%}"}
            if best:
                break                 # first qualifying expiry window is fine
        return {k: v for k, v in best.items() if k != "quality"} if best else None
    except Exception as e:            # noqa: BLE001
        log(f"options_vol: chain fetch failed for {symbol}: {e}")
        return None

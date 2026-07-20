"""Risk governor: the deterministic final gate. Nothing trades without passing
review(); AI has no path around it (dev/ARCHITECTURE.md invariant #2).

Two layers:
1. Kill switches — evaluated once per cycle (check_kill_switches). Tripped
   switches block ALL new entries; exits remain allowed. Recoverable switches
   auto-clear after a stated cooldown. A switch without a clear time is a
   deliberately major, human-action-required condition.
2. Per-order review — the time-step budget (primary check, dev/DECISIONS.md D3)
   plus deployment/position/count/duplicate/staleness/approval checks. The
   governor may REDUCE size to fit caps rather than reject outright.

Worst-case loss definition (user-agreed): full notional for equities, full
premium for options. The budget bounds the max bleed per scan cycle.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

from .models import AccountState, OrderIntent, RiskDecision, TradeCandidate
from .store import Store

KILL_KEY = "kill_switches"          # kv: {switch_name: {reason, tripped_at, auto_clear}}
MIN_ORDER_NOTIONAL = 5.0            # below this fills are noise; reject


class CycleState:
    """Mutable per-scan-cycle accounting shared across order reviews."""

    def __init__(self, budget: float):
        self.budget = budget            # $ worst-case-loss budget for this cycle
        self.budget_used = 0.0
        self.new_positions = 0

    @property
    def budget_left(self) -> float:
        return max(0.0, self.budget - self.budget_used)


class Governor:
    def __init__(self, cfg, store: Store, now_iso: str | None = None):
        self.cfg = cfg
        self.store = store
        self.r = cfg.get("risk", default={})
        # logical clock: backtester injects historical timestamps; live = real now
        self.now_iso = now_iso or datetime.now().astimezone().isoformat()
        # orders in the shared DB are tagged by mode (D26): daily caps and the
        # duplicate cooldown must count only same-mode orders
        self.mode = "live" if cfg.mode == "live" else "paper"

    @property
    def today(self) -> str:
        return self.now_iso[:10]

    def _today_dt(self) -> date:
        return date.fromisoformat(self.today)

    # ---------------- kill switches ----------------
    def active_switches(self) -> dict:
        switches = self.store.kv_get(KILL_KEY, {}) or {}
        today = self.today
        now = datetime.fromisoformat(self.now_iso)
        def expired(v: dict) -> bool:
            if v.get("clear_at"):
                return datetime.fromisoformat(v["clear_at"]) <= now
            return bool(v.get("auto_clear") and v.get("clear_on", "") <= today)
        # auto-clear switches whose window rolled over
        cleared = {k: v for k, v in switches.items() if not expired(v)}
        if cleared != switches:
            if "drawdown" in switches and "drawdown" not in cleared:
                self._reset_drawdown_baseline("auto_clear")
            if "rejected_orders" in switches and "rejected_orders" not in cleared:
                # The rejected rows that caused the cooldown are history, not
                # a reason to re-trip immediately after automatic recovery.
                self.store.kv_set("rejected_orders_reset_ts", self.now_iso)
            for name in switches.keys() - cleared.keys():
                self.store.audit("kill_switch_auto_cleared", {"name": name})
            self.store.kv_set(KILL_KEY, cleared)
        return cleared

    def _reset_drawdown_baseline(self, via: str) -> None:
        """D17: when a drawdown trip clears, the high-water mark restarts at
        the clear date. Without this, an all-cash account sitting 15% below an
        old peak re-trips forever (backtest v2 finding: 3 years frozen)."""
        self.store.kv_set("dd_peak_reset_d", self.today)
        self.store.audit("drawdown_baseline_reset", {"date": self.today, "via": via})

    def trip(self, name: str, reason: str, auto_clear_days: int | None = None,
             auto_clear_minutes: int | None = None) -> None:
        switches = self.store.kv_get(KILL_KEY, {}) or {}
        # Preserve the original trip/cooldown. Re-evaluating the same problem
        # every cycle must not move the recovery window forever into the future.
        if name in switches:
            return
        entry = {"reason": reason, "tripped_at": self.now_iso,
                 "severity": "cooldown" if auto_clear_days is not None or
                 auto_clear_minutes is not None else "major",
                 "requires_human": auto_clear_days is None and auto_clear_minutes is None}
        if auto_clear_days is not None:
            entry["auto_clear"] = True
            entry["clear_on"] = (self._today_dt() + timedelta(days=auto_clear_days)).isoformat()
        if auto_clear_minutes is not None:
            entry["auto_clear"] = True
            entry["clear_at"] = (datetime.fromisoformat(self.now_iso) +
                                  timedelta(minutes=auto_clear_minutes)).isoformat()
        switches[name] = entry
        self.store.kv_set(KILL_KEY, switches)
        self.store.audit("kill_switch_tripped", {"name": name, **entry})

    def reset(self, name: str) -> None:
        switches = self.store.kv_get(KILL_KEY, {}) or {}
        if switches.pop(name, None):
            self.store.kv_set(KILL_KEY, switches)
            self.store.audit("kill_switch_reset", {"name": name})
            if name == "drawdown":
                self._reset_drawdown_baseline("manual_reset")
            if name == "rejected_orders":
                # D39: without this baseline, the switch re-trips on the NEXT
                # cycle (today's rejection count is still over the limit) and
                # the human's reset is silently undone until midnight
                self.store.kv_set("rejected_orders_reset_ts", self.now_iso)

    def _mark(self, symbol: str) -> float | None:
        """Latest settled close — marked exposure, not stale cost basis."""
        row = self.store.db.execute(
            "SELECT close FROM bars WHERE symbol=? ORDER BY d DESC LIMIT 1",
            (symbol,)).fetchone()
        return float(row["close"]) if row and row["close"] else None

    def _marked_value(self, p) -> float:
        """Position exposure at MARKET (R4): an appreciated position must not
        slip past its cap at cost. Options fall back to premium cost basis
        (no reliable mark source here)."""
        if p.asset_type == "option":
            return p.cost_basis
        mark = self._mark(p.symbol)
        return p.qty * (mark if mark else p.avg_cost)

    def _pending_buy_notional(self, symbol: str | None = None) -> float:
        """Open buy orders reserve exposure BEFORE cash/positions update —
        without this, several same-cycle orders could each see full room."""
        q = ("SELECT COALESCE(SUM(notional),0) n FROM orders WHERE side='buy' "
             "AND mode=? AND status IN ('reviewed','placed','pending_approval')")
        args: list = [self.mode]
        if symbol:
            q += " AND symbol=?"
            args.append(symbol)
        return float(self.store.db.execute(q, args).fetchone()["n"])

    def _sector(self, symbol: str) -> str | None:
        row = self.store.db.execute(
            "SELECT sector FROM instruments WHERE symbol=?", (symbol,)).fetchone()
        return row["sector"] if row and row["sector"] else None

    def check_kill_switches(self, account: AccountState, source: str) -> dict:
        """Run at cycle start. Evaluates loss/drawdown limits against the equity
        curve and trips switches. Returns the active set."""
        eq = account.equity
        today = self._today_dt()
        y_eq = self.store.equity_on(source, (today - timedelta(days=1)).isoformat())
        w_eq = self.store.equity_on(source, (today - timedelta(days=7)).isoformat())
        # R4: deposits/withdrawals are not P&L. Comparing equity NET of flows
        # since each baseline means a deposit cannot mask a real loss and a
        # withdrawal cannot trip a false one. (Drawdown peak stays gross —
        # normalizing multi-month peaks needs the full flow ledger; noted.)
        eq_d = eq - self.store.external_flows_since(
            (today - timedelta(days=1)).isoformat())
        eq_w = eq - self.store.external_flows_since(
            (today - timedelta(days=7)).isoformat())
        # high-water mark since the last drawdown-baseline reset (D17)
        reset_d = self.store.kv_get("dd_peak_reset_d", "") or ""
        peak = max(self.store.peak_equity(source, since_d=reset_d), eq)

        if y_eq and eq_d < y_eq * (1 - self.r.get("max_daily_loss", 0.02)):
            self.trip("daily_loss", f"flow-adjusted equity {eq_d:.2f} < "
                                    f"{1-self.r['max_daily_loss']:.0%} "
                                    f"of yesterday {y_eq:.2f}", auto_clear_days=1)
        if w_eq and eq_w < w_eq * (1 - self.r.get("max_weekly_loss", 0.05)):
            self.trip("weekly_loss", f"flow-adjusted equity {eq_w:.2f} breaches "
                                     f"weekly loss vs {w_eq:.2f}",
                      auto_clear_days=7)
        # R4: repeated fractional-fill slippage breaches (recorded by the
        # broker adapter) halt new entries for the day.
        slip = int(self.store.kv_get(f"slippage_breaches:{self.today}", 0) or 0)
        if slip >= self.r.get("max_slippage_breaches_per_day", 3):
            self.trip("slippage", f"{slip} fractional fills breached slippage "
                                  "tolerance today", auto_clear_days=1)
        if peak > 0 and eq < peak * (1 - self.r.get("kill_switch_drawdown", 0.15)):
            # cooldown then auto-resume (D15): a permanent halt turned the 2022
            # bear into 3 years of dead cash in backtest v1. null = manual-only.
            cooldown = self.r.get("drawdown_cooldown_days", 10)
            self.trip("drawdown", f"equity {eq:.2f} is "
                                  f"{(1 - eq/peak):.1%} below peak {peak:.2f}",
                      auto_clear_days=cooldown)
        # broker-level bounces only; governor vetoes carry status 'vetoed'.
        # Counted since the last reset (D39) and auto-clears after a short
        # cooldown — a broker-side storm shouldn't need a human forever.
        reset_ts = self.store.kv_get("rejected_orders_reset_ts", "") or ""
        rejected_today = len([o for o in self.store.orders_today(day=self.today, mode=self.mode)
                              if o["status"] == "rejected" and o["created_at"] > reset_ts])
        if rejected_today > self.r.get("max_rejected_orders_per_day", 5):
            self.trip("rejected_orders",
                      f"{rejected_today} broker-rejected orders today — check the "
                      f"broker_review audit rows for the shared cause",
                      auto_clear_minutes=self.r.get(
                          "rejected_order_cooldown_minutes", 30))
        return self.active_switches()

    # ---------------- cycle budget ----------------
    def cycle_budget(self, account: AccountState, deployment_multiplier: float = 1.0) -> float:
        """min(equity × pct, abs cap) × regime deployment multiplier."""
        pct = self.r.get("time_step_budget_pct", 0.10)
        cap = self.r.get("time_step_budget_abs_cap", 5000)
        return min(account.equity * pct, cap) * max(0.0, deployment_multiplier)

    # ---------------- options availability (D5) ----------------
    def options_unlocked(self, account: AccountState) -> bool:
        setting = self.r.get("options_enabled", "auto")
        if setting is True:
            return True
        if setting is False:
            return False
        per_trade_cap = account.equity * self.r.get("max_single_option_premium_risk", 0.015)
        return per_trade_cap >= self.r.get("min_viable_option_premium", 75)

    def validate_option(self, details: dict) -> list[str]:
        """Bounded-risk option checks per AGENTS.md §22. `details` must carry
        dte, delta, spread_pct, open_interest, premium. Missing key = flag —
        unknown risk is treated as risk."""
        flags = []
        checks = [
            ("dte", lambda v: self.r.get("option_min_dte", 21) <= v <= self.r.get("option_max_dte", 120), "dte_out_of_range"),
            ("delta", lambda v: self.r.get("option_min_delta", 0.25) <= abs(v) <= self.r.get("option_max_delta", 0.70), "delta_out_of_range"),
            ("spread_pct", lambda v: v <= self.r.get("option_max_spread_pct", 0.15), "spread_too_wide"),
            ("open_interest", lambda v: v >= self.r.get("option_min_open_interest", 100), "open_interest_too_low"),
        ]
        for key, ok, flag in checks:
            v = details.get(key)
            if v is None:
                flags.append(f"missing_{key}")
            elif not ok(v):
                flags.append(flag)
        return flags

    # ---------------- per-order review ----------------
    def review(self, intent: OrderIntent, candidate: TradeCandidate,
               account: AccountState, cycle: CycleState,
               data_age_days: int | None,
               skip_duplicate: bool = False) -> RiskDecision:
        rj = lambda *reasons: RiskDecision("REJECTED", list(reasons))  # noqa: E731

        # --- hard blocks first ---
        if intent.side == "buy":
            active = self.active_switches()
            if active:
                return rj(f"kill switches active: {sorted(active)}")
            # Scoped risk exceptions carry a max_equity: above it the tiny-
            # account rationale is gone and every widened limit is void — new
            # buys stop until the operator re-approves config at standard
            # limits. Exits are never blocked by this.
            cap = getattr(self.cfg, "risk_exception_equity_cap", lambda: None)()
            if cap is not None and account.equity > cap:
                return rj(f"risk exception void: equity {account.equity:.2f} "
                          f"exceeds max_equity {cap:.2f} — re-approve config "
                          f"at standard limits")
        if data_age_days is None or data_age_days > self.r.get("stale_data_max_age_days", 4):
            return rj(f"stale data: age={data_age_days} days")
        if not skip_duplicate and self.store.recent_order_exists(
                intent.symbol, intent.side,
                self.r.get("duplicate_order_cooldown_min", 60),
                now_iso=self.now_iso, mode=self.mode):
            return rj("duplicate: same symbol+side within cooldown")
        if intent.asset_type == "option":
            if not self.options_unlocked(account):
                return rj("options locked at current account scale")
            flags = self.validate_option(candidate.option_details or {})
            if flags:
                return rj(*[f"option:{f}" for f in flags])
            # R4: the AGGREGATE premium-at-risk cap finally binds — the sum of
            # every open option premium plus this one stays inside the budget.
            total_premium = sum(p.cost_basis for p in account.positions
                                if p.asset_type == "option") + intent.notional
            opt_cap = account.equity * self.r.get(
                "max_total_options_premium_risk", 0.06)
            if total_premium > opt_cap:
                return rj(f"options aggregate premium cap: ${total_premium:.2f} "
                          f"> ${opt_cap:.2f}")

        # sells (exits) skip entry-side caps — reducing risk is always allowed
        if intent.side == "sell":
            return RiskDecision("APPROVED", ["exit order"], approved_notional=intent.notional)

        # --- worst-case loss of this order ---
        worst_case = intent.notional   # equity notional or option premium (qty*px*100 handled upstream)

        # --- entry caps, with size-reduction where sensible ---
        reasons, notional = [], intent.notional

        # per-DAY cap, not per-cycle (live mode runs 3 scan cycles a day).
        # This cycle's earlier fills are already recorded in orders, so the
        # DB count alone is the complete number — don't add cycle.new_positions.
        buys_today = len([o for o in self.store.orders_today("buy", day=self.today, mode=self.mode)
                          if o["status"] in ("filled", "placed", "reviewed")])
        if buys_today + 1 > self.r.get("max_daily_new_positions", 3):
            return rj(f"max_daily_new_positions reached ({buys_today} placed today)")
        if len([p for p in account.positions if p.qty > 0]) >= self.r.get("max_open_positions", 12):
            return rj("max_open_positions reached")

        # time-step budget (primary): shrink to fit remaining budget
        if worst_case > cycle.budget_left:
            if cycle.budget_left < MIN_ORDER_NOTIONAL:
                return rj(f"time-step budget exhausted (left ${cycle.budget_left:.2f})")
            notional = cycle.budget_left
            reasons.append(f"reduced to fit time-step budget (${cycle.budget_left:.2f} left)")

        # single-position cap at MARKED value + open-order reservation (R4)
        pos_cap = account.equity * self.r.get("max_single_equity_position", 0.08)
        held = sum(self._marked_value(p) for p in account.positions
                   if p.symbol == intent.symbol)
        held += self._pending_buy_notional(intent.symbol)
        if held + notional > pos_cap:
            room = pos_cap - held
            if room < MIN_ORDER_NOTIONAL:
                return rj(f"single-position cap: already ${held:.2f} of ${pos_cap:.2f}")
            notional = min(notional, room)
            reasons.append(f"reduced to single-position cap room ${room:.2f}")

        # sector concentration cap (R4): finally enforced. Unknown sector =
        # exempt but audited once per symbol — visible gap until R5 classifies.
        sector = self._sector(intent.symbol)
        if sector:
            sector_cap = account.equity * self.r.get("max_sector_exposure", 0.25)
            sector_held = sum(self._marked_value(p) for p in account.positions
                              if p.qty > 0 and self._sector(p.symbol) == sector)
            if sector_held + notional > sector_cap:
                room = sector_cap - sector_held
                if room < MIN_ORDER_NOTIONAL:
                    return rj(f"sector cap ({sector}): ${sector_held:.2f} of "
                              f"${sector_cap:.2f} held")
                notional = min(notional, room)
                reasons.append(f"reduced to sector cap room ${room:.2f} ({sector})")
        elif not self.store.kv_get(f"sector_unknown:{intent.symbol}"):
            self.store.kv_set(f"sector_unknown:{intent.symbol}", self.today)
            self.store.audit("sector_unknown", {
                "symbol": intent.symbol,
                "note": "exempt from max_sector_exposure until classified (R5)"})

        # total deployment / cash reserve. Two knobs express the same limit
        # from opposite ends (deploy ≤ X  vs  keep ≥ Y cash); honor whichever
        # is stricter so neither is a silent no-op. MARKED + reserved (R4).
        deployed = sum(self._marked_value(p) for p in account.positions)
        deployed += self._pending_buy_notional()
        deploy_frac = min(self.r.get("max_account_deployment", 0.70),
                          1.0 - self.r.get("min_cash_reserve", 0.0))
        max_deploy = account.equity * deploy_frac
        if deployed + notional > max_deploy:
            room = max_deploy - deployed
            if room < MIN_ORDER_NOTIONAL:
                return rj(f"deployment cap: ${deployed:.2f} of ${max_deploy:.2f} deployed")
            notional = min(notional, room)
            reasons.append(f"reduced to deployment cap room ${room:.2f}")
        # Never treat margin buying power as cash. Conversely, open broker
        # orders can reserve buying power before cash/positions update, so the
        # lower of the two is the only honest amount available for a new buy.
        spendable = min(max(0.0, account.cash), max(0.0, account.buying_power))
        if notional > spendable:
            if spendable < MIN_ORDER_NOTIONAL:
                return rj(f"insufficient spendable cash (${spendable:.2f})")
            notional = spendable
            reasons.append(f"reduced to spendable cash (${spendable:.2f})")

        # --- approval policy (D4) ---
        # Threshold compares the REQUESTED size, pre-reduction: portfolio layer
        # normally proposes cap-respecting sizes, so something asking for a
        # big-bet notional is anomalous and deserves human eyes even though the
        # caps above already shrank what would actually be placed.
        mode = self.r.get("approval_mode", "threshold")
        threshold = account.equity * self.r.get("approval_notional_threshold_pct", 0.05)
        if mode == "all" or (mode == "threshold" and intent.notional > threshold):
            return RiskDecision("REQUIRES_HUMAN_APPROVAL",
                                reasons + [f"notional ${notional:.2f} needs approval "
                                           f"(mode={mode}, threshold ${threshold:.2f})"],
                                approved_notional=notional)

        verdict = "APPROVED_WITH_SIZE_REDUCTION" if notional < intent.notional else "APPROVED"
        return RiskDecision(verdict, reasons or ["all checks passed"], approved_notional=notional)

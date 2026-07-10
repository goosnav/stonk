"""Risk governor: the deterministic final gate. Nothing trades without passing
review(); AI has no path around it (dev/ARCHITECTURE.md invariant #2).

Two layers:
1. Kill switches — evaluated once per cycle (check_kill_switches). Tripped
   switches block ALL new entries; exits remain allowed. daily_loss/weekly_loss
   auto-clear when the calendar rolls; drawdown and operational switches
   (rejected-order storm) need a manual reset (CLI/GUI) after a human looks.
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
        # auto-clear switches whose window rolled over
        cleared = {k: v for k, v in switches.items()
                   if not (v.get("auto_clear") and v.get("clear_on", "") <= today)}
        if cleared != switches:
            if "drawdown" in switches and "drawdown" not in cleared:
                self._reset_drawdown_baseline("auto_clear")
            self.store.kv_set(KILL_KEY, cleared)
        return cleared

    def _reset_drawdown_baseline(self, via: str) -> None:
        """D17: when a drawdown trip clears, the high-water mark restarts at
        the clear date. Without this, an all-cash account sitting 15% below an
        old peak re-trips forever (backtest v2 finding: 3 years frozen)."""
        self.store.kv_set("dd_peak_reset_d", self.today)
        self.store.audit("drawdown_baseline_reset", {"date": self.today, "via": via})

    def trip(self, name: str, reason: str, auto_clear_days: int | None = None) -> None:
        switches = self.store.kv_get(KILL_KEY, {}) or {}
        entry = {"reason": reason, "tripped_at": self.now_iso}
        if auto_clear_days is not None:
            entry["auto_clear"] = True
            entry["clear_on"] = (self._today_dt() + timedelta(days=auto_clear_days)).isoformat()
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

    def check_kill_switches(self, account: AccountState, source: str) -> dict:
        """Run at cycle start. Evaluates loss/drawdown limits against the equity
        curve and trips switches. Returns the active set."""
        eq = account.equity
        today = self._today_dt()
        y_eq = self.store.equity_on(source, (today - timedelta(days=1)).isoformat())
        w_eq = self.store.equity_on(source, (today - timedelta(days=7)).isoformat())
        # high-water mark since the last drawdown-baseline reset (D17)
        reset_d = self.store.kv_get("dd_peak_reset_d", "") or ""
        peak = max(self.store.peak_equity(source, since_d=reset_d), eq)

        if y_eq and eq < y_eq * (1 - self.r.get("max_daily_loss", 0.02)):
            self.trip("daily_loss", f"equity {eq:.2f} < {1-self.r['max_daily_loss']:.0%} "
                                    f"of yesterday {y_eq:.2f}", auto_clear_days=1)
        if w_eq and eq < w_eq * (1 - self.r.get("max_weekly_loss", 0.05)):
            self.trip("weekly_loss", f"equity {eq:.2f} breaches weekly loss vs {w_eq:.2f}",
                      auto_clear_days=7)
        if peak > 0 and eq < peak * (1 - self.r.get("kill_switch_drawdown", 0.15)):
            # cooldown then auto-resume (D15): a permanent halt turned the 2022
            # bear into 3 years of dead cash in backtest v1. null = manual-only.
            cooldown = self.r.get("drawdown_cooldown_days", 10)
            self.trip("drawdown", f"equity {eq:.2f} is "
                                  f"{(1 - eq/peak):.1%} below peak {peak:.2f}",
                      auto_clear_days=cooldown)
        # broker-level bounces only; governor vetoes carry status 'vetoed'.
        # Counted since the last manual reset (D39) and auto-clears next day —
        # a broker-side storm shouldn't need a human forever.
        reset_ts = self.store.kv_get("rejected_orders_reset_ts", "") or ""
        rejected_today = len([o for o in self.store.orders_today(day=self.today, mode=self.mode)
                              if o["status"] == "rejected" and o["created_at"] > reset_ts])
        if rejected_today > self.r.get("max_rejected_orders_per_day", 5):
            self.trip("rejected_orders",
                      f"{rejected_today} broker-rejected orders today — check the "
                      f"broker_review audit rows for the shared cause",
                      auto_clear_days=1)
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

        # single-position cap (cost_basis handles the option ×100 multiplier)
        pos_cap = account.equity * self.r.get("max_single_equity_position", 0.08)
        held = sum(p.cost_basis for p in account.positions if p.symbol == intent.symbol)
        if held + notional > pos_cap:
            room = pos_cap - held
            if room < MIN_ORDER_NOTIONAL:
                return rj(f"single-position cap: already ${held:.2f} of ${pos_cap:.2f}")
            notional = min(notional, room)
            reasons.append(f"reduced to single-position cap room ${room:.2f}")

        # total deployment / cash reserve. Two knobs express the same limit
        # from opposite ends (deploy ≤ X  vs  keep ≥ Y cash); honor whichever
        # is stricter so neither is a silent no-op.
        deployed = sum(p.cost_basis for p in account.positions)
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

"""Adjustment-seam detection and repair.

The bug this guards: `data.refresh()` takes an incremental window when bars are
recent, and `upsert_bars` overwrites per (symbol, d) with no re-adjustment. Both
providers serve RETROACTIVELY split-adjusted series, so after a split the old
rows keep the stale basis while the trailing window arrives on the new one. The
live store carried 1,834 impossible single-session moves, 31% of the training
panel — a +416,566% "return" sitting in the forward targets.
"""
from __future__ import annotations

import numpy as np
import pytest

from conftest import synth_bars
from specforge import bars_audit


def _seamed(n_days=200, factor=50.0, at=120):
    """Bars where everything before `at` sits on a pre-reverse-split basis."""
    rows = synth_bars(n_days=n_days, wiggle=0.004)
    for row in rows[:at]:
        for key in ("open", "high", "low", "close"):
            row[key] /= factor
    return rows


def test_simple_rational_accepts_split_factors_and_rejects_real_moves():
    for ratio in (2.0, 0.5, 10.0, 0.1, 7.0, 3.0, 1.5, 0.05):
        assert bars_audit._simple_rational(ratio), ratio
    for ratio in (1.63, 2.37, 0.71, 1.44, 8.31):
        assert not bars_audit._simple_rational(ratio), ratio


def test_black_monday_is_not_mistaken_for_a_corporate_action(cfg, store):
    """Regression: the first detector classified 1987-10-19 MSFT as a split.

    A -30% crash gives ratio ~0.70, which "any fraction with terms under 20"
    accepts as 7/10, and the drop persisted — so both original tests passed and
    a real market event would have been rewritten out of the record.
    """
    assert not bars_audit._simple_rational(0.6988)
    rows = synth_bars(n_days=200, wiggle=0.004)
    for row in rows[120:]:                    # permanent -30% level shift
        for key in ("open", "high", "low", "close"):
            row[key] *= 0.6988
    store.upsert_bars("CRASH", rows, "test")
    found = bars_audit.detect(store, ["CRASH"])
    assert not [f for f in found if f["kind"] == "seam"]
    assert [f for f in found if f["kind"] == "suspicious"]


def test_detector_finds_an_impossible_jump(cfg, store):
    rows = _seamed(factor=50.0, at=120)
    store.upsert_bars("SEAM", rows, "test")
    found = bars_audit.detect(store, ["SEAM"])
    seams = [f for f in found if f["kind"] == "seam"]
    assert len(seams) == 1
    seam = seams[0]
    assert seam["symbol"] == "SEAM"
    assert seam["d"] == rows[120]["d"]
    assert seam["ratio"] == pytest.approx(50.0, rel=.05)


def test_detector_does_not_flag_a_genuine_spike_that_reverts(cfg, store):
    """A real +60% print must never be 'repaired' out of the record."""
    rows = synth_bars(n_days=200, wiggle=0.004)
    base = rows[120]["close"]
    for key in ("open", "high", "low", "close"):
        rows[120][key] *= 1.6                     # one-session pop, then back
    store.upsert_bars("SPIKE", rows, "test")
    found = bars_audit.detect(store, ["SPIKE"])
    assert not [f for f in found if f["kind"] == "seam"]
    # It is still surfaced for a human, just never auto-repaired.
    assert [f for f in found if f["kind"] == "suspicious"]
    assert rows[121]["close"] < base * 1.2        # sanity: the move did revert


def test_detector_ignores_ordinary_volatility(cfg, store):
    store.upsert_bars("CALM", synth_bars(n_days=200, wiggle=0.01), "test")
    assert bars_audit.detect(store, ["CALM"]) == []


def test_repair_never_deletes_before_a_successful_fetch(cfg, store):
    """A provider returning nothing must leave the stored history untouched."""
    rows = _seamed()
    store.upsert_bars("SEAM", rows, "test")
    before = store.db.execute(
        "SELECT COUNT(*) n FROM bars WHERE symbol='SEAM'").fetchone()["n"]
    result = bars_audit.repair_symbol(store, "SEAM", fetcher=lambda sym: [])
    assert result["status"] == "fetch_failed"
    assert store.db.execute(
        "SELECT COUNT(*) n FROM bars WHERE symbol='SEAM'").fetchone()["n"] == before


def test_repair_refuses_a_fetch_that_truncates_history(cfg, store):
    """Replacing 200 sessions with 20 is data loss, not a repair."""
    rows = _seamed()
    store.upsert_bars("SEAM", rows, "test")
    clean = synth_bars(n_days=20, wiggle=0.004)
    result = bars_audit.repair_symbol(store, "SEAM", fetcher=lambda sym: clean)
    assert result["status"] == "fetch_too_short"
    assert store.db.execute(
        "SELECT COUNT(*) n FROM bars WHERE symbol='SEAM'").fetchone()["n"] == len(rows)


def test_repair_replaces_the_series_and_is_idempotent(cfg, store):
    store.upsert_bars("SEAM", _seamed(), "test")
    assert [f for f in bars_audit.detect(store, ["SEAM"]) if f["kind"] == "seam"]
    clean = synth_bars(n_days=200, wiggle=0.004)
    first = bars_audit.repair_symbol(store, "SEAM", fetcher=lambda sym: clean)
    assert first["status"] == "repaired"
    assert not bars_audit.detect(store, ["SEAM"])          # seam is gone
    snapshot = store.db.execute(
        "SELECT d, close FROM bars WHERE symbol='SEAM' ORDER BY d").fetchall()
    second = bars_audit.repair_symbol(store, "SEAM", fetcher=lambda sym: clean)
    again = store.db.execute(
        "SELECT d, close FROM bars WHERE symbol='SEAM' ORDER BY d").fetchall()
    assert second["status"] == "repaired"
    assert [tuple(r) for r in snapshot] == [tuple(r) for r in again]


def test_repair_leaves_no_row_on_the_stale_basis(cfg, store):
    """Rows the new fetch does not cover must not survive on the old basis."""
    old = _seamed(n_days=200)
    store.upsert_bars("SEAM", old, "test")
    clean = synth_bars(n_days=200, wiggle=0.004)[:150]     # shorter tail
    bars_audit.repair_symbol(store, "SEAM", fetcher=lambda sym: clean,
                             min_coverage=0.5)
    stored = {r["d"] for r in store.db.execute(
        "SELECT d FROM bars WHERE symbol='SEAM'")}
    assert stored == {r["d"] for r in clean}


def test_audit_reports_without_repairing_by_default(cfg, store):
    store.upsert_bars("SEAM", _seamed(), "test")
    report = bars_audit.audit(store, symbols=["SEAM"])
    assert report["seams"] == 1 and report["symbols_affected"] == ["SEAM"]
    assert report["repaired"] == []                # report-only unless asked
    assert store.db.execute(
        "SELECT COUNT(*) n FROM bars WHERE symbol='SEAM'").fetchone()["n"] == 200


def test_extreme_ratios_are_seams_even_without_a_simple_split_factor(cfg, store):
    """Repeated reverse splits compound into ratios no small fraction describes.

    The live store's worst case was 0.34 -> 1433 (roughly 1-for-4200, several
    splits stacked). The simple-rational test rejects it, but no equity moves
    20x in one session by trading, so the data is wrong either way.
    """
    rows = _seamed(factor=4166.667, at=120)
    store.upsert_bars("STACKED", rows, "test")
    seams = [f for f in bars_audit.detect(store, ["STACKED"]) if f["kind"] == "seam"]
    assert len(seams) == 1
    assert seams[0]["simple_rational"] is False      # not a clean split factor...
    assert seams[0]["extreme"] is True               # ...but impossible regardless


def test_a_plain_ten_to_one_split_seam_is_reported_not_auto_repaired(cfg, store):
    """10x is below the impossible threshold, so it needs a human.

    A 1-for-10 reverse split and a genuine 10-bagger session both produce this
    ratio, and nothing offline separates them. Reported, never rewritten.
    """
    store.upsert_bars("TENX", _seamed(factor=10.0, at=120), "test")
    found = bars_audit.detect(store, ["TENX"])
    assert not [f for f in found if f["kind"] == "seam"]
    assert [f for f in found if f["kind"] == "suspicious"]


def test_a_large_but_believable_move_still_needs_both_ordinary_tests(cfg, store):
    """Below the extreme threshold, one test passing is not enough."""
    rows = synth_bars(n_days=200, wiggle=0.004)
    for key in ("open", "high", "low", "close"):     # 2x spike that reverts:
        rows[120][key] *= 2.0                        # rational, but not persistent
    store.upsert_bars("POP", rows, "test")
    found = bars_audit.detect(store, ["POP"])
    assert [f for f in found if f["kind"] == "suspicious"]
    assert not [f for f in found if f["kind"] == "seam"]


# ── prevention: an incremental write must never join two adjustment bases ────

def test_basis_change_detected_when_provider_readjusts(cfg, store):
    from specforge import data
    rows = synth_bars(n_days=100, wiggle=0.004)
    store.upsert_bars("SPLIT", rows, "test")
    # Same sessions, provider now serving a 1:10 reverse-split-adjusted series.
    readjusted = [{**r, **{k: r[k] * 10 for k in ("open", "high", "low", "close")}}
                  for r in rows[-40:]]
    assert data._basis_changed(store, "SPLIT", readjusted) is True
    # Unchanged basis must NOT trigger a full refetch — that would refetch daily.
    assert data._basis_changed(store, "SPLIT", rows[-40:]) is False


def test_basis_check_tolerates_rounding_and_one_corrected_print(cfg, store):
    from specforge import data
    rows = synth_bars(n_days=100, wiggle=0.004)
    store.upsert_bars("NOISE", rows, "test")
    jittered = [{**r, "close": r["close"] * 1.001} for r in rows[-40:]]
    assert data._basis_changed(store, "NOISE", jittered) is False
    one_fixed = [dict(r) for r in rows[-40:]]
    one_fixed[5]["close"] *= 3.0                    # a single corrected bad print
    assert data._basis_changed(store, "NOISE", one_fixed) is False


def test_basis_check_fails_safe_without_enough_overlap(cfg, store):
    from specforge import data
    rows = synth_bars(n_days=100, wiggle=0.004)
    store.upsert_bars("THIN", rows[:50], "test")
    future = synth_bars(n_days=100, wiggle=0.004)[60:]     # no shared sessions
    assert data._basis_changed(store, "THIN", future) is False
    assert data._basis_changed(store, "UNKNOWN", rows) is False


# ── backfill ordering: importance, not alphabet ─────────────────────────────

def test_backfill_prioritizes_importance_over_alphabet(cfg, store):
    """Regression: 530 of 578 symbols that acquired bars began with 'A'."""
    from specforge import research

    def instrument(sym):
        store.db.execute(
            "INSERT INTO instruments(symbol,name,exchange,security_type,is_etf,"
            "is_adr,active,first_seen,last_seen,source,cik,raw_hash) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (sym, sym, "NASDAQ", "common", 0, 0, 1, "2020-01-01",
             "2026-01-01", "test", None, "h"))

    with store.db:
        for sym in ("AAAA", "AAAB", "AAAC", "ZBIG", "ZMID", "MANDATE"):
            instrument(sym)
        # ZBIG/ZMID are the liquid names; alphabetically they are last.
        for rank, sym in enumerate(("ZBIG", "ZMID"), 1):
            store.db.execute("INSERT INTO universe_membership VALUES(?,?,?,?,?,?)",
                             ("2026-07-01", sym, "research", rank, "liquidity", "{}"))
    cfg.data.setdefault("universe", {})["symbols"] = ["MANDATE"]

    order = research._missing_history(store, 10, cfg)
    assert order[0] == "MANDATE"              # configured universe is never starved
    assert order[1:3] == ["ZBIG", "ZMID"]     # then research rank (dollar volume)
    assert set(order[3:]) == {"AAAA", "AAAB", "AAAC"}   # tail still drains


def test_backfill_without_cfg_still_orders_by_membership_rank(cfg, store):
    from specforge import research
    with store.db:
        for sym in ("AAAA", "ZBIG"):
            store.db.execute(
                "INSERT INTO instruments(symbol,name,exchange,security_type,is_etf,"
                "is_adr,active,first_seen,last_seen,source,cik,raw_hash) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (sym, sym, "NASDAQ", "common", 0, 0, 1, "2020-01-01",
                 "2026-01-01", "test", None, "h"))
        store.db.execute("INSERT INTO universe_membership VALUES(?,?,?,?,?,?)",
                         ("2026-07-01", "ZBIG", "research", 1, "liquidity", "{}"))
    assert research._missing_history(store, 10)[0] == "ZBIG"


# ── SEC fact ingestion: one candidate pass, real contact, rate limited ───────

def test_bulk_ingest_refuses_the_placeholder_user_agent(cfg, store):
    """Thousands of requests under a fake contact is how an IP gets blocked."""
    from specforge import universe
    assert "local-user" in universe.sec_user_agent(None)
    result = universe.ingest_filing_facts_batch(store, 10, cfg=cfg,
                                                require_contact=True)
    assert result["status"] == "refused" and "sec_user_agent" in result["reason"]
    cfg.data.setdefault("research", {})["sec_user_agent"] = "Stonk me@example.com"
    assert universe.sec_user_agent(cfg) == "Stonk me@example.com"


def test_batch_walks_one_candidate_pass_not_one_per_issuer(cfg, store, monkeypatch):
    """The marker scan used to restart from index 0 for every single issuer.

    A 1,500-issuer batch cost ~1.1M kv reads for 1,500 fetches; the candidate
    query and the due-check are now hoisted out of the per-issuer function.
    """
    from specforge import universe
    with store.db:
        for i in range(30):
            store.db.execute(
                "INSERT INTO instruments(symbol,name,exchange,security_type,is_etf,"
                "is_adr,active,first_seen,last_seen,source,cik,raw_hash) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (f"S{i:03d}", "n", "NASDAQ", "common", 0, 0, 1, "2020-01-01",
                 "2026-01-01", "test", str(1000 + i), "h"))
    queries = {"candidates": 0}
    real = universe._fact_candidates
    monkeypatch.setattr(universe, "_fact_candidates",
                        lambda s: (queries.__setitem__("candidates",
                                                       queries["candidates"] + 1)
                                   or real(s)))
    monkeypatch.setattr(universe.time, "sleep", lambda *_: None)

    class Client:
        def get(self, url, **kw):
            raise RuntimeError("no network in tests")
    result = universe.ingest_filing_facts_batch(store, 10, cfg=cfg, client=Client())
    assert result["attempted"] == 10
    assert queries["candidates"] == 1          # ONE pass, not one per issuer


def test_batch_rate_limits_between_issuers(cfg, store, monkeypatch):
    from specforge import universe
    with store.db:
        for i in range(5):
            store.db.execute(
                "INSERT INTO instruments(symbol,name,exchange,security_type,is_etf,"
                "is_adr,active,first_seen,last_seen,source,cik,raw_hash) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (f"R{i}", "n", "NASDAQ", "common", 0, 0, 1, "2020-01-01",
                 "2026-01-01", "test", str(2000 + i), "h"))
    slept = []
    monkeypatch.setattr(universe.time, "sleep", lambda s: slept.append(s))

    class Client:
        def get(self, url, **kw):
            raise RuntimeError("offline")
    universe.ingest_filing_facts_batch(store, 5, cfg=cfg, client=Client())
    assert len(slept) == 5
    assert all(s >= 1.0 / universe.SEC_MAX_REQUESTS_PER_SECOND for s in slept)


# ── quarantine: a repair that cannot repair means the provider is broken ─────

def test_symbols_still_seamed_after_refetch_are_quarantined(cfg, store):
    """All 101 live symbols came back seamed after a clean refetch.

    That is not a broken repair — it is a broken provider. Free sources handle
    ultra-thin names with repeated reverse splits badly, and excluding them is
    more honest than pretending a refetch fixed anything.
    """
    store.upsert_bars("BROKEN", _seamed(factor=50.0, at=120), "test")
    still_broken = _seamed(factor=50.0, at=120)      # provider serves it again
    report = bars_audit.audit(store, symbols=["BROKEN"], repair=True,
                              fetcher=lambda sym: still_broken)
    assert report["still_seamed"] == ["BROKEN"]
    assert "BROKEN" in report["quarantined"]
    assert bars_audit.quarantined(store) == {"BROKEN"}
    assert store.db.execute("SELECT 1 FROM audit WHERE "
                            "event_type='bars_quarantined'").fetchone()


def test_quarantine_is_durable_and_additive(cfg, store):
    bars_audit.quarantine(store, ["AAA"])
    bars_audit.quarantine(store, ["BBB"])
    assert bars_audit.quarantined(store) == {"AAA", "BBB"}
    bars_audit.quarantine(store, ["AAA"])            # idempotent
    assert bars_audit.quarantined(store) == {"AAA", "BBB"}


def test_a_successful_repair_does_not_quarantine(cfg, store):
    store.upsert_bars("FIXABLE", _seamed(), "test")
    clean = synth_bars(n_days=200, wiggle=0.004)
    report = bars_audit.audit(store, symbols=["FIXABLE"], repair=True,
                              fetcher=lambda sym: clean)
    assert report["still_seamed"] == []
    assert bars_audit.quarantined(store) == set()


def test_build_dataset_excludes_quarantined_symbols(cfg, store):
    from specforge import neural
    for sym in ("AAA", "BBB", "CCC", "SPY"):
        store.upsert_bars(sym, synth_bars(n_days=700, daily_drift=.001), "test")
    store.upsert_bars("^VIX", [{**r, "open": 15, "high": 16, "low": 14, "close": 15}
                               for r in synth_bars(n_days=700)], "test")
    bars_audit.quarantine(store, ["BBB"])
    cfg.data["neural"]["input_sessions"] = 40
    ds = neural.build_dataset(cfg, store, symbols=["AAA", "BBB", "CCC"])
    assert "error" not in ds
    assert "BBB" not in set(ds["owners"])
    assert ds["pit"]["quarantined_symbols"] == 1


def test_a_real_crash_at_a_split_like_ratio_is_never_auto_repaired(cfg, store):
    """Regression: AMD's 2002 crash (-32%) and MCD were quarantined as splits.

    0.676 sits within tolerance of 2/3, and the drop persisted, so
    "declared split factor + persistent level shift" flagged it. Ninety-three
    symbols were excluded on what were mostly real drawdowns. Only an
    impossible ratio is a seam now.
    """
    rows = synth_bars(n_days=300, wiggle=0.004)
    for row in rows[150:]:                     # permanent -32%, like a bear bottom
        for key in ("open", "high", "low", "close"):
            row[key] *= 0.676
    store.upsert_bars("AMDLIKE", rows, "test")
    found = bars_audit.detect(store, ["AMDLIKE"])
    assert not [f for f in found if f["kind"] == "seam"]
    assert [f for f in found if f["kind"] == "suspicious"]


def test_unreliable_flags_by_density_not_by_any_single_jump(cfg, store):
    """A volatile stock prints a few huge sessions; broken data prints dozens."""
    import numpy as np
    volatile = synth_bars(n_days=2600, wiggle=0.004)
    for i in (300, 900, 1800):                 # three real shocks over ~10 years
        for key in ("open", "high", "low", "close"):
            volatile[i][key] *= 1.8
    store.upsert_bars("VOLATILE", volatile, "test")

    junk = synth_bars(n_days=2600, wiggle=0.004)
    for i in range(100, 2500, 40):             # a jump every 40 sessions
        for key in ("open", "high", "low", "close"):
            junk[i][key] *= 2.2
    store.upsert_bars("JUNK", junk, "test")

    flagged = bars_audit.unreliable(store, ["VOLATILE", "JUNK"])
    assert "JUNK" in flagged and "VOLATILE" not in flagged
    assert flagged["JUNK"]["per_year"] > bars_audit.LARGE_MOVES_PER_YEAR


def test_unranked_tail_is_unbiased_not_alphabetical(cfg, store):
    """Liquidity rank comes FROM stored bars, so unfetched symbols can never
    be ranked — the backfill cannot use the ordering it is meant to produce.

    Alphabetical was the old tiebreak and it produced a systematically biased
    store: 530 of 578 symbols with bars began with "A". A stable hash is still
    arbitrary, but the sample it builds is representative.
    """
    from specforge import research
    import string
    with store.db:
        for letter in string.ascii_uppercase:
            for i in range(4):
                store.db.execute(
                    "INSERT INTO instruments(symbol,name,exchange,security_type,"
                    "is_etf,is_adr,active,first_seen,last_seen,source,cik,raw_hash) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                    (f"{letter}{i:03d}", "n", "NASDAQ", "common", 0, 0, 1,
                     "2020-01-01", "2026-01-01", "test", None, "h"))
    first_50 = research._missing_history(store, 50)
    letters = {s[0] for s in first_50}
    # An alphabetical walk would return only A and B here.
    assert len(letters) > 12, sorted(letters)
    # Stable across calls, so progress is monotonic rather than reshuffled.
    assert research._missing_history(store, 50) == first_50


def test_sec_user_agent_prefers_env_over_tracked_config(cfg, monkeypatch):
    """A contact address is personal data and configs/ is tracked and pushed."""
    from specforge import universe
    cfg.data.setdefault("research", {})["sec_user_agent"] = "from-config"
    monkeypatch.delenv(universe.SEC_USER_AGENT_ENV, raising=False)
    assert universe.sec_user_agent(cfg) == "from-config"
    monkeypatch.setenv(universe.SEC_USER_AGENT_ENV, "from-env me@example.com")
    assert universe.sec_user_agent(cfg) == "from-env me@example.com"

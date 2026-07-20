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


def _seamed(n_days=200, factor=10.0, at=120):
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


def test_detector_finds_a_reverse_split_seam(cfg, store):
    rows = _seamed(factor=10.0, at=120)
    store.upsert_bars("SEAM", rows, "test")
    found = bars_audit.detect(store, ["SEAM"])
    seams = [f for f in found if f["kind"] == "seam"]
    assert len(seams) == 1
    seam = seams[0]
    assert seam["symbol"] == "SEAM"
    assert seam["d"] == rows[120]["d"]
    assert seam["ratio"] == pytest.approx(10.0, rel=.05)


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

"""Static contracts for the dependency-free operator dashboard.

These checks intentionally avoid a browser or live service.  Browser smoke tests
exercise behavior; this file prevents security/accessibility regressions in the
single-file dashboard before a server is ever started.
"""

from pathlib import Path


DASHBOARD = (Path(__file__).parents[1] / "static" / "dashboard.html").read_text()


def test_tabs_have_keyboard_and_aria_contract() -> None:
    assert 'role="tablist"' in DASHBOARD
    assert DASHBOARD.count('role="tab"') == 8
    assert DASHBOARD.count('role="tabpanel"') == 8
    assert "ArrowRight" in DASHBOARD
    assert "ArrowLeft" in DASHBOARD
    assert "event.key==='Enter'||event.key===' '" in DASHBOARD
    assert "aria-selected" in DASHBOARD


def test_feed_failures_are_persistent_and_self_describing() -> None:
    assert 'id="feedstatus" role="status" aria-live="polite"' in DASHBOARD
    assert "FEED ERROR:" in DASHBOARD
    assert "retrying automatically" in DASHBOARD
    assert "fs.textContent" in DASHBOARD


def test_work_progress_is_accessible_and_queued_jobs_are_cancellable() -> None:
    assert 'role="progressbar"' in DASHBOARD
    assert 'aria-valuenow="${value}"' in DASHBOARD
    assert "cancelResearchJob" in DASHBOARD
    assert "'/api/research/jobs/'+encodeURIComponent(id)+'/cancel'" in DASHBOARD
    assert "job.wait_reason" in DASHBOARD


def test_polling_yields_when_hidden_and_slows_when_idle() -> None:
    assert "document.hidden" in DASHBOARD
    assert "visibilitychange" in DASHBOARD
    assert "adaptivePoll" in DASHBOARD
    assert "(window._hasActiveWork||window._engineActive)?activeMs:idleMs" in DASHBOARD
    assert "setInterval(()=>{guard(refreshEngine" not in DASHBOARD


def test_connect_control_requires_live_robinhood_configuration() -> None:
    assert "robinhoodConfigured" in DASHBOARD
    assert "connectApplicable" in DASHBOARD
    assert "!SIM&&robinhoodConfigured" in DASHBOARD


def test_high_risk_api_payloads_are_html_escaped() -> None:
    assert "${esc((r.payload" in DASHBOARD
    assert "${esc(c.thesis" in DASHBOARD
    assert "${esc(p.error)}" in DASHBOARD
    assert "${esc(m.text)}" in DASHBOARD
    assert "${esc(job.kind)}" in DASHBOARD
    assert "data-graph-node=\"${esc(n.id)}\"" in DASHBOARD


def test_compact_reflow_contract_exists() -> None:
    assert "@media (max-width:700px)" in DASHBOARD
    assert "grid-template-columns:minmax(0,1fr)" in DASHBOARD
    assert "overflow-x:auto" in DASHBOARD
    assert "font-size:16px" in DASHBOARD

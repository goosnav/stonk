"""GUI smoke: boot the real app on a random port, render with headless
Chromium, assert zero JS errors and populated key elements on every tab.

Skipped automatically when playwright/chromium isn't installed — the offline
suite stays fast; run this one before any release (`pytest tests/test_gui.py`).
"""
from __future__ import annotations

import socket
import threading
import time

import pytest

playwright = pytest.importorskip("playwright.sync_api")


@pytest.fixture()
def gui_server(cfg, store):
    import uvicorn
    from specforge.app import create_app
    # seed a little state so panels aren't all empty
    store.record_equity(1000, 1000, "paper")
    app = create_app(cfg, store, with_scheduler=False)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port,
                                           log_level="error"))
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    for _ in range(50):
        if server.started:
            break
        time.sleep(0.1)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True


def test_dashboard_renders_all_tabs_without_js_errors(gui_server):
    from playwright.sync_api import sync_playwright
    errors: list[str] = []
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page()
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        pg.goto(gui_server, wait_until="networkidle", timeout=30000)
        pg.wait_for_timeout(1500)
        assert "STONK TERMINAL" in (pg.text_content("#brand") or "")
        assert "$" in (pg.text_content("#acct") or "")          # account cards filled
        for tab in ["trading", "switchboard", "risk", "model", "activity", "overview"]:
            pg.click(f'#tabs div[data-p="{tab}"]')
            pg.wait_for_timeout(300)
        # switchboard table populated from config
        pg.click('#tabs div[data-p="switchboard"]')
        assert "momentum" in (pg.text_content("#nodes") or "")
        # model tab renders the network + ledger (V4)
        pg.click('#tabs div[data-p="model"]')
        pg.wait_for_timeout(800)
        assert "momentum" in (pg.text_content("#model_table") or "")
        assert pg.query_selector("#modelsvg svg") is not None
        b.close()
    # transient fetch aborts during teardown are fine; real JS errors are not
    real = [e for e in errors if "Failed to fetch" not in e and "NetworkError" not in e]
    assert not real, f"JS errors: {real}"


def test_dashboard_keyboard_tabs_and_narrow_reflow(gui_server):
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 400, "height": 900})
        page.goto(gui_server, wait_until="networkidle", timeout=30000)
        tabs = page.get_by_role("tab")
        assert tabs.count() == 8
        tabs.nth(0).focus()
        page.keyboard.press("End")
        assert tabs.nth(7).get_attribute("aria-selected") == "true"
        page.keyboard.press("Home")
        assert tabs.nth(0).get_attribute("aria-selected") == "true"
        width = page.evaluate("document.documentElement.scrollWidth")
        assert width <= 400
        browser.close()


def test_first_load_feed_failure_remains_visible(gui_server):
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.route("**/api/status", lambda route: route.abort())
        page.goto(gui_server, wait_until="networkidle", timeout=30000)
        page.wait_for_timeout(1800)
        status = page.get_by_role("status").filter(has_text="FEED ERROR")
        assert status.count() >= 1
        assert "retrying automatically" in (page.text_content("#feedstatus") or "")
        browser.close()

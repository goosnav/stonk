"""Robinhood Agentic Trading MCP adapter (standalone MCP client, D6 primary).

Talks to https://agent.robinhood.com/mcp/trading over streamable HTTP with
MCP-spec OAuth 2.1 (dynamic client registration + browser flow + local
callback). Robinhood officially supports named clients (Claude, ChatGPT,
Cursor, ...); custom-client registration is UNVERIFIED — if the OAuth
handshake is rejected, connect() raises BrokerAuthError and the operator
should switch config broker to `robinhood_bridge` (see broker/bridge.py).

Tool-mapping facts encoded from the live tool schemas (captured 2026-07-06
from a connected RH MCP session — see dev/DECISIONS.md D12):
- get_accounts (no args) → account list; agentic_allowed accounts only can trade
- get_portfolio(account_number) → market value + buying power
- get_equity_positions(account_number) → symbol/quantity/average cost
- get_equity_quotes(symbols=[...])
- review/place_equity_order: ALL numbers are strings; `ref_id` (UUID) is the
  idempotency key; fractional quantities require type=market + regular_hours
  (limit orders need whole shares); limit_price required for type=limit.
- get_equity_orders(account_number, order_id=...) → poll order state.

Live-order preconditions enforced here (defense in depth; the risk governor
already gates upstream): config triple-gate passes AND account is in
RH_ACCOUNT_WHITELIST AND account is agentic_allowed.
"""
from __future__ import annotations

import asyncio
import json
import os
import threading
import webbrowser
from datetime import datetime
from pathlib import Path

from ..models import AccountState, Fill, OrderIntent, OrderReview, Position

RH_MCP_URL = "https://agent.robinhood.com/mcp/trading"
TOKEN_PATH = Path.home() / ".specforge" / "rh_tokens.json"
CALLBACK_PORT = 8425


class BrokerAuthError(RuntimeError):
    """OAuth to the RH MCP failed — likely custom clients are not allowed.
    Fallback: set broker: robinhood_bridge in config (see scripts/bridge_prompt.md)."""


def _f(v, default=0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _first(d: dict, *keys, default=None):
    """Defensive key lookup — RH response shapes may evolve (AGENTS.md §2)."""
    for k in keys:
        if isinstance(d, dict) and d.get(k) is not None:
            return d[k]
    return default


class RobinhoodMCPBroker:
    name = "robinhood_mcp"

    def __init__(self, cfg, store):
        self.cfg = cfg
        self.store = store
        self.account_number = None      # resolved on first use
        ok, why = cfg.live_trading_allowed()
        self._live_ok, self._live_why = ok, why

    # ---------------- MCP plumbing ----------------
    async def _call_async(self, tool: str, args: dict) -> dict:
        from mcp import ClientSession
        from mcp.client.auth import OAuthClientProvider, TokenStorage
        from mcp.client.streamable_http import streamablehttp_client
        from mcp.shared.auth import (OAuthClientInformationFull,
                                     OAuthClientMetadata, OAuthToken)

        class FileTokenStorage(TokenStorage):
            async def get_tokens(self):
                if TOKEN_PATH.exists():
                    d = json.loads(TOKEN_PATH.read_text()).get("tokens")
                    return OAuthToken(**d) if d else None
                return None

            async def set_tokens(self, tokens):
                TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
                d = json.loads(TOKEN_PATH.read_text()) if TOKEN_PATH.exists() else {}
                d["tokens"] = tokens.model_dump(mode="json")
                TOKEN_PATH.write_text(json.dumps(d))
                TOKEN_PATH.chmod(0o600)

            async def get_client_info(self):
                if TOKEN_PATH.exists():
                    d = json.loads(TOKEN_PATH.read_text()).get("client")
                    return OAuthClientInformationFull(**d) if d else None
                return None

            async def set_client_info(self, info):
                TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
                d = json.loads(TOKEN_PATH.read_text()) if TOKEN_PATH.exists() else {}
                d["client"] = info.model_dump(mode="json")
                TOKEN_PATH.write_text(json.dumps(d))
                TOKEN_PATH.chmod(0o600)

        code_holder: dict = {}

        async def redirect_handler(auth_url: str):
            print(f"\nRobinhood OAuth: opening browser →\n  {auth_url}\n")
            webbrowser.open(auth_url)

        async def callback_handler():
            # one-shot localhost HTTP server catches the OAuth redirect
            from http.server import BaseHTTPRequestHandler, HTTPServer
            from urllib.parse import parse_qs, urlparse
            done = threading.Event()

            class H(BaseHTTPRequestHandler):
                def do_GET(self):          # noqa: N802
                    q = parse_qs(urlparse(self.path).query)
                    code_holder["code"] = (q.get("code") or [None])[0]
                    code_holder["state"] = (q.get("state") or [None])[0]
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b"SpecForge: auth complete, close this tab.")
                    done.set()

                def log_message(self, *a):  # silence
                    pass

            srv = HTTPServer(("127.0.0.1", CALLBACK_PORT), H)
            t = threading.Thread(target=srv.serve_forever, daemon=True)
            t.start()
            ok = await asyncio.get_event_loop().run_in_executor(
                None, done.wait, 300)     # 5 min for the human to log in
            srv.shutdown()
            if not ok or not code_holder.get("code"):
                raise BrokerAuthError("OAuth callback never arrived")
            return code_holder["code"], code_holder["state"]

        oauth = OAuthClientProvider(
            server_url=RH_MCP_URL,
            client_metadata=OAuthClientMetadata(
                client_name="SpecForge",
                redirect_uris=[f"http://127.0.0.1:{CALLBACK_PORT}/callback"],
                grant_types=["authorization_code", "refresh_token"],
                response_types=["code"],
            ),
            storage=FileTokenStorage(),
            redirect_handler=redirect_handler,
            callback_handler=callback_handler,
        )
        try:
            async with streamablehttp_client(RH_MCP_URL, auth=oauth) as (r, w, _):
                async with ClientSession(r, w) as session:
                    await session.initialize()
                    res = await session.call_tool(tool, args)
        except BrokerAuthError:
            raise
        except Exception as e:            # noqa: BLE001 — surface as auth/transport
            raise BrokerAuthError(
                f"RH MCP call failed ({type(e).__name__}: {e}). If this is an "
                f"OAuth/registration rejection, switch config broker to "
                f"robinhood_bridge.") from e
        if getattr(res, "structuredContent", None):
            return res.structuredContent
        for block in res.content or []:
            if getattr(block, "type", "") == "text":
                try:
                    return json.loads(block.text)
                except json.JSONDecodeError:
                    return {"text": block.text}
        return {}

    def _call(self, tool: str, args: dict) -> dict:
        out = asyncio.run(self._call_async(tool, args))
        self.store.audit("rh_mcp_call", {"tool": tool, "args": args,
                                         "response_keys": list(out)[:20]})
        # RH wraps payloads as {"data": {...}, "guide": "..."} (observed live
        # 2026-07-06); unwrap once here so parsers see the payload directly
        if isinstance(out, dict) and isinstance(out.get("data"), (dict, list)):
            out = out["data"] if isinstance(out["data"], dict) else {"results": out["data"]}
        return out

    # ---------------- connect probe (GUI "Connect Robinhood" flow) ----------
    def probe(self) -> dict:
        """Read-only connection test: OAuth + list accounts. Never places
        orders; safe regardless of the live triple-gate. Returns a dict the
        GUI can render directly."""
        res = self._call("get_accounts", {})
        accounts = _first(res, "accounts", "results",
                          default=res if isinstance(res, list) else [])
        wl = os.environ.get("RH_ACCOUNT_WHITELIST", "")
        out = []
        for a in accounts or []:
            out.append({"account_number": _first(a, "account_number", "number", "id"),
                        "agentic_allowed": bool(_first(a, "agentic_allowed",
                                                       "is_agentic", default=False)),
                        "type": _first(a, "type", "account_type", default="")})
        return {"connected": True, "accounts": out, "whitelist": wl,
                "probed_at": datetime.now().astimezone().isoformat(timespec="seconds")}

    # ---------------- account resolution ----------------
    def _account(self) -> str:
        if self.account_number:
            return self.account_number
        whitelist = {a.strip() for a in
                     os.environ.get("RH_ACCOUNT_WHITELIST", "").split(",") if a.strip()}
        res = self._call("get_accounts", {})
        accounts = _first(res, "accounts", "results", default=res if isinstance(res, list) else [])
        usable = []
        for a in accounts or []:
            num = _first(a, "account_number", "number", "id")
            agentic = _first(a, "agentic_allowed", "is_agentic", default=False)
            if num and num in whitelist and agentic:
                usable.append(num)
        if not usable:
            raise BrokerAuthError(
                f"no whitelisted agentic account found (whitelist={sorted(whitelist)}, "
                f"accounts seen={len(accounts or [])}). Fund the RH Agentic account "
                f"and put its number in RH_ACCOUNT_WHITELIST.")
        self.account_number = usable[0]
        return self.account_number

    # ---------------- BrokerAdapter ----------------
    def get_account(self) -> AccountState:
        acct = self._account()
        port = self._call("get_portfolio", {"account_number": acct})
        pos_res = self._call("get_equity_positions", {"account_number": acct})
        positions = []
        for p in _first(pos_res, "positions", "results", default=[]) or []:
            sym = _first(p, "symbol", "ticker")
            qty = _f(_first(p, "quantity", "qty"))
            if sym and qty > 0:
                positions.append(Position(
                    symbol=sym, asset_type="equity", qty=qty,
                    avg_cost=_f(_first(p, "average_cost", "avg_cost",
                                       "average_buy_price")),
                    opened_at=_first(p, "created_at", default="")))
        # live shape (observed 2026-07-06): total_value/cash as strings,
        # buying_power NESTED as {"buying_power": "50.0000", ...}
        equity = _f(_first(port, "total_value", "market_value", "equity"))
        cash = _f(_first(port, "cash", "cash_balance", "uninvested_cash"))
        bp_raw = _first(port, "buying_power", default=cash)
        bp = _f(_first(bp_raw, "buying_power", default=None)) \
            if isinstance(bp_raw, dict) else _f(bp_raw, cash)
        return AccountState(equity=equity or (cash + sum(p.cost_basis for p in positions)),
                            cash=cash, buying_power=bp, positions=positions,
                            as_of=datetime.now().astimezone().isoformat())

    def get_quotes(self, symbols: list[str]) -> dict[str, float]:
        res = self._call("get_equity_quotes", {"symbols": symbols})
        out = {}
        for row in _first(res, "results", "quotes", default=[]) or []:
            q = row.get("quote", row) if isinstance(row, dict) else {}
            sym = _first(q, "symbol", "ticker")
            px = _f(_first(q, "last_trade_price", "last", "price", "mark_price"))
            if sym and px:
                out[sym] = px
        return out

    def _order_args(self, intent: OrderIntent) -> dict:
        """RH constraint: fractional qty ⇒ market order in regular hours;
        whole shares ⇒ limit order at our computed limit price."""
        fractional = abs(intent.qty - round(intent.qty)) > 1e-9
        args = {
            "account_number": self._account(),
            "symbol": intent.symbol,
            "side": intent.side,
            "time_in_force": "gfd",
            "market_hours": "regular_hours",
        }
        if fractional:
            args["type"] = "market"
            args["quantity"] = f"{intent.qty:.6f}"
        else:
            args["type"] = "limit"
            args["quantity"] = str(int(round(intent.qty)))
            args["limit_price"] = f"{intent.limit_price:.2f}"
        return args

    def review_order(self, intent: OrderIntent) -> OrderReview:
        if not self._live_ok:
            return OrderReview(ok=False, warnings=[f"live gate: {self._live_why}"])
        res = self._call("review_equity_order", self._order_args(intent))
        alerts = _first(res, "alerts", "warnings", "pre_trade_alerts", default=[]) or []
        warnings = [str(_first(a, "message", "title", default=a))[:200] for a in alerts] \
            if isinstance(alerts, list) else [str(alerts)[:200]]
        # live shape (observed 2026-07-06): pre-trade alerts arrive as
        # order_checks.alertType. Anything not explicitly known-benign counts
        # as a warning — unknown broker signals must never pass silently.
        checks = res.get("order_checks") or {}
        alert_type = checks.get("alertType") if isinstance(checks, dict) else str(checks)
        benign = {None, "", "EQUITY_OVERNIGHT_MARKET_BUY_FTUX_POPUP"}  # informational FTUX
        if alert_type not in benign:
            warnings.append(f"order_check:{alert_type}")
        # unknown/severe warning ⇒ not ok (AGENTS.md §34.16); engine then skips
        return OrderReview(ok=not warnings, warnings=warnings, raw=res)

    def place_order(self, intent: OrderIntent) -> Fill | None:
        if not self._live_ok:
            raise BrokerAuthError(f"live trading blocked: {self._live_why}")
        args = self._order_args(intent)
        import uuid as _uuid
        args["ref_id"] = str(_uuid.uuid5(_uuid.NAMESPACE_URL, intent.idempotency_key))
        res = self._call("place_equity_order", args)
        order = _first(res, "order", default=res)
        broker_id = _first(order, "id", "order_id")
        self.store.update_order(intent.id, broker_order_id=broker_id)
        state = str(_first(order, "state", "status", default="")).lower()
        if state == "filled":
            return Fill(order_id=intent.id, symbol=intent.symbol, side=intent.side,
                        qty=_f(_first(order, "filled_quantity", "quantity"), intent.qty),
                        price=_f(_first(order, "average_price", "price"), intent.limit_price),
                        filled_at=datetime.now().astimezone().isoformat())
        return None                        # resting — reconciled next cycle

    def poll_order(self, broker_order_id: str, intent: OrderIntent) -> Fill | str | None:
        """→ Fill when filled, 'dead' when cancelled/rejected, None when open."""
        res = self._call("get_equity_orders",
                         {"account_number": self._account(), "order_id": broker_order_id})
        orders = _first(res, "orders", "results", default=[]) or []
        if not orders:
            return None
        o = orders[0]
        state = str(_first(o, "state", "status", default="")).lower()
        if state == "filled":
            return Fill(order_id=intent.id, symbol=intent.symbol, side=intent.side,
                        qty=_f(_first(o, "filled_quantity", "cumulative_quantity"),
                               intent.qty),
                        price=_f(_first(o, "average_price", "executed_price"),
                                 intent.limit_price),
                        filled_at=_first(o, "updated_at",
                                         default=datetime.now().astimezone().isoformat()))
        if state in ("cancelled", "rejected", "failed", "voided"):
            return "dead"
        return None

    def cancel_order(self, broker_order_id: str) -> bool:
        res = self._call("cancel_equity_order",
                         {"account_number": self._account(),
                          "order_id": broker_order_id})
        return "error" not in json.dumps(res).lower()

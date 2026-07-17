"""Config loading: configs/default.yaml overlaid by configs/<mode>.yaml overlaid
by runtime GUI edits stored in the DB (applied by app layer). Dangerous values
are rejected unless advanced_override is set."""
from __future__ import annotations

import copy
import os
import re
import tempfile
import unicodedata
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "configs"
_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _load_dotenv() -> None:
    """Minimal .env loader (stdlib): real env vars win over file values."""
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            key = k.strip()
            if os.environ.get("STONK_RESEARCH_WORKER") == "1" and \
                    key.upper().startswith(("RH_", "ROBINHOOD_", "MCP_")):
                continue
            os.environ.setdefault(key, v.strip())


_load_dotenv()


def _validate_env_pair(key: str, value: str) -> None:
    if not isinstance(key, str) or not _ENV_KEY_RE.fullmatch(key):
        raise ValueError("invalid environment variable name")
    if not isinstance(value, str):
        raise TypeError("environment variable value must be a string")
    if any(unicodedata.category(char).startswith("C") or
           unicodedata.category(char) in {"Zl", "Zp"} for char in value):
        raise ValueError("environment variable value contains control characters")


def set_env_vars(values: dict[str, str]) -> None:
    """Atomically persist one or more validated environment settings."""
    for key, value in values.items():
        _validate_env_pair(key, value)
    env = ROOT / ".env"
    lines = env.read_text().splitlines() if env.exists() else []
    for key, value in values.items():
        prefix = f"{key}="
        for i, line in enumerate(lines):
            if line.lstrip().startswith(prefix):
                lines[i] = f"{key}={value}"
                break
        else:
            lines.append(f"{key}={value}")

    # Write beside the destination and replace it atomically.  The restrictive
    # mode is applied at creation time so a crash cannot briefly expose a
    # secret through a world-readable temporary file.
    fd, temp_name = tempfile.mkstemp(prefix=".env.", dir=ROOT)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as temp_file:
            temp_file.write("\n".join(lines) + "\n")
            temp_file.flush()
            os.fsync(temp_file.fileno())
        os.replace(temp_name, env)
        try:
            env.chmod(0o600)            # secrets live here — keep it owner-only
        except OSError:
            pass                        # temp file was already created as 0600
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise
    os.environ.update(values)


def set_env_var(key: str, value: str) -> None:
    """Upsert one KEY=value into ROOT/.env AND apply it to os.environ live.
    This is the persistent secret store (.env is gitignored, chmod 600). Live
    apply means the next AIClient() — built fresh each scan — picks it up
    without a server restart. Callers must never log `value`."""
    set_env_vars({key: value})

# (path, predicate, message) — governor-level sanity on config itself
_DANGEROUS = [
    (("risk", "kill_switch_drawdown"), lambda v: v > 0.5, "kill_switch_drawdown > 50%"),
    (("risk", "max_daily_loss"), lambda v: v > 0.10, "max_daily_loss > 10%"),
    (("risk", "max_single_equity_position"), lambda v: v > 0.25, "single position > 25% of account"),
    (("risk", "time_step_budget_pct"), lambda v: v > 0.5, "time-step budget > 50% of equity"),
    (("risk", "max_account_deployment"), lambda v: v > 1.0, "deployment > 100% (leverage)"),
    (("execution", "order_type"), lambda v: v == "market", "market orders"),
]


def _deep_merge(base: dict, overlay: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (overlay or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _get(cfg: dict, path: tuple):
    cur = cfg
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


class ConfigError(Exception):
    pass


class Config:
    def __init__(self, data: dict):
        self.data = data

    def __getitem__(self, key):
        return self.data[key]

    def get(self, *path, default=None):
        v = _get(self.data, tuple(path))
        return default if v is None else v

    @property
    def mode(self) -> str:
        return self.data.get("mode", "paper")

    def validate(self) -> list[str]:
        """Return warnings; raise ConfigError on dangerous values w/o override."""
        warnings = []
        for path, pred, msg in _DANGEROUS:
            v = _get(self.data, path)
            if v is not None and pred(v):
                if self.data.get("advanced_override"):
                    warnings.append(f"advanced_override active: {msg}")
                else:
                    raise ConfigError(f"Dangerous config rejected: {msg} "
                                      f"(set advanced_override: true to force)")
        return warnings

    def live_trading_allowed(self) -> tuple[bool, str]:
        """Live orders require config flag AND env var AND account whitelist."""
        if not self.data.get("live_trading_enabled"):
            return False, "config live_trading_enabled is false"
        if os.environ.get("LIVE_TRADING_ENABLED", "").lower() != "true":
            return False, "env LIVE_TRADING_ENABLED != true"
        if self.data.get("broker", "").startswith("robinhood") and \
                not os.environ.get("RH_ACCOUNT_WHITELIST", "").strip():
            return False, "RH_ACCOUNT_WHITELIST is empty"
        return True, "ok"


def load_config(mode: str | None = None, overrides: dict | None = None) -> Config:
    base = yaml.safe_load((CONFIG_DIR / "default.yaml").read_text())
    mode = mode or base.get("mode", "paper")
    mode_file = CONFIG_DIR / f"{mode}.yaml"
    merged = _deep_merge(base, yaml.safe_load(mode_file.read_text()) if mode_file.exists() else {})
    if overrides:
        merged = _deep_merge(merged, overrides)
    cfg = Config(merged)
    cfg.validate()
    return cfg


OVERRIDES_KEY = "config_overrides"


def apply_override(store, mode: str, path: list[str], value, via: str = "gui") -> None:
    """The ONE validated write path for runtime config overrides (GUI and
    steering both route here). Validates the merged result BEFORE persisting —
    no caller can sneak a dangerous value past the governor."""
    ov = store.kv_get(OVERRIDES_KEY, {}) or {}
    cur = ov
    for k in path[:-1]:
        cur = cur.setdefault(k, {})
    cur[path[-1]] = value
    load_config(mode, overrides=ov)          # raises ConfigError on danger
    store.kv_set(OVERRIDES_KEY, ov)
    store.audit("config_override", {"path": path, "value": value, "via": via})

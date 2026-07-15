"""Security regression tests for persistent environment settings."""
from __future__ import annotations

import os
import stat

import pytest

from specforge import config


@pytest.fixture
def isolated_env(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ROOT", tmp_path)
    return tmp_path / ".env"


@pytest.mark.parametrize("key", ["", "1TOKEN", "BAD-KEY", "BAD KEY", "A=B", "A\nB"])
def test_set_env_var_rejects_unsafe_keys_without_mutation(isolated_env, monkeypatch, key):
    isolated_env.write_text("KEEP=safe\n")
    monkeypatch.delenv(key, raising=False)

    with pytest.raises(ValueError, match="invalid environment variable name"):
        config.set_env_var(key, "secret")

    assert isolated_env.read_text() == "KEEP=safe\n"
    assert key not in os.environ


@pytest.mark.parametrize("value", [
    "secret\nINJECTED=true",
    "secret\rINJECTED=true",
    "secret\x00tail",
    "secret\ttail",
    "secret\u0085tail",
    "secret\u2028tail",
    "secret\u2029tail",
    "secret\u200btail",
])
def test_set_env_var_rejects_line_and_control_injection_without_mutation(
        isolated_env, monkeypatch, value):
    isolated_env.write_text("AI_API_KEY=original\n")
    monkeypatch.setenv("AI_API_KEY", "original")

    with pytest.raises(ValueError, match="control characters"):
        config.set_env_var("AI_API_KEY", value)

    assert isolated_env.read_text() == "AI_API_KEY=original\n"
    assert os.environ["AI_API_KEY"] == "original"


def test_set_env_var_overwrites_appends_and_keeps_owner_only_permissions(
        isolated_env, monkeypatch):
    isolated_env.write_text("KEEP=one\nAI_API_KEY=old\n")
    isolated_env.chmod(0o644)
    monkeypatch.delenv("AI_API_KEY", raising=False)
    monkeypatch.delenv("NEW_KEY", raising=False)

    config.set_env_var("AI_API_KEY", "new-secret")
    config.set_env_var("NEW_KEY", "new-value")

    assert isolated_env.read_text() == (
        "KEEP=one\nAI_API_KEY=new-secret\nNEW_KEY=new-value\n")
    assert os.environ["AI_API_KEY"] == "new-secret"
    assert os.environ["NEW_KEY"] == "new-value"
    assert stat.S_IMODE(isolated_env.stat().st_mode) == 0o600


def test_atomic_write_failure_does_not_mutate_file_or_live_environment(
        isolated_env, monkeypatch):
    isolated_env.write_text("AI_API_KEY=original\n")
    monkeypatch.setenv("AI_API_KEY", "original")

    def fail_replace(_source, _destination):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(config.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated replace failure"):
        config.set_env_var("AI_API_KEY", "replacement")

    assert isolated_env.read_text() == "AI_API_KEY=original\n"
    assert os.environ["AI_API_KEY"] == "original"
    assert not list(isolated_env.parent.glob(".env.*"))


def test_research_worker_dotenv_does_not_import_broker_credentials(
        isolated_env, monkeypatch):
    isolated_env.write_text("RH_ACCOUNT_WHITELIST=secret-account\nAI_API_KEY=allowed-ai-key\n")
    monkeypatch.setenv("STONK_RESEARCH_WORKER", "1")
    monkeypatch.delenv("RH_ACCOUNT_WHITELIST", raising=False)
    monkeypatch.delenv("AI_API_KEY", raising=False)
    config._load_dotenv()
    assert "RH_ACCOUNT_WHITELIST" not in os.environ
    assert os.environ["AI_API_KEY"] == "allowed-ai-key"

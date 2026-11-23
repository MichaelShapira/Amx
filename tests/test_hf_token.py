"""The Hugging Face token: shape, the placeholder, the status and the store path shared by the CLI, the deploy script
and the web app's Setup page."""
from __future__ import annotations

import datetime as dt

import pytest

from bench import hf_token
from bench.deployment import HF_TOKEN_UNSET


class FakeSM:
    def __init__(self, value=HF_TOKEN_UNSET):
        self.value, self.puts = value, []

    def describe_secret(self, SecretId):
        return {"Name": "amx-gpu-bench/hf-token", "LastChangedDate": dt.datetime(2026, 11, 23, 12, 0, tzinfo=dt.UTC)}

    def get_secret_value(self, SecretId):
        return {"SecretString": self.value}

    def put_secret_value(self, SecretId, SecretString):
        self.puts.append(SecretString)
        self.value = SecretString


def test_placeholder_and_junk_are_not_tokens():
    assert not hf_token.looks_like_token(HF_TOKEN_UNSET) and not hf_token.looks_like_token("") and not hf_token.looks_like_token(None)
    assert not hf_token.looks_like_token("ghp_abcdefghijklmnopqrstuvwxyz")
    assert hf_token.looks_like_token("hf_" + "a" * 30)


def test_status_reads_the_slot_without_returning_the_value():
    sm = FakeSM()
    st = hf_token.status(sm, "arn")
    assert st == {"configured": False, "name": "amx-gpu-bench/hf-token", "last_changed": "2026-11-23 12:00 UTC"}
    sm.value = "hf_" + "b" * 30
    assert hf_token.status(sm, "arn")["configured"] is True
    assert "hf_" not in str(st)


def test_store_validates_then_writes():
    sm = FakeSM()
    with pytest.raises(ValueError, match="does not look like a Hugging Face token"):
        hf_token.store(sm, "arn", "not-a-token")
    assert sm.puts == []
    st = hf_token.store(sm, "arn", "  hf_" + "c" * 30 + "\n")
    assert sm.puts == ["hf_" + "c" * 30] and st["configured"] is True


def test_verify_reports_a_rejected_token_without_echoing_it(monkeypatch):
    import urllib.error
    import urllib.request

    def boom(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", {}, None)
    monkeypatch.setattr(urllib.request, "urlopen", boom)
    out = hf_token.verify("hf_" + "d" * 30)
    assert out["ok"] is False and "401" in out["error"] and "hf_" not in out["error"]


def test_deploy_script_offers_the_token_as_an_optional_input():
    from pathlib import Path
    sh = (Path(__file__).resolve().parents[1] / "infra/deploy.sh").read_text()
    assert "HF_TOKEN" in sh and "--hf-token-stdin" in sh and "bench hf-token" in sh.replace("$BENCH", "bench")
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text()
    assert "HF_TOKEN=hf_… infra/deploy.sh" in readme and "Setup → Hugging Face token" in readme and "bench hf-token --stdin" in readme

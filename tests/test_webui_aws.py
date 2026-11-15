"""The local API's AWS credentials: refreshable from a file, and an expired session is a sentence,
not "Internal Server Error" (2026-11-15)."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest
from botocore.exceptions import ClientError, NoCredentialsError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from webui.api import aws  # noqa: E402

CREDS = """# pasted from the portal
export ISENGARD_PRODUCTION_ACCOUNT=false
export AWS_ACCESS_KEY_ID=ASIAEXAMPLE000000001
export AWS_SECRET_ACCESS_KEY="secret/one+two"
export AWS_SESSION_TOKEN=IQoJtoken1
"""


@pytest.fixture()
def creds_file(tmp_path, monkeypatch):
    p = tmp_path / "creds.env"
    p.write_text(CREDS)
    monkeypatch.setenv("AMXBENCH_CREDS_FILE", str(p))
    aws.reset_for_tests()
    yield p
    aws.reset_for_tests()


def test_env_file_parsing_keeps_only_the_three_credential_values():
    vals = aws.parse_env_file(CREDS)
    assert vals == {"AWS_ACCESS_KEY_ID": "ASIAEXAMPLE000000001", "AWS_SECRET_ACCESS_KEY": "secret/one+two",
                    "AWS_SESSION_TOKEN": "IQoJtoken1"}
    assert aws.parse_env_file("AWS_ACCESS_KEY_ID=  \n# AWS_SECRET_ACCESS_KEY=x\n") == {}


def test_session_is_rebuilt_when_the_file_changes_and_not_otherwise(creds_file):
    s1 = aws.session()
    assert s1.get_credentials().access_key == "ASIAEXAMPLE000000001"
    assert s1.get_credentials().token == "IQoJtoken1" and s1.region_name == "us-east-1"
    assert aws.session() is s1                      # same mtime: same session, no re-read
    assert "creds.env" in aws.source() and "written" in aws.source()
    time.sleep(0.02)
    creds_file.write_text(CREDS.replace("000000001", "000000002").replace("token1", "token2"))
    os.utime(creds_file, (time.time() + 2, time.time() + 2))   # a new mtime even on coarse filesystems
    s2 = aws.session()
    assert s2 is not s1 and s2.get_credentials().access_key == "ASIAEXAMPLE000000002"
    assert s2.get_credentials().token == "IQoJtoken2"
    # The pasted file is the operator's own machine; the API never copies the values anywhere.
    assert "ASIAEXAMPLE" not in aws.source() and "token2" not in aws.expired_sentence()


def test_without_a_file_the_default_chain_is_used(monkeypatch):
    monkeypatch.setenv("AMXBENCH_CREDS_FILE", "")
    aws.reset_for_tests()
    assert aws.creds_file() is None and aws.source() == "the default AWS credential chain"
    assert "start webui/start.sh again" in aws.expired_sentence() and "amxbench-creds.env" in aws.expired_sentence()
    aws.reset_for_tests()


def test_expired_recognition():
    def ce(code, op="Scan"):
        return ClientError({"Error": {"Code": code, "Message": "x"}}, op)
    for code in ("ExpiredToken", "ExpiredTokenException", "InvalidClientTokenId", "UnrecognizedClientException", "RequestExpired"):
        assert aws.is_expired(ce(code)), code
    assert aws.is_expired(NoCredentialsError())
    assert not aws.is_expired(ce("AccessDeniedException")) and not aws.is_expired(ce("ResourceNotFoundException"))
    assert not aws.is_expired(RuntimeError("x"))


def test_an_expired_session_is_a_401_sentence_not_a_500(creds_file, monkeypatch):
    """The 2026-11-15 screenshot: 'Internal Server Error' on the Runs page, an ExpiredTokenException
    traceback in the log. The same request now answers 401 with what to do, and the status route
    says the same with expired: true so the sidebar can name the problem."""
    from fastapi.testclient import TestClient

    from webui.api import main

    class Boom:
        def scan(self, **kw):
            raise ClientError({"Error": {"Code": "ExpiredTokenException", "Message": "The security token included in the "
                                                                                      "request is expired"}}, "Scan")

    monkeypatch.setattr(main, "_table", lambda: Boom())
    c = TestClient(main.app, raise_server_exceptions=False)
    r = c.get("/api/runs")
    assert r.status_code == 401
    body = r.json()
    assert body["code"] == "aws_session_expired"
    assert body["detail"].startswith("Your AWS session has expired (") and "creds.env" in body["detail"]
    assert "paste a fresh session into" in body["detail"] and "no restart" in body["detail"]
    assert "Internal Server Error" not in r.text and "Traceback" not in r.text

    class Sts:
        def get_caller_identity(self):
            raise ClientError({"Error": {"Code": "ExpiredToken", "Message": "expired"}}, "GetCallerIdentity")

    monkeypatch.setattr(main.aws, "client", lambda name: Sts())
    st = c.get("/api/aws/status").json()
    assert st["ok"] is False and st["expired"] is True and st["error"] == body["detail"] and "creds.env" in st["credentials"]


def test_other_aws_failures_are_a_502_with_the_service_message(creds_file, monkeypatch):
    from fastapi.testclient import TestClient

    from webui.api import main

    class Boom:
        def scan(self, **kw):
            raise ClientError({"Error": {"Code": "ProvisionedThroughputExceededException", "Message": "slow down"}}, "Scan")

    monkeypatch.setattr(main, "_table", lambda: Boom())
    c = TestClient(main.app, raise_server_exceptions=False)
    r = c.get("/api/runs")
    assert r.status_code == 502 and r.json()["code"] == "aws_error"
    assert r.json()["detail"] == "Scan failed: ProvisionedThroughputExceededException: slow down"

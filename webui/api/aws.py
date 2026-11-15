"""AWS clients for the local API, from credentials that can be refreshed without a restart.

The operator works with short-lived (about one hour) session credentials pasted from the account
portal. Until 2026-11-15 the API read them once from its process environment, so an hour after
the start every page said "Internal Server Error" (an ExpiredTokenException traceback in the
log) and only a restart with new credentials helped.

Now the API takes credentials, in this order:

1. A credentials file: `AMXBENCH_CREDS_FILE`, or `~/.aws-tmp/amxbench-creds.env` when that exists.
   Shell `export AWS_ACCESS_KEY_ID=…` lines (the portal's copy-paste format), mode 600. The file is
   re-read whenever its modification time changes, so pasting a new session into it is all that is
   needed; the next request uses it.
2. Otherwise the default botocore chain (environment, profiles, instance role), as before.

`client()` and `table()` are the only constructors the API uses; `is_expired()` recognises the
service errors that mean "the session is gone" so the HTTP layer can say so in one sentence
instead of a stack trace.
"""

from __future__ import annotations

import os
import re
import threading
import time
from pathlib import Path

import boto3
from botocore.exceptions import ClientError, NoCredentialsError, PartialCredentialsError

REGION = "us-east-1"
DEFAULT_CREDS_FILE = Path.home() / ".aws-tmp" / "amxbench-creds.env"
CRED_KEYS = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN")
EXPIRED_CODES = {"ExpiredToken", "ExpiredTokenException", "RequestExpired", "InvalidClientTokenId",
                 "UnrecognizedClientException", "InvalidToken", "AuthFailure"}
_LINE = re.compile(r"^\s*(?:export\s+)?([A-Z_][A-Z0-9_]*)\s*=\s*(.*?)\s*$")

_lock = threading.Lock()
_cache: dict = {"path": None, "mtime": None, "session": None, "source": None}


def creds_file() -> Path | None:
    """The credentials file in use, or None for the default chain."""
    env = os.environ.get("AMXBENCH_CREDS_FILE")
    if env is not None:
        return Path(env).expanduser() if env.strip() else None
    return DEFAULT_CREDS_FILE if DEFAULT_CREDS_FILE.exists() else None


def parse_env_file(text: str) -> dict[str, str]:
    """`export KEY=value` / `KEY="value"` lines -> the three credential values (others ignored)."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        m = _LINE.match(line)
        if not m or line.lstrip().startswith("#"):
            continue
        key, val = m.group(1), m.group(2)
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "'\"":
            val = val[1:-1]
        if key in CRED_KEYS and val:
            out[key] = val
    return out


def session() -> boto3.Session:
    """A boto3 Session for the current credentials; rebuilt when the credentials file changes."""
    path = creds_file()
    with _lock:
        if path is None:
            if _cache["session"] is None or _cache["path"] is not None:
                _cache.update(path=None, mtime=None, session=boto3.Session(), source="the default AWS credential chain")
            return _cache["session"]
        try:
            mtime = path.stat().st_mtime
        except FileNotFoundError:
            mtime = None
        if _cache["session"] is not None and _cache["path"] == path and _cache["mtime"] == mtime:
            return _cache["session"]
        vals = parse_env_file(path.read_text()) if mtime is not None else {}
        if "AWS_ACCESS_KEY_ID" in vals and "AWS_SECRET_ACCESS_KEY" in vals:
            sess = boto3.Session(aws_access_key_id=vals["AWS_ACCESS_KEY_ID"],
                                 aws_secret_access_key=vals["AWS_SECRET_ACCESS_KEY"],
                                 aws_session_token=vals.get("AWS_SESSION_TOKEN"), region_name=REGION)
            when = time.strftime("%H:%M", time.localtime(mtime))
            source = f"{_short(path)} (written {when})"
        else:
            sess, source = boto3.Session(), f"the default AWS credential chain ({_short(path)} has no credentials)"
        _cache.update(path=path, mtime=mtime, session=sess, source=source)
        return sess


def source() -> str:
    """Where the current credentials come from, for status messages."""
    session()
    return _cache["source"] or "the default AWS credential chain"


def client(name: str):
    return session().client(name, region_name=REGION)


def table(name: str = "amx-gpu-bench"):
    return session().resource("dynamodb", region_name=REGION).Table(name)


def is_expired(exc: BaseException) -> bool:
    """Does this exception mean the AWS session is gone (expired, revoked, or absent)?"""
    if isinstance(exc, (NoCredentialsError, PartialCredentialsError)):
        return True
    if isinstance(exc, ClientError):
        return exc.response.get("Error", {}).get("Code") in EXPIRED_CODES
    return False


def expired_sentence() -> str:
    path = creds_file()
    how = (f"paste a fresh session into {_short(path)} (the same `export AWS_…` lines); the app picks it up on its "
           "own, no restart." if path else
           "export a fresh session and start webui/start.sh again, or put the `export AWS_…` lines in "
           f"{_short(DEFAULT_CREDS_FILE)} so the app can re-read them without a restart.")
    return f"Your AWS session has expired ({source()}). To continue, {how}"


def _short(p: Path) -> str:
    try:
        return "~/" + str(p.relative_to(Path.home()))
    except ValueError:
        return str(p)


def reset_for_tests() -> None:
    with _lock:
        _cache.update(path=None, mtime=None, session=None, source=None)

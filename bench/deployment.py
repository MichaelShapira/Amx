"""Where this deployment lives: the account comes from the AWS credentials in use, never from the code.

Everything that used to carry an account id as a literal reads it here: the CDK app (through CDK_DEFAULT_ACCOUNT,
which the CDK CLI exports from the active credentials), the web API (the task role), the CLI and the tools (the
operator's session). The region and the Availability Zone id stay configuration: the instance catalog's prices,
the Spot pools and the AZ id (`use1-az1`) are tied to us-east-1, and a deployment elsewhere changes them together
(AMXBENCH_REGION / AMXBENCH_AZ_ID). The bucket name follows the Core stack's convention so a tool can name it
without an API call; the stack output is still the authority when it is reachable.
"""
from __future__ import annotations

import os
from functools import lru_cache

REGION = os.environ.get("AMXBENCH_REGION", "us-east-1")
AZ_ID = os.environ.get("AMXBENCH_AZ_ID", "use1-az1")
CORE_STACK = "AmxBench-Core"
HF_SECRET_NAME = "amx-gpu-bench/hf-token"      # noqa: S105 - the secret's NAME  # nosec B105
HF_TOKEN_UNSET = "unset"           # noqa: S105 - the placeholder until Setup stores a token  # nosec B105


@lru_cache(maxsize=1)
def account_id(session=None) -> str | None:
    """The account of the active credentials (CDK_DEFAULT_ACCOUNT when the CDK CLI set it, else STS); None when
    no credentials answer, so callers that only need a label can say so instead of failing."""
    env = os.environ.get("CDK_DEFAULT_ACCOUNT") or os.environ.get("AMXBENCH_ACCOUNT")
    if env:
        return env
    try:
        import boto3
        sess = session or boto3.Session(region_name=REGION)
        return sess.client("sts").get_caller_identity()["Account"]
    except Exception:  # noqa: BLE001 - offline (tests, synth without credentials): unknown
        return None


def bucket_name(account: str | None = None) -> str:
    """The Core stack's bucket for this account: the stack output when CloudFormation answers, else the naming
    convention the stack uses (amx-gpu-bench-<account>-<region>)."""
    try:
        import boto3
        st = boto3.client("cloudformation", region_name=REGION).describe_stacks(StackName=CORE_STACK)["Stacks"][0]
        out = {o["OutputKey"]: o["OutputValue"] for o in st.get("Outputs", [])}
        if out.get("BucketName"):
            return out["BucketName"]
    except Exception:  # noqa: BLE001, S110 - no stack reachable: fall through to the convention
        pass
    acct = account or account_id()
    if not acct:
        raise SystemExit("cannot name the bucket: no AWS credentials answer and the Core stack is not reachable")
    return f"amx-gpu-bench-{acct}-{REGION}"

"""Tests never use the operator's AWS session.

The web app reads credentials from ~/.aws-tmp/amxbench-creds.env when that file exists
(webui/api/aws.py). With a valid session in it, a test that creates the app's boto3 Session before
moto has been imported talks to the real account: tests/test_webui_api.py run on its own once
scanned the production run table and tried to delete a run there (it only answered 404 because
the run id did not exist). So, for every test process:

* the credential file is switched off and the environment carries throwaway keys, so a call that
  escapes a moto mock fails fast (InvalidClientTokenId) instead of reaching the account;
* moto is imported here, before any test module, so its request stubber is in botocore's handler
  list for every session created afterwards.
"""

from __future__ import annotations

import os

import moto  # noqa: F401 - imported first on purpose, see the module docstring

os.environ["AMXBENCH_CREDS_FILE"] = ""
os.environ["AWS_ACCESS_KEY_ID"] = "testing"
os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"  # nosec B105 (moto test credentials)
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
for _k in ("AWS_SESSION_TOKEN", "AWS_PROFILE", "AWS_SECURITY_TOKEN"):
    os.environ.pop(_k, None)

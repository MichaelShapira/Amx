"""Container entry point for the web app (webui/Dockerfile, AmxBench-WebApp).

1. Pull the read-only data the Test-data page shows (prompt sets, the DSPM dataset manifest and
   request pools) from the bench bucket into the image's working tree, when AMXBENCH_DATA_SYNC is
   set and the files are not there. Best effort with a time budget: the app serves without them
   (the page then says the data is not built) and never waits on S3 to start.
2. Start uvicorn on the bind main.bind_address() allows: a non-loopback bind is refused unless a
   Cognito user pool is configured, so an unauthenticated API can never be published.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

log = logging.getLogger("entrypoint")
ROOT = Path(__file__).resolve().parents[2]

# S3 prefix -> local directory (relative to ROOT). The same keys the pre-check Lambda reads.
SYNC = {"artifacts/prompts/": "prompts/generated", "datasets/dspm/v1/": "data/dspm/v1",
        "artifacts/dspm/v1/requests/": "data/dspm/v1/requests"}
MAX_OBJECT_BYTES = 64 * 2**20
BUDGET_S = 90


def sync_data(bucket: str, budget_s: float = BUDGET_S) -> dict:
    from webui.api import aws

    s3 = aws.client("s3")
    t0, copied, skipped = time.monotonic(), 0, 0
    for prefix, local in SYNC.items():
        dest = ROOT / local
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                if time.monotonic() - t0 > budget_s:
                    log.warning("data sync stopped after %.0f s; %d copied, %d skipped", budget_s, copied, skipped)
                    return {"copied": copied, "skipped": skipped, "complete": False}
                rel = obj["Key"][len(prefix):]
                if not rel or rel.endswith("/") or obj["Size"] > MAX_OBJECT_BYTES:
                    skipped += 1
                    continue
                target = (dest / rel).resolve()
                if not str(target).startswith(str(dest.resolve())):       # a key with ".." never escapes
                    skipped += 1
                    continue
                if target.exists() and target.stat().st_size == obj["Size"]:
                    skipped += 1
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(bucket, obj["Key"], str(target))
                copied += 1
    log.info("data sync: %d copied, %d already present, %.1f s", copied, skipped, time.monotonic() - t0)
    return {"copied": copied, "skipped": skipped, "complete": True}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    import json

    from webui.api import main as api

    host, port = api.bind_address()                 # refuses a public bind without Cognito
    if os.environ.get("AMXBENCH_DATA_SYNC", "").strip() in ("1", "true", "yes"):
        bucket = json.loads(os.environ.get("AMXBENCH_OUTPUTS", "{}")).get("BucketName")
        if bucket:
            try:
                sync_data(bucket)
            except Exception as e:  # noqa: BLE001 - the app serves without the data
                log.warning("data sync failed: %s: %s", type(e).__name__, str(e)[:200])
    import uvicorn

    uvicorn.run(api.app, host=host, port=port, log_level="info", proxy_headers=True, forwarded_allow_ips="*")
    return 0


if __name__ == "__main__":
    sys.exit(main())

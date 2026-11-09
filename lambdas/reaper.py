"""Reaper (design 8 step 5, Requirement 19.2): every 10 minutes, stop bench instances that
are idle (no activity for IDLE_MINUTES and no active run) or have run longer than
MAX_UPTIME_HOURS with no active run. Only instances tagged bench:managed=true are
considered, and the IAM policy only allows stopping those."""

from __future__ import annotations

import datetime as dt
import os
import time
from typing import Any

from lambdas import common


def _active(run_id: str | None) -> bool:
    if not run_id:
        return False
    item = common.table().get_item(Key={"pk": f"RUN#{run_id}", "sk": "META"}).get("Item")
    return bool(item and item.get("status") in ("running", "planned"))


def decide(instance: dict, activity: dict | None, *, active: bool, now: float, idle_min: int,
           max_uptime_h: float) -> str | None:
    """Return a reason to stop, or None."""
    if active:
        return None
    last = float((activity or {}).get("last_activity_at", 0))
    launched = instance["LaunchTime"].timestamp()
    if now - max(last, launched) > idle_min * 60:
        return f"idle for more than {idle_min} min"
    if now - launched > max_uptime_h * 3600:
        return f"running for more than {max_uptime_h} h"
    return None


def handler(event: dict, context: Any = None) -> dict:
    idle_min = int(os.environ.get("IDLE_MINUTES", "30"))
    max_up = float(os.environ.get("MAX_UPTIME_HOURS", "8"))
    ec2 = common.client("ec2")
    stopped = []
    pages = ec2.get_paginator("describe_instances").paginate(Filters=[
        {"Name": "tag:bench:managed", "Values": ["true"]}, {"Name": "instance-state-name", "Values": ["running"]}])
    now = time.time()
    for page in pages:
        for res in page["Reservations"]:
            for inst in res["Instances"]:
                tags = {t["Key"]: t["Value"] for t in inst.get("Tags", [])}
                tid = tags.get("bench:target", "")
                act = common.table().get_item(Key={"pk": f"TARGET#{tid}", "sk": "ACTIVITY"}).get("Item") or {}
                reason = decide(inst, common.plain(act), active=_active(act.get("active_run_id")), now=now,
                                idle_min=idle_min, max_uptime_h=max_up)
                if reason:
                    # One-time Spot instances cannot be stopped; they are ephemeral by design
                    # (models are re-staged from the S3 cache), so terminate them.
                    if inst.get("InstanceLifecycle") == "spot":
                        ec2.terminate_instances(InstanceIds=[inst["InstanceId"]])
                        action = "terminated"
                    else:
                        ec2.stop_instances(InstanceIds=[inst["InstanceId"]])
                        action = "stopped"
                    stopped.append({"instance_id": inst["InstanceId"], "target": tid, "reason": reason,
                                    "action": action, "at": dt.datetime.now(dt.UTC).isoformat()})
    return {"stopped": stopped}

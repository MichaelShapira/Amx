"""Reaper (design 8 step 5, Requirement 19.2): every 10 minutes, stop bench instances that
are idle (no activity for IDLE_MINUTES and no active run) or have run longer than
MAX_UPTIME_HOURS with no active run, and terminates any managed instance found stopped
(instances are per run and never restarted). Only instances tagged bench:managed=true are
considered, and the IAM policy only allows acting on those."""

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
        {"Name": "tag:bench:managed", "Values": ["true"]},
        {"Name": "instance-state-name", "Values": ["running", "stopped", "stopping"]}])
    now = time.time()
    for page in pages:
        for res in page["Reservations"]:
            for inst in res["Instances"]:
                tags = {t["Key"]: t["Value"] for t in inst.get("Tags", [])}
                tid, run_id = tags.get("bench:target", ""), tags.get("bench:run", "")
                state = inst["State"]["Name"]
                if state in ("stopped", "stopping"):
                    # Instances are per run and never restarted: a stopped one is a leftover that
                    # only keeps its root volume on the bill.
                    reason = "stopped instance left behind (instances are per run and never reused)"
                else:
                    # The run that launched it (bench:run) is the one that may still be using it;
                    # the heartbeat row is per (target, run).
                    act = common.table().get_item(Key=common.activity_key(tid, run_id)).get("Item") or {} if run_id else {}
                    reason = decide(inst, common.plain(act), active=_active(run_id or None), now=now,
                                    idle_min=idle_min, max_uptime_h=max_up)
                if reason:
                    # Always terminate: nothing is reused between runs, and the root volume
                    # (delete_on_termination) goes with the instance.
                    ec2.terminate_instances(InstanceIds=[inst["InstanceId"]])
                    stopped.append({"instance_id": inst["InstanceId"], "target": tid, "run": run_id, "reason": reason,
                                    "action": "terminated", "at": dt.datetime.now(dt.UTC).isoformat()})
    return {"stopped": stopped}

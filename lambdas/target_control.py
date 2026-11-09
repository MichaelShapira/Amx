"""Target control (design 8): launch or start instances, configure cells, recover from
Spot interruptions, stop or terminate instances.

Each target stack owns a launch template; the instance itself is launched here per run,
On-Demand or Spot (one-time, terminate on interruption), and found again by its
bench:target tag. Actions (event["action"]):

  ensure_running  find the target's instance or launch one; returns {"ready": false} while
                  it is pending or capacity is short, so the state machine can wait and
                  call again (every call is idempotent). Spot capacity errors are retried
                  for purchase.spot_capacity_window_s, then an On-Demand instance is
                  launched when spot_fallback_on_demand is set.
  configure_start send `serve.sh start <spec>` through SSM; returns the command id
  configure_poll  {done, ok}; on success returns endpoint, api_path and the levels still
                  to run for the cell (levels already done or failed are skipped)
  check           after a cell step failed: if the instance is gone (interrupted, stopped,
                  terminated) record it and relaunch ("relaunch"), otherwise resume the
                  remaining levels ("resume") or give up ("fail")
  resume_levels   write the remaining level list for the Levels map
  stop            stop an On-Demand instance, terminate a Spot one
"""

from __future__ import annotations

import time
from typing import Any

from botocore.exceptions import ClientError

from lambdas import common

CAPACITY_CODES = {"InsufficientInstanceCapacity", "InstanceLimitExceeded", "Unsupported",
                  "InsufficientCapacityOnHost", "MaxSpotInstanceCountExceeded", "SpotMaxPriceTooLow",
                  "RequestLimitExceeded"}
TOTAL_LAUNCH_WINDOW_S = 1800        # give up on a target after this long without a running instance
CALL_BUDGET_S = 240                 # waiting inside one invocation; the state machine loops otherwise
LIVE_STATES = ("pending", "running", "stopping", "stopped")


def _ec2():
    return common.client("ec2")


def _instance(iid: str) -> dict | None:
    try:
        r = _ec2().describe_instances(InstanceIds=[iid])
    except ClientError as e:
        if e.response["Error"]["Code"].startswith("InvalidInstanceID"):
            return None
        raise
    return r["Reservations"][0]["Instances"][0] if r["Reservations"] else None


def find_instance(target_id: str) -> dict | None:
    """The target's live instance (newest launch), found by tag."""
    r = _ec2().describe_instances(Filters=[
        {"Name": "tag:bench:target", "Values": [target_id]}, {"Name": "tag:bench:managed", "Values": ["true"]},
        {"Name": "instance-state-name", "Values": list(LIVE_STATES)}])
    insts = [i for res in r["Reservations"] for i in res["Instances"]]
    return max(insts, key=lambda i: i["LaunchTime"]) if insts else None


def lifecycle(inst: dict) -> str:
    return "spot" if inst.get("InstanceLifecycle") == "spot" else "on-demand"


def _param(name: str) -> str:
    return common.client("ssm").get_parameter(Name=name)["Parameter"]["Value"]


def _log(run_id: str, target_id: str, key: str, entry: dict) -> None:
    """Append to runs/<run>/<key>/<target>.json."""
    k = f"runs/{run_id}/{key}/{target_id}.json"
    try:
        events = common.get_json(k)
    except ClientError:
        events = []
    events.append(entry)
    common.put_json(k, events)


def _activity(target_id: str) -> dict:
    item = common.table().get_item(Key={"pk": f"TARGET#{target_id}", "sk": "ACTIVITY"}).get("Item") or {}
    return common.plain(item)


def _set_activity(target_id: str, **vals: Any) -> None:
    sets = ", ".join(f"#{k} = :{k}" for k in vals)
    common.table().update_item(Key={"pk": f"TARGET#{target_id}", "sk": "ACTIVITY"},
                               UpdateExpression=f"SET {sets}", ExpressionAttributeNames={f"#{k}": k for k in vals},
                               ExpressionAttributeValues={f":{k}": common._ddb(v) for k, v in vals.items()})


def market_options(purchase: dict) -> dict | None:
    if purchase.get("type", "on-demand") != "spot":
        return None
    opts = {"SpotInstanceType": "one-time", "InstanceInterruptionBehavior": "terminate"}
    if purchase.get("spot_max_price_usd"):
        opts["MaxPrice"] = str(purchase["spot_max_price_usd"])
    return {"MarketType": "spot", "SpotOptions": opts}


def decide_market(purchase: dict, launch_started_at: float, now: float) -> str:
    """Which market to try now: spot until the capacity window is used up, then on-demand
    when the fallback is allowed; raises when nothing is left to try."""
    if purchase.get("type", "on-demand") != "spot":
        return "on-demand"
    if now - launch_started_at <= int(purchase.get("spot_capacity_window_s", 600)):
        return "spot"
    if purchase.get("spot_fallback_on_demand", True):
        return "on-demand"
    raise RuntimeError(f"no Spot capacity within {purchase.get('spot_capacity_window_s', 600)} s and "
                       "spot_fallback_on_demand is off")


def _launch(run_id: str, tid: str, purchase: dict, market: str) -> dict | None:
    template = _param(f"/amx-gpu-bench/targets/{tid}/launch-template")
    tags = [{"Key": "bench:managed", "Value": "true"}, {"Key": "bench:target", "Value": tid},
            {"Key": "bench:run", "Value": run_id}, {"Key": "Name", "Value": f"amx-gpu-bench-{tid}"},
            {"Key": "project", "Value": "amx-gpu-bench"}, {"Key": "owner", "Value": "operator"},
            {"Key": "purpose", "Value": "cpu-amx-vs-gpu-benchmark"}]
    kw: dict[str, Any] = {"LaunchTemplate": {"LaunchTemplateId": template, "Version": "$Latest"},
                          "SubnetId": common.env("TARGET_SUBNET_ID"), "MinCount": 1, "MaxCount": 1,
                          "TagSpecifications": [{"ResourceType": "instance", "Tags": tags},
                                                {"ResourceType": "volume", "Tags": tags}]}
    mo = market_options(purchase) if market == "spot" else None
    if mo:
        kw["InstanceMarketOptions"] = mo
    try:
        inst = _ec2().run_instances(**kw)["Instances"][0]
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code not in CAPACITY_CODES:
            raise
        _log(run_id, tid, "capacity", {"time": common.now_iso(), "code": code, "market": market,
                                       "message": e.response["Error"].get("Message", "")[:300]})
        return None
    _log(run_id, tid, "targets", {"time": common.now_iso(), "event": "launched", "instance_id": inst["InstanceId"],
                                  "instance_type": inst["InstanceType"], "lifecycle": market,
                                  "requested": purchase.get("type", "on-demand"),
                                  "availability_zone": inst["Placement"]["AvailabilityZone"]})
    return inst


def _terminate(iid: str) -> None:
    try:
        _ec2().terminate_instances(InstanceIds=[iid])
    except ClientError as e:
        if not e.response["Error"]["Code"].startswith(("InvalidInstanceID", "IncorrectInstanceState")):
            raise


def _ssm_online(iid: str, deadline: float) -> bool:
    ssm = common.client("ssm")
    while time.time() < deadline:
        info = ssm.describe_instance_information(Filters=[{"Key": "InstanceIds", "Values": [iid]}])
        if info["InstanceInformationList"] and info["InstanceInformationList"][0]["PingStatus"] == "Online":
            return True
        time.sleep(10)
    return False


def ensure_running(event: dict, context: Any = None) -> dict:
    run_id, tid = event["run_id"], event["target_id"]
    purchase = event.get("purchase") or {}
    deadline = time.time() + CALL_BUDGET_S
    inst = find_instance(tid)
    if inst is not None and inst["State"]["Name"] in ("running", "pending") and lifecycle(inst) != \
            purchase.get("type", "on-demand") and not event.get("keep_lifecycle"):
        # The operator changed the purchase option: replace the instance (models live in S3).
        _log(run_id, tid, "targets", {"time": common.now_iso(), "event": "replaced", "instance_id": inst["InstanceId"],
                                      "lifecycle": lifecycle(inst), "requested": purchase.get("type")})
        _terminate(inst["InstanceId"])
        inst = None
    if inst is None:
        act = _activity(tid)
        started = float(act.get("launch_started_at") or 0) or time.time()
        if not act.get("launch_started_at"):
            _set_activity(tid, launch_started_at=started)
        if time.time() - started > TOTAL_LAUNCH_WINDOW_S:
            _set_activity(tid, launch_started_at=0)
            raise RuntimeError(f"{tid}: no capacity after {TOTAL_LAUNCH_WINDOW_S} s")
        inst = _launch(run_id, tid, purchase, decide_market(purchase, started, time.time()))
        if inst is None:
            return {"ready": False, "reason": "capacity"}
        _set_activity(tid, launch_started_at=0)
    iid = inst["InstanceId"]
    state = inst["State"]["Name"]
    if state == "stopping":
        return {"ready": False, "reason": "stopping"}
    if state == "stopped":
        try:
            _ec2().start_instances(InstanceIds=[iid])
        except ClientError as e:
            code = e.response["Error"]["Code"]
            if code not in CAPACITY_CODES:
                raise
            _log(run_id, tid, "capacity", {"time": common.now_iso(), "code": code, "market": lifecycle(inst),
                                           "message": e.response["Error"].get("Message", "")[:300]})
            return {"ready": False, "reason": "capacity"}
        _log(run_id, tid, "targets", {"time": common.now_iso(), "event": "started", "instance_id": iid,
                                      "instance_type": inst["InstanceType"], "lifecycle": lifecycle(inst)})
    while (inst := _instance(iid)) is not None and inst["State"]["Name"] == "pending" and time.time() < deadline:
        time.sleep(10)
    if inst is None or inst["State"]["Name"] in ("shutting-down", "terminated"):
        return {"ready": False, "reason": "gone"}          # next call launches another one
    if inst["State"]["Name"] != "running":
        return {"ready": False, "reason": inst["State"]["Name"]}
    if not _ssm_online(iid, deadline):
        return {"ready": False, "reason": "ssm"}
    common.touch_target(tid, run_id)
    return {"ready": True, "target_id": tid, "instance_id": iid, "private_ip": inst["PrivateIpAddress"],
            "instance_type": inst["InstanceType"], "lifecycle": lifecycle(inst)}


def configure_start(event: dict, context: Any = None) -> dict:
    cell = event["cell"]
    cid = common.ssm_start(event["instance_id"], [f"{common.bench_home()}/serve.sh start {cell['spec_uri']}"],
                           timeout_s=7200, comment=f"bench serve {cell['cell_key']}"[:100])
    common.set_status(event["run_id"], f"CELL#{cell['cell_id']}", "configuring", command_id=cid)
    return {"command_id": cid}


POLL_BUDGET_S = 720    # keep polling inside one invocation so the run history stays small


def _level_status(run_id: str, level_key: str) -> str | None:
    item = common.table().get_item(Key={"pk": f"RUN#{run_id}", "sk": f"LEVEL#{level_key}"}).get("Item") or {}
    return item.get("status")


def remaining_levels(run_id: str, levels: list[dict]) -> list[dict]:
    """Levels not yet done or given up on (design 8: a relaunched target resumes, it does
    not start the cell over)."""
    return [lvl for lvl in levels if _level_status(run_id, lvl["level_key"]) not in ("done", "failed")]


def _write_levels(run_id: str, cell: dict, levels: list[dict], endpoint: str, api_path: str | None,
                  instance_id: str, target_id: str) -> tuple[str, int]:
    for lvl in levels:
        lvl.update(endpoint=endpoint, target_id=target_id, instance_id=instance_id, cpu=cell["cpu"])
        if api_path:
            lvl["api_path"] = api_path
    key = f"runs/{run_id}/plan/{cell['cell_key']}.levels.json"
    common.put_json(key, levels)
    return key, len(levels)


def configure_poll(event: dict, context: Any = None) -> dict:
    run_id, cell = event["run_id"], event["cell"]
    deadline = time.time() + POLL_BUDGET_S
    while True:
        res = common.ssm_poll(event["instance_id"], event["command_id"])
        if res["status"] not in ("Pending", "InProgress", "Delayed"):
            break
        if time.time() > deadline:
            return {"done": False}
        time.sleep(10)
    if res["status"] != "Success":
        common.set_status(run_id, f"CELL#{cell['cell_id']}", "bringup_failed", stderr=res["stderr"][-2000:])
        raise RuntimeError(f"bring-up of {cell['cell_id']} failed ({res['status']}): {res['stderr'][-500:]}")
    bring = common.get_json(f"runs/{run_id}/bringup/{cell['cell_key']}.json")
    cell_plan = common.get_json(f"runs/{run_id}/plan/{cell['cell_key']}.json")
    qual = cell_plan.get("quality")
    endpoint = f"http://{event['private_ip']}:8000"
    api_path = bring.get("api_path")
    levels = remaining_levels(run_id, cell_plan["levels"])
    common.set_status(run_id, f"CELL#{cell['cell_id']}", "ready", bringup=bring.get("phases", {}),
                      instance_id=event["instance_id"])
    common.touch_target(event["target_id"], run_id)
    # The level list goes to S3 for the levels Map (ItemReader) to keep the state payload small.
    key, n = _write_levels(run_id, cell, levels, endpoint, api_path, event["instance_id"], event["target_id"])
    if qual:
        qual.update(endpoint=endpoint, api_path=api_path or qual["api_path"])
    quality_done = common.exists(f"runs/{run_id}/quality/{cell['cell_key']}/outputs.jsonl.gz")
    return {"done": True, "endpoint": endpoint, "api_path": api_path, "levels_key": key, "n_levels": n,
            "quality": None if quality_done else qual}


def resume_levels(event: dict, context: Any = None) -> dict:
    run_id, cell = event["run_id"], event["cell"]
    cell_plan = common.get_json(f"runs/{run_id}/plan/{cell['cell_key']}.json")
    bring = common.get_json(f"runs/{run_id}/bringup/{cell['cell_key']}.json")
    levels = remaining_levels(run_id, cell_plan["levels"])
    key, n = _write_levels(run_id, cell, levels, f"http://{event['private_ip']}:8000", bring.get("api_path"),
                           event["instance_id"], event["target_id"])
    return {"levels_key": key, "n_levels": n}


def decide_after_failure(instance: dict | None, attempt: int, purchase: dict, stage: str) -> str:
    """relaunch when the instance that served the cell is gone (Spot interruption, stop,
    termination); resume when it is still running and a level failed on its own (that level
    is marked failed and the rest continue); fail when bring-up failed on a healthy instance
    or past the relaunch cap."""
    if attempt >= int(purchase.get("max_relaunches_per_cell", 3)):
        return "fail"
    if instance is None or instance["State"]["Name"] not in ("running", "pending"):
        return "relaunch"
    return "resume" if stage == "levels" else "fail"


def _mark_inflight_failed(run_id: str, cell_key: str, error: Any) -> list[str]:
    """The Levels map runs one level at a time, so the level still 'running' is the one that failed."""
    plan_ = common.get_json(f"runs/{run_id}/plan/{cell_key}.json")
    marked = []
    for lvl in plan_["levels"]:
        if _level_status(run_id, lvl["level_key"]) == "running":
            common.set_status(run_id, f"LEVEL#{lvl['level_key']}", "failed", error=str(error)[:500])
            marked.append(lvl["level_key"])
    return marked


def check(event: dict, context: Any = None) -> dict:
    """After a cell step failed. Idempotent: the state machine calls it again every minute
    while the replacement instance is pending; `recovery` carries the attempt count."""
    run_id, tid, iid, cell = event["run_id"], event["target_id"], event["instance_id"], event["cell"]
    purchase = event.get("purchase") or {}
    rec = dict(event.get("recovery") or {})
    attempt = int(rec.get("attempt", 0))
    inst = _instance(iid)
    decision = decide_after_failure(inst, attempt, purchase, event.get("stage", "levels"))
    out = {"decision": decision, "ready": True, "instance_id": iid, "private_ip": event.get("private_ip"),
           "recovery": {"attempt": attempt, "interruption_logged": bool(rec.get("interruption_logged"))}}
    if decision == "fail":
        common.set_status(run_id, f"CELL#{cell['cell_id']}", "failed", reason=f"{event.get('stage')} failed "
                          f"after {attempt} recoveries", error=str(event.get("error"))[:500])
        return out
    if decision == "resume":
        out["marked_failed"] = _mark_inflight_failed(run_id, cell["cell_key"], event.get("error"))
        out["recovery"]["attempt"] = attempt + 1
        return out
    if not rec.get("interruption_logged"):
        reason = (inst or {}).get("StateReason") or {"Code": "NotFound"}
        _log(run_id, tid, "targets", {"time": common.now_iso(), "event": "interrupted", "instance_id": iid,
                                      "state": ((inst or {}).get("State") or {}).get("Name", "terminated"),
                                      "state_reason": reason.get("Code"), "message": reason.get("Message", "")[:200],
                                      "cell": cell["cell_id"]})
        common.set_status(run_id, f"CELL#{cell['cell_id']}", "interrupted", instance_id=iid)
        out["recovery"]["interruption_logged"] = True
    if inst is not None and inst["State"]["Name"] not in ("shutting-down", "terminated"):
        _terminate(iid)
    repl = ensure_running({"run_id": run_id, "target_id": tid, "purchase": purchase}, context)
    out["ready"] = repl["ready"]
    if repl["ready"]:
        out.update(instance_id=repl["instance_id"], private_ip=repl["private_ip"], lifecycle=repl["lifecycle"])
        out["recovery"] = {"attempt": attempt + 1, "interruption_logged": False}
    return out


def stop(event: dict, context: Any = None) -> dict:
    tid = event["target_id"]
    inst = find_instance(tid)
    common.table().update_item(Key={"pk": f"TARGET#{tid}", "sk": "ACTIVITY"}, UpdateExpression="REMOVE active_run_id")
    if inst is None:
        return {"stopped": None}
    if lifecycle(inst) == "spot":
        _terminate(inst["InstanceId"])
        return {"terminated": inst["InstanceId"]}
    if inst["State"]["Name"] in ("running", "pending"):
        _ec2().stop_instances(InstanceIds=[inst["InstanceId"]])
    return {"stopped": inst["InstanceId"]}


ACTIONS = {"ensure_running": ensure_running, "configure_start": configure_start, "configure_poll": configure_poll,
           "check": check, "resume_levels": resume_levels, "stop": stop}


def handler(event: dict, context: Any = None) -> dict:
    return ACTIONS[event["action"]](event, context)

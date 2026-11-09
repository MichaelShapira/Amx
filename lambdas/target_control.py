"""Target control (design 8): start instances, configure cells, stop instances.

Actions (event["action"]):
  ensure_running  start the target if needed and wait for SSM; capacity errors are
                  appended to runs/<run>/capacity/<target>.json and retried for 10 minutes
  configure_start send `serve.sh start <spec>` through SSM; returns the command id
  configure_poll  {done, ok}; on success returns endpoint, api_path and the cell's level list
  stop            stop the instance (the IAM policy only allows tagged bench instances)
Bring-up can take longer than one Lambda invocation (model download, OpenVINO export), so
the state machine polls configure_poll on a Wait loop.
"""

from __future__ import annotations

import time
from typing import Any

from botocore.exceptions import ClientError

from lambdas import common

CAPACITY_CODES = {"InsufficientInstanceCapacity", "InstanceLimitExceeded", "Unsupported",
                  "InsufficientCapacityOnHost", "MaxSpotInstanceCountExceeded"}
CAPACITY_RETRY_S = 600


def _instance(iid: str) -> dict:
    r = common.client("ec2").describe_instances(InstanceIds=[iid])
    return r["Reservations"][0]["Instances"][0]


def _log_capacity(run_id: str, target_id: str, entry: dict) -> None:
    key = f"runs/{run_id}/capacity/{target_id}.json"
    try:
        events = common.get_json(key)
    except ClientError:
        events = []
    events.append(entry)
    common.put_json(key, events)


def ensure_running(event: dict, context: Any = None) -> dict:
    run_id, tid, iid = event["run_id"], event["target_id"], event["instance_id"]
    ec2 = common.client("ec2")
    inst = _instance(iid)
    if inst.get("Tags") and {"Key": "bench:managed", "Value": "true"} not in inst["Tags"]:
        raise RuntimeError(f"{iid} is not tagged bench:managed=true")
    deadline = time.time() + CAPACITY_RETRY_S
    while inst["State"]["Name"] not in ("running", "pending"):
        if inst["State"]["Name"] == "stopping":
            ec2.get_waiter("instance_stopped").wait(InstanceIds=[iid])
        try:
            ec2.start_instances(InstanceIds=[iid])
            break
        except ClientError as e:
            code = e.response["Error"]["Code"]
            if code not in CAPACITY_CODES:
                raise
            _log_capacity(run_id, tid, {"time": common.now_iso(), "code": code,
                                        "message": e.response["Error"].get("Message", ""),
                                        "instance_type": inst["InstanceType"],
                                        "availability_zone": inst["Placement"]["AvailabilityZone"]})
            if time.time() > deadline:
                raise RuntimeError(f"{tid}: no capacity for {inst['InstanceType']} after {CAPACITY_RETRY_S}s") from e
            time.sleep(30)
            inst = _instance(iid)
    ec2.get_waiter("instance_running").wait(InstanceIds=[iid], WaiterConfig={"Delay": 10, "MaxAttempts": 60})
    ssm = common.client("ssm")
    for _ in range(60):
        info = ssm.describe_instance_information(Filters=[{"Key": "InstanceIds", "Values": [iid]}])
        if info["InstanceInformationList"] and info["InstanceInformationList"][0]["PingStatus"] == "Online":
            break
        time.sleep(10)
    else:
        raise RuntimeError(f"{iid} is running but SSM is not online")
    common.touch_target(tid, run_id)
    inst = _instance(iid)
    return {"target_id": tid, "instance_id": iid, "private_ip": inst["PrivateIpAddress"],
            "instance_type": inst["InstanceType"]}


def configure_start(event: dict, context: Any = None) -> dict:
    cell = event["cell"]
    cid = common.ssm_start(event["instance_id"], [f"{common.bench_home()}/serve.sh start {cell['spec_uri']}"],
                           timeout_s=7200, comment=f"bench serve {cell['cell_key']}"[:100])
    common.set_status(event["run_id"], f"CELL#{cell['cell_id']}", "configuring", command_id=cid)
    return {"command_id": cid}


POLL_BUDGET_S = 720    # keep polling inside one invocation so the run history stays small


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
    levels = cell_plan["levels"]
    qual = cell_plan.get("quality")
    endpoint = f"http://{event['private_ip']}:8000"
    api_path = bring.get("api_path")
    for lvl in levels:
        lvl.update(endpoint=endpoint, target_id=event["target_id"], instance_id=event["instance_id"],
                   cpu=cell["cpu"])
        if api_path:
            lvl["api_path"] = api_path
    common.set_status(run_id, f"CELL#{cell['cell_id']}", "ready", bringup=bring.get("phases", {}))
    common.touch_target(event["target_id"], run_id)
    # The level list goes to S3 for the levels Map (ItemReader) to keep the state payload small.
    key = f"runs/{run_id}/plan/{cell['cell_key']}.levels.json"
    common.put_json(key, levels)
    if qual:
        qual.update(endpoint=endpoint, api_path=api_path or qual["api_path"])
    return {"done": True, "endpoint": endpoint, "api_path": api_path, "levels_key": key, "n_levels": len(levels),
            "quality": qual}


def stop(event: dict, context: Any = None) -> dict:
    iid = event["instance_id"]
    common.client("ec2").stop_instances(InstanceIds=[iid])
    common.table().update_item(Key={"pk": f"TARGET#{event['target_id']}", "sk": "ACTIVITY"},
                               UpdateExpression="REMOVE active_run_id")
    return {"stopped": iid}


ACTIONS = {"ensure_running": ensure_running, "configure_start": configure_start,
           "configure_poll": configure_poll, "stop": stop}


def handler(event: dict, context: Any = None) -> dict:
    return ACTIONS[event["action"]](event, context)

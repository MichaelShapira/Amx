"""A/B check of this harness's vLLM CPU configuration against Intel's published Xeon recipe,
on the same EC2 instance, with vLLM's own `vllm bench serve` as the client (the command Intel's
Software Catalog lists next to its numbers).

    .venv/bin/python tools/recipe_check.py --instance-type m8i.4xlarge [--model Qwen/Qwen3-4B]
    .venv/bin/python tools/recipe_check.py --terminate <instance-id>

Launches one On-Demand instance from the t2-m8i-4xl launch template (tagged bench:managed=false,
so the reaper leaves it alone and YOU terminate it), waits for SSM, runs three server
configurations one after another and the same client sweep against each:

  intel_v023    Intel's recipe verbatim: public.ecr.aws/q9t5s3a7/vllm-cpu-release-repo:v0.23.0,
                privileged, VLLM_CPU_SGL_KERNEL=1, VLLM_CPU_KVCACHE_SPACE=40, --distributed-executor-backend
                mp, --enable-chunked-prefill, --max-num-seqs 256, --block-size 128
  ours_v033     what bench/cellspec.py starts for the t2 target: vllm/vllm-openai-cpu v0.33.0 pinned by digest,
                SYS_NICE + seccomp profile, VLLM_CPU_OMP_THREADS_BIND=auto, VLLM_CPU_NUM_OF_RESERVED_CPU=1,
                --kv-cache-memory-bytes 32 GiB, --max-num-seqs 128, --no-enable-prefix-caching
  intel_v033    Intel's flags on the v0.33.0 image (separates "image version" from "flags")

Client: random 128-token prompts, 128 output tokens, ignore_eos, request rate inf, at concurrency
1, 16, 64 and 128 (Intel publishes 128/128 at the highest concurrency that keeps TPOT <= 100 ms).
Results (vllm bench serve JSON, server logs, lscpu) go to s3://<bucket>/runs/recipe-check-<ts>/hosts/
and are summarized on stdout.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import boto3

ROOT = Path(__file__).resolve().parents[1]
REGION = "us-east-1"
OURS_IMAGE = "docker.io/vllm/vllm-openai-cpu@sha256:4154467a886f8b7ffa9dbab32f873f5ca4528409d042a6ec866e8888223263d3"
INTEL_IMAGE = "public.ecr.aws/q9t5s3a7/vllm-cpu-release-repo:v0.23.0"
CONCURRENCY = ((1, 8), (16, 64), (64, 192), (128, 256))      # (max concurrency, prompts)

REMOTE = r"""#!/bin/bash
set -uo pipefail
MODELS="__MODELS__"; CONFIGS="__CONFIGS__"; BUCKET="__BUCKET__"; PREFIX="__PREFIX__"; KV_OURS_BYTES=__KV_OURS__
OURS="__OURS_IMAGE__"; INTEL="__INTEL_IMAGE__"
HF=/opt/models/hfcache; OUT=/var/lib/bench/recipe-check; mkdir -p "$HF" "$OUT"
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
lscpu > "$OUT/lscpu.txt"; nproc > "$OUT/nproc.txt"; free -g > "$OUT/free.txt"
log "pulling images"; docker pull -q "$OURS" >/dev/null; docker pull -q "$INTEL" >/dev/null
for MODEL in $MODELS; do
  log "downloading $MODEL"
  docker run --rm -v "$HF:/root/.cache/huggingface" --entrypoint python3 "$OURS" -c \
    "from huggingface_hub import snapshot_download; print(snapshot_download('$MODEL'))" >>"$OUT/download.txt" 2>&1 || { cat "$OUT/download.txt"; exit 1; }
done
COMMON_ENV=(-e VLLM_RPC_TIMEOUT=100000 -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 -e VLLM_ENGINE_ITERATION_TIMEOUT_S=120)
start_server() {   # name image <docker opts...> -- <cmd...>
  local name=$1; shift; local image=$1; shift; local opts=() ; while [[ "$1" != "--" ]]; do opts+=("$1"); shift; done; shift
  docker rm -f bench-engine >/dev/null 2>&1 || true
  docker run -d --name bench-engine --network host -v "$HF:/root/.cache/huggingface" "${opts[@]}" "$image" "$@" >/dev/null
  local t0=$(date +%s)
  until curl -sf -o /dev/null http://127.0.0.1:8000/v1/models; do
    if ! docker ps --format '{{.Names}}' | grep -qx bench-engine; then log "$name: container exited"; docker logs bench-engine > "$OUT/$name.server.log" 2>&1; return 1; fi
    (( $(date +%s) - t0 < 1500 )) || { log "$name: not healthy after 1500s"; docker logs bench-engine > "$OUT/$name.server.log" 2>&1; return 1; }
    sleep 3
  done
  log "$name: healthy after $(( $(date +%s) - t0 ))s"
  # one real request so the first bench point is not the warm-up
  curl -s -o /dev/null -H 'Content-Type: application/json' -d "{\"model\":\"$MODEL\",\"prompt\":\"Hello\",\"max_tokens\":8}" http://127.0.0.1:8000/v1/completions
  return 0
}
bench() {   # name conc prompts
  local name=$1 conc=$2 n=$3
  docker run --rm --network host -v "$HF:/root/.cache/huggingface" -v "$OUT:/results" --entrypoint vllm "$OURS" bench serve \
    --backend vllm --host 127.0.0.1 --port 8000 --model "$MODEL" --dataset-name random --random-input-len 128 --random-output-len 128 \
    --ignore-eos --request-rate inf --max-concurrency "$conc" --num-prompts "$n" --percentile-metrics ttft,tpot,itl,e2el \
    --save-result --result-dir /results --result-filename "$name.c$conc.json" >"$OUT/$name.c$conc.client.log" 2>&1
  python3 - "$OUT/$name.c$conc.json" <<'EOF'
import json,sys
try:
    d=json.load(open(sys.argv[1])); print(f"  conc={d.get('max_concurrency')} out_tok/s={d.get('output_throughput'):.1f} req/s={d.get('request_throughput'):.2f} ttft_p50={d.get('median_ttft_ms'):.0f} tpot_p50={d.get('median_tpot_ms'):.1f} e2e_p99={d.get('p99_e2el_ms'):.0f} completed={d.get('completed')}")
except Exception as e: print("  no result:", e)
EOF
}
run_config() {   # name image opts... -- cmd...
  local name=$1
  log "=== $name"
  if start_server "$@"; then
    for pair in "1 8" "16 64" "64 192" "128 256"; do set -- $pair; bench "$name" "$1" "$2"; done
    docker logs bench-engine > "$OUT/$name.server.log" 2>&1
  fi
  docker rm -f bench-engine >/dev/null 2>&1 || true
}
for MODEL in $MODELS; do
TAG=$(echo "$MODEL" | tr '/' '_' | tr 'A-Z' 'a-z')
[[ " $CONFIGS " == *" intel_v023 "* ]] && run_config "intel_v023.$TAG" "$INTEL" --privileged --shm-size 4g "${COMMON_ENV[@]}" -e VLLM_CPU_SGL_KERNEL=1 -e VLLM_CPU_KVCACHE_SPACE=40 --entrypoint /bin/bash -- -lc \
  "python -m vllm.entrypoints.openai.api_server --model $MODEL --dtype bfloat16 --distributed-executor-backend mp --trust-remote-code --disable-log-stats --tensor-parallel-size 1 --block-size 128 --enable-chunked-prefill --max-num-batched-tokens 2048 --max-num-seqs 256"
[[ " $CONFIGS " == *" ours_v033 "* ]] && run_config "ours_v033.$TAG" "$OURS" --shm-size 4g --cap-add SYS_NICE --security-opt seccomp=/opt/bench/seccomp-vllm.json -e VLLM_CPU_OMP_THREADS_BIND=auto -e VLLM_CPU_NUM_OF_RESERVED_CPU=1 --entrypoint vllm -- \
  serve "$MODEL" --served-model-name "$MODEL" --dtype bfloat16 --max-model-len 16384 --max-num-seqs 128 --max-num-batched-tokens 2048 --no-enable-prefix-caching --seed 0 --host 0.0.0.0 --port 8000 --kv-cache-memory-bytes "$KV_OURS_BYTES" --block-size 128
[[ " $CONFIGS " == *" intel_v033 "* ]] && run_config "intel_v033.$TAG" "$OURS" --privileged --shm-size 4g "${COMMON_ENV[@]}" -e VLLM_CPU_SGL_KERNEL=1 -e VLLM_CPU_KVCACHE_SPACE=40 --entrypoint vllm -- \
  serve "$MODEL" --dtype bfloat16 --distributed-executor-backend mp --trust-remote-code --disable-log-stats --tensor-parallel-size 1 --block-size 128 --enable-chunked-prefill --max-num-batched-tokens 2048 --max-num-seqs 256 --host 0.0.0.0 --port 8000
done
log "uploading"; aws s3 cp --only-show-errors --recursive "$OUT" "s3://$BUCKET/$PREFIX/" && log "done: s3://$BUCKET/$PREFIX/"
"""


def launch(ec2, instance_type: str) -> str:
    ssm = boto3.client("ssm", region_name=REGION)
    template = ssm.get_parameter(Name="/amx-gpu-bench/targets/t2-m8i-4xl/launch-template")["Parameter"]["Value"]
    tags = [{"Key": "bench:managed", "Value": "false"}, {"Key": "bench:target", "Value": "recipe-check"},
            {"Key": "bench:run", "Value": "recipe-check"}, {"Key": "Name", "Value": "amx-gpu-bench-recipe-check"},
            {"Key": "project", "Value": "amx-gpu-bench"}, {"Key": "owner", "Value": "operator"},
            {"Key": "purpose", "Value": "cpu-amx-vs-gpu-benchmark"}]
    r = ec2.run_instances(LaunchTemplate={"LaunchTemplateId": template, "Version": "$Latest"}, MinCount=1, MaxCount=1,
                          InstanceType=instance_type,
                          TagSpecifications=[{"ResourceType": "instance", "Tags": tags}, {"ResourceType": "volume", "Tags": tags}])
    iid = r["Instances"][0]["InstanceId"]
    ec2.create_tags(Resources=[iid], Tags=[{"Key": "bench:managed", "Value": "false"}])    # the template says true; we win
    return iid


def wait_ssm(iid: str, timeout_s: int = 600) -> None:
    ssm = boto3.client("ssm", region_name=REGION)
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        info = ssm.describe_instance_information(Filters=[{"Key": "InstanceIds", "Values": [iid]}])["InstanceInformationList"]
        if info and info[0]["PingStatus"] == "Online":
            return
        time.sleep(10)
    raise SystemExit(f"{iid} did not register with SSM in {timeout_s}s")


def wait_boot(iid: str) -> None:
    """bench-boot.service syncs /opt/bench (the seccomp profile) from the bucket; wait for it."""
    ssm = boto3.client("ssm", region_name=REGION)
    for _ in range(60):
        r = ssm.send_command(InstanceIds=[iid], DocumentName="AWS-RunShellScript",
                             Parameters={"commands": ["test -f /opt/bench/seccomp-vllm.json && systemctl is-active docker"]})
        cid = r["Command"]["CommandId"]
        time.sleep(5)
        for _ in range(12):
            inv = ssm.get_command_invocation(CommandId=cid, InstanceId=iid)
            if inv["Status"] in ("Success", "Failed", "TimedOut", "Cancelled"):
                break
            time.sleep(3)
        if inv["Status"] == "Success":
            return
        time.sleep(10)
    raise SystemExit("host bootstrap did not finish")


def run_remote(iid: str, script: str, bucket: str, prefix: str) -> str:
    ssm = boto3.client("ssm", region_name=REGION)
    r = ssm.send_command(InstanceIds=[iid], DocumentName="AWS-RunShellScript", TimeoutSeconds=7200,
                         Parameters={"commands": [script], "executionTimeout": ["7000"]}, Comment="recipe check")
    cid = r["Command"]["CommandId"]
    print(f"command {cid} on {iid}; this takes 30-45 minutes", flush=True)
    while True:
        time.sleep(30)
        inv = ssm.get_command_invocation(CommandId=cid, InstanceId=iid)
        if inv["Status"] in ("Success", "Failed", "TimedOut", "Cancelled"):
            print(inv["StandardOutputContent"][-6000:])
            if inv["StandardErrorContent"].strip():
                print("STDERR:", inv["StandardErrorContent"][-3000:])
            return inv["Status"]


def summarize(bucket: str, prefix: str) -> None:
    s3 = boto3.client("s3", region_name=REGION)
    keys = [o["Key"] for o in s3.list_objects_v2(Bucket=bucket, Prefix=prefix + "/").get("Contents", [])]
    rows = []
    for k in sorted(keys):
        if k.endswith(".json") and ".c" in k:
            d = json.loads(s3.get_object(Bucket=bucket, Key=k)["Body"].read())
            name = k.rsplit("/", 1)[1].rsplit(".c", 1)[0]
            rows.append((name, d.get("max_concurrency"), d.get("output_throughput"), d.get("request_throughput"),
                         d.get("median_ttft_ms"), d.get("median_tpot_ms"), d.get("p99_e2el_ms"), d.get("completed")))
    print(f"\n{'config':34} {'conc':>5} {'out tok/s':>10} {'req/s':>7} {'ttft p50':>9} {'tpot p50':>9} {'e2e p99':>9} {'n':>4}")
    for r in sorted(rows, key=lambda x: (x[0], x[1] or 0)):
        print(f"{r[0]:34} {r[1]!s:>5} {r[2] or 0:10.1f} {r[3] or 0:7.2f} {r[4] or 0:9.0f} {r[5] or 0:9.1f} {r[6] or 0:9.0f} {r[7]!s:>4}")
    print(f"\nartifacts: s3://{bucket}/{prefix}/ (server logs: <config>.server.log)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance-type", default="m8i.4xlarge")
    ap.add_argument("--models", default="Qwen/Qwen3-4B", help="space-separated Hugging Face ids")
    ap.add_argument("--configs", default="intel_v023 ours_v033 intel_v033", help="space-separated subset")
    ap.add_argument("--kv-ours-gib", type=int, default=32, help="what cellspec gives the t2 target on this size")
    ap.add_argument("--bucket", default="amx-gpu-bench-346399954218-us-east-1")
    ap.add_argument("--instance-id", default=None, help="reuse a running instance instead of launching")
    ap.add_argument("--keep", action="store_true", help="do not terminate the instance at the end")
    ap.add_argument("--terminate", metavar="INSTANCE_ID", default=None)
    ap.add_argument("--summarize", metavar="PREFIX", default=None)
    args = ap.parse_args(argv)
    ec2 = boto3.client("ec2", region_name=REGION)
    if args.terminate:
        ec2.terminate_instances(InstanceIds=[args.terminate])
        print("terminating", args.terminate)
        return 0
    if args.summarize:
        summarize(args.bucket, args.summarize)
        return 0
    prefix = f"runs/recipe-check-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}/hosts/{args.instance_type}"
    iid = args.instance_id or launch(ec2, args.instance_type)
    print("instance", iid, "prefix", prefix, flush=True)
    try:
        wait_ssm(iid)
        wait_boot(iid)
        script = (REMOTE.replace("__MODELS__", args.models).replace("__CONFIGS__", args.configs)
                  .replace("__BUCKET__", args.bucket).replace("__PREFIX__", prefix)
                  .replace("__KV_OURS__", str(args.kv_ours_gib * 2**30)).replace("__OURS_IMAGE__", OURS_IMAGE)
                  .replace("__INTEL_IMAGE__", INTEL_IMAGE))
        status = run_remote(iid, script, args.bucket, prefix)
        print("status", status)
        summarize(args.bucket, prefix)
    finally:
        if not args.keep and not args.instance_id:
            ec2.terminate_instances(InstanceIds=[iid])
            print("terminated", iid)
    return 0


if __name__ == "__main__":
    sys.exit(main())

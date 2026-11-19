"""A/B of `VLLM_CPU_ATTN_SPLIT_KV=0` for Gemma 4 E4B on vLLM CPU, on one m8i instance.

Intel's Xeon recipe sets `VLLM_CPU_ATTN_SPLIT_KV=0` for Gemma 4 (vLLM's own CPU serving benchmarks set it for
Gemma 4 26B). The harness runs Gemma 4 with the default (split-KV attention on). This runs the harness's exact
server command twice on the same host, default and with the variable set to 0, and the same `vllm bench serve`
sweep against each:

  128/128 at concurrency 1, 16 and 64          what the Intel use case measures
  2048/128 at concurrency 16                   long inputs, where the attention kernel choice can matter

    .venv/bin/python tools/gemma4_splitkv_ab.py [--instance-type m8i.4xlarge] [--kv-gib 25]
    .venv/bin/python tools/gemma4_splitkv_ab.py --summarize runs/gemma4-splitkv-ab-<ts>/hosts/m8i.4xlarge
    .venv/bin/python tools/gemma4_splitkv_ab.py --terminate <instance-id>

Weights come from the bucket's staged copy (models/hf/<repo>/<rev>). Results go to
s3://<bucket>/runs/gemma4-splitkv-ab-<ts>/hosts/<type>/ and are summarized on stdout. The instance is tagged
bench:managed=false (the reaper leaves it alone) and terminated at the end; the remote script also shuts the host
down when it finishes, so an expired operator session cannot leave it running for long (`--keep` disables both).
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import boto3

from tools.gemma4_cpu_probe import REPO, REV
from tools.recipe_check import OURS_IMAGE, REGION, launch, run_remote, wait_boot, wait_ssm

CONFIGS = {"default": "", "splitkv0": "-e VLLM_CPU_ATTN_SPLIT_KV=0"}
SWEEP = (("128", "128", "1", "8"), ("128", "128", "16", "64"), ("128", "128", "64", "192"), ("2048", "128", "16", "48"))

REMOTE = r"""#!/bin/bash
set -uo pipefail
BUCKET="__BUCKET__"; PREFIX="__PREFIX__"; KV_BYTES=__KV__; OURS="__OURS__"; REPO="__REPO__"; REV="__REV__"; SHUTDOWN=__SHUTDOWN__
OUT=/var/lib/bench/gemma4-splitkv-ab; mkdir -p "$OUT" /opt/models/hf
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
finish() { aws s3 cp --only-show-errors --recursive "$OUT" "s3://$BUCKET/$PREFIX/" && log "done: s3://$BUCKET/$PREFIX/"; [[ $SHUTDOWN == 1 ]] && { log "shutting down"; shutdown -h +1; }; }
trap finish EXIT
lscpu > "$OUT/lscpu.txt"; free -g > "$OUT/free.txt"
log "pull"; docker pull -q "$OURS" >/dev/null
log "stage weights"; aws s3 sync --only-show-errors "s3://$BUCKET/models/hf/$REPO/$REV" "/opt/models/hf/$REPO/$REV"
M="/models/hf/$REPO/$REV"
start_server() {   # name extra-env...
  local name=$1; shift
  docker rm -f bench-engine >/dev/null 2>&1 || true
  docker run -d --name bench-engine --network host -v /opt/models:/models:ro --shm-size 4g --cap-add SYS_NICE \
    --security-opt seccomp=/opt/bench/seccomp-vllm.json -e VLLM_CPU_OMP_THREADS_BIND=auto -e VLLM_CPU_NUM_OF_RESERVED_CPU=1 "$@" \
    --entrypoint vllm "$OURS" serve "$M" --served-model-name m6-gemma-4-e4b --dtype bfloat16 --max-model-len 16384 \
    --max-num-seqs 256 --max-num-batched-tokens 2048 --no-enable-prefix-caching --seed 0 --host 0.0.0.0 --port 8000 \
    --kv-cache-memory-bytes "$KV_BYTES" --block-size 128 --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 \
    --limit-mm-per-prompt '{"image": 0, "audio": 0}' >/dev/null
  local t0=$(date +%s)
  until curl -sf -o /dev/null http://127.0.0.1:8000/health; do
    if ! docker ps --format '{{.Names}}' | grep -qx bench-engine; then log "$name: container exited"; docker logs bench-engine > "$OUT/$name.server.log" 2>&1; return 1; fi
    (( $(date +%s) - t0 < 2400 )) || { log "$name: not healthy after 2400s"; docker logs bench-engine > "$OUT/$name.server.log" 2>&1; return 1; }
    sleep 5
  done
  log "$name: healthy after $(( $(date +%s) - t0 ))s"
  curl -s -o /dev/null -H 'Content-Type: application/json' -d '{"model":"m6-gemma-4-e4b","prompt":"Hello","max_tokens":8}' http://127.0.0.1:8000/v1/completions
  docker exec bench-engine env | grep -E '^VLLM_' | sort > "$OUT/$name.env.txt"
}
bench() {   # name in out conc prompts
  local name=$1 inlen=$2 outlen=$3 conc=$4 n=$5 tag="$1.i$2.c$4"
  docker run --rm --network host -v "$OUT:/results" -v /opt/models:/models:ro --entrypoint vllm "$OURS" bench serve --backend vllm \
    --host 127.0.0.1 --port 8000 --model "$M" --tokenizer "$M" --served-model-name m6-gemma-4-e4b --dataset-name random \
    --random-input-len "$inlen" --random-output-len "$outlen" --ignore-eos --request-rate inf --max-concurrency "$conc" --num-prompts "$n" \
    --percentile-metrics ttft,tpot,itl,e2el --save-result --result-dir /results --result-filename "$tag.json" > "$OUT/$tag.client.log" 2>&1
  python3 - "$OUT/$tag.json" <<'EOF'
import json,sys
try:
    d=json.load(open(sys.argv[1])); print(f"  in={d.get('random_input_len') or '?'} conc={d.get('max_concurrency')} out_tok/s={d.get('output_throughput'):.1f} ttft_p50={d.get('median_ttft_ms'):.0f} tpot_p50={d.get('median_tpot_ms'):.1f} e2e_p99={d.get('p99_e2el_ms'):.0f} completed={d.get('completed')}")
except Exception as e: print("  no result:", e)
EOF
}
for CFG in __CONFIG_ORDER__; do
  case "$CFG" in __CONFIG_CASES__ esac
  log "=== $CFG ($EXTRA)"
  if start_server "$CFG" $EXTRA; then
    __SWEEP__
    docker logs bench-engine > "$OUT/$CFG.server.log" 2>&1
    grep -iE 'split|attn|attention backend' "$OUT/$CFG.server.log" | head -5
  fi
  docker rm -f bench-engine >/dev/null 2>&1 || true
done
"""


def render(bucket: str, prefix: str, kv_gib: int, shutdown: bool) -> str:
    cases = " ".join(f'{name}) EXTRA="{env}";;' for name, env in CONFIGS.items())
    sweep = "\n    ".join(f'bench "$CFG" {i} {o} {c} {n}' for i, o, c, n in SWEEP)
    return (REMOTE.replace("__BUCKET__", bucket).replace("__PREFIX__", prefix).replace("__KV__", str(kv_gib * 2**30))
            .replace("__OURS__", OURS_IMAGE).replace("__REPO__", REPO).replace("__REV__", REV)
            .replace("__SHUTDOWN__", "1" if shutdown else "0").replace("__CONFIG_ORDER__", " ".join(CONFIGS))
            .replace("__CONFIG_CASES__", cases).replace("__SWEEP__", sweep))


def summarize(bucket: str, prefix: str) -> list[dict]:
    s3 = boto3.client("s3", region_name=REGION)
    keys = [o["Key"] for o in s3.list_objects_v2(Bucket=bucket, Prefix=prefix + "/").get("Contents", [])]
    rows = []
    for k in sorted(keys):
        name = k.rsplit("/", 1)[1]
        if not (name.endswith(".json") and ".i" in name and ".c" in name):
            continue
        d = json.loads(s3.get_object(Bucket=bucket, Key=k)["Body"].read())
        cfg, rest = name.split(".i", 1)
        inlen = int(rest.split(".c", 1)[0])
        rows.append({"config": cfg, "input": inlen, "conc": d.get("max_concurrency"), "out_tok_s": d.get("output_throughput"),
                     "ttft_p50": d.get("median_ttft_ms"), "tpot_p50": d.get("median_tpot_ms"), "e2e_p99": d.get("p99_e2el_ms"),
                     "completed": d.get("completed")})
    print(f"\n{'config':10} {'in':>5} {'conc':>5} {'out tok/s':>10} {'ttft p50':>9} {'tpot p50':>9} {'e2e p99':>9} {'n':>4}")
    for r in sorted(rows, key=lambda x: (x["input"], x["conc"] or 0, x["config"])):
        print(f"{r['config']:10} {r['input']:>5} {r['conc']!s:>5} {r['out_tok_s'] or 0:10.1f} {r['ttft_p50'] or 0:9.0f} "
              f"{r['tpot_p50'] or 0:9.1f} {r['e2e_p99'] or 0:9.0f} {r['completed']!s:>4}")
    by = {(r["input"], r["conc"], r["config"]): r for r in rows}
    print("\nsplitkv0 vs default (out tok/s):")
    for (inp, conc, cfg), r in sorted(by.items()):
        if cfg == "default" and (inp, conc, "splitkv0") in by and r["out_tok_s"]:
            o = by[(inp, conc, "splitkv0")]
            print(f"  in={inp:<5} conc={conc!s:<4} {100 * (o['out_tok_s'] / r['out_tok_s'] - 1):+.1f}%  (tpot {r['tpot_p50']:.1f} -> {o['tpot_p50']:.1f} ms)")
    print(f"\nartifacts: s3://{bucket}/{prefix}/ (server logs: <config>.server.log, env: <config>.env.txt)")
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance-type", default="m8i.4xlarge")
    ap.add_argument("--kv-gib", type=int, default=25, help="what cellspec gives Gemma 4 E4B on a 64 GiB host")
    ap.add_argument("--bucket", default="amx-gpu-bench-346399954218-us-east-1")
    ap.add_argument("--instance-id", default=None, help="reuse a running instance instead of launching")
    ap.add_argument("--keep", action="store_true", help="neither terminate nor shut the instance down")
    ap.add_argument("--terminate", metavar="INSTANCE_ID", default=None)
    ap.add_argument("--summarize", metavar="PREFIX", default=None)
    ap.add_argument("--print-script", action="store_true", help="print the remote script and exit")
    args = ap.parse_args(argv)
    prefix = f"runs/gemma4-splitkv-ab-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}/hosts/{args.instance_type}"
    if args.print_script:
        print(render(args.bucket, prefix, args.kv_gib, not args.keep))
        return 0
    ec2 = boto3.client("ec2", region_name=REGION)
    if args.terminate:
        ec2.terminate_instances(InstanceIds=[args.terminate])
        print("terminating", args.terminate)
        return 0
    if args.summarize:
        summarize(args.bucket, args.summarize)
        return 0
    iid = args.instance_id or launch(ec2, args.instance_type, shutdown_behavior=None if args.keep else "terminate")
    print("instance", iid, "prefix", prefix, flush=True)
    try:
        wait_ssm(iid)
        wait_boot(iid)
        print("status", run_remote(iid, render(args.bucket, prefix, args.kv_gib, not args.keep), args.bucket, prefix))
        summarize(args.bucket, prefix)
    finally:
        if not args.keep:
            ec2.terminate_instances(InstanceIds=[iid])
            print("terminated", iid)
    return 0


if __name__ == "__main__":
    sys.exit(main())

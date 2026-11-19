"""Bring Gemma 4 E4B up on vLLM CPU the way the harness does, on one m8i instance, and watch the
host's memory while it loads: the first live run (translation-smoke-m6-20261119T081428Z) hung the
64 GiB host (SSM and the CloudWatch agent went silent five minutes after the container started) with
the harness's 32 GiB KV-cache budget, and nothing recorded the peak.

    .venv/bin/python tools/gemma4_cpu_probe.py [--instance-type m8i.4xlarge] [--kv-gib 16]

Weights come from the bucket's staged copy (models/hf/<repo>/<rev>, what stage_model.sh keeps).
Writes mem.log (free -m every 5 s), the server log and a short `vllm bench serve` point to
s3://<bucket>/runs/gemma4-cpu-probe-<ts>/hosts/<type>/. Terminates the instance at the end.
"""

from __future__ import annotations

import argparse
import sys
import time

import boto3

from tools.recipe_check import OURS_IMAGE, REGION, launch, run_remote, wait_boot, wait_ssm

REPO = "google/gemma-4-E4B-it"
REV = "4ecc2ba63bcac23ba4a767fc87025b1450f84c9c"

REMOTE = r"""#!/bin/bash
set -uo pipefail
BUCKET="__BUCKET__"; PREFIX="__PREFIX__"; KV_BYTES=__KV__; OURS="__OURS__"; REPO="__REPO__"; REV="__REV__"
OUT=/var/lib/bench/gemma4-probe; mkdir -p "$OUT" /opt/models/hf
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
free -g > "$OUT/free-before.txt"
log "pull"; docker pull -q "$OURS" >/dev/null
log "stage weights from the bucket"; aws s3 sync --only-show-errors "s3://$BUCKET/models/hf/$REPO/$REV" "/opt/models/hf/$REPO/$REV"
du -sh "/opt/models/hf/$REPO/$REV" | tee "$OUT/weights-size.txt"
( while true; do echo "$(date -u +%H:%M:%S) $(free -m | awk 'NR==2{print "used="$3" free="$4" buff/cache="$6" avail="$7}') $(docker stats --no-stream --format '{{.MemUsage}}' bench-engine 2>/dev/null)"; sleep 5; done ) > "$OUT/mem.log" 2>&1 &
MON=$!
docker rm -f bench-engine >/dev/null 2>&1 || true
docker run -d --name bench-engine --network host -v /opt/models:/models:ro --shm-size 4g --cap-add SYS_NICE \
  --security-opt seccomp=/opt/bench/seccomp-vllm.json -e VLLM_CPU_OMP_THREADS_BIND=auto -e VLLM_CPU_NUM_OF_RESERVED_CPU=1 \
  --entrypoint vllm "$OURS" serve "/models/hf/$REPO/$REV" --served-model-name m6-gemma-4-e4b --dtype bfloat16 --max-model-len 16384 \
  --max-num-seqs 128 --max-num-batched-tokens 2048 --no-enable-prefix-caching --seed 0 --host 0.0.0.0 --port 8000 \
  --kv-cache-memory-bytes "$KV_BYTES" --block-size 128 --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 \
  --limit-mm-per-prompt '{"image": 0, "audio": 0}' >/dev/null
t0=$(date +%s); ok=0
until curl -sf -o /dev/null http://127.0.0.1:8000/health; do
  if ! docker ps --format '{{.Names}}' | grep -qx bench-engine; then log "container exited"; break; fi
  (( $(date +%s) - t0 < 2400 )) || { log "not healthy after 2400s"; break; }
  sleep 5
done
curl -sf -o /dev/null http://127.0.0.1:8000/health && { ok=1; log "healthy after $(( $(date +%s) - t0 ))s"; }
if [[ $ok == 1 ]]; then
  curl -s -H 'Content-Type: application/json' -d '{"model":"m6-gemma-4-e4b","prompt":"<bos><|turn>user\nTranslate the following text from English to German. Reply with the German translation only.\n\nThe truck left the distribution center in Munich at 9:00.<turn|>\n<|turn>model\n","max_tokens":64,"temperature":0}' http://127.0.0.1:8000/v1/completions | tee "$OUT/first.json"; echo
  docker run --rm --network host -v "$OUT:/results" -v /opt/models:/models:ro --entrypoint vllm "$OURS" bench serve --backend vllm --host 127.0.0.1 --port 8000 \
    --model "/models/hf/$REPO/$REV" --tokenizer "/models/hf/$REPO/$REV" --served-model-name m6-gemma-4-e4b --dataset-name random --random-input-len 128 --random-output-len 128 \
    --ignore-eos --request-rate inf --max-concurrency 16 --num-prompts 48 --percentile-metrics ttft,tpot,itl,e2el --save-result --result-dir /results --result-filename c16.json \
    > "$OUT/c16.client.log" 2>&1 || tail -5 "$OUT/c16.client.log"
  grep -E "Output token throughput|Median TTFT|Median TPOT|P99 E2EL|Successful" "$OUT/c16.client.log"
fi
docker logs bench-engine > "$OUT/server.log" 2>&1
kill $MON 2>/dev/null; sleep 1
docker rm -f bench-engine >/dev/null 2>&1 || true
log "peak used (MiB): $(awk '{for(i=1;i<=NF;i++) if($i ~ /^used=/){sub("used=","",$i); if($i+0>m) m=$i+0}} END{print m}' "$OUT/mem.log")"
aws s3 cp --only-show-errors --recursive "$OUT" "s3://$BUCKET/$PREFIX/" && log "done: s3://$BUCKET/$PREFIX/"
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance-type", default="m8i.4xlarge")
    ap.add_argument("--kv-gib", type=int, default=16)
    ap.add_argument("--bucket", default="amx-gpu-bench-346399954218-us-east-1")
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args(argv)
    ec2 = boto3.client("ec2", region_name=REGION)
    prefix = f"runs/gemma4-cpu-probe-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}/hosts/{args.instance_type}"
    iid = launch(ec2, args.instance_type)
    print("instance", iid, "prefix", prefix, flush=True)
    try:
        wait_ssm(iid)
        wait_boot(iid)
        script = (REMOTE.replace("__BUCKET__", args.bucket).replace("__PREFIX__", prefix).replace("__KV__", str(args.kv_gib * 2**30))
                  .replace("__OURS__", OURS_IMAGE).replace("__REPO__", REPO).replace("__REV__", REV))
        print("status", run_remote(iid, script, args.bucket, prefix))
    finally:
        if not args.keep:
            ec2.terminate_instances(InstanceIds=[iid])
            print("terminated", iid)
    return 0


if __name__ == "__main__":
    sys.exit(main())

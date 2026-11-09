#!/usr/bin/env bash
# host_facts.sh TARGET_ID BUCKET
#
# Record what this instance is, at boot (design 8, step 1): CPU and NUMA layout, kernel,
# flags, memory, GPU, Docker and perf versions, instance metadata. Uploaded to
# s3://BUCKET/hosts/TARGET_ID/INSTANCE_ID-BOOT_TIME.json; preflight copies the latest
# file into the run.
# shellcheck source=host/lib.sh
source "$(dirname "$0")/lib.sh"
need jq

target=$1 bucket=$2
iid=$(imds instance-id)
boot=$(date -u -d "$(uptime -s)" +%Y%m%dT%H%M%SZ 2>/dev/null || date -u +%Y%m%dT%H%M%SZ)
cap() { "$@" 2>&1 || true; }

f=$(mktemp)
jq -n \
  --arg target "$target" --arg iid "$iid" --arg boot "$boot" \
  --arg itype "$(imds instance-type)" --arg az "$(imds placement/availability-zone)" \
  --arg ami "$(imds ami-id)" \
  --arg lscpu "$(cap lscpu)" --arg numa "$(cap numactl -H)" --arg uname "$(uname -a)" \
  --arg meminfo "$(head -n 5 /proc/meminfo 2>/dev/null || true)" \
  --arg cmdline "$(cat /proc/cmdline 2>/dev/null || true)" \
  --arg docker "$(cap docker version --format '{{.Server.Version}}')" \
  --arg perf "$(cap perf --version)" \
  --arg gpu "$(command -v nvidia-smi >/dev/null && cap nvidia-smi -q || echo none)" \
  '{target_id: $target, instance_id: $iid, boot_time: $boot, instance_type: $itype,
    availability_zone: $az, ami_id: $ami, uname: $uname, lscpu: $lscpu, numactl: $numa,
    meminfo: $meminfo, kernel_cmdline: $cmdline, docker_version: $docker, perf_version: $perf,
    nvidia_smi: $gpu}' >"$f"
s3_put "$f" "s3://$bucket/hosts/$target/$iid-$boot.json"
log "host_facts: uploaded hosts/$target/$iid-$boot.json"

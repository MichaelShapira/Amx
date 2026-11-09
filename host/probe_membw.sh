#!/usr/bin/env bash
# probe_membw.sh BUCKET RUN_ID
#
# Memory bandwidth of this instance type with the STREAM triad (design 8, step 3), one
# OpenMP thread per physical core, arrays sized well past the last-level cache. STREAM
# 5.10 is downloaded once and checked by sha256. Result: s3://BUCKET/runs/RUN_ID/hosts/membw-TYPE.json
# shellcheck source=host/lib.sh
source "$(dirname "$0")/lib.sh"

bucket=$1 run_id=$2
STREAM_URL=https://www.cs.virginia.edu/stream/FTP/Code/stream.c
STREAM_SHA256=a52bae5e175bea3f7832112af9c085adab47117f7d2ce219165379849231692b
bin="$STATE_DIR/stream"
itype=$(imds instance-type)
cores=$(lscpu -p=CORE,SOCKET | grep -v '^#' | sort -u | wc -l | tr -d ' ')

if [[ ! -x "$bin" ]]; then
  need gcc curl sha256sum
  src=$(mktemp --suffix=.c)
  run curl -sf -o "$src" "$STREAM_URL"
  if [[ "$DRY_RUN" != "1" ]]; then
    echo "$STREAM_SHA256  $src" | sha256sum -c --quiet - || die "stream.c checksum mismatch"
    mkdir -p "$STATE_DIR"
    # 200M doubles per array (4.8 GB total) is far past the LLC of a 4xlarge or 8xlarge.
    gcc -O3 -march=native -fopenmp -DSTREAM_ARRAY_SIZE=200000000 -DNTIMES=20 -mcmodel=large -o "$bin" "$src"
  fi
fi
if [[ "$DRY_RUN" == "1" ]]; then
  run env OMP_NUM_THREADS="$cores" OMP_PROC_BIND=spread OMP_PLACES=cores "$bin"
  exit 0
fi
out=$(OMP_NUM_THREADS="$cores" OMP_PROC_BIND=spread OMP_PLACES=cores "$bin")
triad=$(awk '/^Triad:/ {print $2}' <<<"$out")
copy=$(awk '/^Copy:/ {print $2}' <<<"$out")
[[ -n "$triad" ]] || die "STREAM produced no Triad line"
f=$(mktemp)
jq -n --arg itype "$itype" --argjson cores "$cores" --argjson triad "$triad" --argjson copy "$copy" \
  '{instance_type: $itype, threads: $cores, triad_mb_s: $triad, copy_mb_s: $copy,
    tool: "STREAM 5.10", array_elements: 200000000}' >"$f"
s3_put "$f" "s3://$bucket/runs/$run_id/hosts/membw-$itype.json"
cat "$f"

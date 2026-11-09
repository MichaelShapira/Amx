#!/usr/bin/env bash
# perf_ctl.sh start|stop BUCKET RUN_ID LEVEL_KEY
#
# Hardware counter evidence for one measured level (design 14.1): system-wide
# `perf stat -e exe.amx_busy,cycles` from the level start to its end. `stop` uploads the
# CSV to s3://BUCKET/runs/RUN_ID/amx/perf/LEVEL_KEY.csv and prints it. When the PMU event
# is not exposed (some virtualized hosts), perf prints "<not supported>" and the verdict
# falls back to the engine trace.
# shellcheck source=host/lib.sh
source "$(dirname "$0")/lib.sh"

action=$1 bucket=$2 run_id=$3 level=$4
[[ "$level" =~ ^[A-Za-z0-9._-]+$ ]] || die "bad level key: $level"
dir="$STATE_DIR/perf"
mkdir -p "$dir"
csv="$dir/$level.csv" pidf="$dir/$level.pid"

case "$action" in
  start)
    need perf
    [[ -f "$pidf" ]] && die "perf already running for $level"
    if [[ "$DRY_RUN" == "1" ]]; then
      run perf stat -a -x, -e exe.amx_busy,cycles -o "$csv"
      exit 0
    fi
    nohup perf stat -a -x, -e exe.amx_busy,cycles -o "$csv" -- sleep 86400 >/dev/null 2>&1 &
    echo $! >"$pidf"
    log "perf_ctl: started for $level (pid $(cat "$pidf"))"
    ;;
  stop)
    if [[ "$DRY_RUN" == "1" ]]; then run kill -INT "<pid of $level>"; exit 0; fi
    [[ -f "$pidf" ]] || die "perf is not running for $level"
    pid=$(cat "$pidf")
    # Interrupt the sleep child; perf then writes its counters and exits.
    pkill -INT -P "$pid" sleep || kill -INT "$pid" || true
    for _ in $(seq 50); do kill -0 "$pid" 2>/dev/null || break; sleep 0.1; done
    rm -f "$pidf"
    s3_put "$csv" "s3://$bucket/runs/$run_id/amx/perf/$level.csv"
    cat "$csv"
    ;;
  *) echo "usage: perf_ctl.sh start|stop BUCKET RUN_ID LEVEL_KEY" >&2; exit 2 ;;
esac

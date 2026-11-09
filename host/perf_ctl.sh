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
    # perf's own stderr goes to a file: when the PMU event is unavailable perf exits at once,
    # and `stop` must be able to say why instead of "no such process".
    nohup perf stat -a -x, -e exe.amx_busy,cycles -o "$csv" -- sleep 86400 >/dev/null 2>"$dir/$level.err" &
    echo $! >"$pidf"
    sleep 1
    if ! kill -0 "$(cat "$pidf")" 2>/dev/null; then
      log "perf_ctl: perf exited at start for $level: $(tr '\n' ' ' <"$dir/$level.err" | cut -c1-300)"
    else
      log "perf_ctl: started for $level (pid $(cat "$pidf"))"
    fi
    ;;
  stop)
    if [[ "$DRY_RUN" == "1" ]]; then run kill -INT "<pid of $level>"; exit 0; fi
    [[ -f "$pidf" ]] || die "perf is not running for $level"
    pid=$(cat "$pidf")
    # Interrupt the sleep child; perf then writes its counters and exits.
    pkill -INT -P "$pid" sleep 2>/dev/null || kill -INT "$pid" 2>/dev/null || true
    for _ in $(seq 50); do kill -0 "$pid" 2>/dev/null || break; sleep 0.1; done
    rm -f "$pidf"
    if [[ ! -s "$csv" ]]; then
      die "perf wrote no counters for $level: $(tr '\n' ' ' <"$dir/$level.err" 2>/dev/null | cut -c1-400)"
    fi
    s3_put "$csv" "s3://$bucket/runs/$run_id/amx/perf/$level.csv"
    cat "$csv"
    ;;
  *) echo "usage: perf_ctl.sh start|stop BUCKET RUN_ID LEVEL_KEY" >&2; exit 2 ;;
esac

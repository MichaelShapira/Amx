#!/usr/bin/env bash
# probe_amx.sh CELL_SPEC
#
# Collect the software AMX evidence for one CPU cell, outside the measured window
# (design 14.1), and upload it to s3://BUCKET/runs/RUN/amx/CELL/:
#   flags.txt        lscpu output (the CPU flags check)
#   trace.log.gz     vLLM / PyTorch cells: oneDNN verbose and vLLM GEMM dispatch lines from
#                    a verbose restart that served 5 requests at concurrency 4
#   probe_ov.json    OpenVINO cells: execution types and runtime precision (probe_ov.py)
# The engine is restarted normally afterwards. The hardware counter is per level
# (perf_ctl.sh). Parsing happens in the probes Lambda with bench/amx.py.
# shellcheck source=host/lib.sh
source "$(dirname "$0")/lib.sh"
need jq

spec=$1
if [[ "$spec" == s3://* ]]; then
  mkdir -p "$STATE_DIR/cells"
  aws s3 cp --only-show-errors "$spec" "$STATE_DIR/cells/"
  spec="$STATE_DIR/cells/$(basename "$spec")"
fi
engine=$(jq -r .engine "$spec")
key=$(jq -r .cell_key "$spec")
dest="s3://$(jq -r .bucket "$spec")/runs/$(jq -r .run_id "$spec")/amx/$key"
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

lscpu >"$work/flags.txt"
s3_put "$work/flags.txt" "$dest/flags.txt"

case "$engine" in
  vllm|thin-torch)
    "$BENCH_HOME/serve.sh" start --verbose "$spec" >/dev/null
    ip=$(private_ip)
    path=$(jq -r .health.probe.path "$spec")
    body=$(jq -c '.health.probe.body | if has("max_tokens") then .max_tokens = 16 else . end' "$spec")
    if [[ "$DRY_RUN" != "1" ]]; then
      for wave in 4 1; do
        for _ in $(seq "$wave"); do
          curl -sf -o /dev/null -H 'Content-Type: application/json' -d "$body" "http://$ip:8000$path" &
        done
        wait
      done
      # Keep only the evidence lines; the full verbose log can be gigabytes.
      docker logs bench-engine 2>&1 | grep -E 'onednn_verbose,.*,exec,|CPU unquantized GEMM dispatch|[Ii]nt8|W8A8|scaled_mm|CompressedTensors' \
        | head -n 200000 | gzip >"$work/trace.log.gz" || true
    fi
    s3_put "$work/trace.log.gz" "$dest/trace.log.gz"
    "$BENCH_HOME/serve.sh" start "$spec" >/dev/null
    ;;
  ovms|thin-openvino)
    image=$(jq -r .ov.export_image "$spec")
    # Decoders and classifiers export one openvino_model.xml; an encoder-decoder (T5) exports
    # encoder and decoder IRs, and the encoder is the one the synthetic probe request can feed.
    ovdir="$MODELS_ROOT/ov/$(jq -r .weights.repo "$spec")/$(jq -r .weights.revision "$spec")/$(jq -r .ov.precision "$spec")"
    xml=$(find "$ovdir" -name 'openvino_model.xml' -path '*/model/*' 2>/dev/null | head -n 1 || true)
    [[ -n "$xml" ]] || xml=$(find "$ovdir" -name 'openvino_encoder_model.xml' -path '*/model/*' 2>/dev/null | head -n 1 || true)
    hint=$(jq -r '.ov.hint // empty' "$spec")
    if [[ "$DRY_RUN" == "1" ]]; then xml=${xml:-/opt/models/ov/dry-run/model/openvino_model.xml}; fi
    [[ -n "$xml" ]] || die "no OpenVINO IR found for this cell; run serve.sh start first"
    args=(--xml "/models/${xml#"$MODELS_ROOT"/}")
    if [[ -n "$hint" ]]; then args+=(--hint "$hint"); fi
    if [[ "$DRY_RUN" == "1" ]]; then
      run docker run --rm --network none --entrypoint python -v "$MODELS_ROOT:/models:ro" "$image" /app/probe_ov.py "${args[@]}"
    else
      docker run --rm --network none --entrypoint python -v "$MODELS_ROOT:/models:ro" "$image" \
        /app/probe_ov.py "${args[@]}" >"$work/probe_ov.json"
    fi
    s3_put "$work/probe_ov.json" "$dest/probe_ov.json"
    ;;
  *) die "no software AMX probe for engine $engine" ;;
esac
log "probe_amx: evidence for $key uploaded to $dest"

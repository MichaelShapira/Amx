#!/usr/bin/env bash
# serve.sh start [--verbose] CELL_SPEC | stop | status
#
# Bring up one benchmark cell on this instance (design section 8, step 2). CELL_SPEC is
# the JSON written by bench/cellspec.py (a local path or s3:// URI). The script stops any
# running bench container, pulls the pinned image, stages the weights, exports to
# OpenVINO when the cell needs it, starts the container with the port published on this
# instance's private IP only, waits for health and one real request, and writes phase
# timings to s3://BUCKET/runs/RUN/bringup/CELL.json.
#
# --verbose adds the engine trace environment (ONEDNN_VERBOSE=1 ...) for the AMX probe.
# BENCH_DRY_RUN=1 prints the docker commands instead of running them.
# shellcheck source=host/lib.sh
source "$(dirname "$0")/lib.sh"

CONTAINER=bench-engine

stop_engine() {
  if [[ "$DRY_RUN" == "1" ]] || docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER"; then
    run docker rm -f "$CONTAINER" >/dev/null
  fi
}

# Log in to ECR when the image (or the export image) lives there; tokens last 12 hours.
ecr_login() {
  local image=$1 region=$2 registry
  [[ "$image" == *.dkr.ecr.*.amazonaws.com/* ]] || return 0
  registry=${image%%/*}
  if [[ "$DRY_RUN" == "1" ]]; then run docker login --username AWS --password-stdin "$registry"; return 0; fi
  aws ecr get-login-password --region "$region" | docker login --username AWS --password-stdin "$registry" >/dev/null
}

fetch_spec() {
  local src=$1 dst
  if [[ "$src" == s3://* ]]; then
    dst="$STATE_DIR/cells/$(basename "$src")"
    mkdir -p "$STATE_DIR/cells"
    aws s3 cp --only-show-errors "$src" "$dst"
    echo "$dst"
  else
    echo "$src"
  fi
}

# export_ov SPEC HF_DIR: export (or reuse) the OpenVINO IR; prints the host model directory.
export_ov() {
  local spec=$1 hf_dir=$2 repo rev prec out image exp mtype result hash
  repo=$(jq -r .weights.repo "$spec"); rev=$(jq -r .weights.revision "$spec")
  prec=$(jq -r .ov.precision "$spec"); image=$(jq -r '.ov.export_image // empty' "$spec")
  exp=$(jq -c .ov.export "$spec"); mtype=$(jq -r .ov.model_type "$spec")
  [[ -n "$image" ]] || die "cell needs an OpenVINO export but the spec has no export image"
  ecr_login "$image" "$(jq -r .region "$spec")"
  out="$MODELS_ROOT/ov/$repo/$rev/$prec"
  local cache
  cache="s3://$(jq -r .bucket "$spec")/models/ov/$repo/$rev/$prec"
  run mkdir -p "$out"
  run aws s3 sync --only-show-errors "$cache" "$out"
  # The export container runs as uid 10001 and writes the IR (and later graph.pbtxt).
  run chown -R 10001:10001 "$MODELS_ROOT/ov"
  if [[ "$DRY_RUN" == "1" ]]; then echo "$out/DRYRUNHASH/model"; return; fi
  result=$(docker run --rm --network none -v "$MODELS_ROOT:/models" "$image" ir \
    --model-type "$mtype" --source "/models/hf/$repo/$rev" --source-revision "$rev" \
    --out "/models/ov/$repo/$rev/$prec" --export-json "$exp" | tail -n 1)
  [[ -n "$result" ]] || die "OpenVINO export failed; the converter's error is above"
  hash=$(jq -r .export_hash <<<"$result")
  [[ "$hash" =~ ^[0-9a-f]{16}$ ]] || die "export did not report a hash: $result"
  if [[ "$(jq -r .reused <<<"$result")" != "true" ]]; then
    run aws s3 sync --only-show-errors "$out/$hash" "$cache/$hash"
  fi
  echo "$out/$hash/model"
}

# container_failure SPEC: the engine container exited before becoming healthy. Keep its whole log
# in the run's bringup/ prefix and repeat the exception lines last on stderr. vLLM prints the real
# cause in the engine-core process traceback, far above the final "Engine core initialization
# failed" lines, so the last 50 lines alone never named it (run gemma-20261115T102422Z, W8A8).
# keep_container_log SPEC: the whole container log into the run's bringup/ prefix, the exception
# lines repeated last on stderr (what the run page shows).
keep_container_log() {
  local spec=$1 log
  log="$STATE_DIR/container-$(jq -r .cell_key "$spec").log"
  docker logs "$CONTAINER" >"$log" 2>&1 || true
  docker logs --tail 40 "$CONTAINER" >&2 || true
  {
    echo "--- exception lines from the container log (whole log: bringup/$(jq -r .cell_key "$spec").container.log.gz)"
    grep -E '(^|[[:space:])])([a-z_][a-z0-9_]*\.)*[A-Z][A-Za-z0-9_]*(Error|Exception): ' "$log" \
      | grep -v 'Engine core initialization failed' | tail -n 6 || true
  } >&2
  gzip -f "$log"
  s3_put "$log.gz" "s3://$(jq -r .bucket "$spec")/runs/$(jq -r .run_id "$spec")/bringup/$(jq -r .cell_key "$spec").container.log.gz" \
    || log "could not upload the container log"
}
container_failure() {
  keep_container_log "$1"
  die "container exited before becoming healthy"
}

wait_health() {
  local base=$1 path=$2 timeout=$3 spec=$4 t0
  t0=$(date +%s)
  until curl -sf -o /dev/null "$base$path"; do
    if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
      container_failure "$spec"
    fi
    (( $(date +%s) - t0 < timeout )) || die "not healthy after ${timeout}s"
    sleep 2
  done
}

# first_request BASE SPEC: one real request; prints the API path that worked. When every path fails,
# the server's own answer (status and body) per path and the container log are kept: the server was
# healthy, so only its response says why (run flan-dlp-dspm-20261118T071834Z lost the GPU side with
# nothing but "failed on every API path" recorded).
first_request() {
  local base=$1 spec=$2 body path status resp
  body=$(jq -c .health.probe.body "$spec")
  for path in $(jq -r '.health.probe.path, (.health.probe.alt_paths // [])[]' "$spec"); do
    resp=$(curl -s -m 600 -w '\n%{http_code}' -H 'Content-Type: application/json' -d "$body" "$base$path" 2>&1) || true
    status=${resp##*$'\n'}
    if [[ "$status" =~ ^2 ]]; then
      echo "$path"; return 0
    fi
    log "first request to $path -> HTTP ${status:-none}: $(printf '%s' "${resp%$'\n'*}" | tr '\n' ' ' | cut -c1-600)"
  done
  keep_container_log "$spec"
  die "first request failed on every API path (the server's answers are above; its log is in bringup/)"
}

start() {
  local verbose=0
  if [[ "${1:-}" == "--verbose" ]]; then verbose=1; shift; fi
  need jq
  local spec t0 t ip image hf_dir model_dir model_dir_c engine api_path
  spec=$(fetch_spec "$1")
  [[ "$(jq -r .schema "$spec")" == "1" ]] || die "unsupported cell spec schema"
  engine=$(jq -r .engine "$spec"); image=$(jq -r .container.image "$spec")
  ip=$(private_ip)
  t0=$(now)
  log "serve: $(jq -r .cell_id "$spec") ($engine)"
  stop_engine

  t=$(now)
  ecr_login "$image" "$(jq -r .region "$spec")"
  run docker pull --quiet "$image" >/dev/null
  local image_pull_s; image_pull_s=$(elapsed "$t")
  t=$(now)
  hf_dir=$("$BENCH_HOME/stage_model.sh" "$(jq -r .weights.repo "$spec")" "$(jq -r .weights.revision "$spec")" \
    "$(jq -r .weights.gated "$spec")" "$(jq -r .bucket "$spec")" "$(jq -r .hf_token_secret "$spec")" \
    "$(jq -r .region "$spec")")
  # A decision model needs its base torso next to the adapter: stage it the same way (cache first).
  local base_dir_c=""
  if [[ "$(jq -r '.base_weights != null' "$spec")" == "true" ]]; then
    local base_dir
    base_dir=$("$BENCH_HOME/stage_model.sh" "$(jq -r .base_weights.repo "$spec")" "$(jq -r .base_weights.revision "$spec")" \
      "$(jq -r .base_weights.gated "$spec")" "$(jq -r .bucket "$spec")" "$(jq -r .hf_token_secret "$spec")" \
      "$(jq -r .region "$spec")")
    base_dir_c="/models/${base_dir#"$MODELS_ROOT"/}"
  fi
  local model_fetch_s; model_fetch_s=$(elapsed "$t")
  model_dir=$hf_dir
  local export_s=0
  if [[ "$(jq -r '.ov != null' "$spec")" == "true" ]]; then
    t=$(now); model_dir=$(export_ov "$spec" "$hf_dir"); export_s=$(elapsed "$t")
  fi
  if [[ "$engine" == "ovms" ]]; then
    local g; g=$(jq -c .ovms_graph "$spec")
    run docker run --rm --network none -v "$MODELS_ROOT:/models" "$(jq -r .ov.export_image "$spec")" graph \
      --dir "/models/${model_dir#"$MODELS_ROOT"/}" \
      --enable-prefix-caching "$(jq -r .enable_prefix_caching <<<"$g")" \
      --cache-size "$(jq -r .cache_size <<<"$g")" --max-num-seqs "$(jq -r .max_num_seqs <<<"$g")" \
      --max-num-batched-tokens "$(jq -r .max_num_batched_tokens <<<"$g")" >/dev/null
  fi
  model_dir_c="/models/${model_dir#"$MODELS_ROOT"/}"

  local opts=() cmd=() x
  while IFS= read -r x; do opts+=("${x//\{PRIVATE_IP\}/$ip}"); done < <(jq -r '.container.run_opts[]' "$spec")
  while IFS= read -r x; do cmd+=("${x//\{MODEL_DIR\}/$model_dir_c}"); done < <(jq -r '.container.cmd[]' "$spec")
  local i
  for i in "${!opts[@]}"; do opts[i]=${opts[i]//\{MODEL_DIR\}/$model_dir_c}; opts[i]=${opts[i]//\{BASE_DIR\}/$base_dir_c}; done
  if [[ "$verbose" == "1" ]]; then
    while IFS= read -r x; do opts+=(-e "$x"); done < <(jq -r '.verbose_env | to_entries[] | "\(.key)=\(.value)"' "$spec")
  fi

  t=$(now)
  run docker run -d "${opts[@]}" "$image" ${cmd[@]+"${cmd[@]}"} >/dev/null
  local base="http://$ip:8000"
  if [[ "$DRY_RUN" == "1" ]]; then
    log "dry run: skipping health checks"; return 0
  fi
  wait_health "$base" "$(jq -r .health.path "$spec")" "$(jq -r .health.timeout_s "$spec")" "$spec"
  local load_to_healthy_s; load_to_healthy_s=$(elapsed "$t")
  t=$(now); api_path=$(first_request "$base" "$spec"); local first_request_s; first_request_s=$(elapsed "$t")

  local out
  out="$STATE_DIR/bringup-$(jq -r .cell_key "$spec").json"
  jq -n --arg cell "$(jq -r .cell_id "$spec")" --arg image "$image" --arg api "$api_path" \
    --arg inst "$(imds instance-id)" --arg itype "$(imds instance-type)" \
    --argjson verbose "$verbose" --arg digest "$(docker inspect --format '{{.Image}}' "$CONTAINER")" \
    --argjson pull "$image_pull_s" --argjson fetch "$model_fetch_s" --argjson export "$export_s" \
    --argjson healthy "$load_to_healthy_s" --argjson first "$first_request_s" --argjson total "$(elapsed "$t0")" \
    '{cell_id: $cell, image: $image, image_id: $digest, api_path: $api, instance_id: $inst,
      instance_type: $itype, verbose: ($verbose == 1),
      phases: {image_pull_s: $pull, model_fetch_s: $fetch, export_s: $export,
               load_to_healthy_s: $healthy, first_request_s: $first, total_s: $total}}' >"$out"
  if [[ "$verbose" == "0" ]]; then
    s3_put "$out" "s3://$(jq -r .bucket "$spec")/runs/$(jq -r .run_id "$spec")/bringup/$(jq -r .cell_key "$spec").json"
  fi
  cat "$out"
}

status() {
  docker ps --filter "name=^${CONTAINER}$" --format '{{json .}}'
}

case "${1:-}" in
  start) shift; start "$@" ;;
  stop) stop_engine ;;
  status) status ;;
  *) echo "usage: serve.sh start [--verbose] CELL_SPEC | stop | status" >&2; exit 2 ;;
esac

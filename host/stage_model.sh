#!/usr/bin/env bash
# stage_model.sh REPO REVISION GATED BUCKET SECRET REGION
#
# Make the pinned Hugging Face snapshot available under /opt/models/hf/REPO/REVISION and
# print that path. Order: local copy, then the S3 cache (s3://BUCKET/models/hf/...),
# then a download from Hugging Face at the pinned commit, which then fills the cache.
# LFS files are checked against the sha256 the Hub reports. The token is used only for
# gated repositories, read from Secrets Manager into a 0600 tmpfs file and never logged.
# shellcheck source=host/lib.sh
source "$(dirname "$0")/lib.sh"

repo=$1 rev=$2 gated=$3 bucket=$4 secret=$5 region=$6
[[ "$rev" =~ ^[0-9a-f]{40}$ ]] || die "revision must be a 40-hex commit: $rev"
[[ "$repo" =~ ^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$ ]] || die "bad repo name: $repo"
dest="$MODELS_ROOT/hf/$repo/$rev"
cache="s3://$bucket/models/hf/$repo/$rev"

if [[ -f "$dest/.complete" ]]; then echo "$dest"; exit 0; fi
run mkdir -p "$dest"

if [[ "$DRY_RUN" != "1" ]] && aws s3 ls "$cache/.complete" >/dev/null 2>&1; then
  log "stage: syncing $repo@$rev from the S3 cache"
  run aws s3 sync --only-show-errors "$cache" "$dest"
  [[ -f "$dest/.complete" ]] || die "S3 cache copy of $repo@$rev is incomplete"
  echo "$dest"; exit 0
fi

need curl jq sha256sum
log "stage: downloading $repo@$rev from Hugging Face"
hdr_args=()
hdr=""
if [[ "$gated" == "true" ]]; then
  hdr=$(hf_header_file "$secret" "$region")
  trap 'rm -f "$hdr"' EXIT
  hdr_args=(-H "@$hdr")
fi
manifest=$(mktemp)
run curl -sf --retry 3 ${hdr_args[@]+"${hdr_args[@]}"} -o "$manifest" "https://huggingface.co/api/models/$repo/revision/$rev?blobs=true"
if [[ "$DRY_RUN" == "1" ]]; then echo "$dest"; exit 0; fi

# Weights, configs and tokenizer files only: skip ONNX, original checkpoints and
# PyTorch .bin files when safetensors exist.
has_st=$(jq '[.siblings[].rfilename | select(endswith(".safetensors"))] | length' "$manifest")
jq -r --argjson st "$has_st" '.siblings[]
  | select(.rfilename | test("^(onnx|original|coreml|openvino)/|\\.(onnx|onnx_data|h5|msgpack|ot|gguf|pth|tflite)$") | not)
  | select(($st > 0 and (.rfilename | endswith(".bin"))) | not)
  | [.rfilename, (.lfs.sha256 // "")] | @tsv' "$manifest" |
while IFS=$'\t' read -r file sha; do
  [[ "$file" != *..* ]] || die "unsafe path in manifest: $file"
  mkdir -p "$(dirname "$dest/$file")"
  curl -sfL --retry 3 ${hdr_args[@]+"${hdr_args[@]}"} -o "$dest/$file" "https://huggingface.co/$repo/resolve/$rev/$file"
  if [[ -n "$sha" ]]; then
    echo "$sha  $dest/$file" | sha256sum -c --quiet - || die "checksum mismatch for $file"
  fi
done
echo "$rev" >"$dest/.revision"
rm -f "$manifest"
touch "$dest/.complete"
# The marker goes up last, so an interrupted upload never looks complete.
run aws s3 sync --only-show-errors --exclude .complete "$dest" "$cache"
run aws s3 cp --only-show-errors "$dest/.complete" "$cache/.complete"
echo "$dest"

#!/usr/bin/env bash
# Shared helpers for the host scripts (design section 8). Sourced, not executed.
#
# BENCH_DRY_RUN=1 prints every command that would change state (docker, aws s3, curl
# downloads) instead of running it, and uses placeholder instance facts, so the scripts
# can be reviewed and tested on a laptop.

set -euo pipefail

BENCH_HOME=${BENCH_HOME:-/opt/bench}
MODELS_ROOT=${MODELS_ROOT:-/opt/models}
STATE_DIR=${STATE_DIR:-/var/lib/bench}
DRY_RUN=${BENCH_DRY_RUN:-0}

log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >&2; }
die() { log "ERROR: $*"; exit 1; }

# run CMD...: execute, or print it shell-quoted to stderr in dry-run mode (stderr, so the
# output never mixes into a command substitution that captures a script's result).
run() {
  if [[ "$DRY_RUN" == "1" ]]; then
    { printf 'DRY-RUN:'; printf ' %q' "$@"; printf '\n'; } >&2
  else
    "$@"
  fi
}

now() { date +%s.%N; }
elapsed() { awk -v a="$1" -v b="$(now)" 'BEGIN { printf "%.1f", b - a }'; }

need() { for c in "$@"; do command -v "$c" >/dev/null 2>&1 || die "missing command: $c"; done; }

# IMDSv2 metadata lookup.
imds() {
  if [[ "$DRY_RUN" == "1" ]]; then
    case "$1" in
      local-ipv4) echo "${PRIVATE_IP:-10.0.0.10}" ;;
      instance-id) echo "i-dryrun0000000000" ;;
      instance-type) echo "m8i.4xlarge" ;;
      placement/availability-zone) echo "us-east-1a" ;;
      *) echo "dry-run" ;;
    esac
    return
  fi
  local token
  token=$(curl -sf -X PUT http://169.254.169.254/latest/api/token -H "X-aws-ec2-metadata-token-ttl-seconds: 60")
  curl -sf -H "X-aws-ec2-metadata-token: $token" "http://169.254.169.254/latest/meta-data/$1"
}

private_ip() {
  local ip
  ip=$(imds local-ipv4)
  [[ "$ip" =~ ^10\.|^172\.(1[6-9]|2[0-9]|3[01])\.|^192\.168\. ]] || die "refusing to publish on non-private address $ip"
  echo "$ip"
}

# s3_put LOCAL S3URI: upload a file (no-op in dry-run).
s3_put() { run aws s3 cp --only-show-errors "$1" "$2"; }

# Read the Hugging Face token into a 0600 header file in tmpfs; prints the file path.
# The token never appears in arguments, logs or the environment of other processes.
hf_header_file() {
  local secret=$1 region=$2 f
  if [[ -d /dev/shm ]]; then f=$(mktemp /dev/shm/bench-hf.XXXXXX); else f=$(mktemp); fi
  chmod 600 "$f"
  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'Authorization: Bearer dry-run\n' >"$f"
  else
    { printf 'Authorization: Bearer '; aws secretsmanager get-secret-value --region "$region" \
        --secret-id "$secret" --query SecretString --output text; } >"$f"
  fi
  echo "$f"
}

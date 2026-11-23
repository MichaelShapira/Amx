#!/usr/bin/env bash
# Deploy the whole benchmark into the AWS account the current credentials belong to, in order:
#   1. the stacks: AmxBench-Network, AmxBench-Core, one AmxBench-Target-<id> per enabled hardware option
#   2. the Hugging Face token, if you give one (optional; only gated models such as Gemma need it)
#   3. AmxBench-WebApp and its one sign-in user (infra/deploy_webapp.sh)
#   4. the prompt pools and datasets the runs need (bench upload-artifacts)
#
#   infra/deploy.sh --email you@example.com                         # the web app's password is prompted (hidden)
#   HF_TOKEN=hf_... infra/deploy.sh --email you@example.com          # also store the Hugging Face token
#   infra/deploy.sh --email you@example.com --hf-token-stdin < tok.txt
#   infra/deploy.sh --stacks-only                                     # steps 1 and 2, no web app
#   infra/deploy.sh --skip-stacks --email you@example.com             # steps 2-4 against stacks already deployed
#
# The Hugging Face token can equally be set later, from the application (Setup -> Hugging Face token) or from a
# shell (`.venv/bin/bench hf-token --stdin`); the three paths write the same Secrets Manager secret. Neither the
# token nor the password is written to the repository, to CloudFormation, to the shell history or to any log.
# Needs: AWS credentials for the target account, Docker (images are built here), node, uv, the .venv from Setup.
set -euo pipefail
EMAIL="" HF_STDIN=0 STACKS_ONLY=0 SKIP_STACKS=0
usage() { sed -n '2,19p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --email) EMAIL="${2:-}"; shift 2 ;;
    --hf-token-stdin) HF_STDIN=1; shift ;;
    --stacks-only) STACKS_ONLY=1; shift ;;
    --skip-stacks) SKIP_STACKS=1; shift ;;
    -h|--help) usage 0 ;;
    *) echo "unknown argument: $1" >&2; usage 2 ;;
  esac
done
[[ $STACKS_ONLY == 1 || -n "$EMAIL" ]] || { echo "--email EMAIL is required (the web app's sign-in user), or --stacks-only" >&2; exit 2; }
cd "$(dirname "$0")/.."
BENCH=.venv/bin/bench
[[ -x $BENCH ]] || { echo "no $BENCH: run the Setup steps of the README first (python -m venv .venv; pip install -e '.[dev]')" >&2; exit 2; }

ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
REGION="${AMXBENCH_REGION:-us-east-1}"
echo "==> deploying into account $ACCOUNT, region $REGION (whatever the credentials point at; nothing is hard-coded)"

# ---------------------------------------------------------------- 1. stacks
if [[ $SKIP_STACKS == 0 ]]; then
  TARGETS=$(.venv/bin/python -c "
from bench.config import load_targets
print(' '.join(f'AmxBench-Target-{t}' for t, s in load_targets('config/targets.yaml').resolved().items() if s.enabled))")
  echo "==> stacks: AmxBench-Network AmxBench-Core $TARGETS"
  # shellcheck disable=SC2086
  (cd infra && npx cdk deploy AmxBench-Network AmxBench-Core $TARGETS --require-approval never --outputs-file ../deployment/outputs.json)
fi

# ---------------------------------------------------------------- 2. the Hugging Face token (optional)
if [[ $HF_STDIN == 1 ]]; then
  echo "==> storing the Hugging Face token from stdin"
  $BENCH hf-token --stdin
elif [[ -n "${HF_TOKEN:-}" ]]; then
  echo "==> storing the Hugging Face token from HF_TOKEN"
  $BENCH hf-token
  unset HF_TOKEN
else
  echo "==> no Hugging Face token given (HF_TOKEN or --hf-token-stdin); gated models (Gemma family) stay unavailable"
  echo "    until one is set: Setup -> Hugging Face token in the application, or .venv/bin/bench hf-token --stdin"
fi
[[ $STACKS_ONLY == 1 ]] && { echo "==> done (stacks only)"; exit 0; }

# ---------------------------------------------------------------- 3. the web app and its user
echo "==> web app (AmxBench-WebApp) and the sign-in user $EMAIL"
infra/deploy_webapp.sh --email "$EMAIL"

# ---------------------------------------------------------------- 4. prompt pools and datasets
echo "==> uploading the prompt pools and datasets"
$BENCH upload-artifacts
echo "==> done. $($BENCH hf-token || true)"
echo "    Cost allocation: open Cost Explorer once in this account's console, then run"
echo "    (a day after the first deploy) cd infra && npx cdk deploy AmxBench-Core -c cost_tag_revision=2"

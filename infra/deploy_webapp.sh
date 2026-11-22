#!/usr/bin/env bash
# Deploy the web app to AWS (stack AmxBench-WebApp: ECS Fargate + CloudFront + Cognito + WAF) and make sure its
# one user exists with a permanent password and the operator role.
#
#   infra/deploy_webapp.sh --email you@example.com                 # password prompted (hidden)
#   infra/deploy_webapp.sh --email you@example.com --password-stdin < pw.txt
#   AMXBENCH_INITIAL_PASSWORD=... infra/deploy_webapp.sh --email you@example.com
#   infra/deploy_webapp.sh --email you@example.com --no-deploy      # only (re)create the user
#
# The password never appears on a command line (it goes to the AWS CLI through a 0600 temp file
# that is removed on exit) and is never written to the repository. Needs: AWS credentials for the
# account (the Core stack must already be deployed), Docker (the image is built here), node, uv.
# The user pool admits nobody on its own: this script is the only way an account gets created.
set -euo pipefail

EMAIL="" PASSWORD="" FROM_STDIN=0 DEPLOY=1 REGION="${AWS_REGION:-us-east-1}"  # gitleaks:allow (empty, filled from stdin or the prompt)
usage() { sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --email) EMAIL="${2:-}"; shift 2 ;;
    --password-stdin) FROM_STDIN=1; shift ;;
    --no-deploy) DEPLOY=0; shift ;;
    -h|--help) usage 0 ;;
    *) echo "unknown argument: $1" >&2; usage 2 ;;
  esac
done
[[ "$EMAIL" =~ ^[^[:space:]@]+@[^[:space:]@]+\.[^[:space:]@]+$ ]] || { echo "--email EMAIL is required (an address)" >&2; exit 2; }

if [[ $FROM_STDIN == 1 ]]; then
  IFS= read -r PASSWORD || true
elif [[ -n "${AMXBENCH_INITIAL_PASSWORD:-}" ]]; then
  PASSWORD="$AMXBENCH_INITIAL_PASSWORD"
else
  read -r -s -p "Permanent password for $EMAIL (8+ chars, upper, lower, digit, symbol; not echoed): " PASSWORD; echo
fi
[[ ${#PASSWORD} -ge 8 ]] || { echo "the password must be at least 8 characters" >&2; exit 2; }
unset AMXBENCH_INITIAL_PASSWORD

cd "$(dirname "$0")"
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
echo "account $ACCOUNT, region $REGION"
OUT=../deployment/webapp-outputs.json
if [[ $DEPLOY == 1 ]]; then
  echo "deploying AmxBench-WebApp (builds the web app image)..."
  npx cdk deploy AmxBench-WebApp --require-approval never --outputs-file "$OUT"
fi
[[ -f "$OUT" ]] || { echo "no $OUT; run without --no-deploy first" >&2; exit 1; }
POOL=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['AmxBench-WebApp']['UserPoolId'])" "$OUT")
URL=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['AmxBench-WebApp']['ServiceUrl'])" "$OUT")
GROUP=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['AmxBench-WebApp']['OperatorGroup'])" "$OUT")

# ---------------------------------------------------------------- the one user
if aws cognito-idp admin-get-user --region "$REGION" --user-pool-id "$POOL" --username "$EMAIL" >/dev/null 2>&1; then
  echo "user $EMAIL exists; setting the permanent password"
else
  echo "creating user $EMAIL (no invitation email; the password is set here)"
  aws cognito-idp admin-create-user --region "$REGION" --user-pool-id "$POOL" --username "$EMAIL" \
    --user-attributes Name=email,Value="$EMAIL" Name=email_verified,Value=true --message-action SUPPRESS >/dev/null
fi
TMP=$(mktemp); chmod 600 "$TMP"; trap 'rm -f "$TMP"' EXIT
_PW="$PASSWORD" python3 - "$TMP" "$POOL" "$EMAIL" <<'EOF'
import json, os, sys
json.dump({"UserPoolId": sys.argv[2], "Username": sys.argv[3], "Password": os.environ["_PW"], "Permanent": True},
          open(sys.argv[1], "w"))
EOF
unset PASSWORD
aws cognito-idp admin-set-user-password --region "$REGION" --cli-input-json "file://$TMP"
rm -f "$TMP"
aws cognito-idp admin-add-user-to-group --region "$REGION" --user-pool-id "$POOL" --username "$EMAIL" --group-name "$GROUP"
echo
echo "done. Sign in at $URL as $EMAIL (hosted login; the password you provided is permanent)."
echo "Only accounts created with this script can sign in; the pool allows no self sign-up."

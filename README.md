# amx-gpu-bench

A benchmark harness that compares small-model inference (at most 4B parameters) on
NVIDIA GPU instances (g6/g5) with Intel AMX CPU instances (m8i/c8i, m7i as fallback) in
account `346399954218`, `us-east-1`. It has three parts:

- **Standard suite.** The catalog (reset 2026-11-19 to the models the comparison is about):
  Qwen3-1.7B (M1, the model Intel's Software Catalog publishes Xeon benchmarks for), Gemma 4
  E4B and E2B (M6, M7, from Intel's catalog entries), RoBERTa-large (M3) and EmbeddingGemma 300M (E1),
  on vLLM on both sides (GPU and AMX CPU; Intel's own Xeon recipe). The PyTorch and OpenVINO CPU
  options and Flan-T5 were retired on 2026-11-20/21 (3-10x and 2.3x below vLLM on the same instance;
  they compared engines, not CPU with GPU) and stay disabled so earlier runs still render,
  native PyTorch, driven by a Lambda fan-out load generator. Reports P1, P50, P99 latency,
  throughput, prefill and decode speed, cost from the Price List API, AMX verification and a
  quality check.
- **DSPM/DLP suite.** Narrow-label classification (`SENSITIVE`/`NOT_SENSITIVE`,
  `ALLOW`/`BLOCK`, one category) on a synthetic dataset built in this repo, plus two
  purpose-built detectors (D1 openai/privacy-filter, D2 Roblox PII classifier v2).
- **Translation suite.** English ⇄ German and English ⇄ Spanish on a synthetic parallel
  corpus built in this repo (`tools/translation`): characters translated per second, cost
  per million characters, and translation quality (chrF) against the reference and against
  the GPU's own output. See [Translation suite](#translation-suite).
- **Embedding suite.** The corpus's texts in all three languages embedded by EmbeddingGemma
  (vLLM pooling runner on GPU and CPU, PyTorch through the thin server): embeddings per
  second, cost per million embeddings, cosine agreement with the GPU's vectors and
  cross-lingual retrieval recall@1 (a German or Spanish text's nearest English neighbour
  should be its own translation). All prefill, no decode: the CPU's best case.
- **Intel's benchmark, reproduced.** Intel's Software Catalog method (`vllm bench serve`,
  random 128-token prompts, 128-token answers with `ignore_eos`, pass at TPOT ≤ 100 ms and
  TTFT ≤ 3 s) as a suite, with Intel's own published figures for the same model
  (`config/intel_catalog_benchmarks.json`, 79 rows for Qwen3 1.7B/4B/8B and Llama 3.1 8B on
  two-socket Xeon 6 servers) shown per core next to the run's. Every results page also
  carries a computed "why it came out this way" (`bench/mechanics.py`: reading vs writing
  share, bytes per token, bandwidth reached against bandwidth available, arithmetic against
  peak), and the executive summary ends with a "Technical details" section built from it.
- **Migration calculator.** Estimates the engineering effort to move a model from NVIDIA
  to AMX and recommends a target based on capacity.

GPU capacity is the decision driver: a CPU configuration is recommended when it meets the
SLA, and cost is shown as information. The spec is in
[.kiro/specs/cpu-amx-vs-gpu-inference/](.kiro/specs/cpu-amx-vs-gpu-inference/)
(requirements, design and tasks, revision 3).

## Repository layout

| Path | Contents |
|---|---|
| `bench/` | Config schemas, metrics, load client, cost, AMX verdicts, quality and DSPM scoring, level plans, reports, estimators, CLI |
| `lambdas/` | Preflight, target control, warm-up, wave planning, worker, probes, aggregation, reaper |
| `servers/thin/` | FastAPI server, GPU only since 2026-11-21, for models vLLM does not serve; torch backend active, OpenVINO backends |
| `servers/export/` | OpenVINO export image (retired 2026-11-21, kept for reference) and `probe_ov.py` |
| `host/` | Scripts run on the targets over SSM: host facts, model staging, `serve.sh`, AMX probe, `perf`, STREAM, seccomp profile |
| `infra/` | CDK app: `AmxBench-Network`, `AmxBench-Core`, one `AmxBench-Target-<id>` per target |
| `config/` | `models.yaml`, `targets.yaml`, `migration_effort.yaml`, `runs/*.yaml`, `dspm/` policy and span map |
| `tools/` | Prompt-set builder, DSPM dataset and request-pool builders, seccomp generator |
| `webui/` | Local web app: React front end (`src/`) and its Python API (`api/`) |
| `data/`, `prompts/generated/` | Generated locally, **not in git** |

## Setup

Requirements: Python 3.12, Node.js (for the pinned CDK CLI), Docker (images are built at
deploy time), `uv` (Lambda bundling), AWS credentials for `346399954218`.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/pip install torch==2.14.0          # optional: thin-server contract tests
cd infra && npm ci && cd ..
```

The CDK CLI is pinned to `aws-cdk` 2.1148.0 in `infra/package.json`; always call it as
`npx cdk` from `infra/`. An older global `cdk` cannot read this app's cloud assembly.

Local checks (no AWS calls):

```bash
.venv/bin/ruff check .
.venv/bin/pytest
cd infra && npx cdk synth -c az_name=us-east-1a -c skip_bundling=true -c build_images=false --quiet
```

Tests that need torch, moto, fastapi or shellcheck skip when the package is missing.

## Operator IAM

Deploying uses the CDK bootstrap roles (the account is bootstrapped, version 32), so the
operator needs `sts:AssumeRole` on `arn:aws:iam::346399954218:role/cdk-hnb659fds-*`.
Running and reading results needs:

| Action | Resource | Used by |
|---|---|---|
| `ec2:DescribeAvailabilityZones` | `*` | `cdk synth` (maps AZ ID `use1-az1` to a name) |
| `cloudformation:DescribeStacks` | `AmxBench-Core` | CLI and UI (bucket, table, state machine names) |
| `pricing:GetProducts`, `pricing:ListPriceLists` | `*` | `bench plan`, UI estimate |
| `secretsmanager:GetSecretValue` | `amx-gpu-bench/hf-token` | prompt and dataset builds (gated tokenizers) |
| `s3:PutObject`, `s3:GetObject`, `s3:ListBucket` | the bench bucket | `bench upload-artifacts`, `run`, `report`, UI |
| `states:StartExecution`, `states:StopExecution` | the run state machine and its executions | `bench run`, `bench stop`, UI |
| `dynamodb:Query`, `dynamodb:Scan`, `dynamodb:PutItem` | table `amx-gpu-bench` | `bench status`, UI runs page and bring-up journal |
| `ec2:StopInstances` | instances tagged `bench:managed=true` | manual stop if needed (the reaper normally does this) |

A read-only role is enough for `bench status`, `bench report` and the UI results pages.

## Hugging Face token

The token is stored as the Secrets Manager secret `amx-gpu-bench/hf-token` (plain string); the web app's Setup page writes it there too.
Gemma (M4) is gated; the token's account has accepted its license. Preflight checks that
the token reaches the pinned revision of every gated model and refuses the run otherwise.
The token is never written to logs, S3 or the repo: hosts read it in `stage_model.sh`
through the instance role.

To rotate it without putting the value in shell history:

```bash
umask 077; read -rs HF_NEW; printf %s "$HF_NEW" > /tmp/hf.txt; unset HF_NEW
aws secretsmanager put-secret-value --secret-id amx-gpu-bench/hf-token --secret-string file:///tmp/hf.txt
rm /tmp/hf.txt
```

## Deploy

```bash
cd infra
npx cdk diff  AmxBench-Network AmxBench-Core AmxBench-Target-t1-g6 AmxBench-Target-t2-m8i-4xl
npx cdk deploy AmxBench-Network AmxBench-Core AmxBench-Target-t1-g6 AmxBench-Target-t2-m8i-4xl
```

- **What gets created.** A VPC with one private and one public subnet in `use1-az1`, one
  NAT gateway, S3 and DynamoDB gateway endpoints, the bench bucket (Block Public Access,
  TLS only, SSE-S3, retained on destroy), the DynamoDB table (retained), the Lambdas, two
  state machines, a reaper schedule (every 10 minutes) and one launch template plus
  instance role per target stack. Everything carries the tags `project=amx-gpu-bench`,
  `owner=operator`, `purpose=cpu-amx-vs-gpu-benchmark`.
- **Standing cost while the stacks exist:** the NAT gateway, $0.045/hour (about $1.08 a
  day), plus data processing. No instance runs until a run starts one; the reaper stops
  (On-Demand) or terminates (Spot) any target with no activity for 30 minutes, and any
  target up for more than 8 hours with no active run. Run configs set
  `stop_targets_after_run: true`.
- **Images** are built at deploy time with `DockerImageAsset` and pushed to the CDK
  bootstrap asset repository (`-c build_images=false` skips them for Lambda-only
  changes). They are `linux/amd64`; on Apple Silicon they build under emulation. The
  CUDA thin-server image is 9.8 GB compressed and about 25 GB extracted: the first full
  deploy failed with "no space left on device" on a Docker Desktop VM with 21 GB free, and
  `docker builder prune -af` (build cache only) made room. The full deploy took 25 minutes
  once the pull succeeded. Lambda dependencies are bundled locally with `uv` for `aarch64`
  (Docker is the fallback).
- **Capacity fallback.** After three `InsufficientInstanceCapacity` answers for a target's
  requested type, the launcher alternates with the `alternates` listed in the run config or
  `targets.yaml` (g6.xlarge → g5.xlarge, g6.2xlarge). One launch attempt is made about every
  minute; after `run.purchase.max_launch_attempts` refusals (default 30, set in the wizard as
  "give up on a hardware option after N failed launch attempts") the target is **given up**: its
  cells are marked `not_run` with a plain-language reason (how many refusals, which sizes and
  markets, over how long), the run's META row lists it under `not_run`, and the run continues
  with the other targets. The UI then says "Finished — GPU side did not run" with the reason on
  the run, the instance row, the model row, the Results verdict and the executive summary; the
  report opens with a "GPU SIDE DID NOT RUN" banner. Every attempt is logged under
  `runs/<id>/capacity/` and listed in the report; the report rows carry the type that ran. The whole deployment is
  pinned to one AZ (`use1-az1`) so workers and targets share it; when EC2 says the type is
  available in another AZ, that is a redeploy of the network stack, not a run option.
- More targets (`t2-c8i-4xl`, `t2-m8i-8xl`) are
  separate stacks; deploy them after the smoke reports are reviewed.

## Instance types and fair pairing

Each target has a default instance type in `targets.yaml`, and every run can override it
(`run.instance_types: {t1-g6: g6.2xlarge}`, `bench run … --instance-type t1-g6=g6.2xlarge`,
or the per-target selector in the UI). The choices come from `config/instance_catalog.json`
(specs and On-Demand prices from the EC2 and Price List APIs; refresh with
`tools/refresh_instance_catalog.py`, the snapshot date is in the file). CPU targets accept
AMX families only (m8i, c8i, r8i, m7i, c7i, r7i); GPU targets accept g6, g5 and g6e.

A CPU/GPU pair is **fair** when the GPU instance's hourly On-Demand price is within 0.5x
to 2x the CPU instance's. The benchmark answers "for capacity at a similar spend, what does
each side deliver"; a g6.48xlarge (8 L4s, $13/h) against an m8i.xlarge only measures
instance size. `bench pair m8i.4xlarge` lists comparable GPU types (single-GPU first, then
closest price), the UI shows the same proposals under the cell selectors and flags the
current pair, `bench validate` prints the judgement, and `bench run` refuses an unfair pair
unless `--allow-unfair` is given. The report carries the ratio for the types that actually
ran, with a warning line when a capacity fallback made the pair unfair. Memory-sized engine
arguments (vLLM CPU KV cache, OVMS cache) scale with the override's memory.

## Spot or On-Demand

Target stacks own a launch template, not an instance. Each run launches its instances
from the template (`run.purchase.type`, default `spot`; `bench run ... --purchase
on-demand`, or the "Purchase option" radio in the UI) and finds them again by the
`bench:target` and `bench:run` tags: every run gets its own instance per hardware option, so
any number of runs can be in flight at once without touching each other's machines (a run
cannot reuse an instance left by another run; bring-up is a few minutes per run). Spot instances
are one-time requests that terminate on interruption.
When a cell step fails, the `Check` step looks at the instance:

- instance gone (interrupted, stopped, terminated): record the interruption, launch another
  instance (Spot again while `spot_capacity_window_s` lasts — default 10 minutes, the wizard's
  "fall back to On-Demand when Spot has had no capacity for N minutes" — then On-Demand if
  `spot_fallback_on_demand` is on), re-serve the model, and run only the cell's remaining
  levels; a level cut short is re-run from zero and its partial raw records are dropped;
- instance healthy and a level failed: mark that level failed and continue with the rest;
- bring-up failed on a healthy instance, or more than `max_relaunches_per_cell` (3)
  relaunches: the cell fails.

The report's provenance block lists the instances that served each target, their market,
and every interruption; a warning line names the targets that were interrupted. Cost
columns always use On-Demand prices. At the end of a run (`stop_targets_after_run`) and
from the reaper, instances are terminated whatever their market: nothing is reused between runs, and a
stopped instance would only keep its 200 GiB root volume on the bill. Three independent mechanisms
make sure no test instance outlives its run: (1) the run's own last step terminates its instances,
also when measurements or the report failed; (2) **Stop run** and **Delete** in the UI terminate the
run's instances immediately (an aborted execution never reaches its last step); (3) the reaper runs
every 10 minutes and terminates any managed instance idle for 30 minutes with no live run, running
for 8 hours with no live run, or found stopped. Root volumes are deleted with the instance.

## Prepare artifacts

```bash
.venv/bin/python tools/build_prompts.py --hf-token-secret amx-gpu-bench/hf-token           # standard prompt sets
.venv/bin/python -m tools.dspm.build_dataset --out data/dspm/v1 --hf-token-secret amx-gpu-bench/hf-token
.venv/bin/python -m tools.dspm.build_requests --dataset data/dspm/v1 --hf-token-secret amx-gpu-bench/hf-token
.venv/bin/python -m tools.translation.build_dataset --out data/translation/v1 --n 200          # synthetic EN/DE/ES corpus
.venv/bin/python -m tools.translation.build_requests --hf-token-secret amx-gpu-bench/hf-token  # per-model translation pools
.venv/bin/bench upload-artifacts
.venv/bin/python tools/build_prompts.py --publish-corpus amx-gpu-bench-<account>-us-east-1  # once per account
```

`bench upload-artifacts` copies the prompt sets (the translation pools live next to them),
the DSPM request pools, the DSPM dataset (`dev`, `test`, `manifest.json`) and the
translation corpus (`datasets/translation/v1/`) to the bucket. Run it again after any
rebuild: preflight refuses a DSPM or translation run whose request pools were built from a
different dataset than the uploaded one. `tools/build_prompts.py` also renders the
translation pools when the corpus exists locally, and `--models <id>` builds one model's
pools without dropping the others from the manifest. `--publish-corpus` copies the five public-domain Gutenberg texts the
prompt pools are cut from to `datasets/gutenberg/` in the bucket, after checking each
against its pinned SHA-256 (`GUTENBERG_SHA256`): the web app's Setup page cuts an added
model's pools from these same bytes (gutenberg.org answers the container with HTTP 503,
and a different text would make the new model's pools incomparable with the shipped ones).

## Run

```bash
.venv/bin/bench validate config/runs/smoke-m1.yaml      # offline: cells, levels, pool sizes
.venv/bin/bench plan     config/runs/smoke-m1.yaml      # duration and cost estimate (Price List API)
.venv/bin/bench run      config/runs/smoke-m1.yaml      # uploads run.yaml, starts the run state machine
.venv/bin/bench status   <run-id>
.venv/bin/bench report   <run-id> --out reports/<run-id>
.venv/bin/bench stop     <run-id> --yes                  # finished levels are kept
.venv/bin/bench estimate project.yaml --summary reports/<run-id>/summary.json   # migration calculator
```

### The web app

```bash
webui/start.sh            # builds the React app once, then serves it with the API at http://127.0.0.1:8601
```

A local application: the Python API (`webui/api/main.py`, FastAPI) binds `127.0.0.1:8601`, serves
the built front end from `webui/dist`, and makes every AWS call with the operator's own
credentials. For live-reload development run `npm run dev` inside `webui/` (Vite on
`127.0.0.1:8600`, proxying `/api` to the API). Nothing listens beyond this machine; to use it from
another machine, forward the port over SSH.

Credentials: session credentials from the account portal last about an hour. Put the portal's
`export AWS_ACCESS_KEY_ID=… / AWS_SECRET_ACCESS_KEY=… / AWS_SESSION_TOKEN=…` lines in
`~/.aws-tmp/amxbench-creds.env` (mode 600; or point `AMXBENCH_CREDS_FILE` at another file) and the
API re-reads the file whenever it changes, so a new session never needs a restart. Without the file
the default AWS credential chain is used. When the session has expired every page says so in one
sentence and the sidebar pill reads "AWS session expired"; both recover on their own once the
file holds a fresh session.

### The web app on AWS (AmxBench-WebApp)
```bash
infra/deploy_webapp.sh --email you@example.com        # prompts for the user's permanent password
```
Deploys the same application to ECS Fargate in the bench VPC's private subnet (no public address;
one always-on task with a role instead of credentials), reachable only through an internal load
balancer that only CloudFront can reach, behind an Amazon Cognito user pool and AWS WAF on the
CloudFront distribution (HTTPS on its domain, HTTP redirected), then
creates the one user it is given, with a permanent password, in the `operators` group. Nobody can
sign up: accounts exist only when created by that script (or `admin-create-user`). Every `/api`
call needs the signed-in user's access token (webui/api/auth.py); any pool user may read, only
`operators` may start, stop or delete runs. The service's role is scoped to the bench bucket's run
and data prefixes, the run table, the run state machine, terminating instances tagged
`bench:managed`, the pre-check function's configuration and Claude on Bedrock. The password is
read from a prompt, `--password-stdin` or `AMXBENCH_INITIAL_PASSWORD`, never from an argument,
and is not stored anywhere by the repository. The user pool is deletion-protected and retained
on `cdk destroy`.

It is written for the person deciding which instance to buy, not for the engineer. Light and dark
themes; GPU is always green and CPU always blue.

- **Start here**: what the tool is for, how a run works, how to read the findings and compare runs.
- **New benchmark**, a seven-step wizard: use case → test data with a live preview of the real
  prompts or documents and their expected answers → quick check or full test → models → hardware,
  where the **reasoning model** (Setup → Reasoning model; the newest Claude Opus on Amazon
  Bedrock by default) proposes the GPU size and the AMX CPU size to compare. It receives the
  model's size and kind, the test's shape, every allowed instance size with its catalog specs and
  prices and the fairness rule (hourly prices within half to double of each other), looks up what
  it is not sure of with the AgentCore built-in **web search** (`bedrock-agentcore:InvokeWebSearch`;
  the model's architecture, GPU memory, current prices, serving-software support) and answers with
  a structured proposal: the GPU, the CPU, why each, why the pair is fair, what was checked, the
  caveats and the sources. The page shows it as a card in plain words, applies it the first time
  the step is shown for a model, and keeps manual changes afterwards (**Propose again** asks once
  more). The proposal is validated against the allowed size lists before it may be applied;
  search results are treated as untrusted data; the job runs in the background (one to three
  minutes) and the page shows each search as it happens. There is no sizing table in the code
  any more: the model catalog is open (see Setup below), so sizing is reasoned per model. Private
  or local addresses are never fetched because nothing is fetched: web search is the only tool. Then parallel or sequential execution and Spot or On-Demand → load (single request, N concurrent,
  or batched for classifiers) and acceptance targets → review with the estimate. Every control has a
  help icon.
- **Test data**: the full dataset browser with filters by scenario, expected answer and hard-negative
  type, and the classification policy.
- **Runs**: per hardware option the live instance state, its market (Spot, On-Demand, or On-Demand
  as a fallback after the run's Spot waiting time without capacity), capacity refusals against
  the attempt budget, interruptions, and "Did not run" with the reason when a side was given up;
  per model-on-hardware what it is waiting for (the instance, the previous model, loading);
  per measurement the progress. Runs stopped or failed outside the normal path are reconciled
  with Step Functions. **Delete** (per row, and on the run page) removes a finished, stopped or
  failed run completely — every object under `runs/<id>/` (plan, raw request records, quality
  outputs, capacity and instance logs, the report) and every `RUN#<id>` row in the table — after
  a confirmation that spells this out; a running run must be stopped first, and the shared model
  cache, instances and other runs are untouched (`DELETE /api/runs/<id>`).
  A run whose execution died before the pre-checks (no table row) is still listed, from Step
  Functions, with its cause in words. **Deployed backend check**: the plan reads the preflight
  Lambda's `HARNESS_COMMIT` and leaves out any run setting the deployed Core does not know yet
  (`RunConfig` forbids unknown fields), with a warning naming the deploy command, so an app
  newer than the stack never produces runs that fail before they start.
- **Results**: one run or several side by side. The comparability check runs on every selection
  (same use case, same data and answer length, same request shape, fair instance pairing) and
  blocks or warns in words. Per model: a verdict sentence, head-to-head GPU and CPU cards with
  deltas, how many CPU instances replace one GPU with the limit drawn, and throughput /
  responsiveness / cost across load levels with the target line; "one at a time" shows a single
  hardware option. Targets are editable and everything is re-judged. **Executive summary**: one
  button sends the computed findings (never raw data) to the reasoning model chosen under Setup
  (the newest Claude Opus available on Amazon Bedrock by default, `us.` inference profile) and renders a
  450-word summary with the exact numbers; cached per selection, one Bedrock call per generation.
- **Setup** (operators change, everyone sees): the **Hugging Face token** (written to the
  deployment's Secrets Manager secret; the page shows only whether it is set and when it last
  changed), the **models** the benchmark offers, and the **reasoning model**. *Models* lists the
  live catalog with, per model, the serving software and whether it is **proven in a run**
  (a finished measurement exists in the run table) or **not yet run**. *Add model* takes a
  Hugging Face repository id and runs a job in the web app: look the repository up (exists,
  gated, license, latest commit pinned), derive the catalog entry from `config.json` (kind from
  the architecture suffix, size from the safetensors metadata, context length, chat template,
  labels), cut its prompt pools with its own tokenizer (the same code as `tools/build_prompts.py`
  and `tools/dspm/build_requests.py`, uploaded next to the shipped pools and merged into the
  manifests), and append it to the live catalog. The page says plainly what it cannot promise:
  the serving software is *expected* to load a new model from its architecture family; the first
  run is the proof. *Remove* takes a model off the list (finished runs keep their data; pools stay
  for a re-add). *Reasoning model* lists the Anthropic Claude and OpenAI text models the account
  can call through Bedrock Converse (the extent of the task role's `bedrock:InvokeModel` grant),
  default the newest Opus; it writes the hardware proposals and executive summaries.
  The live catalog is the object `config/models.yaml` in the bench bucket
  (`bench/catalog_store.py`); every reader — the web app, the pre-check and the report Lambdas —
  falls back to the repository's `config/models.yaml` (the seven shipped models) when the object
  is absent. Settings live in the run table (`pk=SETTINGS`, `sk=app`). Locally, export
  `AMXBENCH_HF_SECRET_ARN=<the secret's ARN>` to enable the token form.
`webui/smoke.mjs` drives every page headlessly with Playwright and fails on any console error;
`tests/test_webui_api.py` covers the API offline.

## Cost per run

```
cost = sum over targets (active hours x on-demand $/hour)
     + wall-clock hours x $0.045 (NAT)          + NAT data processing ($0.045/GB)
     + worker GB-seconds x $0.0000133334 (Lambda arm64, 1769 MB per worker)
```

Level time is about `ceil(requests / C) x E2E P50 + waves x 40 s`; each cell adds model
fetch and load (about 6 minutes) and, on CPU, the AMX probe (about 3 minutes). Until a run
has measured a model on a target, `bench plan` uses fixed speed priors. These estimates
used Price List prices from 2026-11-09 (g6.xlarge $0.8048/h, m8i.4xlarge $0.84672/h,
c8i.4xlarge $0.74968/h, m8i.8xlarge $1.69344/h):

| Run config | Wall clock | Instances | NAT | Lambda | Total |
|---|---|---|---|---|---|
| `smoke-m1.yaml` (M1, T1 vs T2, C 1/4/16, 50 requests, 1 repeat) | 0.6 h | $0.74 | $0.03 | $0.18 | **$0.95** |
| `dspm-smoke.yaml` (M1 on W1 and W3, 300 test items) | 0.4 h | $0.47 | $0.02 | $0.32 | **$0.81** |
| `full.yaml` (all models and targets, 200 requests x 3 repeats) | 107 h | $245 | $5 | $289 | **$538** |
| `dspm-full.yaml` | 13 h | $37 | $1 | $55 | **$92** |

These figures leave out instance boot and image pull, up to 30 minutes of idle time
before the reaper stops a target that a run did not stop, the standing NAT cost between
runs, and NAT data processing for model downloads (Qwen3-4B is about 8 GB). The design's
allowance for the smoke run including bring-up is $3-5. For the full standard run, the
T3 OpenVINO target sets the wall clock and Lambda workers are the largest cost. Measured
speeds from the smoke runs replace the priors.

## Methodology rules

- **Closed loop, fixed concurrency.** Each level runs C Lambda workers. Each worker sends
  its next request when the previous one ends. Workers are warmed, pre-connect, and start
  together at a barrier. Waves last at most 12 minutes.
- **Measurement window.** Throughput uses the window from the barrier to the moment the
  first worker finishes, so all C workers are active inside it.
- **Exact percentiles** over every successful request of a level, with n reported. TTFT
  includes queueing, prefill and network. Prefill and decode speeds are reported apart;
  the reports never combine them into one "N times faster" figure.
- **Fixed output length** in the standard suite (`ignore_eos`, fixed `max_tokens`). A
  record whose `completion_tokens` differs is flagged.
- **Prompt-token parity.** Chat templates are applied client-side and every record
  compares `usage.prompt_tokens` with the expected count; mismatches are flagged.
- **Prefix caching** is off in the standard suite and on in the DSPM suite, where a long
  policy is shared by every request.
- **Repeats** are pooled for headline percentiles. Mean, standard deviation and CV across
  repeats are reported; CV above 10% is flagged.
- **AMX verification** for every CPU row: CPU flags at boot, `perf stat` `exe.amx_busy`
  per level, and an engine trace (oneDNN verbose, vLLM GEMM dispatch, OpenVINO execution
  types) once per cell. FP16 and W8A8 rows must show FP16 or INT8 AMX execution.
- **Quality.** 50 fixed prompts per model at temperature 0 against the GPU BF16 cell
  (and an FP32 anchor for M3). The CPU BF16 score is the noise floor for quantized cells.
- **SLA and capacity.** A row qualifies when it meets every SLA constraint with a low error
  rate and is not INVALID. Capacity equivalence is the number of CPU instances needed to
  match the best qualifying GPU row, with the Standard and G-family vCPU quota pools shown
  separately. Cost is reported and does not change the verdict.

## DSPM/DLP suite

Workloads: W1 bulk scan, binary (~512 and ~2,048-token chunks); W2 bulk scan, one
category; W3 inline DLP, `ALLOW`/`BLOCK` (chat, email, GenAI prompts, code; 64-360
tokens); W4 structured columns, one category or `NONE`; W5 redaction is off by default.
SLA presets are `inline_strict` (P99 at most 250 ms), `inline_standard` (P99 at most
1,000 ms) and `bulk` (P99 at most 30 s, maximize documents/s).

- **Policy.** `config/dspm/policy_v1.md` (decoders) and `policy_v1_short.md` (Flan-T5)
  define the categories, decision rules and priority order. Their SHA-256 values go into
  every request manifest, `resolved.json` and the report.
- **Dataset.** `tools/dspm/build_dataset.py` mixes NVIDIA Nemotron-PII (CC BY 4.0) and
  Gretel `synthetic_pii_finance_multilingual` (Apache-2.0) at pinned revisions with a
  seeded generator (92 template families, Faker 39.1.0). The test set has 3,000 items
  (W1 1,400, W3 1,200, W4 400) and the dev set 300; half are sensitive, and about a third
  are hard negatives (masked values, test cards, placeholders, public data). Template
  families and source documents never cross splits. The same inputs give the same
  `dataset_sha256`; the build fails on any quota or leakage check.
- **Handling.** The dataset contains realistic fake secrets, card numbers and IDs that
  secret scanners flag. `data/` is in `.gitignore`; the dataset lives only on the local
  disk and in the bench bucket. Review `data/dspm/v1/review_sample.md` (100 items) before
  a sealed run.
- **Scoring.** Generative outputs are parsed strictly; anything that is not exactly one
  label counts as a format failure and is scored as positive (fail closed). M3 (NLI)
  picks its threshold on the dev set only. Recall, precision, FPR and F1 carry Wilson 95%
  intervals; hard-negative FPR is reported per case type.
- **Verdict.** A CPU row qualifies when it meets the preset and, against the GPU row of the
  same model and precision, keeps label agreement at 98% or higher and loses at most
  1 percentage point of recall. Fitness floors (recall at least 0.90, FPR at most 0.05)
  are reported separately and do not change the CPU-vs-GPU verdict.
- Speed results transfer to fine-tuned models of the same architecture and size;
  accuracy results do not.

## Translation suite

Directions `en-de`, `de-en`, `en-es`, `es-en`; text sizes `sentence` (1-2 sentences, 8-30
words, answer limit 192 tokens) and `paragraph` (5-7 sentences, 40-90 words, limit 448).
Chat models (M1, M2, M4, M6) and Flan-T5 (M5) take part; classifiers do not.
- **Corpus.** `tools/translation/generator.py` fills hand-written trilingual sentence
  frames (78 frames, six business domains: support, logistics, HR, travel, product notes,
  invoices) with seeded slot values rendered per language (dates, times, amounts, weekday
  and month names, localized city names). Every item exists in all three languages from the
  same template, so each text is the reference translation of the other two. 200 items per
  size; nothing is real and no third-party text is used. `build_dataset.py` writes
  `data/translation/v1/{items.jsonl.gz,manifest.json,review_sample.md}` with a
  `dataset_sha256`; `build_requests.py` renders one pool per model, direction and size
  (`prompts/generated/<model>/translate-<dir>-<size>.jsonl`), the source wrapped in
  `bench.translation.INSTRUCTION` and the model's chat format, with the source character
  count the throughput and cost figures use.
- **Plan.** Per direction: the run's concurrency levels (answers end where the model ends
  them, `ignore_eos: false`, greedy) plus one accuracy pass at C=8 over `test_items` items
  with outputs kept.
- **Scoring.** `bench/translation.py`: sentence-level chrF (character n-grams 1-6, beta 2)
  against the reference, mean and P10 per cell and direction; agreement with the model's
  GPU BF16 cell (chrF and identical-answer rate); length ratio and the share of answers
  cut at the token limit. A compressed format is flagged when its chrF falls more than 3
  points below the model's CPU BF16 cell. Read chrF relative to the other cells: a
  paraphrase lowers it for every cell alike.
- **Units.** `chars_s` (source characters per second) and `usd_per_1m_chars`, the unit
  managed translation services bill by (Amazon Translate $15, Google $20 per million
  characters, list prices); the report prints that context, not as like-for-like.

## Is the CPU side configured right?

Checked against Intel's published vLLM Xeon recipe on 2026-11-19 (`tools/recipe_check.py`, same
instance, same client): the harness's configuration ties the recipe at low concurrency and is 30%
ahead at 64–128 concurrent requests (the gap is the vLLM version, 0.33 vs 0.23); single-stream
decode runs at ~70% of the slice's measured memory bandwidth; per-core throughput sits inside the
range Intel publishes for whole two-socket servers. Findings, numbers and what was actually wrong
(W8A8 cells wrongly judged AMX INVALID, now fixed): [docs/recipe-check-2026-11-19.md](docs/recipe-check-2026-11-19.md).

## What the CPU can and cannot win

On AWS On-Demand pricing a GPU instance beats a price-matched Intel AMX CPU instance on **peak
throughput** for every workload measured here, including the ones built for AMX (embeddings,
classifiers): at the same hourly price one L4 (g6.xlarge, $0.80/h) carries about 7.5x the matrix
arithmetic and 2.7x the memory bandwidth of eight Xeon 6 cores (m8i.4xlarge, $0.85/h): 121 dense BF16
TFLOPS and 300 GB/s for the L4 (NVIDIA's L4 datasheet, `bench/mechanics.py` GPU table) against 8 cores x
~2 TFLOPS of AMX BF16 (1,024 FLOPs per cycle per core, Intel) and 107 GB/s measured with STREAM on the
m8i.4xlarge (`hosts/membw-*`). The CPU side runs at 70-80% of its own limits with AMX verified: ~70% of
the slice's memory bandwidth in single-stream decode and inside Intel's own per-core range
(`docs/recipe-check-2026-11-19.md`, runs `recipe-check-20261119T081855Z` and `-085351Z`). "How many CPU
instances replace one GPU" therefore always favours the GPU: 4.3x on embeddings at batch 32 (run
`embedding-full-e1-20261119T141541Z`, 2,531 vs 583 embeddings/s), 7.3x on RoBERTa classification (run
`roberta-int8-full-20261120T211803Z`), 14-16x on inline DLP and short-document scans with Gemma 4 E4B (run
`gemma-bulk-scan-dlp-dspm-20261122T085327Z`), ~4x on chat ($0.49 vs $1.93 per 1M tokens at C=16,
`docs/recipe-check-2026-11-19.md` Result 2). The one measured exception: long-document scans with Gemma 4
E4B, where the m8i.4xlarge matched the g6.2xlarge (4.5 vs 3.5 documents/s, same run). The Results page
says so up front.

The question the CPU can win is **the cheapest option that meets your targets at your load**. The
Results page takes the load to serve (requests per minute, with typical figures to start from) and
the prices to use (On-Demand, a Compute or EC2 Instance Savings Plan at 1 or 3 years with real
published rates, or your own hourly rate per side; 0 for capacity already paid for). Every group
gets an "At your load" card: each option's capacity is the highest measured request rate that still
met the targets, instances = load / capacity rounded up, cost at the chosen prices, the cheapest per
side, the crossover load above which the GPU is cheaper, and a cost-vs-load chart. Smaller sizes of
the measured CPU family are offered as **estimates** (capacity scaled by core count, memory checked
against the model) and marked as such; one run at that size confirms them. Example, EmbeddingGemma
at 600 requests/minute of single sentences: one m8i.large (estimated, $0.08/h) against one
g6.2xlarge ($0.80/h); the GPU becomes cheaper above ~26,000 requests/minute. The executive summary
is written around the same frame (`bench/demand.py`, `webui/api/compare.py`). Savings Plan rates
come from the Savings Plans API via `tools/refresh_instance_catalog.py`.

## Reading the reports

| Label | Meaning |
|---|---|
| `SMOKE` | Fewer than 200 requests per level or fewer than 3 repeats. Fine for checking the pipeline; not a result. |
| `INVALID` | AMX not verified (flag missing, counter below threshold, or no AMX in the engine trace). The row stays in the raw table and is excluded from SLA and cost rankings. |
| FP16 relabeled | The row was configured as FP16 but ran in BF16; it is relabeled and flagged. |
| `Conditional` | The CPU row qualifies but needs more CPU instances per GPU than `max_cpu_instances_per_gpu`. |
| `below floors` | A DSPM model misses the absolute recall or FPR floor (model fitness, not a CPU-vs-GPU verdict). |
| Flags | Prompt-token mismatch, output length mismatch, CV above 10%, late workers, effective concurrency below 0.9 x C, short window (a worker ran fewer than 3 requests at C > 1), quality delta, non-commercial license (M1). |
| Quality flag | Warning only. A compressed format whose answers diverge from the BF16 floor (ROUGE-L for chat, chrF for translation, cosine/recall@1 for embeddings, label agreement for classifiers) is flagged on its card and in the report notes but stays in the SLA and cost rankings: for chat the flag measures different wording against the GPU's text, not a measured loss (two BF16 cells of the same weights agree only 0.73-0.76). |

## Teardown

```bash
cd infra
npx cdk destroy AmxBench-Target-t1-g6 AmxBench-Target-t2-m8i-4xl   # and any other target stacks
npx cdk destroy AmxBench-Core AmxBench-Network
```

The bucket and table are retained (they hold results); delete them by hand when they are
no longer needed. The HF secret is not part of any stack and stays. Images remain in the
CDK bootstrap asset repository.

## Security

Targets run in a private subnet with no public IP, no key pair, IMDSv2 required and an
encrypted root volume. Engines publish port 8000 on the instance's private IP only, and
the target security group admits it only from the worker security group. By decision,
the endpoints have no API key; the network controls are the boundary. Access to hosts is
through SSM. The bucket blocks public access and denies non-TLS requests. The UI binds to
loopback.

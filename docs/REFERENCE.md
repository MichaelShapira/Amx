# Operator reference

The technical companion to the [README](../README.md). The README gets you deployed and running; this
document explains how things work underneath, in the order an operator usually needs them.

**Contents**

1. [Repository layout](#1-repository-layout)
2. [Operator IAM](#2-operator-iam)
3. [What a deploy creates](#3-what-a-deploy-creates)
4. [Hardware options and fair pairing](#4-hardware-options-and-fair-pairing)
5. [Spot or On-Demand](#5-spot-or-on-demand)
6. [When a GPU cannot be launched](#6-when-a-gpu-cannot-be-launched)
7. [Test data: building it by hand](#7-test-data-building-it-by-hand)
8. [The web application, page by page](#8-the-web-application-page-by-page)
9. [Running the web application on your own machine](#9-running-the-web-application-on-your-own-machine)
10. [Tags and cost allocation](#10-tags-and-cost-allocation)
11. [What a run costs](#11-what-a-run-costs)
12. [How the measurements are taken](#12-how-the-measurements-are-taken)
13. [The suites in detail](#13-the-suites-in-detail)
14. [Is the CPU side configured right?](#14-is-the-cpu-side-configured-right)
15. [What the CPU can and cannot win](#15-what-the-cpu-can-and-cannot-win)
16. [Reading the reports](#16-reading-the-reports)
17. [Security](#17-security)
18. [Local checks and tests](#18-local-checks-and-tests)

---

## 1. Repository layout

| Path | Contents |
|---|---|
| `bench/` | Config schemas, metrics, load client, cost, AMX verdicts, quality and DSPM scoring, level plans, reports, estimators, CLI |
| `lambdas/` | Preflight, target control, warm-up, wave planning, worker, probes, aggregation, reaper |
| `servers/thin/` | FastAPI server (GPU only since 2026-11-21) for models vLLM does not serve; torch backend active, OpenVINO backends kept |
| `servers/export/` | OpenVINO export image (retired 2026-11-21, kept for reference) and `probe_ov.py` |
| `host/` | Scripts run on the test hosts over SSM: host facts, model staging, `serve.sh`, AMX probe, `perf`, STREAM, seccomp profile |
| `infra/` | CDK app: `AmxBench-Network`, `AmxBench-Core`, one `AmxBench-Target-<id>` per hardware option, `AmxBench-WebApp`; `deploy.sh`, `deploy_webapp.sh` |
| `config/` | `models.yaml`, `targets.yaml`, `instance_catalog.json`, `migration_effort.yaml`, `runs/*.yaml`, `dspm/` policy and span map |
| `tools/` | Prompt-set builder, DSPM dataset and request-pool builders, translation corpus, seccomp generator, catalog refresh, recipe check |
| `webui/` | The web application: React front end (`src/`) and its Python API (`api/`) |
| `docs/` | User guide, this reference, the architecture diagram, the recipe check |
| `data/`, `prompts/generated/`, `deployment/` | Generated locally, **not in git** |

---

## 2. Operator IAM

Deploying uses the CDK bootstrap roles (the account must be bootstrapped; version 32 was used), so the operator
needs `sts:AssumeRole` on `arn:aws:iam::<account>:role/cdk-hnb659fds-*`.

Running benchmarks and reading results from the CLI needs:

| Action | Resource | Used by |
|---|---|---|
| `ec2:DescribeAvailabilityZones` | `*` | `cdk synth` (maps AZ ID `use1-az1` to a name) |
| `cloudformation:DescribeStacks` | `AmxBench-Core` | CLI and web app (bucket, table, state machine names) |
| `pricing:GetProducts`, `pricing:ListPriceLists` | `*` | `bench plan`, the estimate in the wizard |
| `secretsmanager:GetSecretValue` | `amx-gpu-bench/hf-token` | prompt and dataset builds (gated tokenizers) |
| `s3:PutObject`, `s3:GetObject`, `s3:ListBucket` | the bench bucket | `bench upload-artifacts`, `run`, `report`, web app |
| `states:StartExecution`, `states:StopExecution` | the run state machine and its executions | `bench run`, `bench stop`, web app |
| `dynamodb:Query`, `dynamodb:Scan`, `dynamodb:PutItem` | table `amx-gpu-bench` | `bench status`, Runs page, bring-up journal |
| `ec2:StopInstances` | instances tagged `bench:managed=true` | manual stop if ever needed (the reaper normally does this) |

A read-only role is enough for `bench status`, `bench report` and the results pages.

The hosted web application (`AmxBench-WebApp`) needs none of this from its users: its Fargate task has its own
role, scoped to the bench bucket's run and data prefixes, the run table, the run state machine, terminating
instances tagged `bench:managed`, the pre-check function's configuration, and Claude on Bedrock.

---

## 3. What a deploy creates

**Network and core**

- A VPC with one private and one public subnet in `use1-az1`, one NAT gateway, S3 and DynamoDB gateway endpoints.
- The bench bucket: Block Public Access, TLS only, SSE-S3, **retained** on destroy.
- The DynamoDB table (**retained**), the Lambdas, two state machines, a reaper schedule (every 10 minutes).
- The Hugging Face token secret `amx-gpu-bench/hf-token`, created with the placeholder value `unset`
  (**retained**). An account where that secret already existed before this version imports it instead:
  add `-c hf_secret_arn=<its full ARN>` to every `cdk deploy` of `AmxBench-Core` in that account.
- The EC2 Spot service-linked role, if the account does not have one yet.

**Target stacks** (`AmxBench-Target-<id>`, one per hardware option)

- One launch template and one instance role. No instance runs until a benchmark starts one.
- Shipped: `t1-g6` (GPU), `t2-m8i-4xl`, `t2-c8i-4xl`, `t2-m8i-8xl` (CPU). Three retired engine-comparison
  targets stay in `targets.yaml` disabled so older runs still render.

**Web application** (`AmxBench-WebApp`)

- ECS Fargate (one task, 1 vCPU / 2 GB) in the private subnet, an internal load balancer only CloudFront can
  reach, a CloudFront distribution (HTTPS; HTTP redirected), AWS WAF, and a Cognito user pool with self sign-up
  disabled. The pool is deletion-protected and retained on `cdk destroy`.

**Container images** are built at deploy time (`DockerImageAsset`) and pushed to the CDK bootstrap asset
repository. They are `linux/amd64`; on Apple Silicon they build under emulation. The CUDA thin-server image is
9.8 GB compressed and about 25 GB extracted: the first full deploy failed with "no space left on device" on a
Docker Desktop VM with 21 GB free, and `docker builder prune -af` (build cache only) made room. A full first
deploy takes about 25 minutes once the pulls succeed. `-c build_images=false` skips image builds for Lambda-only
changes. Lambda dependencies are bundled locally with `uv` for `aarch64` (Docker is the fallback).

**Region and Availability Zone** default to `us-east-1` / `use1-az1` and are set in `bench/deployment.py`
(`AMXBENCH_REGION`, `AMXBENCH_AZ_ID`). Nothing in the code names an AWS account: CDK deploys to the account of
the credentials it runs with, the web app and the CLI ask STS, and each report records the account that ran it.
The whole deployment is pinned to one AZ so the load generators and the test hosts share it.

---

## 4. Hardware options and fair pairing

Each hardware option (target) has a default instance type in `config/targets.yaml`. Every run can override it:

- in the wizard, with the per-target selector;
- in a run file, `run.instance_types: {t1-g6: g6.2xlarge}`;
- on the command line, `bench run … --instance-type t1-g6=g6.2xlarge`.

The choices come from `config/instance_catalog.json` (specs and On-Demand prices from the EC2 and Price List
APIs; refresh with `tools/refresh_instance_catalog.py`; the snapshot date is in the file). CPU targets accept AMX
families only (m8i, c8i, r8i, m7i, c7i, r7i); GPU targets accept g6, g5 and g6e.

**A CPU/GPU pair is fair** when the GPU instance's hourly On-Demand price is within 0.5x to 2x the CPU
instance's. The benchmark answers "for capacity at a similar spend, what does each side deliver"; a g6.48xlarge
(8 L4s, $13/h) against an m8i.xlarge only measures instance size.

- `bench pair m8i.4xlarge` lists comparable GPU types (single-GPU first, then closest price).
- The wizard shows the same proposals under the selectors and flags the current pair.
- `bench validate` prints the judgement; `bench run` refuses an unfair pair unless `--allow-unfair` is given.
- The report carries the price ratio of the types that actually ran, with a warning when a capacity fallback
  made the pair unfair.

Memory-sized engine arguments (vLLM CPU KV cache, OVMS cache) scale with the chosen instance's memory.

---

## 5. Spot or On-Demand

Target stacks own a launch template, not an instance. Each run launches its own instances from the template
(`run.purchase.type`, default `spot`; `bench run … --purchase on-demand`; the "Purchase option" radio in the
wizard) and finds them again by the `bench:target` and `bench:run` tags. Every run gets its own instance per
hardware option, so any number of runs can be in flight at once without touching each other's machines. Spot
instances are one-time requests that terminate on interruption.

**When a measurement step fails**, the `Check` step looks at the instance:

| Situation | What happens |
|---|---|
| Instance gone (interrupted, stopped, terminated) | The interruption is recorded; another instance is launched (Spot again while `spot_capacity_window_s` lasts, default 10 minutes, then On-Demand if `spot_fallback_on_demand` is on); the model is re-served; only the cell's remaining levels run. A level cut short is re-run from zero and its partial records dropped. |
| Instance healthy, a level failed | That level is marked failed; the rest continue. |
| Bring-up failed on a healthy instance, or more than `max_relaunches_per_cell` (3) relaunches | The cell fails. |

The report's provenance block lists the instances that served each target, their market and every interruption.
Cost columns always use On-Demand prices.

**No test instance outlives its run.** Three independent mechanisms make sure of it:

1. The run's own last step terminates its instances, also when measurements or the report failed.
2. **Stop run** and **Delete** in the web app terminate the run's instances immediately.
3. The reaper runs every 10 minutes and terminates any managed instance idle for 30 minutes with no live run,
   running for 8 hours with no live run, or found stopped.

Instances are terminated whatever their market; nothing is reused between runs, and a stopped instance would
only keep its 200 GiB root volume on the bill. Root volumes are deleted with the instance.

---

## 6. When a GPU cannot be launched

GPU capacity is sometimes refused (`InsufficientInstanceCapacity`). The launcher handles it in stages:

1. After three refusals for the requested type, it alternates with the `alternates` listed in the run config or
   `targets.yaml` (for example g6.xlarge → g5.xlarge, g6.2xlarge). One launch attempt is made about every minute.
2. After `run.purchase.max_launch_attempts` refusals (default 30; in the wizard, "give up on a hardware option
   after N failed launch attempts") the target is **given up**: its cells are marked `not_run` with a
   plain-language reason (how many refusals, which sizes and markets, over how long), and the run continues with
   the other targets.

The web app then says "Finished — GPU side did not run", with the reason on the run, the instance row, the model
row, the Results verdict and the executive summary; the report opens with a "GPU SIDE DID NOT RUN" banner. Every
attempt is logged under `runs/<id>/capacity/` and listed in the report; report rows carry the type that ran.

When EC2 says the type is available in another AZ, that is a redeploy of the network stack, not a run option:
the deployment is pinned to one AZ.

---

## 7. Test data: building it by hand

`infra/deploy.sh` uploads the shipped prompt pools and datasets (`bench upload-artifacts`). Rebuild them only
when you change the models, the policy or the corpora:

```bash
.venv/bin/python tools/build_prompts.py --hf-token-secret amx-gpu-bench/hf-token           # standard prompt sets
.venv/bin/python -m tools.dspm.build_dataset --out data/dspm/v1 --hf-token-secret amx-gpu-bench/hf-token
.venv/bin/python -m tools.dspm.build_requests --dataset data/dspm/v1 --hf-token-secret amx-gpu-bench/hf-token
.venv/bin/python -m tools.translation.build_dataset --out data/translation/v1 --n 200          # synthetic EN/DE/ES corpus
.venv/bin/python -m tools.translation.build_requests --hf-token-secret amx-gpu-bench/hf-token  # per-model translation pools
.venv/bin/bench upload-artifacts
.venv/bin/python tools/build_prompts.py --publish-corpus amx-gpu-bench-<account>-us-east-1  # once per account
```

- `bench upload-artifacts` copies the prompt sets (translation pools next to them), the DSPM request pools, the
  DSPM dataset (`dev`, `test`, `manifest.json`) and the translation corpus (`datasets/translation/v1/`) to the
  bucket. Run it again after any rebuild: preflight refuses a DSPM or translation run whose request pools were
  built from a different dataset than the uploaded one.
- `tools/build_prompts.py --models <id>` builds one model's pools without dropping the others from the manifest.
- `--publish-corpus` copies the five public-domain Gutenberg texts the prompt pools are cut from to
  `datasets/gutenberg/` in the bucket, after checking each against its pinned SHA-256 (`GUTENBERG_SHA256`). The
  Setup page cuts an added model's pools from these same bytes (gutenberg.org answers the container with HTTP
  503, and a different text would make the new model's pools incomparable with the shipped ones).

---

## 8. The web application, page by page

Written for the person deciding which instance to buy, not for the engineer. Light and dark themes; GPU is
always green and CPU always blue. Every control has a help icon. The [user guide](USER_GUIDE.md) walks through
each screen with pictures; this is the operator's view.

### Start here

What the tool is for, how a run works, how to read the findings and compare runs.

### New benchmark (seven-step wizard)

1. **Use case** (suite).
2. **Test data**, with a live preview of the real prompts or documents and their expected answers.
3. **Quick check or full test.**
4. **Models.**
5. **Hardware.** The **reasoning model** (Setup → Reasoning model; the newest Claude Opus on Amazon Bedrock by
   default) proposes the GPU size and the AMX CPU size to compare. It receives the model's size and kind, the
   test's shape, every allowed instance size with its catalog specs and prices, the fairness rule, and a digest
   of finished runs of the same model or suite (measured runs outrank anything found on the web). It looks up
   what it is unsure of with the AgentCore built-in **web search** (`bedrock-agentcore:InvokeWebSearch`) and
   answers with a structured proposal: the GPU, the CPU, why each, why the pair is fair, what was checked, the
   caveats and the sources. The proposal is validated against the allowed size lists before it may be applied;
   search results are treated as untrusted data; nothing but web search is called, so no private or local
   address is ever fetched. The job runs in the background (one to three minutes); the page shows each search as
   it happens, applies the proposal the first time the step is shown for a model, and keeps manual changes
   afterwards (**Propose again** asks once more). Then: parallel or sequential execution, Spot or On-Demand.
6. **Load** (single request, N concurrent, or batched for classifiers) and acceptance targets.
7. **Review** with the time and cost estimate.

### Test data

The full dataset browser with filters by scenario, expected answer and hard-negative type, and the
classification policy.

### Runs

- Per hardware option: the live instance state, its market (Spot, On-Demand, or On-Demand as a fallback), capacity
  refusals against the attempt budget, interruptions, and "Did not run" with the reason when a side was given up.
- Per model-on-hardware: what it is waiting for (the instance, the previous model, loading). Per measurement: the
  progress.
- Runs stopped or failed outside the normal path are reconciled with Step Functions. A run whose execution died
  before the pre-checks (no table row) is still listed, from Step Functions, with its cause in words.
- **Delete** (per row and on the run page) removes a finished, stopped or failed run completely: every object
  under `runs/<id>/` and every `RUN#<id>` row, after a confirmation that spells this out. A running run must be
  stopped first. The shared model cache, instances and other runs are untouched.
- **Deployed backend check.** The plan reads the preflight Lambda's `HARNESS_COMMIT` and leaves out any run
  setting the deployed Core does not know yet (`RunConfig` forbids unknown fields), with a warning naming the
  deploy command, so an app newer than the stack never produces runs that fail before they start.

### Results

- One run or several side by side. The comparability check runs on every selection (same use case, same data and
  answer length, same request shape, fair instance pairing) and blocks or warns in words.
- Per model: a verdict sentence, head-to-head GPU and CPU cards with deltas, how many CPU instances replace one
  GPU with the limit drawn, and throughput / responsiveness / cost across load levels with the target line.
  Targets are editable and everything is re-judged.
- **At your load**: enter the load to serve and the prices to use (On-Demand, Savings Plans at 1 or 3 years, or
  your own rate; 0 for capacity already paid for) and read the cheapest option per side and the crossover load
  (section 15).
- **Executive summary**: one button sends the computed findings (never raw data) to the reasoning model and
  renders a 450-word summary with the exact numbers; cached per selection, one Bedrock call per generation.

### Setup (operators change, everyone sees)

- **Hugging Face token**: written to the deployment's Secrets Manager secret; the page shows only whether it is
  set and when it last changed.
- **Models**: the live catalog with, per model, the serving software and whether it is **proven in a run** or
  **not yet run**. *Add model* takes a Hugging Face repository id and runs a job in the web app: look the
  repository up (exists, gated, license, latest commit pinned), derive the catalog entry from `config.json`,
  cut its prompt pools with its own tokenizer (the same code as `tools/build_prompts.py` and
  `tools/dspm/build_requests.py`), upload them next to the shipped pools, and append it to the live catalog. The
  page says plainly what it cannot promise: the serving software is *expected* to load a new model from its
  architecture family; the first run is the proof. *Remove* takes a model off the list (finished runs keep their
  data; pools stay for a re-add). The live catalog is the object `config/models.yaml` in the bench bucket
  (`bench/catalog_store.py`); every reader falls back to the repository's `config/models.yaml` when the object
  is absent.
- **Reasoning model**: the Anthropic Claude and OpenAI text models the account can call through Bedrock Converse;
  default the newest Opus. It writes the hardware proposals and executive summaries.
- **Memory**: settings live in the run table (`pk=SETTINGS`, `sk=app`).

### User guide, Technical details, Architecture

The user guide page is `docs/USER_GUIDE.md` served by the API. Technical details go down to the exact container
images. The architecture page shows `docs/architecture.png` and offers the `.drawio` for editing.

---

## 9. Running the web application on your own machine

The hosted deployment is the normal way to use the app. For development, or to use it without the WebApp
stack, it also runs locally:

```bash
webui/start.sh            # builds the React app once, then serves it with the API at http://127.0.0.1:8601
```

- The Python API (`webui/api/main.py`, FastAPI) binds `127.0.0.1:8601`, serves the built front end from
  `webui/dist`, and makes every AWS call with **your** credentials. Nothing listens beyond this machine; to use it
  from another machine, forward the port over SSH.
- For live reload run `npm run dev` inside `webui/` (Vite on `127.0.0.1:8600`, proxying `/api` to the API).
- **Credentials.** Session credentials from an account portal last about an hour. Put the portal's
  `export AWS_ACCESS_KEY_ID=… / AWS_SECRET_ACCESS_KEY=… / AWS_SESSION_TOKEN=…` lines in
  `~/.aws-tmp/amxbench-creds.env` (mode 600; or point `AMXBENCH_CREDS_FILE` at another file). The API re-reads
  the file whenever it changes, so a new session never needs a restart. Without the file the default AWS
  credential chain is used. When the session has expired every page says so in one sentence and the sidebar
  pill reads "AWS session expired"; both recover on their own once the file holds a fresh session.
- To enable the Hugging Face token form locally, export `AMXBENCH_HF_SECRET_ARN=<the secret's ARN>`.

---

## 10. Tags and cost allocation

Every taggable resource of every stack carries the tags in `bench/tags.py` (applied with `cdk.Tags.of(app)` in
`infra/app.py`):

| Tag | Value |
|---|---|
| `Amx` | `amx-gpu-bench` — the **cost allocation tag** |
| `project` | `amx-gpu-bench` |
| `purpose` | `cpu-amx-vs-gpu-benchmark` |
| `owner` | only when `AMXBENCH_OWNER` is set in the deploy's environment (a public checkout names nobody by default) |

- The EC2 instances and volumes a run launches are not stack resources; `lambdas/target_control.py` puts the same
  tags on them at `RunInstances`, plus `bench:managed`, `bench:target` and `bench:run`.
- The web service propagates its tags to its Fargate tasks (`propagate_tags`), which is what the bill meters.
- `tests/test_infra.py` fails when a taggable resource misses one of the tags.

**Activating the cost allocation tag.** Once active, Cost Explorer, budgets and the Cost and Usage Report can
group and filter by `Amx`. The Core stack activates it with a small custom resource
(`CostExplorer.UpdateCostAllocationTagsStatus`). Two AWS behaviours get in the way on a fresh account:

- Billing only learns a tag key after a resource carrying it has been billed (up to 24 hours after the first
  tagged deploy), and the API refuses a key it has not met yet.
- An account that has never opened Cost Explorer has the API disabled; open it once in the console.

So the first activation attempt is tolerated to fail. A day after the first deploy, re-run it:

```bash
aws ce list-cost-allocation-tags --type UserDefined --region us-east-1 --query 'CostAllocationTags[?TagKey==`Amx`]'
cd infra && npx cdk deploy AmxBench-Core -c cost_tag_revision=2        # re-runs the activation (any new value)
# or by hand, same effect:
aws ce update-cost-allocation-tags-status --region us-east-1 --cost-allocation-tags-status TagKey=Amx,Status=Active
```

Cost allocation tags are account-wide billing configuration; in an AWS Organization the call must come from
the management account (or an account delegated for billing). Bedrock model invocations cannot be tagged per
request and are not attributed; Lambda, Fargate, EC2, S3, DynamoDB, CloudFront and CloudWatch costs are.

---

## 11. What a run costs

```
cost = sum over targets (active hours x on-demand $/hour)
     + wall-clock hours x $0.045 (NAT)          + NAT data processing ($0.045/GB)
     + worker GB-seconds x $0.0000133334 (Lambda arm64, 1769 MB per worker)
```

Level time is about `ceil(requests / C) x E2E P50 + waves x 40 s`; each cell adds model fetch and load (about
6 minutes) and, on CPU, the AMX probe (about 3 minutes). Until a run has measured a model on a target,
`bench plan` uses fixed speed priors; measured speeds replace them afterwards.

Estimates with Price List prices from 2026-11-09 (g6.xlarge $0.8048/h, m8i.4xlarge $0.84672/h, c8i.4xlarge
$0.74968/h, m8i.8xlarge $1.69344/h):

| Run config | Wall clock | Instances | NAT | Lambda | Total |
|---|---|---|---|---|---|
| `smoke-m1.yaml` (M1, T1 vs T2, C 1/4/16, 50 requests, 1 repeat) | 0.6 h | $0.74 | $0.03 | $0.18 | **$0.95** |
| `dspm-smoke.yaml` (M1 on W1 and W3, 300 test items) | 0.4 h | $0.47 | $0.02 | $0.32 | **$0.81** |
| `full.yaml` (all models and targets, 200 requests x 3 repeats) | 107 h | $245 | $5 | $289 | **$538** |
| `dspm-full.yaml` | 13 h | $37 | $1 | $55 | **$92** |

These figures leave out instance boot and image pull, up to 30 minutes of idle time before the reaper stops a
target that a run did not stop, the standing NAT cost between runs, and NAT data processing for model downloads
(Qwen3-4B is about 8 GB). The design's allowance for a smoke run including bring-up is $3-5. The `full.yaml`
estimate predates the retirement of the OpenVINO target, which set its wall clock; Lambda workers are the
largest cost of a long run.

---

## 12. How the measurements are taken

- **Closed loop, fixed concurrency.** Each level runs C Lambda workers. Each worker sends its next request when
  the previous one ends. Workers are warmed, pre-connect, and start together at a barrier. Waves last at most
  12 minutes.
- **Measurement window.** Throughput uses the window from the barrier to the moment the first worker finishes,
  so all C workers are active inside it.
- **Exact percentiles** over every successful request of a level, with n reported. TTFT includes queueing,
  prefill and network. Prefill and decode speeds are reported apart; the reports never combine them into one
  "N times faster" figure.
- **Fixed output length** in the standard suite (`ignore_eos`, fixed `max_tokens`). A record whose
  `completion_tokens` differs is flagged.
- **Prompt-token parity.** Chat templates are applied client-side and every record compares
  `usage.prompt_tokens` with the expected count; mismatches are flagged.
- **Prefix caching** is off in the standard suite and on in the DSPM suite, where a long policy is shared by
  every request.
- **Repeats** are pooled for headline percentiles. Mean, standard deviation and CV across repeats are reported;
  CV above 10% is flagged.
- **AMX verification** for every CPU row: CPU flags at boot, `perf stat` `exe.amx_busy` per level, and an engine
  trace (oneDNN verbose, vLLM GEMM dispatch) once per cell. FP16 and W8A8 rows must show FP16 or INT8 AMX
  execution. Engines that run fp32 by design (the decision models' torso) are marked not applicable, not invalid.
- **Quality.** 50 fixed prompts per model at temperature 0 against the GPU BF16 cell (and an FP32 anchor for
  M3). The CPU BF16 score is the noise floor for quantized cells.
- **SLA and capacity.** A row qualifies when it meets every SLA constraint with a low error rate and is not
  INVALID. Capacity equivalence is the number of CPU instances needed to match the best qualifying GPU row, with
  the Standard and G-family vCPU quota pools shown separately. Cost is reported and does not change the verdict.
- **Why it came out this way.** Every results page carries a computed mechanics block (`bench/mechanics.py`:
  reading against writing share, bytes per token, bandwidth reached against bandwidth available, arithmetic
  against peak); the executive summary ends with a technical section built from it.

---

## 13. The suites in detail

### Chat and text generation

The catalog: Qwen3-1.7B (the model Intel's Software Catalog publishes Xeon benchmarks for), Gemma 4 E4B and E2B,
RoBERTa-large (classifier) and EmbeddingGemma 300M, on vLLM on both sides (Intel's own Xeon recipe on the CPU).
Reports P1/P50/P99 latency, throughput, prefill and decode speed, cost from the Price List API, the AMX verdict
and an answer-quality check against the GPU.

The PyTorch and OpenVINO CPU options and Flan-T5 were retired on 2026-11-20/21: on the same m8i.4xlarge the
PyTorch CPU server ran 3-10x below vLLM (run `dspm-1119-2241-20261120T053131Z`) and OpenVINO BF16 2.3x below vLLM
on RoBERTa (run `roberta-int8-full-20261120T211803Z`, 7.9 vs 18.3 items/s). They compared engines, not CPU with
GPU, and stay disabled so earlier runs still render.

### Sensitive-data detection (DSPM/DLP)

Workloads: W1 bulk scan, binary (~512 and ~2,048-token chunks); W2 bulk scan, one category; W3 inline DLP,
`ALLOW`/`BLOCK` (chat, email, GenAI prompts, code; 64-360 tokens); W4 structured columns, one category or `NONE`;
W5 redaction is off by default. SLA presets: `inline_strict` (P99 at most 250 ms), `inline_standard` (P99 at most
1,000 ms) and `bulk` (P99 at most 30 s, maximize documents/s).

- **Policy.** `config/dspm/policy_v1.md` (decoders) and `policy_v1_short.md` define the categories, decision rules
  and priority order. Their SHA-256 values go into every request manifest, `resolved.json` and the report. The
  decision models answer the policy question as one typed decision (yes/no at 0.5, or a choice over seven
  categories with the policy definitions).
- **Dataset.** `tools/dspm/build_dataset.py` mixes NVIDIA Nemotron-PII (CC BY 4.0) and Gretel
  `synthetic_pii_finance_multilingual` (Apache-2.0) at pinned revisions with a seeded generator (92 template
  families, Faker 39.1.0). The test set has 3,000 items (W1 1,400, W3 1,200, W4 400) and the dev set 300; half
  are sensitive, and about a third are hard negatives (masked values, test cards, placeholders, public data).
  Template families and source documents never cross splits. The same inputs give the same `dataset_sha256`;
  the build fails on any quota or leakage check.
- **Handling.** The dataset contains realistic fake secrets, card numbers and IDs that secret scanners flag.
  `data/` is in `.gitignore`; the dataset lives only on the local disk and in the bench bucket. Review
  `data/dspm/v1/review_sample.md` (100 items) before a sealed run.
- **Scoring.** Generative outputs are parsed strictly; anything that is not exactly one label counts as a format
  failure and is scored as positive (fail closed). M3 (NLI) picks its threshold on the dev set only. Recall,
  precision, FPR and F1 carry Wilson 95% intervals; hard-negative FPR is reported per case type.
- **Verdict.** A CPU row qualifies when it meets the preset and, against the GPU row of the same model and
  precision, keeps label agreement at 98% or higher and loses at most 1 percentage point of recall. Fitness
  floors (recall at least 0.90, FPR at most 0.05) are reported separately and do not change the CPU-vs-GPU
  verdict.
- Speed results transfer to fine-tuned models of the same architecture and size; accuracy results do not.

### Translation

Directions `en-de`, `de-en`, `en-es`, `es-en`; text sizes `sentence` (1-2 sentences, 8-30 words, answer limit
192 tokens) and `paragraph` (5-7 sentences, 40-90 words, limit 448). Chat models take part; classifiers do not.

- **Corpus.** `tools/translation/generator.py` fills hand-written trilingual sentence frames (78 frames, six
  business domains: support, logistics, HR, travel, product notes, invoices) with seeded slot values rendered
  per language (dates, times, amounts, weekday and month names, localized city names). Every item exists in all
  three languages from the same template, so each text is the reference translation of the other two. 200 items
  per size; nothing is real and no third-party text is used. `build_dataset.py` writes
  `data/translation/v1/{items.jsonl.gz,manifest.json,review_sample.md}` with a `dataset_sha256`;
  `build_requests.py` renders one pool per model, direction and size.
- **Plan.** Per direction: the run's concurrency levels (answers end where the model ends them,
  `ignore_eos: false`, greedy) plus one accuracy pass at C=8 over `test_items` items with outputs kept.
- **Scoring.** `bench/translation.py`: sentence-level chrF (character n-grams 1-6, beta 2) against the
  reference, mean and P10 per cell and direction; agreement with the model's GPU BF16 cell (chrF and
  identical-answer rate); length ratio and the share of answers cut at the token limit. A compressed format is
  flagged when its chrF falls more than 3 points below the model's CPU BF16 cell. Read chrF relative to the other
  cells: a paraphrase lowers it for every cell alike.
- **Units.** `chars_s` (source characters per second) and `usd_per_1m_chars`, the unit managed translation
  services bill by (Amazon Translate $15, Google $20 per million characters, list prices); the report prints that
  as context, not as like-for-like.

### Embeddings

The corpus in three languages embedded by EmbeddingGemma: embeddings per second, cost per million, cosine
agreement with the GPU's vectors, cross-lingual retrieval recall@1. All prefill: the CPU's best case.

### Agent decisions

The Strands Decider models (E4B and 12B on Gemma 4): typed decisions (choice, yes/no, scale) with calibrated
confidence, decisions per second, accuracy against gold answers, agreement with the GPU. The 12B model needs an
L40S GPU (g6e.xlarge) and an m8i.8xlarge on the CPU side; the host check verifies GPU memory before a run.

### Intel's benchmark, reproduced

Intel's Software Catalog method (`vllm bench serve`, 128-token prompts and answers, pass at TPOT ≤ 100 ms and
TTFT ≤ 3 s), with Intel's published figures for the same model (`config/intel_catalog_benchmarks.json`) shown per
core next to the run's.

---

## 14. Is the CPU side configured right?

Checked against Intel's published vLLM Xeon recipe on 2026-11-19 (`tools/recipe_check.py`, same instance, same
client): the harness's configuration ties the recipe at low concurrency and is 30% ahead at 64–128 concurrent
requests (the gap is the vLLM version, 0.33 vs 0.23); single-stream decode runs at ~70% of the slice's measured
memory bandwidth; per-core throughput sits inside the range Intel publishes for whole two-socket servers.
Findings, numbers and what was actually wrong (W8A8 cells wrongly judged AMX INVALID, since fixed):
[recipe-check-2026-11-19.md](recipe-check-2026-11-19.md).

---

## 15. What the CPU can and cannot win

**Peak throughput per dollar goes to the GPU, every time measured.** On AWS On-Demand pricing a GPU instance
beats a price-matched Intel AMX CPU instance on peak throughput for every workload measured here, including the
ones built for AMX (embeddings, classifiers). At the same hourly price one L4 (g6.xlarge, $0.80/h) carries about
7.5x the matrix arithmetic and 2.7x the memory bandwidth of eight Xeon 6 cores (m8i.4xlarge, $0.85/h):

| | L4 (g6.xlarge) | 8 Xeon 6 cores (m8i.4xlarge) | Source |
|---|---|---|---|
| Dense BF16 arithmetic | 121 TFLOPS | 8 x ~2 TFLOPS (1,024 FLOPs/cycle/core) | NVIDIA L4 datasheet; Intel |
| Memory bandwidth | 300 GB/s | 107 GB/s measured (STREAM) | `bench/mechanics.py`; `hosts/membw-*` |

The CPU side runs at 70-80% of its own limits with AMX verified (section 14), so the gap is the hardware, not
the configuration. "How many CPU instances replace one GPU" therefore always favours the GPU:

| Workload | CPU instances per GPU | Run |
|---|---|---|
| Embeddings at batch 32 | 4.3x (2,531 vs 583 embeddings/s) | `embedding-full-e1-20261119T141541Z` |
| RoBERTa classification | 7.3x | `roberta-int8-full-20261120T211803Z` |
| Inline DLP and short-document scans, Gemma 4 E4B | 14-16x | `gemma-bulk-scan-dlp-dspm-20261122T085327Z` |
| Chat | ~4x ($0.49 vs $1.93 per 1M tokens at C=16) | recipe check, Result 2 |
| **Long-document scans, Gemma 4 E4B** | **1x: the m8i.4xlarge matched the g6.2xlarge (4.5 vs 3.5 documents/s)** | same DSPM run |

The Results page says so up front.

**The question the CPU can win is the cheapest option that meets your targets at your load.** The Results page
takes the load to serve (requests per minute, with typical figures to start from) and the prices to use
(On-Demand, a Compute or EC2 Instance Savings Plan at 1 or 3 years with real published rates, or your own
hourly rate per side; 0 for capacity already paid for). Every group gets an "At your load" card:

- each option's capacity: the highest measured request rate that still met the targets;
- instances needed = load / capacity, rounded up, and the cost at the chosen prices;
- the cheapest option per side, and the crossover load above which the GPU is cheaper;
- a cost-vs-load chart.

Smaller sizes of the measured CPU family are offered as **estimates** (capacity scaled by core count, memory
checked against the model) and marked as such; one run at that size confirms them. Example: EmbeddingGemma at
600 requests/minute of single sentences: one m8i.large (estimated, $0.08/h) against one g6.2xlarge ($0.80/h);
the GPU becomes cheaper above ~26,000 requests/minute.

The CPU also wins on availability when GPU launches are refused (section 6), on the smallest unit you can buy,
and on capacity you already pay for. The executive summary is written around the same frame (`bench/demand.py`,
`webui/api/compare.py`). Savings Plan rates come from the Savings Plans API via
`tools/refresh_instance_catalog.py`.

---

## 16. Reading the reports

| Label | Meaning |
|---|---|
| `SMOKE` | Fewer than 200 requests per level or fewer than 3 repeats. Fine for checking the pipeline; not a result. |
| `INVALID` | AMX not verified (flag missing, counter below threshold, or no AMX in the engine trace). The row stays in the raw table and is excluded from SLA and cost rankings. |
| `NOT_APPLICABLE` | The engine runs fp32 by design (decision models' torso); AMX was not expected. |
| FP16 relabeled | The row was configured as FP16 but ran in BF16; it is relabeled and flagged. The report shows the precision that ran. |
| `Conditional` | The CPU row qualifies but needs more CPU instances per GPU than `max_cpu_instances_per_gpu`. |
| `below floors` | A DSPM model misses the absolute recall or FPR floor (model fitness, not a CPU-vs-GPU verdict). |
| Flags | Prompt-token mismatch, output length mismatch, CV above 10%, late workers, effective concurrency below 0.9 x C, short window (a worker ran fewer than 3 requests at C > 1), quality delta, non-commercial license (M1). |
| Quality flag | Warning only. A compressed format whose answers diverge from the BF16 floor (ROUGE-L for chat, chrF for translation, cosine/recall@1 for embeddings, label agreement for classifiers) is flagged on its card and in the report notes but stays in the SLA and cost rankings. For chat the flag measures different wording against the GPU's text, not a measured loss (two BF16 cells of the same weights agree only 0.73-0.76). |
| Units | Per model type: tokens/s for chat, items/s for classifiers, embeddings/s, decisions/s, source characters/s for translation. |

---

## 17. Security

- Test hosts run in a private subnet with no public IP, no key pair, IMDSv2 required and an encrypted root
  volume. Access is through SSM only.
- Engines publish port 8000 on the instance's private IP only; the target security group admits it only from the
  worker security group. By decision the endpoints have no API key; the network controls are the boundary.
- The bucket blocks public access and denies non-TLS requests.
- The hosted web app sits behind CloudFront, AWS WAF and Cognito; the Fargate task has no public address and its
  role is scoped to the bench resources. Any pool user may read; only members of the `operators` group may start,
  stop or delete runs, change the token, the models or the reasoning model (`webui/api/auth.py`). Nobody can
  sign up; accounts exist only when an operator creates them (README, "Sign in").
- The local web app binds to loopback.
- Secrets: the Hugging Face token and the web app password are never written to the repository, to
  CloudFormation, to logs or to S3. The password reaches the AWS CLI through a 0600 temp file that is removed on
  exit; the token is validated with huggingface.co before it is stored and is read by hosts through their
  instance role.
- Scanned with ProtoShield and Holmes (CSR rubric) before publication; findings fixed or marked with a reason.

---

## 18. Local checks and tests

No AWS calls:

```bash
.venv/bin/ruff check .
.venv/bin/pytest
cd infra && npx cdk synth -c az_name=us-east-1a -c skip_bundling=true -c build_images=false --quiet
cd webui && npx tsc --noEmit && npm test
```

Tests that need torch, moto, fastapi or shellcheck skip when the package is missing
(`.venv/bin/pip install torch==2.14.0` enables the thin-server contract tests). `webui/smoke.mjs` drives every
page headlessly with Playwright and fails on any console error; `tests/test_webui_api.py` covers the API offline.

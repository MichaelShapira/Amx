# Operator reference

The technical companion to the [README](../README.md). The README gets you deployed and signed in; this
document explains how the tool works underneath, in the order an operator usually needs it.

> [!NOTE]
> **About the numbers in this document.** Where a section quotes results (throughput ratios, costs, timings),
> they come from **our own internal test runs** while building the tool, on AWS `us-east-1` prices of
> November 2026. Your deployment starts with no runs; the Results page of your own application is the source of
> truth for your models and your prices. The internal figures are here so you know what to expect, not as a
> substitute for a run.

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
| `servers/thin/` | FastAPI server (GPU only) for models vLLM does not serve |
| `servers/export/` | OpenVINO export image (retired, kept for reference) |
| `host/` | Scripts run on the test hosts over SSM: host facts, model staging, `serve.sh`, AMX probe, `perf`, STREAM, seccomp profile |
| `infra/` | CDK app and the two deploy scripts (`deploy.sh`, `deploy_webapp.sh`) |
| `config/` | `models.yaml`, `targets.yaml`, `instance_catalog.json`, `runs/*.yaml`, `dspm/` policy and span map |
| `tools/` | Prompt-set builder, DSPM dataset and request-pool builders, translation corpus, catalog refresh, recipe check |
| `webui/` | The web application: React front end (`src/`) and its Python API (`api/`) |
| `docs/` | User guide, this reference, the architecture diagram, the recipe check record |
| `data/`, `prompts/generated/`, `deployment/` | Generated locally, **not in git** |

---

## 2. Operator IAM

**Deploying** uses the CDK bootstrap roles, so the operator needs `sts:AssumeRole` on
`arn:aws:iam::<account>:role/cdk-hnb659fds-*`. The account must be bootstrapped (`npx cdk bootstrap`, once).

**Running benchmarks from the command line** needs:

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

**Using the hosted web application** needs none of this. Users sign in with Cognito; the Fargate task has its own
role, scoped to the bench bucket's run and data prefixes, the run table, the run state machine, terminating
instances tagged `bench:managed`, the pre-check function's configuration, and Claude on Bedrock.

---

## 3. What a deploy creates

### Network and core (`AmxBench-Network`, `AmxBench-Core`)

| Resource | Notes |
|---|---|
| VPC, one private and one public subnet in `use1-az1`, one NAT gateway, S3 and DynamoDB gateway endpoints | The whole deployment is pinned to one AZ so the load generators and the test hosts share it |
| The bench bucket | Block Public Access, TLS only, SSE-S3; **retained** on destroy |
| The DynamoDB table | **Retained** on destroy |
| Lambdas, two state machines, a reaper schedule (every 10 minutes) | |
| The secret `amx-gpu-bench/hf-token` | Created with the placeholder value `unset`; **retained**. See the tip below for accounts that already had it |
| The EC2 Spot service-linked role | Only if the account does not have one yet |

> [!TIP]
> An account where the secret `amx-gpu-bench/hf-token` already existed before this version imports it instead
> of creating a new one: add `-c hf_secret_arn=<its full ARN>` to every `cdk deploy` of `AmxBench-Core` in that
> account. A fresh account needs nothing.

### Hardware options (`AmxBench-Target-<id>`, one stack each)

Each stack owns **one launch template and one instance role**. No instance runs until a benchmark starts one.

| Target | Default instance | Side |
|---|---|---|
| `t1-g6` | g6.xlarge (one L4) | GPU |
| `t2-m8i-4xl` | m8i.4xlarge (8 Xeon 6 cores) | CPU |
| `t2-c8i-4xl` | c8i.4xlarge | CPU |
| `t2-m8i-8xl` | m8i.8xlarge (16 cores) | CPU |

Three retired engine-comparison targets stay in `targets.yaml` disabled so older runs still render.

### Web application (`AmxBench-WebApp`)

ECS Fargate (one task, 1 vCPU / 2 GB) in the private subnet; an internal load balancer only CloudFront can reach;
a CloudFront distribution (HTTPS, HTTP redirected) with AWS WAF; a Cognito user pool `amx-gpu-bench` with
self sign-up disabled, deletion-protected and retained on `cdk destroy`.

### Container images

- Built at deploy time (`DockerImageAsset`) and pushed to the CDK bootstrap asset repository.
- `linux/amd64`; on Apple Silicon they build under emulation.
- The CUDA image is 9.8 GB compressed and about 25 GB extracted. In our internal tests the first full deploy
  failed with "no space left on device" on a Docker Desktop VM with 21 GB free; `docker builder prune -af`
  (build cache only) made room. A full first deploy took about 25 minutes once the pulls succeeded.
- `-c build_images=false` skips image builds for Lambda-only changes. Lambda dependencies are bundled locally with
  `uv` for `aarch64` (Docker is the fallback).

### Region, Availability Zone, account

Defaults are `us-east-1` / `use1-az1`, set in `bench/deployment.py` and overridable with `AMXBENCH_REGION` /
`AMXBENCH_AZ_ID`. Nothing in the code names an AWS account: CDK deploys to the account of the credentials it runs
with, the web app and the CLI ask STS, and each report records the account that ran it.

---

## 4. Hardware options and fair pairing

Each hardware option has a default instance type in `config/targets.yaml`. Every run can override it:

| Where | How |
|---|---|
| The wizard | the per-target selector on the hardware step |
| A run file | `run.instance_types: {t1-g6: g6.2xlarge}` |
| The command line | `bench run … --instance-type t1-g6=g6.2xlarge` |

The choices come from `config/instance_catalog.json` (specs and On-Demand prices from the EC2 and Price List APIs;
refresh with `tools/refresh_instance_catalog.py`; the snapshot date is in the file). CPU targets accept AMX
families only (m8i, c8i, r8i, m7i, c7i, r7i); GPU targets accept g6, g5 and g6e.

### The fairness rule

A CPU/GPU pair is **fair** when the GPU instance's hourly On-Demand price is within **0.5x to 2x** the CPU
instance's. The benchmark answers "for capacity at a similar spend, what does each side deliver"; a g6.48xlarge
(8 L4s, about $13/h) against an m8i.xlarge would only measure instance size.

- `bench pair m8i.4xlarge` lists comparable GPU types (single-GPU first, then closest price).
- The wizard shows the same proposals under the selectors and flags the current pair.
- `bench validate` prints the judgement; `bench run` refuses an unfair pair unless `--allow-unfair` is given.
- The report carries the price ratio of the types that actually ran, with a warning when a capacity fallback made
  the pair unfair.

Memory-sized engine arguments (the vLLM CPU KV cache) scale with the chosen instance's memory.

---

## 5. Spot or On-Demand

Each run launches **its own instances** from the target's launch template (`run.purchase.type`, default `spot`;
`bench run … --purchase on-demand`; the "Purchase option" radio in the wizard) and finds them again by the
`bench:target` and `bench:run` tags. Any number of runs can be in flight at once without touching each other's
machines. Spot instances are one-time requests that terminate on interruption.

### When a measurement step fails

| Situation | What happens |
|---|---|
| Instance gone (interrupted, stopped, terminated) | The interruption is recorded. Another instance is launched: Spot again while `spot_capacity_window_s` lasts (default 10 minutes), then On-Demand if `spot_fallback_on_demand` is on. The model is re-served and only the cell's remaining levels run. A level cut short is re-run from zero; its partial records are dropped. |
| Instance healthy, a level failed | That level is marked failed; the rest continue. |
| Bring-up failed on a healthy instance, or more than `max_relaunches_per_cell` (3) relaunches | The cell fails. |

The report's provenance block lists the instances that served each target, their market and every interruption.
**Cost columns always use On-Demand prices**, whatever market the run used.

### No test instance outlives its run

Three independent mechanisms make sure of it:

1. The run's own last step terminates its instances, also when measurements or the report failed.
2. **Stop run** and **Delete** in the web app terminate the run's instances immediately.
3. The reaper runs every 10 minutes and terminates any managed instance idle for 30 minutes with no live run,
   running for 8 hours with no live run, or found stopped.

Instances are terminated, never stopped: nothing is reused between runs, and a stopped instance would keep its
200 GiB root volume on the bill. Root volumes are deleted with the instance.

---

## 6. When a GPU cannot be launched

GPU capacity is sometimes refused (`InsufficientInstanceCapacity`). The launcher handles it in stages:

1. **Alternates.** After three refusals for the requested type, it alternates with the `alternates` listed in the
   run config or `targets.yaml` (for example g6.xlarge → g5.xlarge → g6.2xlarge). One attempt about every minute.
2. **Give up.** After `run.purchase.max_launch_attempts` refusals (default 30; in the wizard, "give up on a
   hardware option after N failed launch attempts") the target is **given up**. Its cells are marked `not_run`
   with a plain-language reason (how many refusals, which sizes and markets, over how long), and the run
   continues with the other side.

The web app then says **"Finished — GPU side did not run"**, with the reason on the run, the instance row, the
model row, the Results verdict and the executive summary; the report opens with a "GPU SIDE DID NOT RUN" banner.
Every attempt is logged under `runs/<id>/capacity/` and listed in the report, and report rows carry the type that
actually ran.

> [!NOTE]
> When EC2 says the type is available in another AZ, that is a redeploy of the network stack, not a run option:
> the deployment is pinned to one AZ. In our internal tests a g6.xlarge was refused 27 times in a row before a
> g5.xlarge alternate launched; the run completed on it and the report said so.

---

## 7. Test data: building it by hand

`infra/deploy.sh` uploads the shipped prompt pools and datasets (`bench upload-artifacts`). Rebuild them only when
you change the models, the policy or the corpora:

```bash
.venv/bin/python tools/build_prompts.py --hf-token-secret amx-gpu-bench/hf-token           # standard prompt sets
.venv/bin/python -m tools.dspm.build_dataset --out data/dspm/v1 --hf-token-secret amx-gpu-bench/hf-token
.venv/bin/python -m tools.dspm.build_requests --dataset data/dspm/v1 --hf-token-secret amx-gpu-bench/hf-token
.venv/bin/python -m tools.translation.build_dataset --out data/translation/v1 --n 200          # synthetic EN/DE/ES corpus
.venv/bin/python -m tools.translation.build_requests --hf-token-secret amx-gpu-bench/hf-token  # per-model translation pools
.venv/bin/bench upload-artifacts
.venv/bin/python tools/build_prompts.py --publish-corpus amx-gpu-bench-<account>-us-east-1  # once per account
```

| Command | What it does |
|---|---|
| `bench upload-artifacts` | Copies the prompt sets, the DSPM request pools, the DSPM dataset and the translation corpus to the bucket. Run it again after any rebuild: the pre-check refuses a DSPM or translation run whose pools were built from a different dataset than the uploaded one. |
| `tools/build_prompts.py --models <id>` | Builds one model's pools without dropping the others from the manifest. |
| `--publish-corpus` | Copies the five public-domain Gutenberg texts the prompt pools are cut from to `datasets/gutenberg/` in the bucket, after checking each against its pinned SHA-256. The Setup page cuts an added model's pools from these same bytes, so new models stay comparable with the shipped ones (gutenberg.org refuses the container with HTTP 503). |

---

## 8. The web application, page by page

Written for the person deciding which instance to buy, not for the engineer. Light and dark themes; GPU is
always green and CPU always blue; every control has a help icon. The [user guide](USER_GUIDE.md) walks through
each screen with pictures. This is the operator's view of what each page does underneath.

### Start here

What the tool is for, how a run works, how to read the findings and compare runs.

### New benchmark: the seven-step wizard

| Step | What happens |
|---|---|
| 1. Use case | Pick a suite (section 13). |
| 2. Test data | Live preview of the real prompts or documents and their expected answers. |
| 3. Quick check or full test | Quick check: fewer requests, one repeat, marked `SMOKE`. Full test: 200+ requests per level, 3 repeats. |
| 4. Models | From the live catalog (Setup → Models). |
| 5. Hardware | The reasoning model proposes the GPU and CPU sizes (below). Then parallel or sequential execution, Spot or On-Demand. |
| 6. Load | Single request, N concurrent, or batched for classifiers; acceptance targets. |
| 7. Review | The plan with its time and cost estimate. |

**The hardware proposal.** The reasoning model (Setup → Reasoning model; the newest Claude Opus on Amazon Bedrock
by default) receives:

- the model's size and kind, and the test's shape;
- every allowed instance size with its catalog specs and prices, and the fairness rule;
- a digest of **your finished runs** of the same model, then of the same suite. Measured runs outrank anything
  found on the web, so the proposal gets better as your deployment accumulates results.

It looks up what it is unsure of with the AgentCore built-in **web search** (`bedrock-agentcore:InvokeWebSearch`),
the only tool it has, so no private or local address is ever fetched. It answers with a structured proposal: the
GPU, the CPU, why each, why the pair is fair, what was checked, the caveats and the sources. The proposal is
validated against the allowed size lists before it may be applied; search results are treated as untrusted data.
The job runs in the background (one to three minutes); the page shows each search as it happens, applies the
proposal the first time the step is shown for a model, and keeps your manual changes afterwards (**Propose
again** asks once more).

### Test data

The full dataset browser with filters by scenario, expected answer and hard-negative type, and the
classification policy.

### Runs

- **Per hardware option:** live instance state, its market (Spot, On-Demand, or On-Demand as a fallback),
  capacity refusals against the attempt budget, interruptions, and "Did not run" with the reason.
- **Per model on hardware:** what it is waiting for (the instance, the previous model, loading).
  **Per measurement:** progress.
- Runs stopped or failed outside the normal path are reconciled with Step Functions. A run whose execution died
  before the pre-checks is still listed, from Step Functions, with its cause in words.
- **Delete** removes a finished, stopped or failed run completely (every object under `runs/<id>/`, every
  `RUN#<id>` row) after a confirmation that spells this out. A running run must be stopped first. The shared model
  cache, instances and other runs are untouched.
- **Deployed backend check.** The plan reads the deployed pre-check Lambda's `HARNESS_COMMIT` and leaves out any
  setting the deployed Core does not know yet, with a warning naming the deploy command, so an app newer than
  the stack never produces runs that fail before they start.

### Results

- One run, or several side by side. The **comparability check** runs on every selection (same use case, same data
  and answer length, same request shape, fair instance pairing) and blocks or warns in words.
- **Per model:** a verdict sentence, head-to-head GPU and CPU cards with deltas, how many CPU instances replace one
  GPU with the limit drawn, and throughput / responsiveness / cost across load levels with the target line.
  Targets are editable and everything is re-judged.
- **At your load:** your traffic and your prices in, the cheapest option per side and the crossover load out
  (section 15).
- **Executive summary:** one button sends the computed findings (never raw data) to the reasoning model and
  renders a 450-word summary with the exact numbers. Cached per selection; one Bedrock call per generation.

### Setup (operators change, everyone sees)

| Setting | What it does |
|---|---|
| **Hugging Face token** | Written to the deployment's Secrets Manager secret. The page shows only whether it is set and when it last changed. |
| **Models** | The live catalog, with per model the serving software and whether it is **proven in a run** or **not yet run**. *Add model* takes a Hugging Face repository id and runs a job: look the repository up (exists, gated, license, latest commit pinned), derive the catalog entry from `config.json`, cut its prompt pools with its own tokenizer, upload them next to the shipped pools, append it to the catalog. The page says what it cannot promise: the serving software is *expected* to load a new model from its architecture family; the first run is the proof. *Remove* takes a model off the list; finished runs keep their data. |
| **Reasoning model** | The Anthropic Claude and OpenAI text models the account can call through Bedrock Converse; default the newest Opus. It writes the hardware proposals and the executive summaries. |

The live catalog is the object `config/models.yaml` in the bench bucket (`bench/catalog_store.py`); every reader
falls back to the repository's `config/models.yaml` when the object is absent. Settings live in the run table
(`pk=SETTINGS`, `sk=app`).

### User guide, Technical details, Architecture

The user guide page serves `docs/USER_GUIDE.md`. Technical details go down to the exact container images.
Architecture shows `docs/architecture.png` and offers the `.drawio` for editing.

---

## 9. Running the web application on your own machine

The hosted deployment is the normal way to use the app. For development, or to use it without the WebApp stack,
it also runs locally:

```bash
webui/start.sh            # builds the React app once, then serves it with the API at http://127.0.0.1:8601
```

| | |
|---|---|
| **Where it listens** | `127.0.0.1:8601` only. Nothing is reachable from another machine; forward the port over SSH if you need that. |
| **Whose credentials** | Yours. Every AWS call uses the credentials of the shell that started it. |
| **Short-lived sessions** | Put the portal's `export AWS_ACCESS_KEY_ID=… / AWS_SECRET_ACCESS_KEY=… / AWS_SESSION_TOKEN=…` lines in `~/.aws-tmp/amxbench-creds.env` (mode 600; or point `AMXBENCH_CREDS_FILE` at another file). The API re-reads the file whenever it changes, so a new session never needs a restart. Without the file the default AWS credential chain is used. |
| **Expired session** | Every page says so in one sentence and the sidebar pill reads "AWS session expired"; both recover on their own once the file holds a fresh session. |
| **Live reload** | `npm run dev` inside `webui/` (Vite on `127.0.0.1:8600`, proxying `/api`). |
| **Token form** | Export `AMXBENCH_HF_SECRET_ARN=<the secret's ARN>` to enable it locally. |

---

## 10. Tags and cost allocation

Every taggable resource of every stack carries the tags in `bench/tags.py` (applied with `cdk.Tags.of(app)`):

| Tag | Value |
|---|---|
| `Amx` | `amx-gpu-bench` — the **cost allocation tag** |
| `project` | `amx-gpu-bench` |
| `purpose` | `cpu-amx-vs-gpu-benchmark` |
| `owner` | Only when `AMXBENCH_OWNER` is set in the deploy's environment. A public checkout names nobody by default. |

- The EC2 instances and volumes a run launches are not stack resources; `lambdas/target_control.py` puts the same
  tags on them at launch, plus `bench:managed`, `bench:target` and `bench:run`.
- The web service propagates its tags to its Fargate tasks, which is what the bill meters.
- `tests/test_infra.py` fails when a taggable resource misses one of the tags.

### Activating the cost allocation tag

Once active, Cost Explorer, budgets and the Cost and Usage Report can group and filter by `Amx`. The Core stack
activates it with a small custom resource (`CostExplorer.UpdateCostAllocationTagsStatus`). Two AWS behaviours
get in the way on a fresh account:

- Billing only learns a tag key after a resource carrying it has been **billed**, up to 24 hours after the first
  tagged deploy, and the API refuses a key it has not met yet.
- An account that has **never opened Cost Explorer** has the API disabled. Open it once in the console.

So the first activation attempt is allowed to fail. A day after the first deploy, re-run it:

```bash
aws ce list-cost-allocation-tags --type UserDefined --region us-east-1 --query 'CostAllocationTags[?TagKey==`Amx`]'
cd infra && npx cdk deploy AmxBench-Core -c cost_tag_revision=2        # re-runs the activation (any new value)
# or by hand, same effect:
aws ce update-cost-allocation-tags-status --region us-east-1 --cost-allocation-tags-status TagKey=Amx,Status=Active
```

> [!NOTE]
> Cost allocation tags are account-wide billing configuration; in an AWS Organization the call must come from the
> management account (or an account delegated for billing). Bedrock model invocations cannot be tagged per request
> and are not attributed; Lambda, Fargate, EC2, S3, DynamoDB, CloudFront and CloudWatch costs are.

---

## 11. What a run costs

```
cost = sum over targets (active hours x on-demand $/hour)
     + wall-clock hours x $0.045 (NAT)          + NAT data processing ($0.045/GB)
     + worker GB-seconds x $0.0000133334 (Lambda arm64, 1769 MB per worker)
```

A measurement level takes about `ceil(requests / C) x E2E P50 + waves x 40 s`. Each model-on-hardware cell adds
model fetch and load (about 6 minutes) and, on CPU, the AMX probe (about 3 minutes). Until your deployment has
measured a model on a target, `bench plan` and the wizard use fixed speed priors; measured speeds replace them
afterwards.

**Estimates for the shipped run configs**, at Price List prices of 2026-11-09 (g6.xlarge $0.8048/h, m8i.4xlarge
$0.84672/h, c8i.4xlarge $0.74968/h, m8i.8xlarge $1.69344/h):

| Run config | Wall clock | Instances | NAT | Lambda | Total |
|---|---|---|---|---|---|
| `smoke-m1.yaml` (Qwen3-1.7B, GPU vs CPU, C 1/4/16, 50 requests, 1 repeat) | 0.6 h | $0.74 | $0.03 | $0.18 | **$0.95** |
| `dspm-smoke.yaml` (Qwen3-1.7B on two DSPM workloads, 300 test items) | 0.4 h | $0.47 | $0.02 | $0.32 | **$0.81** |
| `full.yaml` (all models and targets, 200 requests x 3 repeats) | 107 h | $245 | $5 | $289 | **$538** |
| `dspm-full.yaml` | 13 h | $37 | $1 | $55 | **$92** |

These figures leave out instance boot and image pull, up to 30 minutes of idle time before the reaper stops a
target a run did not stop, the standing NAT cost between runs, and NAT data processing for model downloads
(a 4B model is about 8 GB). In our internal tests a quick check including bring-up came to **$3–5**. The
`full.yaml` estimate predates the retirement of the OpenVINO target, which set its wall clock; in a long run
the Lambda load generators are the largest cost.

---

## 12. How the measurements are taken

| Rule | What it means |
|---|---|
| **Closed loop, fixed concurrency** | Each level runs C Lambda workers. Each worker sends its next request when the previous one ends. Workers are warmed, pre-connect, and start together at a barrier. Waves last at most 12 minutes. |
| **Measurement window** | Throughput uses the window from the barrier to the moment the first worker finishes, so all C workers are active inside it. |
| **Exact percentiles** | Over every successful request of a level, with n reported. TTFT includes queueing, prefill and network. Prefill and decode speeds are reported apart; the reports never combine them into one "N times faster" figure. |
| **Fixed output length** | In the standard suite (`ignore_eos`, fixed `max_tokens`). A record whose `completion_tokens` differs is flagged. |
| **Prompt-token parity** | Chat templates are applied client-side and every record compares `usage.prompt_tokens` with the expected count; mismatches are flagged. |
| **Prefix caching** | Off in the standard suite, on in the DSPM suite, where a long policy is shared by every request. |
| **Repeats** | Pooled for headline percentiles. Mean, standard deviation and CV across repeats are reported; CV above 10% is flagged. |
| **AMX verification** | For every CPU row: CPU flags at boot, `perf stat` `exe.amx_busy` per level, and an engine trace (oneDNN verbose, vLLM GEMM dispatch) once per cell. FP16 and W8A8 rows must show FP16 or INT8 AMX execution. Engines that run fp32 by design (the decision models' torso) are marked not applicable, not invalid. |
| **Quality** | 50 fixed prompts per model at temperature 0 against the GPU BF16 cell. The CPU BF16 score is the noise floor for quantized cells. |
| **SLA and capacity** | A row qualifies when it meets every SLA constraint with a low error rate and is not INVALID. Capacity equivalence is the number of CPU instances needed to match the best qualifying GPU row, with the Standard and G-family vCPU quota pools shown separately. Cost is reported and does not change the verdict. |
| **Why it came out this way** | Every results page carries a computed mechanics block (`bench/mechanics.py`: reading against writing share, bytes per token, bandwidth reached against bandwidth available, arithmetic against peak). The executive summary ends with a technical section built from it. |

---

## 13. The suites in detail

### Chat and text generation

Models: Qwen3-1.7B (the model Intel's Software Catalog publishes Xeon benchmarks for), Gemma 4 E4B and E2B. vLLM
on both sides, Intel's own Xeon recipe on the CPU. Reports P1/P50/P99 latency, throughput, prefill and decode
speed, cost from the Price List API, the AMX verdict and an answer-quality check against the GPU.

> [!NOTE]
> Earlier versions also offered PyTorch and OpenVINO serving on the CPU, and Flan-T5. We retired them after our
> internal tests showed them well behind vLLM on the same m8i.4xlarge (the PyTorch server 3–10x slower; OpenVINO
> BF16 2.3x slower on RoBERTa, 7.9 against 18.3 items/s). They compared serving engines, not CPU with GPU. They
> stay in the configuration disabled so that any older runs still render.

### Sensitive-data detection (DSPM / DLP)

| Workload | What it is |
|---|---|
| W1 | Bulk scan, binary `SENSITIVE` / `NOT_SENSITIVE`, ~512 and ~2,048-token chunks |
| W2 | Bulk scan, one category |
| W3 | Inline DLP, `ALLOW` / `BLOCK` (chat, e-mail, GenAI prompts, code; 64–360 tokens) |
| W4 | Structured columns, one category or `NONE` |
| W5 | Redaction (off by default) |

SLA presets: `inline_strict` (P99 at most 250 ms), `inline_standard` (P99 at most 1,000 ms), `bulk` (P99 at most
30 s, maximize documents per second).

- **Policy.** `config/dspm/policy_v1.md` defines the categories, decision rules and priority order. Its SHA-256
  goes into every request manifest and the report. The decision models (Strands Deciders) answer the policy
  question as one typed decision: yes/no at 0.5, or a choice over seven categories with the policy definitions.
- **Dataset.** `tools/dspm/build_dataset.py` mixes NVIDIA Nemotron-PII (CC BY 4.0) and Gretel
  `synthetic_pii_finance_multilingual` (Apache-2.0) at pinned revisions with a seeded generator (92 template
  families). Test set 3,000 items (W1 1,400, W3 1,200, W4 400), dev set 300; half sensitive, about a third hard
  negatives (masked values, test cards, placeholders, public data). Template families and source documents never
  cross splits. The same inputs give the same `dataset_sha256`; the build fails on any quota or leakage check.
- **Handling.** The dataset contains realistic fake secrets, card numbers and IDs that secret scanners flag.
  `data/` is in `.gitignore`; the dataset lives only on the local disk and in the bench bucket. Review
  `data/dspm/v1/review_sample.md` (100 items) before a sealed run.
- **Scoring.** Generative outputs are parsed strictly; anything that is not exactly one label is a format failure
  and is scored as positive (fail closed). Recall, precision, FPR and F1 carry Wilson 95% intervals; hard-negative
  FPR is reported per case type.
- **Verdict.** A CPU row qualifies when it meets the preset and, against the GPU row of the same model and
  precision, keeps label agreement at 98% or higher and loses at most 1 percentage point of recall. Fitness floors
  (recall at least 0.90, FPR at most 0.05) are reported separately and do not change the CPU-vs-GPU verdict.
- Speed results transfer to fine-tuned models of the same architecture and size; accuracy results do not.

### Translation

Directions `en-de`, `de-en`, `en-es`, `es-en`. Text sizes: `sentence` (1–2 sentences, 8–30 words, answer limit
192 tokens) and `paragraph` (5–7 sentences, 40–90 words, limit 448). Chat models take part; classifiers do not.

- **Corpus.** `tools/translation/generator.py` fills hand-written trilingual sentence frames (78 frames, six
  business domains) with seeded slot values rendered per language. Every item exists in all three languages from
  the same template, so each text is the reference translation of the other two. 200 items per size; nothing is
  real and no third-party text is used.
- **Plan.** Per direction: the run's concurrency levels (answers end where the model ends them, greedy) plus one
  accuracy pass at C=8 with outputs kept.
- **Scoring.** Sentence-level chrF (character n-grams 1–6, beta 2) against the reference, mean and P10 per cell
  and direction; agreement with the model's GPU BF16 cell; length ratio and the share of answers cut at the token
  limit. Read chrF relative to the other cells: a paraphrase lowers it for every cell alike.
- **Units.** Source characters per second and USD per million characters, the unit managed translation services
  bill by (Amazon Translate $15, Google $20 per million characters, list prices); the report prints that as
  context, not as like-for-like.

### Embeddings

The corpus in three languages embedded by EmbeddingGemma 300M: embeddings per second, cost per million, cosine
agreement with the GPU's vectors, cross-lingual retrieval recall@1. All prefill: the CPU's best case.

### Agent decisions

The Strands Decider models (E4B and 12B, built on Gemma 4): typed decisions (choice, yes/no, scale) with
calibrated confidence, decisions per second, accuracy against gold answers, agreement with the GPU. The 12B model
needs an L40S GPU (g6e.xlarge) and an m8i.8xlarge on the CPU side; the host check verifies GPU memory before a
run.

### Intel's benchmark, reproduced

Intel's Software Catalog method (`vllm bench serve`, 128-token prompts and answers, pass at TPOT ≤ 100 ms and
TTFT ≤ 3 s), with Intel's published figures for the same model (`config/intel_catalog_benchmarks.json`) shown per
core next to your run's.

---

## 14. Is the CPU side configured right?

A benchmark that under-configures one side proves nothing, so we checked the CPU configuration against **Intel's
own published vLLM Xeon recipe** (`tools/recipe_check.py`: same instance, same client, the recipe's container
against ours). What our internal check on 2026-11-19 found:

| Check | Result |
|---|---|
| Throughput at low concurrency | Ties the recipe |
| Throughput at 64–128 concurrent requests | 30% ahead of the recipe (the gap is the vLLM version, 0.33 against 0.23) |
| Single-stream decode | About 70% of the slice's measured memory bandwidth |
| Per-core throughput | Inside the range Intel publishes for whole two-socket servers |

The check also found and fixed a real defect: W8A8 cells had been wrongly judged AMX `INVALID`. The full record,
with numbers, is [recipe-check-2026-11-19.md](recipe-check-2026-11-19.md). You can repeat the check in your own
account with `tools/recipe_check.py`.

---

## 15. What the CPU can and cannot win

> [!NOTE]
> The ratios below are from **our internal test runs** (November 2026, `us-east-1` On-Demand prices). Your own
> Results page computes the same figures for your models and your traffic.

### Peak throughput per dollar: the GPU, every time we measured

At the same hourly price, one L4 (g6.xlarge, about $0.80/h) carries far more raw capability than the eight
Xeon 6 cores of an m8i.4xlarge (about $0.85/h):

| | L4 (g6.xlarge) | 8 Xeon 6 cores (m8i.4xlarge) | Ratio | Source |
|---|---|---|---|---|
| Dense BF16 arithmetic | 121 TFLOPS | 8 x ~2 TFLOPS (1,024 FLOPs/cycle/core) | ~7.5x | NVIDIA L4 datasheet; Intel AMX documentation |
| Memory bandwidth | 300 GB/s | 107 GB/s | ~2.7x | NVIDIA datasheet; STREAM, measured by the tool on every CPU host |

The CPU side runs at 70–80% of its own hardware limits with AMX verified (section 14), so the gap is the
hardware, not the configuration. "How many CPU instances replace one GPU" therefore favoured the GPU in every
internal test, including the workloads built for AMX:

| Workload (our internal tests) | CPU instances per GPU |
|---|---|
| Embeddings, EmbeddingGemma, batch 32 | 4.3x (2,531 against 583 embeddings/s) |
| Classification, RoBERTa-large | 7.3x |
| Inline DLP and short-document scans, Gemma 4 E4B | 14–16x |
| Chat, 3–4B models at 16 concurrent requests | about 4x ($0.49 against $1.93 per million output tokens) |
| **Long-document scans, Gemma 4 E4B** | **about 1x: the m8i.4xlarge matched a g6.2xlarge (4.5 against 3.5 documents/s), the one exception we saw** |

The Results page says this up front for each run.

### The question the CPU can win: the cheapest option that meets your targets at your load

Peak throughput is the wrong question for most deployments. The one that matters is: *given the traffic I
actually have, and the prices I actually pay, which option meets my targets for the least money?* The **At your
load** card on the Results page answers it:

| You enter | It computes |
|---|---|
| The load to serve (requests per minute; typical figures are offered to start from) | Each option's **capacity**: the highest measured request rate that still met your targets |
| The prices to use: On-Demand, a Compute or EC2 Instance Savings Plan at 1 or 3 years (published rates), or your own hourly rate per side (0 for capacity you already pay for) | **Instances needed** = load / capacity, rounded up, and the cost at your prices |
| | The **cheapest option per side**, and the **crossover load** above which the GPU becomes cheaper |
| | A cost-against-load chart |

Smaller sizes of the measured CPU family are offered as **estimates** (capacity scaled by core count, memory
checked against the model) and marked as such; one run at that size confirms them.

> **Example from our internal tests.** EmbeddingGemma at 600 requests per minute of single sentences: one
> m8i.large (estimated, about $0.08/h) against one g6.2xlarge (about $0.80/h). The GPU only becomes the cheaper
> choice above roughly 26,000 requests per minute.

### The other three places the CPU wins

- **Availability.** When AWS refuses to launch the GPU size you asked for (section 6), the CPU launches on request.
  The page counts the refusals from the run's own log.
- **The smallest unit you can buy.** A GPU costs at least about $0.80/h, idle or not; a CPU slice that fits a
  small model can cost $0.10/h.
- **Capacity you already pay for.** An existing CPU fleet with idle cycles costs close to nothing extra. Enter 0
  as the CPU's hourly rate and the card recomputes.

The executive summary is written around the same frame (`bench/demand.py`, `webui/api/compare.py`). Savings Plan
rates come from the Savings Plans API via `tools/refresh_instance_catalog.py`.

---

## 16. Reading the reports

| Label | Meaning |
|---|---|
| `SMOKE` | Fewer than 200 requests per level or fewer than 3 repeats. Fine for checking the pipeline; not a result. |
| `INVALID` | AMX not verified (flag missing, counter below threshold, or no AMX in the engine trace). The row stays in the raw table and is excluded from SLA and cost rankings. |
| `NOT_APPLICABLE` | The engine runs fp32 by design (the decision models' torso); AMX was not expected. |
| FP16 relabeled | The row was configured as FP16 but ran in BF16; it is relabeled and flagged. The report shows the precision that ran. |
| `Conditional` | The CPU row qualifies but needs more CPU instances per GPU than `max_cpu_instances_per_gpu`. |
| `below floors` | A DSPM model misses the absolute recall or FPR floor (model fitness, not a CPU-vs-GPU verdict). |
| Flags | Prompt-token mismatch, output length mismatch, CV above 10%, late workers, effective concurrency below 0.9 x C, short window (a worker ran fewer than 3 requests at C > 1), quality delta, non-commercial license. |
| Quality flag | Warning only. A compressed format whose answers diverge from the BF16 floor (ROUGE-L for chat, chrF for translation, cosine/recall@1 for embeddings, label agreement for classifiers) is flagged but stays in the SLA and cost rankings. For chat the flag measures different wording against the GPU's text, not a measured loss: in our tests two BF16 cells of the same weights agreed only 0.73–0.76. |
| Units | Per model type: tokens/s for chat, items/s for classifiers, embeddings/s, decisions/s, source characters/s for translation. |

---

## 17. Security

| Area | What is in place |
|---|---|
| **Test hosts** | Private subnet, no public IP, no key pair, IMDSv2 required, encrypted root volume. Access only through SSM. |
| **Model endpoints** | Port 8000 on the instance's private IP only; the target security group admits it only from the load-generator security group. By decision the endpoints have no API key; the network controls are the boundary. |
| **Bucket** | Blocks public access, denies non-TLS requests. |
| **Hosted web app** | CloudFront + AWS WAF + Cognito. The Fargate task has no public address and a role scoped to the bench resources. Nobody can sign up; accounts exist only when an operator creates them (README, "Sign in"). Any signed-in user may read; only members of the `operators` group may start, stop or delete runs, or change the token, the models or the reasoning model (`webui/api/auth.py`). |
| **Local web app** | Binds to loopback only. |
| **Secrets** | The Hugging Face token and the web app password are never written to the repository, to CloudFormation, to logs or to S3. The password reaches the AWS CLI through a 0600 temp file removed on exit; the token is validated with huggingface.co before it is stored and read by hosts through their instance role. |
| **Reasoning model** | Web search is its only tool; results are treated as untrusted data; proposals are validated against the allowed size lists before they can be applied. |
| **Before publication** | Scanned with ProtoShield and Holmes (content security review rubric); findings fixed or marked with a reason. |

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

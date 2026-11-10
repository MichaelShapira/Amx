# amx-gpu-bench

A benchmark harness that compares small-model inference (at most 4B parameters) on
NVIDIA GPU instances (g6/g5) with Intel AMX CPU instances (m8i/c8i, m7i as fallback) in
account `346399954218`, `us-east-1`. It has three parts:

- **Standard suite.** Five models (M1-M5) on vLLM (GPU and CPU), OpenVINO (OVMS and a thin
  server) and native PyTorch, driven by a Lambda fan-out load generator. Reports P1, P50,
  P99 latency, throughput, prefill and decode speed, cost from the Price List API, AMX
  verification and a quality check.
- **DSPM/DLP suite.** Narrow-label classification (`SENSITIVE`/`NOT_SENSITIVE`,
  `ALLOW`/`BLOCK`, one category) on a synthetic dataset built in this repo, plus two
  purpose-built detectors (D1 openai/privacy-filter, D2 Roblox PII classifier v2).
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
| `servers/thin/` | FastAPI server for models vLLM/OVMS do not serve well (Flan-T5, classifiers, D1 `/detect`), torch and OpenVINO backends |
| `servers/export/` | Pinned OpenVINO export image and `probe_ov.py` |
| `host/` | Scripts run on the targets over SSM: host facts, model staging, `serve.sh`, AMX probe, `perf`, STREAM, seccomp profile |
| `infra/` | CDK app: `AmxBench-Network`, `AmxBench-Core`, one `AmxBench-Target-<id>` per target |
| `config/` | `models.yaml`, `targets.yaml`, `migration_effort.yaml`, `runs/*.yaml`, `dspm/` policy and span map |
| `tools/` | Prompt-set builder, DSPM dataset and request-pool builders, seccomp generator |
| `ui/` | Local Streamlit app |
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

Tests that need torch, moto, streamlit or shellcheck skip when the package is missing.

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

The token is stored as the Secrets Manager secret `amx-gpu-bench/hf-token` (plain string).
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
  `targets.yaml` (g6.xlarge → g5.xlarge, g6.2xlarge) for up to 30 minutes, then fails that target's cells and the run
  continues with the other targets. Every attempt is logged under `runs/<id>/capacity/` and
  listed in the report; the report rows carry the type that ran. The whole deployment is
  pinned to one AZ (`use1-az1`) so workers and targets share it; when EC2 says the type is
  available in another AZ, that is a redeploy of the network stack, not a run option.
- More targets (`t2-c8i-4xl`, `t2-m8i-8xl`, `t3-m8i-4xl-ov`, `t4-m8i-4xl-torch`) are
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
`bench:target` tag. Spot instances are one-time requests that terminate on interruption.
When a cell step fails, the `Check` step looks at the instance:

- instance gone (interrupted, stopped, terminated): record the interruption, launch another
  instance (Spot again while the 10-minute `spot_capacity_window_s` lasts, then On-Demand if
  `spot_fallback_on_demand` is on), re-serve the model, and run only the cell's remaining
  levels; a level cut short is re-run from zero and its partial raw records are dropped;
- instance healthy and a level failed: mark that level failed and continue with the rest;
- bring-up failed on a healthy instance, or more than `max_relaunches_per_cell` (3)
  relaunches: the cell fails.

The report's provenance block lists the instances that served each target, their market,
and every interruption; a warning line names the targets that were interrupted. Cost
columns always use On-Demand prices. At the end of a run (`stop_targets_after_run`) and
from the reaper, On-Demand instances are stopped and Spot instances are terminated.

## Prepare artifacts

```bash
.venv/bin/python tools/build_prompts.py --hf-token-secret amx-gpu-bench/hf-token           # standard prompt sets
.venv/bin/python -m tools.dspm.build_dataset --out data/dspm/v1 --hf-token-secret amx-gpu-bench/hf-token
.venv/bin/python -m tools.dspm.build_requests --dataset data/dspm/v1 --hf-token-secret amx-gpu-bench/hf-token
.venv/bin/bench upload-artifacts
```

`bench upload-artifacts` copies the prompt sets, the DSPM request pools, and the DSPM
dataset (`dev`, `test`, `manifest.json`) to the bucket. Run it again after any rebuild:
preflight refuses a DSPM run whose request pools were built from a different dataset than
the uploaded one.

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

The UI uses the same YAML and the operator's own credentials, and binds to `127.0.0.1`
only:

```bash
cd ui && ../.venv/bin/streamlit run app.py
```

Pages: New run (YAML preview, plan, estimate, start), Runs (progress, stop with
confirmation), Results (table, SLA filters, capacity equivalence, charts, CSV and Markdown
export, dataset and policy hashes), Migration calculator, Bring-up journal.

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

## Reading the reports

| Label | Meaning |
|---|---|
| `SMOKE` | Fewer than 200 requests per level or fewer than 3 repeats. Fine for checking the pipeline; not a result. |
| `INVALID` | AMX not verified (flag missing, counter below threshold, or no AMX in the engine trace). The row stays in the raw table and is excluded from SLA and cost rankings. |
| FP16 relabeled | The row was configured as FP16 but ran in BF16; it is relabeled and flagged. |
| `Conditional` | The CPU row qualifies but needs more CPU instances per GPU than `max_cpu_instances_per_gpu`. |
| `below floors` | A DSPM model misses the absolute recall or FPR floor (model fitness, not a CPU-vs-GPU verdict). |
| Flags | Prompt-token mismatch, output length mismatch, CV above 10%, late workers, effective concurrency below 0.9 x C, quality delta, non-commercial license (M1). |

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

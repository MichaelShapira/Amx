# Design: CPU (Intel AMX) vs GPU inference benchmark harness and migration effort calculator

Status: revision 2, after design review (2026-11-09). Inputs:
[requirements.md](requirements.md), [research.md](research.md). Section 23 records the
review decisions and the defaults kept for questions that were not answered.

Changes in revision 2: deploy to account `346399954218`; models capped at 4B (M2 is now
Qwen3-4B-Instruct-2507, the 7B model is gone); M3 is the classifier roberta-large-mnli;
verdicts are decided by capacity, not cost; vLLM INT8 W8A8 and classifier FP16 cells are
added and prefill is reported separately, based on the two Intel case studies
(research.md section 13).

Changes in revision 3: a DSPM/DLP suite with a synthetic labeled dataset (section 24),
next to the standard suite; two DLP detector models (D1 openai/privacy-filter, D2 Roblox
PII classifier v2); the existing HF token is reused; no AMX-off run.

## 1. Overview

The harness has four parts:

1. **Inference targets.** One EC2 instance per target, each in its own CDK stack, serving
   OpenAI-compatible HTTP on port 8000 in a private subnet.
2. **Load generator.** A Step Functions state machine fans out N worker Lambdas per wave
   through a Distributed Map. Workers start together at a shared timestamp, run a closed
   request loop, and write per-request records to S3.
3. **Analysis.** An aggregator Lambda computes exact percentiles, throughput, cost and
   verdicts (AMX, quality, SLA) and writes CSV, Markdown and JSON reports.
4. **Local UI.** A Streamlit app on 127.0.0.1 builds run configs, starts runs, shows
   progress and results, and hosts the migration calculator and bring-up journal.

Everything a run needs is in one YAML file. The resolved form of that file (digests,
AMI IDs, revisions, prices, engine arguments) is stored next to the results.

## 2. Key decisions

| Topic | Choice | Why | Alternatives |
|---|---|---|---|
| IaC | CDK in Python | Same language as Lambdas, servers and UI | CDK TypeScript |
| Fan-out | Step Functions Standard + Distributed Map, child executions Standard | Inline Map caps at 40 concurrent iterations; Express children stop at 5 minutes; runs last hours | Orchestrator Lambda alone (15-minute limit) |
| Cold starts | Warm-up wave, then a shared start timestamp 20 s in the future; timing starts inside the handler after the barrier | Init time cannot leak into measurements; no provisioned-concurrency cost | Provisioned concurrency (config option) |
| Clocks | Monotonic clock for durations; wall clock only for the barrier and for placing requests on a shared timeline | Durations are immune to clock adjustments | Wall clock only |
| Prompt input | Chat templates applied client-side; rendered text sent to `/v1/completions` | Identical token input on every engine, checkable through `usage.prompt_tokens` | Server-side chat templates (differ between vLLM and OVMS) |
| Output length | `ignore_eos: true` with fixed `max_tokens` for performance runs; natural stop for quality runs | Throughput comparable across engines and precisions | Natural stop everywhere |
| Raw storage | S3, gzip JSONL, one object per worker per wave | Cheap, append-free writes, readable by any tool | DynamoDB items per request |
| Aggregation | Python Lambda with numpy, exact percentiles | A run has at most ~10^5 rows; Athena's `approx_percentile` is approximate | Athena |
| Run state | One DynamoDB table (runs, levels, target activity, journal) | Atomic counters for live progress | Step Functions history only |
| UI | Streamlit, local, bound to 127.0.0.1, operator's own AWS credentials | Simplest option; no hosted auth to build | React on CloudFront (needs a Midway gate, deferred) |
| Instance control | SSM Run Command calling scripts installed at boot | No extra listener on targets; every action is logged | Agent process on the instance |
| Model files | Download the pinned revision once, cache in S3, sync to instances | Reproducible, faster re-deploys, less NAT traffic | Download from HF on every start |
| Egress | One NAT gateway in the benchmark AZ | HF Hub, Docker Hub and public ECR need internet | VPC endpoints + ECR pull-through cache |
| GPU engine | AWS DLC vLLM 0.33.0 | DLC preferred; same vLLM version as CPU | Upstream `vllm/vllm-openai:v0.33.0` |
| CPU vLLM | `vllm/vllm-openai-cpu:v0.33.0-x86_64` | No DLC exists | Build from source |
| OpenVINO | OVMS 2026.4.1 for decoders; optimum-intel inside the thin server for Flan-T5 and the classifier | OVMS has no seq2seq servable and no classification endpoint | ONNX Runtime |
| Plain PyTorch | Thin FastAPI server on DLC PyTorch 2.14 (CPU and CUDA builds) | The "before" stack most customers migrate from; also the only GPU path for Flan-T5 | TGI (not used) |
| Classifier API | vLLM's `/classify` shape on every classifier target | OpenAI has no classification endpoint; vLLM already serves this shape on GPU and CPU | Custom shape |
| INT8 on CPU | Two engines: vLLM W8A8 (INT8 activations) on T2 and OpenVINO weight-only INT8 on T3 | OpenInfer saw 2.7-3.1x prefill differences between engines at the same INT8 family | One engine only |
| Verdict driver | Capacity: does CPU meet the SLA, and how many CPU instances replace one GPU instance | The narrative is about G5/G6 capacity; hourly prices are near parity | Cost-gated verdict |

## 3. Architecture

```mermaid
flowchart LR
  subgraph laptop["Operator laptop"]
    UI["Streamlit UI (127.0.0.1)"]
    CLI["bench CLI"]
  end
  subgraph acct["AWS account, one region"]
    subgraph ctl["Control plane (no VPC)"]
      SFN["Run state machine"]
      LVL["Level state machine"]
      CTL["Control Lambdas: preflight, target control, probes, aggregate"]
      REAP["Reaper Lambda (every 10 min)"]
    end
    subgraph vpc["VPC, one AZ, private subnet"]
      W["Worker Lambdas x N (Distributed Map)"]
      Q["Warm-up and quality Lambdas"]
      T1["T1 g6.xlarge: vLLM GPU / thin CUDA"]
      T2["T2 m8i, c8i 4xl, m8i 8xl: vLLM CPU"]
      T3["T3 m8i.4xl: OVMS / thin OpenVINO"]
      T4["T4 m8i.4xl: thin PyTorch CPU"]
      T5["T5 m7i.4xl: vLLM CPU"]
      NAT["NAT gateway"]
    end
    S3[("S3: models cache, raw records, evidence, reports")]
    DDB[("DynamoDB: runs, progress, activity, journal")]
    PRICE["Price List API"]
    CW["CloudWatch metrics"]
    SM["Secrets Manager: HF token"]
  end
  HF["HF Hub, Docker Hub, public ECR"]
  UI -->|SigV4| SFN
  UI -->|SigV4| S3
  UI -->|SigV4| DDB
  CLI -->|SigV4| SFN
  SFN --> CTL
  SFN --> LVL
  LVL --> W
  LVL --> Q
  W -->|HTTP 8000| T1
  W -->|HTTP 8000| T2
  W -->|HTTP 8000| T3
  W -->|HTTP 8000| T4
  W -->|HTTP 8000| T5
  W --> S3
  W --> DDB
  CTL -->|SSM Run Command| T2
  CTL --> PRICE
  CTL --> CW
  REAP -->|StopInstances| T1
  T2 --> NAT --> HF
  T2 --> SM
```

## 4. Repository layout

New standalone repo at `/Users/operator/Documents/Code/amx-gpu-bench` (hosting is still
open; see section 23).

```
amx-gpu-bench/
  config/
    models.yaml               model catalog (Requirement 1)
    targets.yaml              target catalog (Requirement 2)
    migration_effort.yaml     calculator rules (Requirement 16)
    runs/                     example run configs (smoke-m1.yaml, full-m1-m5.yaml)
  prompts/                    generated, versioned prompt sets + manifest with checksums
  bench/                      shared Python package: config schemas (pydantic), metrics,
                              SSE client, cost, AMX verdict, quality scoring, estimator
  lambdas/                    preflight, target_control, probes, warmup, plan_wave,
                              worker, wave_check, perf_counters, aggregate_level,
                              quality, aggregate_run, reaper
  servers/thin/               FastAPI thin server (torch and openvino backends) + Dockerfiles
  servers/export/             OpenVINO export container (pinned optimum-intel stack)
  host/                       scripts installed on targets: serve.sh, stage_model.sh,
                              probe_amx.sh, probe_ov.py, probe_membw.sh, host_facts.sh,
                              seccomp-vllm.json, cloudwatch-agent.json
  infra/                      CDK app: NetworkStack, CoreStack, TargetStack (one per target)
  ui/                         Streamlit app
  tools/                      build_prompts.py, render_report.py
  tests/                      unit, contract and CDK assertion tests
  reports/sample/             smoke report (Requirement 20)
  README.md
```

## 5. Configuration

All files are validated with pydantic models in `bench/config.py`. A run is refused if any
reference (model, target, precision, prompt set) does not resolve.

### 5.1 `config/models.yaml` (excerpt)

```yaml
models:
  - id: m1-qwen2.5-3b
    slot: M1
    hf_repo: Qwen/Qwen2.5-3B-Instruct
    revision: aa8e72537993ba99e69dfaafa59ed015b17504d1
    type: decoder
    license: qwen-research          # non-commercial; reports show a warning
    gated: false
    load_dtype: bfloat16
    params: 3085938688
    nominal_size_b: 3
    max_input_tokens: 4096
    max_model_len: 8192
    prompt_format: {kind: chat_template}
    quality_reference: {target: t1-g6, precision: bf16}
    precisions:
      bf16:    {source: base}
      w8a8:    {engine: vllm, hf_repo: RedHatAI/Qwen2.5-3B-Instruct-quantized.w8a8,
                revision: cd4a8836f6c1e916f43d7c60c4dc3796b7aca18c}
      int8-ov: {engine: openvino, export: {weight_format: int8}}
      int4-ov: {engine: openvino, export: {weight_format: int4, group_size: 128, ratio: 1.0, sym: false}}  # proposal
      awq-int4: {engine: vllm, hf_repo: Qwen/Qwen2.5-3B-Instruct-AWQ,
                 revision: 3559b226e8ce77211e2c1bd7ddfb7686fec4d6dd, enabled: false}
  - id: m2-qwen3-4b-2507
    slot: M2
    hf_repo: Qwen/Qwen3-4B-Instruct-2507
    revision: cdbee75f17c01a7cc42f958dc650907174af0554
    type: decoder
    license: apache-2.0
    params: 4022468096
    nominal_size_b: 4
    max_input_tokens: 4096           # 14K "xlong" set allowed (context >= 16K)
    max_model_len: 16384
    prompt_format: {kind: chat_template}
    precisions:
      bf16:    {source: base}
      w8a8:    {engine: vllm, hf_repo: RedHatAI/Qwen3-4B-Instruct-2507-quantized.w8a8,
                revision: 31ec5013b3319d7ba82b2a354d6e48b376204dd3}
      int8-ov: {engine: openvino, export: {weight_format: int8}}
      int4-ov: {engine: openvino, export: {weight_format: int4, group_size: 128, ratio: 1.0, sym: false}}
  - id: m3-roberta-large-mnli
    slot: M3
    hf_repo: FacebookAI/roberta-large-mnli
    revision: 2a8f12d27941090092df78e4ba6f0928eb5eac98
    type: encoder-classifier
    license: mit
    gated: false
    params: 356412419
    nominal_size_b: 0.36
    load_dtype: bfloat16
    max_input_tokens: 512            # prompt sets: 128 and 512 tokens
    prompt_format: {kind: pair, template: "{premise}</s></s>{hypothesis}"}   # tokenizer pair encoding
    classifier: {batch_sizes: [1, 8, 32, 64], labels: [CONTRADICTION, NEUTRAL, ENTAILMENT]}
    quality_reference: {target: t1-g6, precision: bf16, fp32_anchor: {target: t4-m8i-4xl-torch, precision: fp32}}
    precisions:
      bf16:    {source: base}
      fp16:    {source: base, requires_cpu_flag: amx_fp16}       # T3 (f16 hint) and T4 only
      fp32:    {source: base, quality_only: true}
      int8-ov: {engine: openvino, export: {weight_format: int8}}
  - id: m5-flan-t5-xl
    slot: M5
    hf_repo: google/flan-t5-xl
    revision: 7d6315df2c2fb742f0f5b556879d730926ca9001
    type: encoder-decoder
    license: apache-2.0
    gated: false
    load_dtype: bfloat16             # avoid FP16 overflow risk
    max_input_tokens: 1024           # long (4K) prompt set skipped for T5
    prompt_format: {kind: plain}
```

Alternatives inside the cap (ModernBERT-large-zeroshot-v2.0, Qwen2.5-1.5B-Instruct,
flan-t5-large) are present with `enabled: false`. The ModernBERT entry lists no OpenVINO
precisions because optimum-intel 2.2.0 refuses to export ModernBERT. Models above the cap
(Qwen2.5-7B, gemma-4-E2B, gemma-3n-E2B, Qwen3.5-4B) are not in the catalog; config
validation rejects `nominal_size_b > 4`.

Revision 3 adds two DSPM/DLP-only entries (section 24): `d1-privacy-filter`
(`openai/privacy-filter`, revision `7ffa9a043d54d1be65afb281eddf0ffbe629385b`, type
`token-classifier`, 1.40B total, BF16) and `d2-roblox-pii-v2`
(`Roblox/roblox-pii-classifier-v2`, revision `44a84be3eba4859a7e2a1f7b9cee8df61131f28b`,
type `encoder-classifier`, multi-label, 512 tokens). Every model entry gains
`suites: [standard, dspm]`.

### 5.2 `config/targets.yaml` (excerpt)

```yaml
targets:
  - id: t1-g6
    label: T1
    instance_type: g6.xlarge
    alternates: [g5.xlarge]
    ami: {ssm: /aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-amazon-linux-2023/latest/ami-id}
    root_volume_gib: 200
    engines:
      vllm: {image: "public.ecr.aws/deep-learning-containers/vllm@sha256:6db6c433557b38d86d47f68e367532d41a8341e56fa2605e60c13d190a5c7d17",
             args: {gpu_memory_utilization: 0.90}}
      thin-torch: {image: "<account ECR>/bench-thin:torch-cuda-<git sha>"}
  - id: t2-m8i-4xl
    label: T2
    instance_type: m8i.4xlarge
    ami: {ssm: /aws/service/ami-amazon-linux-latest/al2023-ami-kernel-6.12-x86_64}
    engines:
      vllm: {image: "docker.io/vllm/vllm-openai-cpu@sha256:4154467a886f8b7ffa9dbab32f873f5ca4528409d042a6ec866e8888223263d3",
             env: {VLLM_CPU_OMP_THREADS_BIND: auto, VLLM_CPU_NUM_OF_RESERVED_CPU: "1"},
             args: {kv_cache_gib: 32, block_size: 128}}
  - {id: t2-c8i-4xl, label: T2, instance_type: c8i.4xlarge, inherits: t2-m8i-4xl, engines: {vllm: {args: {kv_cache_gib: 16}}}}
  - {id: t2-m8i-8xl, label: T2-scale, instance_type: m8i.8xlarge, inherits: t2-m8i-4xl, engines: {vllm: {args: {kv_cache_gib: 64}}}}
  - id: t3-m8i-4xl-ov
    label: T3
    instance_type: m8i.4xlarge
    engines:
      ovms: {image: "docker.io/openvino/model_server@sha256:148e3fe802261ed0b5b634de3bbccdefe7176a218ec34a88e382b40704c34a1c"}
      thin-openvino: {image: "<account ECR>/bench-thin:openvino-<git sha>"}
  - id: t4-m8i-4xl-torch
    label: T4
    instance_type: m8i.4xlarge
    engines:
      thin-torch: {image: "<account ECR>/bench-thin:torch-cpu-<git sha>",
                   env: {OMP_NUM_THREADS: "8", OMP_PROC_BIND: close, OMP_PLACES: cores}}
  - {id: t5-m7i-4xl, label: T5, instance_type: m7i.4xlarge, inherits: t2-m8i-4xl, enabled: false}
```

KV cache sizes are per target and checked at preflight against
`memory - weights - 6 GiB reserve` for the largest model the run places there.

### 5.3 Run config (`config/runs/smoke-m1.yaml`)

```yaml
run:
  name: smoke-m1-t1-t2
  mode: smoke                 # "full" requires >= 200 requests and >= 3 repeats
  account: "346399954218"
  region: us-east-1
  az_id: use1-az1
  models: [m1-qwen2.5-3b]
  cells:
    - {target: t1-g6, precision: bf16}
    - {target: t2-m8i-4xl, precision: bf16}
  prompt_set: short           # short | medium | long | custom
  custom_prompt: null
  sampling: {temperature: 0.0, max_new_tokens: 128, ignore_eos: true, seed: 0}
  concurrency: [1, 4, 16]
  per_level: {requests: 50}   # or {duration_s: 600}
  repeats: 1
  warmup_requests: 20
  request_timeout_s: 600
  quality: {enabled: true, prompt_set: quality-50, max_new_tokens: 256}
  sla: {p99_ttft_ms: 2000, p50_tpot_ms: 150, max_error_pct: 1.0}
  capacity: {max_cpu_instances_per_gpu: 4}    # beyond this the verdict is Conditional (proposal)
  stop_targets_after_run: true
```

The UI writes exactly this format. The CLI is `bench run config/runs/smoke-m1.yaml`.

## 6. Network and security

Applied from the builder-security policy: private by default, least privilege, no
invented authentication, no public storage.

- **VPC.** One private subnet in the benchmark AZ (targets and in-VPC Lambdas), one
  public subnet for the NAT gateway. Gateway endpoints for S3 and DynamoDB.
- **Security groups.** `sg-workers`: egress TCP 8000 to `sg-targets`, 443 to the S3 and
  DynamoDB prefix lists. `sg-targets`: ingress TCP 8000 from `sg-workers` only; egress
  TCP 443 (model and image downloads through NAT). No other ingress, no `0.0.0.0/0`
  ingress anywhere.
- **Instances.** No public IP, no key pair, IMDSv2 required, encrypted gp3 root volume.
  Containers publish to the instance's private IP only (`-p <private-ip>:8000:8000`); the
  in-container `0.0.0.0` bind is behind that publish and the security group.
- **Operator access.** SSM Session Manager. For ad-hoc testing, the operator uses
  `AWS-StartPortForwardingSession` to reach a target on `localhost`.
- **Inference endpoints carry no API key.** They are reachable only from the worker
  security group inside a private subnet, so the network is the boundary. If you want
  defense in depth, vLLM `--api-key` can be fed from Secrets Manager; OVMS has no
  equivalent. Kept as the default (section 23).
- **Account.** Account `346399954218`, us-east-1. `CDKToolkit` is already bootstrapped
  there (bootstrap version 32). CDK deploys use the bootstrap roles; run-time reads use
  the least-privilege roles below.
- **IAM.**
  - Instance role: SSM managed-instance core, CloudWatch agent put-metric, S3 read on
    `artifacts/*` and `models/*`, S3 write on `models/*` (staging), `hosts/*`,
    `runs/*/bringup/*` and `runs/*/amx/*`, `secretsmanager:GetSecretValue` on the HF token
    secret ARN only.
  - Worker, warm-up and quality Lambdas: `s3:PutObject` on `runs/*/raw/*`,
    `runs/*/quality/*`; `dynamodb:UpdateItem` on the table; VPC ENI permissions.
  - Control Lambdas: `ec2:StartInstances`/`StopInstances` conditioned on tag
    `bench:managed=true`, `ec2:Describe*`, `ssm:SendCommand` limited to the bench SSM
    documents and tagged instances, `ssm:GetCommandInvocation`, `pricing:GetProducts`,
    `pricing:ListPriceLists`, `lambda:GetAccountSettings`, `cloudwatch:GetMetricData`,
    `states:StartExecution` on the level state machine, S3 and DynamoDB on bench paths,
    `secretsmanager:GetSecretValue` on the HF token (preflight gating check only).
- **Secrets.** The HF token secret exists outside CDK:
  `arn:aws:secretsmanager:us-east-1:346399954218:secret:amx-gpu-bench/hf-token-naqygX`
  (created 2026-11-09, default `aws/secretsmanager` key, project tags). The value is the
  raw token string. CDK imports it by name and never puts a value in a template. Scripts
  read it into memory, pass it to downloads through the environment, and redact it from
  logs.
- **S3.** Block Public Access on (all four settings), TLS-only bucket policy, SSE-S3.
- **UI.** `ui/.streamlit/config.toml` sets `server.address = "127.0.0.1"`; the app uses the
  operator's AWS profile. No hosted UI in this phase.
- **Golden Path.** The Recommendation Engine tools were not available in this session,
  so no Golden Path guidance is cited.

## 7. Inference targets

### 7.1 Engine matrix

| Model | T1 GPU (g6.xlarge) | T2 vLLM CPU (m8i/c8i 4xl, m8i 8xl) | T3 OpenVINO (m8i.4xl) | T4 PyTorch CPU (m8i.4xl) | T5 vLLM CPU (m7i.4xl) |
|---|---|---|---|---|---|
| M1 (Qwen2.5-3B), M2 (Qwen3-4B-2507) | vLLM BF16 | BF16 on all three sizes; **W8A8 INT8 on m8i.4xl**; AWQ INT4 optional (off) | OVMS BF16 (`bf16-ov`: FP16 IR, run in BF16 by the CPU plugin; the like-for-like baseline on T3), INT8, INT4 weight compression | optional BF16 | BF16 |
| M4 (gemma-2-2b) | vLLM BF16 | BF16; **W8A8 INT8 on m8i.4xl** (not on vLLM validated list) | OVMS BF16 (`bf16-ov`), INT8, INT4 | optional BF16 | BF16 |
| M3 (roberta-large-mnli) | vLLM `/classify` BF16; thin torch CUDA BF16 optional | vLLM `/classify` BF16 | thin + `OVModelForSequenceClassification`: **FP16** (f16 hint), BF16, INT8 | thin torch: BF16, **FP16**; FP32 for the quality anchor only | vLLM `/classify` BF16 |
| M5 (flan-t5-xl) | thin torch CUDA BF16 | not supported | thin + `OVModelForSeq2SeqLM` INT8, INT4 | thin torch BF16 | not supported |

GPU W8A8 is available as an optional cell (off by default). The m8i.8xlarge scaling row
runs M1 and M2 in BF16 only. T5 is disabled by default. vLLM on m7i (Sapphire Rapids) is
outside vLLM's validated hardware list (Xeon 6 and Xeon 5); results are still valid if
AMX verifies. FP16 cells are refused at preflight on any target without `amx_fp16`.

### 7.2 Common vLLM arguments (GPU and CPU)

```
vllm serve /models/<id> --served-model-name <id> --dtype bfloat16
  --max-model-len <model max_model_len> --max-num-seqs 128 --max-num-batched-tokens 2048
  --no-enable-prefix-caching --seed 0 --port 8000
GPU only: --gpu-memory-utilization 0.90
CPU only: --kv-cache-memory-bytes <kv_cache_gib x 2^30> --block-size 128
W8A8 cells: --model points at the pinned compressed-tensors checkpoint (dtype stays bfloat16
  for the unquantized parts)
Classifier: --runner pooling (served on /classify)
```

The same values are used on both platforms where they mean the same thing. Flag names
are checked against `vllm serve --help` of the pinned image at implementation time and
recorded in results.

### 7.3 vLLM CPU container

```
docker run -d --name bench-engine --shm-size 4g
  --cap-add SYS_NICE --security-opt seccomp=/opt/bench/seccomp-vllm.json
  -p <private-ip>:8000:8000 -v /opt/models:/models:ro
  -e VLLM_CPU_OMP_THREADS_BIND=auto -e VLLM_CPU_NUM_OF_RESERVED_CPU=1
  vllm/vllm-openai-cpu@sha256:4154... <args above>
```

`seccomp-vllm.json` is Docker's default profile plus `migrate_pages`, the tighter option
the vLLM docs describe instead of `seccomp=unconfined`. The image already preloads
tcmalloc and Intel OpenMP. The effective thread binding is read from the startup log and
stored with the cell. `VLLM_CPU_SGL_KERNEL` is not set (not defined in v0.33.0).

### 7.4 OpenVINO

- **Export.** `servers/export` is a pinned image (python 3.12, `optimum-intel==2.2.0`,
  `openvino==2026.4.1`, `openvino-tokenizers==2026.4.1.0`, `nncf` pinned,
  `transformers` 5.5.x because optimum-intel requires `<5.6`, `torch==2.14.0+cpu`).
  It runs on the T3 instance and writes IRs to
  `s3://<bucket>/models/ov/<repo>/<revision>/<precision>/<export-hash>/`. The export
  arguments and tool versions are hashed into the path, so a changed setting never reuses
  an old IR. Export time is logged as a bring-up phase.
- **Classifier (M3).** Exported with `--task text-classification` (FP16 IR for the FP16 and
  BF16 cells, `--weight-format int8` for INT8) and served by the thin server's openvino
  backend. FP16 cells compile with `INFERENCE_PRECISION_HINT=f16`; BF16 cells with `bf16`.
  The probe confirms the runtime precision, so a cell labeled FP16 that actually ran in
  BF16 is caught.
- **Decoders on OVMS.** The IR is exported by `optimum-cli export openvino --task
  text-generation-with-past --weight-format int8|int4 ...` run directly (optimum cannot infer
  the task for a local model directory, and the export runs offline; OVMS's `export_model.py`
  omitted `--task`, which failed every decoder export until 2026-11-12), tokenizer and
  detokenizer converted alongside, then OVMS 2026.4.1's `export_model.py` renders the
  `graph.pbtxt` for the exported model (`models_path` normalized to `./`): `device CPU`, `max_num_seqs 128`,
  `max_num_batched_tokens 2048`, `enable_prefix_caching false`, `cache_size` per target.
  The route (`/v1` or `/v3`) is probed at bring-up and pinned in the target record.
- **Precision labels.** OVMS cells are labeled by weights and activations, for example
  `W:int4-g128 A:bf16` or `A:int8-dyn` when dynamic quantization is active. The probe in
  section 14 reads the actual runtime precision.

### 7.5 Thin server (`servers/thin`)

Used for M5 everywhere, for M3 on T3/T4 (and optionally on T1 as a PyTorch GPU classifier
row), and optionally for decoders on T4.

- FastAPI + uvicorn, one process. Inference runs on one dedicated thread; the event loop
  handles HTTP.
- Endpoints: `GET /health`, `GET /v1/models`, `POST /v1/completions` (stream and
  non-stream, `stream_options.include_usage`, `ignore_eos` implemented as
  `min_new_tokens = max_tokens`), `POST /classify` with vLLM's request and response shape
  (`input` as string or list; `data[i].label`, `data[i].probs`; `usage.prompt_tokens`).
- Backends: `torch` (CPU or CUDA; BF16, FP16 or FP32 weights; `torch.inference_mode`) and
  `openvino` (`OVModelForSeq2SeqLM`, `OVModelForSequenceClassification`).
- Batching: a micro-batcher collects up to `max_batch` requests (default 16) or
  `max_wait_ms` (default 5 ms), then runs the batch to completion. Generation streams
  tokens per sequence through a custom multi-row streamer (Hugging Face `BaseStreamer`
  receives one token per row each step). New requests wait for the next batch. This is
  static batching, and it is the same code on GPU and CPU, so the M5 comparison is fair.
  No engine we use offers continuous batching for T5.
- CPU threads: `OMP_NUM_THREADS=8` (physical cores of a 4xlarge), `OMP_PROC_BIND=close`,
  `OMP_PLACES=cores`. One sweep of 8 vs 16 threads on M1 decides the final value.
  `ONEDNN_MAX_CPU_ISA` is left unset.
- Images: `bench-thin:torch-cpu` (from DLC `pytorch:2.14.0-cpu-amzn2023`),
  `bench-thin:torch-cuda` (from DLC `pytorch:2.14.0-cu133-amzn2023`) and
  `bench-thin:openvino` (from the export image's pinned stack). Built as CDK Docker
  assets into a private ECR repo.

## 8. Target lifecycle

1. **Deploy.** `cdk deploy BenchTarget-<id>` creates the instance with tags
   `bench:managed=true`, `bench:target=<id>`, and user data that installs Docker, `perf`
   and the CloudWatch agent, copies `/opt/bench` scripts from S3, pulls images by digest,
   and uploads host facts (`lscpu`, `uname -r`, `numactl -H`, `nvidia-smi -q` on GPU) to
   `hosts/<id>/<instance-id>-<boot-time>.json`. Preflight copies the latest facts into
   `runs/<run>/hosts/`. The instance ID is written to SSM parameter
   `/amx-gpu-bench/targets/<id>`. The AMI ID resolved at deploy time is a stack output and
   is recorded with every result.
2. **Configure a cell.** The target-control Lambda calls
   `serve.sh start <engine> <model> <precision> <args-file>` through SSM. The script stops
   any running bench container, syncs model files from the S3 cache (or downloads the
   pinned revision from HF with the token and fills the cache), starts the container,
   waits for `/health` and a 1-token request, and writes phase timings
   (`image_pull_s`, `model_fetch_s`, `export_s`, `load_to_healthy_s`,
   `first_request_s`) to `runs/<run>/bringup/<cell>.json`.
3. **Probe.** CPU cells run the AMX probe (section 14) and, once per instance type, the
   memory-bandwidth probe (STREAM triad, threads = physical cores; proposal).
4. **Measure.** Levels run (section 9). Targets in one run measure in parallel; cells on
   one target run one after another.
5. **Stop.** At run end, instances stop if `stop_targets_after_run` is true. Otherwise the
   reaper (EventBridge, every 10 minutes) stops instances whose `last_activity_at` in
   DynamoDB is older than `idle_minutes` (default 30) and that have no active run, and
   stops any instance running longer than `max_uptime_hours` (default 8) with no active
   run. Workers update `last_activity_at` once per wave.
6. **Destroy.** `cdk destroy BenchTarget-<id>`. The S3 model cache survives in the core
   stack.

Capacity: if `StartInstances` fails with `InsufficientInstanceCapacity`, the control
Lambda retries for up to 10 minutes, then reports the failure. Switching AZ or type means
redeploying the target stack with the alternate (`-c type=g5.xlarge`); the run records
what was used.

## 9. Load generator

### 9.1 State machines

**Run state machine** (Standard):

1. `Preflight`: validate config; resolve image digests, AMI IDs and model revisions;
   check Lambda unreserved concurrency >= max concurrency + 8; check instance offerings
   in the AZ; check HF access for gated models; refuse FP16 cells on targets whose host
   facts lack `amx_fp16`; refuse models above the size cap; take the price snapshot
   (section 12); write `run.yaml` and `resolved.json`; create the run item.
2. `Map` (inline) over targets, in parallel. Per target: `EnsureRunning` (start, wait
   for SSM online; every `InsufficientInstanceCapacity` or similar error is appended to
   `runs/<run>/capacity/<target>.json` with time, AZ and type, as evidence for the
   capacity narrative), then `Map` (inline, concurrency 1) over the target's cells:
   `ConfigureCell`, `AmxProbe` (CPU), `Quality` (if enabled), then `Map` (inline,
   concurrency 1) over levels x repeats, each a `StartExecution.sync:2` of the level
   state machine.
3. `AggregateRun`: build the summary, verdicts and reports; mark the run complete.
4. A `Catch` on every branch records the failure on the cell or level and lets other cells
   continue. A final cleanup step stops `perf` collection and, if configured, the instances.

**Level state machine** (Standard), input: endpoint, model, precision, C, quota or
duration, prompt-set reference, sampling, repeat index:

1. `Warmup`: one in-VPC Lambda sends `warmup_requests` (default 20) at concurrency
   `min(C, 8)` with asyncio; results go to `warmup/` and are excluded.
2. `PerfStart` (CPU): SSM starts `perf stat -a -x, -e exe.amx_busy,cycles -o ...` in the
   background.
3. Wave loop:
   - `PlanWave`: remaining quota from DynamoDB; `start_at = now + 20 s`; per-worker quotas
     that sum to the remainder; items `[{worker: k, quota: q_k, ...}]`.
   - `PrewarmWave` (first wave of a level only): Distributed Map over C items with
     `mode: warm`; each worker sleeps 3 s so C environments exist at the same time.
   - `MeasureWave`: Distributed Map, `MaxConcurrency = C`, items from `PlanWave`, child
     execution type Standard, `ToleratedFailurePercentage: 0`.
   - `WaveCheck`: if requests remain and waves < `max_waves`, loop.
4. `PerfStop` (CPU): stop `perf`, upload the CSV.
5. `AggregateLevel`: compute level metrics (section 11), pull CloudWatch metrics, write
   `metrics/<cell>/c<C>/r<repeat>.json`, update progress.

A failed wave is retried once with a new barrier. A second failure fails the level.

Each level runs as its own nested execution so the run execution's history stays well
under the 25,000-entry limit of inline Map, however many levels and waves a run has.

### 9.2 Worker (Python, arm64, 1769 MB = one vCPU, timeout 15 min)

```
on cold init: import httpx, load prompt set from S3 (cached per environment), COLD = True
handler(event):
  if event.mode == "warm": sleep(3); return
  client = httpx.Client(http1, keep-alive, timeout=request_timeout_s)
  client.get(endpoint + health)                       # connection opened before the barrier
  sleep until event.start_at (wall clock)
  late_ms = now_wall - start_at
  wall0, mono0 = time.time_ns(), time.perf_counter_ns()   # timeline anchor
  for i in range(quota):                              # or until start_at + duration
      stop if remaining Lambda time < request_timeout_s + 30 s
      prompt = prompts[(worker + i * C) % len(prompts)]    # deterministic
      record = send_streaming(prompt)                 # see 9.3
      buffer.append(record)
  put s3 raw/<cell>/c<C>/r<rep>/w<wave>/worker-<k>.jsonl.gz
  dynamodb ADD completed, errors; SET last_activity_at
  COLD = False
```

Requests are never retried inside a measurement. Errors are recorded and counted.

### 9.3 Streaming client

- POST `/v1/completions` with `stream: true`, `stream_options: {include_usage: true}`,
  `max_tokens`, `temperature`, `seed`, `ignore_eos`.
- Times (monotonic, ns): `t_send` just before the request is written; `t_first` at the
  first SSE event whose `choices[0].text` is non-empty; `t_last` at the last such event;
  `t_end` at `[DONE]` or stream close.
- Usage comes from the final usage chunk. A missing usage chunk is an error
  (`no_usage`); the harness does not count tokens itself.
- SSE events may carry more than one token under load. TPOT therefore uses
  server-reported token counts and first and last token times, not chunk gaps.
- Classifier: POST `/classify` with `input` as a list of `b` texts (batch size is the level
  dimension alongside concurrency); `t_end` when the body is read; `usage.prompt_tokens`
  is checked against the manifest like generation prompts.

### 9.4 Checks inside the harness

- **Start spread.** Spread of first-request wall-clock starts per wave; flagged above
  250 ms (proposal).
- **Late workers.** Workers starting more than 2 s after `start_at` are flagged; a wave
  with more than 10% late workers is flagged.
- **Effective concurrency.** A level below 0.9 x nominal concurrency is flagged.
- **Prompt tokens.** Every record compares `usage.prompt_tokens` with the expected count
  from the prompt manifest; any mismatch is flagged per cell.
- **Output length.** With `ignore_eos`, `completion_tokens != max_tokens` is flagged.

## 10. Data model

### 10.1 S3 layout (`s3://amx-gpu-bench-<account>-<region>/`)

```
artifacts/                         host scripts, prompt sets
hosts/<target>/                    host facts written at each boot
models/hf/<repo>/<revision>/       cached HF snapshots
models/ov/<repo>/<revision>/<precision>/<export-hash>/
runs/<run_id>/run.yaml             as submitted
runs/<run_id>/resolved.json        digests, AMIs, revisions, prices, args, env, harness commit
runs/<run_id>/hosts/<target>.json  host facts, memory bandwidth
runs/<run_id>/bringup/<cell>.json  phase timings
runs/<run_id>/capacity/<target>.json  capacity errors seen while starting instances
runs/<run_id>/amx/<cell>/          evidence.json, onednn_verbose.log.gz, ov_exec_types.json, perf-c<C>-r<rep>.csv
runs/<run_id>/warmup/...           excluded from metrics
runs/<run_id>/raw/<cell>/c<C>/r<rep>/w<wave>/worker-<k>.jsonl.gz
runs/<run_id>/quality/<cell>/      outputs.jsonl.gz, scores.json
runs/<run_id>/metrics/<cell>/c<C>/r<rep>.json
runs/<run_id>/report/              summary.csv, summary.md, summary.json, charts/*.png
```

### 10.2 Raw record (one JSON line per request)

```json
{"run_id": "...", "cell": "m1-qwen2.5-3b|t2-m8i-4xl|bf16", "conc": 16, "repeat": 0,
 "wave": 0, "worker": 7, "req": 12, "prompt_id": "short-0042",
 "wall_send_ns": 0, "send_to_first_ns": 0, "send_to_last_ns": 0, "send_to_end_ns": 0,
 "prompt_tokens_expected": 128, "prompt_tokens": 128, "completion_tokens": 128,
 "n_events": 97, "http_status": 200, "error": null, "timeout": false,
 "cold_env": false, "late_start_ms": 3.1, "output_sha256": "..."}
```

Classifier records replace token-timing fields with `batch_size`, `input_tokens` and
`send_to_end_ns`. Quality records also store the output text, or the label and class
probabilities for the classifier.

### 10.3 DynamoDB (single table `amx-gpu-bench`)

| PK | SK | Content |
|---|---|---|
| `RUN#<id>` | `META` | status, config S3 key, execution ARN, created/updated |
| `RUN#<id>` | `CELL#<cell>` | status, bring-up timings, AMX verdict |
| `RUN#<id>` | `LEVEL#<cell>#c<C>#r<rep>` | status, `completed`, `errors` (atomic counters), waves |
| `TARGET#<id>` | `ACTIVITY` | `last_activity_at`, `active_run_id` |
| `JOURNAL#<model>` | `<target>#<precision>#<ts>` | engineer hours, issues, fixes, calibrated rule ID |

## 11. Metric definitions

Per request i: send `s`, first token `f`, last token `l`, end `e`, completion tokens `n`.
All requests of a wave share one timeline: worker wall anchor plus monotonic offsets.

| Metric | Definition |
|---|---|
| E2E latency | `e - s` |
| TTFT | `f - s` (includes queueing, prefill and network) |
| TPOT | `(l - f) / (n - 1)` for `n >= 2` |
| Output tok/s per request | `n / (e - s)` |
| Measurement window W (per wave) | from `start_at` to the earliest time any worker finishes its last request, so all C workers are active inside W |
| Aggregate output tok/s | sum over requests of `n x overlap([f, l], W) / (l - f)`, divided by the length of W; summed over waves as total tokens over total window time |
| Requests/s | requests whose end falls inside W, divided by the length of W |
| Effective concurrency | sum over requests of `overlap([s, e], W)`, divided by the length of W |
| Error rate | failed / all measured requests in the level |
| Timeout rate | client timeouts plus HTTP 408/504, divided by all measured requests |
| Percentiles | exact, over all successful measured requests of the level (all waves), `numpy.percentile(method="linear")`; n reported |
| Prefill tok/s per request | `prompt_tokens / (f - s)`; at C=1 this is close to pure prefill speed, at higher C it includes queueing, so reports show it per level |
| Aggregate input tok/s | prompt tokens of requests whose first token falls inside W, divided by the length of W |
| Decode tok/s per request | `1 / TPOT` |
| Classifier | per-batch latency `e - s` (P1, P50, P99, mean); classifications/s = inputs completed in W / length of W; levels are concurrency x batch size |

Prefill and decode figures are always reported apart. A combined "N times faster" figure
is never computed (research.md, contradiction 15).

Repeats: each repeat produces its own level metrics. The headline numbers pool all
repeats (more samples for P99). Each row also reports mean, standard deviation and CV of
P50, P99 and aggregate tok/s across repeats; CV above 10% is flagged (proposal).

## 12. Cost model

- **Price snapshot** at preflight: `pricing:GetProducts` (us-east-1 endpoint) for each
  instance type with `regionCode`, `operatingSystem=Linux`, `tenancy=Shared`,
  `preInstalledSw=NA`, `capacitystatus=Used`; store USD/hour, SKU, offer term code, rate
  code, effective date, and the price list ARN and publication date from
  `ListPriceLists`. If any price is missing, the run does not start. No fallback values.
- `$/1M output tokens = price_per_hour / (aggregate_tok_s x 3600) x 10^6`
- `$/1K requests = price_per_hour / (requests_per_s x 3600) x 10^3`
- `$/1M classifications = price_per_hour / (classifications_per_s x 3600) x 10^6`
- These assume the instance is held at that load for the whole hour; reports say so.
- Cost is information. It does not change the SLA ranking's validity rules or the
  migration verdict (section 18.4).

### 12.1 Capacity equivalence

For each model, take the best GPU row that meets the SLA (highest throughput) and each
CPU row that meets the SLA:

- `cpu_instances_per_gpu = ceil(gpu_throughput / cpu_throughput)`, where throughput is
  aggregate output tok/s (classifications/s for M3) at that row's concurrency.
- `standard_vcpus_needed = cpu_instances_per_gpu x vCPUs of the CPU type`, shown next to
  the G-family vCPUs the GPU instance uses, because the two come from separate On-Demand
  quota pools (research.md 8.3).
- `hourly_cost_ratio = cpu_instances_per_gpu x cpu_price / gpu_price`.

If no GPU row meets the SLA, capacity equivalence is not computed and the report says so.

## 13. Resource metrics

- CloudWatch agent at 10 s intervals, namespace `AmxGpuBench`, dimension `target_id`:
  `cpu usage_active` (total and per core), `mem used` and `used_percent`, network, and on
  GPU the agent's `nvidia_gpu` plugin (`utilization_gpu`, `memory_used`, `power_draw`).
- `AggregateLevel` reads mean, P95 and max for the level window with `GetMetricData`.
- Per-core CPU shows whether thread binding landed on physical cores as configured.

## 14. AMX verification

### 14.1 Evidence

| When | Evidence | How |
|---|---|---|
| Boot | CPU flags | `lscpu`; required `amx_bf16 amx_int8 amx_tile`, plus `amx_fp16` for FP16 cells |
| Every measured CPU level | Hardware counter | `perf stat -a -e exe.amx_busy,cycles` for the level; ratio = `amx_busy / cycles` |
| Once per CPU cell, outside the measured window | vLLM / PyTorch engine trace | restart the engine with `ONEDNN_VERBOSE=1` (vLLM also `VLLM_LOGGING_LEVEL=DEBUG`), send 5 requests at C=4, count `exec,cpu,matmul` lines by implementation and the `CPU unquantized GEMM dispatch` lines, restart normally |
| Once per OpenVINO cell | OpenVINO execution types | `probe_ov.py` compiles the same IR with the same OpenVINO version, hints and `PERF_COUNT=YES`, runs one short generation or classification batch, lists execution types of FullyConnected/MatMul nodes and the runtime inference precision |

For FP16 cells the evidence must also show FP16 execution: an `amx_fp16` (or
`avx10_1_512_amx_fp16`) oneDNN implementation for PyTorch, or `f16` runtime precision with
AMX execution types for OpenVINO. A cell labeled FP16 that ran in BF16 is relabeled and
flagged. For `bf16-ov` cells (an FP16 IR with no precision hint) the probe's runtime inference
precision must be `bf16`; anything else relabels and flags the cell the same way. For W8A8
cells, the vLLM dispatch log must show the INT8 AMX kernel.

### 14.2 Verdict

| Condition | Verdict |
|---|---|
| Any required flag missing | INVALID |
| Counter available and `amx_busy > 0` with ratio >= 1e-4 (proposal) | VERIFIED |
| Counter available and ratio below threshold | INVALID |
| Counter unavailable, engine trace shows AMX implementations for the main GEMMs | VERIFIED (software evidence) |
| Counter unavailable, trace shows no AMX | INVALID |

INVALID rows stay in the raw table, are marked in every report, and are excluded from SLA
and cost rankings. `evidence.json` stores the flags, counter values, matched verbose
lines (sample), dispatch counts and OpenVINO execution types.

Optional negative control (off by default, section 23): one M1 level on vLLM CPU with
`ONEDNN_MAX_CPU_ISA=AVX512_CORE_BF16`. This disables AMX for oneDNN GEMMs only (the SGL
kernel and AMX attention check the hardware directly), so it shows part of AMX's
contribution, not all of it.

## 15. Quality check

- **Prompts.** `quality-50`: 50 fixed prompts with IDs and checksums (source proposal:
  databricks-dolly-15k, CC BY-SA 3.0, stratified by category) for M1, M2, M4 and M5.
  `quality-50-nli` for M3: 50 premise/hypothesis pairs (source proposal: SNLI test split,
  CC BY-SA 4.0), balanced across the three labels.
- **Run.** Once per cell, C=4, temperature 0, natural stop, `max_new_tokens` 256.
- **Reference.** The T1 BF16 cell of the same model (thin CUDA for M5). For M3 there is a
  second anchor: PyTorch FP32 on T4, the way Riskfuel judged FP16 against FP32.
- **Scores.** Generation: exact-match rate, ROUGE-L F1 mean and P10 (`rouge-score`).
  Classifier: label agreement rate, and mean and maximum absolute difference of class
  probabilities, against both anchors; label accuracy against the dataset labels is
  reported as context.
- **Flags** (config, proposals): FP16/INT8/INT4 generation cells flagged when mean ROUGE-L
  vs reference falls more than 0.10 below the CPU BF16 cell's value; classifier cells
  flagged when label agreement with FP32 is below 98% or maximum probability difference
  is above 0.05. `quality_delta` = 1 - mean ROUGE-L, or 1 - label agreement.
- CPU BF16 and GPU BF16 will not match exactly under greedy decoding because their
  numerics differ. The CPU BF16 score is the noise floor that quantized cells are judged
  against.

## 16. Reports and charts

- `summary.csv`: one row per model x target x precision x concurrency with every metric
  in sections 11-15 plus provenance columns.
- `summary.md`: per model, the benchmark table with the columns from Requirement 14.1
  plus prefill tok/s; every valid CPU configuration meeting the SLA highlighted, the
  cheapest of them marked, next to the best GPU configuration; a capacity-equivalence
  table (section 12.1); charts; price source block; capacity errors seen during the run;
  warnings (non-commercial license, INVALID AMX, quality flags, CV flags, prompt-token
  mismatches, smoke label).
- Charts (matplotlib PNG in reports, interactive in the UI): P50 and P99 E2E vs
  concurrency; P99 TTFT vs concurrency; aggregate tok/s vs concurrency; prefill tok/s vs
  prompt length; $/1M tokens vs concurrency; classifications/s vs batch size for M3. CPU
  and GPU share axes; log scale optional.
- **SLA evaluation.** A row qualifies when every SLA constraint holds, error rate is at
  most `max_error_pct`, the row is not flagged for quality, and CPU rows are AMX VERIFIED.

## 17. UI (Streamlit, local)

`streamlit run ui/app.py` (address pinned to 127.0.0.1 in config). Pages:

1. **New run.** Model and target pickers, precision per cell, prompt set or custom prompt
   (shows rendered token counts per model), max new tokens, temperature, concurrency list,
   requests per level or duration, repeats. Shows the YAML, the planned cells and levels,
   the duration and cost estimate (section 20) and preflight results. Start calls
   `StartExecution`.
2. **Runs.** Status from DynamoDB and Step Functions, per-level progress bars, errors.
   Stop calls `StopExecution` after a confirmation.
3. **Results.** Run selector, filterable table, SLA inputs with highlighting, charts,
   provenance, CSV and Markdown download.
4. **Migration calculator** (section 18).
5. **Bring-up journal** (section 19).

The operator needs IAM permissions for these calls; the README lists them.

## 18. Migration effort calculator

### 18.1 Config (`config/migration_effort.yaml`)

```yaml
units: engineer_days
base_rules:                  # per model; all matching rules are found, the widest range wins
  - {id: pytorch_or_sagemaker, when: {stack: [pytorch_transformers, sagemaker_pytorch]}, days: [1, 3]}
  - {id: vllm_gpu_to_vllm_cpu, when: {stack: [vllm_gpu]}, days: [3, 7]}
  - {id: flan_t5_to_openvino,  when: {model_type: [encoder-decoder]}, days: [3, 7]}
  - {id: tgi_to_optimum_ovms,  when: {stack: [tgi]}, days: [5, 10]}
  - {id: triton_trtllm,        when: {stack: [triton_trtllm]}, days: [10, 20]}
addons_per_model:
  flash_attention_to_sdpa: [1, 1]
  bitsandbytes_to_ov_awq_gptq: [2, 5]
  custom_cuda_or_triton_kernels: [5, 15]
  tensorrt_engines_outside_triton: [5, 10]      # proposal; not in the brief
  accuracy_validation: {bf16: [2, 3], int8: [2, 4], int4: [4, 5]}   # brief: 2-5, INT4 at the upper end
addons_per_project:
  load_latency_testing: [3, 5]
  infra_ci_changes: {iac_and_cicd_exist: [1, 2], otherwise: [3, 5]}   # brief: 1-5
familiarity_multiplier: {low: 1.3, medium: 1.0, high: 0.85}         # proposal
calibration: {clamp: [0.5, 2.0]}
```

### 18.2 Algorithm

1. For each model: base range from the widest matching rule (an encoder-decoder on vLLM
   gets the larger of the vLLM rule and the Flan-T5 rule); add the per-model add-ons that
   apply; add accuracy validation for the chosen precision.
2. Add project add-ons once.
3. Low = sum of lows; high = sum of highs; likely = sum of midpoints.
4. Multiply all three by the familiarity multiplier.
5. Calibration: for each rule with journal entries, factor = median(actual / predicted
   midpoint), clamped to [0.5, 2.0], applied to that rule's tasks. The breakdown shows
   "calibrated from N entries" or "uncalibrated".
6. Round to half days.

### 18.3 Risks (rule-based)

Custom CUDA/Triton kernels (no CPU equivalent); bitsandbytes (format change, re-validate);
INT4 (accuracy); encoder-decoder on vLLM, TGI or TensorRT-LLM (engine change, no vLLM T5);
Triton + TensorRT-LLM (engine and model format change); gated models; non-commercial
licenses (Qwen2.5-3B); customer models above the 4B benchmark cap (results are
extrapolated); SLA not met at the required concurrency; many CPU instances needed per GPU
instance (Standard vCPU quota); AMX unverified for the proxy results; low CPU-tuning
familiarity.

### 18.4 Recommendation

Capacity decides the verdict; cost is shown but never changes it. The calculator maps each
customer model to the benchmarked model of the same type and nearest parameter count,
then reads that model's summary for the user's SLA:

- **Go:** an AMX-VERIFIED, quality-passing CPU row at the requested precision meets the
  SLA, and `cpu_instances_per_gpu` is at most `max_cpu_instances_per_gpu` (default 4,
  proposal).
- **Conditional:** the SLA is met only with quantization below the requested precision,
  only within a limited concurrency range, or only with more CPU instances per GPU
  instance than the limit.
- **No-go:** no valid CPU row meets the SLA.
- **Insufficient data:** no benchmarked model of that type, or the customer model is far
  outside the benchmarked sizes.

Every verdict shows the capacity equivalence (instances and Standard vCPUs) and the hourly
cost ratio against GPU, and cites the result rows it used.

## 19. Calibration journal

- Machine phases come from `bringup/<cell>.json` automatically.
- The journal page records, per model, target and precision: engineer hours, issues hit
  (for example "OVMS answered on /v3, not /v1"), fixes, and the calculator rule
  the entry calibrates. Entries are stored in DynamoDB and exported with reports.

## 20. Duration and cost estimates

The UI estimates before a run starts:

- Level time ~ `ceil(R / C) x E2E_p50 + waves x 40 s` (barrier lead plus Map start),
  where `E2E_p50` comes from earlier runs of the same model and target, or from
  configurable priors.
- Cell overhead: model fetch and load (2-10 min), AMX probe (about 3 min), quality run.
- Cost = sum over targets of active hours x on-demand price, plus NAT gateway
  ($0.045/hour and $0.045/GB processed in us-east-1, from the Price List API), plus
  Lambda (arm64 $0.0000133334 per GB-second; a 12-minute wave of 64 workers at 1769 MB is
  about $1.09).

Illustrations (assumed speeds, not measurements):

- **Smoke run (deliverable 6).** M1 on T1 and T2-m8i, short prompts, 128 new tokens,
  C in {1, 4, 16}, 50 requests per level, 1 repeat, quality on. If both targets need about
  1.5-2.5 instance-hours, the cost is about $3-5 including NAT and Lambda.
- **Full methodology, worst cell.** M2 (Qwen3-4B) BF16 on m8i.4xlarge at C=1 with 256 new
  tokens: if CPU decode runs at 8 tokens/s, one request takes about 32 s, so 200 requests
  take about 1.8 hours per repeat, and about 5.3 hours for three repeats. The 4B cap
  halves the earlier 7B estimate. The UI shows this per cell before the run starts.

## 21. Error handling

| Failure | Handling |
|---|---|
| No capacity for the instance type | Retry start for 10 min; fail the target's cells with the reason; other targets continue |
| Engine fails to become healthy (20 min timeout) | Upload container logs to `runs/<run>/bringup/`; fail the cell; continue |
| Request error or timeout | Record it; no retry; counts toward error rate |
| Worker Lambda failure | Retry the wave once with a new barrier; then fail the level |
| Price API failure | Run does not start |
| Gated model not accessible | Preflight fails with the model ID and the HF page to request access |
| Prompt-token mismatch | Flag the cell; optional `fail_on_token_mismatch` |
| AMX not verified | Keep rows, mark INVALID, exclude from rankings |
| Run stopped by the user | Cleanup step stops `perf`, marks the run cancelled; reaper handles instances |

## 22. Testing strategy

- **Unit (pytest):** percentile and window math, effective concurrency, cost formulas,
  SSE parsing against recorded fixtures from vLLM GPU, vLLM CPU, OVMS and the thin server,
  config validation, AMX verdict table, estimator rules and calibration, quality scoring.
- **Contract:** the thin server exercised with the OpenAI Python client in streaming and
  non-streaming mode.
- **Local integration:** the worker against a fake OpenAI server with scripted delays;
  computed TTFT and TPOT must match the script within 1 ms.
- **CDK assertions:** no public IPs on targets, security-group ingress only from
  `sg-workers`, IMDSv2 required, S3 Block Public Access, TLS-only bucket policy, tags on
  every resource.
- **On AWS:** the smoke run.

## 23. Review decisions and remaining defaults

### 23.1 Decided in review (2026-11-09)

| # | Question | Decision | Where applied |
|---|---|---|---|
| 1 | Account | Deploy to `346399954218`, us-east-1, AZ `use1-az1` (default kept) | sections 5.3, 6 |
| 2 | M3 use case | Classification. M3 is `FacebookAI/roberta-large-mnli`; ModernBERT classifier optional (no OpenVINO path) | sections 5.1, 7.1, 15 |
| 3 | Cost vs capacity | Capacity decides the verdict; cost is shown only | sections 12, 12.1, 18.4 |
| 4 | Run length | Models capped at 4B. Qwen2.5-7B removed; M2 is Qwen3-4B-Instruct-2507 (4.02B, nominal 4B) | sections 5.1, 20 |
| 5 | Extra rows and quantized cells | Decided from the OpenInfer and Riskfuel case studies: add vLLM W8A8 INT8 (on); keep AWQ INT4 optional (off); add classifier FP16 on 8i with an FP32 anchor; report prefill separately; optional 14K prompt set; gemma-4-E2B not added (above the cap) | research.md 13.3; sections 7.1, 11, 14, 15 |
| 6 | Endpoint auth | No API key; private subnet plus security group is the boundary (confirmed) | section 6 |
| 7 | HF token | Reuse the stored token (`amx-gpu-bench/hf-token`); Gemma access confirmed (HTTP 200) | section 6 |
| 8 | AMX-off control run | Not run | section 14 |
| 9 | Use case | Keep the standard suite and add a DSPM/DLP suite on synthetic data, tested in depth | section 24 |

### 23.2 Not answered; defaults kept

Say if you want any of these changed; otherwise tasks.md uses them.

| # | Topic | Default |
|---|---|---|
| a | `max_new_tokens` | 256 for full runs, 128 for smoke runs; request-count mode (200 x 3) for full runs |
| b | Quality thresholds | ROUGE-L drop 0.10 vs CPU BF16; classifier label agreement 98% and max probability difference 0.05 vs FP32 |
| c | Harness cross-check with `vllm bench serve` | Not included. "Use standard benchmarking" was read as "keep the standard suite"; if it meant this cross-check, it is one task to add |
| e | Prompt sources | Project Gutenberg public-domain text for length sets; databricks-dolly-15k (CC BY-SA 3.0) and SNLI (CC BY-SA 4.0) for quality sets |
| f | Repo hosting | Local folder outside the public `ml` repo; no remote configured until you choose one |
| h | Calculator proposals | Familiarity multipliers 1.3 / 1.0 / 0.85; "TensorRT engines outside Triton" add-on 5-10 days; capacity limit 4 CPU instances per GPU instance |
| i | Qwen3-4B at 4.02B | Counted as nominal 4B. Say "strict" and it is replaced with a model under 4.00B |

## 24. DSPM/DLP suite

Research basis: research.md section 14. The suite reuses the whole harness (targets, load
generator, metrics, AMX verification, reports). What it adds is a labeled dataset, a
policy file, request shapes for narrow-label classification, accuracy scoring, and
DSPM-specific report sections.

### 24.1 Workloads

| ID | Workload | Input | Output | Levels (C = concurrency) | SLA preset | Models |
|---|---|---|---|---|---|---|
| W1 | Bulk scan, binary | Document chunks of ~512 and ~2,048 tokens (512 only for M5, M3) | `SENSITIVE` / `NOT_SENSITIVE` | C in {8, 16, 32, 64} | bulk | M1, M2, M4, M5, M3, D1 |
| W2 | Bulk scan, category | ~512-token chunks | One of `GOV_ID PERSONAL FINANCIAL HEALTH CREDENTIALS CONFIDENTIAL NONE` | C in {16, 32} | bulk | M1, M2, M4, M5 |
| W3 | Inline DLP | Chat messages and threads, emails, GenAI prompts, code questions; 64-512 tokens | `ALLOW` / `BLOCK` | C in {1, 4, 8, 16} | inline standard and inline strict | M1, M2, M4, M5, M3, D1, D2 |
| W4 | Structured columns | CSV header plus 15 sample rows, ~300-800 tokens | One category for the table, or `NONE` | C in {8, 32} | bulk | M1, M2, M4 |
| W5 | Redaction (off by default) | W3 items | JSON list of sensitive spans, at most 128 tokens | C in {4, 16} | inline standard | M1, M2, M4 |

SLA presets (proposals, editable in the run config):

| Preset | Latency limit | Throughput goal |
|---|---|---|
| `inline_strict` | P99 E2E at most 250 ms | none |
| `inline_standard` | P99 E2E at most 1,000 ms (PolicyGuard's gate measured P50 669 ms with a 20B model) | none |
| `bulk` | P99 E2E at most 30 s | maximize documents/s |

### 24.2 Policy file (`config/dspm/policy_v1.md`)

One plain-text file is the single source of truth. It labels the dataset and, as a
system prompt, instructs the generative models (PolicyGuard's policy-as-prompt approach).

| Category | Sensitive when the content contains | Macie analogue |
|---|---|---|
| `GOV_ID` | SSN, passport, driver's license, national ID or tax ID number | Personal information: PII |
| `PERSONAL` | A person identified by name together with at least one of: date of birth, home address, personal phone, personal email, precise location | Personal information: PII |
| `FINANCIAL` | A card number with a valid check digit, a bank account or IBAN with its owner, or routing plus account numbers | Financial information |
| `HEALTH` | A diagnosis, treatment, medical record number or health insurance ID tied to a person | Personal information: PHI |
| `CREDENTIALS` | API keys, access tokens, passwords, private keys, connection strings with passwords, session cookies, JWTs | Credentials |
| `CONFIDENTIAL` | Business content marked confidential or internal: M&A, unreleased financials, roadmaps, customer lists, internal-only source code | none (PolicyGuard's customer_confidential and proprietary_code) |

Decision rules written into the file:

- Masked or truncated values are not sensitive (`***-**-6789`, "card ending 4242").
- Well-known documentation and placeholder values are not sensitive
  (`AKIAIOSFODNN7EXAMPLE`, `YOUR_API_KEY`, `4111 1111 1111 1111` in a test plan).
- Real-format secrets are sensitive even when the text calls them test values, because
  the classifier cannot verify that (PolicyGuard's rule).
- Text about a sensitive topic without a sensitive value is not sensitive.
- Public business information is not sensitive.
- A first name alone is not sensitive.
- When several categories apply, the category answer is the highest in this order:
  `CREDENTIALS > GOV_ID > FINANCIAL > HEALTH > PERSONAL > CONFIDENTIAL`.

The file also holds 10-12 worked examples taken from the dev set. Target size: at most
900 tokens. Flan-T5 gets a short form of the same rules (at most 120 tokens) because it was
trained on inputs of at most 512 tokens. Both files are frozen and hashed before the
first measured run, and the hash is stored with every result.

### 24.3 Dataset build (`tools/dspm/build_dataset.py`)

**Sources**

| ID | Source | Revision | What is taken |
|---|---|---|---|
| S1 | nvidia/Nemotron-PII, test split | `b70ffaf5ff39e079776134c5bf4381f00a9fd1ed` | Unstructured and structured documents with span labels |
| S2 | gretelai/synthetic_pii_finance_multilingual, test split, English | `7b844d16738527a04264f50214cb426a4cea0897` | Long financial and legal documents for ~2,048-token chunks |
| S3 | Seeded generator (this repo) | git commit | Chat threads, GenAI prompts, emails, code (Python, JavaScript, Java, Go, Terraform), configuration (`.env`, YAML, JSON), logs (JSON lines, HTTP access logs), CSV tables, confidential business memos |

**Transforms**

- **Positives:** S1/S2 documents as they are; S3 templates filled with fake values from a
  pinned Faker version. Secrets use real formats with random bodies (`AKIA...`, `ghp_...`,
  `xoxb-...`, `sk_live_...`, PEM private keys, JWTs, database URIs with passwords). Card
  numbers have valid check digits.
- **Negatives:** S1/S2 documents with every labeled span rewritten into non-identifying
  text; S3 templates filled with non-sensitive values.
- **Hard negatives** (at least 30% of negatives): one generator per hard case in
  research.md 14.8 (look-alike numbers, masking, documentation values, topic-only text,
  public business info, first name only, numbers without keywords).
- **Labels:** S3 items carry the categories of the values the generator inserted. S1/S2
  spans map to categories through `config/dspm/span_label_map.yaml` (for example `ssn` to
  `GOV_ID`, `credit_debit_card` to `FINANCIAL`, `medical_record_number` to `HEALTH`), and
  the `PERSONAL` combination rule is applied on top.
- **Chunking:** to target token counts measured with each model's tokenizer; every item
  records its token count per model and its UTF-8 byte count (needed for $/GB).

**Splits and sizes (defaults)**

- Dev: 300 items, used only to write and tune the policy file.
- Test: at least 3,000 items, half sensitive, at least 150 positives per category and per
  workload. Approximate mix: W1/W2 documents 1,400, W3 messages and prompts 1,200, W4
  tables 400.
- Split by template family (S3) and by source document `uid` (S1/S2), so no near-duplicate
  of a test item appears in dev.

**Checks** (fail the build if any fails): label agrees with the generator metadata; every
positive card number passes the check digit; every masked hard negative contains no
unmasked value; no duplicate or near-duplicate text across splits; category and hard-case
quotas met. The build also writes `review_sample.md`, 100 stratified items for a human to
read before the test set is frozen.

**Storage:** `s3://<bucket>/datasets/dspm/v1/{dev,test}.jsonl.gz` and `manifest.json`
(counts, hashes, source revisions, seed, tool versions). A local copy goes to `data/`,
which is in `.gitignore`: the items contain realistic fake secrets that secret scanners
would flag, and nothing generated should be pushed to a public remote. Reports carry the
CC BY 4.0 attribution for Nemotron-PII.

Item schema:

```json
{"id": "w3-s3-chat-0412", "workload": "W3", "channel": "chat", "source": "S3",
 "family": "chat/payroll-question", "text": "...", "label": "BLOCK",
 "categories": ["FINANCIAL"], "hard_case": null, "spans": [{"start": 41, "end": 57, "category": "FINANCIAL"}],
 "tokens": {"m1-qwen2.5-3b": 212, "m5-flan-t5-xl": 231}, "bytes": 803}
```

### 24.4 Request shape per model type

| Model | Request | Decision |
|---|---|---|
| M1, M2, M4 | Chat template rendered client-side: system message = policy file; user message = the content between delimiters plus "Answer with one label."; `/v1/completions`, `max_tokens` 4 (binary) or 8 (category), temperature 0, stop at newline | First word must exactly match a label. Anything else is a format failure, counted and scored as `SENSITIVE`/`BLOCK` (fail closed, as PolicyGuard does) |
| M1, M2, M4, constrained variant | Same plus `structured_outputs: {"choice": [labels]}` | vLLM targets only; reported as a separate variant |
| M5 | Flan-style prompt with the short policy and `OPTIONS: yes, no` (or the category list); input at most 512 tokens | Same strict parsing |
| M3 | `/classify` with the text as premise and one hypothesis per category ("This text contains a government ID number.", and so on) | Sensitive when the highest entailment probability passes a threshold chosen on the dev set |
| D1 | Thin server `POST /detect`: token classification plus the model card's constrained Viterbi span decoding at the default operating point; returns spans and per-category maximum scores, not per-token tensors | Sensitive when a span category maps to a policy category. D1 has no health or confidential-business categories, so its recall there is expected to be low; the report shows coverage per category |
| D2 | `/classify` with the input format from the model card (`Instruct: ... Query: t: {text}`), 512 tokens, sigmoid outputs | `BLOCK` when `privacy_giving_pii` is at least 0.55 (model card threshold); W3 only |

Prefix caching is on for vLLM and OVMS cells, because the policy prefix is identical on
every request; the warm-up requests warm it, as in production. The thin servers and the
classifiers have no prefix to cache.

### 24.5 Cells

| Model | T1 GPU (g6.xlarge) | T2 vLLM CPU (m8i.4xl) | T3 OpenVINO (m8i.4xl) | T4 PyTorch CPU (m8i.4xl) |
|---|---|---|---|---|
| M1, M2, M4 | vLLM BF16 | vLLM BF16, W8A8 | OVMS BF16 (`bf16-ov`), INT8, INT4 | |
| M5 | thin CUDA BF16 | | thin OpenVINO INT8 | thin BF16 |
| M3 | vLLM `/classify` BF16 | vLLM `/classify` BF16 | thin OpenVINO FP16, INT8 | thin BF16, FP16 |
| D1 | thin CUDA BF16 | vLLM (optional; returns large per-token tensors) | not supported | thin BF16 |
| D2 | vLLM `/classify` BF16 | vLLM `/classify` BF16 | thin OpenVINO FP16, INT8 | thin BF16, FP16 |

c8i.4xlarge and m8i.8xlarge rows: M2 in BF16 and W8A8 on W1 and W3, plus D1. The thin
server gains `POST /detect` for D1 and a multi-label mode for D2.

### 24.6 Runs

- **Accuracy pass:** every cell classifies the full test set of its workloads once, at
  C = 16. These requests also feed a throughput figure, but the headline performance
  numbers come from the levels below.
- **Performance levels:** request pools drawn from the test set by workload and length
  bucket, in a fixed order; at least 200 requests per level and 3 repeats in full mode.
- **Order:** the dataset and policy are frozen, then the sealed run happens once per
  cell; re-runs create a new run ID and keep the old one.

### 24.7 Metrics and report

Added to sections 11, 12 and 16:

| Metric | Definition |
|---|---|
| Documents/s | completed items in the window W divided by the length of W |
| $/1M documents | `price_per_hour / (documents_per_s x 3600) x 10^6` |
| $/GB classified | `price_per_hour / (bytes_per_s x 3600) x 10^9`, with bytes = UTF-8 bytes of the content (policy excluded); GB = 10^9 bytes, matching Macie's unit |
| GB/hour per instance | `bytes_per_s x 3600 / 10^9` |
| Recall (block rate), precision, false positive rate, F1 | standard definitions on the binary label, each with a Wilson 95% interval |
| Per-category recall | recall within each policy category; for W2 both strict (top-priority category) and lenient (any gold category) |
| Hard-negative false positive rate | false positive rate per hard-case type |
| Format adherence | share of outputs that parsed to a valid label |
| CPU-vs-GPU agreement | share of items where the CPU cell's label equals the GPU cell's label for the same model and precision |

The DSPM/DLP report adds: an accuracy table per workload with intervals; a hard-negative
breakdown; per-category coverage for D1 and D2; the CPU-vs-GPU agreement table; documents/s
and $/GB charts vs concurrency; capacity equivalence in documents/s; the Macie and Google
list-price block from research.md 14.5, labeled as not like-for-like; and the statement
that speed transfers to fine-tuned models of the same shape while accuracy does not.

**Verdict rule for this suite:** a CPU row meets the SLA when it meets the preset and,
against the GPU row of the same model and precision, keeps label agreement of at least
98% and loses at most 1 percentage point of recall (Requirement 23.3). Absolute accuracy
floors (recall at least 0.90, false positive rate at most 0.05) are reported as model
fitness and do not change the CPU-vs-GPU verdict (Requirement 23.4).

### 24.8 Size of a full DSPM/DLP run

The accuracy pass dominates. W1 alone is about 1,400 items of ~500-2,000 tokens, around
1.5M uncached content tokens per cell. At an assumed 1,000 prefill tokens/s on a CPU
cell (not measured), that is about 25 minutes per CPU cell for W1, and there are about 20
CPU cells. The UI estimator shows the real figure per cell before a run starts, using
measured prefill speeds once the smoke run exists.

# Implementation Plan

Based on [design.md](design.md) revision 3 and [requirements.md](requirements.md)
revision 3. Tasks 1-24 are local code with no AWS changes. Tasks 25-27 create and use AWS
resources in account `346399954218`; before the first `cdk deploy`, I will confirm with
you and list what will be created and what it costs per hour.

- [x] 1. Scaffold the repository
  - Create `pyproject.toml` (Python 3.12, ruff, pytest, pydantic), the package layout from
    design section 4, `.gitignore` (no `.env`, no credentials, no model files) and a local
    `git init` with no remote.
  - Add `ui/.streamlit/config.toml` with `server.address = "127.0.0.1"`.
  - _Requirements: 6.1, 18.6_

- [x] 2. Config schemas and catalogs
  - [x] 2.1 Write pydantic models for `models.yaml`, `targets.yaml`, `migration_effort.yaml`
    and run configs in `bench/config.py`, including target inheritance.
    - Validation: revision must be a 40-hex commit SHA; `nominal_size_b <= 4` (configurable
      cap); FP16 precisions need `requires_cpu_flag: amx_fp16`; `mode: full` needs at
      least 200 requests and 3 repeats, otherwise the run is labeled smoke.
    - _Requirements: 1.1, 1.3, 1.6, 6.1, 7.7_
  - [x] 2.2 Write `config/models.yaml` with M1-M5 and the disabled alternatives, pinned
    revisions and precision variants from research.md sections 2.1, 2.2 and 2.6.
    - _Requirements: 1.2, 1.5_
  - [x] 2.3 Write `config/targets.yaml` (T1, T2 m8i/c8i 4xl, T2 m8i 8xl, T3, T4, T5
    disabled) with image digests, AMI SSM parameters, engine args and env from design
    sections 5.2 and 7.
    - _Requirements: 2.1, 2.3, 2.4, 2.6_
  - [x] 2.4 Write `config/runs/smoke-m1.yaml` and `config/runs/full.yaml`.
  - [x] 2.5 Unit tests for every validation rule above.
    - _Requirements: 1.3, 1.6, 7.7_

- [x] 3. Prompt library
  - [x] 3.1 `tools/build_prompts.py`: build the short (~128), medium (~1,024), long
    (~4,096) and optional xlong (~14,000) sets per model tokenizer from pinned Project
    Gutenberg texts; apply chat templates client-side with explicit BOS handling; record
    expected prompt tokens; write a manifest with SHA-256 checksums.
    - Classifier sets: premise/hypothesis pairs at ~128 and ~512 tokens.
    - _Requirements: 7.1, 7.2, 7.6_
  - [x] 3.2 Build `quality-50` (databricks-dolly-15k, stratified) and `quality-50-nli`
    (SNLI test, label-balanced) with fixed IDs and checksums.
    - _Requirements: 13.1_
  - [x] 3.3 Tests: token counts within tolerance per model, manifest checksums stable,
    deterministic prompt order for a given seed.
    - _Requirements: 6.4, 7.2_

- [x] 4. Metrics core (`bench/metrics.py`)
  - Exact percentiles (P1, P50, P90, P99, mean, min, max), E2E, TTFT, TPOT, per-request
    output tok/s, measurement window W, aggregate output tok/s by interval overlap,
    requests/s, effective concurrency, error and timeout rates, prefill tok/s, aggregate
    input tok/s, decode tok/s, classifier latency and classifications/s, pooled repeats
    with mean, standard deviation and CV.
  - Unit tests on synthetic timelines with known answers, including uneven worker
    finishes and multi-wave levels.
  - _Requirements: 7.5, 8.1-8.8, 9.1_

- [x] 5. Load client and fake server
  - [x] 5.1 `bench/client.py`: streaming `/v1/completions` client (SSE parsing, first and
    last non-empty token times, usage chunk, `no_usage` error) and `/classify` client;
    monotonic timing with a wall-clock anchor.
    - _Requirements: 5.5, 3.1, 3.3_
  - [x] 5.2 `tests/fake_server.py`: an OpenAI-shaped server with scripted delays and
    multi-token SSE events.
  - [x] 5.3 Tests: computed TTFT and TPOT match the script within 1 ms; SSE fixtures
    recorded from vLLM GPU, vLLM CPU, OVMS and the thin server parse identically (fixtures
    added as each engine is brought up in task 20).
    - Done for the fake and thin servers; vLLM and OVMS fixtures are recorded in 25.4.
    - _Requirements: 8.2, 8.3_

- [x] 6. Pricing, cost and capacity equivalence (`bench/cost.py`)
  - Price List API snapshot (`GetProducts` with the filters from design section 12 and
    `ListPriceLists` for ARN and publication date); refuse to proceed if a price is missing.
  - $/1M output tokens, $/1K requests, $/1M classifications; capacity equivalence
    (CPU instances per GPU instance, Standard vCPUs, hourly cost ratio).
  - Tests with botocore Stubber and recorded responses.
  - _Requirements: 10.1-10.4, 14.5_

- [x] 7. AMX verification (`bench/amx.py`)
  - Parsers for `lscpu` flags, `perf stat -x,` CSV, `ONEDNN_VERBOSE` lines, vLLM
    `CPU unquantized GEMM dispatch` lines and OpenVINO execution types and runtime
    precision.
  - Verdict table from design section 14.2, plus the FP16 and W8A8 checks.
  - Tests with fixture files for each verdict branch, including "FP16 label but BF16
    execution".
  - _Requirements: 12.1-12.5_

- [x] 8. Quality scoring (`bench/quality.py`)
  - Exact match and ROUGE-L F1 (mean, P10) for generation; label agreement and mean/max
    probability difference for the classifier against GPU BF16 and FP32 anchors; flags
    with configurable thresholds.
  - Unit tests.
  - _Requirements: 13.1-13.4_

- [x] 9. Thin server (`servers/thin`)
  - [x] 9.1 FastAPI app: `/health`, `/v1/models`, `/v1/completions` (stream and
    non-stream, `include_usage`, `ignore_eos` as `min_new_tokens`), `/classify` in vLLM's
    shape; one inference thread; micro-batcher (`max_batch`, `max_wait_ms`).
    - _Requirements: 3.1-3.4_
  - [x] 9.2 torch backend: seq2seq generation with a multi-row streamer, sequence
    classification; BF16, FP16 and FP32 weights; CPU and CUDA.
  - [x] 9.3 openvino backend: `OVModelForSeq2SeqLM` and `OVModelForSequenceClassification`
    with `INFERENCE_PRECISION_HINT` from config.
  - [x] 9.4 Dockerfiles: `torch-cpu` (DLC `pytorch:2.14.0-cpu-amzn2023`), `torch-cuda`
    (DLC `pytorch:2.14.0-cu133-amzn2023`), `openvino` (pinned optimum-intel stack).
    - Written with digest-pinned bases and hashed locks; first built by `cdk deploy`.
  - [x] 9.5 Contract tests locally on CPU with small checkpoints (a small T5 and a small
    RoBERTa classifier) using the OpenAI Python client and the task-5 client.
    - _Requirements: 3.2_

- [x] 10. OpenVINO export image and probe (`servers/export`)
  - Pinned Dockerfile (python 3.12, `optimum-intel==2.2.0`, `openvino==2026.4.1`,
    `openvino-tokenizers==2026.4.1.0`, `nncf` pinned, `transformers` 5.5.x,
    `torch==2.14.0+cpu`).
  - `export.py`: decoders via OVMS `export_model.py` layout, Flan-T5 and classifier via
    `optimum-cli`; export-config hash in the S3 path; timing written for bring-up logs.
  - `probe_ov.py`: compile with the cell's hints and `PERF_COUNT=YES`, dump execution
    types and runtime precision.
  - _Requirements: 2.1, 12.3, 17.1_

- [x] 11. Host scripts (`host/`)
  - `host_facts.sh`, `serve.sh` (start/stop/status per engine, publish to the private IP
    only, health and 1-token check, phase timings), `stage_model.sh` (S3 cache first, HF
    download with the token second, token never logged), `probe_amx.sh` (verbose restart,
    5 requests, collect evidence), `perf_ctl.sh` (start/stop `perf stat`),
    `probe_membw.sh` (STREAM triad, threads = physical cores), `seccomp-vllm.json`,
    `cloudwatch-agent.json` (CPU per core, memory, `nvidia_gpu`).
  - shellcheck clean; a dry-run mode that prints the docker commands for review.
  - _Requirements: 2.6, 11.1, 11.2, 12.1-12.3, 17.1, 18.2_

- [x] 12. Lambda functions (`lambdas/`)
  - [x] 12.1 `preflight`: config resolution, image digest and AMI checks, Lambda
    concurrency check, AZ offering check, HF gated access check, FP16 flag check, price
    snapshot, `run.yaml` and `resolved.json`.
    - _Requirements: 1.4, 4.4, 5.8, 6.2, 10.2_
  - [x] 12.2 `target_control` (start, wait, configure cell via SSM, capacity-error log)
    and `reaper` (idle and max-uptime stops, tag-conditioned).
    - _Requirements: 2.7, 19.2_
  - [x] 12.3 `warmup`, `plan_wave`, `worker` (warm mode, barrier, pre-connect, closed loop,
    time budget, gzip JSONL to S3, DynamoDB counters, cold flag), `wave_check`.
    - _Requirements: 5.1-5.7, 5.9, 7.3_
  - [x] 12.4 `perf_counters`, `probes` (AMX and memory bandwidth via SSM), `quality`.
    - _Requirements: 11.2, 12.2, 12.3, 13.1_
  - [x] 12.5 `aggregate_level` (metrics, CloudWatch `GetMetricData`, flags) and
    `aggregate_run` (verdicts, SLA, capacity equivalence, reports).
    - _Requirements: 8, 9, 11.1, 14.1-14.5_
  - [x] 12.6 Unit tests with botocore Stubber; the worker tested end to end against the
    fake server.

- [x] 13. Reports (`tools/render_report.py`, used by `aggregate_run`)
  - `summary.csv`, `summary.json`, `summary.md` with the Requirement 14.1 columns plus
    prefill, SLA highlighting, capacity-equivalence table, price source block, capacity
    errors, warnings and smoke label; matplotlib charts from design section 16.
  - Golden-file tests on a synthetic run.
  - _Requirements: 10.3, 14.1-14.5, 1.5, 7.7_

- [x] 14. Migration calculator (`bench/estimator.py`)
  - Rule matching (widest range wins), add-ons, familiarity multiplier, calibration from
    journal entries, risks, capacity-driven recommendation from benchmark summaries
    (design section 18).
  - Tests covering every base rule, add-on, calibration clamp and verdict branch.
  - _Requirements: 16.1-16.5, 17.3_

- [x] 15. CDK app (`infra/`)
  - [x] 15.1 `NetworkStack`: VPC, one private and one public subnet in the configured AZ,
    NAT gateway, S3 and DynamoDB gateway endpoints, `sg-workers`, `sg-targets`.
    - _Requirements: 4.3, 18.3_
  - [x] 15.2 `CoreStack`: S3 bucket (Block Public Access, TLS-only, SSE-S3), DynamoDB table,
    ECR repo and thin-server image assets, all Lambdas with least-privilege roles, run and
    level state machines (Distributed Map, Standard children), reaper schedule, imported
    HF token secret, project tags.
    - _Requirements: 4.1, 5.1, 18.1, 18.2, 18.5, 19.1_
  - [x] 15.3 `TargetStack` (one per target): instance with IMDSv2, no public IP, no key
    pair, encrypted gp3, user data, instance role, SSM parameter with the instance ID, AMI
    ID as output.
    - _Requirements: 2.1, 4.2, 18.3, 18.4_
  - [x] 15.4 CDK assertion tests: no public IPs, ingress on 8000 only from `sg-workers`,
    no `0.0.0.0/0` ingress, IMDSv2 required, S3 Block Public Access and TLS policy, tags
    on every resource, reserved concurrency on the worker.
    - _Requirements: 18.1-18.5, 19.1_

- [x] 16. Command-line interface (`bench` entry point)
  - `bench validate`, `bench estimate`, `bench run`, `bench status`, `bench report`,
    `bench stop`; same YAML as the UI.
  - _Requirements: 6.1, 15.4_

- [x] 17. Streamlit UI (`ui/`)
  - Pages: New run (YAML preview, plan, duration and cost estimate, preflight), Runs
    (progress, stop with confirmation), Results (table, SLA, capacity equivalence, charts,
    CSV/Markdown export), Migration calculator, Bring-up journal.
  - _Requirements: 15.1-15.5, 16, 17.2_

- [x] 18. README
  - Deploy, run, teardown, required IAM for the operator, HF token secret, cost per run
    (formula plus the smoke estimates), the methodology rules, the DSPM/DLP suite and
    dataset handling, and how to read INVALID and smoke labels.
  - _Requirements: 20.1_

- [x] 19. DSPM/DLP policy and taxonomy
  - Write `config/dspm/policy_v1.md` (categories, decision rules, priority order, worked
    examples to be filled from the dev set), the Flan-T5 short form, and
    `config/dspm/span_label_map.yaml` for Nemotron-PII and Gretel span labels.
  - Policy loader computes and records the SHA-256 of both files.
  - _Requirements: 21.3, 22.2_

- [x] 20. Synthetic DSPM/DLP dataset (`tools/dspm/`)
  - [x] 20.1 Source loaders for Nemotron-PII and Gretel at pinned revisions; span
    rewriting (negatives) and masking (hard negatives).
  - [x] 20.2 Seeded generator with pinned Faker: chat threads, GenAI prompts, emails, code,
    configuration, logs, CSV tables, confidential memos; real-format fake secrets; cards
    with valid check digits; one generator per hard case in research.md 14.8.
  - [x] 20.3 Labeling from the policy and span map; chunking to per-model token targets;
    byte counts; template-family and source-document splits; dev and test sets with the
    default sizes and quotas.
  - [x] 20.4 Build checks from design 24.3 (fail the build on any miss), `manifest.json`,
    `review_sample.md`; upload to S3 is done in task 26; `data/` stays out of git.
  - [x] 20.5 Tests: same seed gives the same dataset hash; quotas met; no split leakage;
    every hard-case type present.
  - _Requirements: 22.1-22.8_

- [x] 21. DSPM/DLP request shapes and engines
  - [x] 21.1 Client modes: narrow-label generation (policy as system message,
    `max_tokens` 4/8, stop at newline, strict parser, fail-closed scoring), the optional
    vLLM `structured_outputs.choice` variant, Flan-T5 options prompt, M3 NLI hypothesis
    batches, D2 multi-label `/classify`.
    - _Requirements: 21.3, 21.4_
  - [x] 21.2 Thin server: `POST /detect` for D1 (token classification with the model
    card's Viterbi span decoding, compact response) and multi-label classification for D2;
    contract tests with small checkpoints.
    - _Requirements: 21.2, 3.2_
  - [x] 21.3 Catalog entries for D1 and D2, DSPM cells from design 24.5, prefix caching on
    for vLLM and OVMS DSPM cells, `config/runs/dspm-smoke.yaml` and `dspm-full.yaml`.
    - _Requirements: 21.2, 21.5_

- [x] 22. DSPM/DLP scoring (`bench/dspm_scoring.py`)
  - Recall, precision, false positive rate, F1, per-category recall (strict and lenient),
    hard-negative false positive rate, format adherence, Wilson 95% intervals, CPU-vs-GPU
    agreement, the suite's verdict rule and fitness floors; M3 threshold chosen on dev only.
  - Tests with hand-computed confusion matrices and interval values.
  - _Requirements: 23.1-23.4_

- [x] 23. DSPM/DLP report and UI
  - Report sections from design 24.7 ($/GB, GB/hour, documents/s charts, accuracy and
    hard-negative tables, D1/D2 coverage, Macie and Google reference block, transfer
    statement, CC BY 4.0 attribution); SLA presets and dataset/policy hashes in the UI.
  - _Requirements: 21.6-21.9, 22.7_

- [x] 24. Checkpoint: local verification (2026-11-09: ruff, 263 tests, synth of 8 stacks with
  real bundling, dataset hash `48a1b295…` reproduced, security sweep clean; review sample
  approved)
  - `ruff`, full `pytest`, `cdk synth` for all stacks, a local dataset build with checks
    passing, and the security sweep from the builder-security policy (search for
    `0.0.0.0`, `-p`/`ports:` publishes, `--host`, ingress rules, public bucket settings;
    each hit must be private, loopback or gated). Confirm `data/` and generated secrets are
    ignored by git.
  - You review `review_sample.md` (100 items) before the test set is frozen.
  - Stop and report results before any AWS change.

- [x] 25. Deploy and bring up (AWS, account `346399954218`; go-ahead 2026-11-09)
  - [x] 25.1 HF token stored as `amx-gpu-bench/hf-token` in Secrets Manager (2026-11-09),
    Hugging Face user `MichaelShapira`; reused as decided.
  - [x] 25.2 Gemma license accepted: the token gets HTTP 200 for `google/gemma-2-2b-it`
    (checked 2026-11-09), and for D1 and D2.
  - [x] 25.3 `cdk deploy` of all 8 stacks (Network, Core with the four images, six target
    launch templates), 2026-11-10. Verified: no public IPs, IMDSv2 required, no key pairs,
    targets SG admits 8000 only from the workers SG, bucket returns 403 anonymously.
    Instances are launched per run (Spot by default) from the launch templates.
    - _Requirements: 4.2, 18.3, 18.5_
  - [ ] 25.4 Bring-up checks from research.md section 12 that the smoke runs depend on
    (AMX flags and `exe.amx_busy` on m8i.4xlarge, GPU driver and CUDA compatibility, vLLM
    0.33.0 healthy on both, prompt-token parity for M1). Record SSE fixtures for task 5.3.
    Log bring-up phases and journal entries.
    - Done: AMX flags present and oneDNN trace 5,511/5,511 AMX GEMMs on m8i.4xlarge; vLLM
      healthy on g6.2xlarge (L4) and m8i.4xlarge; zero prompt-token mismatches; bring-up
      phases logged. `exe.amx_busy` is not exposed on m8i (verdict uses the trace). Open:
      SSE fixtures from the live engines, journal entries.
    - _Requirements: 12.1, 12.2, 17.1, 17.2_

- [ ] 26. Smoke runs and sample report (deliverable 6)
  - [x] Upload the frozen dataset to S3 (2026-11-09, `bench upload-artifacts`).
  - [x] Standard smoke T1 (g6.2xlarge Spot) vs T2 (m8i.4xlarge): `reports/sample/`.
  - [x] DSPM smoke on T2 (Spot): `reports/sample/dspm/`; T1 had no GPU capacity in that run.
  - [ ] DSPM smoke with both targets (started 2026-11-10 after the g6.2xlarge alternate
    was deployed).
  - Standard smoke: `config/runs/smoke-m1.yaml` (M1, T1 vs T2 m8i.4xlarge, BF16,
    C in {1, 4, 16}, 50 requests per level, 1 repeat, quality on).
  - DSPM smoke: `config/runs/dspm-smoke.yaml` (M1 on W1 and W3, T1 vs T2, BF16, 300 test
    items, 50 requests per level, 1 repeat).
  - Check the outputs against the requirements: AMX VERIFIED on T2, no prompt-token
    mismatches, smoke label present, price source shown, capacity equivalence computed,
    accuracy intervals and CPU-vs-GPU agreement present.
  - Copy `summary.md`, `summary.csv` and charts to `reports/sample/`; stop the targets.
  - _Requirements: 20.2, 21, 23_

- [ ] 27. Remaining targets and full runs (after you review the smoke reports)
  - Deploy T2 c8i.4xlarge, T2 m8i.8xlarge, T3, T4 (and T5 if enabled), run the remaining
    bring-up checks from research.md section 12, then the full standard run and the full
    DSPM/DLP run. Destroy target stacks when done.
  - _Requirements: 2.1, 4.2, 21, 22, 23_

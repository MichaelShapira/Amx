# Requirements: CPU (Intel AMX) vs GPU inference benchmark harness and migration effort calculator

Revision 2 (2026-11-09): applies the design-review answers. Models are capped at 4B
parameters, M3 is a classification encoder, and the narrative is driven by GPU capacity
rather than cost. Deployment target is account `346399954218`, us-east-1.

Revision 3 (2026-11-09): adds a DSPM/DLP suite (Requirements 21-23) for a cybersecurity
customer, next to the standard suite. Narrow-label classification (Sensitive / Not
sensitive, ALLOW / BLOCK, category) on synthetic data is tested in depth.

## Introduction

The harness measures inference of models with 4B parameters or less on NVIDIA GPU
instances (g6.xlarge, g5.xlarge) and on Intel AMX CPU instances (m8i, c8i, with m7i/c7i
as fallback). It also estimates the engineering effort to move a customer's models from
NVIDIA to AMX. Results back an internal AWS narrative whose main argument is that CPU
capacity is far easier to obtain than G5/G6 capacity. Accuracy and reproducibility take
priority over polish.

Research findings that shaped these requirements are in [research.md](research.md).

## Glossary

- **Model (M1-M5):** an entry in the model catalog. Default slots: M1 Qwen2.5-3B-Instruct,
  M2 Qwen3-4B-Instruct-2507, M3 roberta-large-mnli (classifier), M4 gemma-2-2b-it,
  M5 flan-t5-xl.
- **Target (T1-T5):** an EC2 instance type plus a serving engine. T1 vLLM GPU, T2 vLLM CPU,
  T3 OpenVINO (OVMS or optimum-intel), T4 native PyTorch, T5 vLLM CPU on m7i.
- **Precision:** BF16, FP16, INT8 or INT4, with weights and activations recorded
  separately.
- **Cell:** one model x target x precision combination.
- **Level:** one concurrency value (or batch size, for the classifier) inside a cell.
- **Wave:** one synchronized burst of N concurrent load-generator workers.
- **Run:** the execution of one run config file.
- **SLA:** user-defined limits on result metrics, for example P99 TTFT < 2000 ms.
- **Capacity equivalence:** the number of CPU instances, and Standard vCPUs, needed to
  match one GPU instance's SLA-meeting throughput.
- **Standard suite:** the generic benchmark (prompt sets of fixed lengths, fixed output
  length) in Requirements 1-20.
- **DSPM/DLP suite:** the use-case benchmark in Requirements 21-23: narrow-label
  sensitivity classification of synthetic documents, messages, prompts, code and tables.
- **D1, D2:** DLP-specific detector models used only in the DSPM/DLP suite
  (openai/privacy-filter and Roblox/roblox-pii-classifier-v2).
- **Policy file:** the plain-text definition of what counts as sensitive, used both to
  label the dataset and as the instruction prompt for generative models.
- **Hard negative:** non-sensitive content that looks sensitive (masked values,
  documentation examples, look-alike numbers).

## Requirement 1: Configurable model catalog

**User story:** As a benchmark operator, I want models defined in a config file, so that I
can add or swap models without code changes.

1. THE SYSTEM SHALL read models from a YAML catalog with: id, Hugging Face repo, pinned
   revision (commit SHA), type (decoder, encoder-classifier, token-classifier,
   encoder-decoder), suites the model belongs to (standard, DSPM/DLP), exact
   parameter count, nominal size, license, gated flag, max input tokens, prompt
   formatting rules, and the engines and precisions each target may run.
2. THE SYSTEM SHALL ship the M1-M5 defaults and the alternatives listed in research.md as
   catalog entries, with alternatives disabled by default.
3. WHEN a run references a model whose revision is not a commit SHA, THE SYSTEM SHALL
   reject the run.
4. WHEN a gated model is selected, THE SYSTEM SHALL check before the run that the stored
   Hugging Face token can read the pinned revision, and SHALL stop the run with a clear
   message if it cannot.
5. THE SYSTEM SHALL record each model's license in results, and SHALL show a warning in
   reports for non-commercial licenses (M1).
6. THE SYSTEM SHALL reject any model whose nominal size exceeds a configurable cap
   (default 4B), and SHALL count raw parameters, not "effective" parameters, for
   multimodal or per-layer-embedding models.

## Requirement 2: Inference targets

**User story:** As a benchmark operator, I want each target on its own EC2 instance with an
HTTP endpoint, so that targets can be measured independently and in parallel.

1. THE SYSTEM SHALL provide these targets, each defined in a YAML target catalog:
   - T1 g6.xlarge with vLLM GPU in BF16 (g5.xlarge as alternate);
   - T2 m8i.4xlarge and c8i.4xlarge with vLLM CPU in BF16, plus INT8 W8A8 for decoders on
     m8i.4xlarge;
   - T3 m8i.4xlarge with OpenVINO INT8 and INT4 weight compression for decoders and
     Flan-T5, and FP16 and INT8 for the classifier;
   - T4 m8i.4xlarge with native PyTorch and oneDNN in BF16, plus FP16 for the classifier;
   - T5 (optional) m7i.4xlarge with vLLM CPU in BF16;
   - one larger CPU size (m8i.8xlarge, vLLM CPU BF16) to show scaling.
2. THE SYSTEM SHALL run each model only on engines that support it. Because vLLM has no T5
   support, M5 SHALL run on PyTorch (GPU and CPU) and OpenVINO (CPU) only. Because OVMS
   has no classification endpoint, M3 SHALL run on vLLM (GPU and CPU), PyTorch (GPU and
   CPU) and OpenVINO through optimum-intel.
3. THE SYSTEM SHALL use the same vLLM version on GPU and CPU targets.
4. THE SYSTEM SHALL pin every container image by digest and every AMI by ID, and SHALL
   prefer AWS Deep Learning Containers where one exists for the engine.
5. THE SYSTEM SHALL NOT use Intel Extension for PyTorch (IPEX).
6. THE SYSTEM SHALL disable prefix caching on every engine during performance runs.
7. WHEN the preferred instance type cannot launch for lack of capacity, THE SYSTEM SHALL
   try the configured alternates (other AZ, then alternate type), and SHALL record which
   one was used and every capacity error it received.
8. Optional cells, off by default: vLLM CPU AWQ INT4 for decoders, and GPU W8A8.

## Requirement 3: Common API

**User story:** As a load-generator developer, I want every target to expose the same API
shape, so that one client measures all of them.

1. THE SYSTEM SHALL use OpenAI-compatible `/v1/completions` (and `/v1/chat/completions`
   where available) for generation, and vLLM's `/classify` request and response shape for
   classifiers.
2. WHERE an engine has no compatible server (seq2seq, or classifiers outside vLLM), THE
   SYSTEM SHALL provide a thin server with the same request and response shapes,
   including SSE streaming and a final usage chunk for generation.
3. THE SYSTEM SHALL support `stream_options.include_usage` and an option that forces a
   fixed output length (`ignore_eos` or equivalent) on every generation target.
4. THE SYSTEM SHALL expose a health endpoint on every target.

## Requirement 4: Infrastructure as code

**User story:** As an operator, I want one command to deploy or destroy each target, so
that idle capacity does not cost money.

1. THE SYSTEM SHALL define all infrastructure in AWS CDK (Python).
2. THE SYSTEM SHALL deploy each target as its own stack, deployable and destroyable with
   one command.
3. THE SYSTEM SHALL place targets and load-generator Lambdas in private subnets of one
   VPC, in a single configurable Availability Zone, and SHALL record the AZ ID.
4. THE SYSTEM SHALL make the account and region configurable (defaults `346399954218`,
   us-east-1) and SHALL check instance-type offerings in the chosen AZ before deploying a
   target.

## Requirement 5: Load generator

**User story:** As a benchmark operator, I want a Lambda fan-out that produces true
concurrency, so that results reflect the requested load.

1. THE SYSTEM SHALL fan out N concurrent worker Lambdas through a Step Functions
   Distributed Map, where N is the requested concurrency.
2. Each worker SHALL run a closed loop of requests for a request quota or a time budget.
3. THE SYSTEM SHALL pre-warm workers with a warm-up wave and SHALL start timing only after
   a shared start timestamp, so that cold starts do not affect measurements. Each record
   SHALL carry a cold-start flag for its execution environment.
4. Each worker SHALL wait for the shared start timestamp before its first request, and
   SHALL record how late it started.
5. Each worker SHALL record, per request: send time, first-token time, last-token time,
   end time (monotonic clock), wall-clock send time, server-reported prompt and completion
   token counts, HTTP status, and error type.
6. Each worker SHALL open its HTTP connection before the shared start time, so that
   connection setup is not part of the first request.
7. THE SYSTEM SHALL keep every request timeout and worker time budget below the
   15-minute Lambda limit, and SHALL continue a level in further waves when the quota is
   not reached in one wave.
8. Before a run, THE SYSTEM SHALL check unreserved Lambda concurrency against the highest
   requested concurrency plus headroom, and SHALL refuse runs that do not fit.
9. THE SYSTEM SHALL store raw per-request records in S3 as compressed JSONL.

## Requirement 6: Run configuration and reproducibility

**User story:** As a reviewer of the narrative, I want every result reproducible from one
file, so that numbers can be checked.

1. THE SYSTEM SHALL drive every run from a single YAML run config.
2. THE SYSTEM SHALL store the fully resolved config (image digests, AMI IDs, model
   revisions, prices, engine arguments, environment variables) with the run's results.
3. Every result row SHALL record: instance type, AMI ID and name, AZ ID, kernel version,
   CPU model and flags, NUMA node count, GPU driver and CUDA version (GPU), container image
   and digest, versions of vLLM, PyTorch, OpenVINO, OVMS, transformers and optimum-intel,
   engine arguments, relevant environment variables, model repo and revision, and the
   harness git commit.
4. WHEN the same run config is executed again, THE SYSTEM SHALL produce the same request
   sequence (prompt order and assignment to workers).

## Requirement 7: Benchmark methodology

**User story:** As a reviewer, I want the comparison to be fair, so that the narrative holds
up to scrutiny.

1. THE SYSTEM SHALL use the same prompts, max new tokens, sampling settings (temperature 0
   by default) and tokenizer across all targets for a model.
2. THE SYSTEM SHALL apply chat templates on the client side and send rendered text to
   `/v1/completions`, and SHALL flag any request whose server-reported prompt token count
   differs from the expected count.
3. THE SYSTEM SHALL send at least 20 warm-up requests to each endpoint and cell before
   measuring, and SHALL discard them.
4. THE SYSTEM SHALL run each level for a configured number of requests (default 200) or a
   fixed duration, and SHALL repeat each level a configured number of times (default 3).
5. THE SYSTEM SHALL report the variance across repeats (mean, standard deviation and
   coefficient of variation of P50, P99 and aggregate throughput), and SHALL flag levels
   whose coefficient of variation exceeds a configured threshold.
6. THE SYSTEM SHALL provide prompt sets of about 128, 1,024 and 4,096 input tokens per
   model tokenizer, capped by each model's maximum input length, and an optional set of
   about 14,000 tokens for models with at least 16K context (off by default).
7. WHEN a run uses fewer requests or repeats than the methodology defaults, THE SYSTEM SHALL
   label its results "smoke" in every output.
8. THE SYSTEM SHALL use one OpenMP thread per physical core as the default CPU thread
   policy, and SHALL record the effective binding.

## Requirement 8: Generation metrics

**User story:** As an analyst, I want a full latency and throughput profile per cell and
level, so that I can compare CPU and GPU.

1. For each model x target x precision x concurrency, THE SYSTEM SHALL report end-to-end
   latency P1, P50, P90, P99, mean, min and max.
2. THE SYSTEM SHALL report TTFT P1, P50 and P99 from streamed responses.
3. THE SYSTEM SHALL report TPOT (time per output token after the first) P1, P50 and P99.
4. THE SYSTEM SHALL report output tokens/s per request and aggregate output tokens/s
   across the system.
5. THE SYSTEM SHALL report requests/s, error rate and timeout rate.
6. THE SYSTEM SHALL report the effective concurrency achieved in each level.
7. THE SYSTEM SHALL compute percentiles exactly from all measured requests, not from
   approximations.
8. THE SYSTEM SHALL report prefill throughput (prompt tokens per second of TTFT, per
   request, and aggregate input tokens/s) separately from decode throughput.

## Requirement 9: Classifier metrics

1. For classifier models, THE SYSTEM SHALL report classifications/s and latency per batch
   (P1, P50, P99, mean) instead of token metrics, for configurable batch sizes (default
   1, 8, 32, 64) and concurrency levels.

## Requirement 10: Cost metrics

**User story:** As the narrative author, I want cost figures with a traceable price source,
so that cost statements can be audited.

1. THE SYSTEM SHALL compute USD per 1M output tokens and USD per 1K requests (USD per 1M
   classifications for classifiers) from the instance on-demand price.
2. THE SYSTEM SHALL read prices from the AWS Price List API for the chosen region at run
   time, and SHALL NOT hard-code prices.
3. Every report SHALL show the price, its source, the price list publication date, the
   effective date, and the SKU.
4. Cost SHALL be reported as information; it SHALL NOT decide the migration verdict
   (Requirement 16.4).

## Requirement 11: Server resource metrics

1. THE SYSTEM SHALL collect CPU utilization (total and per core), memory use, and for GPU
   targets GPU utilization, GPU memory use and power, for each level's time window.
2. THE SYSTEM SHALL measure memory bandwidth on each CPU instance type once at bring-up.

## Requirement 12: AMX verification

**User story:** As a reviewer, I want proof that AMX actually ran, so that CPU results are
not silently AVX-512 results.

1. THE SYSTEM SHALL capture the CPU flags (`lscpu`) on every CPU target and SHALL confirm
   `amx_bf16`, `amx_int8` and `amx_tile` (and `amx_fp16` on Xeon 6, required for FP16
   cells).
2. THE SYSTEM SHALL collect the `exe.amx_busy` hardware counter during each measured CPU
   level where the instance exposes it.
3. THE SYSTEM SHALL capture engine-level evidence once per CPU cell, outside the measured
   window: `ONEDNN_VERBOSE` output for PyTorch and vLLM, and OpenVINO execution types and
   runtime precision for OpenVINO.
4. THE SYSTEM SHALL mark each CPU result "AMX verified" only when the flags are present
   and either the hardware counter or the engine evidence shows AMX kernels executing.
5. WHEN AMX is not verified, THE SYSTEM SHALL mark the CPU result invalid, keep it out of
   SLA and cost rankings, and show it as invalid in every report.

## Requirement 13: Quality check

**User story:** As the narrative author, I want to know whether CPU or quantized outputs
differ from GPU outputs, so that a CPU configuration is not a worse one.

1. THE SYSTEM SHALL run 50 fixed prompts per model on each cell, with temperature 0 and
   natural stopping, and SHALL compare outputs with the GPU BF16 reference for that model.
2. For generation, THE SYSTEM SHALL report exact-match rate and ROUGE-L F1 (mean and
   10th percentile).
3. For classifiers, THE SYSTEM SHALL report label agreement rate and the maximum and mean
   absolute difference in class probabilities, against the GPU BF16 reference and
   against a PyTorch FP32 CPU reference.
4. THE SYSTEM SHALL flag FP16, INT8 and INT4 cells whose quality falls below a configurable
   threshold, measured against the references and against the CPU BF16 result.

## Requirement 14: Results, reports and charts

1. THE SYSTEM SHALL aggregate results into a benchmark table with at least: Model, Target,
   Instance, Precision, Concurrency, P1/P50/P99 E2E (ms), P1/P50/P99 TTFT (ms), P50 TPOT
   (ms), aggregate tok/s, req/s, error %, $/hr, $/1M tokens, AMX verified, quality delta.
2. WHEN the user defines an SLA, THE SYSTEM SHALL highlight, per model, the valid CPU
   configurations that meet it, and among those the cheapest one, next to the best GPU
   configuration.
3. THE SYSTEM SHALL chart latency vs concurrency and $/1M tokens vs concurrency, with CPU
   and GPU targets on the same chart.
4. THE SYSTEM SHALL export results as CSV and Markdown.
5. For every SLA-meeting CPU configuration, THE SYSTEM SHALL report capacity equivalence:
   CPU instances and Standard vCPUs needed to match one GPU instance's SLA-meeting
   throughput, and the resulting hourly cost compared with the GPU instance.

## Requirement 15: User interface

**User story:** As an operator, I want a simple UI to configure, run and review
benchmarks.

1. THE UI SHALL let the user select models and targets; enter or paste a prompt or pick a
   prompt set (short, medium, long); and set max new tokens, temperature (default 0),
   concurrency levels, and requests per level or duration.
2. THE UI SHALL show the planned cells, the estimated duration and the estimated cost
   before a run starts.
3. THE UI SHALL start a run, show live progress, and show and export results.
4. THE UI SHALL produce the same YAML run config that the command line uses, and SHALL
   store it with the run.
5. THE UI SHALL include the migration calculator page (Requirement 16) and the bring-up
   journal (Requirement 17).

## Requirement 16: Migration effort calculator

**User story:** As a solutions architect, I want an effort estimate in engineer-days for
moving a customer from NVIDIA to AMX, so that I can scope the migration.

1. THE CALCULATOR SHALL accept: number and type of models; current serving stack
   (PyTorch/Transformers, SageMaker PyTorch endpoint, vLLM GPU, TGI, Triton +
   TensorRT-LLM); CUDA-only dependencies (flash-attention, bitsandbytes, custom CUDA or
   Triton kernels, TensorRT engines); target precision (BF16, INT8, INT4); infrastructure
   (EKS, ECS, EC2, SageMaker) and whether CI/CD and IaC exist; team familiarity with CPU
   tuning (low, medium, high).
2. THE CALCULATOR SHALL read baseline efforts and add-ons from a tunable config file,
   seeded with the values in the task brief.
3. THE CALCULATOR SHALL output a low, likely and high estimate, a per-task breakdown, a
   risk list, and a recommendation of Go, Conditional or No-go.
4. THE CALCULATOR SHALL base the recommendation on measured benchmark results for the
   matching or nearest benchmarked model and the user's SLA, with capacity as the
   deciding factor:
   - Go when an AMX-verified, quality-passing CPU configuration at the requested
     precision meets the SLA;
   - Conditional when the SLA is met only with quantization below the requested
     precision, only within a limited concurrency range, or only by scaling out to more
     CPU instances than a configurable limit per replaced GPU instance;
   - No-go when no valid CPU configuration meets the SLA.
   The CALCULATOR SHALL show the cost difference against GPU and the capacity
   equivalence for every verdict, without letting cost change the verdict.
5. THE CALCULATOR SHALL show which inputs and which benchmark results drove each part of
   the estimate.

## Requirement 17: Calibration from real bring-up data

1. THE SYSTEM SHALL log machine-measured bring-up phases for every cell: image pull, model
   download, export or conversion, server start to healthy, and first successful request.
2. THE SYSTEM SHALL let an engineer record actual effort (hours), issues hit and fixes for
   bringing each model up on each target.
3. THE CALCULATOR SHALL use these records to adjust its estimates, and SHALL show the
   calibration source.

## Requirement 18: Security

1. THE SYSTEM SHALL grant each role only the permissions it uses.
2. THE SYSTEM SHALL keep the Hugging Face token in AWS Secrets Manager, readable only by
   the roles that download models, and SHALL never write it to logs or results.
3. Inference endpoints SHALL have no public IP and SHALL accept traffic only from the
   load-generator security group.
4. Operator access to instances SHALL use SSM Session Manager; no SSH keys.
5. S3 buckets SHALL block all public access, enforce TLS and encrypt at rest.
6. A locally run UI SHALL bind to 127.0.0.1 only and use the operator's own AWS
   credentials.

## Requirement 19: Tagging, auto-stop and cost control

1. THE SYSTEM SHALL tag all resources with project, owner and environment tags, and run
   resources with the run ID.
2. THE SYSTEM SHALL stop target instances that have had no benchmark activity for a
   configurable idle period (default 30 minutes), and SHALL stop any target that exceeds
   a configurable maximum uptime when no run is active.

## Requirement 20: Documentation and deliverables

1. THE SYSTEM SHALL include a README covering deployment, running a benchmark, teardown and
   expected cost per benchmark run.
2. THE SYSTEM SHALL include a sample results report (Markdown and CSV) from a short smoke
   run of M1 comparing T1 and T2, labeled as smoke.

## Requirement 21: DSPM/DLP benchmark suite

**User story:** As the author of a narrative for a DSPM and DLP vendor, I want the benchmark
to run their real classification shapes, so that the CPU-vs-GPU results apply to their
product.

1. THE SYSTEM SHALL provide a DSPM/DLP suite in addition to the standard suite, with these
   workloads: W1 bulk-scan binary, W2 bulk-scan category, W3 inline DLP binary (chat
   messages, emails, GenAI prompts), W4 structured column classification, and W5 redaction
   (optional, off by default).
2. THE SYSTEM SHALL run M1, M2, M4 and M5 as policy-prompted classifiers, M3 as a zero-shot
   NLI baseline, D1 (openai/privacy-filter) as a span detector, and D2
   (Roblox/roblox-pii-classifier-v2) on W3 only.
3. For generative models, THE SYSTEM SHALL use one frozen policy prompt shared by all
   models, ask for exactly one label, cap output at a configurable small token count
   (default 4 for binary, 8 for category), stop naturally, and record whether each output
   parsed to a valid label.
4. WHERE an engine supports constrained choice decoding, THE SYSTEM SHALL offer it as an
   optional variant, labeled as such.
5. THE SYSTEM SHALL run the suite with prefix caching on for engines that support it,
   because every request shares the policy prefix, and SHALL record the setting per cell.
6. THE SYSTEM SHALL report documents/s, latency P1/P50/P99 (E2E and TTFT), prefill tok/s,
   $/1M documents, $/GB of text classified, GB/hour per instance, and capacity
   equivalence in documents/s.
7. THE SYSTEM SHALL provide SLA presets (inline strict, inline standard, bulk scan) that
   the user can edit.
8. Every DSPM/DLP report SHALL state that speed results transfer to fine-tuned models of
   the same architecture, size and precision, and accuracy results do not.
9. THE SYSTEM SHALL show Amazon Macie and Google Sensitive Data Protection list prices per
   GB as context, labeled as not like-for-like.

## Requirement 22: Synthetic DSPM/DLP dataset

**User story:** As a benchmark operator, I want a realistic labeled dataset that contains no
real personal data, so that the DSPM/DLP suite can be run and shared safely.

1. THE SYSTEM SHALL build the dataset from public synthetic sources (Nemotron-PII,
   CC BY 4.0; Gretel synthetic PII finance, Apache-2.0) and a seeded generator for chat
   messages, GenAI prompts, emails, source code, configuration files, logs and tables
   that uses fake values only.
2. THE SYSTEM SHALL label every item from the policy file: a binary label and one or more
   categories (government ID, personal data, financial, health, credentials, confidential
   business), derived deterministically from generator metadata and span labels.
3. Hard negatives SHALL make up at least 30% of negatives, covering every hard case in
   research.md section 14.8.
4. THE SYSTEM SHALL split the data into a dev set (for policy tuning) and a test set,
   separated by template family and source document, and SHALL freeze the test set and
   the policy file with recorded hashes before any measured run.
5. The default test set SHALL have at least 3,000 items, balanced between sensitive and
   not sensitive, with at least 150 positives per category and per workload; the default
   dev set 300 items.
6. THE SYSTEM SHALL run automatic dataset checks (label consistency, check digits on
   positive card numbers, masking on masked hard negatives, no duplicates across splits)
   and SHALL produce a 100-item sample for human review.
7. THE SYSTEM SHALL store generated data in S3 and outside git, because it contains
   realistic-looking fake secrets, and SHALL carry the CC BY 4.0 attribution in reports.
8. THE SYSTEM SHALL rebuild the same dataset from the same seed and source revisions.

## Requirement 23: DSPM/DLP accuracy measurement

1. THE SYSTEM SHALL score every DSPM/DLP cell on the full test set for its workload with
   recall (block rate), precision, false positive rate, F1, per-category recall,
   hard-negative false positive rate and format adherence, each with a Wilson 95%
   confidence interval.
2. THE SYSTEM SHALL report per-item label agreement between each CPU cell and the GPU cell
   of the same model and precision.
3. For the capacity verdict, a CPU configuration SHALL count as meeting the SLA only when
   it meets the latency or throughput preset and loses no more than a configurable amount
   of accuracy against the GPU cell of the same model (default: label agreement at least
   98% and recall drop at most 1 percentage point).
4. THE SYSTEM SHALL report absolute accuracy against configurable floors (default recall
   at least 0.90 and false positive rate at most 0.05) as model fitness, separately from
   the CPU-vs-GPU verdict.

## Out of scope

- Models above 4B parameters (nominal size; raw parameters for multimodal models).
- Fine-tuning models on the synthetic data (a customer's fine-tuned weights can be added
  to the catalog later; their speed will match the public checkpoint of the same shape).
- File parsing, OCR and connectors to data stores; the DSPM/DLP suite classifies text that
  has already been extracted.
- Training or fine-tuning.
- Multi-instance autoscaling tests (capacity equivalence is computed, not deployed).
- A hosted, authenticated UI. Hosting it would need Midway authentication in front of
  CloudFront; this is deferred unless requested.
- Spot and Savings Plans pricing (on-demand only).
- Engines other than vLLM, OpenVINO and native PyTorch (for example llama.cpp or the
  OpenInfer runtime).

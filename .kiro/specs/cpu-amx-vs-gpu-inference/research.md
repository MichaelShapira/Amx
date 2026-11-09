# Research: CPU (Intel AMX) vs GPU inference for models of 4B parameters or less

Date of research: 2026-11-09. Region: us-east-1.

Revision 2 (2026-11-09, after design review): model scope capped at 4B parameters, M3
is a classification encoder, and GPU capacity (not cost) is the main driver for the
narrative. Section 13 covers the two Intel case studies used for the extra-row and
quantization decisions.

Revision 3 (2026-11-09): section 14 adds DSPM and DLP research for a cybersecurity
customer, which drives a new DSPM/DLP benchmark suite with synthetic data.

AWS facts below were read live with read-only calls (`describe-instance-type-offerings`,
`describe-instance-types`, `pricing get-products`, `lambda get-account-settings`,
`service-quotas get-service-quota`, `ssm get-parameter`, `ec2 describe-images`,
`cloudformation describe-stacks`) using account `346399954218`, role `MLGod` (Isengard).
Software facts come from the projects' release APIs, container registries, docs and
source trees (vLLM v0.33.0, optimum-intel v2.2.0). Anything not verified this way is
marked **unverified**.

## 1. Findings that change the plan

1. **Hourly prices are close to parity, and capacity is the real argument.** g6.xlarge is
   $0.8048/h, m8i.4xlarge $0.84672/h, c8i.4xlarge $0.74968/h. The verdict logic therefore
   asks whether CPU meets the SLA at all; cost is reported but does not decide the
   verdict. The harness adds a capacity-equivalence figure: CPU instances (and Standard
   vCPUs) needed to replace one GPU instance at the SLA.
2. **vLLM does not support T5.** vLLM 0.33.0 lists no T5 architecture and says
   encoder-decoder models should be added through its plugin system. M5 (Flan-T5) runs on
   PyTorch and OpenVINO only, on CPU and on GPU.
3. **OpenVINO Model Server has no seq2seq servable and no classification endpoint.** OVMS
   text generation is decoder-only, and its GenAI endpoints cover chat, completions,
   embeddings and rerank. Flan-T5 and the classifier run on OpenVINO through optimum-intel
   inside our own thin server.
4. **optimum-intel 2.2.0 refuses to export ModernBERT.** `modernbert` is in its
   `ONNX_SUPPORTED_ARCHITECTURES` set, and `convert.py` raises "The OpenVINO export of
   modernbert models is not officially supported by optimum-intel and has been removed."
   RoBERTa is supported for `text-classification`. This decides the M3 pick (section 2.2).
5. **Qwen2.5-3B-Instruct is not Apache-licensed.** It uses the Qwen Research License
   ("Non-Commercial shall mean for research or evaluation purposes only"). Benchmarking is
   evaluation, so the test is fine. A customer production recommendation for this exact
   model needs a commercial license from Alibaba Cloud. Qwen3-4B-Instruct-2507 (added in
   revision 2) is Apache-2.0.
6. **Not every vLLM CPU GEMM goes through oneDNN.** In v0.33.0, dense BF16 projections
   above 1 MiB use oneDNN (`onednn_mm`); smaller weights use the SGL AMX kernel; attention
   picks an "amx" ISA when available. `ONEDNN_VERBOSE` sees the oneDNN part only. AMX
   verification therefore also uses a hardware counter (`exe.amx_busy`), which Intel's
   published matrix shows as available on m7i.2xlarge, m7i.24xlarge and m8i.16xlarge.
7. **Prefix caching must be off.** vLLM V1 enables prefix caching by default. Repeated
   benchmark prompts would hit the cache and make TTFT look better than it is.
8. **Some AWS-blog tuning is wrong for this stack.** `KMP_AFFINITY` only applies to Intel
   OpenMP; `DNNL_MAX_CPU_ISA=AVX512_CORE_AMX` caps oneDNN below AMX-FP16 on Xeon 6;
   `OMP_NUM_THREADS=$(nproc)` counts hyperthreads. Details in section 10.
9. **There is no AWS DLC for vLLM CPU or OVMS.** AWS DLCs exist for vLLM GPU 0.33.0 and
   PyTorch 2.14 CPU/CUDA. The CPU vLLM image comes from Docker Hub, OVMS from Intel.
10. **Engine choice matters as much as quantization on CPU.** OpenInfer's Xeon 6 study
    (section 13) saw 2.7-3.1x prefill differences between two AMX-enabled engines at the
    same INT8 quantization, with decode near parity. The harness therefore measures INT8
    in two engines (vLLM W8A8 and OpenVINO) and reports prefill throughput separately
    from decode.
11. **Full methodology runs on CPU are still long at 4B.** 200 requests per level at
    concurrency 1, repeated 3 times, can take hours per level for a 4B model on 16 vCPUs.
    The UI shows an estimate before a run starts (design, section 20).

## 2. Models

Parameter counts are the `safetensors.total` field from the Hugging Face Hub API. The
revision is the `main` commit at research time; the harness pins it. The cap is 4B
parameters by nominal size; Qwen3-4B counts 4.02B (0.6% above a strict 4.00B line).

### 2.1 Selected models

| Slot | Model | Params | Checkpoint dtype | License | Gated | Revision (pin) |
|---|---|---|---|---|---|---|
| M1 | Qwen/Qwen2.5-3B-Instruct | 3,085,938,688 | BF16 | Qwen Research License (non-commercial) | No | `aa8e72537993ba99e69dfaafa59ed015b17504d1` |
| M2 | Qwen/Qwen3-4B-Instruct-2507 | 4,022,468,096 | BF16 | Apache-2.0 | No | `cdbee75f17c01a7cc42f958dc650907174af0554` |
| M3 | FacebookAI/roberta-large-mnli | 356,412,419 | F32 | MIT | No | `2a8f12d27941090092df78e4ba6f0928eb5eac98` |
| M4 | google/gemma-2-2b-it | 2,614,341,888 | BF16 | Gemma | Yes (manual approval) | `299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8` |
| M5 | google/flan-t5-xl | 2,849,757,184 | F32 | Apache-2.0 | No | `7d6315df2c2fb742f0f5b556879d730926ca9001` |

### 2.2 Alternatives and exclusions

| Model | Params | License | Status | Reason |
|---|---|---|---|---|
| Qwen/Qwen2.5-7B-Instruct | 7.62B | Apache-2.0 | **Removed** (was M2) | Above the 4B cap |
| MoritzLaurer/ModernBERT-large-zeroshot-v2.0 | 395,833,346 | Apache-2.0 | Optional M3 alt (vLLM and PyTorch only) | ModernBERT-large backbone with a trained 2-class head; no OpenVINO export path (finding 4). Revision `a51e07b524299e309dd2b88d48b0cfa2bd9ec598` |
| answerdotai/ModernBERT-large | 395,881,664 | Apache-2.0 | Not used | Masked-LM base model with no classification head |
| google/embeddinggemma-300m | 302,863,104 | Gemma | Not used | Embedding model; M3 is classification. Only small community classifiers exist on it, and vLLM has no Gemma3Text classifier |
| BAAI/bge-large-en-v1.5 | 335,142,400 | MIT | Not used | Embedding model |
| Qwen/Qwen2.5-1.5B-Instruct | 1,543,714,304 | Apache-2.0 | Optional M4 alt | Non-Gemma 2B-class fallback |
| google/gemma-4-E2B-it | 5,123,178,051 raw | Apache-2.0 | **Excluded** | 5.1B raw parameters, above the cap |
| google/gemma-3n-E2B-it | 5,439,438,272 raw | Gemma | **Excluded** | 5.4B raw parameters, above the cap |
| Qwen/Qwen3.5-2B | 2,274,069,824 | Apache-2.0 | Not used | Same family as OpenInfer's model, but multimodal with hybrid linear attention, a different kernel path from the other decoders and not part of the customer's stack. Can be added later as a catalog entry |
| Qwen/Qwen3.5-4B | 4,659,865,088 | Apache-2.0 | **Excluded** | Above the cap |
| google/flan-t5-large | 783,150,080 | Apache-2.0 | Optional M5 alt | Smaller step |
| T5Gemma family | 2.1-3.8B near the M5 size | Gemma | Not used | No OpenVINO export config in optimum-intel 2.2.0 and no vLLM support |

### 2.3 Engine support per selected model

| Model | transformers 5.x | vLLM 0.33 GPU | vLLM 0.33 CPU | OpenVINO (optimum-intel 2.2.0) | OVMS 2026.4.1 |
|---|---|---|---|---|---|
| Qwen2.5-3B | Qwen2ForCausalLM | Yes | Architecture validated on Xeon (QwQ-32B uses Qwen2ForCausalLM); model itself not on the list | `qwen2` text-generation-with-past | LLM servable |
| Qwen3-4B-Instruct-2507 | Qwen3ForCausalLM | Yes | Qwen3-4B validated, with a Xeon 6 recipe | `qwen3` | LLM servable; OpenVINO org publishes Qwen3-4B int4/int8 IRs |
| roberta-large-mnli | RobertaForSequenceClassification | Yes (`/classify`) | Pooling runner on CPU (vLLM CPU embed benchmark exists); classifier on CPU **unverified** | `roberta` text-classification | No classification endpoint |
| gemma-2-2b-it | Gemma2ForCausalLM | Yes | Not on validated list. CPU attention supports sliding window and logit soft-cap (code check) | `gemma2` text-generation-with-past | LLM servable |
| flan-t5-xl | T5ForConditionalGeneration | **No** | **No** | `t5` text2text-generation-with-past | **No seq2seq servable** |

Sources: [vLLM supported models](https://docs.vllm.ai/en/stable/models/supported_models/),
[vLLM classification models](https://docs.vllm.ai/en/stable/models/pooling_models/classify/),
[vLLM CPU validated models](https://docs.vllm.ai/en/stable/models/hardware_supported_models/cpu/),
optimum-intel `v2.2.0` `optimum/exporters/openvino/{model_configs,utils,convert}.py`,
[OVMS LLM reference](https://docs.openvino.ai/2026/model-server/ovms_docs_llm_reference.html),
[OVMS GenAI endpoints](https://docs.openvino.ai/2026/model-server/ovms_docs_clients_genai.html).

### 2.4 Why these picks

- **M2 → Qwen3-4B-Instruct-2507.** With the 7B model removed, this is the largest decoder
  inside the cap. It is Apache-2.0, on vLLM's Xeon 6 validated list, and has a
  pre-quantized W8A8 checkpoint. The 2507 Instruct release is non-thinking, so outputs
  carry no reasoning tokens.
- **M3 → roberta-large-mnli.** It has a trained classification head (3-class NLI), is the
  closest trained classifier to 300M (356M), is MIT-licensed and ungated, and runs on every
  engine in scope: vLLM GPU and CPU (`RobertaForSequenceClassification`), OpenVINO
  (`roberta` text-classification), and PyTorch. Its context limit is 512 tokens. The
  ModernBERT classifier is the modern alternative but cannot run on the OpenVINO path.
  **Open point:** if the customer's classifier architecture is known, it should replace
  this proxy.
- **M4 → gemma-2-2b-it** (unchanged). Gemma 3 has no 2B size and both "E2B" models exceed
  the cap.
- **M5 → flan-t5-xl** (unchanged).

### 2.5 Dtype notes

- Use BF16 on every target by default. L4 and A10G support BF16, and Xeon AMX accelerates
  BF16.
- **FP16 on Xeon 6 for M3:** m8i/c8i support AMX-FP16. The Riskfuel study (section 13)
  ran OpenVINO with AMX-FP16 on Granite Rapids. M3 gets FP16 cells on 8i targets (T3, T4);
  m7i lacks `amx_fp16`. vLLM CPU is not used for FP16 because its docs call torch CPU FP16
  unstable.
- The Flan-T5 model card shows FP16 loading, but T5 v1.1-based checkpoints are known to
  overflow in FP16 (**unverified for flan-t5-xl in this session**). BF16 avoids the question.
- roberta-large-mnli and flan-t5 checkpoints are F32 and get cast at load. Record the cast.

### 2.6 Pre-quantized checkpoints

| Repo | License | Revision | Use |
|---|---|---|---|
| RedHatAI/Qwen2.5-3B-Instruct-quantized.w8a8 | none stated (base: Qwen research) | `cd4a8836f6c1e916f43d7c60c4dc3796b7aca18c` | vLLM INT8 W8A8 |
| RedHatAI/Qwen3-4B-Instruct-2507-quantized.w8a8 | Apache-2.0 | `31ec5013b3319d7ba82b2a354d6e48b376204dd3` | vLLM INT8 W8A8 |
| RedHatAI/gemma-2-2b-it-quantized.w8a8 | Gemma | `9bafbb745168370d51b0e7e2c9356d6bc2fd1a5d` | vLLM INT8 W8A8 |
| Qwen/Qwen2.5-3B-Instruct-AWQ | Qwen research | `3559b226e8ce77211e2c1bd7ddfb7686fec4d6dd` | optional AWQ INT4 |
| Qwen/Qwen3-4B-AWQ | Apache-2.0 | `74d4bd2bd4bff9cafc9345221320bffb08b406a3` | optional AWQ INT4 (base Qwen3-4B, not 2507) |
| OpenVINO/Qwen3-4B-int4-ov, -int8-ov | Apache-2.0 | | reference only; we export our own |

We export OpenVINO IRs ourselves with pinned tool versions so the quantization settings
are known. Third-party quantized checkpoints are recorded by repo and revision.

## 3. Software versions (as of 2026-11-09)

| Component | Version | Image / source | Digest |
|---|---|---|---|
| vLLM | 0.33.0 (tag; GitHub release API still shows `v0.33.0rc3` as latest, `v0.32.0` is the last final release object) | | |
| vLLM GPU (AWS DLC) | 0.33.0, CUDA 13.0, Ubuntu 24.04 | `public.ecr.aws/deep-learning-containers/vllm:0.33.0-gpu-py312-cu130-ubuntu24.04-ec2` | `sha256:6db6c433557b38d86d47f68e367532d41a8341e56fa2605e60c13d190a5c7d17` |
| vLLM GPU (upstream, fallback) | 0.33.0 | `vllm/vllm-openai:v0.33.0` | `sha256:d825baf9ce4b1b6bfc5ecc7b79880c7e6748c65a72d697ce3d47607aafa84c6e` |
| vLLM CPU | 0.33.0, torch 2.14.0+cpu, Ubuntu 22.04 | `vllm/vllm-openai-cpu:v0.33.0-x86_64` (Docker Hub) | `sha256:4154467a886f8b7ffa9dbab32f873f5ca4528409d042a6ec866e8888223263d3` |
| OVMS | 2026.4.1 (2026-10-06) | `openvino/model_server:2026.4.1` | `sha256:148e3fe802261ed0b5b634de3bbccdefe7176a218ec34a88e382b40704c34a1c` |
| OpenVINO / GenAI | 2026.4.1 / 2026.4.1.0 (2026-10-01) | PyPI | |
| optimum-intel | 2.2.0 (2026-09-17); needs `transformers>=4.51,<5.6`, `openvino>=2026.0`, `optimum~=2.3.0` | PyPI | |
| PyTorch | 2.15.0 latest (2026-10-28); 2.14.0 matches vLLM CPU | | |
| PyTorch CPU (AWS DLC) | 2.14.0, Amazon Linux 2023 | `public.ecr.aws/deep-learning-containers/pytorch:2.14.0-cpu-amzn2023` | `sha256:e9aa07c11afc5f7eff183782cf3dde397012e2f08ab700dffd2828203688c159` |
| PyTorch CUDA (AWS DLC) | 2.14.0, CUDA 13.3 | `public.ecr.aws/deep-learning-containers/pytorch:2.14.0-cu133-amzn2023` | pin at build |
| transformers | 5.22.2 latest; the OpenVINO path must stay below 5.6 | | |
| oneDNN | 3.14.2 latest; vLLM 0.33.0 builds oneDNN 3.13 statically with `ONEDNN_VERBOSE=ON` and `ONEDNN_ENABLE_MAX_CPU_ISA=ON` | | |
| IPEX | last CPU release 2.8.0 (2025-08-12). Do not use | | |
| CDK bootstrap in 346399954218 / us-east-1 | `CDKToolkit` stack `UPDATE_COMPLETE`, bootstrap version 32 | | |

AMIs in us-east-1 (resolved from public SSM parameters):

| Use | Name | AMI |
|---|---|---|
| GPU | Deep Learning Base OSS Nvidia Driver GPU AMI (Amazon Linux 2023) 20261106 | `ami-0fa51b517f148a6a4` |
| CPU | al2023-ami-2023.12.20261102.0-kernel-6.12-x86_64 | `ami-07c1b6f58fb89ce9f` |
| CPU (newer kernel) | al2023-ami-2023.12.20261102.0-kernel-6.18-x86_64 | `ami-0c230065b91270bc6` |

The September 2026 release of the GPU AMI shipped NVIDIA driver 595.91.07 and CUDA 13.2
([release notes](https://docs.aws.amazon.com/dlami/latest/devguide/aws-deep-learning-ami-gpubaseoss-al2023-2026-09-04.html)).
The November driver version is **unverified**; check `nvidia-smi` at bring-up. The
PyTorch CUDA DLC targets CUDA 13.3, which relies on CUDA minor-version compatibility with
that driver (**verify at bring-up**).

Sources: GitHub releases API, Docker Hub tags API, public ECR registry API,
[AWS DLC image list](https://github.com/aws/deep-learning-containers/tree/main/docs/src/data).

## 4. vLLM CPU backend (v0.33.0)

From the [CPU install docs](https://docs.vllm.ai/en/stable/getting_started/installation/cpu.html)
(page dated 2026-10-15) and the v0.33.0 source:

- **Dtypes:** FP32, FP16, BF16 on x86. Docs recommend `--dtype=bfloat16` because torch
  CPU FP16 is unstable.
- **Quantization on x86:** AWQ, GPTQ, compressed-tensors INT8 W8A8. An INT4 W4A8 path is
  enabled by default (`VLLM_CPU_INT4_W4A8=1` in `vllm/envs.py`). INT8 W8A8 GEMMs go to
  the SGL `int8_scaled_mm` AMX kernel when shapes allow (`check_cpu_sgl_kernel`).
- **AMX use (code):** `dispatch_cpu_unquantized_gemm` in
  `vllm/model_executor/layers/utils.py` routes BF16/FP16/INT8 GEMMs to the SGL AMX kernel
  when the weight is at most 1 MiB, and to oneDNN `onednn_mm` above that. AMX-FP16 has its
  own capability check (Sapphire Rapids lacks it). CPU attention selects the `amx` ISA for
  BF16 when `block_size % 32 == 0` (`vllm/v1/attention/backends/cpu_attn.py`). Set
  `VLLM_LOGGING_LEVEL=DEBUG` to see the `CPU unquantized GEMM dispatch:` lines.
- **Classification:** served by the pooling runner on `/classify`, which is a vLLM API, not
  an OpenAI one. CPU attention supports encoder-only attention.
- **Encoder-decoder:** CPU attention supports encoder and encoder-decoder attention types,
  but no T5 model exists. Whisper is the validated encoder-decoder on CPU.
- **Prebuilt artifacts:** x86 wheels since 0.17.0; Docker images on Docker Hub
  (`vllm/vllm-openai-cpu`). The image sets `LD_PRELOAD` to tcmalloc and Intel OpenMP.
- **Docker flags:** `--cap-add SYS_NICE --security-opt seccomp=unconfined` (for
  `get_mempolicy` and `migrate_pages`), `--shm-size=4g`. The docs suggest a custom seccomp
  profile (default plus `migrate_pages`) as the tighter option.
- **Threads:** `VLLM_CPU_OMP_THREADS_BIND=auto` (default) binds one OpenMP thread per
  physical core within a NUMA node and reserves CPUs for the frontend. Docs recommend
  reserving 1 CPU for the frontend when `world_size == 1`
  (`VLLM_CPU_NUM_OF_RESERVED_CPU=1`). On a 16-vCPU/8-core host, bind to logical CPUs 0-7 or
  8-15, not to hyperthread siblings. OpenInfer's study also used one thread per physical
  core.
- **KV cache:** `VLLM_CPU_KVCACHE_SPACE` (GiB) is now called legacy; use
  `--kv-cache-memory-bytes`.
- **Batching:** online defaults are `--max-num-batched-tokens 2048` and
  `--max-num-seqs 128` per rank; `--block-size` should be a multiple of 32 (default 128).
- **NUMA / TP:** set tensor-parallel size to the NUMA node count on multi-socket hosts.
  4xlarge and 8xlarge are expected to be one NUMA node (**verify with `lscpu`**).
- **Reference configs:** `.buildkite/performance-benchmarks/tests/serving-tests-cpu-text.json`
  and `serving-tests-cpu-embed.json` at v0.33.0 use BF16, `block_size 128`,
  `max_num_batched_tokens 2048`, `max_num_seqs 256`, temperature 0, `ignore-eos`,
  200 prompts, and concurrency 12-200. Gemma models run with `VLLM_CPU_SGL_KERNEL=0`, and
  Gemma 4 26B also with `VLLM_CPU_ATTN_SPLIT_KV=0`.
- **Recipes:** the CPU image includes `serve_with_recipe.sh --hardware xeon6`, which fetches
  the latest recipe from recipes.vllm.ai at run time (see contradiction 8).

## 5. OpenVINO and OVMS

- **OVMS 2026.4.1 endpoints:** `chat/completions`, `completions` and `responses` for text
  generation, `embeddings` and `rerank`, under `/v1` (2026 demos) and `/v3` (API
  reference). We probe both at bring-up and pin one.
- **Servables:** decoder-only LLM (continuous batching on CPU) and VLM. No seq2seq, no
  sequence classification.
- **Benchmark-relevant parameters:** `ignore_eos` and `stream_options.include_usage` are
  supported on `/completions` (continuous batching servables). Only a single string
  prompt is accepted. `enable_prefix_caching` defaults to false. `cache_size` (GB) set to 0
  means dynamic allocation. `max_num_seqs` defaults to 256 and `max_num_batched_tokens`
  to 256 with dynamic split-fuse on.
- **Precision on AMX hosts:** the CPU plugin executes FP32/FP16 IRs in half precision
  automatically when the platform supports it (execution mode PERFORMANCE); BF16 is the
  expected default on Xeon, and `INFERENCE_PRECISION_HINT=f16` selects FP16 (AMX-FP16 on
  Xeon 6). Weight-compressed INT8/INT4 models keep activations in floating point or
  dynamically quantized INT8 (`DYNAMIC_QUANTIZATION_GROUP_SIZE`). Results label weights
  and activations separately, and the probe reads the actual runtime precision.
  ([CPU device docs](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/cpu-device.html))
- **Export:** `optimum-cli export openvino --model <id> --weight-format int8|int4 <dir>`
  ([Optimum Intel 2.0 blog](https://huggingface.co/blog/jeffboudier/optimum-intel-v2)).
  optimum-intel 2.0 removed the IPEX and INC integrations and installs OpenVINO and NNCF by
  default. OVMS also provides `export_model.py` for its directory layout and
  `graph.pbtxt`.
- **Kernel visibility:** OpenVINO exposes per-node execution type through performance
  counters (`PERF_COUNT`), for example `brgemm_avx512_amx_bf16`. OVMS is a C++ server, so
  the check runs as a Python probe on the same IR and OpenVINO version.

## 6. Native PyTorch + oneDNN

- From the [AWS Compute Blog](https://aws.amazon.com/blogs/compute/accelerate-cpu-based-ai-inference-workloads-using-intel-amx-on-amazon-ec2/):
  BF16 weights or `torch.amp.autocast('cpu', dtype=torch.bfloat16)`, `ONEDNN_DEFAULT_FPMATH_MODE=bf16`
  (matters for FP32 models only), `ONEDNN_PRIMITIVE_CACHE_CAPACITY`, thread settings.
  The blog used PyTorch 2.8.0 with HF `generate()`, 5 iterations, latency including
  tokenization, in us-west-2.
- PyTorch pip wheels use GNU OpenMP. Thread pinning there uses `OMP_PROC_BIND` and
  `OMP_PLACES` (or `GOMP_CPU_AFFINITY`). `KMP_*` variables need Intel OpenMP preloaded.
- FP16 linear layers on CPU use oneDNN, which picks AMX-FP16 on Xeon 6 when
  `ONEDNN_MAX_CPU_ISA` does not cap it (**verify in the AMX probe**).
- **IPEX:** active development stopped after 2.8 with two more quarters of fixes
  ([GitHub #867](https://github.com/intel/intel-extension-for-pytorch/issues/867)); the
  install site says EOL by end of March 2026 ([pytorch-extension.intel.com](https://pytorch-extension.intel.com/));
  [INTEL-SA-01497](https://www.intel.com/content/www/us/en/security-center/advisory/intel-sa-01497.html)
  reports a privilege-escalation issue with no fix planned (read from the search snippet;
  the page returned HTTP 403 to the fetcher).

## 7. AMX verification methods

| Method | What it proves | Where it works | Cost |
|---|---|---|---|
| `lscpu` flags `amx_bf16 amx_int8 amx_tile` (+ `amx_fp16` on Xeon 6) | Hardware and kernel expose AMX | Any instance | None |
| `perf stat -e exe.amx_busy` | AMX units were busy during the run | Event listed as supported on m7i.2xlarge, m7i.24xlarge, m8i.16xlarge in [Intel's vPMU matrix](https://intel.github.io/optimization-zone/hardware/PMU/). m8i.4xlarge/8xlarge **unverified** | Counting mode, negligible |
| `ONEDNN_VERBOSE=1` | oneDNN primitives executed with an `*_amx*` implementation | PyTorch eager and vLLM's oneDNN GEMMs; not vLLM SGL kernels or attention | High; run outside the measured window |
| OpenVINO `PERF_COUNT` exec types | OpenVINO nodes ran AMX kernels | optimum-intel; OVMS via a probe on the same IR | Probe only |
| vLLM `VLLM_LOGGING_LEVEL=DEBUG` dispatch lines | Which GEMM backend each layer bound to | vLLM CPU | Startup only |

Notes:
- oneDNN prints the ISA in the implementation name, for example `brg_matmul:avx10_1_512_amx`
  with the AVX10.1 naming. Match on `amx`, not on one exact string.
- Linux needs kernel 5.16 or later for AMX state; AL2023 ships 6.12 and 6.18.

## 8. AWS facts (us-east-1, 2026-11-09)

### 8.1 Availability by AZ

| Type | us-east-1a (use1-az1) | 1b (az2) | 1c (az4) | 1d (az6) | 1e (az3) | 1f (az5) |
|---|---|---|---|---|---|---|
| g6.xlarge, g5.xlarge | yes | yes | yes | yes | no | yes |
| m8i.4xlarge/8xlarge, c8i.4xlarge/8xlarge | yes | yes | yes | yes | no | yes |
| m7i.4xlarge, c7i.4xlarge | yes | yes | yes | yes | no | yes |

8i is offered, so the m7i/c7i fallback is not needed in us-east-1. Offerings do not
guarantee capacity at launch time, which is the premise of the narrative.

### 8.2 Specs and on-demand prices

| Type | vCPU / cores | Memory | Accelerator | CPU | $/hour |
|---|---|---|---|---|---|
| g6.xlarge | 4 / 2 | 16 GiB | 1 x L4 (22,888 MiB) | AMD EPYC 7R13 | 0.8048 |
| g5.xlarge | 4 / 2 | 16 GiB | 1 x A10G (22,888 MiB) | AMD EPYC 7R32 | 1.0060 |
| m8i.4xlarge | 16 / 8 | 64 GiB | | Xeon 6 (Granite Rapids), 3.9 GHz | 0.84672 |
| m8i.8xlarge | 32 / 16 | 128 GiB | | same | 1.69344 |
| c8i.4xlarge | 16 / 8 | 32 GiB | | same | 0.74968 |
| c8i.8xlarge | 32 / 16 | 64 GiB | | same | 1.49936 |
| m7i.4xlarge | 16 / 8 | 64 GiB | | Xeon (Sapphire Rapids), 3.2 GHz | 0.8064 |
| c7i.4xlarge | 16 / 8 | 32 GiB | | same | 0.7140 |

Price source: AWS Price List API `GetProducts`, service `AmazonEC2`, filters
`regionCode=us-east-1, operatingSystem=Linux, tenancy=Shared, preInstalledSw=NA,
capacitystatus=Used`; OnDemand term effective 2026-11-01; publication
2026-11-07T00:28:52Z; price list `arn:aws:pricing:::price-list/aws/AmazonEC2/USD/20261107002852/us-east-1`.
The AWS compute blog quotes the same m8i/m7i prices for us-west-2.

With the 4B cap, the largest model (Qwen3-4B, about 8 GB in BF16) leaves ample memory for
KV cache even on c8i.4xlarge (32 GiB).

### 8.3 Account limits and quotas

- Lambda: `ConcurrentExecutions` 1000, `UnreservedConcurrentExecutions` 890.
- Lambda scales each function by up to 1,000 execution environments every 10 seconds
  ([scaling behavior](https://docs.aws.amazon.com/lambda/latest/dg/scaling-behavior.html)).
- EC2 On-Demand vCPU quotas: G and VT instances 768; Standard (A, C, D, H, I, M, R, T, Z)
  1152. These are separate pools: replacing one g6.xlarge (4 G vCPUs) with m8i.4xlarge
  instances draws 16 Standard vCPUs each. The capacity-equivalence figure uses this.
- Step Functions Distributed Map runs up to 10,000 parallel child executions; inline Map
  supports up to 40 concurrent iterations and 25,000 execution-history entries
  ([Map state](https://docs.aws.amazon.com/step-functions/latest/dg/state-map.html));
  Express workflows run at most 5 minutes
  ([Express workflows](https://docs.aws.amazon.com/wellarchitected/latest/serverless-applications-lens/step-functions-workflows.html)).

## 9. Expectations to test (hypotheses, not results)

- Single-stream decode is memory-bandwidth bound on both platforms. A BF16 4B model reads
  about 8 GB of weights per generated token. NVIDIA lists 300 GB/s for the L4 (datasheet;
  **not re-checked in this session**), which caps batch-1 decode near 37 tokens/s.
  Per-vCPU memory bandwidth for m8i.4xlarge is not published; the harness measures it.
  OpenInfer's Xeon 6 data shows the same pattern: decode near parity across engines,
  prefill sensitive to kernel quality.
- AMX helps compute-bound phases (prefill, encoders, larger batches) more than batch-1
  decode. Expect the strongest CPU results for M3 at larger batch sizes and for
  prefill-heavy prompts.
- INT8 weights halve bytes per token on CPU, so W8A8 and OpenVINO INT8 are the most likely
  way to close the decode gap for M1 and M2.

## 10. Contradictions between sources, and which to trust

| # | Conflict | Trusted source and why | Action |
|---|---|---|---|
| 1 | AWS blog sets `KMP_AFFINITY`. Stock PyTorch wheels load GNU OpenMP, which ignores `KMP_*`. | OpenMP runtime behavior; vLLM docs require preloading `libiomp5` before `KMP_*` matters | T4 uses `OMP_PROC_BIND=close`, `OMP_PLACES=cores`; record the loaded OpenMP runtime |
| 2 | AWS blog sets `OMP_NUM_THREADS=$(nproc)` (16 incl. hyperthreads). vLLM binds one thread per physical core and reserves 1 CPU; OpenInfer also used one thread per physical core. | vLLM docs and OpenInfer's data; for T4, measure | T4: one sweep of 8 vs 16 threads on M1, then fix the value in config |
| 3 | AWS blog sets `DNNL_MAX_CPU_ISA=AVX512_CORE_AMX`. oneDNN orders `AVX10_1_512_AMX` (= `AVX512_CORE_AMX`) below `AVX10_1_512_AMX_FP16`, so this caps Xeon 6 below AMX-FP16, which Riskfuel relied on. The documented variable name is `ONEDNN_MAX_CPU_ISA`. | [oneDNN dispatcher docs](https://uxlfoundation.github.io/oneDNN/dev_guide_cpu_dispatcher_control.html) | Leave unset (DEFAULT = best ISA); record the effective ISA from verbose output |
| 4 | AWS blog: batch 8-16 is best and saturates beyond 16; its own table recommends batch 1-8 for 3B+. vLLM docs: larger batch gives more throughput, tune `max-num-seqs`. | Neither for serving: the blog measured static HF `generate()` batching | Sweep concurrency 1-64 and report the curve |
| 5 | AWS blog: AMX overhead hurts batch 1 (DialoGPT -44%), but Llama-3.2-3B gains 27% at batch 1 in the same post | Both are data points; effect depends on model size | Measure C=1 for every model |
| 6 | vLLM docs: use `SYS_NICE` + seccomp change instead of `--privileged`. HF Gemma 4 Xeon blog uses `--privileged`. | vLLM docs (least privilege) | `SYS_NICE` plus a custom seccomp profile adding `migrate_pages` |
| 7 | vLLM docs call `VLLM_CPU_KVCACHE_SPACE` legacy; vLLM's own CPU benchmark configs still set it and also set `VLLM_CPU_SGL_KERNEL`, which `vllm/envs.py` v0.33.0 does not define | Source code | Use `--kv-cache-memory-bytes`; do not set `VLLM_CPU_SGL_KERNEL` |
| 8 | vLLM Recipes fetch the latest recipe at run time; the methodology requires reproducible runs | Reproducibility rule | Use a recipe once to derive settings, then pin explicit args |
| 9 | The provided HF Transformers CPU guide (`perf_infer_cpu`) is gone from v5.22.2 docs; the main-branch page only points to Optimum/ONNX Runtime | Current docs | Treat as stale |
| 10 | The provided `huggingface.co/blog/MatrixYao/xeon` link now resolves to "Run Gemma 4 on Intel Xeon Out-Of-the-Box", not general small-model guidance | Current page | Not needed now that Gemma 4 is excluded |
| 11 | Intel AMX Quick Start Guide (PDF) covers only 4th/5th Gen Xeon and has no verification method | oneDNN docs and Intel's Xeon 6 article (AMX adds FP16 on Xeon 6) | Do not rely on the PDF for Xeon 6 |
| 12 | Intel's EC2 M7i LLM blog (May 2024) predates IPEX EOL; page returned HTTP 403 | Not used | None |
| 13 | C8i launch blog: 2.5x memory bandwidth vs C7i; AWS compute blog: "up to 2.5x DDR5 bandwidth" m8i vs m7i. Both are whole-instance claims | Measure per instance size | Run a memory-bandwidth probe at bring-up |
| 14 | OVMS docs show both `/v1` and `/v3` routes | Probe | Pin the route that answers at bring-up |
| 15 | OpenInfer's headline is "3x faster" but its own table shows decode gains of 1.07-1.17x; the 2.7-3.1x is prefill only | OpenInfer's table | Report prefill and decode separately; never quote one combined speedup |

## 11. Intel programs and reference benchmarks

- **Funded POC programs:** no official public Intel page for an AWS-customer POC fund was
  found. A login-gated third-party site (`iapflow.com`, "Deal Desk") says an Intel-funded
  program pays customers through The Channel Company against Cost Explorer exports and
  that AWS makes no payments; it returned HTTP 401, so this is **unverified**. Ask the
  Intel account team or the AWS Intel partner team. AWS-funded POC credits go through
  account teams.
- **Reference benchmarks:** vLLM publishes a Xeon 6/5 validated list with recipes
  (Llama-3.1-8B, Llama-3.2-1B/3B, Qwen3-1.7B/4B/8B, granite-3.2-2b, Gemma 3/4) and a CPU
  [performance dashboard](https://hud.pytorch.org/benchmark/llms?repoName=vllm-project%2Fvllm&deviceName=cpu).
  The AWS compute blog has m8i vs m7i numbers for Llama-3.2-3B and Gemma-3-1B with HF
  `generate()`. The two Intel case studies in section 13 cover a 27B LLM on Xeon 6 SoC and a
  neural surrogate on x8i. Nothing published covers Qwen2.5-3B, Qwen3-4B-Instruct-2507,
  gemma-2-2b, roberta-large-mnli or flan-t5-xl serving on m8i against a GPU, so this
  harness fills that gap.

## 12. Items to verify at bring-up

1. `lscpu` shows `amx_fp16` on m8i/c8i, and NUMA node count is 1 on 4xlarge and 8xlarge.
2. `perf stat -e exe.amx_busy` works on m8i.4xlarge, c8i.4xlarge, m7i.4xlarge.
3. GPU AMI driver version; CUDA 13.0 (vLLM DLC) and 13.3 (PyTorch DLC) containers run on it.
4. OVMS route (`/v1` or `/v3`) and `ignore_eos` behavior.
5. roberta-large-mnli on vLLM CPU `/classify` returns the same labels as PyTorch FP32.
6. gemma-2-2b-it runs correctly on vLLM CPU (not on the validated list), BF16 and W8A8.
7. W8A8 on vLLM CPU binds to the SGL `int8_scaled_mm` AMX kernel (debug dispatch log).
8. OpenVINO FP16 cells really run in f16 (probe reads the runtime inference precision) and
   PyTorch FP16 cells hit an AMX-FP16 oneDNN implementation.
9. Prompt token counts match across engines for every prompt (chat template and BOS).

## 13. Intel case studies (read for the revision-2 decisions)

### 13.1 OpenInfer on Xeon 6 SoC (Intel content 926388, PDF dated 2026-08-19)

Source: [How OpenInfer delivers 3x faster AI inference on Intel Xeon 6 SoC](https://www.intel.com/content/www/us/en/content-details/926388/how-openinfer-delivers-3x-faster-ai-inference-on-intel-xeon-6-soc.html)
(PDF via `cdrdv2.intel.com/v1/dl/getContent/926388`; the landing page returned HTTP 403).

What was tested: Qwen3.5 27B (dense, hybrid attention) on a Xeon 6726P-B (42 cores) under
VMware, one VM with 64 vCPUs pinned to 32 cores and 64 GB RAM. OpenInfer's own runtime
(QINT8) against llama.cpp built with native AMX (Q8_0). One thread per physical core.

Results as published:

| Test | Threads | OpenInfer prefill / decode (tok/s) | llama.cpp prefill / decode (tok/s) |
|---|---|---|---|
| 1 prompt, 14,120-token context, 256-token decode | 32 | 207.2 / 5.4 | 76.9 / 4.6 |
| same | 16 | 153.4 / 4.4 | 49.5 / 4.1 |
| 8 concurrent prompts, same shape (OpenInfer only) | 32 | 148.3 / 2.7 | n/a |

Their reading: prefill is compute-bound, so kernel quality shows (2.7x at 32 threads,
about 3.1x at 16). Decode on a 27B model is governed by memory bandwidth and is near
parity. They also frame GPU capacity as procurement- and lead-time-constrained, which
matches the narrative's capacity argument.

Caveats: vendor case study (Intel says it does not audit third-party data); Xeon 6 SoC in
VMware, not an EC2 instance; a 27B model, far above our cap; no GPU baseline; OpenInfer's
runtime is not one of the allowed engines (PyTorch, OpenVINO, vLLM CPU).

### 13.2 Riskfuel on x8i (Intel content 927556, PDF dated 2026-09-03)

Source: [Millions of valuations per second: how Riskfuel and Intel are transforming structured investment product design](https://www.intel.com/content/www/us/en/content-details/927556/millions-of-valuations-per-second-how-riskfuel-and-intel-are-transforming-structured-investment-product-design.html)
(PDF via `cdrdv2.intel.com/v1/dl/getContent/927556`).

What was tested: a neural-network surrogate of a bank's Monte Carlo pricer (4,161 input
features), served with OpenVINO and AMX-FP16 on an AWS x8i metal instance (Xeon 6 Granite
Rapids, 96 cores, single socket), at peak batched throughput across 192 threads.

Results as published: about 2.27 million valuations per second against about 1 per second
for the FP64 Monte Carlo pricer on the same node. Over more than 90,000 client trades, the
FP16 surrogate's error (mean 0.029%, P90 0.059%, P99 0.14%) was identical to FP32.

Caveats: not a language model; full metal socket; no GPU baseline; the speedup compares a
neural surrogate against Monte Carlo, not CPU against GPU.

### 13.3 What changes in the design because of these sources

| Decision | Choice | Basis |
|---|---|---|
| vLLM CPU INT8 W8A8 cells | **Add, on by default** for M1, M2, M4 on T2 (m8i.4xlarge) | OpenInfer: at the same INT8 quantization family, the engine changed prefill throughput 2.7-3.1x on the same Xeon 6. One INT8 engine is not enough evidence. W8A8 is vLLM's AMX-INT8 path with INT8 activations; OpenVINO INT8 is weight-only with optional dynamic activation quantization, so the two measure different things |
| vLLM CPU AWQ INT4 cells | **Keep optional, off by default** | Neither source tests INT4. OpenInfer shows decode is bandwidth-bound and near parity across engines, so INT4's effect is about bytes per token, which OpenVINO INT4 on T3 already measures |
| GPU W8A8 | Optional, off by default | The narrative compares the customer's current GPU baseline (BF16) against CPU options |
| FP16 cells for M3 | **Add** on T3 (OpenVINO, `INFERENCE_PRECISION_HINT=f16`) and T4 (PyTorch FP16) on 8i targets | Riskfuel ran OpenVINO with AMX-FP16 on the same CPU generation (Granite Rapids on AWS) for a batched, compute-bound network and saw FP32-identical accuracy |
| FP32 anchor for M3 quality | **Add** a PyTorch FP32 CPU reference next to the GPU BF16 reference | Riskfuel judged FP16 against FP32 |
| M3 batch sweep | Batch sizes 1, 8, 32, 64 | Riskfuel's result is peak batched throughput; AMX gains grow with batch |
| Prefill throughput metric | **Add** per-request prefill tok/s and aggregate input tok/s, reported apart from decode | OpenInfer's gains were prefill-only |
| Long-context set | Keep the 4K set as required; add an optional "xlong" set of about 14K tokens for models with at least 16K context (M1, M2), off by default | Matches OpenInfer's test shape so results can be compared with it |
| Thread policy | One OpenMP thread per physical core as the default everywhere | vLLM docs and OpenInfer agree |
| gemma-4-E2B-it | **Not added** | 5.1B raw parameters, above the cap; neither source covers it |
| Qwen3-4B-Instruct-2507 | **Added as M2** | Largest decoder inside the cap; Apache-2.0; vLLM Xeon 6 validated. Not decided by these sources |
| llama.cpp or OpenInfer runtime | **Not added** | Outside the allowed engines |

## 14. DSPM and DLP use cases (revision 3, 2026-11-09)

Added for a cybersecurity customer that sells DSPM and DLP. The goal is a benchmark suite
that looks like their production classification traffic.

### 14.1 What the two product categories do

- **DSPM (data security posture management)** works on data at rest. Gartner describes it
  as discovering, classifying and cataloging structured and unstructured data across
  on-premises and cloud sources, then assessing exposure
  ([Gartner market page](https://www.gartner.com/reviews/market/data-security-posture-management)).
  The workload is bulk scanning of buckets, databases, file shares and SaaS stores. It is
  throughput-bound and sensitive to cost per GB.
- **DLP (data loss prevention)** works on data in motion and in use: email, web uploads,
  endpoints, SaaS sharing, and now prompts sent to GenAI tools and agents. The workload is
  inline. Each message or prompt must be judged before it is allowed through, so it is
  latency-bound.

Both categories run the same core step: decide whether a piece of content is sensitive,
and often which kind. The answer is a label, not prose.

### 14.2 How vendors classify (what the benchmark has to imitate)

| Source | What it says | Consequence for the benchmark |
|---|---|---|
| [CSA / Sentra, "Building a better DSPM"](https://cloudsecurityalliance.org/articles/building-a-better-dspm-by-combining-data-classification-techniques/) | Regex catches patterns but needs context: a 9-digit number may be an SSN or a license number; a 16-digit number may be a card, user ID or tracking number. Validation (check digits), table structure (column names, neighbouring columns) and NLP/NER supply that context | Hard negatives with look-alike numbers; a structured-table workload where column names and neighbours matter |
| [Cyera, "AI-driven sensitive data classification"](https://www.cyera.com/blog/advancing-sensitive-data-classification-in-the-age-of-ai) | Combines identifiers with LLM classifiers built on open foundation models, naming **FLAN-T5 and Mistral**, fine-tuned in-house; samples clustered files instead of reading everything; claims 95% precision | Flan-T5 (M5) is a realistic DSPM model, not only a size proxy; bulk-scan chunks are samples, not whole files |
| [Microsoft Purview](https://learn.microsoft.com/en-us/purview/deploymentmodels/depmod-reduce-false-positives) | Pattern-based sensitive information types plus Exact Data Match, trainable classifiers and fingerprinting, layered to cut false positives | Label policy must separate "pattern present" from "actually sensitive" |
| [Amazon Macie managed data identifiers](https://docs.aws.amazon.com/macie/latest/user/mdis-reference-quick.html) | Four categories: Credentials, Financial information, Personal information (PHI), Personal information (PII); many identifiers need a keyword nearby | Category taxonomy anchored on Macie's, plus confidential business data |
| [PolicyGuard, arXiv 2608.02687](https://arxiv.org/html/2608.02687v1) (Aug 2026) | LLM classifier for prompts to coding agents, configured by a plain-text policy (8 categories, ALLOW/BLOCK/REVIEW). 2,000 synthetic multilingual prompts, about half sensitive; splits by template family; reported 96.5% block rate and 3.0% false positive rate; P50 669 ms and P95 2,397 ms with gpt-oss-safeguard-20b on Bedrock; same-size 8B models ranged from 64.1% to 89.7% block rate; zero-shot prompting was far worse than a policy prompt | Policy-as-prompt with examples, frozen before the test run; template-family splits; block rate (recall) and false positive rate as headline accuracy metrics; per-model accuracy varies a lot at the same size |

### 14.3 Workload shapes

| Workload | Input | Output | Load pattern | What decides success |
|---|---|---|---|---|
| Bulk scan, binary (DSPM) | Document chunks of about 512-2,048 tokens: emails, forms, contracts, notes, logs | One label (SENSITIVE / NOT_SENSITIVE) | High concurrency, latency-tolerant | Documents/s, $/GB scanned, recall |
| Bulk scan, category (DSPM) | Same | One category label | Same | Same plus per-category recall |
| Inline DLP | Chat messages, emails, GenAI prompts, code questions; about 64-512 tokens | One label (ALLOW / BLOCK) | Low to medium concurrency, strict latency | P99 latency, recall, false positive rate |
| Structured column classification (DSPM) | A table header plus 10-20 sample rows | One category per column set | Batch | Accuracy on column context |
| Redaction (optional) | Same as inline | Span list (tens to hundreds of tokens) | Medium | Decode speed; this is the only decode-heavy DSPM/DLP shape |

Narrow labels mean 1-5 generated tokens per request, so these workloads are almost all
prefill. Prefill is compute-bound, which is where AMX helps most (research.md section 13,
OpenInfer). The "PrefillOnly" engine paper describes the same class of single-output-token
workloads ([arXiv 2505.07203](https://arxiv.org/html/2505.07203v1)).

Every request in a DLP or DSPM deployment shares the same long policy instructions, and
only the content changes. In production that prefix is cached, so the DSPM suite runs
with prefix caching on, unlike the standard suite (where repeated prompts would make
caching misleading).

### 14.4 Latency and accuracy expectations

- **Latency:** there is no industry standard budget for inline DLP. Regex-only proxies
  claim sub-millisecond scanning; PolicyGuard's LLM gate measured P50 669 ms with a 20B
  model. The harness therefore uses configurable SLA presets rather than one number.
- **False positives:** a Cyberhaven survey of 300 security leaders put the average share
  of DLP alerts that are false positives at 51%; Cyberhaven also says a rate above 20-30%
  signals a structural problem
  ([Cyberhaven](https://www.cyberhaven.com/blog/dlp-false-positives)). Missed detections
  are the costlier error (PolicyGuard fails closed for that reason). The report leads with
  recall and false positive rate, not accuracy.
- **Speed vs accuracy transfer:** the speed of a classifier depends on architecture,
  size, precision and input length, not on its weights. A customer's fine-tuned Flan-T5 or
  RoBERTa runs at the same speed as the public checkpoint of the same shape. Accuracy does
  not transfer: zero-shot public checkpoints will score lower than fine-tuned production
  models. The report must say this next to every accuracy figure.

### 14.5 Managed-service price points (context for $/GB)

| Service | Price (us-east-1 / list) | Source |
|---|---|---|
| Amazon Macie sensitive data discovery | $1.00/GB for the first 50,000 GB per month, then $0.50 and $0.25 | Price List API, `AmazonMacie`, usage type `USE1-SensitiveDataDiscovery`, 2026-11-09 |
| Google Sensitive Data Protection storage inspection | $1.00/GiB from 1 GiB to 50 TiB; content API $3.00/GiB to 1 TiB; discovery profiling $0.03/GB | [Google pricing page](https://cloud.google.com/sensitive-data-protection/pricing) |

These services use pattern and ML detectors with their own coverage, so they are a cost
reference, not a like-for-like competitor. The harness reports self-hosted $/GB of text
classified (classification only; file parsing and OCR are out of scope).

### 14.6 Public synthetic datasets

| Dataset | License | Contents | Use |
|---|---|---|---|
| [nvidia/Nemotron-PII](https://huggingface.co/datasets/nvidia/Nemotron-PII) | CC BY 4.0 | 100K train + 100K test fully synthetic records, 55+ PII types, 50+ industries, structured and unstructured formats, span labels, locale field; revision `b70ffaf5ff39e079776134c5bf4381f00a9fd1ed` | Positive documents, and negatives made by sanitizing their spans |
| [gretelai/synthetic_pii_finance_multilingual](https://huggingface.co/datasets/gretelai/synthetic_pii_finance_multilingual) | Apache-2.0 | 50,346 train + 5,594 test synthetic financial documents (contracts, statements, applications), multilingual, span labels; revision `7b844d16738527a04264f50214cb426a4cea0897` | Long financial and legal documents |
| ai4privacy pii-masking sets | "other" (custom terms) | Large PII masking corpora | **Not used** (license) |

Both usable sets are almost entirely PII-positive, so negatives, hard negatives,
credentials, code, chat, prompts and tables come from our own seeded generator.

### 14.7 Open DLP-specific models inside the cap

| Model | Params | License | Type | Engines in scope | Use |
|---|---|---|---|---|---|
| [openai/privacy-filter](https://huggingface.co/openai/privacy-filter) (2026-04-17) | 1.40B total, about 50M active (MoE) | Apache-2.0 | Bidirectional token classifier, 8 span categories including `secret`, 128K context; built for high-throughput on-premises sanitization | vLLM (registered architecture, encoder-only attention) and PyTorch; no OpenVINO export | Detector baseline for bulk scan and inline DLP; revision `7ffa9a043d54d1be65afb281eddf0ffbe629385b` |
| [Roblox/roblox-pii-classifier-v2](https://huggingface.co/Roblox/roblox-pii-classifier-v2) (2026-08-03) | 559.9M (XLM-RoBERTa-large) | Apache-2.0 | Multi-label sequence classifier (asking for PII, giving PII, moving users off-platform), fixed 512 tokens, `problem_type: multi_label_classification` (vLLM applies sigmoid) | vLLM `/classify`, OpenVINO (`xlm-roberta` text-classification), PyTorch | Production chat-DLP classifier for the inline workload; revision `44a84be3eba4859a7e2a1f7b9cee8df61131f28b` |
| nvidia/gliner-PII | GLiNER | "other" | Span model | Not in our engines | Not used |
| OpenMed PII (DeBERTa-v3, ModernBERT) | 141-434M | Apache-2.0 | Token classifiers | DeBERTa has no vLLM support; ModernBERT has no OpenVINO path | Not used |
| iiiorg/piiranha-v1 | 278M | CC BY-NC-ND 4.0 | Token classifier | | **Not used** (non-commercial) |

All three of the Hugging Face model files above, and Gemma, returned HTTP 200 with the
stored token on 2026-11-09 (the Gemma license is now accepted).

### 14.8 Hard cases the dataset must contain

Taken from the CSA/Sentra article, PolicyGuard's error analysis and Macie's keyword rules:

- Look-alike numbers: 16-digit order or tracking numbers, Luhn-invalid card-like numbers,
  9-digit invoice numbers, UUIDs, hashes.
- Masked or truncated values: `***-**-6789`, "card ending 4242".
- Documentation and placeholder values: `AKIAIOSFODNN7EXAMPLE`, `YOUR_API_KEY`, the
  well-known test card `4111 1111 1111 1111` in a test plan.
- Text about sensitive topics without sensitive values: "how to rotate API keys", a policy
  on handling SSNs.
- Public business information: headquarters addresses, executive names in a press
  release.
- Partial identity: a first name alone vs a name combined with date of birth and address.
- Keyword proximity: the same number with and without "SSN" or "card" nearby.
- Real-format secrets labeled as "test" (PolicyGuard's policy blocks these; the policy
  file must state the rule either way).

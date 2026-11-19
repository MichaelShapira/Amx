# Is the CPU side configured right? Checking this harness against Intel's vLLM Xeon recipe

Question (2026-11-19): every run so far shows a clear GPU advantage, while Intel's Software
Catalog says Xeon with AMX competes with GPUs for some model types and sizes. Is the harness
running vLLM on the CPU the way Intel does (`public.ecr.aws/q9t5s3a7/vllm-cpu-release-repo:v0.23.0`,
`VLLM_CPU_SGL_KERNEL=1`, `VLLM_CPU_KVCACHE_SPACE=40`, `--distributed-executor-backend mp`,
`--enable-chunked-prefill`, `--max-num-seqs 256`, privileged container), or is it handicapping it?

Method: `tools/recipe_check.py`. One m8i.4xlarge (the size the harness pairs with g6.xlarge),
On-Demand, three server configurations one after another, the same client against each:
`vllm bench serve --dataset-name random --random-input-len 128 --random-output-len 128 --ignore-eos
--request-rate inf`, the command Intel prints next to its own numbers. Model `Qwen/Qwen3-4B`
(the same architecture and size as M2 in this catalog, and a model Intel publishes figures for).
Artifacts: `s3://amx-gpu-bench-346399954218-us-east-1/runs/recipe-check-20261119T081855Z/` and
`.../recipe-check-20261119T085351Z/` (bench JSON, server logs, lscpu).

## Result 1: the harness's configuration is as fast as Intel's recipe, or faster

m8i.4xlarge (8 physical cores of a Xeon 6975P-C, 16 vCPUs, 64 GiB; STREAM triad 107 GB/s), Qwen3-4B BF16:

| config | C=1 tok/s | C=16 tok/s (TPOT ms) | C=64 tok/s (TPOT) | C=128 tok/s (TPOT) |
|---|---|---|---|---|
| `intel_v023` — Intel's recipe verbatim, vLLM v0.23.0 image | 9.1 | 114 (115) | 197 (282) | 239 (489) |
| `ours_v033` — what `bench/cellspec.py` starts (vLLM v0.33.0, SYS_NICE + seccomp, OMP bind auto, 1 reserved core, 32 GiB KV, max-num-seqs 128) | 9.5 | 122 (115) | 259 (215) | 317 (367) |
| `intel_v033` — Intel's flags on the v0.33.0 image | 9.5 | 122 (115) | 258 (216) | 315 (369) |

Same instance, same client, same model. Our configuration ties Intel's at low concurrency and is
30% ahead at 64–128 concurrent requests. The third row shows why: the difference is the vLLM
version (0.33 vs 0.23), not the flags. `VLLM_CPU_SGL_KERNEL` no longer exists in 0.33 (the AMX
small-batch kernels are selected automatically: `vllm/model_executor/layers/utils.py::
check_cpu_sgl_kernel`), `--enable-chunked-prefill` is the default, and `privileged` buys nothing
over `--cap-add SYS_NICE` with the seccomp profile that admits `sched_setaffinity`, `mbind` and
`set_mempolicy` (both logs show the same binding: 7 compute cores, 1 reserved, NUMA membind).
`VLLM_CPU_KVCACHE_SPACE=40` and `--kv-cache-memory-bytes` are the same knob.

The CPU side is also at the hardware's limit, not the software's: at C=1 the engine streams the
8.0 GB of weights 9.5 times a second = 76 GB/s, 71% of what STREAM measures on this slice (107 GB/s).
No flag recovers the other 29%. At C=128 the step is compute-bound (~2.8 TFLOPS on 7 cores).

## Result 2: a bigger CPU slice does not change the per-dollar picture

m8i.16xlarge (32 cores, 256 GiB, $3.39/h = 4.0x the 4xlarge), `ours_v033`:

| model | C=1 tok/s | C=16 (TPOT) | C=64 (TPOT) | C=128 (TPOT) |
|---|---|---|---|---|
| Qwen3-4B BF16 | 20.1 | 261 (56) | 636 (90) | 824 (143) |
| Qwen3-1.7B BF16 | 36.6 | 501 (29) | 1172 (50) | 1461 (81) |

4x the cores and price bought 2.1x (C=1) to 2.6x (C=128) the throughput: the slice's share of the
socket's memory bandwidth does not grow with its core count (C=1 implies ~160 GB/s here, 5 GB/s
per core against 12 GB/s per core on the 4xlarge). Cost per million output tokens, On-Demand:

| | C=16 | C=64 | C=128 |
|---|---|---|---|
| m8i.4xlarge, Qwen3-4B | $1.93 | $0.91 | $0.74 |
| m8i.16xlarge, Qwen3-4B | $3.60 | $1.48 | $1.14 |
| m8i.16xlarge, Qwen3-1.7B | $1.88 | $0.80 | $0.64 |
| g6.2xlarge (L4), Qwen2.5-3B, run qwen-2-5-20261115T102048Z | $0.49 | — | — |

The GPU figure is at C=16 with TPOT 27 ms; at C=64–128 an L4 would be several times cheaper still.

## Result 3: our numbers agree with Intel's own, once scaled

Intel's catalog publishes, for Qwen3-4B 128/128 at the highest concurrency that keeps TPOT ≤ 100 ms
(entries `qwen-qwen3-4b`, "Test by Intel", vLLM v0.18, TP=4):

| system | cores | tok/s | per core |
|---|---|---|---|
| 2x Xeon 6747P, DDR5-6400 | 96 | 3,562 @ C=192 | 37 |
| 2x Xeon 6767P, DDR5-8000 | 128 | 1,544 @ C=80 | 12 |
| 2x Xeon 6787P, DDR5-8000 | 172 | 4,608 @ C=248 | 27 |
| 2x Xeon 6972P, DDR5-6400 | 192 | 3,443 @ C=184 | 18 |
| 2x Xeon 6980P, DDR5-6400 | 256 | 3,281 @ C=176 | 13 |
| this harness, m8i.4xlarge | 8 | 259 @ C=64 (TPOT 215) / 122 @ C=16 (TPOT 115) | 32 / 15 |
| this harness, m8i.16xlarge | 32 | 636 @ C=64 (TPOT 90) | 20 |

Per core at a ~100 ms TPOT, the AWS slices land inside Intel's own range (12–37 tok/s/core). Intel's
headline numbers are whole two-socket servers (96–256 cores, 1–1.5 TB of RAM) at 176–400 concurrent
requests, with a 100 ms-per-token, 3 s-to-first-token service level: a bulk-throughput regime on
hardware that is not what AWS sells at the price of one L4. They do not claim the slice this tool
pairs with a g6.xlarge keeps up with it, and on these measurements it does not.

## What WAS wrong in the harness, and is fixed

1. **Every W8A8 (INT8) vLLM CPU cell was judged "AMX INVALID" and dropped from the SLA and cost
   rankings.** vLLM's INT8 path runs through oneDNN's brgemm micro-kernel, logged as
   `ukernel,exec,cpu,brgemm,,undef,src:u8 wei:s8` with no implementation name; the parser only
   counted primitive lines, so the trace read `gemm_total: 0`. PyTorch admits u8 inputs to that
   micro-kernel only when the ISA is `avx512_core_amx`, so those 200,000 lines WERE the AMX INT8
   path. Fixed in `bench/amx.py`; `bench rejudge-amx <run> --render` re-parsed and re-rendered the
   Qwen run: W8A8 at C=16 gives 246 tok/s against 162 for BF16 (1.5x) and $0.96 per 1M tokens.
   (Those rows are still excluded for a different reason, see "open" below.)
2. **The default concurrency sweep (1, 4, 16) stops where the CPU is still climbing.** Throughput
   doubled again between C=16 and C=128 on both slices. The wizard allows up to 512; a run that
   wants Intel's framing should test 64 and 128 with a 100 ms per-token target.
3. Not a CPU handicap, but found on the way: the vLLM CPU KV-cache budget was sized without the
   engine's own footprint (2.15x the checkpoint, measured), which hung a 64 GiB host under
   Gemma 4 E4B; the budget now comes from the measured footprint and too-small hosts are refused.

## Open: the quality flag excludes the compressed formats on divergence, not on measured loss

The standard suite scores answer quality as ROUGE-L of a cell's greedy 256-token answers against the
GPU's, and flags a compressed format when its mean falls more than 0.10 below the CPU BF16 cell's.
Two BF16 cells of the SAME weights score only 0.73–0.76 against each other (greedy decodes diverge
after the first differing token), so W8A8 at 0.50 and INT8-OV at 0.61 are flagged and excluded from
the capacity and cost verdict. That measures divergence from the GPU's text, not accuracy. The DSPM
suite (labels) and the translation suite (chrF against a reference) measure the task itself;
whether a divergence flag should remove a row from the rankings is a policy decision for the
owner, not something this check settles.

## Bottom line

The CPU side is not misconfigured. On the same instance it matches or beats Intel's recipe; its
single-stream decode runs at ~70% of the slice's measured memory bandwidth; its per-core
throughput sits inside Intel's own published range. The GPU advantage in these reports comes from
the pairing (one L4 against 8 Xeon cores at the same hourly price) and from the regime tested
(interactive latency at 1–16 concurrent requests), not from a software handicap. Where the CPU case
is strongest on AWS pricing: classifiers and encoders, models of ~1–2B parameters, INT8 weights,
bulk workloads at 64+ concurrent requests with a 100 ms per-token target, and the situation the
tool was built for, no GPU capacity available.

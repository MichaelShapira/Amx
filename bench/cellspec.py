"""Cell specification for host/serve.sh (design 7.2-7.5 and 8).

Everything engine-specific is decided here, in Python, where it is tested: which weights
to stage, whether an OpenVINO export is needed, and the exact `docker run` argument
vector. host/serve.sh only stages files, substitutes {PRIVATE_IP} and {MODEL_DIR}, starts
the container and checks health. The container port is published on the instance's
private IP only (design section 6).
"""

from __future__ import annotations

import re
from typing import Any

from bench.config import ModelSpec, ResolvedCell, RunConfig, TargetSpec

CONTAINER = "bench-engine"
PORT = 8000
THIN_TASK = {"decoder": "causal", "encoder-decoder": "seq2seq", "encoder-classifier": "seq-cls",
             "token-classifier": "token-cls"}
API_TASK = {"decoder": "generate", "encoder-decoder": "generate", "encoder-classifier": "classify",
            "token-classifier": "detect"}
OV_TASK = {"decoder": "text-generation-with-past", "encoder-decoder": "text2text-generation-with-past",
           "encoder-classifier": "text-classification"}


def cell_key(cell_id: str) -> str:
    """Filesystem- and S3-safe form of a cell id (m1-qwen2.5-3b|t2-m8i-4xl|bf16 -> m1-qwen2.5-3b__t2-m8i-4xl__bf16)."""
    return re.sub(r"[^A-Za-z0-9._-]", "__", cell_id.replace("|", "__"))


def resolve_image(image: str, ecr_repo_uri: str | None, built_tags: dict[str, str] | None) -> str:
    if not image.startswith("built:"):
        return image
    name = image.split(":", 1)[1]
    if not ecr_repo_uri or not built_tags or name not in built_tags:
        raise ValueError(f"image {image} needs the ECR repository and a built tag for {name}")
    return f"{ecr_repo_uri}:{built_tags[name]}"


def _weights(model: ModelSpec, precision: str) -> dict[str, Any]:
    p = model.precisions[precision]
    if p.hf_repo:
        return {"repo": p.hf_repo, "revision": p.revision, "gated": model.gated}
    return {"repo": model.hf_repo, "revision": model.revision, "gated": model.gated}


def _ov(model: ModelSpec, precision: str, engine: str) -> dict[str, Any] | None:
    if engine not in ("ovms", "thin-openvino"):
        return None
    p = model.precisions[precision]
    exp = dict(p.export or {"weight_format": "fp16"})   # BF16/FP16 classifier cells share the FP16 IR
    exp.setdefault("task", OV_TASK[model.type])
    return {"model_type": model.type, "export": exp, "precision": precision, "hint": p.inference_precision}


def _prefix_caching(run: RunConfig) -> bool:
    return bool(run.suite == "dspm" and run.dspm and run.dspm.prefix_caching)


def _env_flags(env: dict[str, str]) -> list[str]:
    out: list[str] = []
    for k in sorted(env):
        out += ["-e", f"{k}={env[k]}"]
    return out


def docker_argv(cell: ResolvedCell, model: ModelSpec, target: TargetSpec, run: RunConfig, image: str,
                instance_type: str | None = None) -> dict:
    """Return {"run_opts": [...], "image": str, "cmd": [...]} for the detached container start.
    instance_type: the type this run launches; memory-sized arguments scale from the target's default."""
    from bench.pairing import scale_kv_cache

    eng = target.engines[cell.engine]
    gpu = target.accelerator == "gpu"
    opts = ["--name", CONTAINER, "--restart", "no", "-p", f"{{PRIVATE_IP}}:{PORT}:{PORT}",
            "-v", "/opt/models:/models:ro", "--log-opt", "max-size=200m", "--log-opt", "max-file=2"]
    if gpu:
        opts += ["--gpus", "all"]
    cmd: list[str] = []
    a = eng.args
    if cell.engine == "vllm":
        opts += ["--shm-size", "4g"]
        if not gpu:
            opts += ["--cap-add", "SYS_NICE", "--security-opt", "seccomp=/opt/bench/seccomp-vllm.json"]
        opts += _env_flags(eng.env)
        opts += ["--entrypoint", "vllm"]
        max_len = model.max_model_len or model.max_input_tokens
        if model.type in ("encoder-classifier", "token-classifier"):
            max_len = (model.classifier.max_length if model.classifier else model.max_input_tokens)
        cmd = ["serve", "{MODEL_DIR}", "--served-model-name", model.id, "--dtype", "bfloat16",
               "--max-model-len", str(max_len), "--max-num-seqs", str(a.get("max_num_seqs", 128)),
               "--max-num-batched-tokens", str(a.get("max_num_batched_tokens", 2048)),
               "--enable-prefix-caching" if _prefix_caching(run) else "--no-enable-prefix-caching",
               "--seed", str(a.get("seed", 0)), "--host", "0.0.0.0", "--port", str(PORT)]  # noqa: S104 - in-container bind; published on the private IP only
        if gpu:
            cmd += ["--gpu-memory-utilization", str(a.get("gpu_memory_utilization", 0.9))]
        else:
            kv = scale_kv_cache(int(a.get("kv_cache_gib", 16)), target.instance_type, instance_type)
            cmd += ["--kv-cache-memory-bytes", str(kv * 2**30),
                    "--block-size", str(a.get("block_size", 128))]
        if model.type in ("encoder-classifier", "token-classifier"):
            cmd += ["--runner", "pooling"]
    elif cell.engine == "ovms":
        opts += _env_flags(eng.env)
        cmd = ["--rest_port", str(PORT), "--rest_bind_address", "0.0.0.0", "--model_name", model.id,  # noqa: S104 - in-container bind
               "--model_path", "{MODEL_DIR}", "--log_level", "INFO"]
    elif cell.engine in ("thin-torch", "thin-openvino"):
        p = model.precisions[cell.precision]
        env = {
            "THIN_MODEL_DIR": "{MODEL_DIR}", "THIN_MODEL_ID": model.id, "THIN_TASK": THIN_TASK[model.type],
            "THIN_BACKEND": "openvino" if cell.engine == "thin-openvino" else "torch",
            "THIN_DTYPE": cell.precision if cell.precision in ("bf16", "fp16", "fp32") else "bf16",
            "THIN_DEVICE": "cuda" if gpu else "cpu",
            "THIN_MAX_BATCH": str(a.get("max_batch", 16)), "THIN_MAX_WAIT_MS": str(a.get("max_wait_ms", 5)),
            "THIN_MAX_INPUT_TOKENS": str(model.classifier.max_length if model.classifier else model.max_input_tokens),
            "THIN_HOST": "0.0.0.0", "THIN_PORT": str(PORT),  # noqa: S104 - in-container bind; published on the private IP only
        }
        if cell.engine == "thin-openvino" and p.inference_precision:
            env["THIN_OV_PRECISION_HINT"] = p.inference_precision
        if model.classifier and model.classifier.multi_label:
            env["THIN_MULTI_LABEL"] = "true"
        opts += _env_flags({**eng.env, **env})
    else:
        raise ValueError(f"unknown engine {cell.engine}")
    return {"run_opts": opts, "image": image, "cmd": cmd}


def health(cell: ResolvedCell, model: ModelSpec) -> dict:
    task = API_TASK[model.type]
    if task == "generate":
        probe = {"path": "/v1/completions", "body": {"model": model.id, "prompt": "Hello", "max_tokens": 1,
                                                    "temperature": 0}}
        if cell.engine == "ovms":
            probe["alt_paths"] = ["/v3/completions"]
    elif task == "classify":
        probe = {"path": "/classify", "body": {"model": model.id, "input": ["hello"]}}
    else:
        probe = {"path": "/detect", "body": {"model": model.id, "input": ["hello"]}}
    path = "/v2/health/ready" if cell.engine == "ovms" else "/health"
    return {"path": path, "probe": probe, "timeout_s": 2400}


def verbose_env(cell: ResolvedCell) -> dict[str, str]:
    """Extra environment for the AMX trace restart (design 14.1)."""
    if cell.engine == "vllm":
        return {"ONEDNN_VERBOSE": "1", "VLLM_LOGGING_LEVEL": "DEBUG"}
    if cell.engine == "thin-torch":
        return {"ONEDNN_VERBOSE": "1"}
    return {}


def build(cell: ResolvedCell, model: ModelSpec, target: TargetSpec, run: RunConfig, *, run_id: str,
          bucket: str, hf_token_secret: str, ecr_repo_uri: str | None = None,
          built_tags: dict[str, str] | None = None, instance_type: str | None = None) -> dict:
    """instance_type: the type this run launches for the target (run.instance_types override);
    memory-dependent engine arguments are scaled from the target's default size."""
    image = resolve_image(target.engines[cell.engine].image, ecr_repo_uri, built_tags)
    ov = _ov(model, cell.precision, cell.engine)
    if ov is not None:
        ov["export_image"] = resolve_image("built:export", ecr_repo_uri, built_tags) if (
            ecr_repo_uri and built_tags and "export" in built_tags) else None
    spec = {
        "schema": 1, "run_id": run_id, "cell_id": cell.cell_id, "cell_key": cell_key(cell.cell_id),
        "target_id": target.id, "engine": cell.engine, "model_id": model.id, "model_type": model.type,
        "precision": cell.precision, "variant": cell.variant, "api_task": API_TASK[model.type],
        "weights": _weights(model, cell.precision), "ov": ov,
        "prefix_caching": _prefix_caching(run) if cell.engine in ("vllm", "ovms") else None,
        "instance_type": instance_type or target.instance_type,
        "container": docker_argv(cell, model, target, run, image, instance_type),
        "health": health(cell, model), "verbose_env": verbose_env(cell),
        "bucket": bucket, "hf_token_secret": hf_token_secret, "region": run.region,
    }
    if cell.engine == "ovms":
        a = target.engines["ovms"].args
        from bench.pairing import scale_kv_cache

        spec["ovms_graph"] = {"enable_prefix_caching": "true" if spec["prefix_caching"] else "false",
                              "cache_size": scale_kv_cache(int(a.get("cache_size_gb", 16)), target.instance_type,
                                                           instance_type),
                              "max_num_seqs": a.get("max_num_seqs", 128),
                              "max_num_batched_tokens": a.get("max_num_batched_tokens", 2048)}
    return spec

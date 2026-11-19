"""Config schemas for the model catalog, target catalog and run configs.

Every run is driven by one YAML run config (Requirement 6.1). Catalogs are validated
strictly: unpinned revisions, unpinned images and oversized models are rejected.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any, ClassVar, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SHA40 = re.compile(r"^[0-9a-f]{40}$")
IMAGE_DIGEST = re.compile(r"^[^\s@]+@sha256:[0-9a-f]{64}$")
BUILT_IMAGE = re.compile(r"^built:[a-z0-9-]+$")

ModelType = Literal["decoder", "encoder-classifier", "token-classifier", "encoder-decoder", "embedding"]
Suite = Literal["standard", "dspm", "translation", "embedding", "intel"]
TranslationDirection = Literal["en-de", "de-en", "en-es", "es-en"]
TranslationSize = Literal["sentence", "paragraph"]
EngineName = Literal["vllm", "ovms", "thin-torch", "thin-openvino"]

# Methodology defaults (Requirement 7.4).
FULL_MIN_REQUESTS = 200
FULL_MIN_REPEATS = 3
# Requests must finish well inside the 15-minute Lambda limit (Requirement 5.7).
MAX_REQUEST_TIMEOUT_S = 780


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PromptFormat(Strict):
    kind: Literal["chat_template", "plain", "pair", "prefix", "roblox_chat"]
    template: str | None = None
    prefix: str | None = None
    # Extra keyword arguments for apply_chat_template, e.g. {enable_thinking: false} for Qwen3, whose
    # template thinks by default (Intel's recipe passes the same through --default-chat-template-kwargs).
    template_kwargs: dict[str, Any] | None = None


class ClassifierSpec(Strict):
    batch_sizes: list[int] = Field(default_factory=lambda: [1, 8, 32, 64])
    labels: list[str] = Field(default_factory=list)
    multi_label: bool = False
    max_length: int = 512


class EmbeddingSpec(Strict):
    """An embedding model (sentence-transformers layout: pooling, Dense projections, normalize).
    prefix: the text prepended to every input (EmbeddingGemma's document prompt); both engines get it."""
    batch_sizes: list[int] = Field(default_factory=lambda: [1, 8, 32, 64])
    max_length: int = 512
    dim: int
    prefix: str = ""


class QualityReference(Strict):
    target: str
    precision: str
    fp32_anchor: dict[str, str] | None = None


class PrecisionSpec(Strict):
    engines: list[EngineName]
    hf_repo: str | None = None
    revision: str | None = None
    export: dict[str, Any] | None = None
    inference_precision: Literal["bf16", "f16", "f32"] | None = None
    weights: str | None = None
    activations: str | None = None
    requires_cpu_flag: str | None = None
    quality_only: bool = False
    enabled: bool = True

    @model_validator(mode="after")
    def _pinned(self) -> PrecisionSpec:
        if self.hf_repo and not (self.revision and SHA40.match(self.revision)):
            raise ValueError(f"precision checkpoint {self.hf_repo} needs a 40-hex commit revision")
        return self


class ModelSpec(Strict):
    id: str
    slot: str
    hf_repo: str
    revision: str
    type: ModelType
    license: str
    gated: bool = False
    params: int
    nominal_size_b: float
    suites: list[Suite] = Field(default_factory=lambda: ["standard"])
    load_dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    max_input_tokens: int
    max_model_len: int | None = None
    prompt_format: PromptFormat
    quality_reference: QualityReference | None = None
    classifier: ClassifierSpec | None = None
    embedding: EmbeddingSpec | None = None
    precisions: dict[str, PrecisionSpec]
    # Extra `vllm serve` arguments this model needs on every vLLM target (a chat template, a
    # tool-call or reasoning parser, multimodal limits), appended verbatim after the harness's own
    # arguments. Flags only: the harness decides the port, model path and scheduler settings.
    vllm_args: list[str] = Field(default_factory=list)
    enabled: bool = True
    notes: str | None = None

    @field_validator("vllm_args")
    @classmethod
    def _flags_only(cls, v: list[str]) -> list[str]:
        reserved = {"--host", "--port", "--served-model-name", "--max-model-len", "--max-num-seqs",
                    "--max-num-batched-tokens", "--dtype", "--seed", "--kv-cache-memory-bytes", "--gpu-memory-utilization"}
        for a in v:
            if a.split("=", 1)[0] in reserved:
                raise ValueError(f"vllm_args may not set {a.split('=', 1)[0]}; the harness owns it")
        return v

    @field_validator("revision")
    @classmethod
    def _sha(cls, v: str) -> str:
        if not SHA40.match(v):
            raise ValueError("revision must be a 40-hex commit SHA (Requirement 1.3)")
        return v

    @property
    def non_commercial(self) -> bool:
        return self.license in {"qwen-research", "cc-by-nc-4.0", "cc-by-nc-nd-4.0"}

    # The DSPM suite's rules (which workloads, how to read an answer) were written per shipped
    # model and keyed by slot. A shipped alternate (slot "M3-alt") stands in for its slot. A model
    # added later gets the rules of the shipped model it is most like: any chat model behaves like
    # M1, any encoder-decoder like M5; classifiers and detectors added later have no label mapping,
    # so no DSPM rules (standard suite only).
    DSPM_SHIPPED: ClassVar[frozenset[str]] = frozenset({"M1", "M2", "M3", "M4", "M5", "D1", "D2"})

    @property
    def dspm_profile(self) -> str | None:
        base = self.slot.split("-")[0]
        if base in self.DSPM_SHIPPED:
            return base
        return {"decoder": "M1", "encoder-decoder": "M5"}.get(self.type)

    @property
    def user_added(self) -> bool:
        return self.slot.startswith("U")


class ModelCatalog(Strict):
    size_cap_b: float = 4.0
    models: list[ModelSpec]
    # Shipped model ids a user removed in Setup (bench.catalog_store merges the live copy with the seed).
    removed: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _cap_and_unique(self) -> ModelCatalog:
        ids = [m.id for m in self.models]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate model ids")
        for m in self.models:
            if m.nominal_size_b > self.size_cap_b:
                raise ValueError(
                    f"{m.id}: nominal size {m.nominal_size_b}B exceeds cap {self.size_cap_b}B"
                )
        return self

    def get(self, model_id: str) -> ModelSpec:
        for m in self.models:
            if m.id == model_id:
                return m
        raise KeyError(f"unknown model {model_id}")


class EngineSpec(Strict):
    image: str
    port: int = 8000
    env: dict[str, str] = Field(default_factory=dict)
    args: dict[str, Any] = Field(default_factory=dict)

    @field_validator("image")
    @classmethod
    def _pinned_image(cls, v: str) -> str:
        if not (IMAGE_DIGEST.match(v) or BUILT_IMAGE.match(v)):
            raise ValueError(f"image {v!r} must be pinned by digest or be built:<name>")
        return v


class TargetSpec(Strict):
    id: str
    label: str
    instance_type: str | None = None
    alternates: list[str] = Field(default_factory=list)
    ami_ssm: str | None = None
    accelerator: Literal["gpu", "cpu"] | None = None
    cpu_features: list[str] = Field(default_factory=list)
    root_volume_gib: int = 200
    engines: dict[str, EngineSpec] = Field(default_factory=dict)
    inherits: str | None = None
    enabled: bool = True


class TargetCatalog(Strict):
    targets: list[TargetSpec]

    def resolved(self) -> dict[str, TargetSpec]:
        raw = {t.id: t for t in self.targets}
        out: dict[str, TargetSpec] = {}

        def resolve(tid: str, seen: tuple[str, ...] = ()) -> TargetSpec:
            if tid in out:
                return out[tid]
            if tid in seen:
                raise ValueError(f"inheritance cycle at {tid}")
            t = raw[tid]
            if t.inherits:
                base = resolve(t.inherits, (*seen, tid)).model_dump()
                mine = t.model_dump(exclude_unset=True)
                merged = _deep_merge(base, mine)
                merged["inherits"] = None
                if "enabled" not in mine:
                    merged["enabled"] = True
                t = TargetSpec.model_validate(merged)
            for field in ("instance_type", "ami_ssm", "accelerator"):
                if getattr(t, field) is None:
                    raise ValueError(f"target {tid} has no {field}")
            out[tid] = t
            return t

        for tid in raw:
            resolve(tid)
        return out


def _deep_merge(base: dict, over: dict) -> dict:
    result = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


class Cell(Strict):
    target: str
    precision: str
    model: str | None = None
    variant: Literal["default", "constrained"] = "default"


class Sampling(Strict):
    temperature: float = 0.0
    max_new_tokens: int = 256
    ignore_eos: bool = True
    seed: int = 0


class PerLevel(Strict):
    requests: int | None = None
    duration_s: int | None = None

    @model_validator(mode="after")
    def _one(self) -> PerLevel:
        if (self.requests is None) == (self.duration_s is None):
            raise ValueError("per_level needs exactly one of requests or duration_s")
        return self


class QualityCfg(Strict):
    enabled: bool = True
    prompt_set: str = "quality-50"
    max_new_tokens: int = 256


class SLA(Strict):
    p99_ttft_ms: float | None = None
    p50_tpot_ms: float | None = None
    p99_e2e_ms: float | None = None
    max_error_pct: float = 1.0


class CapacityCfg(Strict):
    max_cpu_instances_per_gpu: int = 4


class PurchaseCfg(Strict):
    """How target instances are launched for this run. Spot (the default) instances are
    one-time requests that terminate on interruption; the run then launches another instance
    (Spot again, or On-Demand once the Spot capacity window is used up) and resumes the
    cell's remaining levels. Cost figures always use On-Demand prices."""
    type: Literal["on-demand", "spot"] = "spot"
    spot_fallback_on_demand: bool = True
    spot_max_price_usd: float | None = None          # None: EC2 caps at the On-Demand price
    spot_capacity_window_s: int = Field(default=600, ge=60, le=43_200)   # retry Spot this long before falling back
    # Launch attempts (about one per minute, Spot and On-Demand together) before the run gives
    # up on a hardware option; the other options continue and the report says which side did
    # not run and why. Applies to the first launch and to replacements after an interruption.
    max_launch_attempts: int = Field(default=30, ge=1, le=720)
    max_relaunches_per_cell: int = 3


class DspmCfg(Strict):
    workloads: list[Literal["W1", "W2", "W3", "W4", "W5"]] = Field(
        default_factory=lambda: ["W1", "W2", "W3", "W4"]
    )
    sla_preset: Literal["inline_strict", "inline_standard", "bulk"] = "bulk"
    dataset_version: str = "v1"
    test_items: int | None = None
    prefix_caching: bool = True
    agreement_min: float = 0.98
    recall_drop_max_pp: float = 1.0
    fitness_recall_min: float = 0.90
    fitness_fpr_max: float = 0.05


class TranslationCfg(Strict):
    """Translation suite (synthetic EN/DE/ES parallel corpus, tools/translation). One request pool
    per direction and text size; every direction gets an accuracy pass (each item once, outputs
    kept) scored with chrF against the reference and against the GPU BF16 cell."""
    directions: list[TranslationDirection] = Field(default_factory=lambda: ["en-de", "de-en", "en-es", "es-en"])
    test_items: int | None = None          # items per direction in the accuracy pass; None = the whole pool

    @model_validator(mode="after")
    def _some(self) -> TranslationCfg:
        if not self.directions:
            raise ValueError("translation needs at least one direction")
        if len(set(self.directions)) != len(self.directions):
            raise ValueError("duplicate translation directions")
        return self


class EmbeddingCfg(Strict):
    """Embedding suite: the translation corpus's texts in all three languages, one pool per text size;
    an accuracy pass keeps the vectors for the agreement and cross-lingual retrieval scores."""
    languages: list[Literal["en", "de", "es"]] = Field(default_factory=lambda: ["en", "de", "es"])
    test_items: int | None = None


class IntelCfg(Strict):
    """Intel's published vLLM benchmark method (Intel Software Catalog): random prompts of a fixed
    token count, a fixed answer length with ignore_eos, request rate unbounded, one concurrency at a
    time; pass/fail at TPOT <= 100 ms and TTFT <= 3 s. The report shows Intel's figures for the same
    model next to ours (config/intel_catalog_benchmarks.json)."""
    input_tokens: Literal[128, 2048] = 128
    output_tokens: Literal[128, 2048] = 128


class ExecutionCfg(Strict):
    """parallel_targets: run every hardware option at the same time (default) or one after
    another (fewer instances up at once; the run takes longer)."""
    parallel_targets: bool = True


class RunConfig(Strict):
    name: str
    suite: Suite = "standard"
    mode: Literal["smoke", "full"] = "smoke"
    account: str
    region: str = "us-east-1"
    az_id: str = "use1-az1"
    models: list[str]
    cells: list[Cell]
    prompt_set: str = "short"
    custom_prompt: str | None = None
    sampling: Sampling = Field(default_factory=Sampling)
    concurrency: list[int]
    per_level: PerLevel
    repeats: int = 1
    warmup_requests: int = 20
    request_timeout_s: int = 600
    quality: QualityCfg = Field(default_factory=QualityCfg)
    sla: SLA = Field(default_factory=SLA)
    capacity: CapacityCfg = Field(default_factory=CapacityCfg)
    purchase: PurchaseCfg = Field(default_factory=PurchaseCfg)
    # Per-target instance type for this run (target id -> EC2 type), e.g. {t1-g6: g6.2xlarge}.
    # Must be a catalog type of the target's accelerator (AMX family for CPU targets); the
    # pairing check (bench/pairing.py) reports whether the CPU and GPU sides are comparable.
    instance_types: dict[str, str] = Field(default_factory=dict)
    alternates: dict[str, list[str]] = Field(default_factory=dict)      # per-target capacity fallbacks
    # Inputs per request for classifier models (overrides the model's classifier.batch_sizes);
    # generation models always take one prompt per request (the server batches concurrent ones).
    batch_sizes: list[int] | None = None
    execution: ExecutionCfg = Field(default_factory=ExecutionCfg)
    dspm: DspmCfg | None = None
    translation: TranslationCfg | None = None
    embedding: EmbeddingCfg | None = None
    intel: IntelCfg | None = None
    stop_targets_after_run: bool = True
    max_waves: int = 20

    @model_validator(mode="after")
    def _methodology(self) -> RunConfig:
        if not self.concurrency or any(c < 1 or c > 512 for c in self.concurrency):
            raise ValueError("concurrency levels must be between 1 and 512")
        if self.batch_sizes is not None and (not self.batch_sizes or any(b < 1 or b > 256 for b in self.batch_sizes)):
            raise ValueError("batch_sizes must be between 1 and 256")
        if self.warmup_requests < 20:
            raise ValueError("at least 20 warm-up requests are required (Requirement 7.3)")
        if self.request_timeout_s > MAX_REQUEST_TIMEOUT_S:
            raise ValueError(f"request_timeout_s must be <= {MAX_REQUEST_TIMEOUT_S}")
        if self.mode == "full":
            if self.per_level.requests is not None and self.per_level.requests < FULL_MIN_REQUESTS:
                raise ValueError(f"full mode needs >= {FULL_MIN_REQUESTS} requests per level")
            if self.repeats < FULL_MIN_REPEATS:
                raise ValueError(f"full mode needs >= {FULL_MIN_REPEATS} repeats")
        if self.suite == "dspm" and self.dspm is None:
            self.dspm = DspmCfg()
        if self.suite == "embedding":
            if self.embedding is None:
                self.embedding = EmbeddingCfg()
            if self.prompt_set not in ("sentence", "paragraph"):
                raise ValueError("the embedding suite's prompt_set is the text size: sentence or paragraph")
        if self.suite == "intel":
            if self.intel is None:
                self.intel = IntelCfg()
            if self.prompt_set != "intel":
                raise ValueError("the intel suite's prompt_set is 'intel' (random prompts of a fixed token count)")
            if not self.sampling.ignore_eos or self.sampling.max_new_tokens != self.intel.output_tokens:
                raise ValueError("the intel suite generates exactly intel.output_tokens per answer (ignore_eos: true)")
        if self.suite == "translation":
            if self.translation is None:
                self.translation = TranslationCfg()
            if self.prompt_set not in ("sentence", "paragraph"):
                raise ValueError("the translation suite's prompt_set is the text size: sentence or paragraph")
            if self.sampling.ignore_eos:
                raise ValueError("translation answers must end where the model ends them (sampling.ignore_eos: false)")
        if self.custom_prompt and self.prompt_set != "custom":
            raise ValueError("custom_prompt requires prompt_set: custom")
        return self

    @property
    def is_smoke(self) -> bool:
        """Requirement 7.7: anything below the methodology defaults is labeled smoke."""
        below = (
            self.per_level.requests is not None and self.per_level.requests < FULL_MIN_REQUESTS
        ) or self.repeats < FULL_MIN_REPEATS
        return self.mode == "smoke" or below


class ResolvedCell(Strict):
    cell_id: str
    model: str
    target: str
    precision: str
    engine: str
    variant: str = "default"
    quality_only: bool = False


def resolve_cells(run: RunConfig, models: ModelCatalog, targets: TargetCatalog) -> list[ResolvedCell]:
    """Expand run cells against the catalogs; refuse anything that does not resolve."""
    tmap = targets.resolved()
    out: list[ResolvedCell] = []
    for cell in run.cells:
        model_ids = [cell.model] if cell.model else run.models
        for mid in model_ids:
            if mid not in run.models:
                raise ValueError(f"cell model {mid} is not listed in run.models")
            m = models.get(mid)
            if not m.enabled:
                raise ValueError(f"model {mid} is disabled")
            if run.suite not in m.suites:
                raise ValueError(f"model {mid} is not part of the {run.suite} suite")
            if cell.target not in tmap:
                raise ValueError(f"unknown target {cell.target}")
            t = tmap[cell.target]
            if not t.enabled:
                raise ValueError(f"target {t.id} is disabled")
            p = m.precisions.get(cell.precision)
            if p is None or not p.enabled:
                raise ValueError(f"{mid} has no enabled precision {cell.precision}")
            if p.requires_cpu_flag and t.accelerator == "cpu" and p.requires_cpu_flag not in t.cpu_features:
                raise ValueError(
                    f"{mid}/{cell.precision} needs CPU flag {p.requires_cpu_flag}, which {t.id} lacks"
                )
            engine = next((e for e in p.engines if e in t.engines), None)
            if engine is None:
                raise ValueError(
                    f"{mid}/{cell.precision}: none of {p.engines} is available on {t.id}"
                )
            if cell.variant == "constrained" and engine != "vllm":
                raise ValueError("the constrained variant is only available on vLLM targets")
            cid = f"{mid}|{t.id}|{cell.precision}" + ("" if cell.variant == "default" else f"|{cell.variant}")
            out.append(
                ResolvedCell(
                    cell_id=cid, model=mid, target=t.id, precision=cell.precision,
                    engine=engine, variant=cell.variant, quality_only=p.quality_only,
                )
            )
    ids = [c.cell_id for c in out]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate cells in run")
    used = {c.target for c in out}
    for tid, itype in run.instance_types.items():
        if tid not in tmap:
            raise ValueError(f"instance_types names unknown target {tid}")
        if tid not in used:
            raise ValueError(f"instance_types names target {tid}, which has no cell in this run")
        from bench.pairing import validate_for_target

        t = tmap[tid]
        try:
            validate_for_target(itype, t.accelerator, t.cpu_features)
        except ValueError as e:
            raise ValueError(f"instance_types[{tid}]: {e}") from e
    return out


def run_instance_types(run: RunConfig, cells: list[ResolvedCell], tmap: dict[str, TargetSpec]) -> dict[str, str]:
    """Instance type per target used by this run: the run's override or the target's default."""
    return {tid: run.instance_types.get(tid) or tmap[tid].instance_type for tid in sorted({c.target for c in cells})}


def run_alternates(run: RunConfig, tid: str, tmap: dict[str, TargetSpec]) -> list[str]:
    alts = run.alternates.get(tid)
    return list(alts) if alts is not None else list(tmap[tid].alternates)


def load_yaml(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _strip_anchors(data: dict) -> dict:
    """Top-level keys starting with 'x-' only hold YAML anchors; drop them before validation."""
    return {k: v for k, v in data.items() if not str(k).startswith("x-")}


def load_models(path: str | Path = "config/models.yaml") -> ModelCatalog:
    return ModelCatalog.model_validate(_strip_anchors(load_yaml(path)))


def load_targets(path: str | Path = "config/targets.yaml") -> TargetCatalog:
    return TargetCatalog.model_validate(_strip_anchors(load_yaml(path)))


def load_run(path: str | Path) -> RunConfig:
    data = load_yaml(path)
    return RunConfig.model_validate(data["run"] if "run" in data else data)


def config_hash(obj: Any) -> str:
    """Stable SHA-256 of a config object, used to tie results to the exact inputs."""
    if isinstance(obj, BaseModel):
        obj = obj.model_dump(mode="json")
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()

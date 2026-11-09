"""Thin server settings (design 7.5). Everything comes from THIN_* environment variables so
the same image serves every model; serve.sh writes them per cell."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

TASKS = ("seq2seq", "causal", "seq-cls", "token-cls")
DTYPES = ("bf16", "fp16", "fp32")


def _env(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name)
    return v if v not in (None, "") else default


@dataclass
class Settings:
    model_dir: str
    model_id: str
    task: str
    backend: str = "torch"                 # torch | openvino
    dtype: str = "bf16"                    # torch weights and compute
    device: str = "cpu"                    # cpu | cuda
    ov_precision_hint: str | None = None   # bf16 | f16 | f32 (openvino INFERENCE_PRECISION_HINT)
    max_batch: int = 16
    max_wait_ms: float = 5.0
    max_input_tokens: int = 1024
    max_new_tokens_limit: int = 4096
    multi_label: bool | None = None        # None: from config.problem_type
    viterbi_calibration: str | None = None
    host: str = "127.0.0.1"
    port: int = 8000
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.task not in TASKS:
            raise ValueError(f"THIN_TASK must be one of {TASKS}, got {self.task!r}")
        if self.dtype not in DTYPES:
            raise ValueError(f"THIN_DTYPE must be one of {DTYPES}, got {self.dtype!r}")
        if self.backend not in ("torch", "openvino"):
            raise ValueError("THIN_BACKEND must be torch or openvino")
        if self.backend == "openvino" and self.task in ("causal", "token-cls"):
            raise ValueError("the openvino backend serves seq2seq and seq-cls only (design 7.4, 24.5)")
        if self.max_batch < 1 or self.max_wait_ms < 0:
            raise ValueError("max_batch must be >= 1 and max_wait_ms >= 0")

    @classmethod
    def from_env(cls) -> Settings:
        model_dir = _env("THIN_MODEL_DIR")
        if not model_dir or not Path(model_dir).exists():
            raise SystemExit(f"THIN_MODEL_DIR is not set or does not exist: {model_dir!r}")
        ml = _env("THIN_MULTI_LABEL")
        return cls(
            model_dir=model_dir,
            model_id=_env("THIN_MODEL_ID", Path(model_dir).name),
            task=_env("THIN_TASK", ""),
            backend=_env("THIN_BACKEND", "torch"),
            dtype=_env("THIN_DTYPE", "bf16"),
            device=_env("THIN_DEVICE", "cpu"),
            ov_precision_hint=_env("THIN_OV_PRECISION_HINT"),
            max_batch=int(_env("THIN_MAX_BATCH", "16")),
            max_wait_ms=float(_env("THIN_MAX_WAIT_MS", "5")),
            max_input_tokens=int(_env("THIN_MAX_INPUT_TOKENS", "1024")),
            multi_label=None if ml is None else ml.lower() in ("1", "true", "yes"),
            viterbi_calibration=_env("THIN_VITERBI_CALIBRATION"),
            host=_env("THIN_HOST", "127.0.0.1"),
            port=int(_env("THIN_PORT", "8000")),
        )

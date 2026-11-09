"""Model backends for the thin server: torch (CPU or CUDA) and openvino (design 7.5, 7.4)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch

from servers.thin import generation, viterbi
from servers.thin.settings import Settings

log = logging.getLogger("thin")
TORCH_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


class Backend:
    """Common inference code; subclasses load the model."""

    settings: Settings
    tok = None
    model = None
    encoder_decoder = False
    device = "cpu"

    def __init__(self, settings: Settings):
        self.settings = settings
        self._load()
        self.config = self.model.config
        id2label = getattr(self.config, "id2label", None) or {}
        self.labels = [id2label[i] if i in id2label else id2label.get(str(i), str(i)) for i in range(len(id2label))]
        ml = settings.multi_label
        self.multi_label = (getattr(self.config, "problem_type", None) == "multi_label_classification") \
            if ml is None else ml
        if settings.task == "token-cls":
            calib = settings.viterbi_calibration or str(Path(settings.model_dir) / "viterbi_calibration.json")
            self.decoder = viterbi.Decoder({i: lab for i, lab in enumerate(self.labels)}, viterbi.load_biases(calib))
            self.viterbi_biases = self.decoder.biases

    def _load(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def describe(self) -> dict:
        return {"task": self.settings.task, "backend": self.settings.backend, "dtype": self.settings.dtype,
                "device": self.device, "labels": self.labels, "multi_label": self.multi_label,
                "ov_precision_hint": self.settings.ov_precision_hint,
                "max_batch": self.settings.max_batch, "max_wait_ms": self.settings.max_wait_ms}

    def count_tokens(self, text: str) -> int:
        return len(self.tok(text, add_special_tokens=True)["input_ids"])

    # -- generation
    def generate(self, jobs: list[generation.GenJob]) -> None:
        generation.run_batch(self.model, self.tok, jobs, device=self.device,
                             encoder_decoder=self.encoder_decoder,
                             max_input_tokens=self.settings.max_input_tokens)

    # -- sequence classification
    def _encode_cls(self, inputs: list):
        kw = dict(return_tensors="pt", padding=True, truncation=True, max_length=self.settings.max_input_tokens)
        if all(isinstance(x, (list, tuple)) for x in inputs):
            enc = self.tok([x[0] for x in inputs], [x[1] for x in inputs], **kw)
        else:
            enc = self.tok(list(inputs), **kw)
        return enc

    def _logits(self, enc) -> np.ndarray:
        with torch.inference_mode():
            out = self.model(**{k: v.to(self.device) for k, v in enc.items()})
        return out.logits.float().cpu().numpy()

    def classify(self, inputs: list) -> tuple[list[list[float]], list[int]]:
        enc = self._encode_cls(inputs)
        logits = self._logits(enc)
        if self.multi_label:
            probs = 1.0 / (1.0 + np.exp(-logits))
        else:
            z = logits - logits.max(axis=1, keepdims=True)
            probs = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
        return probs.tolist(), enc["attention_mask"].sum(dim=1).tolist()

    # -- token classification (D1)
    def detect(self, texts: list[str]) -> tuple[list[dict], list[int]]:
        enc = self.tok(texts, return_tensors="pt", padding=True, truncation=True,
                       max_length=self.settings.max_input_tokens, return_offsets_mapping=True)
        offsets = enc.pop("offset_mapping").tolist()
        logits = torch.from_numpy(self._logits(enc))
        logprobs = torch.log_softmax(logits, dim=-1).numpy()
        probs = np.exp(logprobs)
        mask = enc["attention_mask"].numpy().astype(bool)
        results = []
        for r, text in enumerate(texts):
            keep = [k for k in range(mask.shape[1]) if mask[r, k] and offsets[r][k][1] > offsets[r][k][0]]
            lp, pr = logprobs[r, keep], probs[r, keep]
            offs = [tuple(offsets[r][k]) for k in keep]
            path = self.decoder.decode(lp)
            spans = viterbi.char_spans(text, offs, self.decoder.spans(path), pr, path)
            results.append({"spans": spans,
                            "category_max": viterbi.category_max(pr, self.decoder.names, np.ones(len(keep), bool))})
        return results, enc["attention_mask"].sum(dim=1).tolist()


class TorchBackend(Backend):
    def _load(self) -> None:
        from transformers import (
            AutoModelForCausalLM,
            AutoModelForSeq2SeqLM,
            AutoModelForSequenceClassification,
            AutoModelForTokenClassification,
            AutoTokenizer,
        )

        s = self.settings
        self.device = s.device
        if s.device == "cuda" and not torch.cuda.is_available():
            raise SystemExit("THIN_DEVICE=cuda but CUDA is not available")
        cls = {"seq2seq": AutoModelForSeq2SeqLM, "causal": AutoModelForCausalLM,
               "seq-cls": AutoModelForSequenceClassification, "token-cls": AutoModelForTokenClassification}[s.task]
        self.tok = AutoTokenizer.from_pretrained(s.model_dir)
        if s.task == "causal":
            self.tok.padding_side = "left"
            if self.tok.pad_token is None:
                self.tok.pad_token = self.tok.eos_token
        self.model = cls.from_pretrained(s.model_dir, dtype=TORCH_DTYPES[s.dtype]).to(s.device).eval()
        self.encoder_decoder = bool(getattr(self.model.config, "is_encoder_decoder", False))
        log.info("torch backend: %s %s on %s, torch %s, threads %d", s.task, s.dtype, s.device,
                 torch.__version__, torch.get_num_threads())


class OpenVINOBackend(Backend):
    """optimum-intel models compiled with the cell's INFERENCE_PRECISION_HINT."""

    def _load(self) -> None:
        from optimum.intel import OVModelForSeq2SeqLM, OVModelForSequenceClassification
        from transformers import AutoTokenizer

        s = self.settings
        ov_config = {"PERFORMANCE_HINT": "THROUGHPUT" if s.max_batch > 1 else "LATENCY"}
        if s.ov_precision_hint:
            ov_config["INFERENCE_PRECISION_HINT"] = s.ov_precision_hint
        cls = {"seq2seq": OVModelForSeq2SeqLM, "seq-cls": OVModelForSequenceClassification}[s.task]
        self.tok = AutoTokenizer.from_pretrained(s.model_dir)
        self.model = cls.from_pretrained(s.model_dir, ov_config=ov_config, device="CPU", compile=True)
        self.encoder_decoder = s.task == "seq2seq"
        self.ov_config = ov_config
        log.info("openvino backend: %s, ov_config %s", s.task, json.dumps(ov_config))

    def _logits(self, enc) -> np.ndarray:
        out = self.model(**dict(enc))
        logits = out.logits
        return logits.float().cpu().numpy() if hasattr(logits, "cpu") else np.asarray(logits, dtype=np.float32)


def load(settings: Settings) -> Backend:
    return (OpenVINOBackend if settings.backend == "openvino" else TorchBackend)(settings)

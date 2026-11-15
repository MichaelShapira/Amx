"""Export helpers that do not need OpenVINO installed (task 10)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("export", ROOT / "servers/export/export.py")
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)

GRAPH = """input_stream: "HTTP_REQUEST_PAYLOAD:input"
node: {
  name: "LLMExecutor"
  calculator: "HttpLLMCalculator"
  node_options: {
      [type.googleapis.com/mediapipe.LLMCalculatorOptions]: {
          models_path: "./",
          plugin_config: '{ }',
          enable_prefix_caching: true,
          cache_size: 0,
          max_num_seqs: 256,
          device: "CPU",
      }
  }
}
"""


def test_hash_changes_with_any_input():
    tv = {"openvino": "2026.4.1"}
    base = export.export_hash({"weight_format": "int4", "group_size": 128}, "abc", tv)
    assert base == export.export_hash({"group_size": 128, "weight_format": "int4"}, "abc", tv)
    assert base != export.export_hash({"weight_format": "int8"}, "abc", tv)
    assert base != export.export_hash({"weight_format": "int4", "group_size": 128}, "abd", tv)
    assert base != export.export_hash({"weight_format": "int4", "group_size": 128}, "abc", {"openvino": "2026.4.0"})


def test_commands_per_model_type(tmp_path):
    """Decoders: optimum-cli with an explicit --task (optimum cannot infer it for a local directory,
    which is what sank every OVMS decoder export until 2026-11-12), then OVMS's export_model.py
    only for graph.pbtxt, pointed at the exported IR with a writable config path."""
    steps = export.commands("decoder", {"weight_format": "int4", "group_size": 128, "ratio": 1.0, "sym": False},
                            "/models/hf/m1", tmp_path)
    assert len(steps) == 2
    opt, graph = steps
    assert opt[1].endswith("optimum_export.py") and opt[2:4] == ["export", "openvino"]
    assert opt[opt.index("--task") + 1] == "text-generation-with-past"
    assert opt[opt.index("--weight-format") + 1] == "int4" and opt[-1] == str(tmp_path / "model") and "--trust-remote-code" in opt
    assert opt[opt.index("--group-size") + 1] == "128" and opt[opt.index("--ratio") + 1] == "1.0" and "--sym" not in opt
    assert "text_generation" in graph and graph[graph.index("--source_model") + 1] == str(tmp_path / "model")
    assert graph[graph.index("--config_file_path") + 1] == str(tmp_path / "config.json")
    assert graph[graph.index("--enable_prefix_caching") + 1] == "false" and "--extra_quantization_params" not in graph
    int8 = export.commands("decoder", {"weight_format": "int8", "task": "text-generation-with-past"}, "/models/hf/m4", tmp_path)[0]
    assert int8[int8.index("--weight-format") + 1] == "int8" and "--group-size" not in int8
    tok = export.tokenizer_command("/models/hf/m4", tmp_path / "model")
    assert tok[0] == "convert_tokenizer" and "--with-detokenizer" in tok and tok[-1] == "/models/hf/m4"
    g = tmp_path / "graph.pbtxt"
    g.write_text('node {\n  models_path: "/models/ov/google/x/int8-ov/abc/model",\n  cache_size: 2,\n}')
    assert export.normalize_graph_model_path(g) is True and 'models_path: "./"' in g.read_text()
    assert export.normalize_graph_model_path(g) is False
    t5 = export.commands("encoder-decoder", {"weight_format": "int8", "task": "text2text-generation-with-past"},
                         "/models/hf/m5", tmp_path)[0]
    assert t5[1].endswith("optimum_export.py") and t5[2:4] == ["export", "openvino"] and "text2text-generation-with-past" in t5
    cls = export.commands("encoder-classifier", {"weight_format": "fp16"}, "/models/hf/m3", tmp_path)[0]
    assert cls[cls.index("--task") + 1] == "text-classification"
    # bf16-ov decoders: the same two steps as INT8 with the FP16 weight format and no quantization flags.
    bf, bf_graph = export.commands("decoder", {"weight_format": "fp16", "task": "text-generation-with-past"}, "/models/hf/m1", tmp_path)
    assert bf[bf.index("--weight-format") + 1] == "fp16" and not {"--group-size", "--ratio", "--sym"} & set(bf)
    assert bf_graph[bf_graph.index("--weight-format") + 1] == "fp16"


def test_graph_rewrite_sets_cell_fields():
    out = export.rewrite_graph(GRAPH, {"enable_prefix_caching": False, "cache_size": 32, "max_num_seqs": 128,
                                       "max_num_batched_tokens": 2048})
    assert "enable_prefix_caching: false" in out and "cache_size: 32" in out
    assert "max_num_seqs: 128" in out and "max_num_batched_tokens: 2048" in out
    assert out.count("cache_size") == 1


def test_graph_rewrite_accepts_the_header_export_model_py_renders():
    """export_model.py (model_server 2026.4.1) renders the options header as
    "[type.googleapis.com / mediapipe.LLMCalculatorOptions]" with spaces around the slash. The
    anchor regex matched only the unspaced form, so adding max_num_batched_tokens (absent from
    the rendered graph) died with "graph.pbtxt has no LLMCalculatorOptions block" — the first
    live bf16-ov bring-up, 2026-11-13."""
    rendered = GRAPH.replace("[type.googleapis.com/mediapipe.LLMCalculatorOptions]",
                             "[type.googleapis.com / mediapipe.LLMCalculatorOptions]")
    assert "max_num_batched_tokens" not in rendered
    out = export.rewrite_graph(rendered, {"enable_prefix_caching": False, "cache_size": 32, "max_num_seqs": 128,
                                          "max_num_batched_tokens": 2048})
    assert "max_num_batched_tokens: 2048" in out and "cache_size: 32" in out and "enable_prefix_caching: false" in out
    assert out.index("max_num_batched_tokens") > out.index("LLMCalculatorOptions]")


def test_every_export_runs_through_the_cap_lifting_wrapper(tmp_path):
    """Gemma 2 exports were refused by optimum-intel 2.2.0's inherited Transformers cap (5.0) under
    the pinned 5.5.4 (2026-11-15). All optimum-cli calls go through optimum_export.py, which lifts
    exactly the caps upstream lifted (Gemma 2: commit 2092b03) and passes every argument through."""
    import importlib.util as ilu

    wrapper = Path(export.__file__).with_name("optimum_export.py")
    assert wrapper.exists()
    for mtype, exp in (("decoder", {"weight_format": "int8"}), ("encoder-decoder", {"weight_format": "int8"}),
                       ("encoder-classifier", {"weight_format": "fp16"})):
        first = export.commands(mtype, exp, "/models/hf/x", tmp_path)[0]
        assert first[1] == str(wrapper) and first[2:4] == ["export", "openvino"], first[:4]
        assert "optimum-cli" not in first
    spec = ilu.spec_from_file_location("optimum_export", wrapper)
    mod = ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.LIFTED == {"gemma2": ("Gemma2OpenVINOConfig", "2092b03709")}
    # The Dockerfile ships the wrapper next to export.py.
    assert "servers/export/optimum_export.py" in (Path(export.__file__).parent / "Dockerfile").read_text()


def test_cap_lift_on_the_pinned_converter():
    """With optimum-intel installed (the export image, or a dev venv), lifting removes only the
    Gemma 2 cap and leaves Gemma (1) and every other config alone."""
    pytest.importorskip("optimum.exporters.openvino")
    import importlib.util as ilu

    from optimum.exporters.openvino import model_configs as mc

    spec = ilu.spec_from_file_location("optimum_export", Path(export.__file__).with_name("optimum_export.py"))
    mod = ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    before = {n: getattr(c, "MAX_TRANSFORMERS_VERSION", None) for n, c in vars(mc).items()
              if isinstance(c, type) and hasattr(c, "MAX_TRANSFORMERS_VERSION")}
    assert mod.lift() == ["Gemma2OpenVINOConfig"]
    after = {n: getattr(c, "MAX_TRANSFORMERS_VERSION", None) for n, c in vars(mc).items()
             if isinstance(c, type) and hasattr(c, "MAX_TRANSFORMERS_VERSION")}
    assert after["Gemma2OpenVINOConfig"] is None and after["GemmaOpenVINOConfig"] == before["GemmaOpenVINOConfig"]
    # Only Gemma 2 and the configs that inherit its cap (Gemma 3/3n/4 text) change, the same as
    # upstream's class attribute; Gemma (1), Llama, Qwen, T5, RoBERTa keep theirs.
    changed = {n for n in before if before[n] != after[n]}
    assert "Gemma2OpenVINOConfig" in changed
    assert all(issubclass(getattr(mc, n), mc.Gemma2OpenVINOConfig) for n in changed), changed
    for keep in ("GemmaOpenVINOConfig", "LlamaOpenVINOConfig", "Qwen2OpenVINOConfig", "T5OpenVINOConfig"):
        if keep in before:
            assert keep not in changed

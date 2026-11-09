"""Export helpers that do not need OpenVINO installed (task 10)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

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
    dec = export.commands("decoder", {"weight_format": "int4", "group_size": 128, "ratio": 1.0, "sym": False},
                          "/models/hf/m1", tmp_path)[0]
    assert "text_generation" in dec and dec[dec.index("--weight-format") + 1] == "int4"
    assert dec[dec.index("--extra_quantization_params") + 1] == "--group-size 128 --ratio 1.0"
    assert dec[dec.index("--enable_prefix_caching") + 1] == "false"
    t5 = export.commands("encoder-decoder", {"weight_format": "int8", "task": "text2text-generation-with-past"},
                         "/models/hf/m5", tmp_path)[0]
    assert t5[:3] == ["optimum-cli", "export", "openvino"] and "text2text-generation-with-past" in t5
    cls = export.commands("encoder-classifier", {"weight_format": "fp16"}, "/models/hf/m3", tmp_path)[0]
    assert cls[cls.index("--task") + 1] == "text-classification"


def test_graph_rewrite_sets_cell_fields():
    out = export.rewrite_graph(GRAPH, {"enable_prefix_caching": False, "cache_size": 32, "max_num_seqs": 128,
                                       "max_num_batched_tokens": 2048})
    assert "enable_prefix_caching: false" in out and "cache_size: 32" in out
    assert "max_num_seqs: 128" in out and "max_num_batched_tokens: 2048" in out
    assert out.count("cache_size") == 1

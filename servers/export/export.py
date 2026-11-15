"""Export a catalog model to OpenVINO IR (design 7.4).

    python export.py ir --model-type decoder --precision int4-ov --source /models/hf/m1 \
        --out /models/ov --export-json '{"weight_format": "int4", "group_size": 128, ...}'
    python export.py graph --dir /models/ov/<hash> --enable-prefix-caching false --cache-size 32

`ir` writes the IR to <out>/<export-hash>/ where the hash covers the export arguments,
the source revision and the tool versions, so a changed setting never reuses an old IR.
Decoders use OVMS's export_model.py (pinned at model_server v2026.4.1) so the layout and
graph.pbtxt are what OVMS expects; Flan-T5 and the classifiers use optimum-cli. `graph`
rewrites the per-cell serving fields of an existing graph.pbtxt without re-exporting.
Every run writes export.json with the command, versions and wall time.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as md
import json
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

OVMS_EXPORT = Path(__file__).with_name("ovms") / "export_model.py"
# optimum-cli with the Transformers caps upstream has lifted (optimum_export.LIFTED); same arguments.
OPTIMUM = [sys.executable, str(Path(__file__).with_name("optimum_export.py"))]
OVMS_COMMIT = "6c25e51e064b967b371decb305483d6fb76032b1"
TOOLS = ("optimum-intel", "optimum", "openvino", "openvino-tokenizers", "nncf", "transformers", "torch")


def versions() -> dict[str, str]:
    out = {}
    for p in TOOLS:
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = "missing"
    out["ovms_export_model"] = OVMS_COMMIT
    return out


def export_hash(args: dict, source_revision: str, tool_versions: dict) -> str:
    blob = json.dumps({"args": args, "source_revision": source_revision, "tools": tool_versions}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _quant_flags(exp: dict) -> list[str]:
    flags = []
    if exp.get("group_size") is not None:
        flags += ["--group-size", str(exp["group_size"])]
    if exp.get("ratio") is not None:
        flags += ["--ratio", str(exp["ratio"])]
    if exp.get("sym"):
        flags.append("--sym")
    return flags


def commands(model_type: str, exp: dict, source: str, dest: Path) -> list[list[str]]:
    """The export steps. Every path runs optimum-cli ourselves with an explicit --task: optimum
    cannot infer the task for a local directory (it raises "Cannot infer the task from a local
    directory yet"), and the sources here are always local, pinned directories with no network.
    OVMS's export_model.py delegated to optimum-cli WITHOUT --task, so every decoder export failed
    (2026-11-12, Gemma INT8 on the OpenVINO option); it is now used only for what it is good at,
    rendering graph.pbtxt for an already exported model."""
    wf = exp["weight_format"]
    if model_type == "decoder":
        task = exp.get("task") or "text-generation-with-past"
        model_dir = dest / "model"
        optimum = [*OPTIMUM, "export", "openvino", "--model", source, "--task", task, "--weight-format", wf,
                   *_quant_flags(exp), "--trust-remote-code", str(model_dir)]
        # export_model.py sees openvino_model.xml in --source_model, skips conversion and only writes
        # <dest>/model/graph.pbtxt (+ a servable config next to it; its default config.json would land
        # in the read-only /app).
        graph = [sys.executable, str(OVMS_EXPORT), "text_generation", "--source_model", str(model_dir),
                 "--model_name", "model", "--weight-format", wf, "--model_repository_path", str(dest),
                 "--config_file_path", str(dest / "config.json"),
                 "--target_device", "CPU", "--enable_prefix_caching", "false", "--overwrite_models"]
        return [optimum, graph]
    task = exp.get("task") or ("text2text-generation-with-past" if model_type == "encoder-decoder"
                               else "text-classification")
    return [[*OPTIMUM, "export", "openvino", "--model", source, "--task", task, "--weight-format", wf,
             *_quant_flags(exp), str(dest / "model")]]


def tokenizer_command(source: str, model_dir: Path) -> list[str]:
    """OVMS needs openvino_tokenizer/detokenizer next to the model; optimum-cli writes them when
    openvino-tokenizers is installed, this is the fallback export_model.py would have run."""
    return ["convert_tokenizer", "--with-detokenizer", "--trust-remote-code", "-o", str(model_dir), source]


def normalize_graph_model_path(graph: Path) -> bool:
    """export_model.py renders models_path as the absolute source directory; the model files sit
    next to graph.pbtxt, and "./" keeps the IR relocatable (host mount paths differ by instance)."""
    text = graph.read_text()
    new = re.sub(r'(models_path\s*:\s*)"[^"]*"', r'\1"./"', text, count=1)
    if new != text:
        graph.write_text(new)
    return new != text


def run_ir(a: argparse.Namespace) -> int:
    exp = json.loads(a.export_json)
    if "weight_format" not in exp:
        raise SystemExit("--export-json needs weight_format")
    rev_file = Path(a.source) / ".revision"          # written by host/stage_model.sh
    src_rev = a.source_revision or (rev_file.read_text().strip() if rev_file.exists() else "unknown")
    tv = versions()
    h = export_hash({"model_type": a.model_type, **exp}, src_rev, tv)
    dest = Path(a.out) / h
    # The lifted caps are recorded, not hashed: they change nothing for a model that was not refused,
    # so IRs already cached for other architectures stay valid.
    sys.path.insert(0, str(Path(__file__).parent))
    from optimum_export import LIFTED
    record = {"export_hash": h, "model_type": a.model_type, "export": exp, "source": a.source,
              "source_revision": src_rev, "tools": tv,
              "lifted_transformers_caps": {k: {"config": v[0], "upstream_commit": v[1]} for k, v in LIFTED.items()}}
    if (dest / "export.json").exists() and not a.force:
        print(json.dumps({**record, "reused": True}))
        return 0
    dest.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    ran = []

    def run(cmd: list[str]) -> None:
        print("+ " + shlex.join(cmd), file=sys.stderr, flush=True)
        subprocess.run(cmd, check=True)
        ran.append(shlex.join(cmd))

    steps = commands(a.model_type, exp, a.source, dest)
    run(steps[0])
    model_dir = dest / "model"
    if a.model_type == "decoder":
        if not (model_dir / "openvino_detokenizer.xml").exists():
            run(tokenizer_command(a.source, model_dir))
        for cmd in steps[1:]:
            run(cmd)
        graph = model_dir / "graph.pbtxt"
        if not graph.exists():
            raise SystemExit(f"export_model.py did not write {graph}")
        normalize_graph_model_path(graph)
    else:
        for cmd in steps[1:]:
            run(cmd)
    record.update(export_s=round(time.monotonic() - t0, 1), commands=ran)
    (dest / "export.json").write_text(json.dumps(record, indent=2, sort_keys=True))
    print(json.dumps({**record, "reused": False}))
    return 0


GRAPH_FIELDS = {"enable_prefix_caching": "bool", "cache_size": "int", "max_num_seqs": "int",
                "max_num_batched_tokens": "int"}


def rewrite_graph(text: str, values: dict) -> str:
    """Set fields inside the LLM node options of an OVMS graph.pbtxt."""
    for key, val in values.items():
        if val is None:
            continue
        rendered = ("true" if val else "false") if GRAPH_FIELDS[key] == "bool" else str(int(val))
        pat = re.compile(rf"(\b{key}\s*:\s*)([^\s,}}]+)")
        if pat.search(text):
            text = pat.sub(lambda m, r=rendered: m.group(1) + r, text, count=1)
        else:
            # export_model.py renders the header as "[type.googleapis.com / mediapipe.LLMCalculatorOptions]"
            # (spaces around the slash); accept both spellings.
            anchor = re.search(r"(\[type\.googleapis\.com\s*/\s*mediapipe\.LLMCalculatorOptions\]\s*:\s*\{)", text)
            if not anchor:
                raise SystemExit("graph.pbtxt has no LLMCalculatorOptions block")
            text = text[:anchor.end()] + f"\n          {key}: {rendered}," + text[anchor.end():]
    return text


def run_graph(a: argparse.Namespace) -> int:
    paths = list(Path(a.dir).rglob("graph.pbtxt"))
    if len(paths) != 1:
        raise SystemExit(f"expected one graph.pbtxt under {a.dir}, found {len(paths)}")
    vals = {"enable_prefix_caching": None if a.enable_prefix_caching is None else a.enable_prefix_caching == "true",
            "cache_size": a.cache_size, "max_num_seqs": a.max_num_seqs,
            "max_num_batched_tokens": a.max_num_batched_tokens}
    paths[0].write_text(rewrite_graph(paths[0].read_text(), vals))
    print(json.dumps({"graph": str(paths[0]), **{k: v for k, v in vals.items() if v is not None}}))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ir = sub.add_parser("ir")
    ir.add_argument("--model-type", required=True, choices=["decoder", "encoder-decoder", "encoder-classifier"])
    ir.add_argument("--source", required=True, help="local directory with the pinned Hugging Face files")
    ir.add_argument("--source-revision", default=None)
    ir.add_argument("--out", required=True)
    ir.add_argument("--export-json", required=True, help="the precision's export block from models.yaml")
    ir.add_argument("--force", action="store_true")
    g = sub.add_parser("graph")
    g.add_argument("--dir", required=True)
    g.add_argument("--enable-prefix-caching", choices=["true", "false"], default=None)
    g.add_argument("--cache-size", type=int, default=None)
    g.add_argument("--max-num-seqs", type=int, default=None)
    g.add_argument("--max-num-batched-tokens", type=int, default=None)
    a = ap.parse_args(argv)
    return run_ir(a) if a.cmd == "ir" else run_graph(a)


if __name__ == "__main__":
    sys.exit(main())

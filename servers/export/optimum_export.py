"""`optimum-cli` with upstream-lifted Transformers version caps (design 7.4).

    python optimum_export.py export openvino --model <dir> --task ... <out>

Same arguments as `optimum-cli`. Before it runs, the export configs listed in LIFTED get
their MAX_TRANSFORMERS_VERSION removed, exactly as optimum-intel main has done, so a model
whose conversion code is unchanged upstream is not refused only because the pinned release
still carries the old cap. Nothing else is altered.

Why: optimum-intel 2.2.0 (the latest release, 2026-09-17) inherits MAX_TRANSFORMERS_VERSION
"5.0" for Gemma 2 from its Gemma config, and the export image pins transformers 5.5.4, so every
Gemma 2 export failed with "Maximum required is 5.0, got: 5.5.4" (run gemma-20261115T102422Z).
Upstream commit 2092b03 (2026-10-23, "Add support for Transformers 5.11-5.15") sets
Gemma2OpenVINOConfig.MAX_TRANSFORMERS_VERSION = None and does not touch Gemma2ModelPatcher.
Checked with these exact pins on optimum-intel-internal-testing/tiny-random-gemma2: prefill
logits within 1.1e-6 of PyTorch FP32 and 24 greedy tokens identical. Remove an entry when the
pinned optimum-intel release carries the change.
"""

from __future__ import annotations

import sys

# HF model_type -> (export config class in optimum.exporters.openvino.model_configs, upstream commit)
LIFTED = {"gemma2": ("Gemma2OpenVINOConfig", "2092b03709")}


def lift(model_types=None) -> list[str]:
    """Remove the cap from the listed configs; returns the class names changed."""
    from optimum.exporters.openvino import model_configs

    done = []
    for mt, (cls_name, _commit) in LIFTED.items():
        if model_types is not None and mt not in model_types:
            continue
        cls = getattr(model_configs, cls_name)
        # Set on the class itself: an inherited cap ("5.0" from GemmaOpenVINOConfig) is what refuses.
        cls.MAX_TRANSFORMERS_VERSION = None
        done.append(cls_name)
    return done


def main(argv: list[str]) -> int:
    changed = lift()
    print(f"optimum_export: Transformers cap lifted for {', '.join(changed)} (as upstream {LIFTED['gemma2'][1]})",
          file=sys.stderr, flush=True)
    from optimum.commands.optimum_cli import main as optimum_main

    sys.argv = ["optimum-cli", *argv]
    return optimum_main() or 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

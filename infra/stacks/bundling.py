"""Lambda code bundle: bench/, lambdas/ and config/ plus pinned dependencies for
linux/arm64, built locally with uv (no Docker needed at synth)."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path

import jsii
from aws_cdk import AssetHashType, BundlingOptions, DockerImage, ILocalBundling
from aws_cdk import aws_lambda as lambda_

ROOT = Path(__file__).resolve().parents[2]
REQUIREMENTS = ["httpx==0.28.1", "pydantic==2.14.3", "pyyaml==6.0.3", "numpy==2.5.4", "matplotlib==3.11.2",
                "boto3==1.43.126"]
PACKAGES = ("bench", "lambdas")


@jsii.implements(ILocalBundling)
class _UvBundling:
    def try_bundle(self, output_dir: str, *, image=None, **_kw) -> bool:
        uv = shutil.which("uv")
        if uv is None:
            return False
        out = Path(output_dir)
        # Lambda python3.12 runs on Amazon Linux 2023 (glibc 2.34); numpy 2.5 ships aarch64
        # wheels for manylinux_2_28 only, so manylinux2014 finds no wheel.
        subprocess.run([uv, "pip", "install", "--quiet", "--target", str(out), "--python-platform",  # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-audit (argv list, paths from this file)
                        "aarch64-manylinux_2_28", "--python-version", "3.12", "--only-binary", ":all:",
                        *REQUIREMENTS], check=True)
        for pkg in PACKAGES:
            shutil.copytree(ROOT / pkg, out / pkg, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
                            dirs_exist_ok=True)
        shutil.copytree(ROOT / "config", out / "config", dirs_exist_ok=True)
        return True


def source_hash() -> str:
    """Hash of every file that goes into the bundle plus the pinned requirements. CDK's own
    source fingerprint sees nothing through the `**` + `!dir/**` exclude list (a directory
    excluded by `**` cannot be re-included), so without this the asset hash never changed
    and code edits were not deployed (seen 2026-11-09)."""
    h = hashlib.sha256("\n".join(REQUIREMENTS).encode())
    for pkg in (*PACKAGES, "config"):
        for f in sorted((ROOT / pkg).rglob("*")):
            if f.is_file() and "__pycache__" not in f.parts and f.suffix != ".pyc":
                h.update(str(f.relative_to(ROOT)).encode())
                h.update(f.read_bytes())
    return h.hexdigest()


def lambda_code(skip: bool = False) -> lambda_.Code:
    """skip=True (unit tests): the source tree only, no dependency install."""
    if skip:
        return lambda_.Code.from_asset(str(ROOT), exclude=["**", "!bench/**", "!lambdas/**", "!config/**",
                                                            "**/__pycache__/**"])
    return lambda_.Code.from_asset(
        str(ROOT), exclude=["**", "!bench/**", "!lambdas/**", "!config/**", "**/__pycache__/**"],
        asset_hash=source_hash(), asset_hash_type=AssetHashType.CUSTOM,
        bundling=BundlingOptions(image=DockerImage.from_registry("public.ecr.aws/sam/build-python3.12:latest-arm64"),
                                 local=_UvBundling(),
                                 command=["bash", "-c", "pip install -t /asset-output " + " ".join(REQUIREMENTS)
                                          + " && cp -r /asset-input/bench /asset-input/lambdas /asset-input/config"
                                            " /asset-output/"]))

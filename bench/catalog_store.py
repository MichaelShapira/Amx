"""The model catalog as data, not as a file baked into every deployment.

`config/models.yaml` in the repository is the seed: the seven models the benchmark shipped with.
The live catalog is the object `config/models.yaml` in the bench bucket, written by the web app's
Setup page when a user adds or removes a model. Every reader (the web app, the pre-check and the
report Lambdas) calls `load_catalog()`: the bucket copy when there is one, else the seed. The
bucket name comes from the environment the deployment already sets (`BENCH_BUCKET` in the Lambdas,
`AMXBENCH_OUTPUTS` in the web app) or is passed explicitly. The result is cached for a minute per
process; `invalidate()` after a write.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import yaml

from bench.config import ModelCatalog, load_models

ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / "config" / "models.yaml"
KEY = "config/models.yaml"
TTL_S = 60

_lock = threading.Lock()
_cache: dict = {"at": 0.0, "bucket": None, "catalog": None, "source": None}


def default_bucket() -> str | None:
    for var in ("BENCH_BUCKET", "BUCKET"):                 # the Lambdas (lambdas/common.py), then a plain override
        b = os.environ.get(var, "").strip()
        if b:
            return b
    out = os.environ.get("AMXBENCH_OUTPUTS", "").strip()
    if out:
        try:
            return json.loads(out).get("BucketName") or None
        except ValueError:
            return None
    return None


def _default_s3():
    import boto3

    return boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-east-1"))


# The web app swaps this for its credential-file-aware client (webui/api/aws.py); the Lambdas keep boto3.
s3_factory = _default_s3


def _s3():
    return s3_factory()


def load_catalog(bucket: str | None = None, *, fresh: bool = False) -> ModelCatalog:
    """The live catalog: the bucket object when it exists and parses, else the seed file."""
    bucket = bucket or default_bucket()
    with _lock:
        if (not fresh and _cache["catalog"] is not None and _cache["bucket"] == bucket
                and time.time() - _cache["at"] < TTL_S):
            return _cache["catalog"]
        catalog, source = None, "seed"
        if bucket:
            try:
                body = _s3().get_object(Bucket=bucket, Key=KEY)["Body"].read()
                catalog = ModelCatalog.model_validate(yaml.safe_load(body))
                source = f"s3://{bucket}/{KEY}"
            except Exception as e:  # noqa: BLE001 - no object yet, no permission, or invalid: the seed answers
                if type(e).__name__ not in ("NoSuchKey",):
                    _cache["last_error"] = f"{type(e).__name__}: {str(e)[:160]}"
        if catalog is None:
            catalog = load_models(SEED)
        _cache.update(at=time.time(), bucket=bucket, catalog=catalog, source=source)
        return catalog


def source() -> str | None:
    return _cache["source"]


def save_catalog(catalog: ModelCatalog, bucket: str | None = None) -> str:
    """Validate and write the catalog to the bucket; returns the key. The YAML is written from
    the validated model so a hand edit can never leave the object unparseable."""
    bucket = bucket or default_bucket()
    if not bucket:
        raise RuntimeError("no bucket to save the catalog to")
    ModelCatalog.model_validate(catalog.model_dump(exclude_none=True))     # re-validate before writing
    text = yaml.safe_dump(json.loads(catalog.model_dump_json(exclude_none=True)), sort_keys=False, allow_unicode=True)
    _s3().put_object(Bucket=bucket, Key=KEY, Body=text.encode("utf-8"), ContentType="application/yaml")
    invalidate()
    return KEY


def invalidate() -> None:
    with _lock:
        _cache.update(at=0.0, catalog=None)

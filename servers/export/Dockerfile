# syntax=docker/dockerfile:1.7
# bench-export (design 7.4): the pinned OpenVINO export stack plus OVMS's export_model.py
# at model_server v2026.4.1 (commit 6c25e51e), checked by sha256. Runs on the T3 instance.
# Build from the repository root:
#   docker build -f servers/export/Dockerfile .
FROM public.ecr.aws/docker/library/python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b

WORKDIR /app
COPY servers/export/requirements.lock /app/requirements.lock
RUN python -m pip install --no-cache-dir --require-hashes --no-deps -r /app/requirements.lock \
        --extra-index-url https://download.pytorch.org/whl/cpu
# OVMS's export_model.py at a pinned commit, fetched and sha256-checked in one step (the slim base has no curl).
RUN python - <<'PY'
import hashlib, pathlib, urllib.request
url = ("https://raw.githubusercontent.com/openvinotoolkit/model_server/6c25e51e064b967b371decb305483d6fb76032b1"
       "/demos/common/export_models/export_model.py")
want = "783df4b5bfbadd6b19574bef5a6deb3b5d51dc1218b8e375740a2a7edea2fb88"
with urllib.request.urlopen(url, timeout=60) as r:  # noqa: S310 - fixed https URL, checksum verified below
    data = r.read()
if hashlib.sha256(data).hexdigest() != want:
    raise SystemExit("export_model.py: checksum mismatch")
pathlib.Path("/app/ovms").mkdir(parents=True, exist_ok=True)
pathlib.Path("/app/ovms/export_model.py").write_bytes(data)
PY
COPY servers/export/export.py servers/export/optimum_export.py servers/export/probe_ov.py /app/

ENV PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin bench && chmod a+r /app/ovms/export_model.py
USER 10001
# A one-shot export job, not a service: there is no endpoint to probe.
HEALTHCHECK NONE
ENTRYPOINT ["python", "/app/export.py"]

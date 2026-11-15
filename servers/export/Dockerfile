# syntax=docker/dockerfile:1.7
# bench-export (design 7.4): the pinned OpenVINO export stack plus OVMS's export_model.py
# at model_server v2026.4.1 (commit 6c25e51e), checked by sha256. Runs on the T3 instance.
# Build from the repository root:
#   docker build -f servers/export/Dockerfile .
FROM python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b

WORKDIR /app
COPY servers/export/requirements.lock /app/requirements.lock
RUN python -m pip install --no-cache-dir --require-hashes --no-deps -r /app/requirements.lock \
        --extra-index-url https://download.pytorch.org/whl/cpu
ADD --checksum=sha256:783df4b5bfbadd6b19574bef5a6deb3b5d51dc1218b8e375740a2a7edea2fb88 \
    https://raw.githubusercontent.com/openvinotoolkit/model_server/6c25e51e064b967b371decb305483d6fb76032b1/demos/common/export_models/export_model.py \
    /app/ovms/export_model.py
COPY servers/export/export.py servers/export/optimum_export.py servers/export/probe_ov.py /app/

ENV PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin bench && chmod a+r /app/ovms/export_model.py
USER 10001
ENTRYPOINT ["python", "/app/export.py"]

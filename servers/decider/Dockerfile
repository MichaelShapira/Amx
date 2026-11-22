# bench-decider:cpu and bench-decider:cuda — the official Strands Decider server on the AWS DLC PyTorch base.
# Build from the repository root:
#   docker build -f servers/decider/Dockerfile --build-arg BASE_IMAGE=<cpu or cuda digest> .
# torch/torchvision come from the base; everything else is pinned with hashes in requirements.lock, and the
# decider itself is pinned to a commit of its main branch (Gemma 4 support is not on PyPI yet).
# checkov:skip=CKV_DOCKER_7: BASE_IMAGE is always a digest-pinned reference (the default below, or config/images.yaml)
ARG BASE_IMAGE=public.ecr.aws/deep-learning-containers/pytorch:2.14.0-cpu-amzn2023@sha256:e9aa07c11afc5f7eff183782cf3dde397012e2f08ab700dffd2828203688c159
FROM ${BASE_IMAGE}
WORKDIR /app
RUN dnf install -y -q git && dnf clean all
COPY servers/decider/requirements.lock servers/decider/requirements-git.txt /app/
RUN python -m pip install --no-cache-dir --require-hashes --no-deps -r /app/requirements.lock \
    && python -m pip install --no-cache-dir --no-deps -r /app/requirements-git.txt \
    && python -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__)" \
    && python -m pip show strands-decider peft | grep -E '^(Name|Version)' \
    && python -c "import importlib.util as u; assert u.find_spec('strands_decider') and u.find_spec('peft')"
# Importing peft or strands_decider is left to run time: on the CUDA base, peft probes transformer_engine, which
# dlopens libcuda.so.1, absent at image build time (the build failed on it, 2026-11-22).
COPY servers/decider/entrypoint.py /app/entrypoint.py
# Models are mounted read-only at /models; the entrypoint lays the staged torso into an offline Hub cache under
# HF_HOME. The server runs as a user with no home directory, so every cache goes to /tmp.
ENV PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    HOME=/tmp XDG_CACHE_HOME=/tmp/cache TRITON_CACHE_DIR=/tmp/cache/triton TORCHINDUCTOR_CACHE_DIR=/tmp/cache/inductor \
    HF_HOME=/tmp/cache/huggingface
RUN useradd --uid 10001 --no-create-home --shell /sbin/nologin bench && mkdir -p /tmp/cache && chmod 1777 /tmp /tmp/cache
USER 10001
EXPOSE 8000
# The model takes minutes to load (start-period); afterwards /health answers once the server is up.
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen(f\"http://127.0.0.1:{os.environ.get('DECIDER_PORT', '8000')}/health\", timeout=4)"]
ENTRYPOINT []
CMD ["python", "/app/entrypoint.py"]

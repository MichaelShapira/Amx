# bench-decider:cpu and bench-decider:cuda — the official Strands Decider server on the AWS DLC PyTorch base.
# Build from the repository root:
#   docker build -f servers/decider/Dockerfile --build-arg BASE_IMAGE=<cpu or cuda digest> .
# torch/torchvision come from the base; everything else is pinned with hashes in requirements.lock, and the
# decider itself is pinned to a commit of its main branch (Gemma 4 support is not on PyPI yet).
ARG BASE_IMAGE=public.ecr.aws/deep-learning-containers/pytorch:2.14.0-cpu-amzn2023@sha256:e9aa07c11afc5f7eff183782cf3dde397012e2f08ab700dffd2828203688c159
FROM ${BASE_IMAGE}
WORKDIR /app
RUN dnf install -y -q git && dnf clean all
COPY servers/decider/requirements.lock servers/decider/requirements-git.txt /app/
RUN python -m pip install --no-cache-dir --require-hashes --no-deps -r /app/requirements.lock \
    && python -m pip install --no-cache-dir --no-deps -r /app/requirements-git.txt \
    && python -c "import torch, transformers, peft, strands_decider; print('torch', torch.__version__, 'transformers', transformers.__version__, 'peft', peft.__version__)"
COPY servers/decider/entrypoint.py /app/entrypoint.py
# Models are mounted read-only at /models; the entrypoint lays the staged torso into an offline Hub cache under
# HF_HOME. The server runs as a user with no home directory, so every cache goes to /tmp.
ENV PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    HOME=/tmp XDG_CACHE_HOME=/tmp/cache TRITON_CACHE_DIR=/tmp/cache/triton TORCHINDUCTOR_CACHE_DIR=/tmp/cache/inductor \
    HF_HOME=/tmp/cache/huggingface
RUN useradd --uid 10001 --no-create-home --shell /sbin/nologin bench && mkdir -p /tmp/cache && chmod 1777 /tmp /tmp/cache
USER 10001
EXPOSE 8000
ENTRYPOINT []
CMD ["python", "/app/entrypoint.py"]

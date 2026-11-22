# syntax=docker/dockerfile:1.7
# The web app for ECS Fargate (infra/stacks/webapp.py): the React front end built from source
# and served by the FastAPI API, Python dependencies from the hashed lock only, non-root, port 8080
# published by the platform. Build from the repository root:
#   docker build -f webui/Dockerfile .

FROM public.ecr.aws/docker/library/node:22-bookworm-slim@sha256:e9f565d32859f0cbfb52125eafe8be39390f24dac302dc3cdf566e8ba3ab21ee AS ui
WORKDIR /ui
COPY webui/package.json webui/package-lock.json ./
RUN npm ci --no-fund --no-audit
COPY webui/index.html webui/vite.config.ts webui/tsconfig.json ./
COPY webui/src ./src
RUN npm run build

FROM public.ecr.aws/docker/library/python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b
WORKDIR /app
COPY webui/requirements.lock /app/webui/requirements.lock
RUN python -m pip install --no-cache-dir --require-hashes --no-deps -r /app/webui/requirements.lock
COPY bench /app/bench
COPY lambdas /app/lambdas
COPY tools /app/tools
COPY config /app/config
# Shown verbatim on the Technical details page: the engine Dockerfiles, the export tooling, the host
# scripts, this Dockerfile and the architecture diagram (docs/architecture.svg from tools/architecture_diagram.py).
COPY servers/thin/Dockerfile.torch servers/thin/Dockerfile.openvino /app/servers/thin/
COPY servers/export/Dockerfile servers/export/export.py servers/export/optimum_export.py /app/servers/export/
COPY host/stage_model.sh host/serve.sh /app/host/
COPY webui/Dockerfile /app/webui/Dockerfile
COPY docs/architecture.svg docs/architecture.drawio docs/USER_GUIDE.md /app/docs/
COPY webui/__init__.py /app/webui/__init__.py
COPY webui/api /app/webui/api
COPY --from=ui /ui/dist /app/webui/dist

# The app writes only under prompts/ and data/: the sync from the bench bucket at start (entrypoint.py)
# and the prompt pools a model added in Setup gets (onboarding.py).
RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin app \
    && mkdir -p /app/prompts /app/data/home /app/data/cache && chown -R 10001:10001 /app/prompts /app/data \
    && find /app -name __pycache__ -type d -prune -exec rm -rf {} +
# HOME and the Hugging Face cache point under the writable data tree: the app user has no home
# directory, and the onboarding job downloads tokenizers and the quality-set sources there.
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/app \
    HOME=/app/data/home HF_HOME=/app/data/cache/hf-home \
    AMXBENCH_BIND=0.0.0.0 PORT=8080 AMXBENCH_DATA_SYNC=1 AMXBENCH_CREDS_FILE=
USER 10001
EXPOSE 8080
# Liveness only (the ECS task definition runs the same probe); says nothing about data or identity.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/api/health', timeout=3).status == 200 else 1)"]
CMD ["python", "-m", "webui.api.entrypoint"]

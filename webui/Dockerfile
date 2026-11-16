# syntax=docker/dockerfile:1.7
# The web app for AWS App Runner (infra/stacks/webapp.py): the React front end built from source
# and served by the FastAPI API, Python dependencies from the hashed lock only, non-root, port 8080
# published by the platform. Build from the repository root:
#   docker build -f webui/Dockerfile .

FROM node:22-bookworm-slim@sha256:e9f565d32859f0cbfb52125eafe8be39390f24dac302dc3cdf566e8ba3ab21ee AS ui
WORKDIR /ui
COPY webui/package.json webui/package-lock.json ./
RUN npm ci --no-fund --no-audit
COPY webui/index.html webui/vite.config.ts webui/tsconfig.json ./
COPY webui/src ./src
RUN npm run build

FROM python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b
WORKDIR /app
COPY webui/requirements.lock /app/webui/requirements.lock
RUN python -m pip install --no-cache-dir --require-hashes --no-deps -r /app/webui/requirements.lock
COPY bench /app/bench
COPY lambdas /app/lambdas
COPY config /app/config
COPY webui/__init__.py /app/webui/__init__.py
COPY webui/api /app/webui/api
COPY --from=ui /ui/dist /app/webui/dist

# The app writes only the data it syncs from the bench bucket at start (entrypoint.py).
RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin app \
    && mkdir -p /app/prompts /app/data && chown -R 10001:10001 /app/prompts /app/data \
    && find /app -name __pycache__ -type d -prune -exec rm -rf {} +
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/app \
    AMXBENCH_BIND=0.0.0.0 PORT=8080 AMXBENCH_DATA_SYNC=1 AMXBENCH_CREDS_FILE=
USER 10001
EXPOSE 8080
CMD ["python", "-m", "webui.api.entrypoint"]

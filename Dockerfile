# PillSpark: FastAPI service + landing page in one image. Render sets $PORT.
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
COPY backend/requirements.txt backend/requirements-sie.txt backend/
RUN pip install -r backend/requirements.txt && pip install --no-deps -r backend/requirements-sie.txt
COPY backend backend
COPY landing landing
# Runtime data (orders, uploads, memory, cognee stores) lives outside the code; on Render Free it is ephemeral.
ENV PILLSPARK_DATA_DIR=/tmp/pillspark MODEL_BACKEND=sie MEMORY_BACKEND=auto PILLSPARK_RECORD=0
RUN useradd --create-home --uid 10001 app && mkdir -p /tmp/pillspark && chown -R app /app /tmp/pillspark
USER app
EXPOSE 10000
HEALTHCHECK CMD python -c "import urllib.request,os;urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"10000\")}/healthz')"
CMD ["sh", "-c", "uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port ${PORT:-10000} --proxy-headers --forwarded-allow-ips='*'"]

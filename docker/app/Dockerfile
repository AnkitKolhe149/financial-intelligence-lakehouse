# ---------------------------------------------------------------------------
# Application image for the serving API, the source-system API and the dashboard.
#
# No JVM and no PySpark: these services never touch Spark, and adding a JDK
# would triple the image for nothing.
#
# Build:  docker compose build api
# ---------------------------------------------------------------------------
FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# curl is used by the compose healthchecks; build-essential is dropped again
# after the wheels are built so it does not ship in the final image.
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY api/ ./api/
COPY dashboard/ ./dashboard/
COPY config/ ./config/
COPY sql/ ./sql/
COPY scripts/ ./scripts/
COPY pyproject.toml ./

RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/data /app/metadata \
    && chown -R appuser:appuser /app
USER appuser

ENV PYTHONPATH=/app

EXPOSE 8000 8001 8501

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ---------------------------------------------------------------------------
# Spark + Delta execution image.
#
# Python 3.12, not 3.13+: PySpark 4.0 supports 3.9-3.12 and fails to import on
# newer interpreters. This pin is the entire reason Spark is containerised
# rather than run on the host - the host here runs Python 3.14.
#
# Build:  docker compose build spark
# Run:    docker compose run --rm spark python -m src.pipelines.spark_pipeline
# ---------------------------------------------------------------------------
FROM python:3.12-slim-bookworm

# JDK 17 is what Spark 4 targets. headless keeps the image ~150MB smaller than
# the full JDK, and nothing here draws a window.
RUN apt-get update && apt-get install -y --no-install-recommends \
        openjdk-17-jdk-headless \
        procps \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64 \
    PATH="/usr/lib/jvm/java-17-openjdk-amd64/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYSPARK_PYTHON=python \
    PYSPARK_DRIVER_PYTHON=python

WORKDIR /app

# Requirements first so the dependency layer is cached across source changes -
# the Spark jars alone are ~300MB and should not be re-fetched for a typo fix.
COPY requirements.txt requirements-spark.txt ./
RUN pip install --no-cache-dir -r requirements-spark.txt

# Pre-fetch the Delta jars at build time. Without this, the first job pays a
# Maven download inside the run, which then fails on an air-gapped host.
RUN python -c "\
from pyspark.sql import SparkSession; \
import delta; \
b = SparkSession.builder.appName('warm-delta-jars') \
    .config('spark.sql.extensions','io.delta.sql.DeltaSparkSessionExtension') \
    .config('spark.sql.catalog.spark_catalog','org.apache.spark.sql.delta.catalog.DeltaCatalog') \
    .master('local[1]'); \
s = delta.configure_spark_with_delta_pip(b).getOrCreate(); \
print('delta jars ready, spark', s.version); \
s.stop()"

COPY src/ ./src/
COPY api/ ./api/
COPY config/ ./config/
COPY sql/ ./sql/
COPY scripts/ ./scripts/

# Non-root: a container that writes to a bind-mounted lake as root leaves files
# the host user cannot delete.
RUN useradd --create-home --uid 1000 spark \
    && mkdir -p /app/data /app/metadata \
    && chown -R spark:spark /app
USER spark

ENV PYTHONPATH=/app

CMD ["python", "-m", "src.pipelines.spark_pipeline", "--help"]

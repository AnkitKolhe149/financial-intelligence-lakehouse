# ---------------------------------------------------------------------------
# Financial Intelligence Lakehouse
#
#   make install    then    make demo
#
# Every target works on a machine with nothing running: no containers, no
# database, no network. Targets that need an optional service say so and
# degrade rather than fail.
# ---------------------------------------------------------------------------

# Use the virtualenv interpreter when it exists, otherwise whatever python is
# on PATH. Keeps the same Makefile working inside the containers.
VENV      := .venv
ifeq ($(OS),Windows_NT)
PYTHON    := $(if $(wildcard $(VENV)/Scripts/python.exe),$(VENV)/Scripts/python.exe,python)
else
PYTHON    := $(if $(wildcard $(VENV)/bin/python),$(VENV)/bin/python,python3)
endif

SCALE     ?= default
EVENTS    ?= 400

.DEFAULT_GOAL := help
.PHONY: help install install-dev venv generate-data ingest pipeline stream stream-events \
        demo test test-fast coverage lint format typecheck sql api source-api dashboard \
        docker-up docker-down docker-build spark spark-shell load-graph benchmark \
        schemas docs clean clean-all check

# ---------------------------------------------------------------------------
help:
	@echo "Financial Intelligence Lakehouse"
	@echo ""
	@echo "  Setup"
	@echo "    install         Create .venv and install runtime dependencies"
	@echo "    install-dev     Install runtime + development dependencies"
	@echo ""
	@echo "  Run"
	@echo "    demo            Generate data and run every stage end to end"
	@echo "    generate-data   Generate the multi-format source landscape"
	@echo "    pipeline        Run the batch pipeline (Bronze -> Silver -> Gold)"
	@echo "    stream          Run the streaming pipeline over the event inbox"
	@echo "    stream-events   Emit simulated events into the inbox"
	@echo "    sql             Execute the SQL analytics library"
	@echo "    api             Serve the platform API on :8000"
	@echo "    source-api      Serve the simulated source system on :8001"
	@echo "    dashboard       Launch the Streamlit dashboard on :8501"
	@echo ""
	@echo "  Quality"
	@echo "    test            Run the test suite"
	@echo "    test-fast       Run the test suite, skipping slow tests"
	@echo "    coverage        Run tests with a coverage report"
	@echo "    lint            Ruff + Black checks"
	@echo "    format          Apply Black and Ruff fixes"
	@echo "    typecheck       Run mypy"
	@echo "    check           lint + typecheck + test (what CI runs)"
	@echo "    benchmark       Measure performance across dataset sizes"
	@echo ""
	@echo "  Optional services"
	@echo "    docker-up       Start PostgreSQL and Neo4j"
	@echo "    docker-down     Stop all services"
	@echo "    spark           Run the pipeline on PySpark + Delta in Docker"
	@echo "    load-graph      Load the investigation graph into Neo4j"
	@echo ""
	@echo "  Housekeeping"
	@echo "    schemas         Regenerate schemas/*.json from the schema registry"
	@echo "    docs            Regenerate the generated documentation"
	@echo "    clean           Remove the lake, quarantine and models"
	@echo "    clean-all       clean + caches + local metadata"

# --- setup -----------------------------------------------------------------
venv:
	python -m venv $(VENV)

install: venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	@echo "Installed. Next: make demo"

install-dev: venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements-dev.txt

# --- running ---------------------------------------------------------------
generate-data:
	$(PYTHON) -m src.data_generation.generate_all --scale $(SCALE)

ingest pipeline:
	$(PYTHON) -m src.pipelines.batch_pipeline

stream-events:
	$(PYTHON) -m src.streaming.event_generator --events $(EVENTS) --interval 0.1

stream:
	$(PYTHON) -m src.pipelines.streaming_pipeline --idle-exit 10

demo:
	$(PYTHON) scripts/run_demo.py --scale $(SCALE) --events $(EVENTS)

sql:
	$(PYTHON) -m src.analytics.run_sql

api:
	$(PYTHON) -m uvicorn src.api.main:app --reload --port 8000

source-api:
	$(PYTHON) -m uvicorn api.source_system.main:app --reload --port 8001

dashboard:
	$(PYTHON) -m streamlit run dashboard/app.py

# --- quality ---------------------------------------------------------------
test:
	$(PYTHON) -m pytest tests/ -v

test-fast:
	$(PYTHON) -m pytest tests/ -q -m "not slow"

coverage:
	$(PYTHON) -m pytest tests/ --cov=src --cov-report=term-missing --cov-report=html
	@echo "HTML report: htmlcov/index.html"

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m black --check src tests dashboard api scripts

format:
	$(PYTHON) -m ruff check . --fix
	$(PYTHON) -m black src tests dashboard api scripts

typecheck:
	$(PYTHON) -m mypy src --ignore-missing-imports

check: lint typecheck test

benchmark:
	$(PYTHON) scripts/benchmark.py --scales small default --compare-optimisations --memory

# --- optional services -----------------------------------------------------
docker-up:
	docker compose up -d postgres neo4j
	@echo "PostgreSQL :5432   Neo4j browser http://localhost:7474"

docker-down:
	docker compose down

docker-build:
	docker compose build

spark:
	docker compose --profile spark run --rm spark python -m src.pipelines.spark_pipeline

spark-shell:
	docker compose --profile spark run --rm spark python

load-graph:
	$(PYTHON) -c "from src.gold.builder import load_silver; from src.graph.graph_loader import load_graph; from src.core.config import get_settings; s=get_settings(); print(load_graph(load_silver(s), s).summary())"

# --- housekeeping ----------------------------------------------------------
schemas:
	$(PYTHON) scripts/export_schemas.py

docs:
	$(PYTHON) scripts/generate_docs.py

clean:
	$(PYTHON) -c "import shutil,pathlib; [shutil.rmtree(p, ignore_errors=True) for p in ['data/lake','data/quarantine','data/models','data/raw']]; print('lake removed')"

clean-all: clean
	$(PYTHON) -c "import shutil,pathlib; [shutil.rmtree(p, ignore_errors=True) for p in ['metadata/local','.pytest_cache','.ruff_cache','.mypy_cache','htmlcov']]; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]; print('caches removed')"

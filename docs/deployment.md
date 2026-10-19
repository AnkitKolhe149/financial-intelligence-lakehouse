# Deployment and Local Setup

## Requirements

| Component | Requirement | Notes |
| --- | --- | --- |
| Python | 3.11+ | 3.14 works for everything except the Spark path |
| Disk | ~500 MB | Includes the default-scale lake |
| Memory | 2 GB | Peak Python allocation measured at 66 MB |
| Docker | Optional | Only for PostgreSQL, Neo4j and Spark |
| Java | Not required | Only inside the Spark container |

**The full demo runs with none of the optional components.** No containers, no
database, no network.

## Linux and macOS

```bash
git clone <repo> && cd financial-intelligence-lakehouse
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                 # every value has a working default

python scripts/run_demo.py           # generate, ingest, build, score, analyse
streamlit run dashboard/app.py       # http://localhost:8501
```

## Windows (PowerShell)

```powershell
git clone <repo>; cd financial-intelligence-lakehouse
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env

python scripts\run_demo.py
streamlit run dashboard\app.py
```

Windows is a first-class target here — the project was built and tested on it.
Every path goes through `pathlib`, no shell-specific commands appear in the
Python code, and the atomic-write path retries briefly because Windows refuses
to rename a directory while any handle into it is open.

`make` is not standard on Windows; run the underlying commands directly, or use
the Makefile under Git Bash or WSL.

## What the demo does

```bash
python scripts/run_demo.py --scale small        # ~30 s
python scripts/run_demo.py                      # default, ~70 s
python scripts/run_demo.py --scale large        # 500k transactions
python scripts/run_demo.py --external           # also call the live USAspending API
```

1. Clean the previous lake
2. Generate CSV, JSON, XML, Excel, REST payloads and a stream
3. Probe the REST source system (falls back to files if it is not running)
4. Run Bronze → Silver → quality → ML → risk → Gold
5. Load the investigation graph (skipped cleanly if Neo4j is absent)
6. Simulate and process streaming events
7. Execute all 53 analytical queries
8. Print a summary built entirely from measured values

## Individual components

```bash
make generate-data          # or: python -m src.data_generation.generate_all
make pipeline               # python -m src.pipelines.batch_pipeline
make stream-events          # emit events into the inbox
make stream                 # process them
make sql                    # run the SQL library
make api                    # serving API on :8000, docs at /docs
make source-api             # simulated source system on :8001
make dashboard              # Streamlit on :8501
make test lint typecheck    # or: make check
make benchmark              # measured performance across scales
```

## Optional services

```bash
docker compose up -d postgres neo4j
```

Then set in `.env`:

```env
FIL_POSTGRES_DSN=postgresql+psycopg://fil:fil_local_password@localhost:5432/financial_intelligence
```

Metadata moves from SQLite to PostgreSQL with no code change — same schema, same
queries, both through SQLAlchemy Core. Neo4j is picked up automatically; until
it is, the graph queries use the in-process fallback and say so.

Full stack, including the API and dashboard as containers:

```bash
docker compose up -d
# API        http://localhost:8000/docs
# Dashboard  http://localhost:8501
# Neo4j      http://localhost:7474
```

## Running Spark

PySpark needs a JVM and supports Python ≤ 3.12, so it runs in a container rather
than on the host:

```bash
docker compose --profile spark build spark
docker compose --profile spark run --rm spark python -m src.pipelines.spark_pipeline
docker compose --profile spark run --rm spark pytest tests/test_engine_parity.py -v
```

The lake is bind-mounted, so the Spark engine reads exactly the files the pandas
engine wrote and vice versa. That is what makes the parity test meaningful
rather than two separate universes.

## Verification status

Built and tested on Windows 11, Python 3.14.5, pandas 3.0.5.

**Verified by execution on this machine:**

| Component | Evidence |
| --- | --- |
| Generation, ingestion (5 formats), Bronze, Silver, quality, ML, risk, Gold | `scripts/run_demo.py` completes; every stage `ok` |
| Idempotency, schema evolution, quarantine | Asserted in tests and visible in demo output |
| SQL analytics | 53/53 queries succeed |
| Serving API | All endpoints exercised via `TestClient` |
| Dashboard | All 7 pages render (`streamlit.testing.v1.AppTest`) |
| Streaming | 300 events, checkpointing, watermarks, alerts |
| Graph (fallback path) | 66-node neighbourhood returned |
| USAspending | 4 endpoints called live; real awards returned |
| Tests | 317 passed, 8 skipped |
| Lint | `ruff check .` clean |
| Benchmarks | Two scales plus optimisation comparisons |

**Not verified by execution here:**

| Component | Why | Status |
| --- | --- | --- |
| PySpark + Delta pipeline | Docker's engine would not start on this machine — the Desktop process runs but `com.docker.service` is stopped and starting it needs elevation | Code complete; image builds in CI; parity tests written and marked `spark` |
| PostgreSQL metadata backend | Same | Code complete; identical SQLAlchemy path to the SQLite backend, which is exercised by every test |
| Neo4j Cypher path | Same | Code complete; integration tests written and marked `neo4j`; fallback verified |

To verify them, start Docker Desktop (its engine service needs administrator
rights on this machine), then:

```bash
docker compose up -d postgres neo4j
docker compose --profile spark run --rm spark pytest tests/test_engine_parity.py -v
pytest tests/test_graph.py -v         # the neo4j-marked tests will now run
```

## Configuration

Three tiers: code defaults → `config/settings.yaml` → environment variables.

Non-secret tunables live in YAML, reviewable in a pull request. Secrets and
host-specific values live only in the environment. The YAML is parsed into
Pydantic models, so a typo like `contaminaton: 0.01` fails at start-up with a
clear message rather than silently changing model behaviour.

| Variable | Default | Effect |
| --- | --- | --- |
| `FIL_ENGINE` | `pandas` | `pandas` or `spark` |
| `FIL_POSTGRES_DSN` | empty | Empty uses the SQLite fallback |
| `FIL_LOG_FORMAT` | `console` | `json` for log shipping |
| `FIL_GRAPH_ENABLED` | `true` | Disables the graph stage entirely |
| `USASPENDING_ENABLED` | `true` | `false` uses the synthetic federal source (CI sets this) |

## Troubleshooting

**"The lake has not been built yet"** — run `python scripts/run_demo.py`. The
dashboard and API degrade to this message rather than erroring.

**Source API unavailable** — expected unless `make source-api` is running. The
pipeline swaps in file-based fallbacks and logs the reason.

**`metadata_store_unavailable`** — PostgreSQL is configured but not reachable.
The run continues; metadata is not persisted for it. Clear `FIL_POSTGRES_DSN` to
use SQLite.

**PySpark import error on the host** — expected. Spark runs in its container.

**Windows: file in use during clean** — the metadata store holds the SQLite
file open. `scripts/run_demo.py` disposes it before cleaning; if you clean by
hand, close any running dashboard or API first.

-- ---------------------------------------------------------------------------
-- PostgreSQL initialisation.
--
-- Runs once, on first container start, before the application connects.
--
-- The metadata *tables* are created by SQLAlchemy (src/metadata/schema.py) so
-- that the SQLite fallback and PostgreSQL stay identical - defining them twice
-- is how the two backends drift apart. This file only sets up what SQLAlchemy
-- cannot: the schema, extensions and a read-only analyst role.
-- ---------------------------------------------------------------------------

-- Query statistics, for the performance work in docs/performance.md.
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- Trigram indexing, so ILIKE searches over run ids and dataset names in the
-- dashboard do not degrade to sequential scans as history accumulates.
CREATE EXTENSION IF NOT EXISTS pg_trgm;

COMMENT ON DATABASE financial_intelligence IS
  'Operational metadata for the Financial Intelligence Lakehouse: pipeline runs, data quality results, lineage, ingestion ledger and model registry. Analytical data lives in the Parquet/Delta lake, not here.';

-- A read-only role for humans and BI tools. Nobody should be exploring
-- metadata with the account the pipelines write with.
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'fil_readonly') THEN
        CREATE ROLE fil_readonly LOGIN PASSWORD 'fil_readonly_password';
    END IF;
END
$$;

GRANT CONNECT ON DATABASE financial_intelligence TO fil_readonly;
GRANT USAGE ON SCHEMA public TO fil_readonly;

-- Applies to tables SQLAlchemy creates later, which is the point: without
-- default privileges the role would see nothing after the first migration.
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO fil_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO fil_readonly;

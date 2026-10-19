"""Pluggable execution engines.

The platform runs the same Bronze -> Silver -> Gold logic on two engines:

``pandas``
    The default. Runs anywhere Python runs, needs no JVM, and is genuinely
    faster than Spark for the local dataset sizes - Spark's scheduling overhead
    dominates until data no longer fits comfortably in memory.

``spark``
    PySpark + Delta Lake, executed inside the ``spark`` container (PySpark needs
    a JVM and supports Python <= 3.12). This is the path that would run in
    production and the one that maps directly onto Databricks.

Both engines interpret the **same declarative specifications**
(:mod:`src.silver.spec`, :mod:`src.data_quality.suites`, the risk rules in
``config/settings.yaml``), which is what keeps them in agreement. That agreement
is not assumed - ``tests/test_engine_parity.py`` runs both over identical input
and asserts identical output.

Choosing an engine:

    FIL_ENGINE=pandas   python -m src.pipelines.batch_pipeline
    FIL_ENGINE=spark    python -m src.pipelines.spark_pipeline   # inside the container
"""

from src.engines.base import EngineCapabilities, LakehouseEngine, get_engine

__all__ = ["EngineCapabilities", "LakehouseEngine", "get_engine"]

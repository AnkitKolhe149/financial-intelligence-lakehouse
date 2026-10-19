"""The platform's serving layer.

Distinct from :mod:`api.source_system`, which simulates an upstream system the
platform *reads from*. This package is what the platform *offers*: risk
profiles, anomalies, quality scores, lineage and pipeline health, as JSON.

The API reads Gold datasets directly. There is no application database and no
caching layer, because the Gold layer is already the pre-aggregated, query-ready
form - adding a second copy would introduce a staleness problem that does not
currently exist.
"""

from src.api.main import app

__all__ = ["app"]

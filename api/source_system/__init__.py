"""A local stand-in for an upstream operational system that exposes REST.

This is deliberately a *separate* FastAPI application from the platform's own
serving API (:mod:`src.api`). Conflating them would hide the most important
property of the ingestion path: the platform pulls from a system it does not
control, over HTTP, with pagination, filters, latency and failures it must
tolerate.
"""

from api.source_system.main import app

__all__ = ["app"]

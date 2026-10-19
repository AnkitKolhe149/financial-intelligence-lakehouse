"""Resilient HTTP client shared by every API-based source.

Written explicitly rather than pulled from a library because the retry policy is
a design decision worth being able to explain and test:

* **Only transient failures are retried** - connection errors, timeouts, 429 and
  5xx. A 400 or 404 is a bug in our request; retrying it four times just delays
  the error and hammers the source.
* **Exponential backoff with full jitter.** Without jitter, every worker that
  hit the same outage retries in lockstep and re-creates the thundering herd
  that caused it.
* **``Retry-After`` is honoured** when the server sends it. The server knows
  better than our backoff formula.
* **A token-bucket rate limiter** caps our own request rate, because being a
  good client of a public API is part of the job.
* **Every attempt is logged** with the run id, so a slow load can be traced to
  the exact pages that were retried.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any

import requests

from src.core.exceptions import IngestionError
from src.core.logging import get_logger

log = get_logger(__name__)

#: Status codes worth retrying. Everything else is a permanent answer.
RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


@dataclass
class RateLimiter:
    """Token bucket. ``requests_per_second <= 0`` disables limiting."""

    requests_per_second: float
    _allowance: float = field(init=False, default=0.0)
    _last_check: float = field(init=False, default_factory=time.monotonic)

    def __post_init__(self) -> None:
        self._allowance = max(self.requests_per_second, 0.0)

    def acquire(self) -> None:
        if self.requests_per_second <= 0:
            return
        now = time.monotonic()
        elapsed = now - self._last_check
        self._last_check = now
        self._allowance = min(
            self._allowance + elapsed * self.requests_per_second, self.requests_per_second
        )
        if self._allowance < 1.0:
            sleep_for = (1.0 - self._allowance) / self.requests_per_second
            time.sleep(sleep_for)
            self._allowance = 0.0
        else:
            self._allowance -= 1.0


@dataclass
class HttpStats:
    """Per-client counters, surfaced in ingestion metrics."""

    requests: int = 0
    retries: int = 0
    failures: int = 0
    total_seconds: float = 0.0
    bytes_received: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "http_requests": self.requests,
            "http_retries": self.retries,
            "http_failures": self.failures,
            "http_seconds": round(self.total_seconds, 3),
            "http_bytes": self.bytes_received,
        }


class ResilientHttpClient:
    """A small ``requests`` wrapper with retry, backoff and rate limiting."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout: int = 30,
        max_retries: int = 4,
        rate_limit_per_sec: float = 4.0,
        backoff_base: float = 0.5,
        backoff_cap: float = 30.0,
        user_agent: str = "financial-intelligence-lakehouse/1.0",
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_cap = backoff_cap
        self.limiter = RateLimiter(rate_limit_per_sec)
        self.stats = HttpStats()
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": user_agent, "Accept": "application/json"})

    # -- public API ---------------------------------------------------------

    def get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._request("GET", path, params=params)

    def post(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._request("POST", path, json_body=payload)

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> ResilientHttpClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # -- internals ----------------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = path if path.startswith("http") else f"{self.base_url}/{path.lstrip('/')}"
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            self.limiter.acquire()
            started = time.perf_counter()
            try:
                response = self.session.request(
                    method, url, params=params, json=json_body, timeout=self.timeout
                )
                elapsed = time.perf_counter() - started
                self.stats.requests += 1
                self.stats.total_seconds += elapsed
                self.stats.bytes_received += len(response.content or b"")

                if response.status_code in RETRYABLE_STATUS:
                    last_error = IngestionError(
                        "Retryable HTTP status",
                        status=response.status_code,
                        url=url,
                    )
                    if attempt == self.max_retries:
                        break
                    self._sleep_before_retry(attempt, response)
                    continue

                response.raise_for_status()
                return self._decode(response, url)

            except (requests.ConnectionError, requests.Timeout) as exc:
                self.stats.total_seconds += time.perf_counter() - started
                last_error = exc
                if attempt == self.max_retries:
                    break
                self._sleep_before_retry(attempt, None)
            except requests.HTTPError as exc:
                # Non-retryable status: fail immediately rather than burn retries.
                self.stats.failures += 1
                raise IngestionError(
                    "HTTP request failed",
                    url=url,
                    status=exc.response.status_code if exc.response is not None else None,
                    detail=str(exc)[:300],
                ) from exc

        self.stats.failures += 1
        raise IngestionError(
            "HTTP request failed after retries",
            url=url,
            attempts=self.max_retries,
            detail=str(last_error)[:300],
        ) from last_error

    def _decode(self, response: requests.Response, url: str) -> dict[str, Any]:
        try:
            payload = response.json()
        except ValueError as exc:
            self.stats.failures += 1
            raise IngestionError(
                "Response body was not valid JSON", url=url, body=response.text[:200]
            ) from exc
        return payload if isinstance(payload, dict) else {"results": payload}

    def _sleep_before_retry(self, attempt: int, response: requests.Response | None) -> None:
        """Back off, honouring ``Retry-After`` when present."""
        self.stats.retries += 1
        delay: float
        retry_after = response.headers.get("Retry-After") if response is not None else None
        if retry_after and str(retry_after).isdigit():
            delay = float(retry_after)
        else:
            # Exponential with full jitter: sleep ~U(0, base * 2^(attempt-1)).
            ceiling = min(self.backoff_base * (2 ** (attempt - 1)), self.backoff_cap)
            delay = random.uniform(0, ceiling)
        log.warning("http_retry", attempt=attempt, sleep_seconds=round(delay, 2))
        time.sleep(delay)

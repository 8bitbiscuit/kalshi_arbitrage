"""Shared async HTTP plumbing: rate limiting, retries, error typing."""

from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Any, Mapping

import httpx

log = logging.getLogger(__name__)

RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}


class ApiError(RuntimeError):
    """Non-retryable API failure, or a retryable one that ran out of attempts."""

    def __init__(self, message: str, status: int | None = None, body: str | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.body = body


class RateLimiter:
    """Simple token-per-interval limiter shared across a client's requests.

    Both venues publish per-key rate limits (Kalshi's basic tier is ~10 reads/s,
    Polymarket's gateway is looser). Staying under them politely is cheaper than
    handling the 429s.
    """

    def __init__(self, requests_per_second: float) -> None:
        self.min_interval = 1.0 / requests_per_second if requests_per_second > 0 else 0.0
        self._next_slot = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        if self.min_interval <= 0:
            return
        async with self._lock:
            now = time.monotonic()
            wait = self._next_slot - now
            if wait > 0:
                await asyncio.sleep(wait)
                now = time.monotonic()
            self._next_slot = now + self.min_interval


class HttpClient:
    """Thin wrapper over ``httpx.AsyncClient`` with retry and rate limiting."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 25.0,
        retries: int = 4,
        requests_per_second: float = 8.0,
        user_agent: str = "arbmon/0.1",
        headers: Mapping[str, str] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.retries = retries
        self.limiter = RateLimiter(requests_per_second)
        default_headers = {"Accept": "application/json", "User-Agent": user_agent}
        if headers:
            default_headers.update(headers)
        # trust_env keeps HTTPS_PROXY / REQUESTS_CA_BUNDLE working for anyone
        # running this behind a corporate proxy.
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout,
            headers=default_headers,
            follow_redirects=True,
            trust_env=True,
        )

    async def __aenter__(self) -> "HttpClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    def _auth_headers(self, method: str, path: str) -> dict[str, str]:
        """Hook for subclasses that sign requests. Default: no extra headers."""
        return {}

    async def request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        json_body: Any = None,
    ) -> Any:
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            await self.limiter.acquire()
            try:
                response = await self._client.request(
                    method,
                    path,
                    params=_clean_params(params),
                    json=json_body,
                    headers=self._auth_headers(method, path),
                )
            except httpx.HTTPError as exc:
                last_error = exc
                if attempt >= self.retries:
                    break
                await self._backoff(attempt, reason=str(exc))
                continue

            if response.status_code in RETRYABLE_STATUS and attempt < self.retries:
                retry_after = _retry_after_seconds(response)
                await self._backoff(attempt, reason=f"HTTP {response.status_code}", floor=retry_after)
                continue

            if response.status_code >= 400:
                raise ApiError(
                    f"{method} {self.base_url}{path} -> HTTP {response.status_code}",
                    status=response.status_code,
                    body=response.text[:1000],
                )
            if not response.content:
                return None
            try:
                return response.json()
            except ValueError as exc:
                raise ApiError(
                    f"{method} {path} returned non-JSON body", body=response.text[:500]
                ) from exc

        raise ApiError(f"{method} {self.base_url}{path} failed after {self.retries + 1} attempts: {last_error}")

    async def _backoff(self, attempt: int, reason: str, floor: float | None = None) -> None:
        delay = min(2.0**attempt, 30.0) + random.uniform(0, 0.4)
        if floor is not None:
            delay = max(delay, floor)
        log.debug("retrying in %.1fs (attempt %d): %s", delay, attempt + 1, reason)
        await asyncio.sleep(delay)

    async def get(self, path: str, **kwargs: Any) -> Any:
        return await self.request("GET", path, **kwargs)

    async def post(self, path: str, **kwargs: Any) -> Any:
        return await self.request("POST", path, **kwargs)


def _clean_params(params: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Drop None values so they don't turn into literal 'None' query strings."""
    if not params:
        return None
    return {k: v for k, v in params.items() if v is not None}


def _retry_after_seconds(response: httpx.Response) -> float | None:
    raw = response.headers.get("Retry-After")
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None

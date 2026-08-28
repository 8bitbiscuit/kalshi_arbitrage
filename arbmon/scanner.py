"""Scan orchestration: fetch both venues, match, detect, persist, repeat."""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
import time
from dataclasses import dataclass, field

from .arbitrage import find_opportunities
from .clients.kalshi import KalshiClient
from .clients.polymarket import PolymarketClient
from .config import Config
from .matching import MatchedPair, find_pairs
from .models import NormalizedMarket, Opportunity
from .report import format_opportunities, format_pairs, scan_header
from .storage import Storage, append_jsonl

log = logging.getLogger(__name__)


@dataclass
class ScanResult:
    started_at: dt.datetime
    elapsed: float
    kalshi_markets: list[NormalizedMarket] = field(default_factory=list)
    polymarket_markets: list[NormalizedMarket] = field(default_factory=list)
    pairs: list[MatchedPair] = field(default_factory=list)
    opportunities: list[Opportunity] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def all_markets(self) -> list[NormalizedMarket]:
        return self.kalshi_markets + self.polymarket_markets


class Scanner:
    """Owns the clients and the storage handle across repeated scans."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.kalshi = KalshiClient(config) if config.kalshi.enabled else None
        self.polymarket = PolymarketClient(config) if config.polymarket.enabled else None
        self.storage = Storage(config.storage.sqlite_path) if config.storage.enabled else None
        self._scan_number = 0

    async def aclose(self) -> None:
        if self.kalshi:
            await self.kalshi.aclose()
        if self.polymarket:
            await self.polymarket.aclose()
        if self.storage:
            self.storage.close()

    async def __aenter__(self) -> "Scanner":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def collect_markets(
        self,
    ) -> tuple[list[NormalizedMarket], list[NormalizedMarket], list[str]]:
        """Fetch both venues concurrently; one venue failing is not fatal.

        Returns ``(kalshi, polymarket, errors)``. Exposed separately from
        :meth:`scan_once` so the ``markets`` and ``pairs`` CLI commands can
        inspect what was collected without pricing or persisting anything.
        """
        tasks = []
        tasks.append(self.kalshi.collect() if self.kalshi else _empty())
        tasks.append(self.polymarket.collect() if self.polymarket else _empty())
        kalshi_result, poly_result = await asyncio.gather(*tasks, return_exceptions=True)

        errors: list[str] = []
        kalshi_markets: list[NormalizedMarket] = []
        poly_markets: list[NormalizedMarket] = []

        if isinstance(kalshi_result, BaseException):
            errors.append(f"kalshi: {kalshi_result}")
            log.error("kalshi collection failed: %s", kalshi_result)
        else:
            kalshi_markets = kalshi_result

        if isinstance(poly_result, BaseException):
            errors.append(f"polymarket: {poly_result}")
            log.error("polymarket collection failed: %s", poly_result)
        else:
            poly_markets = poly_result

        return kalshi_markets, poly_markets, errors

    async def scan_once(self) -> ScanResult:
        self._scan_number += 1
        started_at = dt.datetime.now(dt.timezone.utc)
        clock = time.monotonic()
        scan_id = self.storage.start_scan(started_at) if self.storage else 0

        kalshi_markets, poly_markets, errors = await self.collect_markets()

        pairs = find_pairs(kalshi_markets, poly_markets, self.config.matching)
        opportunities = find_opportunities(pairs, kalshi_markets + poly_markets, self.config)

        result = ScanResult(
            started_at=started_at,
            elapsed=time.monotonic() - clock,
            kalshi_markets=kalshi_markets,
            polymarket_markets=poly_markets,
            pairs=pairs,
            opportunities=opportunities,
            errors=errors,
        )

        if self.storage:
            if self.config.storage.record_snapshots:
                self.storage.record_snapshots(scan_id, result.all_markets)
            self.storage.record_opportunities(scan_id, opportunities)
            self.storage.finish_scan(
                scan_id,
                finished_at=dt.datetime.now(dt.timezone.utc),
                kalshi_count=len(kalshi_markets),
                poly_count=len(poly_markets),
                pair_count=len(pairs),
                opp_count=len(opportunities),
                error="; ".join(errors) or None,
            )

        if self.config.output.jsonl_path and opportunities:
            append_jsonl(self.config.output.jsonl_path, opportunities)

        return result

    def print_result(self, result: ScanResult) -> None:
        print(
            scan_header(
                self._scan_number,
                len(result.kalshi_markets),
                len(result.polymarket_markets),
                len(result.pairs),
                result.elapsed,
            )
        )
        for error in result.errors:
            print(f"     ! {error}")
        if self.config.output.verbose_pairs:
            print(format_pairs(result.pairs, self.config.output.max_rows))
        print(format_opportunities(result.opportunities, self.config.output.max_rows))
        print()

    async def run(self) -> None:
        """Scan on a fixed interval until interrupted or ``max_scans`` is hit."""
        max_scans = self.config.max_scans
        while max_scans == 0 or self._scan_number < max_scans:
            try:
                result = await self.scan_once()
            except Exception as exc:  # noqa: BLE001 - a bad scan must not kill the loop
                log.exception("scan failed: %s", exc)
                await asyncio.sleep(self.config.scan_interval_seconds)
                continue

            self.print_result(result)
            if max_scans and self._scan_number >= max_scans:
                break
            await asyncio.sleep(self.config.scan_interval_seconds)


async def _empty() -> list[NormalizedMarket]:
    return []

"""Polymarket client: Gamma for market metadata, CLOB for order books.

Gamma (``gamma-api.polymarket.com``) is the indexed metadata service - it knows
the question text, resolution date, token ids and a cached top of book. The CLOB
(``clob.polymarket.com``) serves the live order book. Both are public and need
no authentication for reads.

A Polymarket binary market is two separate ERC-1155 tokens (the YES token and
the NO token), each with its own book, so buying NO is a real purchase of the NO
token rather than a synthetic short of YES.
"""

from __future__ import annotations

import json
import logging
import math
from decimal import Decimal
from typing import Any, Iterable

from ..config import Config
from ..models import NormalizedMarket, PriceLevel, Venue
from ..normalize import (
    detect_asset,
    parse_datetime,
    parse_polymarket_spec,
    to_decimal,
)
from .base import ApiError, HttpClient

log = logging.getLogger(__name__)

POLYMARKET_WEB_URL = "https://polymarket.com/event"

_YES_LABELS = {"yes", "up", "above", "over", "higher"}
_NO_LABELS = {"no", "down", "below", "under", "lower"}


class PolymarketClient:
    """Wraps the two Polymarket services behind one interface."""

    def __init__(self, config: Config) -> None:
        pc = config.polymarket
        self.config = config
        self.pc = pc
        self.gamma = HttpClient(
            pc.gamma_url,
            timeout=config.http_timeout_seconds,
            retries=config.http_retries,
            requests_per_second=pc.requests_per_second,
            user_agent=config.user_agent,
        )
        self.clob = HttpClient(
            pc.clob_url,
            timeout=config.http_timeout_seconds,
            retries=config.http_retries,
            requests_per_second=pc.requests_per_second,
            user_agent=config.user_agent,
        )

    async def __aenter__(self) -> "PolymarketClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self.gamma.aclose()
        await self.clob.aclose()

    # -- raw endpoints ------------------------------------------------------

    async def fetch_markets(self) -> list[dict[str, Any]]:
        """Page through Gamma's open markets, most liquid first."""
        out: list[dict[str, Any]] = []
        for page in range(self.pc.max_pages):
            payload = await self.gamma.get(
                "/markets",
                params={
                    "limit": self.pc.page_limit,
                    "offset": page * self.pc.page_limit,
                    "active": "true",
                    "closed": "false",
                    "archived": "false",
                    "order": "volumeNum",
                    "ascending": "false",
                    "liquidity_num_min": _as_param(self.pc.min_liquidity),
                    "volume_num_min": _as_param(self.pc.min_volume),
                },
            )
            page_markets = payload if isinstance(payload, list) else (payload or {}).get("data") or []
            if not page_markets:
                break
            out.extend(page_markets)
            if len(page_markets) < self.pc.page_limit:
                break
        return out

    async def fetch_books(self, token_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Batch-fetch CLOB books. Falls back to single fetches on failure."""
        books: dict[str, dict[str, Any]] = {}
        for chunk in _chunks(token_ids, self.pc.orderbook_batch_size):
            try:
                payload = await self.clob.post(
                    "/books", json_body=[{"token_id": tid} for tid in chunk]
                )
            except ApiError as exc:
                log.debug("batch book fetch failed (%s); falling back to singles", exc)
                for token_id in chunk:
                    book = await self._fetch_book(token_id)
                    if book:
                        books[token_id] = book
                continue
            for book in payload or []:
                asset_id = str(book.get("asset_id") or "")
                if asset_id:
                    books[asset_id] = book
        return books

    async def _fetch_book(self, token_id: str) -> dict[str, Any] | None:
        try:
            return await self.clob.get("/book", params={"token_id": token_id})
        except ApiError as exc:
            log.debug("book fetch failed for %s: %s", token_id, exc)
            return None

    # -- normalisation ------------------------------------------------------

    def normalize(
        self,
        market: dict[str, Any],
        books: dict[str, dict[str, Any]] | None = None,
    ) -> NormalizedMarket | None:
        tokens = _json_list(market.get("clobTokenIds"))
        outcomes = [str(o).strip().lower() for o in _json_list(market.get("outcomes"))]
        if len(tokens) != 2 or len(outcomes) != 2:
            return None

        yes_index = _yes_index(outcomes)
        if yes_index is None:
            return None
        no_index = 1 - yes_index
        yes_token, no_token = str(tokens[yes_index]), str(tokens[no_index])

        event_title = _event_title(market)
        prices = [to_decimal(p) for p in _json_list(market.get("outcomePrices"))]
        yes_mid = prices[yes_index] if len(prices) == 2 else None
        no_mid = prices[no_index] if len(prices) == 2 else None

        yes_asks: list[PriceLevel] = []
        no_asks: list[PriceLevel] = []
        if books:
            yes_asks = ask_ladder(books.get(yes_token))
            no_asks = ask_ladder(books.get(no_token))

        spec = parse_polymarket_spec(
            market, allowed_assets=self.config.matching.assets, event_title=event_title
        )

        # Gamma's bestAsk/bestBid are quoted against the *first* outcome token,
        # so they only describe the YES leg when YES is outcome 0.
        best_ask = to_decimal(market.get("bestAsk"))
        best_bid = to_decimal(market.get("bestBid"))
        if yes_index == 1:
            best_ask, best_bid = _complement(best_bid), _complement(best_ask)

        yes_ask = yes_asks[0].price if yes_asks else best_ask
        no_ask = no_asks[0].price if no_asks else _complement(best_bid)

        slug = str(market.get("slug") or "")
        event_slug = _event_slug(market) or slug

        return NormalizedMarket(
            venue=Venue.POLYMARKET,
            venue_id=yes_token,
            title=str(market.get("question") or slug),
            spec=spec,
            yes_asks=yes_asks,
            no_asks=no_asks,
            yes_bid=best_bid,
            yes_ask=yes_ask,
            no_bid=_complement(best_ask),
            no_ask=no_ask,
            close_time=parse_datetime(market.get("endDate")),
            volume=to_decimal(market.get("volumeNum")) or to_decimal(market.get("volume")),
            liquidity=to_decimal(market.get("liquidityNum")) or to_decimal(market.get("liquidity")),
            open_interest=None,
            url=f"{POLYMARKET_WEB_URL}/{event_slug}" if event_slug else None,
            taker_fee_bps=to_decimal(market.get("takerBaseFee")),
            maker_fee_bps=to_decimal(market.get("makerBaseFee")),
            raw={
                **market,
                "_yes_token_id": yes_token,
                "_no_token_id": no_token,
                "_yes_mid": str(yes_mid) if yes_mid is not None else None,
                "_no_mid": str(no_mid) if no_mid is not None else None,
            },
        )

    # -- high level ---------------------------------------------------------

    async def collect(self) -> list[NormalizedMarket]:
        raw = await self.fetch_markets()
        assets = self.config.matching.assets
        # Cheap keyword pre-filter before the (much more expensive) spec parse.
        candidates = [
            m
            for m in raw
            if detect_asset(
                str(m.get("question") or ""),
                _event_title(m),
                str(m.get("slug") or ""),
                allowed=assets,
            )
        ]

        markets = [m for m in (self.normalize(m) for m in candidates) if m is not None]
        tradeable = [m for m in markets if m.spec is not None]
        log.info(
            "polymarket: %d markets fetched, %d crypto candidates, %d contracts parsed",
            len(raw),
            len(candidates),
            len(tradeable),
        )

        if self.pc.fetch_orderbook and tradeable:
            token_ids: list[str] = []
            for market in tradeable:
                token_ids.append(market.raw["_yes_token_id"])
                token_ids.append(market.raw["_no_token_id"])
            books = await self.fetch_books(token_ids)
            for market in tradeable:
                market.yes_asks = ask_ladder(books.get(market.raw["_yes_token_id"]))
                market.no_asks = ask_ladder(books.get(market.raw["_no_token_id"]))
                if market.yes_asks:
                    market.yes_ask = market.yes_asks[0].price
                if market.no_asks:
                    market.no_ask = market.no_asks[0].price
        return tradeable


def ask_ladder(book: dict[str, Any] | None) -> list[PriceLevel]:
    """Extract the buy-side ladder (cheapest ask first) from a CLOB book.

    Polymarket returns asks worst-price-first, so we always re-sort rather than
    trusting the order. Sizes are shares and may be fractional; they are floored
    to whole contracts because the paired Kalshi leg can only trade in integers.
    """
    if not book:
        return []
    levels: list[PriceLevel] = []
    for entry in book.get("asks") or []:
        px = to_decimal(entry.get("price") if isinstance(entry, dict) else None)
        raw_size = to_decimal(entry.get("size") if isinstance(entry, dict) else None)
        if px is None or raw_size is None:
            continue
        size = int(math.floor(raw_size))
        if size <= 0 or not (Decimal(0) < px < Decimal(1)):
            continue
        levels.append(PriceLevel(px, size))
    levels.sort(key=lambda level: level.price)
    return levels


def _yes_index(outcomes: list[str]) -> int | None:
    """Which of the two outcome tokens is the YES leg?"""
    for index, label in enumerate(outcomes):
        if label in _YES_LABELS:
            return index
    for index, label in enumerate(outcomes):
        if label in _NO_LABELS:
            return 1 - index
    return None


def _json_list(value: Any) -> list[Any]:
    """Gamma encodes list fields as JSON strings, but not always."""
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def _complement(value: Decimal | None) -> Decimal | None:
    return None if value is None else Decimal(1) - value


def _event_title(market: dict[str, Any]) -> str | None:
    events = market.get("events")
    if isinstance(events, list) and events and isinstance(events[0], dict):
        return str(events[0].get("title") or "") or None
    return None


def _event_slug(market: dict[str, Any]) -> str | None:
    events = market.get("events")
    if isinstance(events, list) and events and isinstance(events[0], dict):
        return str(events[0].get("slug") or "") or None
    return None


def _as_param(value: Decimal | None) -> str | None:
    if value is None or value <= 0:
        return None
    return str(value)


def _chunks(items: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), max(1, size)):
        yield items[start : start + size]

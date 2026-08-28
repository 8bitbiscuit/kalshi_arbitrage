"""Kalshi trade-api v2 client (read-only).

The ``/markets`` endpoint is public and already carries top-of-book quotes, so
the scanner works with no credentials at all. Supplying an API key unlocks
``/markets/{ticker}/orderbook``, which is what makes depth-aware sizing
possible - without it we can only see the best bid/ask and have to assume a
size.
"""

from __future__ import annotations

import base64
import logging
import time
from decimal import Decimal
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlsplit

from ..config import Config
from ..models import NormalizedMarket, PriceLevel, Venue, cents_to_price
from ..normalize import parse_datetime, parse_kalshi_spec, to_decimal
from .base import ApiError, HttpClient

log = logging.getLogger(__name__)

KALSHI_WEB_URL = "https://kalshi.com/markets"

# Kalshi quotes in whole cents. A 0 bid or a 100 ask means "nobody is there".
NO_LIQUIDITY_ASK = 100
NO_LIQUIDITY_BID = 0


class KalshiClient(HttpClient):
    def __init__(self, config: Config) -> None:
        kc = config.kalshi
        super().__init__(
            kc.base_url,
            timeout=config.http_timeout_seconds,
            retries=config.http_retries,
            requests_per_second=kc.requests_per_second,
            user_agent=config.user_agent,
        )
        self.config = config
        self.kc = kc
        self._base_path = urlsplit(kc.base_url).path.rstrip("/")
        self._private_key = self._load_private_key()
        self._orderbook_forbidden = False

    # -- auth ---------------------------------------------------------------

    def _load_private_key(self) -> Any:
        if not self.kc.has_credentials():
            return None
        try:
            from cryptography.hazmat.primitives import serialization
        except ImportError:  # pragma: no cover - cryptography is a hard dep
            log.warning("cryptography not installed; Kalshi order books disabled")
            return None
        path = Path(str(self.kc.resolved_private_key_path())).expanduser()
        password = self.kc.resolved_private_key_password()
        try:
            return serialization.load_pem_private_key(
                path.read_bytes(),
                password=password.encode() if password else None,
            )
        except Exception as exc:  # noqa: BLE001 - surface any key problem clearly
            log.warning("could not load Kalshi private key at %s: %s", path, exc)
            return None

    @property
    def authenticated(self) -> bool:
        return self._private_key is not None

    def _auth_headers(self, method: str, path: str) -> dict[str, str]:
        """Sign with RSA-PSS over ``timestamp_ms + METHOD + full_path``.

        The query string is deliberately excluded from the signed message, and
        the path must include the ``/trade-api/v2`` prefix that lives in
        ``base_url``.
        """
        if self._private_key is None:
            return {}
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding

        timestamp_ms = str(int(time.time() * 1000))
        full_path = f"{self._base_path}{path.split('?', 1)[0]}"
        message = f"{timestamp_ms}{method.upper()}{full_path}".encode()
        signature = self._private_key.sign(
            message,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=hashes.SHA256.digest_size),
            hashes.SHA256(),
        )
        return {
            "KALSHI-ACCESS-KEY": str(self.kc.resolved_api_key_id()),
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode(),
            "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
        }

    # -- raw endpoints ------------------------------------------------------

    async def fetch_markets(
        self,
        status: str = "open",
        series_ticker: str | None = None,
    ) -> list[dict[str, Any]]:
        """Page through ``/markets``, honouring the cursor until exhausted."""
        markets: list[dict[str, Any]] = []
        cursor: str | None = None
        for _ in range(self.kc.max_pages):
            payload = await self.get(
                "/markets",
                params={
                    "limit": self.kc.page_limit,
                    "status": status,
                    "cursor": cursor,
                    "series_ticker": series_ticker,
                },
            )
            page = (payload or {}).get("markets") or []
            markets.extend(page)
            cursor = (payload or {}).get("cursor") or None
            # Kalshi echoes the cursor back on the final page; an empty page or
            # a repeated cursor both mean we are done.
            if not cursor or not page:
                break
        return markets

    async def fetch_orderbook(self, ticker: str) -> dict[str, Any] | None:
        """Fetch resting depth for one market. Requires credentials."""
        if not self.authenticated or self._orderbook_forbidden:
            return None
        try:
            payload = await self.get(
                f"/markets/{ticker}/orderbook", params={"depth": self.kc.orderbook_depth}
            )
        except ApiError as exc:
            if exc.status in (401, 403):
                # One rejection is enough - stop hammering the endpoint.
                log.warning("Kalshi order book access denied (HTTP %s); using top-of-book", exc.status)
                self._orderbook_forbidden = True
                return None
            log.debug("order book fetch failed for %s: %s", ticker, exc)
            return None
        return (payload or {}).get("orderbook")

    # -- normalisation ------------------------------------------------------

    def normalize(self, market: dict[str, Any], orderbook: dict[str, Any] | None = None) -> NormalizedMarket:
        ticker = str(market.get("ticker") or "")
        yes_bid = _quote(market.get("yes_bid"), is_ask=False)
        yes_ask = _quote(market.get("yes_ask"), is_ask=True)
        no_bid = _quote(market.get("no_bid"), is_ask=False)
        no_ask = _quote(market.get("no_ask"), is_ask=True)

        yes_asks: list[PriceLevel] = []
        no_asks: list[PriceLevel] = []
        if orderbook:
            yes_asks, no_asks = derive_ask_ladders(orderbook)

        return NormalizedMarket(
            venue=Venue.KALSHI,
            venue_id=ticker,
            title=str(market.get("title") or ticker),
            spec=parse_kalshi_spec(market, allowed_assets=self.config.matching.assets),
            yes_asks=yes_asks,
            no_asks=no_asks,
            yes_bid=yes_bid,
            yes_ask=yes_ask,
            no_bid=no_bid,
            no_ask=no_ask,
            close_time=parse_datetime(market.get("close_time")),
            volume=to_decimal(market.get("volume")),
            liquidity=_liquidity_dollars(market.get("liquidity")),
            open_interest=_as_int(market.get("open_interest")),
            url=f"{KALSHI_WEB_URL}/{ticker}",
            raw=market,
        )

    # -- high level ---------------------------------------------------------

    async def collect(self) -> list[NormalizedMarket]:
        """Fetch every crypto market we can parse, with depth where available."""
        raw: list[dict[str, Any]] = []
        seen: set[str] = set()
        series = self.kc.series_prefixes or [None]  # type: ignore[list-item]
        for status in self.kc.statuses:
            for series_ticker in series:
                for market in await self.fetch_markets(status=status, series_ticker=series_ticker):
                    ticker = str(market.get("ticker") or "")
                    if ticker and ticker not in seen:
                        seen.add(ticker)
                        raw.append(market)

        normalized = [self.normalize(m) for m in raw]
        # Only crypto markets we could parse are worth spending order book
        # requests on.
        tradeable = [m for m in normalized if m.spec is not None and _is_quotable(m)]
        log.info("kalshi: %d markets fetched, %d crypto contracts parsed", len(raw), len(tradeable))

        if self.kc.fetch_orderbook and self.authenticated:
            for market in tradeable:
                book = await self.fetch_orderbook(market.venue_id)
                if book:
                    market.yes_asks, market.no_asks = derive_ask_ladders(book)
                if self._orderbook_forbidden:
                    break
        return tradeable


def derive_ask_ladders(orderbook: dict[str, Any]) -> tuple[list[PriceLevel], list[PriceLevel]]:
    """Convert Kalshi's two bid ladders into the ask ladders we can buy from.

    Kalshi's order book is expressed as two lists of *resting bids*, one on each
    side, each entry ``[price_in_cents, contracts]``. There is no explicit ask
    ladder, because a resting NO bid at 40c *is* an offer to sell YES at 60c:
    the two traders together post the $1 that one of them will collect.

    So the price at which we can buy YES is ``100 - (NO bid price)``, in units
    of that NO bid's size, and vice versa. Getting this backwards is the single
    easiest way to invent an arbitrage that is not there, which is why it lives
    in its own function with its own test.
    """
    yes_bids = _levels(orderbook.get("yes"))
    no_bids = _levels(orderbook.get("no"))

    # Best (cheapest) ask comes from the highest bid on the opposite side.
    yes_asks = [PriceLevel(cents_to_price(100 - cents), size) for cents, size in sorted(no_bids, reverse=True)]
    no_asks = [PriceLevel(cents_to_price(100 - cents), size) for cents, size in sorted(yes_bids, reverse=True)]
    return yes_asks, no_asks


def _levels(raw: Any) -> list[tuple[int, int]]:
    if not isinstance(raw, Sequence):
        return []
    out: list[tuple[int, int]] = []
    for entry in raw:
        if not isinstance(entry, Sequence) or len(entry) < 2:
            continue
        try:
            cents, size = int(entry[0]), int(entry[1])
        except (TypeError, ValueError):
            continue
        if 0 < cents < 100 and size > 0:
            out.append((cents, size))
    return out


def _quote(value: Any, is_ask: bool) -> Decimal | None:
    """Convert a cent quote to dollars, mapping the empty-book sentinels to None."""
    number = _as_int(value)
    if number is None:
        return None
    if is_ask and number >= NO_LIQUIDITY_ASK:
        return None
    if not is_ask and number <= NO_LIQUIDITY_BID:
        return None
    return cents_to_price(number)


def _as_int(value: Any) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _liquidity_dollars(value: Any) -> Decimal | None:
    """Kalshi reports market liquidity in cents."""
    number = to_decimal(value)
    return None if number is None else number / Decimal(100)


def _is_quotable(market: NormalizedMarket) -> bool:
    """Drop markets with no two-sided quote - nothing to arbitrage against."""
    return market.yes_ask is not None or market.no_ask is not None


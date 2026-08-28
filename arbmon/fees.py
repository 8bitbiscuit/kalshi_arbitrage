"""Venue fee schedules.

An "arbitrage" that ignores fees is almost always not one, so this module is
deliberately explicit and unit-tested. Every function returns a fee in dollars
as a ``Decimal``.

Kalshi
------
Trading fee, charged on execution (there is no settlement fee)::

    fee = roundup_to_cent(multiplier * C * P * (1 - P))

``C`` is the contract count, ``P`` the execution price in dollars, and the
multiplier is 0.07 on general markets. A handful of series (the S&P 500 and
Nasdaq range markets) use 0.035, so the multiplier is overridable per series
prefix. The ``P * (1 - P)`` shape means the fee peaks at 1.75c/contract around
50c and falls away towards the wings - which matters here, because arbitrage
pairs are usually one cheap leg and one expensive leg.

Kalshi additionally charges a maker fee of ``roundup_to_cent(0.0025 * C)`` on
selected series, but only on resting orders. Arbitrage execution crosses the
spread, so the taker schedule is the default and the maker fee is opt-in.

Polymarket
----------
The CLOB charged no trading fee for most of its history, and many markets are
still fee-free. Where a fee applies it is proportional to the *cheaper* side of
the book::

    fee = base_rate * min(P, 1 - P) * C

with ``base_rate`` published per market in basis points. We read the venue's
reported rate when it is available and fall back to a configured default.
Polymarket relays transactions, so there is no per-trade gas cost, but a fixed
per-trade cost knob is provided for anyone who wants to charge deposit and
withdrawal costs against each trade.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import ROUND_CEILING, ROUND_HALF_UP, Decimal

from .models import CENT, PRICE_QUANTUM, Venue

DEFAULT_KALSHI_MULTIPLIER = Decimal("0.07")
DEFAULT_KALSHI_MAKER_FEE_PER_CONTRACT = Decimal("0.0025")

# Series that use the reduced 0.035 trading multiplier. Matched as ticker
# prefixes against the Kalshi market ticker.
REDUCED_MULTIPLIER_SERIES: tuple[str, ...] = ("KXINX", "KXNASDAQ100", "INX", "NASDAQ100")
REDUCED_MULTIPLIER = Decimal("0.035")


def roundup_cent(amount: Decimal) -> Decimal:
    """Round a dollar amount up to the next whole cent, as Kalshi does."""
    return amount.quantize(CENT, rounding=ROUND_CEILING)


def kalshi_multiplier_for(ticker: str, override: Decimal | None = None) -> Decimal:
    if override is not None:
        return override
    upper = (ticker or "").upper()
    for prefix in REDUCED_MULTIPLIER_SERIES:
        if upper.startswith(prefix):
            return REDUCED_MULTIPLIER
    return DEFAULT_KALSHI_MULTIPLIER


@dataclass(frozen=True)
class FeeQuote:
    """Result of a fee calculation, kept alongside its own explanation."""

    amount: Decimal
    detail: str

    def __add__(self, other: "FeeQuote") -> "FeeQuote":
        return FeeQuote(self.amount + other.amount, f"{self.detail} + {other.detail}")


class FeeModel:
    """Base interface: what does it cost to buy ``qty`` contracts at ``px``?"""

    venue: Venue

    def fee(self, qty: int, px: Decimal) -> FeeQuote:  # pragma: no cover - interface
        raise NotImplementedError

    def amount(self, qty: int, px: Decimal) -> Decimal:
        return self.fee(qty, px).amount


@dataclass
class KalshiFeeModel(FeeModel):
    """Kalshi's ``roundup(m * C * P * (1-P))`` trading fee."""

    venue: Venue = field(default=Venue.KALSHI, init=False)
    multiplier: Decimal = DEFAULT_KALSHI_MULTIPLIER
    maker: bool = False
    maker_fee_per_contract: Decimal = DEFAULT_KALSHI_MAKER_FEE_PER_CONTRACT
    fixed_cost_per_trade: Decimal = Decimal("0")

    @classmethod
    def for_ticker(
        cls,
        ticker: str,
        multiplier_override: Decimal | None = None,
        **kwargs: object,
    ) -> "KalshiFeeModel":
        return cls(multiplier=kalshi_multiplier_for(ticker, multiplier_override), **kwargs)  # type: ignore[arg-type]

    def fee(self, qty: int, px: Decimal) -> FeeQuote:
        if qty <= 0:
            return FeeQuote(Decimal("0"), "kalshi: no fill")
        px = Decimal(px)
        raw = self.multiplier * Decimal(qty) * px * (Decimal(1) - px)
        total = roundup_cent(raw)
        detail = f"kalshi taker ceil({self.multiplier}x{qty}x{px:.4f}x{1 - px:.4f})=${total}"
        if self.maker:
            maker_fee = roundup_cent(self.maker_fee_per_contract * Decimal(qty))
            total += maker_fee
            detail += f" + maker ${maker_fee}"
        if self.fixed_cost_per_trade:
            total += self.fixed_cost_per_trade
            detail += f" + fixed ${self.fixed_cost_per_trade}"
        return FeeQuote(total.quantize(PRICE_QUANTUM), detail)


@dataclass
class PolymarketFeeModel(FeeModel):
    """Polymarket's ``rate * min(P, 1-P) * C`` fee. Rate is often zero."""

    venue: Venue = field(default=Venue.POLYMARKET, init=False)
    base_fee_bps: Decimal = Decimal("0")
    fixed_cost_per_trade: Decimal = Decimal("0")

    @classmethod
    def for_market(
        cls,
        reported_bps: Decimal | None,
        default_bps: Decimal = Decimal("0"),
        **kwargs: object,
    ) -> "PolymarketFeeModel":
        """Trust the venue's own number when it gives us one."""
        bps = reported_bps if reported_bps is not None else default_bps
        return cls(base_fee_bps=Decimal(bps), **kwargs)  # type: ignore[arg-type]

    def fee(self, qty: int, px: Decimal) -> FeeQuote:
        if qty <= 0:
            return FeeQuote(Decimal("0"), "polymarket: no fill")
        px = Decimal(px)
        rate = self.base_fee_bps / Decimal(10_000)
        cheaper = min(px, Decimal(1) - px)
        total = (rate * cheaper * Decimal(qty)).quantize(CENT, rounding=ROUND_HALF_UP)
        detail = f"polymarket {self.base_fee_bps}bps x min({px:.4f},{1 - px:.4f}) x {qty}=${total}"
        if self.fixed_cost_per_trade:
            total += self.fixed_cost_per_trade
            detail += f" + fixed ${self.fixed_cost_per_trade}"
        return FeeQuote(total.quantize(PRICE_QUANTUM), detail)


class ZeroFeeModel(FeeModel):
    """Escape hatch for 'what would this look like with no fees at all'."""

    def __init__(self, venue: Venue = Venue.KALSHI) -> None:
        self.venue = venue

    def fee(self, qty: int, px: Decimal) -> FeeQuote:
        return FeeQuote(Decimal("0"), "zero-fee model")

"""Normalised data model shared by both venues.

Every price in this package is a probability price expressed in dollars per
contract as a ``Decimal`` in ``[0, 1]``. A contract pays exactly $1.00 if it
resolves YES and $0.00 otherwise, on both venues, so a Kalshi price of 63c and
a Polymarket price of 0.63 are directly comparable once normalised here.

``Decimal`` is used rather than ``float`` throughout: fee schedules round to the
nearest cent and the whole point of the scanner is to decide whether a number
sits a fraction of a cent above or below a threshold.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any

# Prices are quantised to this many decimal places internally. Polymarket ticks
# at 0.001 on some markets, Kalshi at 0.01, so 6dp leaves plenty of headroom for
# intermediate VWAP arithmetic.
PRICE_QUANTUM = Decimal("0.000001")
CENT = Decimal("0.01")

ONE_DOLLAR = Decimal("1")


def price(value: Any) -> Decimal:
    """Coerce anything numeric into a quantised probability price."""
    return Decimal(str(value)).quantize(PRICE_QUANTUM)


def cents_to_price(value: Any) -> Decimal:
    """Kalshi quotes integer cents (1..99). Convert to dollars."""
    return (Decimal(str(value)) / Decimal(100)).quantize(PRICE_QUANTUM)


class Venue(str, Enum):
    KALSHI = "kalshi"
    POLYMARKET = "polymarket"


class Side(str, Enum):
    """Which leg of a binary contract is being bought."""

    YES = "yes"
    NO = "no"

    @property
    def other(self) -> "Side":
        return Side.NO if self is Side.YES else Side.YES


class Comparator(str, Enum):
    """How the contract's strike relates to the settlement value."""

    ABOVE = "above"  # settles YES if value > strike (or >=; see note below)
    BELOW = "below"  # settles YES if value < strike
    BETWEEN = "between"  # settles YES if low <= value <= high
    # The negation of a range is two disjoint intervals, which no venue lists
    # directly. It exists so that the NO leg of a range market has a spec and
    # can still be matched against another range market's NO leg.
    OUTSIDE = "outside"

    @property
    def inverse(self) -> "Comparator":
        return _INVERSE[self]


_INVERSE: dict[Comparator, Comparator] = {
    Comparator.ABOVE: Comparator.BELOW,
    Comparator.BELOW: Comparator.ABOVE,
    Comparator.BETWEEN: Comparator.OUTSIDE,
    Comparator.OUTSIDE: Comparator.BETWEEN,
}

RANGE_COMPARATORS = (Comparator.BETWEEN, Comparator.OUTSIDE)


class Style(str, Enum):
    """Barrier style. Mixing these up is the classic source of fake arbitrage.

    TERMINAL - resolves on the value at one specific moment ("BTC above $70k at
    5pm on Aug 27"). This is a European digital.

    TOUCH    - resolves YES if the value is *ever* reached before expiry ("Will
    BTC hit $70k by Aug 31"). This is an American one-touch and is strictly more
    valuable than the terminal contract with the same strike and expiry, so the
    two must never be paired against each other.
    """

    TERMINAL = "terminal"
    TOUCH = "touch"


@dataclass(frozen=True)
class ContractSpec:
    """Canonical description of what a contract actually pays out on.

    Two markets from different venues are considered the same economic bet if
    and only if their specs match under :func:`arbmon.matching.specs_match`.
    """

    asset: str  # normalised ticker, e.g. "BTC"
    comparator: Comparator
    style: Style
    strike_low: Decimal | None = None  # for ABOVE and BETWEEN
    strike_high: Decimal | None = None  # for BELOW and BETWEEN
    resolve_at: dt.datetime | None = None  # UTC
    # Free-text description of the settlement index, e.g. "CF Benchmarks BRTI"
    # or "Binance BTCUSDT". Never used to *reject* a match, only to warn.
    resolution_source: str | None = None

    def inverted(self) -> "ContractSpec":
        """The spec of the complementary contract (this contract's NO leg).

        The NO leg of "BTC above $70k" is "BTC at-or-below $70k", so the single
        strike moves from ``strike_low`` to ``strike_high`` as the comparator
        flips. The boundary case - the settlement value landing exactly on the
        strike, where strict inequalities on both venues could resolve both legs
        NO - is a measure-zero difference that :mod:`arbmon.matching` reports as
        a caveat rather than trying to encode here.

        A range keeps both of its bounds; only its comparator flips, because the
        negation of "inside [low, high]" is "outside [low, high]".
        """
        if self.comparator in RANGE_COMPARATORS:
            strike_low, strike_high = self.strike_low, self.strike_high
        elif self.comparator is Comparator.ABOVE:
            strike_low, strike_high = None, self.strike_low
        else:
            strike_low, strike_high = self.strike_high, None

        return ContractSpec(
            asset=self.asset,
            comparator=self.comparator.inverse,
            style=self.style,
            strike_low=strike_low,
            strike_high=strike_high,
            resolve_at=self.resolve_at,
            resolution_source=self.resolution_source,
        )

    def describe(self) -> str:
        when = self.resolve_at.strftime("%Y-%m-%d %H:%MZ") if self.resolve_at else "?"
        verb = "reaches" if self.style is Style.TOUCH else "is"
        if self.comparator is Comparator.BETWEEN:
            return f"{self.asset} {verb} in [{self.strike_low}, {self.strike_high}] @ {when}"
        if self.comparator is Comparator.OUTSIDE:
            return f"{self.asset} {verb} outside [{self.strike_low}, {self.strike_high}] @ {when}"
        if self.comparator is Comparator.ABOVE:
            return f"{self.asset} {verb} above {self.strike_low} @ {when}"
        return f"{self.asset} {verb} below {self.strike_high} @ {when}"


@dataclass(frozen=True)
class PriceLevel:
    """One resting order book level. ``size`` is in whole contracts."""

    price: Decimal
    size: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "price", price(self.price))
        object.__setattr__(self, "size", int(self.size))


@dataclass
class NormalizedMarket:
    """A single binary market from one venue, in common units."""

    venue: Venue
    venue_id: str  # Kalshi ticker, or Polymarket YES clob token id
    title: str
    spec: ContractSpec | None
    # Executable asks: the prices at which *we* can buy, cheapest first.
    yes_asks: list[PriceLevel] = field(default_factory=list)
    no_asks: list[PriceLevel] = field(default_factory=list)
    # Top of book, kept separately because Kalshi hands these to us even when we
    # have no credentials for the depth endpoint.
    yes_bid: Decimal | None = None
    yes_ask: Decimal | None = None
    no_bid: Decimal | None = None
    no_ask: Decimal | None = None
    close_time: dt.datetime | None = None
    volume: Decimal | None = None
    liquidity: Decimal | None = None
    open_interest: int | None = None
    url: str | None = None
    # Venue-reported fee parameters, when the venue exposes them.
    taker_fee_bps: Decimal | None = None
    maker_fee_bps: Decimal | None = None
    fetched_at: dt.datetime = field(default_factory=lambda: dt.datetime.now(dt.timezone.utc))
    raw: dict[str, Any] = field(default_factory=dict)

    def asks_for(self, side: Side) -> list[PriceLevel]:
        return self.yes_asks if side is Side.YES else self.no_asks

    def best_ask(self, side: Side) -> Decimal | None:
        levels = self.asks_for(side)
        if levels:
            return levels[0].price
        return self.yes_ask if side is Side.YES else self.no_ask

    def has_depth(self) -> bool:
        return bool(self.yes_asks or self.no_asks)

    def spec_for(self, side: Side) -> ContractSpec | None:
        """The economic bet represented by buying ``side`` of this market."""
        if self.spec is None:
            return None
        return self.spec if side is Side.YES else self.spec.inverted()


@dataclass
class Leg:
    """One side of a proposed arbitrage: buy ``size`` of ``side`` on ``venue``."""

    venue: Venue
    venue_id: str
    title: str
    side: Side
    size: int
    vwap: Decimal  # average fill price in dollars, for display
    notional: Decimal  # exact cash paid for the contracts, before fees
    fee: Decimal  # total fee in dollars for this leg
    url: str | None = None
    levels_consumed: list[PriceLevel] = field(default_factory=list)

    @property
    def cost(self) -> Decimal:
        return (self.notional + self.fee).quantize(PRICE_QUANTUM)


@dataclass
class Opportunity:
    """A pair of legs that together pay $1.00 per contract at resolution."""

    kind: str  # "cross_venue" or "single_venue"
    spec: ContractSpec | None
    legs: list[Leg]
    size: int
    gross_cost: Decimal  # sum of leg notionals, before fees
    total_fees: Decimal
    profit: Decimal  # payout - gross_cost - fees, in dollars
    caveats: list[str] = field(default_factory=list)
    detected_at: dt.datetime = field(default_factory=lambda: dt.datetime.now(dt.timezone.utc))

    @property
    def payout(self) -> Decimal:
        return Decimal(self.size) * ONE_DOLLAR

    @property
    def capital(self) -> Decimal:
        """Cash actually required to put the trade on."""
        return (self.gross_cost + self.total_fees).quantize(PRICE_QUANTUM)

    @property
    def roi(self) -> Decimal:
        """Return on committed capital, as a fraction (0.012 == 1.2%)."""
        cap = self.capital
        if cap <= 0:
            return Decimal(0)
        return (self.profit / cap).quantize(PRICE_QUANTUM)

    @property
    def edge_per_contract(self) -> Decimal:
        if self.size <= 0:
            return Decimal(0)
        return (self.profit / Decimal(self.size)).quantize(PRICE_QUANTUM)

    def signature(self) -> str:
        """Stable identity for de-duplicating repeat sightings across scans."""
        parts = sorted(f"{leg.venue.value}:{leg.venue_id}:{leg.side.value}" for leg in self.legs)
        return "|".join(parts)

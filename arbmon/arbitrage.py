"""Detect and size arbitrage between two complementary contract legs.

The trade
---------
Buy one contract of each leg of a complementary pair. Exactly one of them
settles at $1.00 and the other at $0.00, whatever the underlying does. So the
position is riskless (modulo the caveats attached by :mod:`arbmon.matching`) and
its profit is::

    profit = size - (cost of leg A + cost of leg B) - fees

Sizing
------
Naively comparing the two best asks overstates the trade: the top of book might
be 5 contracts deep. This module walks both ask ladders together and reports the
size that maximises total profit.

That maximum always sits at a cumulative depth boundary of one of the two
ladders. Within a segment between boundaries the marginal cost of another
contract is constant, so marginal profit is constant too: if it is positive you
should take the whole segment, and if it is negative you should have stopped at
the segment's start. Evaluating only the boundaries is therefore exact, not an
approximation, and it keeps the search linear in the number of price levels.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Iterable, Sequence

from .config import ArbitrageConfig, Config, FeeConfig
from .fees import FeeModel, KalshiFeeModel, PolymarketFeeModel
from .matching import MatchedPair
from .models import (
    PRICE_QUANTUM,
    Leg,
    NormalizedMarket,
    Opportunity,
    PriceLevel,
    Side,
    Venue,
)


@dataclass(frozen=True)
class Fill:
    """The result of sweeping ``size`` contracts off an ask ladder.

    ``notional`` is the exact sum of price x quantity across the levels taken.
    ``vwap`` is derived from it and is therefore rounded; recomputing the
    notional from the rounded VWAP would let a sub-cent error into every profit
    figure, so the exact number is the one that is stored.
    """

    size: int
    notional: Decimal
    levels: tuple[PriceLevel, ...]

    @property
    def vwap(self) -> Decimal:
        return (self.notional / Decimal(self.size)).quantize(PRICE_QUANTUM)


def build_fee_model(market: NormalizedMarket, cfg: FeeConfig) -> FeeModel:
    """Pick the right fee schedule for a market, preferring venue-reported rates."""
    if market.venue is Venue.KALSHI:
        return KalshiFeeModel.for_ticker(
            market.venue_id,
            multiplier_override=cfg.kalshi_multiplier,
            maker=cfg.kalshi_maker,
            fixed_cost_per_trade=cfg.kalshi_fixed_cost_per_trade,
        )
    return PolymarketFeeModel.for_market(
        market.taker_fee_bps,
        default_bps=cfg.polymarket_default_bps,
        fixed_cost_per_trade=cfg.polymarket_fixed_cost_per_trade,
    )


def effective_ladder(
    market: NormalizedMarket, side: Side, cfg: ArbitrageConfig
) -> list[PriceLevel]:
    """The ask ladder we will trade against, with the slippage buffer applied.

    When we have no order book (no Kalshi credentials, or a Polymarket book
    fetch that failed) we fall back to a single synthetic level at the quoted
    best ask, sized at ``assumed_size_without_depth``. That is a guess, and any
    opportunity built on it is flagged as such.
    """
    levels = list(market.asks_for(side))
    if not levels:
        best = market.best_ask(side)
        if best is None:
            return []
        levels = [PriceLevel(best, cfg.assumed_size_without_depth)]

    buffer = cfg.slippage_buffer
    if buffer:
        levels = [PriceLevel(min(level.price + buffer, Decimal(1)), level.size) for level in levels]
    return [level for level in levels if Decimal(0) < level.price < Decimal(1)]


def sweep(levels: Sequence[PriceLevel], size: int) -> Fill | None:
    """Volume-weighted cost of buying ``size`` contracts off ``levels``."""
    if size <= 0:
        return None
    remaining = size
    spend = Decimal(0)
    taken: list[PriceLevel] = []
    for level in levels:
        if remaining <= 0:
            break
        take = min(remaining, level.size)
        spend += level.price * take
        taken.append(PriceLevel(level.price, take))
        remaining -= take
    if remaining > 0:
        return None  # not enough depth
    return Fill(size=size, notional=spend.quantize(PRICE_QUANTUM), levels=tuple(taken))


def candidate_sizes(
    a: Sequence[PriceLevel], b: Sequence[PriceLevel], max_contracts: int
) -> list[int]:
    """Cumulative depth boundaries common to both ladders, ascending."""
    depth_a = _cumulative(a)
    depth_b = _cumulative(b)
    if not depth_a or not depth_b:
        return []
    ceiling = min(depth_a[-1], depth_b[-1], max_contracts)
    if ceiling <= 0:
        return []
    sizes = {q for q in depth_a + depth_b if 0 < q <= ceiling}
    sizes.add(ceiling)
    return sorted(sizes)


def _cumulative(levels: Sequence[PriceLevel]) -> list[int]:
    total = 0
    out: list[int] = []
    for level in levels:
        total += level.size
        out.append(total)
    return out


def best_execution(
    levels_a: Sequence[PriceLevel],
    model_a: FeeModel,
    levels_b: Sequence[PriceLevel],
    model_b: FeeModel,
    cfg: ArbitrageConfig,
) -> tuple[int, Fill, Fill, Decimal, Decimal, Decimal] | None:
    """Maximise total profit over the shared depth of two ask ladders.

    Returns ``(size, fill_a, fill_b, fee_a, fee_b, profit)`` or None if no size
    is profitable.
    """
    best: tuple[int, Fill, Fill, Decimal, Decimal, Decimal] | None = None
    for size in candidate_sizes(levels_a, levels_b, cfg.max_contracts):
        fill_a = sweep(levels_a, size)
        fill_b = sweep(levels_b, size)
        if fill_a is None or fill_b is None:
            continue
        fee_a = model_a.amount(size, fill_a.vwap)
        fee_b = model_b.amount(size, fill_b.vwap)
        payout = Decimal(size)
        profit = (payout - fill_a.notional - fill_b.notional - fee_a - fee_b).quantize(PRICE_QUANTUM)
        if best is None or profit > best[5]:
            best = (size, fill_a, fill_b, fee_a, fee_b, profit)
    if best is None or best[5] <= 0:
        return None
    return best


def _leg(market: NormalizedMarket, side: Side, fill: Fill, fee: Decimal) -> Leg:
    return Leg(
        venue=market.venue,
        venue_id=market.venue_id,
        title=market.title,
        side=side,
        size=fill.size,
        vwap=fill.vwap,
        notional=fill.notional,
        fee=fee,
        url=market.url,
        levels_consumed=list(fill.levels),
    )


def _passes_filters(opportunity: Opportunity, cfg: ArbitrageConfig) -> bool:
    return (
        opportunity.profit >= cfg.min_profit
        and opportunity.edge_per_contract >= cfg.min_edge_per_contract
        and opportunity.roi >= cfg.min_roi
    )


def scan_pair(pair: MatchedPair, config: Config) -> list[Opportunity]:
    """Evaluate both hedged side combinations for one matched pair."""
    cfg = config.arbitrage
    if cfg.require_depth and not (pair.kalshi.has_depth() and pair.polymarket.has_depth()):
        return []

    kalshi_fees = build_fee_model(pair.kalshi, config.fees)
    poly_fees = build_fee_model(pair.polymarket, config.fees)

    found: list[Opportunity] = []
    for kalshi_side, poly_side in pair.side_combinations():
        kalshi_levels = effective_ladder(pair.kalshi, kalshi_side, cfg)
        poly_levels = effective_ladder(pair.polymarket, poly_side, cfg)
        if not kalshi_levels or not poly_levels:
            continue

        result = best_execution(kalshi_levels, kalshi_fees, poly_levels, poly_fees, cfg)
        if result is None:
            continue
        size, kalshi_fill, poly_fill, kalshi_fee, poly_fee, profit = result

        caveats = list(pair.caveats)
        if cfg.slippage_buffer:
            caveats.append(f"prices include a {cfg.slippage_buffer} slippage buffer per contract")

        opportunity = Opportunity(
            kind="cross_venue",
            spec=pair.spec,
            legs=[
                _leg(pair.kalshi, kalshi_side, kalshi_fill, kalshi_fee),
                _leg(pair.polymarket, poly_side, poly_fill, poly_fee),
            ],
            size=size,
            gross_cost=(kalshi_fill.notional + poly_fill.notional).quantize(PRICE_QUANTUM),
            total_fees=(kalshi_fee + poly_fee).quantize(PRICE_QUANTUM),
            profit=profit,
            caveats=caveats,
        )
        if _passes_filters(opportunity, cfg):
            found.append(opportunity)
    return found


def scan_single_venue(market: NormalizedMarket, config: Config) -> Opportunity | None:
    """Look for ``yes_ask + no_ask < 1`` inside a single market.

    This needs no cross-venue matching and carries none of the settlement-basis
    risk, so when it appears it is the cleanest trade on the board. It is also
    rare and short-lived, which is exactly why it is worth scanning for.
    """
    cfg = config.arbitrage
    if cfg.require_depth and not market.has_depth():
        return None

    yes_levels = effective_ladder(market, Side.YES, cfg)
    no_levels = effective_ladder(market, Side.NO, cfg)
    if not yes_levels or not no_levels:
        return None

    fees = build_fee_model(market, config.fees)
    result = best_execution(yes_levels, fees, no_levels, fees, cfg)
    if result is None:
        return None
    size, yes_fill, no_fill, yes_fee, no_fee, profit = result

    caveats: list[str] = []
    if not market.has_depth():
        caveats.append("size assumed from top-of-book quotes, not the live order book")
    if cfg.slippage_buffer:
        caveats.append(f"prices include a {cfg.slippage_buffer} slippage buffer per contract")

    opportunity = Opportunity(
        kind="single_venue",
        spec=market.spec,
        legs=[
            _leg(market, Side.YES, yes_fill, yes_fee),
            _leg(market, Side.NO, no_fill, no_fee),
        ],
        size=size,
        gross_cost=(yes_fill.notional + no_fill.notional).quantize(PRICE_QUANTUM),
        total_fees=(yes_fee + no_fee).quantize(PRICE_QUANTUM),
        profit=profit,
        caveats=caveats,
    )
    return opportunity if _passes_filters(opportunity, cfg) else None


def find_opportunities(
    pairs: Iterable[MatchedPair],
    markets: Iterable[NormalizedMarket],
    config: Config,
) -> list[Opportunity]:
    """Run every detector and return the results, most profitable first."""
    found: list[Opportunity] = []
    for pair in pairs:
        found.extend(scan_pair(pair, config))

    if config.arbitrage.include_single_venue:
        for market in markets:
            opportunity = scan_single_venue(market, config)
            if opportunity is not None:
                found.append(opportunity)

    found.sort(key=lambda o: o.profit, reverse=True)
    return found

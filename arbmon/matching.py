"""Pair economically equivalent contracts across the two venues.

Two markets are worth pairing when buying one leg on each venue guarantees that
exactly one of them pays $1. That happens in two shapes:

``same``
    Both venues ask the same question ("BTC above $70k at 5pm Aug 27"). The
    hedged combinations are Kalshi-YES with Polymarket-NO, and Kalshi-NO with
    Polymarket-YES.

``complement``
    The venues ask opposite questions (Kalshi "above $70k", Polymarket "below
    $70k"). The hedged combinations are YES/YES and NO/NO.

Matching only ever *proposes* a pair; every reason to be suspicious of it is
attached as a caveat and carried all the way through to the report, because a
cross-venue "arbitrage" is only riskless if the two contracts really do settle
on the same number at the same instant.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Iterable

from .config import MatchingConfig
from .models import RANGE_COMPARATORS, ContractSpec, NormalizedMarket, Side

log = logging.getLogger(__name__)

Relationship = str  # "same" | "complement"


@dataclass
class MatchedPair:
    kalshi: NormalizedMarket
    polymarket: NormalizedMarket
    relationship: Relationship
    caveats: list[str] = field(default_factory=list)

    def side_combinations(self) -> list[tuple[Side, Side]]:
        """The (kalshi_side, polymarket_side) pairs that hedge to $1."""
        if self.relationship == "same":
            return [(Side.YES, Side.NO), (Side.NO, Side.YES)]
        return [(Side.YES, Side.YES), (Side.NO, Side.NO)]

    @property
    def spec(self) -> ContractSpec | None:
        return self.kalshi.spec


def strikes_agree(a: Decimal | None, b: Decimal | None, cfg: MatchingConfig) -> bool:
    """Compare two strikes under a relative tolerance with an absolute floor.

    The absolute floor matters for cheap assets: 5 basis points of a $0.14 DOGE
    strike is a hundredth of a cent, which no venue quotes to.
    """
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    tolerance = max(abs(a) * cfg.strike_rel_tolerance, cfg.strike_abs_tolerance)
    return abs(a - b) <= tolerance


def times_agree(a: dt.datetime | None, b: dt.datetime | None, cfg: MatchingConfig) -> bool:
    if a is None or b is None:
        return False
    return abs((a - b).total_seconds()) <= cfg.max_resolution_delta_seconds


def specs_match(a: ContractSpec | None, b: ContractSpec | None, cfg: MatchingConfig) -> bool:
    """True when ``a`` and ``b`` are the same bet."""
    if a is None or b is None:
        return False
    if a.asset != b.asset:
        return False
    if a.comparator is not b.comparator:
        return False
    if a.style is not b.style and not cfg.allow_style_mismatch:
        return False
    if not times_agree(a.resolve_at, b.resolve_at, cfg):
        return False
    return strikes_agree(a.strike_low, b.strike_low, cfg) and strikes_agree(
        a.strike_high, b.strike_high, cfg
    )


def specs_complementary(a: ContractSpec | None, b: ContractSpec | None, cfg: MatchingConfig) -> bool:
    """True when ``b`` is the logical negation of ``a``."""
    if a is None or b is None:
        return False
    return specs_match(a.inverted(), b, cfg)


def pair_caveats(
    kalshi: NormalizedMarket,
    polymarket: NormalizedMarket,
    relationship: Relationship,
    cfg: MatchingConfig,
) -> list[str]:
    """Everything about this pair that makes it less than perfectly riskless."""
    caveats: list[str] = []
    k_spec, p_spec = kalshi.spec, polymarket.spec
    if k_spec is None or p_spec is None:
        return caveats

    if cfg.warn_on_source_mismatch:
        k_src, p_src = k_spec.resolution_source, p_spec.resolution_source
        if k_src and p_src and k_src != p_src:
            caveats.append(
                f"settlement index differs: Kalshi uses {k_src}, Polymarket uses {p_src} - "
                "the two can disagree at the strike"
            )
        elif not p_src:
            caveats.append(
                "Polymarket settlement source not identified from the market description - "
                "verify it matches Kalshi's index before trading"
            )

    if k_spec.resolve_at and p_spec.resolve_at:
        delta = abs((k_spec.resolve_at - p_spec.resolve_at).total_seconds())
        if delta > 60:
            caveats.append(
                f"resolution times differ by {int(delta)}s - the price can move in between"
            )

    if k_spec.style is not p_spec.style:
        caveats.append(
            f"barrier style mismatch ({k_spec.style.value} vs {p_spec.style.value}) - "
            "these are NOT the same contract; this pair is only shown because "
            "allow_style_mismatch is enabled"
        )

    if relationship == "complement" and k_spec.comparator not in RANGE_COMPARATORS:
        caveats.append(
            "complementary strikes: if the settlement value lands exactly on the strike, "
            "both legs can resolve NO under strict inequality rules"
        )

    if not kalshi.has_depth():
        caveats.append("Kalshi size is from top-of-book only (no order book credentials)")
    if not polymarket.has_depth():
        caveats.append("Polymarket size is from cached Gamma quotes, not the live book")

    return caveats


def _bucket_key(spec: ContractSpec, cfg: MatchingConfig) -> tuple[str, int]:
    """Coarse index key: asset plus a resolution-time bucket.

    The bucket is the tolerance window so that two contracts within tolerance
    always share a bucket or sit in adjacent ones (both of which we check).
    """
    if spec.resolve_at is None:
        return (spec.asset, 0)
    window = max(cfg.max_resolution_delta_seconds, 1)
    return (spec.asset, int(spec.resolve_at.timestamp()) // window)


def find_pairs(
    kalshi_markets: Iterable[NormalizedMarket],
    polymarket_markets: Iterable[NormalizedMarket],
    cfg: MatchingConfig,
) -> list[MatchedPair]:
    """Cross-join the two venues, bucketed by asset and resolution window."""
    index: dict[tuple[str, int], list[NormalizedMarket]] = {}
    for market in polymarket_markets:
        if market.spec is None:
            continue
        index.setdefault(_bucket_key(market.spec, cfg), []).append(market)

    pairs: list[MatchedPair] = []
    seen: set[tuple[str, str]] = set()
    for kalshi in kalshi_markets:
        if kalshi.spec is None:
            continue
        asset, bucket = _bucket_key(kalshi.spec, cfg)
        candidates: list[NormalizedMarket] = []
        for offset in (-1, 0, 1):
            candidates.extend(index.get((asset, bucket + offset), []))

        for polymarket in candidates:
            key = (kalshi.venue_id, polymarket.venue_id)
            if key in seen:
                continue
            if specs_match(kalshi.spec, polymarket.spec, cfg):
                relationship = "same"
            elif specs_complementary(kalshi.spec, polymarket.spec, cfg):
                relationship = "complement"
            else:
                continue
            seen.add(key)
            pairs.append(
                MatchedPair(
                    kalshi=kalshi,
                    polymarket=polymarket,
                    relationship=relationship,
                    caveats=pair_caveats(kalshi, polymarket, relationship, cfg),
                )
            )

    log.info("matched %d cross-venue contract pairs", len(pairs))
    return pairs


def summarize_unmatched(
    markets: Iterable[NormalizedMarket], pairs: Iterable[MatchedPair]
) -> list[NormalizedMarket]:
    """Markets that parsed fine but found no counterpart. Useful for tuning."""
    paired = {p.kalshi.venue_id for p in pairs} | {p.polymarket.venue_id for p in pairs}
    return [m for m in markets if m.spec is not None and m.venue_id not in paired]


__all__ = [
    "MatchedPair",
    "Relationship",
    "find_pairs",
    "pair_caveats",
    "specs_complementary",
    "specs_match",
    "strikes_agree",
    "summarize_unmatched",
    "times_agree",
]

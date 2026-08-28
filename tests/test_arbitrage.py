import datetime as dt
from decimal import Decimal

import pytest

from arbmon.arbitrage import (
    best_execution,
    build_fee_model,
    candidate_sizes,
    effective_ladder,
    find_opportunities,
    scan_pair,
    scan_single_venue,
    sweep,
)
from arbmon.config import Config
from arbmon.fees import KalshiFeeModel, PolymarketFeeModel, ZeroFeeModel
from arbmon.matching import MatchedPair
from arbmon.models import (
    Comparator,
    ContractSpec,
    NormalizedMarket,
    PriceLevel,
    Side,
    Style,
    Venue,
)

RESOLVE = dt.datetime(2025, 8, 27, 21, 0, tzinfo=dt.timezone.utc)
SPEC = ContractSpec(
    asset="BTC",
    comparator=Comparator.ABOVE,
    style=Style.TERMINAL,
    strike_low=Decimal("65000"),
    resolve_at=RESOLVE,
    resolution_source="CF Benchmarks",
)


def permissive_config(**arb_overrides) -> Config:
    """Thresholds low enough that the maths, not the filters, decides."""
    config = Config()
    config.arbitrage.min_profit = Decimal("0")
    config.arbitrage.min_edge_per_contract = Decimal("0")
    config.arbitrage.min_roi = Decimal("0")
    for key, value in arb_overrides.items():
        setattr(config.arbitrage, key, value)
    return config


def levels(*pairs) -> list[PriceLevel]:
    return [PriceLevel(Decimal(p), s) for p, s in pairs]


def make_market(venue, venue_id, yes=None, no=None, yes_ask=None, no_ask=None, spec=SPEC):
    return NormalizedMarket(
        venue=venue,
        venue_id=venue_id,
        title=f"{venue_id} contract",
        spec=spec,
        yes_asks=yes or [],
        no_asks=no or [],
        yes_ask=Decimal(yes_ask) if yes_ask else (yes[0].price if yes else None),
        no_ask=Decimal(no_ask) if no_ask else (no[0].price if no else None),
    )


def make_pair(kalshi, polymarket, relationship="same"):
    return MatchedPair(kalshi=kalshi, polymarket=polymarket, relationship=relationship)


# -- sweeping the book ------------------------------------------------------


def test_sweep_within_one_level():
    fill = sweep(levels(("0.40", 100)), 30)
    assert fill.size == 30
    assert fill.vwap == Decimal("0.400000")
    assert fill.notional == Decimal("12.000000")


def test_sweep_across_levels_uses_a_volume_weighted_price():
    fill = sweep(levels(("0.40", 10), ("0.50", 10)), 20)
    assert fill.vwap == Decimal("0.450000")
    assert [(str(l.price), l.size) for l in fill.levels] == [("0.400000", 10), ("0.500000", 10)]


def test_sweep_takes_only_what_it_needs_from_the_last_level():
    fill = sweep(levels(("0.40", 10), ("0.60", 100)), 15)
    # (10 * 0.40 + 5 * 0.60) / 15 = 7.0 / 15
    assert fill.vwap == Decimal("0.466667")
    assert fill.levels[-1].size == 5


def test_sweep_returns_none_when_the_book_is_too_thin():
    assert sweep(levels(("0.40", 10)), 11) is None


def test_sweep_rejects_non_positive_size():
    assert sweep(levels(("0.40", 10)), 0) is None


# -- candidate sizing -------------------------------------------------------


def test_candidate_sizes_are_the_shared_depth_boundaries():
    a = levels(("0.40", 50), ("0.42", 50))
    b = levels(("0.55", 60), ("0.60", 100))
    assert candidate_sizes(a, b, max_contracts=10_000) == [50, 60, 100]


def test_candidate_sizes_respect_the_configured_cap():
    a = levels(("0.40", 500))
    b = levels(("0.55", 500))
    assert candidate_sizes(a, b, max_contracts=120) == [120]


def test_candidate_sizes_empty_when_a_side_has_no_book():
    assert candidate_sizes(levels(("0.4", 10)), [], 100) == []


# -- profit maximisation ----------------------------------------------------


def test_best_execution_picks_an_interior_boundary():
    """Deeper is not always better: level two can wipe out the edge."""
    kalshi = levels(("0.40", 50), ("0.42", 50))
    poly = levels(("0.55", 60), ("0.60", 100))
    cfg = permissive_config().arbitrage

    result = best_execution(kalshi, KalshiFeeModel(), poly, PolymarketFeeModel(), cfg)
    assert result is not None
    size, _, _, _, _, profit = result
    assert size == 60
    assert profit == Decimal("1.780000")


def test_best_execution_stops_before_an_unprofitable_level():
    kalshi = levels(("0.40", 10), ("0.44", 100))
    poly = levels(("0.55", 500))
    cfg = permissive_config().arbitrage

    size, _, _, _, _, profit = best_execution(
        kalshi, KalshiFeeModel(), poly, PolymarketFeeModel(), cfg
    )
    assert size == 10
    assert profit == Decimal("0.330000")


def test_best_execution_returns_none_when_nothing_is_profitable():
    kalshi = levels(("0.60", 100))
    poly = levels(("0.55", 100))  # 1.15 total, hopeless
    cfg = permissive_config().arbitrage
    assert best_execution(kalshi, KalshiFeeModel(), poly, PolymarketFeeModel(), cfg) is None


def test_fees_can_eliminate_a_one_cent_edge():
    """0.49 + 0.50 = 0.99 looks like an edge until Kalshi's 1.75c fee lands."""
    kalshi = levels(("0.49", 100))
    poly = levels(("0.50", 100))
    cfg = permissive_config().arbitrage

    assert best_execution(kalshi, KalshiFeeModel(), poly, PolymarketFeeModel(), cfg) is None
    # The same trade is profitable if you pretend fees do not exist.
    assert best_execution(kalshi, ZeroFeeModel(), poly, ZeroFeeModel(), cfg) is not None


# -- ladders ----------------------------------------------------------------


def test_effective_ladder_falls_back_to_top_of_book():
    cfg = permissive_config(assumed_size_without_depth=42).arbitrage
    market = make_market(Venue.KALSHI, "K1", yes_ask="0.40", no_ask="0.55")
    ladder = effective_ladder(market, Side.YES, cfg)
    assert [(str(l.price), l.size) for l in ladder] == [("0.400000", 42)]


def test_effective_ladder_applies_the_slippage_buffer():
    cfg = permissive_config(slippage_buffer=Decimal("0.01")).arbitrage
    market = make_market(Venue.KALSHI, "K1", yes=levels(("0.40", 10), ("0.42", 10)))
    ladder = effective_ladder(market, Side.YES, cfg)
    assert [str(l.price) for l in ladder] == ["0.410000", "0.430000"]


def test_effective_ladder_is_empty_without_any_quote():
    cfg = permissive_config().arbitrage
    market = make_market(Venue.KALSHI, "K1")
    assert effective_ladder(market, Side.YES, cfg) == []


# -- fee model selection ----------------------------------------------------


def test_build_fee_model_picks_the_venue_schedule():
    config = Config()
    kalshi = make_market(Venue.KALSHI, "KXBTCD-1", yes_ask="0.4")
    poly = make_market(Venue.POLYMARKET, "111", yes_ask="0.4")
    assert isinstance(build_fee_model(kalshi, config.fees), KalshiFeeModel)
    assert isinstance(build_fee_model(poly, config.fees), PolymarketFeeModel)


def test_build_fee_model_uses_the_venue_reported_polymarket_rate():
    config = Config()
    poly = make_market(Venue.POLYMARKET, "111", yes_ask="0.4")
    poly.taker_fee_bps = Decimal("30")
    assert build_fee_model(poly, config.fees).base_fee_bps == Decimal("30")


# -- cross-venue detection --------------------------------------------------


def test_scan_pair_finds_the_textbook_arbitrage():
    config = permissive_config()
    kalshi = make_market(Venue.KALSHI, "K1", yes=levels(("0.40", 100)), no=levels(("0.62", 100)))
    poly = make_market(Venue.POLYMARKET, "P1", yes=levels(("0.63", 100)), no=levels(("0.55", 100)))

    found = scan_pair(make_pair(kalshi, poly), config)
    assert len(found) == 1

    opp = found[0]
    assert opp.kind == "cross_venue"
    assert opp.size == 100
    assert [(leg.venue, leg.side) for leg in opp.legs] == [
        (Venue.KALSHI, Side.YES),
        (Venue.POLYMARKET, Side.NO),
    ]
    assert opp.gross_cost == Decimal("95.000000")
    assert opp.total_fees == Decimal("1.680000")  # ceil(0.07 * 100 * 0.40 * 0.60)
    assert opp.profit == Decimal("3.320000")
    assert opp.payout == Decimal("100")
    assert opp.capital == Decimal("96.680000")
    assert opp.roi > Decimal("0.034")


def test_scan_pair_checks_both_hedged_directions():
    """Kalshi NO + Polymarket YES is a different trade at different prices."""
    config = permissive_config()
    kalshi = make_market(Venue.KALSHI, "K1", yes=levels(("0.90", 100)), no=levels(("0.30", 100)))
    poly = make_market(Venue.POLYMARKET, "P1", yes=levels(("0.60", 100)), no=levels(("0.95", 100)))

    found = scan_pair(make_pair(kalshi, poly), config)
    assert len(found) == 1
    assert [(leg.venue, leg.side) for leg in found[0].legs] == [
        (Venue.KALSHI, Side.NO),
        (Venue.POLYMARKET, Side.YES),
    ]


def test_scan_pair_uses_yes_yes_for_complementary_questions():
    config = permissive_config()
    below_spec = SPEC.inverted()
    kalshi = make_market(Venue.KALSHI, "K1", yes=levels(("0.40", 100)), no=levels(("0.62", 100)))
    poly = make_market(
        Venue.POLYMARKET, "P1", yes=levels(("0.55", 100)), no=levels(("0.46", 100)), spec=below_spec
    )

    found = scan_pair(make_pair(kalshi, poly, relationship="complement"), config)
    assert len(found) == 1
    assert [leg.side for leg in found[0].legs] == [Side.YES, Side.YES]


def test_scan_pair_finds_nothing_when_the_market_is_efficient():
    config = permissive_config()
    kalshi = make_market(Venue.KALSHI, "K1", yes=levels(("0.62", 100)), no=levels(("0.40", 100)))
    poly = make_market(Venue.POLYMARKET, "P1", yes=levels(("0.63", 100)), no=levels(("0.39", 100)))
    assert scan_pair(make_pair(kalshi, poly), config) == []


def test_scan_pair_carries_the_pair_caveats_through():
    config = permissive_config()
    kalshi = make_market(Venue.KALSHI, "K1", yes=levels(("0.40", 100)))
    poly = make_market(Venue.POLYMARKET, "P1", no=levels(("0.55", 100)))
    pair = make_pair(kalshi, poly)
    pair.caveats = ["settlement index differs"]

    found = scan_pair(pair, config)
    assert found[0].caveats == ["settlement index differs"]


def test_scan_pair_notes_the_slippage_buffer_in_the_caveats():
    config = permissive_config(slippage_buffer=Decimal("0.005"))
    kalshi = make_market(Venue.KALSHI, "K1", yes=levels(("0.35", 100)))
    poly = make_market(Venue.POLYMARKET, "P1", no=levels(("0.55", 100)))
    found = scan_pair(make_pair(kalshi, poly), config)
    assert any("slippage buffer" in c for c in found[0].caveats)


def test_require_depth_rejects_top_of_book_only_pairs():
    config = permissive_config(require_depth=True)
    kalshi = make_market(Venue.KALSHI, "K1", yes_ask="0.40", no_ask="0.62")
    poly = make_market(Venue.POLYMARKET, "P1", yes_ask="0.63", no_ask="0.55")
    assert scan_pair(make_pair(kalshi, poly), config) == []


# -- thresholds -------------------------------------------------------------


@pytest.mark.parametrize(
    "field,value",
    [
        ("min_profit", Decimal("10")),
        ("min_edge_per_contract", Decimal("0.10")),
        ("min_roi", Decimal("0.50")),
    ],
)
def test_thresholds_suppress_marginal_opportunities(field, value):
    config = permissive_config(**{field: value})
    kalshi = make_market(Venue.KALSHI, "K1", yes=levels(("0.40", 100)))
    poly = make_market(Venue.POLYMARKET, "P1", no=levels(("0.55", 100)))
    assert scan_pair(make_pair(kalshi, poly), config) == []


def test_default_thresholds_still_admit_a_real_edge():
    config = Config()  # untouched defaults
    kalshi = make_market(Venue.KALSHI, "K1", yes=levels(("0.40", 500)))
    poly = make_market(Venue.POLYMARKET, "P1", no=levels(("0.55", 500)))
    assert len(scan_pair(make_pair(kalshi, poly), config)) == 1


# -- single-venue detection -------------------------------------------------


def test_scan_single_venue_finds_a_crossed_book():
    config = permissive_config()
    market = make_market(Venue.KALSHI, "K1", yes=levels(("0.45", 100)), no=levels(("0.50", 100)))

    opp = scan_single_venue(market, config)
    assert opp is not None
    assert opp.kind == "single_venue"
    assert opp.size == 100
    assert opp.gross_cost == Decimal("95.000000")
    # ceil(0.07*100*0.45*0.55) + ceil(0.07*100*0.50*0.50) = 1.74 + 1.75
    assert opp.total_fees == Decimal("3.490000")
    assert opp.profit == Decimal("1.510000")


def test_scan_single_venue_ignores_a_normal_book():
    config = permissive_config()
    market = make_market(Venue.KALSHI, "K1", yes=levels(("0.52", 100)), no=levels(("0.50", 100)))
    assert scan_single_venue(market, config) is None


def test_scan_single_venue_flags_assumed_size():
    config = permissive_config()
    market = make_market(Venue.KALSHI, "K1", yes_ask="0.45", no_ask="0.50")
    opp = scan_single_venue(market, config)
    assert any("size assumed from top-of-book" in c for c in opp.caveats)


# -- aggregation ------------------------------------------------------------


def test_find_opportunities_sorts_by_profit_and_includes_both_kinds():
    config = permissive_config()
    kalshi = make_market(Venue.KALSHI, "K1", yes=levels(("0.40", 100)), no=levels(("0.55", 100)))
    poly = make_market(Venue.POLYMARKET, "P1", yes=levels(("0.63", 100)), no=levels(("0.55", 100)))

    found = find_opportunities([make_pair(kalshi, poly)], [kalshi, poly], config)
    kinds = {o.kind for o in found}
    assert kinds == {"cross_venue", "single_venue"}
    assert [o.profit for o in found] == sorted((o.profit for o in found), reverse=True)


def test_find_opportunities_can_skip_single_venue_scanning():
    config = permissive_config(include_single_venue=False)
    kalshi = make_market(Venue.KALSHI, "K1", yes=levels(("0.40", 100)), no=levels(("0.55", 100)))
    found = find_opportunities([], [kalshi], config)
    assert found == []


def test_opportunity_signature_is_stable_and_leg_order_independent():
    config = permissive_config()
    kalshi = make_market(Venue.KALSHI, "K1", yes=levels(("0.40", 100)))
    poly = make_market(Venue.POLYMARKET, "P1", no=levels(("0.55", 100)))
    opp = scan_pair(make_pair(kalshi, poly), config)[0]
    assert opp.signature() == "kalshi:K1:yes|polymarket:P1:no"

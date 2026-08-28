import datetime as dt
from decimal import Decimal

from arbmon.config import MatchingConfig
from arbmon.matching import (
    find_pairs,
    pair_caveats,
    specs_complementary,
    specs_match,
    strikes_agree,
    summarize_unmatched,
    times_agree,
)
from arbmon.models import Comparator, ContractSpec, NormalizedMarket, Side, Style, Venue

RESOLVE = dt.datetime(2025, 8, 27, 21, 0, tzinfo=dt.timezone.utc)


def spec(
    asset="BTC",
    comparator=Comparator.ABOVE,
    style=Style.TERMINAL,
    low=Decimal("65000"),
    high=None,
    resolve_at=RESOLVE,
    source="CF Benchmarks",
):
    return ContractSpec(
        asset=asset,
        comparator=comparator,
        style=style,
        strike_low=low,
        strike_high=high,
        resolve_at=resolve_at,
        resolution_source=source,
    )


def market(venue, venue_id, contract_spec, yes_ask="0.40", no_ask="0.55"):
    return NormalizedMarket(
        venue=venue,
        venue_id=venue_id,
        title=f"{venue_id} title",
        spec=contract_spec,
        yes_ask=Decimal(yes_ask),
        no_ask=Decimal(no_ask),
        yes_bid=Decimal(no_ask) and Decimal(1) - Decimal(no_ask),
        no_bid=Decimal(1) - Decimal(yes_ask),
    )


CFG = MatchingConfig()


# -- tolerances -------------------------------------------------------------


def test_strikes_agree_within_relative_tolerance():
    # 5bp of 65000 is 32.50
    assert strikes_agree(Decimal("65000"), Decimal("64999.99"), CFG)
    assert strikes_agree(Decimal("65000"), Decimal("65030"), CFG)
    assert not strikes_agree(Decimal("65000"), Decimal("65100"), CFG)


def test_strikes_agree_uses_the_absolute_floor_for_cheap_assets():
    # 5bp of 0.14 is far below any quoted tick, so the 0.50 floor takes over.
    assert strikes_agree(Decimal("0.14"), Decimal("0.20"), CFG)


def test_strikes_agree_requires_both_or_neither():
    assert strikes_agree(None, None, CFG)
    assert not strikes_agree(Decimal("1"), None, CFG)


def test_times_agree_within_window():
    later = RESOLVE + dt.timedelta(minutes=30)
    assert times_agree(RESOLVE, later, CFG)
    assert not times_agree(RESOLVE, RESOLVE + dt.timedelta(hours=2), CFG)


def test_times_agree_needs_both_timestamps():
    assert not times_agree(RESOLVE, None, CFG)


# -- spec equivalence -------------------------------------------------------


def test_identical_specs_match():
    assert specs_match(spec(), spec(), CFG)


def test_different_assets_do_not_match():
    assert not specs_match(spec(asset="BTC"), spec(asset="ETH"), CFG)


def test_different_strikes_do_not_match():
    assert not specs_match(spec(low=Decimal("65000")), spec(low=Decimal("70000")), CFG)


def test_touch_never_matches_terminal_by_default():
    """The whole point of tracking barrier style: these are different bets."""
    assert not specs_match(spec(style=Style.TOUCH), spec(style=Style.TERMINAL), CFG)


def test_style_mismatch_can_be_opted_into():
    lenient = MatchingConfig(allow_style_mismatch=True)
    assert specs_match(spec(style=Style.TOUCH), spec(style=Style.TERMINAL), lenient)


def test_above_and_below_are_complementary_not_equal():
    above = spec(comparator=Comparator.ABOVE, low=Decimal("65000"), high=None)
    below = spec(comparator=Comparator.BELOW, low=None, high=Decimal("65000"))
    assert not specs_match(above, below, CFG)
    assert specs_complementary(above, below, CFG)
    assert specs_complementary(below, above, CFG)


def test_range_complement_is_the_outside_comparator():
    inside = spec(comparator=Comparator.BETWEEN, low=Decimal("65000"), high=Decimal("66000"))
    outside = inside.inverted()
    assert outside.comparator is Comparator.OUTSIDE
    assert specs_complementary(inside, outside, CFG)
    assert not specs_match(inside, outside, CFG)


def test_specs_match_handles_none():
    assert not specs_match(None, spec(), CFG)
    assert not specs_complementary(spec(), None, CFG)


# -- pairing ----------------------------------------------------------------


def test_find_pairs_matches_the_same_question():
    k = market(Venue.KALSHI, "K1", spec())
    p = market(Venue.POLYMARKET, "P1", spec(source="Binance"))
    pairs = find_pairs([k], [p], CFG)
    assert len(pairs) == 1
    assert pairs[0].relationship == "same"
    assert pairs[0].side_combinations() == [(Side.YES, Side.NO), (Side.NO, Side.YES)]


def test_find_pairs_matches_opposite_questions():
    k = market(Venue.KALSHI, "K1", spec(comparator=Comparator.ABOVE, low=Decimal("65000")))
    p = market(
        Venue.POLYMARKET,
        "P1",
        spec(comparator=Comparator.BELOW, low=None, high=Decimal("65000")),
    )
    pairs = find_pairs([k], [p], CFG)
    assert len(pairs) == 1
    assert pairs[0].relationship == "complement"
    assert pairs[0].side_combinations() == [(Side.YES, Side.YES), (Side.NO, Side.NO)]


def test_find_pairs_ignores_unrelated_markets():
    k = market(Venue.KALSHI, "K1", spec(asset="BTC"))
    p = market(Venue.POLYMARKET, "P1", spec(asset="ETH"))
    assert find_pairs([k], [p], CFG) == []


def test_find_pairs_skips_markets_with_no_spec():
    k = market(Venue.KALSHI, "K1", None)
    p = market(Venue.POLYMARKET, "P1", spec())
    assert find_pairs([k], [p], CFG) == []


def test_find_pairs_spans_adjacent_time_buckets():
    """A pair 10 minutes apart must still match even across a bucket edge."""
    early = RESOLVE
    late = RESOLVE + dt.timedelta(minutes=10)
    k = market(Venue.KALSHI, "K1", spec(resolve_at=early))
    p = market(Venue.POLYMARKET, "P1", spec(resolve_at=late))
    assert len(find_pairs([k], [p], CFG)) == 1


def test_find_pairs_does_not_duplicate_the_same_two_markets():
    k = market(Venue.KALSHI, "K1", spec())
    p = market(Venue.POLYMARKET, "P1", spec())
    pairs = find_pairs([k, k], [p], CFG)
    assert len(pairs) == 1


def test_find_pairs_returns_one_entry_per_distinct_counterpart():
    k = market(Venue.KALSHI, "K1", spec())
    p1 = market(Venue.POLYMARKET, "P1", spec())
    p2 = market(Venue.POLYMARKET, "P2", spec(resolve_at=RESOLVE + dt.timedelta(minutes=5)))
    assert len(find_pairs([k], [p1, p2], CFG)) == 2


# -- caveats ----------------------------------------------------------------


def test_caveat_on_settlement_source_mismatch():
    k = market(Venue.KALSHI, "K1", spec(source="CF Benchmarks"))
    p = market(Venue.POLYMARKET, "P1", spec(source="Binance"))
    caveats = pair_caveats(k, p, "same", CFG)
    assert any("settlement index differs" in c for c in caveats)


def test_caveat_on_unknown_polymarket_source():
    k = market(Venue.KALSHI, "K1", spec(source="CF Benchmarks"))
    p = market(Venue.POLYMARKET, "P1", spec(source=None))
    caveats = pair_caveats(k, p, "same", CFG)
    assert any("settlement source not identified" in c for c in caveats)


def test_caveat_on_resolution_time_drift():
    k = market(Venue.KALSHI, "K1", spec())
    p = market(Venue.POLYMARKET, "P1", spec(resolve_at=RESOLVE + dt.timedelta(minutes=15)))
    caveats = pair_caveats(k, p, "same", CFG)
    assert any("resolution times differ by 900s" in c for c in caveats)


def test_caveat_on_boundary_risk_for_complementary_strikes():
    k = market(Venue.KALSHI, "K1", spec(comparator=Comparator.ABOVE))
    p = market(
        Venue.POLYMARKET, "P1", spec(comparator=Comparator.BELOW, low=None, high=Decimal("65000"))
    )
    caveats = pair_caveats(k, p, "complement", CFG)
    assert any("exactly on the strike" in c for c in caveats)


def test_caveat_when_depth_is_missing():
    k = market(Venue.KALSHI, "K1", spec())
    p = market(Venue.POLYMARKET, "P1", spec())
    caveats = pair_caveats(k, p, "same", CFG)
    assert any("top-of-book only" in c for c in caveats)
    assert any("cached Gamma quotes" in c for c in caveats)


def test_caveat_loudly_flags_a_style_mismatch():
    lenient = MatchingConfig(allow_style_mismatch=True)
    k = market(Venue.KALSHI, "K1", spec(style=Style.TERMINAL))
    p = market(Venue.POLYMARKET, "P1", spec(style=Style.TOUCH))
    caveats = pair_caveats(k, p, "same", lenient)
    assert any("NOT the same contract" in c for c in caveats)


# -- diagnostics ------------------------------------------------------------


def test_summarize_unmatched_lists_contracts_with_no_counterpart():
    k = market(Venue.KALSHI, "K1", spec())
    p = market(Venue.POLYMARKET, "P1", spec())
    orphan = market(Venue.KALSHI, "K2", spec(asset="SOL", low=Decimal("200")))
    pairs = find_pairs([k, orphan], [p], CFG)
    unmatched = summarize_unmatched([k, orphan, p], pairs)
    assert [m.venue_id for m in unmatched] == ["K2"]

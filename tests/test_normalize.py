import datetime as dt
from decimal import Decimal

import pytest

from arbmon.models import Comparator, Style
from arbmon.normalize import (
    detect_asset,
    detect_comparator,
    detect_source,
    detect_style,
    parse_all_money,
    parse_datetime,
    parse_kalshi_spec,
    parse_money,
    parse_polymarket_spec,
)
from tests.fixtures import (
    KALSHI_BTC_ABOVE,
    KALSHI_BTC_RANGE,
    KALSHI_ETH_BELOW,
    POLY_BTC_ABOVE,
    POLY_BTC_TOUCH,
    clone,
)


# -- assets -----------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Will Bitcoin be above $65,000?", "BTC"),
        ("Ethereum price at 5pm", "ETH"),
        ("SOL price today", "SOL"),
        ("Dogecoin to the moon", "DOGE"),
        ("Bitcoin Cash halving", "BCH"),
        ("Will the S&P 500 close higher?", None),
    ],
)
def test_detect_asset(text, expected):
    assert detect_asset(text) == expected


def test_detect_asset_respects_word_boundaries():
    # "SOLD" must not become Solana, "LINKED" must not become Chainlink.
    assert detect_asset("Tickets SOLD out") is None
    assert detect_asset("Accounts LINKED this week") is None


def test_detect_asset_prefers_the_longest_alias():
    assert detect_asset("Bitcoin Cash above $500") == "BCH"


def test_detect_asset_honours_the_allowed_list():
    assert detect_asset("Solana above $200", allowed=["BTC", "ETH"]) is None
    assert detect_asset("Solana above $200", allowed=["SOL"]) == "SOL"


def test_detect_asset_scans_later_arguments():
    assert detect_asset(None, "", "ethereum-price-slug") == "ETH"


# -- numbers ----------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("$65,000", Decimal("65000")),
        ("$70k", Decimal("70000")),
        ("1.5m", Decimal("1500000")),
        ("above 3000", Decimal("3000")),
        ("$64,999.99", Decimal("64999.99")),
        ("no numbers here", None),
    ],
)
def test_parse_money(text, expected):
    assert parse_money(text) == expected


def test_parse_all_money_finds_every_value():
    assert parse_all_money("between $60,000 and $65,000") == [
        Decimal("60000"),
        Decimal("65000"),
    ]


# -- dates ------------------------------------------------------------------


def test_parse_datetime_handles_z_suffix_and_offsets():
    expected = dt.datetime(2025, 8, 27, 21, 0, tzinfo=dt.timezone.utc)
    assert parse_datetime("2025-08-27T21:00:00Z") == expected
    assert parse_datetime("2025-08-27T17:00:00-04:00") == expected


def test_parse_datetime_assumes_utc_when_naive():
    assert parse_datetime("2025-08-27T21:00:00").tzinfo == dt.timezone.utc


def test_parse_datetime_rejects_garbage():
    assert parse_datetime("not a date") is None
    assert parse_datetime(None) is None


# -- style and direction ----------------------------------------------------


def test_detect_style_touch_vs_terminal():
    assert detect_style("Will Bitcoin hit $70,000 by August 31?") is Style.TOUCH
    assert detect_style("Will Bitcoin reach $70,000 this month?") is Style.TOUCH
    assert detect_style("Bitcoin above $65,000 on August 27?") is Style.TERMINAL
    assert detect_style("Bitcoin above $65,000 at 5pm?") is Style.TERMINAL


def test_detect_style_returns_none_when_unsure():
    assert detect_style("Bitcoin above $65,000?") is None
    assert detect_style("") is None


@pytest.mark.parametrize(
    "text,expected",
    [
        ("above $65,000", Comparator.ABOVE),
        ("$65,000 or above", Comparator.ABOVE),
        ("greater than $65,000", Comparator.ABOVE),
        ("below $3,000", Comparator.BELOW),
        ("under $3,000", Comparator.BELOW),
        ("dips to $50,000", Comparator.BELOW),
        (">= $65,000", Comparator.ABOVE),
        ("<= $65,000", Comparator.BELOW),
        ("between $60,000 and $65,000", Comparator.BETWEEN),
        ("something else entirely", None),
    ],
)
def test_detect_comparator(text, expected):
    assert detect_comparator(text) == expected


def test_detect_source():
    assert detect_source("settles on the CF Benchmarks BRTI") == "CF Benchmarks"
    assert detect_source("Binance BTCUSDT 1 minute candle") == "Binance"
    assert detect_source("no source named") is None


# -- Kalshi -----------------------------------------------------------------


def test_parse_kalshi_above_uses_structured_fields():
    spec = parse_kalshi_spec(KALSHI_BTC_ABOVE)
    assert spec is not None
    assert spec.asset == "BTC"
    assert spec.comparator is Comparator.ABOVE
    assert spec.style is Style.TERMINAL
    assert spec.strike_low == Decimal("64999.99")
    assert spec.strike_high is None
    assert spec.resolve_at == dt.datetime(2025, 8, 27, 21, 0, tzinfo=dt.timezone.utc)
    assert spec.resolution_source == "CF Benchmarks"


def test_parse_kalshi_below():
    spec = parse_kalshi_spec(KALSHI_ETH_BELOW)
    assert spec is not None
    assert spec.asset == "ETH"
    assert spec.comparator is Comparator.BELOW
    assert spec.strike_high == Decimal("3000.0")
    assert spec.strike_low is None


def test_parse_kalshi_range():
    spec = parse_kalshi_spec(KALSHI_BTC_RANGE)
    assert spec is not None
    assert spec.comparator is Comparator.BETWEEN
    assert spec.strike_low == Decimal("65000.0")
    assert spec.strike_high == Decimal("65999.99")


def test_parse_kalshi_falls_back_to_the_title_when_strikes_are_missing():
    market = clone(KALSHI_BTC_ABOVE, strike_type="custom", floor_strike=None, cap_strike=None)
    spec = parse_kalshi_spec(market)
    assert spec is not None
    assert spec.comparator is Comparator.ABOVE
    assert spec.strike_low == Decimal("65000")


def test_parse_kalshi_rejects_non_crypto_markets():
    market = clone(KALSHI_BTC_ABOVE, title="Will the Fed cut rates?", subtitle="", yes_sub_title="")
    market["ticker"] = "KXFED-25SEP"
    assert parse_kalshi_spec(market) is None


def test_parse_kalshi_respects_the_allowed_asset_list():
    assert parse_kalshi_spec(KALSHI_ETH_BELOW, allowed_assets=["BTC"]) is None


# -- Polymarket -------------------------------------------------------------


def test_parse_polymarket_terminal_above():
    spec = parse_polymarket_spec(POLY_BTC_ABOVE, event_title="Bitcoin price on August 27")
    assert spec is not None
    assert spec.asset == "BTC"
    assert spec.comparator is Comparator.ABOVE
    assert spec.style is Style.TERMINAL
    assert spec.strike_low == Decimal("65000")
    assert spec.resolution_source == "Binance"


def test_parse_polymarket_touch_defaults_to_above():
    spec = parse_polymarket_spec(POLY_BTC_TOUCH)
    assert spec is not None
    assert spec.style is Style.TOUCH
    assert spec.comparator is Comparator.ABOVE
    assert spec.strike_low == Decimal("65000")


def test_parse_polymarket_group_item_title_carries_the_strike():
    market = clone(
        POLY_BTC_ABOVE,
        question="What price will Bitcoin hit in August?",
        groupItemTitle="$72,000",
    )
    spec = parse_polymarket_spec(market)
    assert spec is not None
    assert spec.strike_low == Decimal("72000")
    assert spec.style is Style.TOUCH


def test_parse_polymarket_range_row():
    market = clone(
        POLY_BTC_ABOVE,
        question="What price will Bitcoin close at on August 27?",
        groupItemTitle="$65,000-$65,999",
    )
    spec = parse_polymarket_spec(market)
    assert spec is not None
    assert spec.comparator is Comparator.BETWEEN
    assert spec.strike_low == Decimal("65000")
    assert spec.strike_high == Decimal("65999")


def test_parse_polymarket_refuses_when_style_is_ambiguous():
    # No "hit"/"reach" and no date phrasing: we cannot tell touch from terminal.
    market = clone(POLY_BTC_ABOVE, question="Will Bitcoin be above $65,000?", events=[])
    assert parse_polymarket_spec(market) is None


def test_parse_polymarket_refuses_up_or_down_markets():
    # "Bitcoin Up or Down on August 27" has a date but no strike; the trailing
    # "27" must not be mistaken for one.
    market = clone(POLY_BTC_ABOVE, question="Bitcoin Up or Down on August 27?", events=[])
    assert parse_polymarket_spec(market) is None


def test_parse_polymarket_rejects_non_crypto():
    market = clone(POLY_BTC_ABOVE, question="Will the Knicks win above 50 games?", slug="knicks", events=[])
    assert parse_polymarket_spec(market) is None


# -- inversion --------------------------------------------------------------


def test_spec_inversion_flips_the_comparator_and_moves_the_strike():
    spec = parse_kalshi_spec(KALSHI_BTC_ABOVE)
    inverted = spec.inverted()
    assert inverted.comparator is Comparator.BELOW
    assert inverted.strike_high == Decimal("64999.99")
    assert inverted.strike_low is None
    # Inverting twice is the identity.
    assert inverted.inverted() == spec


def test_spec_inversion_preserves_style_and_time():
    spec = parse_kalshi_spec(KALSHI_BTC_ABOVE)
    inverted = spec.inverted()
    assert inverted.style is spec.style
    assert inverted.resolve_at == spec.resolve_at

import asyncio
from decimal import Decimal

import pytest

from arbmon.clients.kalshi import KalshiClient, derive_ask_ladders
from arbmon.clients.polymarket import (
    PolymarketClient,
    _complement,
    _json_list,
    _yes_index,
    ask_ladder,
)
from arbmon.config import Config
from arbmon.models import Side, Venue
from tests.fixtures import (
    KALSHI_BTC_ABOVE,
    KALSHI_ORDERBOOK,
    POLY_BOOK_NO,
    POLY_BOOK_YES,
    POLY_BTC_ABOVE,
    POLY_BTC_BELOW_REVERSED_OUTCOMES,
    clone,
)


@pytest.fixture
def kalshi_client():
    client = KalshiClient(Config())
    try:
        yield client
    finally:
        asyncio.run(client.aclose())


@pytest.fixture
def poly_client():
    client = PolymarketClient(Config())
    try:
        yield client
    finally:
        asyncio.run(client.aclose())


# -- Kalshi order book inversion -------------------------------------------


def test_derive_ask_ladders_inverts_the_opposite_side():
    """A resting NO bid at 36c is an offer to sell YES at 64c."""
    yes_asks, no_asks = derive_ask_ladders(KALSHI_ORDERBOOK)

    assert [(str(l.price), l.size) for l in yes_asks] == [
        ("0.640000", 150),
        ("0.650000", 300),
        ("0.660000", 500),
    ]
    assert [(str(l.price), l.size) for l in no_asks] == [
        ("0.380000", 100),
        ("0.400000", 250),
        ("0.420000", 400),
    ]


def test_derived_best_asks_agree_with_the_quoted_top_of_book():
    yes_asks, no_asks = derive_ask_ladders(KALSHI_ORDERBOOK)
    assert yes_asks[0].price == Decimal("0.64")  # KALSHI_BTC_ABOVE["yes_ask"] == 64
    assert no_asks[0].price == Decimal("0.38")  # KALSHI_BTC_ABOVE["no_ask"] == 38


def test_derive_ask_ladders_is_ascending_in_price():
    yes_asks, no_asks = derive_ask_ladders(KALSHI_ORDERBOOK)
    for ladder in (yes_asks, no_asks):
        prices = [level.price for level in ladder]
        assert prices == sorted(prices)


def test_derive_ask_ladders_drops_malformed_and_out_of_range_levels():
    book = {
        "yes": [[62, 100], [0, 50], [100, 50], [55, 0], ["x", 10], [7]],
        "no": [[36, 150]],
    }
    yes_asks, no_asks = derive_ask_ladders(book)
    assert len(yes_asks) == 1  # only the single valid NO bid
    assert len(no_asks) == 1  # only the 62c YES bid survives


def test_derive_ask_ladders_handles_an_empty_book():
    assert derive_ask_ladders({}) == ([], [])
    assert derive_ask_ladders({"yes": None, "no": None}) == ([], [])


# -- Kalshi normalisation ---------------------------------------------------


def test_kalshi_normalize_converts_cents_to_dollars(kalshi_client):
    market = kalshi_client.normalize(KALSHI_BTC_ABOVE)
    assert market.venue is Venue.KALSHI
    assert market.yes_ask == Decimal("0.64")
    assert market.no_ask == Decimal("0.38")
    assert market.yes_bid == Decimal("0.62")
    assert market.no_bid == Decimal("0.36")
    assert market.spec is not None
    assert market.url.endswith("KXBTCD-25AUG2717-T64999.99")


def test_kalshi_normalize_treats_sentinel_quotes_as_no_liquidity(kalshi_client):
    market = kalshi_client.normalize(clone(KALSHI_BTC_ABOVE, yes_ask=100, no_bid=0))
    assert market.yes_ask is None
    assert market.no_bid is None


def test_kalshi_normalize_reports_liquidity_in_dollars(kalshi_client):
    market = kalshi_client.normalize(KALSHI_BTC_ABOVE)
    assert market.liquidity == Decimal("4500")  # 450000 cents


def test_kalshi_normalize_attaches_depth_when_supplied(kalshi_client):
    market = kalshi_client.normalize(KALSHI_BTC_ABOVE, orderbook=KALSHI_ORDERBOOK)
    assert market.has_depth()
    assert market.best_ask(Side.YES) == Decimal("0.64")
    assert sum(level.size for level in market.yes_asks) == 950


def test_kalshi_client_is_unauthenticated_without_credentials(kalshi_client):
    assert kalshi_client.authenticated is False


# -- Polymarket helpers -----------------------------------------------------


def test_json_list_handles_encoded_and_native_lists():
    assert _json_list('["Yes", "No"]') == ["Yes", "No"]
    assert _json_list(["Yes", "No"]) == ["Yes", "No"]
    assert _json_list("not json") == []
    assert _json_list(None) == []


def test_yes_index_finds_the_yes_token():
    assert _yes_index(["yes", "no"]) == 0
    assert _yes_index(["no", "yes"]) == 1
    assert _yes_index(["up", "down"]) == 0
    assert _yes_index(["down", "up"]) == 1
    assert _yes_index(["lakers", "celtics"]) is None


def test_complement():
    assert _complement(Decimal("0.63")) == Decimal("0.37")
    assert _complement(None) is None


def test_ask_ladder_sorts_ascending_and_floors_fractional_size():
    ladder = ask_ladder(POLY_BOOK_YES)
    assert [(str(l.price), l.size) for l in ladder] == [
        ("0.640000", 300),  # 300.5 shares floored
        ("0.680000", 900),
    ]


def test_ask_ladder_rejects_degenerate_prices():
    book = {"asks": [{"price": "0", "size": "10"}, {"price": "1", "size": "10"}]}
    assert ask_ladder(book) == []


def test_ask_ladder_handles_a_missing_book():
    assert ask_ladder(None) == []
    assert ask_ladder({}) == []


# -- Polymarket normalisation ----------------------------------------------


def test_polymarket_normalize_standard_outcome_order(poly_client):
    market = poly_client.normalize(POLY_BTC_ABOVE)
    assert market is not None
    assert market.venue is Venue.POLYMARKET
    assert market.venue_id == "11111111"
    assert market.raw["_no_token_id"] == "22222222"
    assert market.yes_ask == Decimal("0.64")
    assert market.no_ask == Decimal("0.38")  # 1 - bestBid
    assert market.url == "https://polymarket.com/event/bitcoin-price-august-27"


def test_polymarket_normalize_reversed_outcome_order(poly_client):
    """When the NO token is listed first, Gamma's quotes describe the NO leg."""
    market = poly_client.normalize(POLY_BTC_BELOW_REVERSED_OUTCOMES)
    assert market is not None
    assert market.venue_id == "66666666"  # the YES token
    assert market.raw["_no_token_id"] == "55555555"
    assert market.yes_ask == Decimal("0.39")  # 1 - 0.61
    assert market.yes_bid == Decimal("0.37")  # 1 - 0.63
    assert market.no_ask == Decimal("0.63")  # the raw outcome-0 ask
    assert market.no_bid == Decimal("0.61")  # the raw outcome-0 bid


def test_polymarket_normalize_prefers_live_book_over_cached_quotes(poly_client):
    books = {"11111111": POLY_BOOK_YES, "22222222": POLY_BOOK_NO}
    market = poly_client.normalize(POLY_BTC_ABOVE, books=books)
    assert market is not None
    assert market.has_depth()
    assert market.yes_ask == Decimal("0.64")
    assert market.no_ask == Decimal("0.36")
    assert [l.size for l in market.no_asks] == [250, 800]


def test_polymarket_normalize_rejects_non_binary_markets(poly_client):
    market = clone(POLY_BTC_ABOVE, outcomes='["A", "B", "C"]', clobTokenIds='["1","2","3"]')
    assert poly_client.normalize(market) is None


def test_polymarket_normalize_rejects_unrecognised_outcome_labels(poly_client):
    market = clone(POLY_BTC_ABOVE, outcomes='["Lakers", "Celtics"]')
    assert poly_client.normalize(market) is None


def test_polymarket_normalize_reads_the_reported_fee_rate(poly_client):
    market = poly_client.normalize(clone(POLY_BTC_ABOVE, takerBaseFee=25))
    assert market is not None
    assert market.taker_fee_bps == Decimal("25")

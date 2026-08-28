"""Realistic payload fixtures, shaped like what each venue actually returns."""

from __future__ import annotations

import copy
from typing import Any

KALSHI_BTC_ABOVE: dict[str, Any] = {
    "ticker": "KXBTCD-25AUG2717-T64999.99",
    "event_ticker": "KXBTCD-25AUG2717",
    "market_type": "binary",
    "title": "Bitcoin price at 5pm EDT on Aug 27, 2025?",
    "subtitle": "$65,000 or above",
    "yes_sub_title": "$65,000 or above",
    "no_sub_title": "Below $65,000",
    "open_time": "2025-08-26T21:00:00Z",
    "close_time": "2025-08-27T21:00:00Z",
    "expiration_time": "2025-08-27T21:00:00Z",
    "expected_expiration_time": "2025-08-27T21:00:00Z",
    "status": "active",
    "yes_bid": 62,
    "yes_ask": 64,
    "no_bid": 36,
    "no_ask": 38,
    "last_price": 63,
    "volume": 120_000,
    "volume_24h": 40_000,
    "liquidity": 450_000,
    "open_interest": 5_000,
    "strike_type": "greater_or_equal",
    "floor_strike": 64999.99,
    "category": "Crypto",
    "rules_primary": (
        "If the price of Bitcoin (BTC) according to the CF Benchmarks Bitcoin Real Time "
        "Index (BRTI) at 5:00 PM EDT is above $64,999.99, then the market resolves to Yes."
    ),
}

KALSHI_ETH_BELOW: dict[str, Any] = {
    "ticker": "KXETHD-25AUG2717-B3000",
    "event_ticker": "KXETHD-25AUG2717",
    "title": "Ethereum price at 5pm EDT on Aug 27, 2025?",
    "yes_sub_title": "Below $3,000",
    "close_time": "2025-08-27T21:00:00Z",
    "expected_expiration_time": "2025-08-27T21:00:00Z",
    "status": "active",
    "yes_bid": 40,
    "yes_ask": 42,
    "no_bid": 58,
    "no_ask": 60,
    "strike_type": "less",
    "cap_strike": 3000.0,
    "category": "Crypto",
    "rules_primary": "Settles on the CF Benchmarks Ether Real Time Index (ETHUSD_RTI).",
}

KALSHI_BTC_RANGE: dict[str, Any] = {
    "ticker": "KXBTCD-25AUG2717-R65000-66000",
    "title": "Bitcoin price at 5pm EDT on Aug 27, 2025?",
    "yes_sub_title": "$65,000 to $65,999.99",
    "close_time": "2025-08-27T21:00:00Z",
    "expected_expiration_time": "2025-08-27T21:00:00Z",
    "status": "active",
    "yes_bid": 10,
    "yes_ask": 12,
    "no_bid": 88,
    "no_ask": 90,
    "strike_type": "between",
    "floor_strike": 65000.0,
    "cap_strike": 65999.99,
    "rules_primary": "CF Benchmarks BRTI.",
}

POLY_BTC_ABOVE: dict[str, Any] = {
    "id": "512345",
    "question": "Will Bitcoin be above $65,000 on August 27?",
    "conditionId": "0xabc123",
    "slug": "bitcoin-above-65000-on-august-27",
    "endDate": "2025-08-27T21:00:00Z",
    "startDate": "2025-08-20T00:00:00Z",
    "description": (
        "This market resolves to Yes if the Binance 1 minute candle for BTCUSDT at "
        "17:00 ET on August 27 has a close price above $65,000."
    ),
    "outcomes": '["Yes", "No"]',
    "outcomePrices": '["0.63", "0.37"]',
    "clobTokenIds": '["11111111", "22222222"]',
    "volumeNum": 900_000,
    "liquidityNum": 42_000,
    "bestBid": 0.62,
    "bestAsk": 0.64,
    "spread": 0.02,
    "active": True,
    "closed": False,
    "makerBaseFee": 0,
    "takerBaseFee": 0,
    "events": [
        {"title": "Bitcoin price on August 27", "slug": "bitcoin-price-august-27", "ticker": "btc-aug-27"}
    ],
}

POLY_BTC_TOUCH: dict[str, Any] = {
    "id": "512999",
    "question": "Will Bitcoin hit $65,000 by August 31?",
    "slug": "will-bitcoin-hit-65000-by-august-31",
    "endDate": "2025-08-27T21:00:00Z",
    "description": "Resolves Yes if BTCUSDT on Binance trades at or above $65,000 at any point.",
    "outcomes": '["Yes", "No"]',
    "outcomePrices": '["0.71", "0.29"]',
    "clobTokenIds": '["33333333", "44444444"]',
    "volumeNum": 300_000,
    "liquidityNum": 15_000,
    "bestBid": 0.70,
    "bestAsk": 0.72,
    "active": True,
    "closed": False,
    "events": [{"title": "Bitcoin price in August", "slug": "bitcoin-august"}],
}

POLY_BTC_BELOW_REVERSED_OUTCOMES: dict[str, Any] = {
    "id": "513000",
    "question": "Will Bitcoin be below $65,000 on August 27?",
    "slug": "bitcoin-below-65000-august-27",
    "endDate": "2025-08-27T21:00:00Z",
    "description": "Binance BTCUSDT close at 17:00 ET.",
    # Outcome order is reversed here: the NO token is listed first.
    "outcomes": '["No", "Yes"]',
    "outcomePrices": '["0.62", "0.38"]',
    "clobTokenIds": '["55555555", "66666666"]',
    "volumeNum": 50_000,
    "liquidityNum": 9_000,
    "bestBid": 0.61,
    "bestAsk": 0.63,
    "active": True,
    "closed": False,
    "events": [{"title": "Bitcoin price on August 27", "slug": "bitcoin-price-august-27"}],
}

KALSHI_ORDERBOOK: dict[str, Any] = {
    # Kalshi returns resting BIDS on each side, ascending by price.
    "yes": [[58, 400], [60, 250], [62, 100]],
    "no": [[34, 500], [35, 300], [36, 150]],
}

POLY_BOOK_YES: dict[str, Any] = {
    "market": "0xabc123",
    "asset_id": "11111111",
    "bids": [{"price": "0.60", "size": "500"}, {"price": "0.62", "size": "200"}],
    # Polymarket returns asks worst-first.
    "asks": [{"price": "0.68", "size": "900"}, {"price": "0.64", "size": "300.5"}],
}

POLY_BOOK_NO: dict[str, Any] = {
    "market": "0xabc123",
    "asset_id": "22222222",
    "bids": [{"price": "0.34", "size": "400"}],
    "asks": [{"price": "0.40", "size": "800"}, {"price": "0.36", "size": "250"}],
}


def clone(payload: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    out = copy.deepcopy(payload)
    out.update(overrides)
    return out

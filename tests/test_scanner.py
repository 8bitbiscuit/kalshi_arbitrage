"""End-to-end pipeline test: HTTP payloads in, priced opportunities out.

The two venues are stubbed at the transport layer rather than at the client
methods, so this exercises the real paging, signing, normalisation, order book
inversion, matching and sizing code against realistic payloads.
"""

from __future__ import annotations

import asyncio
import json
from decimal import Decimal

import httpx
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from arbmon.config import Config
from arbmon.scanner import Scanner
from arbmon.storage import Storage, opportunity_to_dict

# Kalshi: BTC >= $64,999.99 at 21:00Z. Book gives YES at 40c (300 deep) and
# NO at 62c (500 deep).
KALSHI_MARKET = {
    "ticker": "KXBTCD-25AUG2717-T64999.99",
    "event_ticker": "KXBTCD-25AUG2717",
    "title": "Bitcoin price at 5pm EDT on Aug 27, 2025?",
    "yes_sub_title": "$65,000 or above",
    "close_time": "2025-08-27T21:00:00Z",
    "expected_expiration_time": "2025-08-27T21:00:00Z",
    "status": "active",
    "yes_bid": 38,
    "yes_ask": 40,
    "no_bid": 60,
    "no_ask": 62,
    "volume": 100_000,
    "liquidity": 250_000,
    "open_interest": 4_000,
    "strike_type": "greater_or_equal",
    "floor_strike": 64999.99,
    "rules_primary": "Settles on the CF Benchmarks Bitcoin Real Time Index (BRTI) at 5pm EDT.",
}

KALSHI_ORDERBOOK = {"yes": [[38, 500]], "no": [[60, 300]]}

# A non-crypto market that must be filtered out before matching.
KALSHI_NOISE = {
    "ticker": "KXFED-25SEP-C25",
    "title": "Will the Fed cut rates in September?",
    "yes_sub_title": "25 bps or more",
    "close_time": "2025-09-17T18:00:00Z",
    "expected_expiration_time": "2025-09-17T18:00:00Z",
    "status": "active",
    "yes_bid": 55,
    "yes_ask": 57,
    "no_bid": 43,
    "no_ask": 45,
}

# Polymarket: the same question, with the NO token offered at 55c.
POLY_MARKET = {
    "id": "512345",
    "question": "Will Bitcoin be above $65,000 on August 27?",
    "conditionId": "0xabc123",
    "slug": "bitcoin-above-65000-on-august-27",
    "endDate": "2025-08-27T21:00:00Z",
    "description": "Resolves using the Binance BTCUSDT 1 minute candle close at 17:00 ET.",
    "outcomes": '["Yes", "No"]',
    "outcomePrices": '["0.46", "0.54"]',
    "clobTokenIds": '["77777777", "88888888"]',
    "volumeNum": 900_000,
    "liquidityNum": 42_000,
    "bestBid": 0.45,
    "bestAsk": 0.47,
    "active": True,
    "closed": False,
    "takerBaseFee": 0,
    "events": [{"title": "Bitcoin price on August 27", "slug": "bitcoin-price-august-27"}],
}

POLY_BOOKS = {
    "77777777": {"asset_id": "77777777", "asks": [{"price": "0.47", "size": "400"}], "bids": []},
    "88888888": {"asset_id": "88888888", "asks": [{"price": "0.55", "size": "400"}], "bids": []},
}


@pytest.fixture
def signing_key(tmp_path):
    """A throwaway RSA key so the Kalshi order book path is exercised."""
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    path = tmp_path / "kalshi.pem"
    path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    return path


@pytest.fixture
def config(tmp_path, signing_key):
    config = Config()
    config.max_scans = 1
    config.storage.sqlite_path = str(tmp_path / "arbmon.sqlite")
    config.output.jsonl_path = str(tmp_path / "opps.jsonl")
    config.kalshi.api_key_id = "test-key-id"
    config.kalshi.private_key_path = str(signing_key)
    return config


class Recorder:
    """Captures the requests each stubbed venue received."""

    def __init__(self) -> None:
        self.paths: list[str] = []
        self.headers: list[httpx.Headers] = []

    def note(self, request: httpx.Request) -> None:
        self.paths.append(request.url.path)
        self.headers.append(request.headers)


def kalshi_handler(recorder: Recorder):
    def handle(request: httpx.Request) -> httpx.Response:
        recorder.note(request)
        path = request.url.path
        if path.endswith("/orderbook"):
            return httpx.Response(200, json={"orderbook": KALSHI_ORDERBOOK})
        if path.endswith("/markets"):
            return httpx.Response(
                200, json={"markets": [KALSHI_MARKET, KALSHI_NOISE], "cursor": ""}
            )
        return httpx.Response(404, json={"error": "not found"})

    return handle


def gamma_handler(recorder: Recorder):
    def handle(request: httpx.Request) -> httpx.Response:
        recorder.note(request)
        offset = int(request.url.params.get("offset", 0))
        return httpx.Response(200, json=[POLY_MARKET] if offset == 0 else [])

    return handle


def clob_handler(recorder: Recorder):
    def handle(request: httpx.Request) -> httpx.Response:
        recorder.note(request)
        requested = json.loads(request.content.decode())
        return httpx.Response(
            200, json=[POLY_BOOKS[item["token_id"]] for item in requested if item["token_id"] in POLY_BOOKS]
        )

    return handle


def stub(client, handler) -> None:
    """Swap a live AsyncClient for one backed by a mock transport."""
    base_url = str(client._client.base_url)
    client._client = httpx.AsyncClient(base_url=base_url, transport=httpx.MockTransport(handler))


async def run_scan(config: Config):
    scanner = Scanner(config)
    kalshi_rec, gamma_rec, clob_rec = Recorder(), Recorder(), Recorder()
    stub(scanner.kalshi, kalshi_handler(kalshi_rec))
    stub(scanner.polymarket.gamma, gamma_handler(gamma_rec))
    stub(scanner.polymarket.clob, clob_handler(clob_rec))
    try:
        result = await scanner.scan_once()
    finally:
        await scanner.aclose()
    return result, kalshi_rec, gamma_rec, clob_rec


def test_scan_finds_the_cross_venue_arbitrage(config):
    result, *_ = asyncio.run(run_scan(config))

    assert result.errors == []
    # The Fed market is filtered out before matching.
    assert [m.venue_id for m in result.kalshi_markets] == ["KXBTCD-25AUG2717-T64999.99"]
    assert [m.venue_id for m in result.polymarket_markets] == ["77777777"]
    assert len(result.pairs) == 1
    assert result.pairs[0].relationship == "same"

    assert len(result.opportunities) == 1
    opp = result.opportunities[0]
    assert opp.kind == "cross_venue"
    # Kalshi YES is 300 deep at 40c, Polymarket NO is 400 deep at 55c.
    assert opp.size == 300
    assert [(leg.venue.value, leg.side.value, str(leg.vwap)) for leg in opp.legs] == [
        ("kalshi", "yes", "0.400000"),
        ("polymarket", "no", "0.550000"),
    ]
    assert opp.gross_cost == Decimal("285.000000")
    assert opp.total_fees == Decimal("5.040000")  # ceil(0.07 * 300 * 0.4 * 0.6)
    assert opp.profit == Decimal("9.960000")


def test_scan_flags_the_settlement_index_mismatch(config):
    result, *_ = asyncio.run(run_scan(config))
    caveats = result.opportunities[0].caveats
    assert any("CF Benchmarks" in c and "Binance" in c for c in caveats)


def test_scan_signs_kalshi_requests_and_fetches_the_order_book(config):
    result, kalshi_rec, _, _ = asyncio.run(run_scan(config))

    assert any(path.endswith("/orderbook") for path in kalshi_rec.paths)
    headers = kalshi_rec.headers[0]
    assert headers["KALSHI-ACCESS-KEY"] == "test-key-id"
    assert headers["KALSHI-ACCESS-TIMESTAMP"].isdigit()
    assert len(headers["KALSHI-ACCESS-SIGNATURE"]) > 40
    # Real depth, not the synthetic top-of-book fallback.
    assert result.kalshi_markets[0].has_depth()


def test_scan_fetches_both_polymarket_token_books(config):
    result, _, _, clob_rec = asyncio.run(run_scan(config))
    assert clob_rec.paths == ["/books"]
    market = result.polymarket_markets[0]
    assert [str(l.price) for l in market.yes_asks] == ["0.470000"]
    assert [str(l.price) for l in market.no_asks] == ["0.550000"]


def test_scan_persists_everything_it_saw(config):
    asyncio.run(run_scan(config))

    with Storage(config.storage.sqlite_path) as storage:
        scans = storage.scan_history()
        assert len(scans) == 1
        assert scans[0]["kalshi_count"] == 1
        assert scans[0]["poly_count"] == 1
        assert scans[0]["pair_count"] == 1
        assert scans[0]["opp_count"] == 1
        assert scans[0]["error"] is None
        assert scans[0]["finished_at"] is not None

        stats = storage.opportunity_stats()
        assert len(stats) == 1
        assert stats[0]["signature"] == "kalshi:KXBTCD-25AUG2717-T64999.99:yes|polymarket:77777777:no"
        assert stats[0]["sightings"] == 1

        rows = storage.conn.execute("SELECT venue, yes_ask, no_ask FROM snapshots").fetchall()
        assert {row["venue"] for row in rows} == {"kalshi", "polymarket"}


def test_scan_appends_to_the_jsonl_sink(config):
    asyncio.run(run_scan(config))
    lines = [json.loads(line) for line in open(config.output.jsonl_path)]
    assert len(lines) == 1
    assert lines[0]["profit"] == "9.960000"
    assert lines[0]["asset"] == "BTC"
    assert len(lines[0]["legs"]) == 2


def test_scan_survives_one_venue_failing(config):
    async def run():
        scanner = Scanner(config)
        stub(scanner.kalshi, lambda request: httpx.Response(500, json={"error": "boom"}))
        stub(scanner.polymarket.gamma, gamma_handler(Recorder()))
        stub(scanner.polymarket.clob, clob_handler(Recorder()))
        scanner.kalshi.retries = 0
        try:
            return await scanner.scan_once()
        finally:
            await scanner.aclose()

    result = asyncio.run(run())
    assert result.kalshi_markets == []
    assert len(result.polymarket_markets) == 1
    assert any("kalshi" in error for error in result.errors)
    assert result.opportunities == []

    with Storage(config.storage.sqlite_path) as storage:
        assert "kalshi" in storage.scan_history()[0]["error"]


def test_opportunity_serialisation_round_trips(config):
    result, *_ = asyncio.run(run_scan(config))
    payload = opportunity_to_dict(result.opportunities[0])
    assert json.loads(json.dumps(payload))["capital"] == "290.040000"
    assert payload["contract"].startswith("BTC is above 64999.99")

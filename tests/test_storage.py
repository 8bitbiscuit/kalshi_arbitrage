import datetime as dt
import json
from decimal import Decimal

from arbmon.models import (
    Comparator,
    ContractSpec,
    Leg,
    NormalizedMarket,
    Opportunity,
    PriceLevel,
    Side,
    Style,
    Venue,
)
from arbmon.storage import Storage, append_jsonl, opportunity_to_dict

NOW = dt.datetime(2025, 8, 27, 20, 0, tzinfo=dt.timezone.utc)
SPEC = ContractSpec(
    asset="BTC",
    comparator=Comparator.ABOVE,
    style=Style.TERMINAL,
    strike_low=Decimal("65000"),
    resolve_at=dt.datetime(2025, 8, 27, 21, 0, tzinfo=dt.timezone.utc),
    resolution_source="CF Benchmarks",
)


def sample_market():
    return NormalizedMarket(
        venue=Venue.KALSHI,
        venue_id="K1",
        title="Bitcoin above $65,000",
        spec=SPEC,
        yes_asks=[PriceLevel(Decimal("0.40"), 300)],
        no_asks=[PriceLevel(Decimal("0.62"), 500)],
        yes_bid=Decimal("0.38"),
        yes_ask=Decimal("0.40"),
        no_bid=Decimal("0.60"),
        no_ask=Decimal("0.62"),
        volume=Decimal("100000"),
        liquidity=Decimal("2500"),
        fetched_at=NOW,
    )


def sample_opportunity():
    legs = [
        Leg(Venue.KALSHI, "K1", "Kalshi BTC", Side.YES, 300, Decimal("0.40"), Decimal("120"), Decimal("5.04")),
        Leg(Venue.POLYMARKET, "P1", "Poly BTC", Side.NO, 300, Decimal("0.55"), Decimal("165"), Decimal("0")),
    ]
    return Opportunity(
        kind="cross_venue",
        spec=SPEC,
        legs=legs,
        size=300,
        gross_cost=Decimal("285"),
        total_fees=Decimal("5.04"),
        profit=Decimal("9.96"),
        caveats=["settlement index differs"],
        detected_at=NOW,
    )


def test_schema_is_created_on_open(tmp_path):
    with Storage(tmp_path / "nested" / "db.sqlite") as storage:
        tables = {
            row[0]
            for row in storage.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    assert {"scans", "snapshots", "opportunities"} <= tables


def test_scan_lifecycle(tmp_path):
    with Storage(tmp_path / "db.sqlite") as storage:
        scan_id = storage.start_scan(NOW)
        assert scan_id > 0
        storage.finish_scan(
            scan_id,
            finished_at=NOW + dt.timedelta(seconds=3),
            kalshi_count=12,
            poly_count=7,
            pair_count=3,
            opp_count=1,
        )
        row = storage.scan_history()[0]
        assert row["kalshi_count"] == 12
        assert row["opp_count"] == 1
        assert row["error"] is None


def test_record_snapshots_stores_depth_as_json(tmp_path):
    with Storage(tmp_path / "db.sqlite") as storage:
        scan_id = storage.start_scan(NOW)
        assert storage.record_snapshots(scan_id, [sample_market()]) == 1
        row = storage.conn.execute("SELECT * FROM snapshots").fetchone()
        assert row["venue"] == "kalshi"
        assert row["asset"] == "BTC"
        assert row["comparator"] == "above"
        assert row["style"] == "terminal"
        assert row["yes_ask"] == "0.40"
        assert json.loads(row["yes_depth"]) == [["0.400000", 300]]


def test_record_snapshots_tolerates_an_unparsed_spec(tmp_path):
    market = sample_market()
    market.spec = None
    with Storage(tmp_path / "db.sqlite") as storage:
        scan_id = storage.start_scan(NOW)
        storage.record_snapshots(scan_id, [market])
        row = storage.conn.execute("SELECT asset, resolve_at FROM snapshots").fetchone()
        assert row["asset"] is None
        assert row["resolve_at"] is None


def test_record_snapshots_of_nothing_is_a_noop(tmp_path):
    with Storage(tmp_path / "db.sqlite") as storage:
        assert storage.record_snapshots(storage.start_scan(NOW), []) == 0


def test_opportunity_stats_aggregate_repeat_sightings(tmp_path):
    with Storage(tmp_path / "db.sqlite") as storage:
        for _ in range(3):
            scan_id = storage.start_scan(NOW)
            storage.record_opportunities(scan_id, [sample_opportunity()])
        stats = storage.opportunity_stats()

    assert len(stats) == 1
    assert stats[0]["sightings"] == 3
    assert stats[0]["best_profit"] == 9.96
    assert stats[0]["contract"].startswith("BTC is above 65000")


def test_opportunity_to_dict_is_json_serialisable():
    payload = opportunity_to_dict(sample_opportunity())
    restored = json.loads(json.dumps(payload))
    assert restored["profit"] == "9.96"
    assert restored["capital"] == "290.040000"
    assert restored["payout"] == "300"
    assert restored["legs"][0]["cost"] == "125.040000"
    assert restored["caveats"] == ["settlement index differs"]


def test_append_jsonl_creates_the_file_and_appends(tmp_path):
    path = tmp_path / "out" / "opps.jsonl"
    assert append_jsonl(path, [sample_opportunity()]) == 1
    assert append_jsonl(path, [sample_opportunity()]) == 1
    assert len(path.read_text().strip().splitlines()) == 2


def test_append_jsonl_of_nothing_does_not_create_a_file(tmp_path):
    path = tmp_path / "opps.jsonl"
    assert append_jsonl(path, []) == 0
    assert not path.exists()

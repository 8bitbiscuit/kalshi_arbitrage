"""SQLite persistence for quotes, opportunities and scan runs.

Kept deliberately boring: one file, three tables, no ORM. The point is to be
able to answer "how often does an edge actually appear, and how long does it
survive?" after a few days of collection, which is the question that decides
whether any of this is worth trading.
"""

from __future__ import annotations

import datetime as dt
import json
import sqlite3
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable

from .models import NormalizedMarket, Opportunity

SCHEMA = """
CREATE TABLE IF NOT EXISTS scans (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at    TEXT NOT NULL,
    finished_at   TEXT,
    kalshi_count  INTEGER DEFAULT 0,
    poly_count    INTEGER DEFAULT 0,
    pair_count    INTEGER DEFAULT 0,
    opp_count     INTEGER DEFAULT 0,
    error         TEXT
);

CREATE TABLE IF NOT EXISTS snapshots (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_id     INTEGER NOT NULL REFERENCES scans(id),
    ts          TEXT NOT NULL,
    venue       TEXT NOT NULL,
    venue_id    TEXT NOT NULL,
    title       TEXT,
    asset       TEXT,
    comparator  TEXT,
    style       TEXT,
    strike_low  TEXT,
    strike_high TEXT,
    resolve_at  TEXT,
    yes_bid     TEXT,
    yes_ask     TEXT,
    no_bid      TEXT,
    no_ask      TEXT,
    volume      TEXT,
    liquidity   TEXT,
    yes_depth   TEXT,
    no_depth    TEXT
);

CREATE INDEX IF NOT EXISTS idx_snapshots_lookup ON snapshots(venue, venue_id, ts);
CREATE INDEX IF NOT EXISTS idx_snapshots_scan ON snapshots(scan_id);

CREATE TABLE IF NOT EXISTS opportunities (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_id           INTEGER NOT NULL REFERENCES scans(id),
    ts                TEXT NOT NULL,
    signature         TEXT NOT NULL,
    kind              TEXT NOT NULL,
    contract          TEXT,
    size              INTEGER NOT NULL,
    gross_cost        TEXT NOT NULL,
    total_fees        TEXT NOT NULL,
    profit            TEXT NOT NULL,
    roi               TEXT NOT NULL,
    edge_per_contract TEXT NOT NULL,
    legs_json         TEXT NOT NULL,
    caveats_json      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_opportunities_sig ON opportunities(signature, ts);
"""


class Storage:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Storage":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- writes -------------------------------------------------------------

    def start_scan(self, started_at: dt.datetime) -> int:
        cursor = self.conn.execute(
            "INSERT INTO scans (started_at) VALUES (?)", (_iso(started_at),)
        )
        self.conn.commit()
        return int(cursor.lastrowid or 0)

    def finish_scan(
        self,
        scan_id: int,
        *,
        finished_at: dt.datetime,
        kalshi_count: int,
        poly_count: int,
        pair_count: int,
        opp_count: int,
        error: str | None = None,
    ) -> None:
        self.conn.execute(
            """
            UPDATE scans
               SET finished_at = ?, kalshi_count = ?, poly_count = ?,
                   pair_count = ?, opp_count = ?, error = ?
             WHERE id = ?
            """,
            (_iso(finished_at), kalshi_count, poly_count, pair_count, opp_count, error, scan_id),
        )
        self.conn.commit()

    def record_snapshots(self, scan_id: int, markets: Iterable[NormalizedMarket]) -> int:
        rows = []
        for market in markets:
            spec = market.spec
            rows.append(
                (
                    scan_id,
                    _iso(market.fetched_at),
                    market.venue.value,
                    market.venue_id,
                    market.title,
                    spec.asset if spec else None,
                    spec.comparator.value if spec else None,
                    spec.style.value if spec else None,
                    _dec(spec.strike_low) if spec else None,
                    _dec(spec.strike_high) if spec else None,
                    _iso(spec.resolve_at) if spec and spec.resolve_at else None,
                    _dec(market.yes_bid),
                    _dec(market.yes_ask),
                    _dec(market.no_bid),
                    _dec(market.no_ask),
                    _dec(market.volume),
                    _dec(market.liquidity),
                    json.dumps([[str(l.price), l.size] for l in market.yes_asks]),
                    json.dumps([[str(l.price), l.size] for l in market.no_asks]),
                )
            )
        if not rows:
            return 0
        self.conn.executemany(
            """
            INSERT INTO snapshots (
                scan_id, ts, venue, venue_id, title, asset, comparator, style,
                strike_low, strike_high, resolve_at, yes_bid, yes_ask, no_bid,
                no_ask, volume, liquidity, yes_depth, no_depth
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            rows,
        )
        self.conn.commit()
        return len(rows)

    def record_opportunities(self, scan_id: int, opportunities: Iterable[Opportunity]) -> int:
        rows = []
        for opp in opportunities:
            rows.append(
                (
                    scan_id,
                    _iso(opp.detected_at),
                    opp.signature(),
                    opp.kind,
                    opp.spec.describe() if opp.spec else None,
                    opp.size,
                    _dec(opp.gross_cost),
                    _dec(opp.total_fees),
                    _dec(opp.profit),
                    _dec(opp.roi),
                    _dec(opp.edge_per_contract),
                    json.dumps([leg_to_dict(leg) for leg in opp.legs]),
                    json.dumps(opp.caveats),
                )
            )
        if not rows:
            return 0
        self.conn.executemany(
            """
            INSERT INTO opportunities (
                scan_id, ts, signature, kind, contract, size, gross_cost,
                total_fees, profit, roi, edge_per_contract, legs_json, caveats_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            rows,
        )
        self.conn.commit()
        return len(rows)

    # -- reads --------------------------------------------------------------

    def opportunity_stats(self) -> list[dict[str, Any]]:
        """How often each opportunity has been seen, and its best profit."""
        cursor = self.conn.execute(
            """
            SELECT signature, kind, contract, COUNT(*) AS sightings,
                   MIN(ts) AS first_seen, MAX(ts) AS last_seen,
                   MAX(CAST(profit AS REAL)) AS best_profit
              FROM opportunities
             GROUP BY signature
             ORDER BY sightings DESC
            """
        )
        return [dict(row) for row in cursor.fetchall()]

    def scan_history(self, limit: int = 20) -> list[dict[str, Any]]:
        cursor = self.conn.execute(
            "SELECT * FROM scans ORDER BY id DESC LIMIT ?", (limit,)
        )
        return [dict(row) for row in cursor.fetchall()]


def leg_to_dict(leg: Any) -> dict[str, Any]:
    return {
        "venue": leg.venue.value,
        "venue_id": leg.venue_id,
        "title": leg.title,
        "side": leg.side.value,
        "size": leg.size,
        "vwap": str(leg.vwap),
        "fee": str(leg.fee),
        "cost": str(leg.cost),
        "url": leg.url,
    }


def opportunity_to_dict(opp: Opportunity) -> dict[str, Any]:
    """JSON-serialisable form, used for both the JSONL sink and the API."""
    return {
        "detected_at": _iso(opp.detected_at),
        "signature": opp.signature(),
        "kind": opp.kind,
        "contract": opp.spec.describe() if opp.spec else None,
        "asset": opp.spec.asset if opp.spec else None,
        "size": opp.size,
        "gross_cost": str(opp.gross_cost),
        "total_fees": str(opp.total_fees),
        "capital": str(opp.capital),
        "payout": str(opp.payout),
        "profit": str(opp.profit),
        "roi": str(opp.roi),
        "edge_per_contract": str(opp.edge_per_contract),
        "legs": [leg_to_dict(leg) for leg in opp.legs],
        "caveats": list(opp.caveats),
    }


def append_jsonl(path: str | Path, opportunities: Iterable[Opportunity]) -> int:
    items = list(opportunities)
    if not items:
        return 0
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        for opp in items:
            handle.write(json.dumps(opportunity_to_dict(opp)) + "\n")
    return len(items)


def _iso(value: dt.datetime | None) -> str | None:
    return value.isoformat() if value else None


def _dec(value: Decimal | None) -> str | None:
    return str(value) if value is not None else None

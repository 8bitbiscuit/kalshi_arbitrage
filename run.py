#!/usr/bin/env python3
"""Command line entry point for the arbitrage scanner.

    python run.py scan                 # continuous scan loop
    python run.py once                 # a single scan, then exit
    python run.py markets --venue all  # what each venue looks like after parsing
    python run.py pairs                # which contracts matched across venues
    python run.py stats                # summarise what has been collected so far
    python run.py fees --venue kalshi --price 0.62 --qty 100

Nothing in here places an order. The scanner is read-only by construction:
neither client has an order-entry method.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from decimal import Decimal

from arbmon.config import Config, load_config
from arbmon.fees import KalshiFeeModel, PolymarketFeeModel
from arbmon.matching import find_pairs, summarize_unmatched
from arbmon.report import bold, dim, format_pairs, money
from arbmon.scanner import Scanner
from arbmon.storage import Storage


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Scan Kalshi and Polymarket for crypto contract arbitrage (read-only).",
    )
    parser.add_argument("-c", "--config", default=None, help="path to a YAML config file")
    parser.add_argument(
        "-l",
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="logging verbosity (default: INFO)",
    )
    sub = parser.add_subparsers(dest="command")

    scan = sub.add_parser("scan", help="run the scan loop until interrupted")
    scan.add_argument("--interval", type=float, default=None, help="seconds between scans")
    scan.add_argument("--max-scans", type=int, default=None, help="stop after N scans")

    once = sub.add_parser("once", help="run exactly one scan and exit")
    once.add_argument("--pairs", action="store_true", help="also print the matched-pair table")

    markets = sub.add_parser("markets", help="dump the contracts each venue parsed")
    markets.add_argument("--venue", choices=["kalshi", "polymarket", "all"], default="all")
    markets.add_argument("--limit", type=int, default=40)

    pairs = sub.add_parser("pairs", help="dump cross-venue matches without pricing them")
    pairs.add_argument("--limit", type=int, default=60)
    pairs.add_argument("--unmatched", action="store_true", help="also list contracts with no pair")

    sub.add_parser("stats", help="summarise the SQLite collection so far")

    fees = sub.add_parser("fees", help="compute the fee for a hypothetical fill")
    fees.add_argument("--venue", choices=["kalshi", "polymarket"], required=True)
    fees.add_argument("--price", type=str, required=True, help="fill price in dollars, e.g. 0.62")
    fees.add_argument("--qty", type=int, required=True)
    fees.add_argument("--bps", type=str, default="0", help="Polymarket base fee, in bps")

    return parser


def resolve_config(args: argparse.Namespace) -> Config:
    config = load_config(args.config)
    if getattr(args, "interval", None) is not None:
        config.scan_interval_seconds = args.interval
    if getattr(args, "max_scans", None) is not None:
        config.max_scans = args.max_scans
    if getattr(args, "pairs", False):
        config.output.verbose_pairs = True
    return config


async def cmd_scan(config: Config) -> int:
    async with Scanner(config) as scanner:
        try:
            await scanner.run()
        except KeyboardInterrupt:
            print("\nstopped")
    return 0


async def cmd_once(config: Config) -> int:
    config.max_scans = 1
    async with Scanner(config) as scanner:
        result = await scanner.scan_once()
        scanner.print_result(result)
        return 1 if result.errors and not result.all_markets else 0


async def cmd_markets(config: Config, venue: str, limit: int) -> int:
    async with Scanner(config) as scanner:
        kalshi_markets, poly_markets, errors = await scanner.collect_markets()
    for error in errors:
        print(f"! {error}", file=sys.stderr)

    groups = []
    if venue in ("kalshi", "all"):
        groups.append(("kalshi", kalshi_markets))
    if venue in ("polymarket", "all"):
        groups.append(("polymarket", poly_markets))

    for name, markets in groups:
        print(bold(f"\n{name}: {len(markets)} parsed crypto contracts"))
        for market in markets[:limit]:
            spec = market.spec.describe() if market.spec else "?"
            depth = f"{len(market.yes_asks)}/{len(market.no_asks)} levels" if market.has_depth() else "top-of-book"
            yes = f"{market.yes_ask:.3f}" if market.yes_ask is not None else "  -  "
            no = f"{market.no_ask:.3f}" if market.no_ask is not None else "  -  "
            print(f"  yes_ask={yes} no_ask={no}  {spec:<50} {dim(depth)}")
        if len(markets) > limit:
            print(dim(f"  ... and {len(markets) - limit} more"))
    return 0


async def cmd_pairs(config: Config, limit: int, show_unmatched: bool) -> int:
    async with Scanner(config) as scanner:
        kalshi_markets, poly_markets, errors = await scanner.collect_markets()
    for error in errors:
        print(f"! {error}", file=sys.stderr)

    pairs = find_pairs(kalshi_markets, poly_markets, config.matching)
    print(bold(f"{len(pairs)} matched pairs"))
    print(format_pairs(pairs, limit))

    if show_unmatched:
        unmatched = summarize_unmatched(kalshi_markets + poly_markets, pairs)
        print(bold(f"\n{len(unmatched)} contracts with no counterpart"))
        for market in unmatched[:limit]:
            spec = market.spec.describe() if market.spec else "?"
            print(f"  {market.venue.value:<11} {spec}")
    return 0


def cmd_stats(config: Config) -> int:
    with Storage(config.storage.sqlite_path) as storage:
        scans = storage.scan_history(limit=10)
        print(bold(f"last {len(scans)} scans"))
        for scan in scans:
            print(
                f"  #{scan['id']:<5} {scan['started_at']}  "
                f"kalshi={scan['kalshi_count']:<4} poly={scan['poly_count']:<4} "
                f"pairs={scan['pair_count']:<4} opps={scan['opp_count']:<3}"
                + (f"  error={scan['error']}" if scan["error"] else "")
            )

        stats = storage.opportunity_stats()
        print(bold(f"\n{len(stats)} distinct opportunities recorded"))
        for row in stats[:25]:
            print(
                f"  seen {row['sightings']:>4}x  best {money(Decimal(str(row['best_profit'] or 0)))}  "
                f"{row['kind']:<12} {row['contract'] or ''}"
            )
    return 0


def cmd_fees(venue: str, price: str, qty: int, bps: str) -> int:
    px = Decimal(price)
    model = (
        KalshiFeeModel()
        if venue == "kalshi"
        else PolymarketFeeModel(base_fee_bps=Decimal(bps))
    )
    quote = model.fee(qty, px)
    per_contract = quote.amount / Decimal(qty) if qty else Decimal(0)
    print(f"{venue}: {qty} contracts @ ${px}")
    print(f"  fee        {money(quote.amount)}  ({per_contract * 100:.3f} cents/contract)")
    print(f"  breakdown  {quote.detail}")
    print(f"  notional   {money(px * qty)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    command = args.command or "scan"

    if command == "fees":
        return cmd_fees(args.venue, args.price, args.qty, args.bps)

    config = resolve_config(args)

    if command == "stats":
        return cmd_stats(config)
    if command == "markets":
        return asyncio.run(cmd_markets(config, args.venue, args.limit))
    if command == "pairs":
        return asyncio.run(cmd_pairs(config, args.limit, args.unmatched))
    if command == "once":
        return asyncio.run(cmd_once(config))
    return asyncio.run(cmd_scan(config))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)

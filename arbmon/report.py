"""Human-readable console output."""

from __future__ import annotations

import os
import sys
from decimal import Decimal
from typing import Iterable, Sequence

from .matching import MatchedPair
from .models import Opportunity, Side

_USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


def bold(text: str) -> str:
    return _c(text, "1")


def green(text: str) -> str:
    return _c(text, "32")


def yellow(text: str) -> str:
    return _c(text, "33")


def dim(text: str) -> str:
    return _c(text, "2")


def money(value: Decimal) -> str:
    return f"${value:,.2f}"


def pct(value: Decimal) -> str:
    return f"{value * 100:.2f}%"


def scan_header(
    scan_number: int,
    kalshi_count: int,
    poly_count: int,
    pair_count: int,
    elapsed: float,
) -> str:
    return (
        f"{bold(f'scan #{scan_number}')}  "
        f"kalshi={kalshi_count} polymarket={poly_count} "
        f"matched_pairs={pair_count} in {elapsed:.1f}s"
    )


def format_opportunity(opp: Opportunity, index: int) -> str:
    """Multi-line detail for one opportunity, including the exact legs."""
    lines: list[str] = []
    label = "CROSS-VENUE" if opp.kind == "cross_venue" else "SAME-VENUE"
    headline = (
        f"[{index}] {label}  profit {money(opp.profit)}  "
        f"roi {pct(opp.roi)}  size {opp.size}  "
        f"edge {opp.edge_per_contract * 100:.2f}c/contract"
    )
    lines.append(green(bold(headline)))
    if opp.spec:
        lines.append(f"     contract: {opp.spec.describe()}")

    for leg in opp.legs:
        side = "YES" if leg.side is Side.YES else "NO "
        lines.append(
            f"     buy {side} x{leg.size:<6} @ {leg.vwap:.4f}  "
            f"fee {money(leg.fee):>8}  on {leg.venue.value:<11} {_truncate(leg.title, 58)}"
        )
        if leg.url:
            lines.append(dim(f"         {leg.url}"))

    lines.append(
        f"     cost {money(opp.gross_cost)} + fees {money(opp.total_fees)} "
        f"= {money(opp.capital)}   ->  payout {money(opp.payout)}"
    )
    for caveat in opp.caveats:
        lines.append(yellow(f"     ! {caveat}"))
    return "\n".join(lines)


def format_opportunities(opportunities: Sequence[Opportunity], max_rows: int = 25) -> str:
    if not opportunities:
        return dim("     no opportunities above thresholds")
    shown = opportunities[:max_rows]
    blocks = [format_opportunity(opp, i + 1) for i, opp in enumerate(shown)]
    if len(opportunities) > len(shown):
        blocks.append(dim(f"     ... and {len(opportunities) - len(shown)} more"))
    return "\n".join(blocks)


def format_pairs(pairs: Iterable[MatchedPair], max_rows: int = 25) -> str:
    """The matched-pair table, for verifying that matching is behaving."""
    rows = list(pairs)[:max_rows]
    if not rows:
        return dim("     no matched pairs")
    lines = [
        f"     {'contract':<52} {'rel':<11} {'k yes/no ask':<15} {'p yes/no ask':<15}"
    ]
    for pair in rows:
        contract = pair.spec.describe() if pair.spec else "?"
        lines.append(
            f"     {_truncate(contract, 52):<52} {pair.relationship:<11} "
            f"{_quote_pair(pair.kalshi.yes_ask, pair.kalshi.no_ask):<15} "
            f"{_quote_pair(pair.polymarket.yes_ask, pair.polymarket.no_ask):<15}"
        )
    return "\n".join(lines)


def _quote_pair(yes: Decimal | None, no: Decimal | None) -> str:
    left = f"{yes:.3f}" if yes is not None else "  -  "
    right = f"{no:.3f}" if no is not None else "  -  "
    return f"{left}/{right}"


def _truncate(text: str, width: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 1] + "…"

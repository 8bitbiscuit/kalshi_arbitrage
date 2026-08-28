"""Turn venue-specific market descriptions into a common :class:`ContractSpec`.

This module is where most of the risk in the whole project lives. If the parser
decides that Kalshi's "BTC above $70,000 at 5pm ET on Aug 27" and Polymarket's
"Will BTC hit $70,000 by Aug 31" are the same contract, the scanner will report
a beautiful arbitrage that is really a naked short of a one-touch option. So the
rules here are deliberately conservative:

* Anything that cannot be parsed with confidence returns ``None`` and is dropped
  rather than guessed at.
* Barrier style (terminal vs one-touch) is parsed explicitly and never inferred
  by default.
* Kalshi's structured ``strike_type`` / ``floor_strike`` / ``cap_strike`` fields
  are preferred over its prose title wherever they are present.
"""

from __future__ import annotations

import datetime as dt
import re
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable

from .models import Comparator, ContractSpec, Style

# --------------------------------------------------------------------------
# Assets
# --------------------------------------------------------------------------

# Canonical ticker -> the strings that mean it. Order matters within a value
# list only for readability; matching is longest-alias-first across the table.
ASSET_ALIASES: dict[str, tuple[str, ...]] = {
    "BTC": ("bitcoin", "btc", "xbt"),
    "ETH": ("ethereum", "ether", "eth"),
    "SOL": ("solana", "sol"),
    "XRP": ("ripple", "xrp"),
    "DOGE": ("dogecoin", "doge"),
    "ADA": ("cardano", "ada"),
    "AVAX": ("avalanche", "avax"),
    "LINK": ("chainlink", "link"),
    "LTC": ("litecoin", "ltc"),
    "BCH": ("bitcoin cash", "bch"),
    "MATIC": ("polygon", "matic", "pol"),
    "DOT": ("polkadot", "dot"),
    "SHIB": ("shiba inu", "shib"),
    "TRX": ("tron", "trx"),
    "BNB": ("binance coin", "bnb"),
    "TON": ("toncoin", "ton"),
    "PEPE": ("pepe",),
    "SUI": ("sui",),
    "HYPE": ("hyperliquid", "hype"),
}

# Longest first so "bitcoin cash" wins over "bitcoin".
_ALIAS_TABLE: list[tuple[str, str]] = sorted(
    ((alias, asset) for asset, aliases in ASSET_ALIASES.items() for alias in aliases),
    key=lambda pair: len(pair[0]),
    reverse=True,
)


def detect_asset(*texts: str | None, allowed: Iterable[str] | None = None) -> str | None:
    """Find the crypto asset referenced by any of ``texts``.

    Aliases are matched on word boundaries so that "LINKED" does not become
    Chainlink and the "SOL" in "SOLD" does not become Solana.
    """
    allowed_set = {a.upper() for a in allowed} if allowed else None
    for text in texts:
        if not text:
            continue
        haystack = text.lower()
        for alias, asset in _ALIAS_TABLE:
            if allowed_set is not None and asset not in allowed_set:
                continue
            if re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", haystack):
                return asset
    return None


# --------------------------------------------------------------------------
# Numbers
# --------------------------------------------------------------------------

_MONEY_RE = re.compile(
    r"\$?\s*(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s*(k|m|b)?\b",
    re.IGNORECASE,
)
_SUFFIX_MULTIPLIER = {"k": Decimal(1_000), "m": Decimal(1_000_000), "b": Decimal(1_000_000_000)}


def parse_money(text: str | None) -> Decimal | None:
    """Parse the first monetary value in ``text``. ``$70,000``, ``70k``, ``0.35``."""
    if not text:
        return None
    match = _MONEY_RE.search(text)
    if not match:
        return None
    return _money_from_match(match)


def parse_all_money(text: str | None) -> list[Decimal]:
    if not text:
        return []
    return [_money_from_match(m) for m in _MONEY_RE.finditer(text)]


def _money_from_match(match: re.Match[str]) -> Decimal:
    raw = match.group(1).replace(",", "")
    value = Decimal(raw)
    suffix = (match.group(2) or "").lower()
    if suffix:
        value *= _SUFFIX_MULTIPLIER[suffix]
    return value


def to_decimal(value: Any) -> Decimal | None:
    if value is None or value == "":
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


# --------------------------------------------------------------------------
# Dates
# --------------------------------------------------------------------------


def parse_datetime(value: Any) -> dt.datetime | None:
    """Parse RFC3339 / ISO8601 timestamps from either venue into aware UTC."""
    if value is None or value == "":
        return None
    if isinstance(value, dt.datetime):
        return value.astimezone(dt.timezone.utc) if value.tzinfo else value.replace(tzinfo=dt.timezone.utc)
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = dt.datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


# --------------------------------------------------------------------------
# Barrier style and direction
# --------------------------------------------------------------------------

# "hit"/"reach"/"touch" describe a path-dependent barrier: the contract pays if
# the price is EVER at or beyond the strike before expiry.
_TOUCH_WORDS = re.compile(
    r"\b(hit|hits|reach(?:es)?|touch(?:es)?|dip(?:s)?\s+to|at\s+any\s+(?:point|time)|ever)\b",
    re.IGNORECASE,
)
# "above X on <date>" / "at <time>" describe the value at a single instant.
_TERMINAL_WORDS = re.compile(
    r"\b(close|closes|closing|end\s+of|at\s+\d{1,2}\s*(?::\d{2})?\s*(?:am|pm)|"
    r"\bon\s+\w+\s+\d{1,2}\b)",
    re.IGNORECASE,
)

_ABOVE_WORDS = re.compile(
    r"\b(above|over|greater\s+than|higher\s+than|more\s+than|exceed(?:s)?|at\s+least|"
    r"or\s+(?:more|higher|above)|gte)\b",
    re.IGNORECASE,
)
_BELOW_WORDS = re.compile(
    r"\b(below|under|less\s+than|lower\s+than|beneath|at\s+most|"
    r"or\s+(?:less|lower|below)|down\s+to|dip(?:s)?\s+to|lte)\b",
    re.IGNORECASE,
)
# Comparison symbols need their own patterns: ">=" has no word boundary on
# either side, so it can never match inside a ``\b(...)\b`` alternation.
_ABOVE_SYMBOL = re.compile(r">=?|≥")
_BELOW_SYMBOL = re.compile(r"<=?|≤")
_BETWEEN_WORDS = re.compile(r"\b(between|and)\b", re.IGNORECASE)


def detect_style(text: str | None) -> Style | None:
    """Classify a question as a one-touch or a terminal (at-expiry) contract."""
    if not text:
        return None
    touch = bool(_TOUCH_WORDS.search(text))
    if touch:
        # "hit $X by <date>" is a touch; "hit $X at 5pm" would be terminal, but
        # that phrasing is rare enough that we prefer the safer classification.
        return Style.TOUCH
    if _TERMINAL_WORDS.search(text):
        return Style.TERMINAL
    return None


def detect_comparator(text: str | None) -> Comparator | None:
    if not text:
        return None
    above = _ABOVE_WORDS.search(text) or _ABOVE_SYMBOL.search(text)
    below = _BELOW_WORDS.search(text) or _BELOW_SYMBOL.search(text)
    if above and below:
        # Ambiguous phrasing such as "above $60,000 and below $65,000".
        if _BETWEEN_WORDS.search(text) and len(parse_all_money(text)) >= 2:
            return Comparator.BETWEEN
        # Otherwise trust whichever keyword appears first.
        return Comparator.ABOVE if above.start() < below.start() else Comparator.BELOW
    if above:
        return Comparator.ABOVE
    if below:
        return Comparator.BELOW
    if _BETWEEN_WORDS.search(text) and len(parse_all_money(text)) >= 2:
        return Comparator.BETWEEN
    return None


# --------------------------------------------------------------------------
# Kalshi
# --------------------------------------------------------------------------

_KALSHI_STRIKE_TYPES = {
    "greater": Comparator.ABOVE,
    "greater_or_equal": Comparator.ABOVE,
    "less": Comparator.BELOW,
    "less_or_equal": Comparator.BELOW,
    "between": Comparator.BETWEEN,
}

# Settlement source, extracted from the market's own rules text so that a
# source mismatch against Polymarket can be surfaced to the user.
_SOURCE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("CF Benchmarks", re.compile(r"cf\s*benchmark|\bbrti\b|\bbrr\b", re.I)),
    ("Coinbase", re.compile(r"coinbase", re.I)),
    ("Binance", re.compile(r"binance", re.I)),
    ("Chainlink", re.compile(r"chainlink", re.I)),
    ("Pyth", re.compile(r"\bpyth\b", re.I)),
    ("CoinDesk", re.compile(r"coindesk", re.I)),
    ("Kraken", re.compile(r"kraken", re.I)),
)


def detect_source(*texts: str | None) -> str | None:
    for text in texts:
        if not text:
            continue
        for name, pattern in _SOURCE_PATTERNS:
            if pattern.search(text):
                return name
    return None


def parse_kalshi_spec(market: dict[str, Any], allowed_assets: Iterable[str] | None = None) -> ContractSpec | None:
    """Build a :class:`ContractSpec` from a Kalshi market object.

    Prefers the structured strike fields; falls back to the prose title only
    when ``strike_type`` is ``custom``/``functional`` or the strikes are absent.
    """
    ticker = str(market.get("ticker") or "")
    title = str(market.get("title") or "")
    subtitle = str(market.get("subtitle") or "")
    yes_sub = str(market.get("yes_sub_title") or "")
    rules = str(market.get("rules_primary") or "")

    asset = detect_asset(ticker, title, subtitle, yes_sub, allowed=allowed_assets)
    if asset is None:
        return None

    comparator = _KALSHI_STRIKE_TYPES.get(str(market.get("strike_type") or "").lower())
    floor_strike = to_decimal(market.get("floor_strike"))
    cap_strike = to_decimal(market.get("cap_strike"))

    if comparator is None or (floor_strike is None and cap_strike is None):
        # Structured fields unusable - fall back to the human-readable strings.
        text = " ".join(part for part in (yes_sub, subtitle, title) if part)
        comparator = comparator or detect_comparator(text)
        values = parse_all_money(text)
        if comparator is None or not values:
            return None
        if comparator is Comparator.BETWEEN and len(values) >= 2:
            floor_strike, cap_strike = min(values[:2]), max(values[:2])
        elif comparator is Comparator.ABOVE:
            floor_strike = floor_strike or values[0]
        else:
            cap_strike = cap_strike or values[0]

    if comparator is Comparator.ABOVE and floor_strike is None:
        return None
    if comparator is Comparator.BELOW and cap_strike is None:
        # Kalshi sometimes encodes a "less than" market with only floor_strike.
        cap_strike = floor_strike
        if cap_strike is None:
            return None
    if comparator is Comparator.BETWEEN and (floor_strike is None or cap_strike is None):
        return None

    # Kalshi crypto markets settle on the value at expiration_time, so they are
    # terminal contracts unless the rules explicitly describe a touch.
    style = Style.TOUCH if _TOUCH_WORDS.search(f"{title} {subtitle} {yes_sub}") else Style.TERMINAL

    resolve_at = (
        parse_datetime(market.get("expected_expiration_time"))
        or parse_datetime(market.get("expiration_time"))
        or parse_datetime(market.get("close_time"))
    )

    return ContractSpec(
        asset=asset,
        comparator=comparator,
        style=style,
        strike_low=floor_strike if comparator in (Comparator.ABOVE, Comparator.BETWEEN) else None,
        strike_high=cap_strike if comparator in (Comparator.BELOW, Comparator.BETWEEN) else None,
        resolve_at=resolve_at,
        resolution_source=detect_source(rules, title) or "CF Benchmarks",
    )


# --------------------------------------------------------------------------
# Polymarket
# --------------------------------------------------------------------------

# Grouped markets ("What price will Bitcoin hit in August?") put the strike in
# groupItemTitle, e.g. "$70,000" or ">= $70,000" or "$70,000-$74,999".
_RANGE_RE = re.compile(
    r"\$?\s*([\d,]+(?:\.\d+)?)\s*(?:k|m)?\s*(?:-|to|–|—)\s*\$?\s*([\d,]+(?:\.\d+)?)\s*(k|m)?",
    re.IGNORECASE,
)


def parse_polymarket_spec(
    market: dict[str, Any],
    allowed_assets: Iterable[str] | None = None,
    event_title: str | None = None,
) -> ContractSpec | None:
    """Build a :class:`ContractSpec` from a Gamma market object."""
    question = str(market.get("question") or "")
    slug = str(market.get("slug") or "")
    group_item = str(market.get("groupItemTitle") or "")
    description = str(market.get("description") or "")

    asset = detect_asset(question, event_title, slug, allowed=allowed_assets)
    if asset is None:
        return None

    # The question carries the direction and style; the group item title, when
    # present, carries the strike for that particular row of a grouped event.
    text = f"{event_title or ''} {question}".strip()

    style = detect_style(question) or detect_style(event_title)
    if style is None:
        # No explicit signal. Refusing to guess is the whole point of this
        # module: an unmatched market costs us one missed trade, a mis-styled
        # match costs real money.
        return None

    comparator = detect_comparator(group_item) or detect_comparator(text)

    # Range rows like "$70,000-$74,999" are BETWEEN regardless of wording.
    range_match = _RANGE_RE.search(group_item) or _RANGE_RE.search(question)
    if range_match:
        low = _money_from_match_pair(range_match, 1)
        high = _money_from_match_pair(range_match, 2)
        if low is not None and high is not None:
            return ContractSpec(
                asset=asset,
                comparator=Comparator.BETWEEN,
                style=style,
                strike_low=min(low, high),
                strike_high=max(low, high),
                resolve_at=_polymarket_resolution_time(market),
                resolution_source=detect_source(description, question),
            )

    strike = parse_money(group_item) or parse_money(question)
    if strike is None:
        return None
    if comparator is None:
        # "Will BTC hit $70,000 by Aug 31" has no directional keyword but a
        # touch contract on an upside strike is unambiguously an "above".
        if style is Style.TOUCH:
            comparator = Comparator.ABOVE
        else:
            return None

    if comparator is Comparator.BETWEEN:
        return None  # needed two strikes and did not find a range

    return ContractSpec(
        asset=asset,
        comparator=comparator,
        style=style,
        strike_low=strike if comparator is Comparator.ABOVE else None,
        strike_high=strike if comparator is Comparator.BELOW else None,
        resolve_at=_polymarket_resolution_time(market),
        resolution_source=detect_source(description, question),
    )


def _money_from_match_pair(match: re.Match[str], group: int) -> Decimal | None:
    raw = match.group(group)
    if raw is None:
        return None
    value = to_decimal(raw.replace(",", ""))
    if value is None:
        return None
    suffix = (match.group(3) or "").lower()
    if suffix:
        value *= _SUFFIX_MULTIPLIER[suffix]
    return value


def _polymarket_resolution_time(market: dict[str, Any]) -> dt.datetime | None:
    return (
        parse_datetime(market.get("endDate"))
        or parse_datetime(market.get("end_date_iso"))
        or parse_datetime(market.get("gameStartTime"))
    )

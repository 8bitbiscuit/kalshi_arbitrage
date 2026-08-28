"""Configuration objects and YAML loading.

Everything the scanner does is driven from a :class:`Config`. ``config.example.yaml``
in the repo root documents each field; this module is the authority on defaults.
Secrets (Kalshi API key id, private key path) are read from the environment
first so the YAML file can stay checked in.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, is_dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, get_args, get_type_hints

import yaml

DEFAULT_ASSETS = ["BTC", "ETH", "SOL", "XRP", "DOGE"]


@dataclass
class KalshiConfig:
    base_url: str = "https://api.elections.kalshi.com/trade-api/v2"
    enabled: bool = True
    # Kalshi returns up to 1000 markets per page.
    page_limit: int = 1000
    max_pages: int = 20
    # Only these market statuses are tradeable.
    statuses: list[str] = field(default_factory=lambda: ["open"])
    # Restrict the crawl to these series prefixes. Empty means "crawl all open
    # markets and filter by category/title", which is slower but catches series
    # that have been renamed.
    series_prefixes: list[str] = field(default_factory=list)
    # Depth (order book) requires authentication. Without credentials the
    # scanner falls back to top-of-book quotes from the markets endpoint.
    fetch_orderbook: bool = True
    orderbook_depth: int = 20
    api_key_id: str | None = None
    private_key_path: str | None = None
    private_key_password: str | None = None
    requests_per_second: float = 8.0

    def resolved_api_key_id(self) -> str | None:
        return self.api_key_id or os.environ.get("KALSHI_API_KEY_ID")

    def resolved_private_key_path(self) -> str | None:
        return self.private_key_path or os.environ.get("KALSHI_PRIVATE_KEY_PATH")

    def resolved_private_key_password(self) -> str | None:
        return self.private_key_password or os.environ.get("KALSHI_PRIVATE_KEY_PASSWORD")

    def has_credentials(self) -> bool:
        path = self.resolved_private_key_path()
        return bool(self.resolved_api_key_id() and path and Path(path).expanduser().exists())


@dataclass
class PolymarketConfig:
    gamma_url: str = "https://gamma-api.polymarket.com"
    clob_url: str = "https://clob.polymarket.com"
    enabled: bool = True
    page_limit: int = 500
    max_pages: int = 20
    # Skip illiquid noise before we spend order book requests on it.
    min_liquidity: Decimal = Decimal("500")
    min_volume: Decimal = Decimal("0")
    fetch_orderbook: bool = True
    # /books accepts a batch of token ids per call.
    orderbook_batch_size: int = 100
    requests_per_second: float = 8.0


@dataclass
class FeeConfig:
    # Override Kalshi's 0.07 trading multiplier (None = use the per-series default).
    kalshi_multiplier: Decimal | None = None
    kalshi_maker: bool = False
    # Used when Polymarket does not report a fee rate for a market.
    polymarket_default_bps: Decimal = Decimal("0")
    # Flat per-leg cost, e.g. amortised transfer costs. Applied to every leg.
    kalshi_fixed_cost_per_trade: Decimal = Decimal("0")
    polymarket_fixed_cost_per_trade: Decimal = Decimal("0")


@dataclass
class MatchingConfig:
    assets: list[str] = field(default_factory=lambda: list(DEFAULT_ASSETS))
    # Two contracts must resolve within this many seconds of each other.
    max_resolution_delta_seconds: int = 3600
    # Strikes must agree within this relative tolerance (0.0005 == 5bp).
    strike_rel_tolerance: Decimal = Decimal("0.0005")
    # Absolute floor on the strike tolerance, for low-priced assets.
    strike_abs_tolerance: Decimal = Decimal("0.5")
    # Never pair a one-touch contract against a terminal one.
    allow_style_mismatch: bool = False
    # Emit a caveat (rather than reject) when settlement indices differ.
    warn_on_source_mismatch: bool = True


@dataclass
class ArbitrageConfig:
    # Minimum total dollar profit for an opportunity to be reported.
    min_profit: Decimal = Decimal("1.00")
    # Minimum per-contract edge in dollars. 0.005 == half a cent per contract.
    min_edge_per_contract: Decimal = Decimal("0.005")
    # Minimum return on committed capital, as a fraction.
    min_roi: Decimal = Decimal("0.005")
    # Cap the size we are willing to claim, regardless of book depth.
    max_contracts: int = 5000
    # When a venue gives us no order book, assume the quoted best ask is good
    # for this many contracts. Any opportunity relying on this is flagged.
    assumed_size_without_depth: int = 100
    # Subtracted from every fill price as a safety margin against the book
    # moving between the scan and the (hypothetical) execution.
    slippage_buffer: Decimal = Decimal("0.00")
    # Only report opportunities where we saw real order book depth on both legs.
    require_depth: bool = False
    # Also look for within-venue arbitrage (yes_ask + no_ask < 1).
    include_single_venue: bool = True


@dataclass
class StorageConfig:
    enabled: bool = True
    sqlite_path: str = "data/arbmon.sqlite"
    # Persist every quote we see, not just the opportunities. Useful for
    # backtesting how long edges actually survive; grows fast.
    record_snapshots: bool = True


@dataclass
class OutputConfig:
    # Append every opportunity to this JSON Lines file.
    jsonl_path: str | None = "data/opportunities.jsonl"
    # Print the full matched-pair table each scan, not just the hits.
    verbose_pairs: bool = False
    max_rows: int = 25


@dataclass
class Config:
    scan_interval_seconds: float = 30.0
    # 0 means "run forever".
    max_scans: int = 0
    http_timeout_seconds: float = 25.0
    http_retries: int = 4
    user_agent: str = "arbmon/0.1 (research; read-only)"
    kalshi: KalshiConfig = field(default_factory=KalshiConfig)
    polymarket: PolymarketConfig = field(default_factory=PolymarketConfig)
    fees: FeeConfig = field(default_factory=FeeConfig)
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    arbitrage: ArbitrageConfig = field(default_factory=ArbitrageConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def _coerce(value: Any, target_type: Any) -> Any:
    """Best-effort coercion of YAML scalars into the dataclass field types."""
    if value is None:
        return None
    args = get_args(target_type)
    if args:  # Optional[X] / X | None / list[X]
        non_none = [t for t in args if t is not type(None)]
        if len(non_none) == 1:
            return _coerce(value, non_none[0])
        return value
    if target_type is Decimal:
        return Decimal(str(value))
    if target_type is bool:
        return bool(value)
    if target_type is int:
        return int(value)
    if target_type is float:
        return float(value)
    if target_type is str:
        return str(value)
    return value


def _apply(section: Any, data: dict[str, Any]) -> None:
    # ``from __future__ import annotations`` makes ``field.type`` a string, so
    # resolve the real annotations against this module's namespace instead.
    hints = get_type_hints(type(section))
    known = {f.name for f in fields(section)}
    for key, value in data.items():
        if key not in known:
            raise ValueError(f"unknown config key {key!r} in {type(section).__name__}")
        current = getattr(section, key)
        if is_dataclass(current) and isinstance(value, dict):
            _apply(current, value)
        elif isinstance(value, list):
            setattr(section, key, list(value))
        else:
            setattr(section, key, _coerce(value, hints[key]))


def load_config(path: str | Path | None) -> Config:
    """Load a :class:`Config`, falling back to defaults when ``path`` is None."""
    cfg = Config()
    if path is None:
        return cfg
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"config file not found: {p}")
    data = yaml.safe_load(p.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config file must contain a mapping, got {type(data).__name__}")
    _apply(cfg, data)
    return cfg

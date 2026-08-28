from decimal import Decimal

import pytest

from arbmon.config import Config, load_config


def test_defaults_are_sane():
    config = Config()
    assert config.kalshi.base_url.endswith("/trade-api/v2")
    assert config.polymarket.gamma_url.startswith("https://")
    assert config.arbitrage.min_profit == Decimal("1.00")
    assert config.matching.allow_style_mismatch is False


def test_load_config_returns_defaults_for_none():
    assert load_config(None).scan_interval_seconds == 30.0


def test_load_config_reads_nested_sections(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        """
scan_interval_seconds: 5
arbitrage:
  min_profit: 2.5
  max_contracts: 250
matching:
  assets: [BTC, ETH]
  allow_style_mismatch: true
kalshi:
  fetch_orderbook: false
"""
    )
    config = load_config(path)
    assert config.scan_interval_seconds == 5
    assert config.arbitrage.min_profit == Decimal("2.5")
    assert config.arbitrage.max_contracts == 250
    assert config.matching.assets == ["BTC", "ETH"]
    assert config.matching.allow_style_mismatch is True
    assert config.kalshi.fetch_orderbook is False
    # Untouched fields keep their defaults.
    assert config.polymarket.enabled is True


def test_decimal_fields_survive_yaml_floats(tmp_path):
    """YAML gives us a float; money fields must land as exact Decimals."""
    path = tmp_path / "config.yaml"
    path.write_text("arbitrage:\n  min_edge_per_contract: 0.005\n")
    value = load_config(path).arbitrage.min_edge_per_contract
    assert isinstance(value, Decimal)
    assert value == Decimal("0.005")


def test_optional_decimal_field_accepts_a_value(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("fees:\n  kalshi_multiplier: 0.035\n")
    assert load_config(path).fees.kalshi_multiplier == Decimal("0.035")


def test_unknown_keys_are_rejected(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("arbitrage:\n  min_prophet: 3\n")
    with pytest.raises(ValueError, match="unknown config key 'min_prophet'"):
        load_config(path)


def test_missing_file_is_an_error():
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/config.yaml")


def test_non_mapping_file_is_an_error(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("- just\n- a list\n")
    with pytest.raises(ValueError, match="must contain a mapping"):
        load_config(path)


def test_empty_file_yields_defaults(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("")
    assert load_config(path).scan_interval_seconds == Config().scan_interval_seconds


def test_credentials_come_from_the_environment(monkeypatch, tmp_path):
    key = tmp_path / "key.pem"
    key.write_text("not a real key")
    monkeypatch.setenv("KALSHI_API_KEY_ID", "abc-123")
    monkeypatch.setenv("KALSHI_PRIVATE_KEY_PATH", str(key))
    config = Config()
    assert config.kalshi.resolved_api_key_id() == "abc-123"
    assert config.kalshi.has_credentials() is True


def test_missing_key_file_means_no_credentials(monkeypatch):
    monkeypatch.setenv("KALSHI_API_KEY_ID", "abc-123")
    monkeypatch.setenv("KALSHI_PRIVATE_KEY_PATH", "/nope/key.pem")
    assert Config().kalshi.has_credentials() is False


def test_explicit_config_beats_the_environment(monkeypatch, tmp_path):
    key = tmp_path / "key.pem"
    key.write_text("x")
    monkeypatch.setenv("KALSHI_API_KEY_ID", "from-env")
    config = Config()
    config.kalshi.api_key_id = "from-file"
    assert config.kalshi.resolved_api_key_id() == "from-file"

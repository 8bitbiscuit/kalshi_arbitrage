from decimal import Decimal

import pytest

from arbmon.fees import (
    KalshiFeeModel,
    PolymarketFeeModel,
    ZeroFeeModel,
    kalshi_multiplier_for,
    roundup_cent,
)


def test_roundup_cent_always_rounds_up():
    assert roundup_cent(Decimal("1.001")) == Decimal("1.01")
    assert roundup_cent(Decimal("1.010")) == Decimal("1.01")
    assert roundup_cent(Decimal("0.0001")) == Decimal("0.01")


def test_kalshi_fee_matches_published_formula():
    # 0.07 * 100 * 0.40 * 0.60 = 1.68 exactly, so no rounding kicks in.
    model = KalshiFeeModel()
    assert model.amount(100, Decimal("0.40")) == Decimal("1.68")


def test_kalshi_fee_rounds_up_to_the_next_cent():
    # 0.07 * 1 * 0.50 * 0.50 = 0.0175 -> 0.02
    model = KalshiFeeModel()
    assert model.amount(1, Decimal("0.50")) == Decimal("0.02")


def test_kalshi_fee_is_symmetric_around_fifty_cents():
    model = KalshiFeeModel()
    assert model.amount(250, Decimal("0.30")) == model.amount(250, Decimal("0.70"))


def test_kalshi_fee_peaks_at_the_midpoint():
    model = KalshiFeeModel()
    mid = model.amount(1000, Decimal("0.50"))
    wing = model.amount(1000, Decimal("0.05"))
    assert mid > wing
    # 1.75c per contract is the theoretical maximum at P = 0.5.
    assert mid == Decimal("17.50")


def test_kalshi_reduced_multiplier_for_index_series():
    assert kalshi_multiplier_for("KXINX-25AUG27-B5000") == Decimal("0.035")
    assert kalshi_multiplier_for("KXBTCD-25AUG2717-T64999.99") == Decimal("0.07")
    assert kalshi_multiplier_for("KXBTCD-x", override=Decimal("0.01")) == Decimal("0.01")


def test_kalshi_for_ticker_picks_the_series_multiplier():
    model = KalshiFeeModel.for_ticker("KXINX-25AUG27-B5000")
    assert model.multiplier == Decimal("0.035")


def test_kalshi_maker_fee_is_opt_in():
    taker = KalshiFeeModel().amount(400, Decimal("0.25"))
    maker = KalshiFeeModel(maker=True).amount(400, Decimal("0.25"))
    # ceil(0.0025 * 400) = 1.00 added on top.
    assert maker - taker == Decimal("1.00")


def test_kalshi_zero_quantity_is_free():
    assert KalshiFeeModel().amount(0, Decimal("0.5")) == Decimal("0")


def test_polymarket_default_is_free():
    assert PolymarketFeeModel().amount(1000, Decimal("0.42")) == Decimal("0")


def test_polymarket_fee_uses_the_cheaper_side():
    model = PolymarketFeeModel(base_fee_bps=Decimal("100"))  # 1%
    # min(0.80, 0.20) = 0.20; 1% * 0.20 * 500 = 1.00
    assert model.amount(500, Decimal("0.80")) == Decimal("1.00")
    assert model.amount(500, Decimal("0.20")) == Decimal("1.00")


def test_polymarket_for_market_prefers_the_reported_rate():
    model = PolymarketFeeModel.for_market(Decimal("50"), default_bps=Decimal("200"))
    assert model.base_fee_bps == Decimal("50")
    fallback = PolymarketFeeModel.for_market(None, default_bps=Decimal("200"))
    assert fallback.base_fee_bps == Decimal("200")


def test_fixed_cost_is_added_once_per_leg():
    model = PolymarketFeeModel(fixed_cost_per_trade=Decimal("0.25"))
    assert model.amount(10, Decimal("0.5")) == Decimal("0.25")


def test_zero_fee_model():
    assert ZeroFeeModel().amount(9999, Decimal("0.5")) == Decimal("0")


@pytest.mark.parametrize("price", ["0.01", "0.25", "0.50", "0.75", "0.99"])
def test_kalshi_fee_never_exceeds_two_cents_per_contract(price):
    model = KalshiFeeModel()
    qty = 1000
    per_contract = model.amount(qty, Decimal(price)) / qty
    assert per_contract <= Decimal("0.0175") + Decimal("0.0001")

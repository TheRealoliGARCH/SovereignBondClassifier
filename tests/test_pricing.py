import numpy as np
import pytest
from sovereign_bond_classifier.pricing import continuous_bond_price, continuous_discount_factor, continuous_duration


def test_flat_rate_discount_factor():
    got = continuous_discount_factor(0.0, 2.0, lambda _: 0.05, integration_steps=200)
    assert got == pytest.approx(np.exp(-0.10), rel=1e-8)


def test_zero_coupon_price_and_duration():
    rate, maturity = 0.04, 5.0
    price = continuous_bond_price(0.0, maturity, lambda _: 0.0, 100.0, lambda _: rate)
    duration = continuous_duration(0.0, maturity, lambda _: 0.0, 100.0, lambda _: rate)
    assert price == pytest.approx(100.0 * np.exp(-rate * maturity), rel=1e-5)
    assert duration == pytest.approx(maturity, rel=1e-5)


def test_maturity_now():
    assert continuous_bond_price(1, 1, lambda _: 0, 100, lambda _: 0.03) == 100
    assert continuous_duration(1, 1, lambda _: 0, 100, lambda _: 0.03) == 0


def test_invalid_time_order():
    with pytest.raises(ValueError):
        continuous_discount_factor(2, 1, lambda _: 0.02)

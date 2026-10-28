"""Continuous-time bond valuation helpers using deterministic discounting."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

RateFunction = Callable[[float], float]


def _validate_time(t: float, T: float) -> None:
    if not np.isfinite(t) or not np.isfinite(T):
        raise ValueError("t and T must be finite.")
    if T < t:
        raise ValueError("Maturity must be at or after valuation time.")


def _discount_grid(t: float, grid: np.ndarray, rate: RateFunction) -> np.ndarray:
    rates = np.asarray([rate(float(u)) for u in grid], dtype=float)
    if not np.all(np.isfinite(rates)):
        raise ValueError("Rate function returned non-finite values.")
    cumulative = np.zeros(len(grid), dtype=float)
    if len(grid) > 1:
        increments = 0.5 * (rates[:-1] + rates[1:]) * np.diff(grid)
        cumulative[1:] = np.cumsum(increments)
    return np.exp(-cumulative)


def continuous_discount_factor(
    t: float, s: float, rate: RateFunction, *, integration_steps: int = 1000
) -> float:
    """Return exp(-integral_t^s r(u) du) using trapezoidal integration."""
    _validate_time(t, s)
    if integration_steps < 2:
        raise ValueError("integration_steps must be at least 2.")
    if t == s:
        return 1.0
    grid = np.linspace(t, s, integration_steps)
    rates = np.asarray([rate(float(u)) for u in grid], dtype=float)
    if not np.all(np.isfinite(rates)):
        raise ValueError("Rate function returned non-finite values.")
    return float(np.exp(-np.trapezoid(rates, grid)))


def continuous_bond_price(
    t: float,
    maturity: float,
    coupon_flow: RateFunction,
    terminal_payment: float,
    rate: RateFunction,
    *,
    integration_steps: int = 2000,
) -> float:
    """Price continuous coupons plus terminal payment under deterministic discounting.

    Rates are continuously compounded annual rates and times are measured in years.
    This helper is not a stochastic defaultable-bond pricing model.
    """
    _validate_time(t, maturity)
    if integration_steps < 2:
        raise ValueError("integration_steps must be at least 2.")
    if not np.isfinite(terminal_payment):
        raise ValueError("terminal_payment must be finite.")
    if maturity == t:
        return float(terminal_payment)
    grid = np.linspace(t, maturity, integration_steps)
    coupons = np.asarray([coupon_flow(float(s)) for s in grid], dtype=float)
    if not np.all(np.isfinite(coupons)):
        raise ValueError("coupon_flow returned non-finite values.")
    discounts = _discount_grid(t, grid, rate)
    return float(np.trapezoid(coupons * discounts, grid) + terminal_payment * discounts[-1])


def continuous_duration(
    t: float,
    maturity: float,
    coupon_flow: RateFunction,
    terminal_payment: float,
    rate: RateFunction,
    *,
    integration_steps: int = 2000,
) -> float:
    """Return duration under a parallel shift of the discount curve."""
    _validate_time(t, maturity)
    if integration_steps < 2:
        raise ValueError("integration_steps must be at least 2.")
    if maturity == t:
        return 0.0
    grid = np.linspace(t, maturity, integration_steps)
    coupons = np.asarray([coupon_flow(float(s)) for s in grid], dtype=float)
    if not np.all(np.isfinite(coupons)):
        raise ValueError("coupon_flow returned non-finite values.")
    discounts = _discount_grid(t, grid, rate)
    price = float(np.trapezoid(coupons * discounts, grid) + terminal_payment * discounts[-1])
    if not np.isfinite(price) or price <= 0:
        raise ValueError("Duration requires a finite, strictly positive price.")
    weighted_pv = float(
        np.trapezoid((grid - t) * coupons * discounts, grid)
        + (maturity - t) * terminal_payment * discounts[-1]
    )
    return weighted_pv / price

"""Black-Scholes-Merton prices, Greeks and implied volatility.

This is the baseline every other method is compared with. Heston with vol-of-vol -> 0
and v0 = theta must collapse to these formulas, which makes it the first sanity check.

Conventions used everywhere in the package:
    S: spot, K: strike, T: years to expiry, r: risk-free rate, q: dividend yield,
    sigma: volatility (annualised, e.g. 0.2), call: True for a call, False for a put.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def _d1_d2(S, K, T, r, q, sigma):
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrtT)
    return d1, d1 - sigma * sqrtT


def bs_price(S, K, T, r, sigma, q=0.0, call=True):
    """Black-Scholes price. Works on numpy arrays (broadcasts)."""
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    df_r, df_q = np.exp(-r * T), np.exp(-q * T)
    if call:
        return S * df_q * norm.cdf(d1) - K * df_r * norm.cdf(d2)
    return K * df_r * norm.cdf(-d2) - S * df_q * norm.cdf(-d1)


def bs_greeks(S, K, T, r, sigma, q=0.0, call=True) -> dict:
    """Analytic delta, gamma, vega (per 1.00 of vol), theta (per year) and rho."""
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    df_r, df_q = np.exp(-r * T), np.exp(-q * T)
    pdf = norm.pdf(d1)
    gamma = df_q * pdf / (S * sigma * np.sqrt(T))
    vega = S * df_q * pdf * np.sqrt(T)
    common = -S * df_q * pdf * sigma / (2 * np.sqrt(T))
    if call:
        delta = df_q * norm.cdf(d1)
        theta = common - r * K * df_r * norm.cdf(d2) + q * S * df_q * norm.cdf(d1)
        rho = K * T * df_r * norm.cdf(d2)
    else:
        delta = -df_q * norm.cdf(-d1)
        theta = common + r * K * df_r * norm.cdf(-d2) - q * S * df_q * norm.cdf(-d1)
        rho = -K * T * df_r * norm.cdf(-d2)
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


def implied_vol(price, S, K, T, r, q=0.0, call=True, tol=1e-10, max_iter=50):
    """Implied volatility of one option price.

    Newton-Raphson first (quadratic convergence, uses vega as the derivative), started
    from the Brenner-Subrahmanyam guess. If Newton steps outside (0, 5] or vega is tiny
    (deep in/out of the money), fall back to Brent's method on a bracket, which always
    converges because the BS price is monotonic in sigma.

    Returns nan if the price is outside the no-arbitrage bounds.
    """
    df_r, df_q = np.exp(-r * T), np.exp(-q * T)
    fwd_intrinsic = max(S * df_q - K * df_r, 0.0) if call else max(K * df_r - S * df_q, 0.0)
    upper = S * df_q if call else K * df_r
    if not (fwd_intrinsic - 1e-12 <= price < upper):
        return np.nan
    if price - fwd_intrinsic < 1e-14:
        return 0.0

    sigma = max(np.sqrt(2 * np.pi / T) * price / S, 0.05)  # Brenner-Subrahmanyam
    for _ in range(max_iter):
        diff = bs_price(S, K, T, r, sigma, q, call) - price
        if abs(diff) < tol:
            return float(sigma)
        vega = bs_greeks(S, K, T, r, sigma, q, call)["vega"]
        if vega < 1e-8:
            break
        sigma -= diff / vega
        if not (0.0 < sigma <= 5.0):
            break

    f = lambda s: bs_price(S, K, T, r, s, q, call) - price
    try:
        return float(brentq(f, 1e-6, 5.0, xtol=tol, maxiter=200))
    except ValueError:
        return np.nan

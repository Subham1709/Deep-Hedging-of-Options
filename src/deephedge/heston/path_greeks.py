"""Heston put/call prices and Greeks for many market states at once, by the COS method.

The benchmark hedges need the Heston delta and vega at every hedging date on every test
path: 30 dates x 100,000 paths = 3 million option valuations. The engine's `cos_price`
prices many strikes for one (S, v); here it is the other way round, one strike for many
(S, v), so the COS sum is laid out as a (paths x N) matrix.

Two facts make this cheap.

1. The Heston characteristic function of the log-return over tau is affine in the current
   variance v:  phi(u; v) = exp(C(u, tau) + D(u, tau) v)  (with r = q = 0 here).
   C and D depend only on tau, so they are computed once per date and shared by all paths.
2. Because of (1), every cumulant of the log-return is affine in v too. Evaluating the
   cumulants at v = 0 and v = 1 gives them for every path, so each path gets its own COS
   truncation interval [a, b] sized to its own distribution at no extra cost.

Greeks come from differentiating the COS series itself (a and b held fixed):

    d/dx  multiplies term k by  i u_k        (x = log S/K, so delta = (1/S) dV/dx)
    d/dv  multiplies term k by  D(u_k)       (the variance sensitivity, "vega" in v)
"""
from __future__ import annotations

import numpy as np

from .model import HestonParams, cumulants


def _C_D(u, tau, p: HestonParams):
    """C(u, tau) and D(u, tau) of the Heston cf of log(S_tau/S_0), r = q = 0.

    Same "little Heston trap" form as model.heston_cf.
    """
    k, th, s, rho = p.kappa, p.theta, p.sigma, p.rho
    iu = 1j * u
    beta = k - rho * s * iu
    d = np.sqrt(beta**2 + s**2 * (iu + u**2))
    g = (beta - d) / (beta + d)
    e = np.exp(-d * tau)
    C = (k * th / s**2) * ((beta - d) * tau - 2.0 * np.log((1.0 - g * e) / (1.0 - g)))
    D = ((beta - d) / s**2) * (1.0 - e) / (1.0 - g * e)
    return C, D


def heston_put_greeks(S, v, K, tau, p: HestonParams, N=256, L=10.0, chunk=20_000):
    """European put price, dP/dS and dP/dv for arrays S, v sharing one strike and tau.

    Zero rates and dividends. Returns three float64 arrays shaped like S.
    Call: price + S - K, delta + 1, same dP/dv (put-call parity).
    """
    S = np.asarray(S, dtype=float).ravel()
    v = np.maximum(np.asarray(v, dtype=float).ravel(), 0.0)
    price, dS, dv = (np.empty_like(S) for _ in range(3))

    # cumulants are affine in v: c(v) = c(0) + v (c(1) - c(0))
    p0 = HestonParams(0.0, p.kappa, p.theta, p.sigma, p.rho)
    p1 = HestonParams(1.0, p.kappa, p.theta, p.sigma, p.rho)
    c0, c1_ = np.array(cumulants(tau, p0)), np.array(cumulants(tau, p1))

    k = np.arange(N)[None, :]
    for lo in range(0, S.size, chunk):
        sl = slice(lo, lo + chunk)
        Sc, vc = S[sl, None], v[sl, None]
        cm1, cm2, cm4 = (c0[j] + vc * (c1_[j] - c0[j]) for j in range(3))
        width = L * np.sqrt(np.abs(cm2) + np.sqrt(np.abs(cm4)))
        x = np.log(Sc / K)
        a, b = x + cm1 - width, x + cm1 + width

        u = k * np.pi / (b - a)                      # (n, N): each path has its own grid
        C, D = _C_D(u, tau, p)
        top = np.minimum(b, 0.0)
        w = u
        chi = (np.cos(w * (top - a)) * np.exp(top) - np.exp(a)
               + w * np.sin(w * (top - a)) * np.exp(top)) / (1 + w**2)
        psi = np.where(k == 0, top - a, np.sin(w * (top - a)) / np.where(k == 0, 1, w))
        Vk = np.where(top > a, 2.0 / (b - a) * (psi - chi), 0.0)  # coeffs of (1 - e^y)^+
        Vk[:, 0] *= 0.5                                            # the ' in sum'

        base = np.exp(C + D * vc + 1j * u * (x - a))
        price[sl] = K * (base.real * Vk).sum(axis=1)
        dS[sl] = K * ((1j * u * base).real * Vk).sum(axis=1) / Sc[:, 0]
        dv[sl] = K * ((D * base).real * Vk).sum(axis=1)
    return price, dS, dv


def variance_swap_level(v, tau, p: HestonParams):
    """E[ int_t^{t+tau} v_s ds | v_t = v ]: the fair (variance-unit) value of the floating
    leg of a variance swap still to accrue. Exact for the CIR variance process:

        theta tau + (v - theta) (1 - e^{-kappa tau}) / kappa
    """
    k, th = p.kappa, p.theta
    return th * tau + (np.asarray(v) - th) * (-np.expm1(-k * tau)) / k


def variance_swap_dv(tau, p: HestonParams):
    """d/dv of variance_swap_level: (1 - e^{-kappa tau}) / kappa."""
    return -np.expm1(-p.kappa * tau) / p.kappa

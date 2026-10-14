"""The Heston (1993) stochastic volatility model and its characteristic function.

Under the risk-neutral measure:

    dS_t = (r - q) S_t dt + sqrt(v_t) S_t dW1_t
    dv_t = kappa (theta - v_t) dt + sigma sqrt(v_t) dW2_t,    dW1 dW2 = rho dt

    v0     initial variance           (vol = sqrt(v0), e.g. 0.04 -> 20%)
    kappa  speed of mean reversion    (half-life of a variance shock = ln 2 / kappa)
    theta  long-run variance
    sigma  volatility of variance ("vol of vol"): controls smile curvature
    rho    spot-variance correlation: controls skew (equities: rho < 0)

Everything Fourier-based in this package needs only one function: the characteristic
function of log(S_T), which Heston has in closed form.
"""
from __future__ import annotations

from dataclasses import astuple, dataclass

import numpy as np


@dataclass(frozen=True)
class HestonParams:
    v0: float
    kappa: float
    theta: float
    sigma: float
    rho: float

    def feller(self) -> float:
        """2 kappa theta / sigma^2. If >= 1 the variance can never touch zero.

        Calibrated equity parameters usually violate it (< 1), which is exactly why the
        Monte Carlo and PDE schemes have to deal carefully with v = 0.
        """
        return 2 * self.kappa * self.theta / self.sigma**2

    def as_array(self) -> np.ndarray:
        return np.array(astuple(self), dtype=float)


def heston_cf(u, T, p: HestonParams, r=0.0, q=0.0):
    """E[exp(i u log(S_T / S_0))] under the risk-neutral measure.

    Uses the "little Heston trap" form of Albrecher, Mayer, Schoutens & Tistaert (2007).
    Heston's original formula has g -> 1/g, which makes the complex log jump branches
    for long maturities, silently giving wrong prices. This form is continuous in u.

    u may be a complex numpy array.
    """
    u = np.asarray(u, dtype=complex)
    k, th, s, rho, v0 = p.kappa, p.theta, p.sigma, p.rho, p.v0
    iu = 1j * u
    beta = k - rho * s * iu
    d = np.sqrt(beta**2 + s**2 * (iu + u**2))
    g = (beta - d) / (beta + d)
    e = np.exp(-d * T)
    C = (k * th / s**2) * ((beta - d) * T - 2.0 * np.log((1.0 - g * e) / (1.0 - g)))
    D = ((beta - d) / s**2) * (1.0 - e) / (1.0 - g * e)
    return np.exp(iu * (r - q) * T + C + D * v0)


def cumulants(T, p: HestonParams, r=0.0, q=0.0, h=0.05):
    """Cumulants c1 (mean), c2 (variance) and c4 of log(S_T/S_0), read off the
    characteristic function by finite differences of its log:

        log phi(u) = i c1 u - c2 u^2 / 2 - i c3 u^3 / 6 + c4 u^4 / 24 + ...

    Used to choose the truncation range of the COS method. c4 matters: Heston has fat
    tails when vol-of-vol is high, and a range based on the variance alone is too narrow.
    """
    lp = lambda u: np.log(heston_cf(u, T, p, r, q))
    l0, lp1, lm1, lp2, lm2 = lp(0.0), lp(h), lp(-h), lp(2 * h), lp(-2 * h)
    c1 = ((lp1 - lm1) / (2j * h)).real
    c2 = -((lp1 - 2 * l0 + lm1) / h**2).real
    c4 = ((lp2 - 4 * lp1 + 6 * l0 - 4 * lm1 + lm2) / h**4).real
    return c1, c2, c4

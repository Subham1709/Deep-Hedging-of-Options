"""Monte Carlo simulation of Heston paths.

Two discretisations of the variance process:

    euler_full_truncation  Euler on v, but use max(v, 0) inside drift and diffusion
                           (Lord, Koekkoek & van Dijk, 2010). Simple; biased at large dt.
    andersen_qe            Andersen's (2008) Quadratic-Exponential scheme: sample v_{t+dt}
                           from a distribution that matches the exact conditional mean
                           and variance. Much smaller bias at the same dt; the standard in
                           practice.

Both simulate log S with the correlated Brownian motion, so the simulated S_T can be used
for any payoff, including path-dependent ones (pass return_paths=True).

The deep hedging project reuses `simulate` as its market simulator.
"""
from __future__ import annotations

import numpy as np

from .model import HestonParams


def simulate(S0, T, r, p: HestonParams, n_paths=100_000, n_steps=100, q=0.0,
             scheme="qe", seed=None, antithetic=True, return_paths=False):
    """Simulate (S, v). Returns S_T and v_T, or full paths of shape (n_steps+1, n_paths)."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    half = n_paths // 2 if antithetic else n_paths
    k, th, s, rho = p.kappa, p.theta, p.sigma, p.rho

    x = np.full(n_paths, np.log(S0))
    v = np.full(n_paths, p.v0)
    if return_paths:
        S_path = np.empty((n_steps + 1, n_paths)); S_path[0] = S0
        v_path = np.empty((n_steps + 1, n_paths)); v_path[0] = p.v0

    # constants for QE
    e = np.exp(-k * dt)
    K0 = -rho * k * th * dt / s
    K1 = 0.5 * dt * (k * rho / s - 0.5) - rho / s
    K2 = 0.5 * dt * (k * rho / s - 0.5) + rho / s
    K3 = 0.5 * dt * (1 - rho**2)

    def normals():
        z = rng.standard_normal(half)
        return np.concatenate([z, -z]) if antithetic else z

    def uniforms():
        u = rng.random(half)
        return np.concatenate([u, 1 - u]) if antithetic else u

    for t in range(n_steps):
        z1 = normals()
        if scheme == "euler":
            z2 = rho * z1 + np.sqrt(1 - rho**2) * normals()
            vp = np.maximum(v, 0.0)
            x = x + (r - q - 0.5 * vp) * dt + np.sqrt(vp * dt) * z1
            v = v + k * (th - vp) * dt + s * np.sqrt(vp * dt) * z2
        elif scheme == "qe":
            # exact conditional mean and variance of v_{t+dt} given v_t (CIR process)
            m = th + (v - th) * e
            s2 = v * s**2 * e / k * (1 - e) + th * s**2 / (2 * k) * (1 - e) ** 2
            psi = s2 / m**2
            v_new = np.empty_like(v)
            quad = psi <= 1.5
            # psi small: v_new = a (b + Z)^2, a scaled non-central chi-square with 1 dof
            pq = psi[quad]
            b2 = 2 / pq - 1 + np.sqrt(2 / pq) * np.sqrt(2 / pq - 1)
            a = m[quad] / (1 + b2)
            zv = normals()[quad]
            v_new[quad] = a * (np.sqrt(b2) + zv) ** 2
            # psi large: point mass at 0 plus an exponential tail
            ex = ~quad
            pe = psi[ex]
            prob0 = (pe - 1) / (pe + 1)
            beta = (1 - prob0) / m[ex]
            uv = uniforms()[ex]
            v_new[ex] = np.where(uv <= prob0, 0.0,
                                 np.log((1 - prob0) / np.maximum(1 - uv, 1e-300)) / beta)
            # log-spot step: integrate v over the step with the trapezoid rule and use the
            # fact that int sqrt(v) dW2 can be backed out of the variance SDE
            x = x + (r - q) * dt + K0 + K1 * v + K2 * v_new + np.sqrt(K3 * (v + v_new)) * z1
            v = v_new
        else:
            raise ValueError(scheme)
        if return_paths:
            S_path[t + 1] = np.exp(x); v_path[t + 1] = np.maximum(v, 0.0)

    if return_paths:
        return S_path, v_path
    return np.exp(x), np.maximum(v, 0.0)


def mc_price(S0, K, T, r, p: HestonParams, q=0.0, call=True, n_paths=200_000,
             n_steps=100, scheme="qe", seed=0, control_variate=True):
    """European option by Monte Carlo. Returns (price, standard error).

    Control variate: the discounted terminal stock has a known mean, S0 e^{-qT}. Payoffs
    are strongly correlated with S_T, so regressing them on S_T removes most of the noise
    for free. For in-the-money calls the variance drops by 80-95%.
    """
    ST, _ = simulate(S0, T, r, p, n_paths, n_steps, q, scheme, seed)
    df = np.exp(-r * T)
    payoff = df * (np.maximum(ST - K, 0) if call else np.maximum(K - ST, 0))
    if control_variate:
        cv = df * ST
        c = np.cov(payoff, cv)
        beta = c[0, 1] / c[1, 1]
        payoff = payoff - beta * (cv - S0 * np.exp(-q * T))
    # antithetic pairs are not independent: average each pair before the std error
    n = len(payoff) // 2
    pairs = 0.5 * (payoff[:n] + payoff[n:2 * n])
    return float(pairs.mean()), float(pairs.std(ddof=1) / np.sqrt(n))

"""Market data for deep hedging: simulated Heston and GBM paths, and real NIFTY 50 windows.

The hedging problem (same set-up as Buehler, Gonon, Teichmann & Wood, "Deep Hedging", 2019):
a dealer sells one European call (strike K = S0 = 100, 30 trading days) and hedges it daily
until expiry. Rates and dividends are zero, so prices need no discounting.

Each dataset is a set of paths on the hedging dates t = 0, 1, ..., 30 (in trading days).
Everything the network may look at on date t, and every benchmark hedge, is stored per path
and date, so training only has to read arrays.

Time convention: one step = one trading day = 1/252 year, so T = 30/252. The Heston
parameters are annualised and are used as they are.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from .heston.black_scholes import bs_price, implied_vol
from .heston.model import HestonParams
from .heston.monte_carlo import simulate
from .heston.path_greeks import (heston_put_greeks, variance_swap_dv,
                                 variance_swap_level)

# Calibrated to SPX option quotes of 12 Oct 2026 in Heston_Pricing_Engine (results/calibration.md)
SPX_HESTON = HestonParams(v0=0.0136, kappa=7.41, theta=0.0421, sigma=1.89, rho=-0.65)


@dataclass
class DataConfig:
    S0: float = 100.0
    K: float = 100.0
    n_steps: int = 30                 # hedging dates, one per trading day
    days_per_year: int = 252
    substeps: int = 4                 # Heston simulated at 4 steps per day, observed daily
    heston: HestonParams = SPX_HESTON
    n_paths: dict = field(default_factory=lambda: {"train": 100_000, "val": 20_000,
                                                   "test": 100_000})
    seeds: dict = field(default_factory=lambda: {
        "heston": {"train": 1, "val": 2, "test": 3},
        "gbm": {"train": 11, "val": 12, "test": 13}})

    @property
    def T(self):
        return self.n_steps / self.days_per_year

    @property
    def taus(self):
        """Time to expiry (years) on each hedging date, shape (n_steps + 1,)."""
        return (self.n_steps - np.arange(self.n_steps + 1)) / self.days_per_year


# ------------------------------------------------------------------- reference prices

def heston_call(S, v, tau, cfg: DataConfig, N=256):
    """Heston call price, delta dC/dS and variance sensitivity dC/dv, via put-call parity."""
    put, dS, dv = heston_put_greeks(S, v, cfg.K, tau, cfg.heston, N=N)
    return put + np.ravel(S) - cfg.K, dS + 1.0, dv


def reference_prices(cfg: DataConfig) -> dict:
    """Option price at t = 0 under Heston, its BS implied vol, and the GBM price at that vol."""
    p0 = float(heston_call([cfg.S0], [cfg.heston.v0], cfg.T, cfg)[0][0])
    iv = float(implied_vol(p0, cfg.S0, cfg.K, cfg.T, 0.0))
    return {"heston_price": p0, "heston_atm_iv": iv,
            "gbm_sigma": iv, "gbm_price": float(bs_price(cfg.S0, cfg.K, cfg.T, 0.0, iv))}


def bs_delta(S, tau, K, sigma):
    """Call delta; at expiry (tau = 0) it is the indicator S > K."""
    S, tau = np.asarray(S, float), np.asarray(tau, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(S / K) + 0.5 * sigma**2 * tau) / (sigma * np.sqrt(tau))
    return np.where(tau > 0, norm.cdf(d1), (S > K).astype(float))


# ------------------------------------------------------------------- simulated markets

def simulate_gbm(n, cfg: DataConfig, sigma, seed):
    """GBM paths with zero drift (risk-neutral, r = 0), exact log-normal steps."""
    rng = np.random.default_rng(seed)
    dt = 1.0 / cfg.days_per_year
    half = n // 2
    z = rng.standard_normal((half, cfg.n_steps))
    z = np.concatenate([z, -z])                       # antithetic pairs
    logret = -0.5 * sigma**2 * dt + sigma * np.sqrt(dt) * z
    S = cfg.S0 * np.exp(np.concatenate([np.zeros((n, 1)), np.cumsum(logret, axis=1)], axis=1))
    return S


def simulate_heston(n, cfg: DataConfig, seed):
    """Heston paths observed daily: S, v and the variance swap price.

    Simulated with Andersen QE at `substeps` steps per day. The fine grid is used for the
    realised variance int_0^t v ds (trapezoid rule), which the variance swap needs.

    Variance swap (second hedging instrument, as in Buehler et al.): it pays int_0^T v_s ds
    at T. Its price on date t is what has accrued plus the expected remainder,
        VS_t = int_0^t v_s ds + E[int_t^T v_s ds | v_t],
    a martingale under the risk-neutral measure, so hedging with it costs nothing on average.
    """
    m = cfg.substeps
    S_f, v_f = simulate(cfg.S0, cfg.T, 0.0, cfg.heston, n_paths=n, n_steps=cfg.n_steps * m,
                        scheme="qe", seed=seed, antithetic=True, return_paths=True)
    dt_f = cfg.T / (cfg.n_steps * m)
    acc_f = np.concatenate([np.zeros((1, n)),
                            np.cumsum(0.5 * (v_f[1:] + v_f[:-1]) * dt_f, axis=0)])
    S, v, acc = (a[::m].T.copy() for a in (S_f, v_f, acc_f))     # (n, n_steps + 1)
    vs = acc + variance_swap_level(v, cfg.taus[None, :], cfg.heston)
    return S, v, vs


def heston_benchmarks(S, v, cfg: DataConfig, iv):
    """Benchmark hedge ratios on every date before expiry, shape (n, n_steps).

    bs_delta      Black-Scholes delta at the t = 0 implied vol (what a desk without a
                  stochastic-vol model would do)
    heston_delta  dC/dS: the Heston model delta, holding v fixed
    mv_delta      minimum-variance delta, dC/dS + (rho sigma / S) dC/dv: the stock-only
                  hedge that also offsets the part of a vol move that is correlated with
                  spot. The best Heston hedge when only the stock can be traded.
    vs_units      units of variance swap that neutralise dC/dv: with heston_delta in the
                  stock, this is the complete hedge (exact in continuous time)
    """
    p = cfg.heston
    n = S.shape[0]
    out = {k: np.empty((n, cfg.n_steps), np.float32)
           for k in ("bs_delta", "heston_delta", "mv_delta", "vs_units")}
    for t in range(cfg.n_steps):
        tau = cfg.taus[t]
        _, dS, dv = heston_call(S[:, t], v[:, t], tau, cfg)
        out["bs_delta"][:, t] = bs_delta(S[:, t], tau, cfg.K, iv)
        out["heston_delta"][:, t] = dS
        out["mv_delta"][:, t] = dS + p.rho * p.sigma / S[:, t] * dv
        out["vs_units"][:, t] = dv / variance_swap_dv(tau, p)
    return out


# ------------------------------------------------------------------- real NIFTY 50 paths

def nifty_windows(closes: pd.Series, cfg: DataConfig, vol_lookback=60):
    """Every run of n_steps + 1 consecutive NIFTY 50 closes, rescaled to start at S0.

    Used only as an out-of-sample test: a model trained on simulated paths is run on real
    ones. Windows overlap (one starts every trading day), so neighbouring windows share
    most of their moves; count non-overlapping windows when judging significance.

    hist_vol is the annualised realised vol of the `vol_lookback` daily log returns before
    the window starts: information a trader had on the start date, used to price the
    option and set the Black-Scholes hedge (no look-ahead).
    """
    closes = closes.dropna().sort_index()
    x = closes.to_numpy(float)
    lr = np.diff(np.log(x))
    L = cfg.n_steps + 1
    starts = np.arange(vol_lookback, len(x) - L + 1)
    S = np.stack([x[s:s + L] / x[s] * cfg.S0 for s in starts])
    hv = np.array([lr[s - vol_lookback:s].std(ddof=1) * np.sqrt(cfg.days_per_year)
                   for s in starts])
    dates = closes.index[starts]
    return S, hv, np.array(dates, dtype="datetime64[D]")


def save_npz(path: Path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k: (a.astype(np.float32) if a.dtype == np.float64 else a)
                                 for k, a in arrays.items()})


def config_json(cfg: DataConfig, extra: dict) -> str:
    d = asdict(cfg)
    d["heston"] = asdict(cfg.heston)
    d["T"] = cfg.T
    d.update(extra)
    return json.dumps(d, indent=2)

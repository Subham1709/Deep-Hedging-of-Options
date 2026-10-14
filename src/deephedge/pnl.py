"""Hedging P&L and risk measures (numpy; the training code will use a torch twin of these).

The dealer sells the call for a premium p0 and holds delta_t shares over (t, t+1]:

    PnL = p0 - max(S_T - K, 0) + sum_t delta_t (S_{t+1} - S_t)
             [+ sum_t units_t (VS_{t+1} - VS_t)]               second instrument, if used
             - cost * sum_t |delta_t - delta_{t-1}| S_t         proportional costs
    with delta_{-1} = 0 and the position closed at T (delta_T = 0), which also costs.

Costs are a fraction of the traded notional: cost = 0.001 means 10 bps per trade.
"""
from __future__ import annotations

import numpy as np


def hedge_pnl(S, delta, K, p0=0.0, cost=0.0, vs=None, units=None, vs_cost=0.0):
    """S: (n, T+1); delta, units: (n, T). Returns P&L per path, shape (n,)."""
    S = np.asarray(S, float)
    delta = np.asarray(delta, float)
    gains = (delta * np.diff(S, axis=1)).sum(axis=1)
    pos = np.concatenate([np.zeros((len(S), 1)), delta, np.zeros((len(S), 1))], axis=1)
    tc = cost * (np.abs(np.diff(pos, axis=1)) * S).sum(axis=1)
    if vs is not None:
        units = np.asarray(units, float)
        gains += (units * np.diff(vs, axis=1)).sum(axis=1)
        upos = np.concatenate([np.zeros((len(S), 1)), units, np.zeros((len(S), 1))], axis=1)
        tc += vs_cost * np.abs(np.diff(upos, axis=1)).sum(axis=1)
    return p0 - np.maximum(S[:, -1] - K, 0.0) + gains - tc


def cvar(pnl, alpha=0.95):
    """Expected shortfall: average loss in the worst (1 - alpha) share of paths.

    Reported as a positive number (a loss). This is the risk measure the network will
    minimise; Buehler et al. use it too (alpha = 0.5 and 0.99 in their experiments).
    """
    loss = -np.asarray(pnl)
    q = np.quantile(loss, alpha)
    return float(loss[loss >= q].mean())


def summary(pnl, alpha=0.95):
    pnl = np.asarray(pnl)
    return {"mean": float(pnl.mean()), "std": float(pnl.std()),
            f"cvar{int(alpha * 100)}": cvar(pnl, alpha)}

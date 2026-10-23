"""Numpy pieces the web app needs: run a trained network, the benchmark hedges, summaries.

Mirrors deephedge.model.rollout and scripts/04_evaluate.py without torch, so the hosted app
only needs numpy, scipy and pandas.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deephedge.data import DataConfig, bs_delta, heston_benchmarks  # noqa: E402
from deephedge.pnl import cvar, hedge_pnl  # noqa: E402

# same constants as deephedge/model.py
MONEYNESS_SCALE, VOL_SCALE, VS_UNITS_SCALE = 0.05, 0.15, 300.0
K = 100.0
COSTS = {"0 bp": 0.0, "5 bp": 0.0005, "20 bp": 0.002}
MARKETS = {"GBM, stock only": ("gbm", False),
           "Heston, stock only": ("heston", False),
           "Heston, stock + variance swap": ("heston", True)}
# no-trade band widths tuned on the validation sets by scripts/04_evaluate.py (results.md)
BANDS = {("gbm", 0.0): 0.01, ("gbm", 0.0005): 0.02, ("gbm", 0.002): 0.05,
         ("heston", 0.0): 0.01, ("heston", 0.0005): 0.02, ("heston", 0.002): 0.05}


def model_name(market, cost, use_vs):
    return f"{market}_{'stock_vs' if use_vs else 'stock'}_{cost * 1e4:g}bp"


def forward(weights, name, x):
    """The network: Linear -> ReLU, three times, then a last Linear."""
    n_layers = sum(1 for k in weights if k.startswith(name + "/W"))
    for i in range(n_layers):
        x = x @ weights[f"{name}/W{i}"].T + weights[f"{name}/b{i}"]
        if i < n_layers - 1:
            x = np.maximum(x, 0.0)
    return x


def features(S, sqrt_v=None, vol_rescale=None):
    """Network inputs on dates 0..29: log-moneyness, time left / T, [sqrt(v)]."""
    n_steps = S.shape[1] - 1
    m = np.log(S[:, :-1] / K)
    if vol_rescale is not None:
        m = m * np.asarray(vol_rescale)[:, None]
    tau = np.broadcast_to((n_steps - np.arange(n_steps)) / n_steps, m.shape)
    f = [m / MONEYNESS_SCALE, tau]
    if sqrt_v is not None:
        f.append(sqrt_v[:, :-1] / VOL_SCALE)
    return np.stack(f, axis=-1)


def policy_positions(weights, name, S, sqrt_v=None, vol_rescale=None):
    """Roll the network along each path, feeding back yesterday's position.
    Returns (n, 30, n_instruments) in real units (shares; variance swap units)."""
    F = features(S, sqrt_v, vol_rescale)
    n_layers = sum(1 for k in weights if k.startswith(name + "/W"))
    n_inst = weights[f"{name}/b{n_layers - 1}"].shape[0]
    prev = np.zeros((len(S), n_inst))
    out = []
    for t in range(F.shape[1]):
        prev = forward(weights, name, np.concatenate([F[:, t], prev], axis=1))
        out.append(prev)
    pos = np.stack(out, axis=1)
    if n_inst == 2:
        pos[..., 1] *= VS_UNITS_SCALE
    return pos


def band(target, b):
    """No-trade band: keep yesterday's position while it is within b of the target delta,
    otherwise trade just to the edge of the band."""
    pos = np.empty_like(target)
    prev = np.zeros(len(target))
    for t in range(target.shape[1]):
        prev = np.clip(prev, target[:, t] - b, target[:, t] + b)
        pos[:, t] = prev
    return pos


def turnover(delta):
    z = np.zeros((len(delta), 1))
    return float(np.abs(np.diff(np.concatenate([z, delta, z], 1), axis=1)).sum(1).mean())


def summary_row(pnl, delta):
    return {"mean P&L": pnl.mean(), "std": pnl.std(), "CVaR95": cvar(pnl),
            "shares traded": turnover(delta)}


def hedge_all(weights, market, use_vs, cost, S, v=None, vs=None, iv=None, p0=None, bm=None):
    """Deep hedge and benchmarks on the same paths. Returns {label: (pnl, stock position)}
    and the network's full positions. bm: Heston benchmark hedges from heston_benchmarks
    (slow, so the app computes them once per set of paths and passes them in)."""
    cfg = DataConfig()
    name = model_name(market, cost, use_vs)
    sqrt_v = np.sqrt(v) if market == "heston" else None
    pos = policy_positions(weights, name, S, sqrt_v)
    out = {"deep hedge": (hedge_pnl(S, pos[..., 0], K, p0, cost,
                                    vs=vs if use_vs else None,
                                    units=pos[..., 1] if use_vs else None), pos[..., 0])}
    n = len(S)
    out["no hedge"] = (hedge_pnl(S, np.zeros((n, 30)), K, p0, cost), np.zeros((n, 30)))
    taus = cfg.taus[None, :-1]
    b = BANDS.get((market, cost))
    if market == "gbm":
        d = bs_delta(S[:, :-1], taus, K, iv)
        out["BS delta"] = hedge_pnl(S, d, K, p0, cost), d
        db = band(d, b)
        out[f"BS delta, band ±{b:g}"] = hedge_pnl(S, db, K, p0, cost), db
    else:
        if bm is None:
            bm = heston_benchmarks(S, v, cfg, iv)
        if use_vs:
            out["Heston delta + var swap"] = (hedge_pnl(S, bm["heston_delta"], K, p0, cost,
                                                        vs=vs, units=bm["vs_units"]),
                                              bm["heston_delta"])
        else:
            out["BS delta (t=0 vol)"] = hedge_pnl(S, bm["bs_delta"], K, p0, cost), bm["bs_delta"]
            mv = bm["mv_delta"].astype(float)
            out["min-variance delta"] = hedge_pnl(S, mv, K, p0, cost), mv
            mb = band(mv, b)
            out[f"min-var delta, band ±{b:g}"] = hedge_pnl(S, mb, K, p0, cost), mb
    return out, pos

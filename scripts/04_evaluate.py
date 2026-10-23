"""Evaluate the trained networks on the test sets and on real NIFTY 50 paths.

    python scripts/04_evaluate.py

Compares each network with the benchmarks at the same cost level, including a no-trade
band around the benchmark delta whose width is tuned on the validation set. Writes
results/results.md and figures.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deephedge.data import DataConfig, bs_delta, heston_benchmarks  # noqa: E402
from deephedge.heston.black_scholes import bs_price  # noqa: E402
from deephedge.model import HedgePolicy, market_features  # noqa: E402
from deephedge.pnl import cvar, hedge_pnl  # noqa: E402
from deephedge.train import evaluate, load_split, model_name  # noqa: E402

DATA, MODELS, RES = ROOT / "data", ROOT / "models", ROOT / "results"
COSTS = [0.0, 0.0005, 0.002]
BAND_GRID = [0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3]
K = 100.0


def band(target, b):
    """No-trade band: keep yesterday's position while it is within b of the target delta,
    otherwise trade just to the edge of the band. b = 0 is the plain delta hedge."""
    pos = np.empty_like(target)
    prev = np.zeros(len(target))
    for t in range(target.shape[1]):
        prev = np.clip(prev, target[:, t] - b, target[:, t] + b)
        pos[:, t] = prev
    return pos


def turnover(delta):
    """Average number of shares traded per path, including opening and closing."""
    z = np.zeros((len(delta), 1))
    return float(np.abs(np.diff(np.concatenate([z, delta, z], 1), axis=1)).sum(1).mean())


def tune_band(S_val, target_val, p0, cost, alpha=0.95):
    scores = [cvar(hedge_pnl(S_val, band(target_val, b), K, p0, cost), alpha) for b in BAND_GRID]
    return BAND_GRID[int(np.argmin(scores))]


def paired_bootstrap(pnl_a, pnl_b, n=200, alpha=0.95, seed=0):
    """Std error of CVaR(a) - CVaR(b), resampling the same paths for both."""
    rng = np.random.default_rng(seed)
    d = [cvar(pnl_a[i], alpha) - cvar(pnl_b[i], alpha)
         for i in (rng.integers(0, len(pnl_a), len(pnl_a)) for _ in range(n))]
    return float(np.std(d))


def load_policy(name):
    meta = json.loads((MODELS / f"{name}.json").read_text())
    n_feat = 3 if meta["market"] == "heston" else 2
    pol = HedgePolicy(n_feat, 2 if meta["use_vs"] else 1, meta["config"]["width"],
                      meta["config"]["depth"])
    pol.load_state_dict(torch.load(MODELS / f"{name}.pt"))
    return pol, meta


def row(label, pnl, delta, extra=""):
    return [label, f"{pnl.mean():+.3f}", f"{pnl.std():.3f}", f"{cvar(pnl):.3f}",
            f"{turnover(delta):.2f}", extra]


def table(rows, header):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(out + ["| " + " | ".join(r) + " |" for r in rows])


def main():
    meta = json.loads((DATA / "meta.json").read_text())
    ref = meta["reference"]
    cfg = DataConfig()
    iv = ref["heston_atm_iv"]
    RES.mkdir(exist_ok=True)
    taus = cfg.taus[None, :-1]
    HDR = ["hedge", "mean P&L", "std", "CVaR95", "shares traded", "note"]

    g = dict(np.load(DATA / "gbm_test.npz")); h = dict(np.load(DATA / "heston_test.npz"))
    gS, hS = g["S"].astype(float), h["S"].astype(float)
    gval = np.load(DATA / "gbm_val.npz")["S"].astype(float)
    hv_ = dict(np.load(DATA / "heston_val.npz"))
    hval_S = hv_["S"].astype(float)
    cache = DATA / "heston_val_benchmarks.npz"
    if not cache.exists():
        print("computing Heston Greeks on the validation set for band tuning ...", flush=True)
        np.savez_compressed(cache, **heston_benchmarks(hval_S, hv_["v"].astype(float), cfg, iv))
    hval_b = dict(np.load(cache))

    md = ["# Results", "",
          "Test sets: 100k fresh paths per market, never used for training or any choice. "
          "CVaR95 = average loss in the worst 5% of paths (lower is better). Mean P&L "
          "includes the premium (the model price, 1.6121). The band width of each "
          "no-trade band is tuned on the validation set. ± is one bootstrap standard "
          "error of the CVaR difference between the network and the best benchmark.", ""]
    summary_rows, store = [], {}

    for market, use_vs, title in (("gbm", False, "GBM market, stock only"),
                                  ("heston", False, "Heston market, stock only"),
                                  ("heston", True, "Heston market, stock + variance swap")):
        md += [f"## {title}", ""]
        for cost in COSTS:
            name = model_name(market, cost, use_vs)
            pol, mm = load_policy(name)
            S_t, F_t, V_t = load_split(DATA, market, "test", use_vs)
            p0 = mm["p0"]
            net_pnl, net_pos = evaluate(pol, S_t, F_t, V_t, p0, cost)
            store[name] = (net_pnl, net_pos)
            rows = [row("**deep hedge**", net_pnl, net_pos[..., 0],
                        f"{mm['epochs']} epochs")]
            bench = {}
            if market == "gbm":
                tgt, tgt_val = g["bs_delta"].astype(float), bs_delta(gval[:, :-1], taus, K, iv)
                bench["BS delta"] = hedge_pnl(gS, tgt, K, p0, cost), tgt
                b = tune_band(gval, tgt_val, p0, cost)
                d = band(tgt, b); bench[f"BS delta, band ±{b:g}"] = hedge_pnl(gS, d, K, p0, cost), d
            elif not use_vs:
                bench["BS delta (t=0 vol)"] = hedge_pnl(hS, h["bs_delta"], K, p0, cost), h["bs_delta"]
                tgt = h["mv_delta"].astype(float)
                bench["min-variance delta"] = hedge_pnl(hS, tgt, K, p0, cost), tgt
                b = tune_band(hval_S, hval_b["mv_delta"].astype(float), p0, cost)
                d = band(tgt, b); bench[f"min-var delta, band ±{b:g}"] = hedge_pnl(hS, d, K, p0, cost), d
            else:
                bench["Heston delta + var swap"] = (hedge_pnl(hS, h["heston_delta"], K, p0, cost,
                                                              vs=h["vs"], units=h["vs_units"]),
                                                    h["heston_delta"])
            for lbl, (p, dl) in bench.items():
                rows.append(row(lbl, p, dl))
            best_lbl = min(bench, key=lambda k: cvar(bench[k][0]))
            best_p = bench[best_lbl][0]
            se = paired_bootstrap(net_pnl, best_p)
            diff = cvar(net_pnl) - cvar(best_p)
            summary_rows.append([title, f"{cost * 1e4:g} bp", f"{cvar(net_pnl):.3f}",
                                 f"{cvar(best_p):.3f} ({best_lbl})",
                                 f"{diff:+.3f} ± {se:.3f}",
                                 f"{100 * diff / cvar(best_p):+.0f}%"])
            md += [f"**Costs {cost * 1e4:g} bps**", "", table(rows, HDR), ""]
            print(name, f"net {cvar(net_pnl):.3f} best {best_lbl} {cvar(best_p):.3f}", flush=True)

    md[4:4] = ["## Summary", "",
               table(summary_rows, ["market", "costs", "deep hedge CVaR95",
                                    "best benchmark CVaR95", "difference", "relative"]), ""]

    # real NIFTY 50: GBM-trained stock networks, moneyness rescaled by training vol / hist vol
    nf = dict(np.load(DATA / "nifty50_windows.npz"))
    S, hv = nf["S"].astype(float), nf["hist_vol"].astype(float)
    p0 = bs_price(100.0, K, cfg.T, 0.0, hv)
    tgt = bs_delta(S[:, :-1], taus, K, hv[:, None])
    St = torch.tensor(S)
    tau_frac = torch.tensor(cfg.taus[:-1] / cfg.T)
    F = market_features(St, tau_frac, vol_rescale=torch.tensor(iv / hv)).float()
    rows = []
    for cost in COSTS:
        pol, _ = load_policy(model_name("gbm", cost, False))
        net_pnl, net_pos = evaluate(pol, St.float(), F, None, torch.tensor(p0).float(), cost)
        rows.append(row(f"deep hedge, {cost * 1e4:g} bp", net_pnl, net_pos[..., 0]))
        rows.append(row(f"BS delta, {cost * 1e4:g} bp", hedge_pnl(S, tgt, K, p0, cost), tgt))
        # independent blocks: every 30th window
        nb = slice(0, None, 30)
        rows[-2][-1] = f"non-overlapping CVaR {cvar(net_pnl[nb]):.3f}"
        rows[-1][-1] = f"non-overlapping CVaR {cvar(hedge_pnl(S[nb], tgt[nb], K, p0[nb], cost)):.3f}"
    md += ["## Real NIFTY 50 paths (2010-2026)", "",
           f"{len(S)} overlapping 30-day windows. The networks trained on GBM are applied "
           "with log-moneyness rescaled by (training vol / 60-day historical vol), so a move "
           "of one standard deviation looks the same to the network as in training. Premium "
           "and BS delta use the same historical vol. Overlapping windows share most of their "
           "moves, so the last column repeats the CVaR on every 30th window (134 independent "
           "windows; noisy).", "",
           table(rows, HDR), ""]
    (RES / "results.md").write_text("\n".join(md))
    print("\n".join(md))

    # figures
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    i = int(np.argmin(np.abs(gS[:, -1] - 101)))          # a path ending near the money
    for cost, col in zip(COSTS, ("C0", "C1", "C3")):
        ax[0].step(range(30), store[model_name("gbm", cost, False)][1][i, :, 0], where="post",
                   color=col, label=f"deep hedge {cost * 1e4:g} bp")
    ax[0].step(range(30), g["bs_delta"][i], where="post", color="k", ls="--", label="BS delta")
    ax[0].set(title="Position along one GBM path", xlabel="trading day", ylabel="shares held")
    ax[0].legend(fontsize=8)
    # learned position vs moneyness at mid-life, GBM, 20 bp vs 0 bp
    m = np.linspace(-0.08, 0.08, 200)
    for cost, col in zip(COSTS, ("C0", "C1", "C3")):
        pol, _ = load_policy(model_name("gbm", cost, False))
        with torch.no_grad():
            for prev, ls in ((0.2, ":"), (0.8, "-")):
                f = torch.tensor(np.stack([m / 0.05, np.full_like(m, 0.5),
                                           np.full_like(m, prev)], 1)).float()
                ax[1].plot(m, pol.net(f)[:, 0].numpy(), color=col, ls=ls,
                           label=f"{cost * 1e4:g} bp, holding {prev}")
    ax[1].plot(m, bs_delta(100 * np.exp(m), cfg.T / 2, K, iv), "k--", label="BS delta")
    ax[1].set(title="Learned position vs moneyness (15 days left)", xlabel="log(S/K)")
    ax[1].legend(fontsize=7)
    name = model_name("heston", 0.002, False)
    for p, lbl, col in ((store[name][0], "deep hedge", "C3"),
                        (hedge_pnl(hS, h["mv_delta"], K, ref["heston_price"], 0.002),
                         "min-variance delta", "k")):
        ax[2].hist(p, bins=150, range=(-6, 2), density=True, histtype="step", color=col, label=lbl)
    ax[2].set(title="Heston, stock only, 20 bp: test P&L", yscale="log")
    ax[2].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(RES / "deep_hedge_results.png", dpi=130)

    fig, ax = plt.subplots(figsize=(7, 4))
    for f in sorted(MODELS.glob("*.json")):
        hist = json.loads(f.read_text())["history"]
        ax.plot([x["epoch"] for x in hist], [x["val_cvar"] for x in hist], lw=1, label=f.stem)
    ax.set(xlabel="epoch", ylabel="validation CVaR95", yscale="log", title="Training curves")
    ax.legend(fontsize=6, ncol=2); fig.tight_layout(); fig.savefig(RES / "training_curves.png", dpi=130)


if __name__ == "__main__":
    main()

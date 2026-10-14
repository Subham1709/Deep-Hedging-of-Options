"""Sanity checks on the prepared data, and the benchmark hedges the network has to beat.

    python scripts/02_check_data.py [--data data]

Writes results/data_checks.md and results/data_overview.png.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deephedge.data import bs_delta  # noqa: E402
from deephedge.heston.black_scholes import bs_price  # noqa: E402
from deephedge.pnl import hedge_pnl, summary  # noqa: E402

COSTS = [0.0, 0.0005, 0.002]   # 0, 5 and 20 bps of traded notional


def mean_se(x):
    x = np.asarray(x, float)
    n = len(x) // 2   # antithetic pairs are not independent: average each pair first
    pairs = 0.5 * (x[:n] + x[n:2 * n])
    return pairs.mean(), pairs.std(ddof=1) / np.sqrt(n)


def table(rows, header):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data"))
    args = ap.parse_args()
    d = Path(args.data)
    meta = json.loads((d / "meta.json").read_text())
    K, ref = meta["K"], meta["reference"]
    load = lambda name: dict(np.load(d / f"{name}.npz"))
    md = ["# Data checks", "",
          f"Option: European call, S0 = K = {K:g}, {meta['n_steps']} trading days, "
          f"daily hedging, zero rates.",
          f"Heston price at t = 0: **{ref['heston_price']:.4f}** "
          f"(Black-Scholes implied vol {ref['heston_atm_iv']:.2%}); "
          f"GBM uses that vol, so both markets charge the same premium.", ""]

    # 1. martingale and pricing checks: the simulator must reproduce the model prices
    rows = []
    for name in ("gbm", "heston"):
        for split in ("train", "val", "test"):
            a = load(f"{name}_{split}")
            S = a["S"].astype(float)
            m, se = mean_se(S[:, -1])
            pay_m, pay_se = mean_se(np.maximum(S[:, -1] - K, 0))
            price = ref["gbm_price"] if name == "gbm" else ref["heston_price"]
            row = [name, split, len(S), f"{m:.3f} ± {se:.3f}", f"{pay_m:.4f} ± {pay_se:.4f}",
                   f"{price:.4f}"]
            if name == "heston":
                vs = a["vs"].astype(float)
                vm, vse = mean_se(vs[:, -1] - vs[:, 0])
                row.append(f"{vm * 1e4:.2f} ± {vse * 1e4:.2f}")
            else:
                row.append("")
            rows.append(row)
    md += ["## 1. Do the simulated paths reproduce the model?", "",
           "Under the pricing measure with zero rates, S and the variance swap price are "
           "martingales (mean change zero), and the average call payoff is the option "
           "price. Errors are ± 1 standard error.", "",
           table(rows, ["market", "split", "paths", "mean S_T (100)", "mean payoff",
                        "model price", "mean VS change (x1e-4)"]), ""]

    # 2. benchmark hedges on the test sets
    g, h = load("gbm_test"), load("heston_test")
    hS, gS = h["S"].astype(float), g["S"].astype(float)
    strategies = {
        ("GBM", "no hedge"): (gS, np.zeros_like(g["bs_delta"]), ref["gbm_price"], {}),
        ("GBM", "BS delta"): (gS, g["bs_delta"], ref["gbm_price"], {}),
        ("Heston", "no hedge"): (hS, np.zeros_like(h["bs_delta"]), ref["heston_price"], {}),
        ("Heston", "BS delta (t=0 implied vol)"): (hS, h["bs_delta"], ref["heston_price"], {}),
        ("Heston", "Heston delta"): (hS, h["heston_delta"], ref["heston_price"], {}),
        ("Heston", "min-variance delta"): (hS, h["mv_delta"], ref["heston_price"], {}),
        ("Heston", "Heston delta + variance swap"):
            (hS, h["heston_delta"], ref["heston_price"], {"vs": h["vs"], "units": h["vs_units"]}),
    }
    rows = []
    results = {}
    for (mkt, name), (S, dl, p0, kw) in strategies.items():
        row = [mkt, name]
        for c in COSTS:
            s = summary(hedge_pnl(S, dl, K, p0, cost=c, **kw))
            results[(mkt, name, c)] = s
            row += [f"{s['mean']:+.3f}", f"{s['std']:.3f}", f"{s['cvar95']:.3f}"]
        rows.append(row)
    hdr = ["market", "hedge"]
    for c in COSTS:
        bp = f"{c * 1e4:g}bp"
        hdr += [f"mean {bp}", f"std {bp}", f"CVaR95 {bp}"]
    md += ["## 2. Benchmark hedges (test sets, 100k paths)", "",
           "P&L of the dealer who sold the call at the model price and hedges daily. "
           "CVaR95 = average loss in the worst 5% of paths (lower is better). Costs are "
           "charged on the stock only, as a fraction of traded notional, including the "
           "trade that closes the hedge at expiry.", "",
           table(rows, hdr), "",
           "What to read from it:",
           "- Without costs, BS delta on GBM leaves only the error from hedging daily "
           "instead of continuously. The network should match it, not beat it.",
           "- On Heston, a stock-only hedge cannot remove the volatility risk; adding the "
           "variance swap removes most of what is left.",
           "- Costs shift the mean down and fatten the loss tail. A delta hedger trades "
           "every day regardless; that is the room a learned hedge has to trade less.", ""]

    # 3. real NIFTY 50 windows
    nf = load("nifty50_windows")
    S, hv = nf["S"].astype(float), nf["hist_vol"].astype(float)
    T = meta["n_steps"] / meta["days_per_year"]
    taus = (meta["n_steps"] - np.arange(meta["n_steps"])) / meta["days_per_year"]
    p0 = bs_price(meta["S0"], K, T, 0.0, hv)
    dl = bs_delta(S[:, :-1], taus[None, :], K, hv[:, None])
    real_ret = np.log(S[:, -1] / S[:, 0])
    rows = []
    for c in COSTS:
        s = summary(hedge_pnl(S, dl, K, p0, cost=c))
        rows.append([f"{c * 1e4:g}bp", f"{s['mean']:+.3f}", f"{s['std']:.3f}", f"{s['cvar95']:.3f}"])
    dates = nf["start_date"]
    md += ["## 3. Real NIFTY 50 windows (out-of-sample test only)", "",
           f"{len(S)} overlapping 30-day windows starting {str(dates[0])[:10]} to "
           f"{str(dates[-1])[:10]} (about {len(S) // 30} non-overlapping). Premium and BS "
           f"delta use the 60-day historical vol known on the start date "
           f"(median {np.median(hv):.1%}, range {hv.min():.1%} to {hv.max():.1%}).", "",
           f"30-day log return: std {real_ret.std():.3f}, worst {real_ret.min():.3f}, "
           f"best {real_ret.max():.3f}.", "",
           table(rows, ["cost", "mean", "std", "CVaR95"]), ""]
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "data_checks.md").write_text("\n".join(md))
    print("\n".join(md))

    # figure
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.5))
    t = np.arange(meta["n_steps"] + 1)
    for i in range(40):
        ax[0, 0].plot(t, hS[i], lw=0.7, alpha=0.7)
        ax[0, 1].plot(t, np.sqrt(h["v"][i]), lw=0.7, alpha=0.7)
    ax[0, 0].set(title="Heston paths (SPX-calibrated)", xlabel="trading day", ylabel="S")
    ax[0, 1].set(title="Instantaneous vol sqrt(v) on the same paths", xlabel="trading day")
    for name, a, col in (("GBM", gS, "C0"), ("Heston", hS, "C3"), ("NIFTY 50 real", S, "C2")):
        ax[1, 0].hist(np.log(a[:, -1] / a[:, 0]), bins=120, density=True, histtype="step",
                      color=col, label=name, range=(-0.35, 0.25))
    ax[1, 0].set(title="30-day log return", yscale="log")
    ax[1, 0].legend()
    for (mkt, name, col) in (("GBM", "BS delta", "C0"), ("Heston", "BS delta (t=0 implied vol)", "C3"),
                             ("Heston", "Heston delta + variance swap", "C1")):
        S_, dl_, p0_, kw_ = strategies[(mkt, name)]
        ax[1, 1].hist(hedge_pnl(S_, dl_, K, p0_, **kw_), bins=150, range=(-4, 2), density=True,
                      histtype="step", color=col, label=f"{mkt}: {name}")
    ax[1, 1].set(title="Hedging P&L, no costs (test set)", yscale="log")
    ax[1, 1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "data_overview.png", dpi=130)


if __name__ == "__main__":
    main()

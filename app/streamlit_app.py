"""Deep Hedging: interactive web app.

    streamlit run app/streamlit_app.py

Runs the trained networks live (numpy forward pass, see app/hedging.py) on fresh simulated
paths and on real NIFTY 50 windows, next to the benchmark hedges. Nothing is retrained here.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

APP = Path(__file__).resolve().parent
ROOT = APP.parent
sys.path.insert(0, str(APP))

from hedging import (COSTS, K, MARKETS, forward, hedge_all, model_name,  # noqa: E402
                     policy_positions, summary_row)
from deephedge.data import (DataConfig, bs_delta, heston_benchmarks, nifty_windows,  # noqa: E402
                            simulate_gbm, simulate_heston)
from deephedge.heston.black_scholes import bs_price  # noqa: E402
from deephedge.pnl import hedge_pnl  # noqa: E402

REPO = "https://github.com/Subham1709/Deep-Hedging-of-Options"
COLORS = {"deep hedge": "#d62728", "no hedge": "#bbbbbb"}
BENCH_COLORS = ["#1f77b4", "#2ca02c", "#9467bd", "#8c564b"]

st.set_page_config(page_title="Deep Hedging", page_icon="📉", layout="wide")


@st.cache_resource
def load_weights():
    return dict(np.load(APP / "assets" / "policies.npz"))


@st.cache_data
def load_meta():
    return json.loads((ROOT / "data" / "meta.json").read_text())


@st.cache_data
def load_histories():
    out = {}
    for f in sorted((ROOT / "models").glob("*.json")):
        m = json.loads(f.read_text())
        out[f.stem] = {"epochs": m["epochs"], "best_val_cvar": m["best_val_cvar"],
                       "val": [h["val_cvar"] for h in m["history"]]}
    return out


@st.cache_data(show_spinner="Simulating paths ...")
def simulate(market, n, seed):
    cfg, ref = DataConfig(), load_meta()["reference"]
    if market == "gbm":
        return simulate_gbm(n, cfg, ref["gbm_sigma"], seed), None, None, None
    S, v, vs = simulate_heston(n, cfg, seed)
    with st.spinner("Computing Heston Greeks for the benchmark hedges (COS method) ..."):
        bm = heston_benchmarks(S, v, cfg, ref["heston_atm_iv"])
    return S, v, vs, bm


@st.cache_data
def load_nifty():
    closes = pd.read_csv(ROOT / "data" / "raw" / "nifty50_close.csv",
                         parse_dates=["date"]).set_index("date")["close"]
    S, hv, dates = nifty_windows(closes, DataConfig())
    return closes, S, hv, pd.to_datetime(dates)


def fmt_table(rows: dict) -> pd.DataFrame:
    df = pd.DataFrame(rows).T
    return df.style.format({"mean P&L": "{:+.3f}", "std": "{:.3f}", "CVaR95": "{:.3f}",
                            "shares traded": "{:.2f}"}) \
        .highlight_min(subset=["CVaR95"], props="background-color:#d4f4dd;color:#151515;font-weight:600")


weights, meta = load_weights(), load_meta()
ref = meta["reference"]
cfg = DataConfig()

st.title("Deep Hedging of options")
st.markdown(
    "A neural network learns how to hedge a **short call option** with daily trades, "
    "under transaction costs and stochastic volatility, by minimising the **CVaR** "
    "(average loss in the worst 5% of outcomes). It is compared with Black-Scholes, Heston "
    "and minimum-variance delta hedging. Based on Buehler, Gonon, Teichmann & Wood, "
    f"*Deep Hedging* (2019). Code: [GitHub]({REPO})."
)

tab_live, tab_learned, tab_nifty, tab_results = st.tabs(
    ["Hedge live", "What the network learned", "Real NIFTY 50", "Results and method"])

# --------------------------------------------------------------------------- hedge live
with tab_live:
    st.markdown(
        "The dealer sells a 30-trading-day at-the-money call (S0 = K = 100) for its model "
        f"price **{ref['heston_price']:.4f}** and hedges once a day. Pick a market and a cost; "
        "the app simulates fresh paths the networks never saw and hedges them with every "
        "strategy.")
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    mkt_label = c1.selectbox("Market", list(MARKETS))
    cost_label = c2.radio("Transaction cost", list(COSTS), index=2, horizontal=True)
    market, use_vs = MARKETS[mkt_label]
    n = c3.select_slider("Paths", [1_000, 2_000, 4_000, 8_000, 20_000][:4 if market == "heston" else 5],
                         value=4_000)
    seed = c4.number_input("Random seed", 0, 10_000, 2026)
    cost = COSTS[cost_label]

    S, v, vs, bm = simulate(market, n, int(seed))
    p0 = ref["gbm_price"] if market == "gbm" else ref["heston_price"]
    res, pos = hedge_all(weights, market, use_vs, cost, S, v, vs, ref["heston_atm_iv"], p0, bm)

    rows = {k: summary_row(*v_) for k, v_ in res.items()}
    best_bench = min((k for k in res if k not in ("deep hedge", "no hedge")),
                     key=lambda k: rows[k]["CVaR95"])
    d, b = rows["deep hedge"]["CVaR95"], rows[best_bench]["CVaR95"]
    m1, m2, m3 = st.columns(3)
    m1.metric("Deep hedge CVaR95", f"{d:.3f}")
    m2.metric(f"Best benchmark: {best_bench}", f"{b:.3f}")
    m3.metric("Change in tail loss", f"{100 * (d - b) / b:+.0f}%", delta_color="inverse",
              delta=f"{d - b:+.3f}")
    st.dataframe(fmt_table(rows), width="stretch")
    st.caption("P&L per path = premium - payoff + hedge gains - costs. CVaR95 lower is better. "
               "Shares traded counts opening and closing the hedge. Band widths were tuned on "
               "a validation set, not on these paths.")

    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        edges = np.linspace(-6 if market == "heston" else -3, 1.5, 151)
        for i, (k, (p, _)) in enumerate(res.items()):
            if k == "no hedge":
                continue
            dens, _ = np.histogram(p, edges, density=True)
            fig.add_trace(go.Scatter(x=edges[:-1], y=np.where(dens > 0, dens, None), name=k,
                                     line_shape="hv",
                                     line=dict(color=COLORS.get(k, BENCH_COLORS[i % 4]),
                                               width=3 if k == "deep hedge" else 1.5)))
        fig.update_layout(title="Distribution of hedging P&L (left tail = losses)",
                          xaxis_title="P&L per option", yaxis_title="density",
                          height=420, legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig, width="stretch")
    with right:
        i = st.slider("Path to inspect", 0, n - 1, int(np.argmin(np.abs(S[:, -1] - 101))))
        fig = go.Figure()
        days = np.arange(30)
        for j, (k, (_, dl)) in enumerate(res.items()):
            if k == "no hedge":
                continue
            fig.add_trace(go.Scatter(x=days, y=dl[i], name=k, line_shape="hv",
                                     line=dict(color=COLORS.get(k, BENCH_COLORS[j % 4]),
                                               width=3 if k == "deep hedge" else 1.5)))
        fig.add_trace(go.Scatter(x=np.arange(31), y=S[i], name="stock price", yaxis="y2",
                                 line=dict(color="black", dash="dot")))
        fig.update_layout(title=f"Shares held along path {i} "
                                f"(final price {S[i, -1]:.1f}, strike 100)",
                          xaxis_title="trading day", yaxis_title="shares held",
                          yaxis2=dict(overlaying="y", side="right", title="stock price"),
                          height=420, legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig, width="stretch")
        pnl_i = {k: f"{p[i]:+.3f}" for k, (p, _) in res.items()}
        st.caption("P&L on this path: " + ", ".join(f"{k} {x}" for k, x in pnl_i.items()))
    if use_vs:
        st.caption("This network also trades a variance swap; the chart shows its stock "
                   "position only.")

# ------------------------------------------------------------------ what it learned
with tab_learned:
    st.markdown(
        "The network's inputs on each day are log-moneyness log(S/K), the time left, "
        "(under Heston) the current volatility, and **the position it already holds**. "
        "Feeding the position back in is what lets it learn to *not trade*: with costs, "
        "the best position today depends on what you hold. Below, the network's chosen "
        "position as the stock price moves, for two starting holdings.")
    c1, c2, c3 = st.columns(3)
    mk = c1.radio("Network trained on", ["GBM", "Heston"], horizontal=True)
    days_left = c2.slider("Trading days to expiry", 1, 30, 15)
    held = c3.slider("Shares already held", 0.0, 1.0, (0.2, 0.8), 0.05)
    vol = 0.117
    if mk == "Heston":
        vol = st.slider("Current volatility sqrt(v)", 0.05, 0.40, 0.117, 0.005)
    mny = np.linspace(-0.08, 0.08, 241)
    tau_frac = days_left / 30
    fig = go.Figure()
    for (lbl, c), col in zip(COSTS.items(), ["#1f77b4", "#ff7f0e", "#d62728"]):
        name = model_name(mk.lower(), c, False)
        for h, dash in zip(held, ["dot", "solid"]):
            f = [mny / 0.05, np.full_like(mny, tau_frac)]
            if mk == "Heston":
                f.append(np.full_like(mny, vol / 0.15))
            f.append(np.full_like(mny, h))
            y = forward(weights, name, np.stack(f, 1))[:, 0]
            fig.add_trace(go.Scatter(x=mny, y=y, name=f"{lbl}, holding {h:.2f}",
                                     line=dict(color=col, dash=dash)))
    fig.add_trace(go.Scatter(x=mny, y=bs_delta(100 * np.exp(mny), days_left / 252, K, vol),
                             name=f"BS delta at {vol:.1%} vol",
                             line=dict(color="black", dash="dash", width=2)))
    fig.update_layout(xaxis_title="log(S/K)", yaxis_title="new position (shares)", height=480,
                      title=f"Learned position with {days_left} days left")
    st.plotly_chart(fig, width="stretch")
    st.info(
        "How to read it: at 0 bp the two curves for different holdings sit almost on top of "
        "each other and close to BS delta, so yesterday's position barely matters. At 20 bp "
        "they separate: holding 0.8 it stays higher, holding 0.2 it stays lower, i.e. it "
        "moves only part of the way towards delta. That is a learned **no-trade band**, and "
        "it is how the network saves costs. Move the sliders to see it change.")

    hist = load_histories()
    fig = go.Figure()
    for k, h in hist.items():
        fig.add_trace(go.Scatter(y=h["val"], x=np.arange(1, len(h["val"]) + 1), name=k,
                                 line=dict(width=1.3)))
    fig.update_layout(title="Training: validation CVaR95 per epoch (early stopping, patience 20)",
                      xaxis_title="epoch", yaxis_title="validation CVaR95", yaxis_type="log",
                      height=380)
    with st.expander("Training curves of the 9 networks"):
        st.plotly_chart(fig, width="stretch")

# -------------------------------------------------------------------------- nifty
with tab_nifty:
    closes, NS, NHV, ND = load_nifty()
    st.markdown(
        f"Out-of-sample test on **real NIFTY 50 prices**: {len(NS):,} overlapping 30-day "
        f"windows from {ND[0]:%b %Y} to {ND[-1]:%b %Y}, each rescaled to start at 100. The "
        "networks trained on simulated GBM paths are applied as they are; moneyness is "
        "rescaled by (training vol / 60-day historical vol) so a one-standard-deviation move "
        "looks the same as in training. Premium and BS delta use that historical vol, known "
        "on the start date (no look-ahead).")
    iv = ref["gbm_sigma"]
    p0_all = bs_price(100.0, K, cfg.T, 0.0, NHV)
    tgt_all = bs_delta(NS[:, :-1], cfg.taus[None, :-1], K, NHV[:, None])
    rows = {}
    nets = {}
    for lbl, c in COSTS.items():
        pos_n = policy_positions(weights, model_name("gbm", c, False), NS,
                                 vol_rescale=iv / NHV)[..., 0]
        nets[lbl] = pos_n
        rows[f"deep hedge, {lbl}"] = summary_row(hedge_pnl(NS, pos_n, K, p0_all, c), pos_n)
        rows[f"BS delta, {lbl}"] = summary_row(hedge_pnl(NS, tgt_all, K, p0_all, c), tgt_all)
    st.dataframe(fmt_table(rows), width="stretch")
    st.caption("Overlapping windows share most of their moves (about 134 independent "
               "windows), so treat these differences as indicative, not significant.")

    c1, c2 = st.columns([1, 1])
    date = c1.date_input("Window starting on (nearest trading day)", ND[-200].date(),
                         min_value=ND[0].date(), max_value=ND[-1].date())
    cost_lbl = c2.radio("Cost", list(COSTS), index=2, horizontal=True, key="nifty_cost")
    w = int(np.argmin(np.abs(ND - pd.Timestamp(date))))
    c = COSTS[cost_lbl]
    real = closes.loc[ND[w]:].iloc[:31]
    pn = hedge_pnl(NS[w:w + 1], nets[cost_lbl][w:w + 1], K, p0_all[w], c)[0]
    pb = hedge_pnl(NS[w:w + 1], tgt_all[w:w + 1], K, p0_all[w], c)[0]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Start", f"{ND[w]:%d %b %Y}")
    m2.metric("60-day hist vol", f"{NHV[w]:.1%}")
    m3.metric("Deep hedge P&L", f"{pn:+.3f}")
    m4.metric("BS delta P&L", f"{pb:+.3f}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=real.index[:30], y=nets[cost_lbl][w], name="deep hedge",
                             line_shape="hv", line=dict(color="#d62728", width=3)))
    fig.add_trace(go.Scatter(x=real.index[:30], y=tgt_all[w], name="BS delta",
                             line_shape="hv", line=dict(color="#1f77b4")))
    fig.add_trace(go.Scatter(x=real.index, y=real.values, name="NIFTY 50 close", yaxis="y2",
                             line=dict(color="black", dash="dot")))
    fig.update_layout(title="Shares held per unit of option (strike = first close)",
                      yaxis_title="shares held", height=440,
                      yaxis2=dict(overlaying="y", side="right", title="NIFTY 50"),
                      legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, width="stretch")
    st.caption("P&L is per option on a stock rescaled to 100 at the start; premium "
               f"{p0_all[w]:.3f} at the historical vol.")

# ------------------------------------------------------------------------- results
with tab_results:
    st.subheader("Full test results (100,000 fresh paths per market)")
    md = (ROOT / "results" / "results.md").read_text()
    summary = md.split("## Summary")[1].split("## ")[0]
    st.markdown(summary)
    st.markdown(f"Every table: [results/results.md]({REPO}/blob/main/results/results.md). "
                f"Step-by-step write-up: [docs/LEARNING_LOG.md]({REPO}/blob/main/docs/LEARNING_LOG.md).")
    st.image(str(ROOT / "results" / "deep_hedge_results.png"), width="stretch")
    st.subheader("Method")
    h = meta["heston"]
    st.markdown(f"""
- **Problem.** Sell a 30-day ATM call, hedge daily, pay proportional costs (0, 5 or 20 bp of
  traded notional) on every trade including the close. Minimise CVaR95 of the hedging loss.
- **Markets.** Heston with parameters calibrated to SPX options in
  [Heston_Pricing_Engine](https://github.com/Subham1709/Heston_Pricing_Engine)
  (v0 {h['v0']}, kappa {h['kappa']}, theta {h['theta']}, sigma {h['sigma']}, rho {h['rho']}),
  simulated with Andersen's QE scheme at 4 steps a day; GBM at the same implied vol
  ({ref['gbm_sigma']:.2%}) so both charge the same premium; a variance swap as an optional
  second instrument. 100k training, 20k validation, 100k test paths.
- **Network.** One feedforward net (3 hidden layers of 64, ReLU) shared by all 30 dates.
  Inputs: log(S/K), time left, sqrt(v) under Heston, and the previous position.
- **Training.** Adam (lr 1e-3, batch 2048) on the Rockafellar-Uryasev form of CVaR, with the
  VaR term set to the batch quantile each step; early stopping on validation CVaR.
- **Benchmarks.** BS delta, Heston delta, minimum-variance delta (Heston delta plus the part
  of a vol move correlated with spot), Heston delta plus variance swap, and no-trade bands
  around delta tuned on the validation set. Heston Greeks come from a COS pricer vectorised
  over paths.
- **This app.** Runs the saved networks with a numpy forward pass (checked against torch in
  `app/export_weights.py`); it does not retrain anything.
""")
    st.caption("Honest limits: without costs the network beats delta hedging on CVaR but not "
               "on variance (it optimises a different objective). Gains on real NIFTY are "
               "small and come from overlapping windows.")

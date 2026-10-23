# Web app

Live: https://subham-deep-hedging.streamlit.app (Streamlit Community Cloud, main file
`app/streamlit_app.py`, requirements in `app/requirements.txt`).

```
pip install -r app/requirements.txt
streamlit run app/streamlit_app.py
```

## What each tab does

| Tab | What happens |
|---|---|
| Hedge live | Simulates fresh GBM or Heston paths (seed and count you choose), hedges each with the trained network and with the benchmarks, shows CVaR95, P&L distribution and the position along any one path. |
| What the network learned | Feeds a grid of stock prices into one network with a fixed holding and days to expiry, and plots the new position against BS delta. With costs the curves for different holdings separate: the learned no-trade band. |
| Real NIFTY 50 | Runs the GBM-trained networks on all 4,023 real 30-day windows (2010-2026) and on any window you pick, next to BS delta at the 60-day historical vol. |
| Results and method | The 100k-path test table from `results/results.md` and a short method summary. |

## Files

| File | Role |
|---|---|
| `streamlit_app.py` | The page: controls, tables, charts (plotly). |
| `hedging.py` | Numpy forward pass of the network, feeding back yesterday's position (same as `rollout` in `src/deephedge/model.py`), benchmark hedges, no-trade band, summary stats. |
| `export_weights.py` | Copies the weights from `models/*.pt` into `assets/policies.npz` and checks numpy and torch give the same positions. Rerun after retraining. |

Why numpy and not torch: each network is three Linear + ReLU layers, so the forward pass is
a few matrix products (`x @ W.T + b`, then `max(x, 0)`). Leaving torch out keeps the hosted
app small and quick to wake up.

Speed: Heston benchmarks need the Heston Greeks on every path and day (COS method), so
Heston runs take ~10-20 s for 4,000 paths the first time; results are cached per seed.

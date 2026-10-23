# Deep Hedging

A neural network learns how to hedge a short option position under transaction costs and
stochastic volatility, following Buehler, Gonon, Teichmann & Wood, *Deep Hedging*
(Quantitative Finance, 2019). It is compared with Black-Scholes, Heston and
minimum-variance delta hedging on simulated Heston and GBM markets and on real NIFTY 50
price paths.

## Results (100k test paths, CVaR95 of hedging loss, lower is better)

| Market | Costs | Deep hedge | Best benchmark | Improvement |
|---|---|---|---|---|
| GBM, stock | 0 / 20 bp | 0.524 / 1.031 | 0.584 / 1.157 (BS delta + tuned no-trade band) | 10% / 11% |
| Heston, stock | 0 / 20 bp | 2.856 / 3.301 | 2.998 / 3.447 (min-variance delta + band) | 5% / 4% |
| Heston, stock + variance swap | 0 / 20 bp | 0.705 / 1.153 | 1.038 / 1.734 (Heston delta + var swap) | 32% / 33% |
| Real NIFTY 50 (GBM-trained) | 0 / 20 bp | 2.05 / 2.54 | 2.15 / 2.93 (BS delta) | 5% / 13% |

With costs the network learns to trade less (1.06 shares per path at 20 bp vs 2.27 for
the min-variance delta). Without costs it beats delta hedging on CVaR but not on variance:
it optimises a different objective. Full tables: [results/results.md](results/results.md).

![results](results/deep_hedge_results.png)


## The problem

Sell a 30-trading-day at-the-money call (S0 = K = 100), hedge daily, pay proportional
costs on every trade. Minimise the CVaR of the hedging loss.

## Data (`scripts/01_make_data.py`)

| Market | Paths | Notes |
|---|---|---|
| Heston, SPX-calibrated | 100k train / 20k val / 100k test | v0 0.0136, kappa 7.41, theta 0.0421, sigma 1.89, rho -0.65 (from [Heston_Pricing_Engine](https://github.com/Subham1709/Heston_Pricing_Engine)); Andersen QE, 4 sub-steps a day; variance swap as a 2nd hedging instrument |
| GBM | 100k / 20k / 100k | vol 11.7% = the Heston option's implied vol, so both markets charge the same premium |
| Real NIFTY 50 | 4,023 overlapping 30-day windows, 2010-2026 | out-of-sample test only; premium and BS delta from 60-day historical vol |

Test sets also carry the benchmark hedge ratios on every path and date (BS delta, Heston
delta, minimum-variance delta, variance-swap hedge), computed with a COS pricer vectorised
over paths (`src/deephedge/heston/path_greeks.py`).

```bash
pip install -r requirements.txt
python scripts/01_make_data.py      # ~15 min; --small for a 1/20 version
python scripts/02_check_data.py     # results/data_checks.md, results/data_overview.png
python scripts/03_train.py          # 9 models into models/ (~45 min on 4 CPU cores)
python scripts/04_evaluate.py       # results/results.md and figures
pytest -q
```

## Benchmarks to beat

See [results/data_checks.md](results/data_checks.md).

## Layout

```
src/deephedge/data.py          simulators, benchmark hedges, NIFTY windows
src/deephedge/pnl.py           hedging P&L with costs, CVaR
src/deephedge/model.py         policy network, rollout, torch P&L, CVaR objective
src/deephedge/train.py         Adam + early stopping on validation CVaR
models/                        trained weights (.pt) and training histories (.json)
src/deephedge/heston/          Heston model, QE Monte Carlo, BS, COS (from Heston_Pricing_Engine)
                               + path_greeks.py: COS prices and Greeks for many (S, v) at once
scripts/                       01 make data, 02 check data, 03 train, 04 evaluate
docs/LEARNING_LOG.md           what, why and how, step by step
```

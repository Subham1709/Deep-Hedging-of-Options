# Data checks

Option: European call, S0 = K = 100, 30 trading days, daily hedging, zero rates.
Heston price at t = 0: **1.6121** (Black-Scholes implied vol 11.71%); GBM uses that vol, so both markets charge the same premium.

## 1. Do the simulated paths reproduce the model?

Under the pricing measure with zero rates, S and the variance swap price are martingales (mean change zero), and the average call payoff is the option price. Errors are ± 1 standard error.

| market | split | paths | mean S_T (100) | mean payoff | model price | mean VS change (x1e-4) |
|---|---|---|---|---|---|---|
| gbm | train | 100000 | 100.000 ± 0.001 | 1.6052 ± 0.0057 | 1.6121 |  |
| gbm | val | 20000 | 100.000 ± 0.001 | 1.6047 ± 0.0128 | 1.6121 |  |
| gbm | test | 100000 | 100.000 ± 0.001 | 1.6086 ± 0.0057 | 1.6121 |  |
| heston | train | 100000 | 99.969 ± 0.014 | 1.6044 ± 0.0058 | 1.6121 | 0.18 ± 0.13 |
| heston | val | 20000 | 99.982 ± 0.031 | 1.5980 ± 0.0126 | 1.6121 | -0.06 ± 0.29 |
| heston | test | 100000 | 100.017 ± 0.014 | 1.6171 ± 0.0058 | 1.6121 | -0.11 ± 0.13 |

## 2. Benchmark hedges (test sets, 100k paths)

P&L of the dealer who sold the call at the model price and hedges daily. CVaR95 = average loss in the worst 5% of paths (lower is better). Costs are charged on the stock only, as a fraction of traded notional, including the trade that closes the hedge at expiry.

| market | hedge | mean 0bp | std 0bp | CVaR95 0bp | mean 5bp | std 5bp | CVaR95 5bp | mean 20bp | std 20bp | CVaR95 20bp |
|---|---|---|---|---|---|---|---|---|---|---|
| GBM | no hedge | +0.003 | 2.415 | 7.001 | +0.003 | 2.415 | 7.001 | +0.003 | 2.415 | 7.001 |
| GBM | BS delta | +0.000 | 0.253 | 0.590 | -0.136 | 0.261 | 0.767 | -0.546 | 0.314 | 1.329 |
| Heston | no hedge | -0.005 | 2.121 | 6.383 | -0.005 | 2.121 | 6.383 | -0.005 | 2.121 | 6.383 |
| Heston | BS delta (t=0 implied vol) | -0.002 | 1.236 | 3.857 | -0.126 | 1.261 | 4.074 | -0.499 | 1.342 | 4.732 |
| Heston | Heston delta | +0.000 | 1.357 | 3.987 | -0.136 | 1.363 | 4.133 | -0.543 | 1.386 | 4.582 |
| Heston | min-variance delta | -0.001 | 1.029 | 3.000 | -0.116 | 1.042 | 3.146 | -0.459 | 1.088 | 3.595 |
| Heston | Heston delta + variance swap | +0.001 | 0.396 | 1.038 | -0.135 | 0.400 | 1.206 | -0.542 | 0.429 | 1.734 |

What to read from it:
- Without costs, BS delta on GBM leaves only the error from hedging daily instead of continuously. The network should match it, not beat it.
- On Heston, a stock-only hedge cannot remove the volatility risk; adding the variance swap removes most of what is left.
- Costs shift the mean down and fatten the loss tail. A delta hedger trades every day regardless; that is the room a learned hedge has to trade less.

## 3. Real NIFTY 50 windows (out-of-sample test only)

4023 overlapping 30-day windows starting 2010-04-05 to 2026-08-18 (about 134 non-overlapping). Premium and BS delta use the 60-day historical vol known on the start date (median 14.1%, range 7.3% to 58.6%).

30-day log return: std 0.057, worst -0.467, best 0.194.

| cost | mean | std | CVaR95 |
|---|---|---|---|
| 0bp | +0.058 | 0.883 | 2.145 |
| 5bp | -0.080 | 0.894 | 2.338 |
| 20bp | -0.495 | 0.936 | 2.934 |

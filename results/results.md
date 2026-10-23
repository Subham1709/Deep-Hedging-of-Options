# Results

Test sets: 100k fresh paths per market, never used for training or any choice. CVaR95 = average loss in the worst 5% of paths (lower is better). Mean P&L includes the premium (the model price, 1.6121). The band width of each no-trade band is tuned on the validation set. ± is one bootstrap standard error of the CVaR difference between the network and the best benchmark.

## Summary

| market | costs | deep hedge CVaR95 | best benchmark CVaR95 | difference | relative |
|---|---|---|---|---|---|
| GBM market, stock only | 0 bp | 0.524 | 0.584 (BS delta, band ±0.01) | -0.061 ± 0.002 | -10% |
| GBM market, stock only | 5 bp | 0.662 | 0.738 (BS delta, band ±0.02) | -0.077 ± 0.002 | -10% |
| GBM market, stock only | 20 bp | 1.031 | 1.157 (BS delta, band ±0.05) | -0.126 ± 0.002 | -11% |
| Heston market, stock only | 0 bp | 2.856 | 2.998 (min-var delta, band ±0.01) | -0.141 ± 0.009 | -5% |
| Heston market, stock only | 5 bp | 2.982 | 3.124 (min-var delta, band ±0.02) | -0.143 ± 0.009 | -5% |
| Heston market, stock only | 20 bp | 3.301 | 3.447 (min-var delta, band ±0.05) | -0.147 ± 0.009 | -4% |
| Heston market, stock + variance swap | 0 bp | 0.705 | 1.038 (Heston delta + var swap) | -0.334 ± 0.007 | -32% |
| Heston market, stock + variance swap | 5 bp | 0.828 | 1.206 (Heston delta + var swap) | -0.378 ± 0.007 | -31% |
| Heston market, stock + variance swap | 20 bp | 1.153 | 1.734 (Heston delta + var swap) | -0.581 ± 0.008 | -33% |

## GBM market, stock only

**Costs 0 bps**

| hedge | mean P&L | std | CVaR95 | shares traded | note |
|---|---|---|---|---|---|
| **deep hedge** | +0.001 | 0.285 | 0.524 | 2.57 | 55 epochs |
| BS delta | +0.000 | 0.253 | 0.590 | 2.71 |  |
| BS delta, band ±0.01 | +0.000 | 0.256 | 0.584 | 2.48 |  |

**Costs 5 bps**

| hedge | mean P&L | std | CVaR95 | shares traded | note |
|---|---|---|---|---|---|
| **deep hedge** | -0.122 | 0.294 | 0.662 | 2.45 | 40 epochs |
| BS delta | -0.136 | 0.261 | 0.767 | 2.71 |  |
| BS delta, band ±0.02 | -0.115 | 0.271 | 0.738 | 2.29 |  |

**Costs 20 bps**

| hedge | mean P&L | std | CVaR95 | shares traded | note |
|---|---|---|---|---|---|
| **deep hedge** | -0.441 | 0.322 | 1.031 | 2.20 | 59 epochs |
| BS delta | -0.546 | 0.314 | 1.329 | 2.71 |  |
| BS delta, band ±0.05 | -0.385 | 0.344 | 1.157 | 1.91 |  |

## Heston market, stock only

**Costs 0 bps**

| hedge | mean P&L | std | CVaR95 | shares traded | note |
|---|---|---|---|---|---|
| **deep hedge** | -0.001 | 1.173 | 2.856 | 2.44 | 75 epochs |
| BS delta (t=0 vol) | -0.002 | 1.236 | 3.857 | 2.47 |  |
| min-variance delta | -0.001 | 1.029 | 3.000 | 2.27 |  |
| min-var delta, band ±0.01 | -0.001 | 1.030 | 2.998 | 2.09 |  |

**Costs 5 bps**

| hedge | mean P&L | std | CVaR95 | shares traded | note |
|---|---|---|---|---|---|
| **deep hedge** | -0.080 | 1.175 | 2.982 | 1.58 | 82 epochs |
| BS delta (t=0 vol) | -0.126 | 1.261 | 4.074 | 2.47 |  |
| min-variance delta | -0.116 | 1.042 | 3.146 | 2.27 |  |
| min-var delta, band ±0.02 | -0.099 | 1.044 | 3.124 | 1.95 |  |

**Costs 20 bps**

| hedge | mean P&L | std | CVaR95 | shares traded | note |
|---|---|---|---|---|---|
| **deep hedge** | -0.216 | 1.225 | 3.301 | 1.06 | 56 epochs |
| BS delta (t=0 vol) | -0.499 | 1.342 | 4.732 | 2.47 |  |
| min-variance delta | -0.459 | 1.088 | 3.595 | 2.27 |  |
| min-var delta, band ±0.05 | -0.341 | 1.090 | 3.447 | 1.69 |  |

## Heston market, stock + variance swap

**Costs 0 bps**

| hedge | mean P&L | std | CVaR95 | shares traded | note |
|---|---|---|---|---|---|
| **deep hedge** | +0.002 | 0.555 | 0.705 | 2.44 | 112 epochs |
| Heston delta + var swap | +0.001 | 0.396 | 1.038 | 2.71 |  |

**Costs 5 bps**

| hedge | mean P&L | std | CVaR95 | shares traded | note |
|---|---|---|---|---|---|
| **deep hedge** | -0.108 | 0.568 | 0.828 | 2.18 | 88 epochs |
| Heston delta + var swap | -0.135 | 0.400 | 1.206 | 2.71 |  |

**Costs 20 bps**

| hedge | mean P&L | std | CVaR95 | shares traded | note |
|---|---|---|---|---|---|
| **deep hedge** | -0.386 | 0.586 | 1.153 | 1.93 | 84 epochs |
| Heston delta + var swap | -0.542 | 0.429 | 1.734 | 2.71 |  |

## Real NIFTY 50 paths (2010-2026)

4023 overlapping 30-day windows. The networks trained on GBM are applied with log-moneyness rescaled by (training vol / 60-day historical vol), so a move of one standard deviation looks the same to the network as in training. Premium and BS delta use the same historical vol. Overlapping windows share most of their moves, so the last column repeats the CVaR on every 30th window (134 independent windows; noisy).

| hedge | mean P&L | std | CVaR95 | shares traded | note |
|---|---|---|---|---|---|
| deep hedge, 0 bp | +0.039 | 0.877 | 2.046 | 2.60 | non-overlapping CVaR 2.165 |
| BS delta, 0 bp | +0.058 | 0.883 | 2.145 | 2.73 | non-overlapping CVaR 2.187 |
| deep hedge, 5 bp | -0.094 | 0.888 | 2.201 | 2.49 | non-overlapping CVaR 2.343 |
| BS delta, 5 bp | -0.080 | 0.894 | 2.338 | 2.73 | non-overlapping CVaR 2.379 |
| deep hedge, 20 bp | -0.432 | 0.923 | 2.543 | 2.28 | non-overlapping CVaR 2.771 |
| BS delta, 20 bp | -0.495 | 0.936 | 2.934 | 2.73 | non-overlapping CVaR 2.954 |

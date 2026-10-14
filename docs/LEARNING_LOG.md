# Learning log

Notes written while building the project, in the order things were done. Each step has
**what**, **why**, and **what to say in an interview**. This part covers the data; the
training steps come after you choose the hyperparameters.

---

## Step 0. The big picture

**The question.** Black-Scholes says: hold delta shares, rebalance continuously, and the
option is perfectly replicated. Every assumption behind that is false on a real desk:

1. You rebalance discretely (daily), not continuously.
2. Every trade costs money (bid-ask spread, fees, impact).
3. Volatility is not constant, so the stock alone cannot hedge the option.

With any of these, there is no perfect hedge, and "hedge with delta" is no longer optimal.
The question becomes: *given a risk measure, what trading strategy minimises the risk of
the hedged position?* That is an optimisation over strategies, which is hard to solve
analytically, but a neural network can parametrise the strategy and gradient descent can
optimise it. That is **deep hedging** (Buehler, Gonon, Teichmann & Wood, JP Morgan, 2019).

**The pipeline.**

```
market simulator (Heston, GBM) --> paths S_t, v_t, variance swap price
                                         |
            network:  (features on date t, yesterday's position) --> position today
                                         |
     P&L = premium - payoff + trading gains - transaction costs   (per path)
                                         |
          loss = CVaR of -P&L over a batch of paths --> backprop through all 30 dates
                                         |
     compare with: BS delta, Heston delta, min-variance delta, Heston delta + var swap
     test on: fresh simulated paths, then real NIFTY 50 paths (model misspecification)
```

**What makes the project credible.** Showing *when the network wins* (with costs, it learns
to trade less and to hedge vol risk better than BS) and *when it can't* (GBM, no costs:
delta hedging is already optimal, so the network should only match it).

---

## Step 1. The hedging problem and the P&L (`src/deephedge/pnl.py`)

**Set-up (same as the paper).** The dealer sells one European call, strike K = S0 = 100,
expiring in 30 trading days, and rebalances once a day. Rates are zero (over 6 weeks the
interest effect is tiny and it removes discounting from every formula).

**P&L of one path:**

```
PnL = p0 - max(S_T - K, 0) + sum_t delta_t (S_{t+1} - S_t) - c sum_t |delta_t - delta_{t-1}| S_t
```

- `p0` the premium received, `delta_t` shares held from day t to t+1.
- Cost is proportional to traded notional (`c = 0.001` = 10 bps). It starts from no
  position and closes the position at expiry, both of which cost money.
- With a second instrument (the variance swap, step 3) its gains are added the same way.

**Risk measure: CVaR (expected shortfall).** `CVaR_alpha` = average loss over the worst
(1 - alpha) share of paths. Why not variance? Variance penalises gains as much as losses,
and a short option's P&L is skewed (small gains, rare big losses). CVaR looks only at the
loss tail, it is what Basel uses for market risk capital, and it is *convex*, so the
optimisation is well behaved.

The trick that makes CVaR trainable (Rockafellar & Uryasev, 2000):

```
CVaR_alpha(X) = min_w { w + E[(X - w)^+] / (1 - alpha) },    X = loss
```

and the minimising `w` is the VaR. So `w` becomes one more trainable parameter next to the
network weights, and the loss is a plain average over the batch. No sorting needed.

**Interview lines.**
- "Why CVaR and not variance?" Asymmetric risk; coherent; convex; regulatory standard.
- "Why does `p0` matter if it is a constant?" It doesn't change the optimal hedge for CVaR
  (cash invariance: adding cash c to every path lowers CVaR by exactly c), but it makes the P&L numbers readable. The
  minimum CVaR is itself a price: the premium that makes the dealer's risk zero
  (indifference price), which with costs is above the model price.

---

## Step 2. The market simulator (`src/deephedge/data.py`, `heston/`)

**Reused, not rewritten.** The Heston engine's Andersen QE simulator, Black-Scholes code
and COS pricer are copied unchanged from `Heston_Pricing_Engine` (commit `15565ee`). That
engine was validated against published benchmarks, so this project inherits that trust.

**Parameters.** The SPX calibration from that project (12 Oct 2026 quotes):
v0 = 0.0136 (11.7% vol), kappa = 7.41, theta = 0.0421 (20.5%), sigma = 1.89, rho = -0.65.
These are wild: vol-of-vol 1.89 and a Feller ratio of 0.17 mean variance often hits
zero and spikes. Good for this project: it is a hard market to hedge, so differences
between strategies are large.

**Why not NIFTY parameters?** A NIFTY option chain can't be downloaded from here (NSE
blocks cloud servers). With a chain downloaded from a home connection, the engine's
calibration script would produce NIFTY parameters and the data step reruns unchanged.

**Simulation details.**
- 30 hedging dates, one per trading day; one day = 1/252 year, so T = 30/252.
- Simulated at 4 sub-steps per day and observed daily. The sub-steps make the realised
  variance (needed for the variance swap) accurate; QE is already accurate at daily steps.
- Antithetic pairs, as in the engine.
- **GBM market** with vol = the Heston option's implied vol (11.7%), so both markets charge
  the same premium (1.6121). Any difference in hedging results comes from the dynamics.
- Splits: 100k train, 20k validation, 100k test paths per market, each from its own seed.
  The test set is never used for any choice (model, epoch, hyperparameter).

**Interview lines.**
- "Why simulate instead of using history?" There are only ~134 non-overlapping 30-day
  NIFTY windows since 2010; a network needs 10^5 paths. Simulation gives unlimited data
  from a model calibrated to today's option prices; real history is the reality check.
- "What's the risk of that?" The network learns the model, including its mistakes. That is
  why step 5 tests on real NIFTY paths.

---

## Step 3. The second hedging instrument: a variance swap

**What.** A contract that pays the realised variance `int_0^T v_s ds` at expiry. Its fair
price on date t is what has already accrued plus the expected rest:

```
VS_t = int_0^t v_s ds + E[ int_t^T v_s ds | v_t ]
E[ int_t^T v_s ds | v_t ] = theta tau + (v_t - theta)(1 - e^{-kappa tau}) / kappa,   tau = T - t
```

The second formula is exact because the CIR variance has a linear expected path,
`E[v_s | v_t] = theta + (v_t - theta) e^{-kappa (s - t)}`; integrate it over s.

**Why.** In Heston there are two random drivers (spot and variance), so one hedging
instrument can't remove all the risk. The market is *incomplete*. A second instrument
whose price moves with v completes it. Buehler et al. use exactly this one.

**Check.** VS_t is a martingale under the pricing measure: its average change over 30 days
is zero within the Monte Carlo error (results/data_checks.md, table 1).

---

## Step 4. Benchmark hedges and the vectorised Heston Greeks

The network has to beat something. Four benchmarks, stored for every test path and date:

| Benchmark | Formula | Idea |
|---|---|---|
| BS delta | N(d1) at the t = 0 implied vol (11.7%) | what a desk without a stoch-vol model does |
| Heston delta | dC/dS, v held fixed | the model's own spot sensitivity |
| Min-variance delta | dC/dS + (rho sigma / S) dC/dv | also hedges the part of a vol move correlated with spot (Bakshi-Cao-Chen) |
| Heston delta + var swap | dC/dS shares, (dC/dv) / (dVS/dv) swaps | the complete hedge; exact in continuous time |

**Where the min-variance delta comes from.** Over a short step, `dC = C_S dS + C_v dv`.
With only the stock, choose h to minimise Var(dC - h dS):
`h = C_S + C_v Cov(dv, dS)/Var(dS) = C_S + C_v (rho sigma sqrt(v) S sqrt(v)) / (v S^2)
  = C_S + rho sigma C_v / S`.

**Computing 3 million Heston Greeks.** 100k test paths x 30 dates. The engine's COS pricer
prices many strikes for one state; here it is one strike for many states (S, v). Two
facts make it fast:
1. The characteristic function is `exp(C(u, tau) + D(u, tau) v)`: C and D depend only on
   the date, so they're computed once per date and reused on all paths.
2. All cumulants of the log-return are therefore affine in v, so every path gets its own
   COS truncation range from two cumulant evaluations (v = 0 and v = 1).

Greeks come from differentiating the COS series term by term: d/dx multiplies term k by
`i u_k` (delta = that / S), d/dv multiplies it by `D(u_k)`. Checked against the Gil-Pelaez
integral pricer and finite differences (tests/test_data.py): price error below 2e-5, delta
error below 1e-4 except at v = 0 with 1-2 days left, where it's ~1e-3 (the density is a
near-spike there and N = 256 terms can't resolve it; harmless for a benchmark).

---

## Step 5. Real NIFTY 50 paths (out-of-sample only)

From the Nifty_ML_Alpha data snapshot (Yahoo ^NSEI closes, 2010-01-04 to 2026-09-30,
4,113 days): every run of 31 consecutive closes, rescaled to start at 100. That gives
4,023 overlapping windows (about 134 independent ones).

For each window, the 60-day historical vol before the start date prices the option and
sets the BS delta. It's information the trader had that day, so there is no look-ahead.

**Why it matters.** The network is trained on SPX-calibrated Heston. NIFTY has different
vol (median 14% vs 11.7% implied), different jumps (March 2020: -47% in 30 days), and no
model is exactly right. If the network's advantage survives on real paths, it learned
something general (like trading less under costs); if not, it overfit the simulator.

---

## Step 6. Data checks (`scripts/02_check_data.py`, `results/data_checks.md`)

**Do the paths reproduce the model?** Yes. On every split the mean of S_T is 100 and the
mean call payoff matches the Heston price 1.6121 within about one standard error; the
variance swap's mean change is zero within error. If these failed, any hedging result
would be meaningless.

**Benchmarks on the 100k test paths (std of P&L / CVaR95 of loss):**

| Market, hedge | no costs | 5 bps | 20 bps |
|---|---|---|---|
| GBM, no hedge | 2.42 / 7.00 | same | same |
| GBM, BS delta | 0.25 / 0.59 | 0.26 / 0.77 | 0.31 / 1.33 |
| Heston, BS delta (t = 0 implied vol) | 1.24 / 3.86 | 1.26 / 4.07 | 1.34 / 4.73 |
| Heston, Heston delta | 1.36 / 3.99 | 1.36 / 4.13 | 1.39 / 4.58 |
| Heston, min-variance delta | 1.03 / 3.00 | 1.04 / 3.15 | 1.09 / 3.60 |
| Heston, Heston delta + var swap | 0.40 / 1.04 | 0.40 / 1.21 | 0.43 / 1.73 |

Three things to notice, each a talking point:
1. **The model's own delta is not the best stock-only hedge.** Heston delta (v held fixed)
   is worse than plain BS delta, and the minimum-variance delta beats both. Spot and
   variance move together (rho = -0.65), and only the min-variance delta uses that. This
   is the empirical answer to the question left open at the end of the Heston engine log.
2. **Stochastic vol is the big risk.** Even the best stock-only hedge leaves CVaR 3.0, five
   times the GBM case. The variance swap cuts it to 1.0. What is left is mostly from
   hedging once a day while vol-of-vol is 1.89.
3. **Costs.** At 20 bps a daily delta hedger loses ~0.5 on average, a third of the 1.61
   premium. The network can cut that by trading less, and that's where it should win.

The figure (`results/data_overview.png`) shows the Heston return distribution has a much
fatter left tail than GBM, and the real NIFTY 50 one sits in between, with March 2020 as
the extreme.

---

## Next: hyperparameters

Chosen by Subh before training (see the thread). Then: the network (`src/deephedge/model.py`),
the training loop in PyTorch, and results against every benchmark above.

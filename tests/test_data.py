import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from deephedge.data import (DataConfig, bs_delta, heston_call, reference_prices,  # noqa: E402
                            simulate_gbm, simulate_heston)
from deephedge.heston.black_scholes import bs_greeks  # noqa: E402
from deephedge.heston.fourier import heston_price_integral  # noqa: E402
from deephedge.heston.model import HestonParams  # noqa: E402
from deephedge.heston.path_greeks import heston_put_greeks  # noqa: E402
from deephedge.pnl import cvar, hedge_pnl  # noqa: E402

P = HestonParams(0.0136, 7.41, 0.0421, 1.89, -0.65)


@pytest.mark.parametrize("S,v,tau", [(100, 0.0136, 30 / 252), (95, 0.05, 10 / 252),
                                     (110, 0.02, 30 / 252), (100, 0.3, 2 / 252)])
def test_cos_put_matches_integral(S, v, tau):
    price, _, _ = heston_put_greeks([S], [v], 100.0, tau, P)
    ref = heston_price_integral(S, 100.0, tau, 0.0, HestonParams(v, *P.as_array()[1:]), call=False)
    assert price[0] == pytest.approx(ref, abs=5e-5)   # N = 256: ~1e-5 on fat tails


@pytest.mark.parametrize("S,v,tau", [(100, 0.0136, 30 / 252), (97, 0.04, 5 / 252)])
def test_cos_greeks_match_finite_differences(S, v, tau):
    f = lambda s, w: heston_put_greeks([s], [w], 100.0, tau, P, N=1024)[0][0]
    _, dS, dv = heston_put_greeks([S], [v], 100.0, tau, P)
    assert dS[0] == pytest.approx((f(S + 1e-3, v) - f(S - 1e-3, v)) / 2e-3, abs=1e-4)
    assert dv[0] == pytest.approx((f(S, v + 1e-5) - f(S, v - 1e-5)) / 2e-5, rel=1e-3)


def test_vectorised_equals_one_by_one():
    S = np.array([90.0, 100.0, 120.0]); v = np.array([0.0, 0.02, 0.1])
    batch = heston_put_greeks(S, v, 100.0, 0.05, P)
    for i in range(3):
        one = heston_put_greeks(S[i:i + 1], v[i:i + 1], 100.0, 0.05, P)
        for b, o in zip(batch, one):
            assert b[i] == pytest.approx(o[0], abs=1e-12)


def test_heston_collapses_to_black_scholes():
    # vol-of-vol -> 0 and v0 = theta: variance stays at theta, so Heston = BS(sqrt(theta))
    cfg = DataConfig(heston=HestonParams(0.04, 2.0, 0.04, 1e-4, -0.5))
    price, delta, _ = heston_call([105.0], [0.04], 0.1, cfg)
    g = bs_greeks(105.0, 100.0, 0.1, 0.0, 0.2)
    assert delta[0] == pytest.approx(g["delta"], abs=1e-5)


def test_reference_price_and_iv_consistent():
    ref = reference_prices(DataConfig())
    assert ref["gbm_price"] == pytest.approx(ref["heston_price"], abs=1e-8)


def test_gbm_martingale_and_shapes():
    cfg = DataConfig()
    S = simulate_gbm(20_000, cfg, 0.2, seed=0)
    assert S.shape == (20_000, 31) and np.all(S[:, 0] == 100)
    assert S[:, -1].mean() == pytest.approx(100, abs=0.1)


def test_heston_paths_and_variance_swap():
    cfg = DataConfig()
    S, v, vs = simulate_heston(20_000, cfg, seed=0)
    assert S.shape == v.shape == vs.shape == (20_000, 31)
    assert (v >= 0).all()
    # the variance swap ends at the realised variance and is a martingale
    assert vs[:, 0].std() < 1e-12
    assert (vs[:, -1] - vs[:, 0]).mean() == pytest.approx(0, abs=3e-4)
    assert S[:, -1].mean() == pytest.approx(100, abs=0.3)


def test_pnl_conventions():
    S = np.array([[100.0, 101.0, 103.0]])
    # hold one share throughout: gains 3, payoff 3, premium 1 -> P&L 1
    assert hedge_pnl(S, np.ones((1, 2)), 100.0, p0=1.0)[0] == pytest.approx(1.0)
    # 10 bp cost on buying 1 at 100 and selling 1 at 103
    assert hedge_pnl(S, np.ones((1, 2)), 100.0, p0=1.0, cost=1e-3)[0] == pytest.approx(1.0 - 0.203)
    assert bs_delta(np.array([99.0, 101.0]), 0.0, 100.0, 0.2).tolist() == [0.0, 1.0]


def test_cvar():
    pnl = -np.arange(100.0)          # losses 0..99
    assert cvar(pnl, 0.95) == pytest.approx(np.mean(np.arange(95, 100)), abs=0.6)

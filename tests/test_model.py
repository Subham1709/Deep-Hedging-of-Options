import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from deephedge.model import cvar_objective, pnl  # noqa: E402
from deephedge.pnl import cvar, hedge_pnl  # noqa: E402


def test_torch_pnl_matches_numpy():
    rng = np.random.default_rng(0)
    S = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, (50, 31)), 1))
    S[:, 0] = 100
    d = rng.uniform(0, 1, (50, 30))
    vs = np.cumsum(rng.normal(0, 1e-3, (50, 31)), 1)
    u = rng.normal(0, 100, (50, 30))
    ref = hedge_pnl(S, d, 100.0, 1.5, cost=0.002, vs=vs, units=u)
    pos = torch.tensor(np.stack([d, u], -1))
    out = pnl(torch.tensor(S), pos, 1.5, 0.002, vs=torch.tensor(vs)).numpy()
    assert np.allclose(out, ref)


def test_cvar_objective_equals_empirical_cvar():
    x = torch.tensor(np.random.default_rng(1).normal(size=100_000))
    assert cvar_objective(x, 0.95).item() == pytest.approx(cvar(x.numpy(), 0.95), rel=1e-3)

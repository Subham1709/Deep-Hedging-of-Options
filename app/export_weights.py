"""Copy the trained networks' weights into one numpy file for the web app.

    python app/export_weights.py

The app runs the networks with numpy instead of torch: each network is only three
Linear + ReLU layers, and leaving torch out keeps the hosted app small and quick to start.
Writes app/assets/policies.npz with keys "<model name>/W0", "<model name>/b0", ...
and checks that the numpy forward pass gives the same positions as torch.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))

from deephedge.model import HedgePolicy, market_features, rollout  # noqa: E402
from hedging import policy_positions  # noqa: E402

MODELS, OUT = ROOT / "models", ROOT / "app" / "assets" / "policies.npz"


def main():
    arrays = {}
    for pt in sorted(MODELS.glob("*.pt")):
        state = torch.load(pt)
        linears = sorted({k.split(".")[1] for k in state if k.startswith("net.")}, key=int)
        for i, idx in enumerate(linears):
            arrays[f"{pt.stem}/W{i}"] = state[f"net.{idx}.weight"].numpy()
            arrays[f"{pt.stem}/b{i}"] = state[f"net.{idx}.bias"].numpy()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, **arrays)
    print(f"wrote {OUT} ({len(arrays) // 8} networks)")

    # same positions from torch and numpy on random paths
    weights = dict(np.load(OUT))
    rng = np.random.default_rng(0)
    S = 100 * np.exp(np.cumsum(np.c_[np.zeros(500), 0.01 * rng.standard_normal((500, 30))], 1))
    sv = 0.1 + 0.05 * rng.random((500, 31))
    tau_frac = (30 - np.arange(30)) / 30
    for pt in sorted(MODELS.glob("*.pt")):
        heston, vs = pt.stem.startswith("heston"), "_vs_" in pt.stem
        pol = HedgePolicy(3 if heston else 2, 2 if vs else 1)
        pol.load_state_dict(torch.load(pt))
        with torch.no_grad():
            F = market_features(torch.tensor(S), torch.tensor(tau_frac),
                                torch.tensor(sv) if heston else None).float()
            ref = rollout(pol, F).numpy()
        got = policy_positions(weights, pt.stem, S, sv if heston else None)
        err = (np.abs(ref - got) / (1 + np.abs(ref))).max()   # float32 vs float64
        assert err < 1e-4, (pt.stem, err)
    print("numpy and torch positions agree")


if __name__ == "__main__":
    main()

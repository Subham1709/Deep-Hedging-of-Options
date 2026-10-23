"""Training loop: Adam on the CVaR objective, early stopping on validation CVaR."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from .model import HedgePolicy, cvar_objective, market_features, pnl, rollout
from .pnl import cvar


@dataclass
class TrainConfig:
    # the defaults Subh chose on 2026-10-22
    width: int = 64
    depth: int = 3
    alpha: float = 0.95          # CVaR level
    lr: float = 1e-3
    batch: int = 2048
    max_epochs: int = 200
    patience: int = 20           # epochs without a better validation CVaR before stopping
    seed: int = 0


def load_split(data_dir: Path, market: str, split: str, use_vs: bool):
    """Tensors for one dataset: S, features, variance swap (or None)."""
    a = np.load(data_dir / f"{market}_{split}.npz")
    S = torch.tensor(a["S"], dtype=torch.float64)
    n_steps = S.shape[1] - 1
    tau_frac = torch.tensor((n_steps - np.arange(n_steps)) / n_steps, dtype=torch.float64)
    sqrt_v = torch.tensor(np.sqrt(a["v"]), dtype=torch.float64) if market == "heston" else None
    feats = market_features(S, tau_frac, sqrt_v)
    vs = torch.tensor(a["vs"], dtype=torch.float64) if use_vs else None
    return S.float(), feats.float(), (vs.float() if vs is not None else None)


def evaluate(policy, S, feats, vs, p0, cost, chunk=50_000):
    """P&L per path (numpy) and positions (numpy), no gradients."""
    policy.eval()
    pnls, poss = [], []
    with torch.no_grad():
        for i in range(0, len(S), chunk):
            sl = slice(i, i + chunk)
            pos = rollout(policy, feats[sl])
            pnls.append(pnl(S[sl], pos, p0, cost, vs=None if vs is None else vs[sl]))
            poss.append(pos)
    return torch.cat(pnls).numpy().astype(float), torch.cat(poss).numpy()


def train_one(data_dir: Path, market: str, cost: float, use_vs: bool, p0: float,
              cfg: TrainConfig, out_dir: Path, log=print):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    S, F, V = load_split(data_dir, market, "train", use_vs)
    Sv, Fv, Vv = load_split(data_dir, market, "val", use_vs)
    policy = HedgePolicy(F.shape[-1], 2 if use_vs else 1, cfg.width, cfg.depth)
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr)

    best, best_state, bad, history = np.inf, None, 0, []
    t0 = time.time()
    for epoch in range(cfg.max_epochs):
        policy.train()
        perm = torch.randperm(len(S))
        for i in range(0, len(S), cfg.batch):
            idx = perm[i:i + cfg.batch]
            pos = rollout(policy, F[idx])
            loss = cvar_objective(pnl(S[idx], pos, p0, cost,
                                              vs=None if V is None else V[idx]), cfg.alpha)
            opt.zero_grad()
            loss.backward()
            opt.step()
        val_pnl, _ = evaluate(policy, Sv, Fv, Vv, p0, cost)
        val_cvar = cvar(val_pnl, cfg.alpha)
        history.append({"epoch": epoch + 1, "val_cvar": val_cvar, "val_mean": float(val_pnl.mean())})
        if val_cvar < best - 1e-5:
            best, bad = val_cvar, 0
            best_state = {k: v.clone() for k, v in policy.state_dict().items()}
        else:
            bad += 1
        if (epoch + 1) % 10 == 0:
            log(f"  epoch {epoch + 1}: val CVaR {val_cvar:.4f} (best {best:.4f}), "
                f"{time.time() - t0:.0f} s")
        if bad >= cfg.patience:
            break
    policy.load_state_dict(best_state)
    name = model_name(market, cost, use_vs)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), out_dir / f"{name}.pt")
    (out_dir / f"{name}.json").write_text(json.dumps({
        "market": market, "cost": cost, "use_vs": use_vs, "p0": p0, "config": asdict(cfg),
        "best_val_cvar": best, "epochs": len(history), "seconds": time.time() - t0,
        "history": history}, indent=1))
    return policy, best, len(history)


def model_name(market, cost, use_vs):
    return f"{market}_{'stock_vs' if use_vs else 'stock'}_{cost * 1e4:g}bp"

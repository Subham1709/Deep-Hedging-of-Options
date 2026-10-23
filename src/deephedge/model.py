"""The deep hedging policy and its training objective (PyTorch).

One feedforward network is shared by all 30 hedging dates. On date t it sees the market
state and the position it already holds, and outputs the new position:

    position_t = f_theta( log(S_t/K), tau_t / T, [sqrt(v_t)], position_{t-1} )

Feeding the previous position back in is what lets the network learn not to trade: with
transaction costs the best position today depends on what you already hold. (This is the
"semi-recurrent" structure of Buehler et al.; the paper uses one network per date, a
shared network with time-to-expiry as an input does the same job with 30x fewer weights.)

Training minimises CVaR of the hedging loss through the Rockafellar-Uryasev formula

    CVaR_a(L) = min_w  w + E[(L - w)^+] / (1 - a)

The minimising w is the VaR (the alpha-quantile of the loss). Buehler et al. train w as
one more parameter; here it is set to the batch quantile on every step (detached), which
is that minimiser exactly, instead of waiting for gradient descent to find it (with a
learning rate of 1e-3, w would need ~60 epochs just to travel from 0 to a CVaR of 3).
Everything else is differentiable, so gradients flow back through all 30 decisions.
"""
from __future__ import annotations

import torch
from torch import nn

# Scales that put every network input and output near 1. Fixed constants (not fitted on
# data) so a trained model can be applied to any path, real ones included.
MONEYNESS_SCALE = 0.05      # log(S/K) of +-5% is +-1
VOL_SCALE = 0.15            # sqrt(v) of 15% vol is 1
VS_UNITS_SCALE = 300.0      # benchmark variance-swap hedges are a few hundred units


class HedgePolicy(nn.Module):
    def __init__(self, n_market_features: int, n_instruments: int, width=64, depth=3):
        super().__init__()
        layers, d_in = [], n_market_features + n_instruments
        for _ in range(depth):
            layers += [nn.Linear(d_in, width), nn.ReLU()]
            d_in = width
        layers.append(nn.Linear(d_in, n_instruments))
        self.net = nn.Sequential(*layers)
        self.n_instruments = n_instruments

    def forward(self, features, prev_position):
        """features: (batch, n_market_features); prev_position: (batch, n_instruments),
        in network units (variance swap units divided by VS_UNITS_SCALE)."""
        return self.net(torch.cat([features, prev_position], dim=1))


def market_features(S, tau_frac, sqrt_v=None, vol_rescale=None):
    """Network inputs on every date before expiry, shape (n, n_steps, n_features).

    S: (n, n_steps + 1); tau_frac: (n_steps,) time to expiry / T; sqrt_v: (n, n_steps + 1).
    vol_rescale (n,): multiplies log-moneyness, used to transfer a GBM-trained network to
    real paths whose volatility differs from the training vol (see evaluate script).
    """
    m = torch.log(S[:, :-1] / 100.0)
    if vol_rescale is not None:
        m = m * vol_rescale[:, None]
    feats = [m / MONEYNESS_SCALE, tau_frac.expand_as(m)]
    if sqrt_v is not None:
        feats.append(sqrt_v[:, :-1] / VOL_SCALE)
    return torch.stack(feats, dim=-1)


def rollout(policy: HedgePolicy, feats):
    """Run the policy along the paths. Returns positions (n, n_steps, n_instruments) in
    real units (shares; variance swap units)."""
    n, T, _ = feats.shape
    prev = feats.new_zeros(n, policy.n_instruments)
    out = []
    for t in range(T):
        prev = policy(feats[:, t], prev)
        out.append(prev)
    pos = torch.stack(out, dim=1)
    if policy.n_instruments == 2:
        pos = torch.cat([pos[..., :1], pos[..., 1:] * VS_UNITS_SCALE], dim=-1)
    return pos


def pnl(S, pos, p0, cost, K=100.0, vs=None):
    """Torch twin of deephedge.pnl.hedge_pnl (same conventions; costs on the stock only)."""
    delta = pos[..., 0]
    gains = (delta * (S[:, 1:] - S[:, :-1])).sum(1)
    z = delta.new_zeros(len(S), 1)
    trades = torch.diff(torch.cat([z, delta, z], 1), dim=1).abs()
    tc = cost * (trades * S).sum(1)
    if vs is not None:
        gains = gains + (pos[..., 1] * (vs[:, 1:] - vs[:, :-1])).sum(1)
    return p0 - torch.clamp(S[:, -1] - K, min=0.0) + gains - tc


def cvar_objective(pnl_values, alpha):
    loss = -pnl_values
    w = torch.quantile(loss.detach(), alpha)
    return w + torch.clamp(loss - w, min=0.0).mean() / (1 - alpha)

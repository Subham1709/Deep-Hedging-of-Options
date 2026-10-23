"""Train the deep hedging networks: every market x hedging instruments x cost level.

    python scripts/03_train.py [--only heston_stock_20bp] [--max-epochs 200]

Nine models: GBM stock-only, Heston stock-only and Heston stock + variance swap, each at
0, 5 and 20 bps. Weights and training histories go to models/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deephedge.train import TrainConfig, model_name, train_one  # noqa: E402

COSTS = [0.0, 0.0005, 0.002]
RUNS = [(m, c, vs) for (m, vs) in (("gbm", False), ("heston", False), ("heston", True))
        for c in COSTS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--out", default=str(ROOT / "models"))
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--max-epochs", type=int, default=TrainConfig.max_epochs)
    args = ap.parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    data = Path(args.data)
    ref = json.loads((data / "meta.json").read_text())["reference"]
    cfg = TrainConfig(max_epochs=args.max_epochs)
    for market, cost, use_vs in RUNS:
        name = model_name(market, cost, use_vs)
        if args.only and name not in args.only:
            continue
        print(f"{name}:", flush=True)
        p0 = ref["gbm_price"] if market == "gbm" else ref["heston_price"]
        _, best, epochs = train_one(data, market, cost, use_vs, p0, cfg, Path(args.out),
                                    log=lambda s: print(s, flush=True))
        print(f"  done: best val CVaR95 {best:.4f} after {epochs} epochs", flush=True)


if __name__ == "__main__":
    main()

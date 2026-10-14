"""Build every dataset the deep hedging experiments use, into data/.

    python scripts/01_make_data.py            # all of it (~15 min on a laptop, mostly the
                                              # Heston Greeks on the test set)
    python scripts/01_make_data.py --small    # 1/20 of the paths, for a quick try

Outputs (float32, compressed .npz; shapes use n paths and 31 dates t = 0..30):
    gbm_{train,val,test}.npz      S (n, 31); test also bs_delta (n, 30)
    heston_{train,val,test}.npz   S, v, vs (n, 31); test also bs_delta, heston_delta,
                                  mv_delta, vs_units (n, 30)
    nifty50_windows.npz           S (m, 31), hist_vol (m,), start_date (m,)
    meta.json                     config, seeds, reference prices
Deterministic: the same seeds give the same files.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deephedge.data import (DataConfig, bs_delta, config_json, heston_benchmarks,  # noqa: E402
                            nifty_windows, reference_prices, save_npz, simulate_gbm,
                            simulate_heston)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "data"))
    ap.add_argument("--nifty", default=str(ROOT / "data" / "raw" / "nifty50_close.csv"))
    args = ap.parse_args()
    out = Path(args.out)

    cfg = DataConfig()
    if args.small:
        cfg.n_paths = {k: v // 20 for k, v in cfg.n_paths.items()}
    ref = reference_prices(cfg)
    print("reference prices:", ref)
    taus = cfg.taus[None, :-1]

    for split, n in cfg.n_paths.items():
        t0 = time.time()
        S = simulate_gbm(n, cfg, ref["gbm_sigma"], cfg.seeds["gbm"][split])
        extra = {"bs_delta": bs_delta(S[:, :-1], taus, cfg.K, ref["gbm_sigma"])} \
            if split == "test" else {}
        save_npz(out / f"gbm_{split}.npz", S=S, **extra)

        S, v, vs = simulate_heston(n, cfg, cfg.seeds["heston"][split])
        extra = heston_benchmarks(S, v, cfg, ref["heston_atm_iv"]) if split == "test" else {}
        save_npz(out / f"heston_{split}.npz", S=S, v=v, vs=vs, **extra)
        print(f"{split}: {n} paths each, {time.time() - t0:.0f} s")

    nifty = pd.read_csv(args.nifty, parse_dates=["date"]).set_index("date")["close"]
    S, hv, dates = nifty_windows(nifty, cfg)
    save_npz(out / "nifty50_windows.npz", S=S, hist_vol=hv, start_date=dates)
    print(f"nifty: {len(S)} windows {dates[0]} to {dates[-1]}")

    (out / "meta.json").write_text(config_json(cfg, {
        "reference": ref,
        "nifty": {"source": "Nifty_ML_Alpha data/indices.parquet (Yahoo ^NSEI closes)",
                  "first_close": str(nifty.index[0].date()),
                  "last_close": str(nifty.index[-1].date()), "windows": int(len(S))}}))


if __name__ == "__main__":
    main()

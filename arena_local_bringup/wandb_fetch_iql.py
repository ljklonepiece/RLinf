"""Pull open-loop action MSE/MAE (and config) from a wandb run, to compare the internal
IQL training profile against my public N1.7 BC finetune.

Usage:
  python wandb_fetch_iql.py nv-gear/offline_rl/7hjvyj11
"""

import json
import sys

import wandb

RUN = sys.argv[1] if len(sys.argv) > 1 else "nv-gear/offline_rl/7hjvyj11"

api = wandb.Api()
run = api.run(RUN)
print(f"RUN name={run.name!r} state={run.state} id={run.id}")

cfg_terms = ("backbone", "embod", "loss", "base_model", "start_from", "pretrained",
             "dataset", "task", "action_horizon", "reward", "cr2", "eagle")
cfg = {k: v for k, v in run.config.items() if any(t in k.lower() for t in cfg_terms)}
print("CONFIG", json.dumps(cfg, default=str)[:2500])


def _flat(d, p=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(_flat(v, f"{p}.{k}" if p else k))
    else:
        out[p] = d
    return out


_f = _flat(dict(run.config))
_terms = ("learning_rate", "warmup", "weight_decay", "batch", "optim", "scheduler",
          "max_steps", "gradient_accum", "num_gpus", "lr_", "color_jitter", "state_dropout",
          "tune_", "use_relative", "action_horizon", "loss_mode", "polyak")
print("TRAIN_CFG")
for k in sorted(_f):
    if any(t in k.lower() for t in _terms):
        print(f"  {k} = {_f[k]}")

metric_terms = ("mse", "mae", "open_loop", "action", "recon", "l1", "l2", "loss")
summ = {k: v for k, v in dict(run.summary).items()
        if any(t in k.lower() for t in metric_terms) and not k.startswith("_")}
print("SUMMARY_METRICS", json.dumps(summ, default=str)[:2500])

allkeys = sorted(k for k in dict(run.summary).keys() if not k.startswith("_"))
print("ALL_SUMMARY_KEYS", allkeys)

try:
    h = run.history(samples=5000, pandas=True)
    cols = [c for c in h.columns
            if any(t in c.lower() for t in ("mse", "mae", "open_loop", "action_error", "recon", "l1", "l2"))]
    print("HISTORY_METRIC_COLS", cols)
    for c in cols:
        s = h[c].dropna()
        if len(s):
            print(f"  {c}: n={len(s)} min={float(s.min()):.6g} "
                  f"last={float(s.iloc[-1]):.6g} mean_last10={float(s.tail(10).mean()):.6g}")
except Exception as e:  # noqa: BLE001
    print("history error:", repr(e))

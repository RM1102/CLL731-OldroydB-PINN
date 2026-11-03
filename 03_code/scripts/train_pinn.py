"""Train one PINN run from the command line.

Examples
--------
  # lubrication limit, one Deborah number
  python train_pinn.py --eps 0 --De 0.1 --name eps0_De0.1

  # full 2-D at eps = 0.02, warm-started from a previous run
  python train_pinn.py --eps 0.02 --De 0.2 --init-from ../../04_results/pinn/eps0.02_De0.1

  # one network for the whole range 0 <= De <= 0.3
  python train_pinn.py --eps 0.02 --De-range 0 0.3 --name eps0.02_param

Every field of PINNConfig (see oldroydb/pinn.py) can be set with --field-name.
Results go to 04_results/pinn/<name>/ (model.pt, config.json, history.json).
"""
import argparse
import sys
from dataclasses import fields
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "03_code"))
from oldroydb.pinn import PINNConfig, train  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--name", required=True, help="run folder name under 04_results/pinn/")
parser.add_argument("--threads", type=int, default=4, help="CPU threads for PyTorch (4 = fastest on an M4)")
for f in fields(PINNConfig):
    flag = "--" + f.name.replace("_", "-")
    if f.name == "De_range":
        parser.add_argument(flag, type=float, nargs=2, default=None)
    elif f.type in ("float", float):
        parser.add_argument(flag, type=float, default=f.default)
    elif f.type in ("int", int):
        parser.add_argument(flag, type=int, default=f.default)
    else:
        parser.add_argument(flag, type=str, default=f.default)
args = parser.parse_args()
torch.set_num_threads(args.threads)

cfg = PINNConfig(**{f.name: getattr(args, f.name) for f in fields(PINNConfig)})
if cfg.De_range is not None:
    cfg.De_range = tuple(cfg.De_range)
cfg.tag = args.name
out = ROOT / "04_results" / "pinn" / args.name
print(f"training {args.name}: {cfg}")
model, hist = train(cfg, out)
print(f"done in {hist['train_seconds']:.0f}s, held-out loss {hist['test_loss']:.3e}")
for De in ([cfg.De] if cfg.De_range is None else [cfg.De_range[0], sum(cfg.De_range) / 2,
                                                     cfg.De_range[1]]):
    dp = model.pressure_drop(De)
    print(f"  De={De:.3f}: Delta P direct={dp['direct']:.5f} momentum={dp['momentum']:.5f}")

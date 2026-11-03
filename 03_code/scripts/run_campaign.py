"""Train every PINN used in the notebook.

Each chain steps through De, warm-starting each run from the previous one
(continuation in De): the solution at De = 0.1 is a good first guess at
De = 0.15. The first run of a chain starts from scratch.

  python run_campaign.py --chain all        # everything, one run after another
  python run_campaign.py --chain eps0.1 --gpu   # a second lane on the Mac GPU
  python run_campaign.py --chain eps0       # lubrication limit, 7 values of De
  python run_campaign.py --chain eps0.02    # full 2-D, paper's case I
  python run_campaign.py --chain eps0.1     # full 2-D, paper's case II
  python run_campaign.py --chain param      # De-as-input network (eps = 0.02)

Runs that already have a model.pt are skipped, so a stopped campaign resumes.
Run one chain at a time: two runs side by side share the CPU and take longer
in total (measured on an Apple M4).
Results: 04_results/pinn/<run name>/, logs in 04_results/pinn/logs/.
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
TRAIN = str(Path(__file__).with_name("train_pinn.py"))
PINN = ROOT / "04_results" / "pinn"

COMMON = ["--geo-features", "1"]
DEVICE = []                                        # set by --gpu below
# Full 2-D runs (eps > 0) need three extra settings, found by testing (plan, section 9):
#  - corners rounded over 0.03 (the 2-D corner flow is otherwise too sharp to learn)
#  - y-momentum divided by its own natural size 3/H^2, and no residual divided by
#    less than 10 % of its outlet size (else the slow inlet corner dominates the loss)
#  - L-BFGS in 3 rounds on fresh points, with more points (stops over-fitting)
FULL_2D = ["--delta", "0.03", "--ymom-power", "2", "--norm-floor", "0.1",
           "--n-points", "6144", "--lbfgs-points", "12288", "--lbfgs-rounds", "3"]
CHAINS = {
    "eps0": dict(eps=0.0, De=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]),
    "eps0.02": dict(eps=0.02, De=[0.0, 0.1, 0.2, 0.3]),
    "eps0.1": dict(eps=0.1, De=[0.0, 0.1, 0.2, 0.3]),
}


def run(name, args):
    log = PINN / "logs" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    lock = PINN / f"{name}.running"                # stops two lanes training one case
    if (PINN / name / "model.pt").exists():
        print(f"skip {name} (already trained)")
        return
    if lock.exists():
        print(f"skip {name} (being trained by another lane)")
        return
    lock.touch()
    print(f"train {name} ...", flush=True)
    try:
        with open(log, "w") as fh:
            subprocess.run([PY, TRAIN, "--name", name] + COMMON + DEVICE + args, stdout=fh,
                           stderr=subprocess.STDOUT, check=True)
    finally:
        lock.unlink(missing_ok=True)
    print(f"  done: {log.read_text().strip().splitlines()[-2]}", flush=True)


def chain(key):
    """One continuation chain in De; each run warm-starts from the previous one."""
    c = CHAINS[key]
    prev = None
    for De in c["De"]:
        name = f"eps{c['eps']:g}_De{De:g}"
        extra = ["--eps", str(c["eps"]), "--De", str(De)] + (FULL_2D if c["eps"] > 0 else [])
        if prev is not None:                       # warm start: shorter, gentler Adam
            extra += ["--init-from", str(PINN / prev), "--adam-iters", "3000", "--lr", "5e-4"]
            if c["eps"] > 0:                       # the warm start already did most work
                extra += ["--adam-iters", "2000", "--n-points", "4096",
                          "--lbfgs-points", "8192", "--lbfgs-iters", "900"]
        run(name, extra)
        prev = name


def param(eps_list=(0.02,)):
    """Networks with De as an input (one network for 0 <= De <= 0.3)."""
    for eps in eps_list:
        extra = FULL_2D if eps > 0 else ["--n-points", "6144", "--lbfgs-points", "12288",
                                          "--lbfgs-rounds", "3"]
        run(f"eps{eps:g}_param", ["--eps", str(eps), "--De-range", "0", "0.3",
                                   "--adam-iters", "12000", "--lbfgs-iters", "1500",
                                   "--width", "80"] + extra)


parser = argparse.ArgumentParser()
parser.add_argument("--chain", required=True, nargs="+",
                    choices=list(CHAINS) + ["param", "all"],
                    help="one or more chains, run in order; 'all' = every chain")
parser.add_argument("--gpu", action="store_true",
                    help="train on the Mac GPU (mps, float32). A GPU lane can run next to"
                         " a CPU lane: they do not share cores.")
args = parser.parse_args()
DEVICE = ["--device", "mps", "--lbfgs-device", "mps", "--threads", "1"] if args.gpu else []

for key in (list(CHAINS) + ["param"] if "all" in args.chain else args.chain):
    param() if key == "param" else chain(key)

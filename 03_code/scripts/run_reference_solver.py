"""Solve the lubrication-limit (eps -> 0) equations for a sweep of De and save them.

Output: 04_results/reference/fd_eps0.npz with every field on the (Z, eta) grid
for each De, plus Delta P. Takes about a minute on a laptop.
Also writes a grid-convergence table (04_results/reference/convergence.txt).

Run:  python run_reference_solver.py [--quick]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "03_code"))
from oldroydb import analytical as an  # noqa: E402
from oldroydb import lubrication_fd as fd  # noqa: E402

OUT = ROOT / "04_results" / "reference"
DE_LIST = [0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3]

parser = argparse.ArgumentParser()
parser.add_argument("--quick", action="store_true", help="skip the convergence study")
args = parser.parse_args()
OUT.mkdir(parents=True, exist_ok=True)

# ---- main sweep (N = 20 Chebyshev intervals, corner rounding delta = 0.01)
t0 = time.time()
sols, solver = fd.solve(DE_LIST, N=20, delta=0.01, dDe=0.025)
keys = ["Uz", "Uy", "Azz", "Ayy", "Ayz", "P", "G"]
data = {"De": np.array(DE_LIST), "Z": solver.Z, "eta": solver.eta, "H": solver.H,
        "dP": np.array([sols[d]["dP"] for d in DE_LIST])}
for k in keys:
    data[k] = np.stack([sols[d][k] for d in DE_LIST])
np.savez_compressed(OUT / "fd_eps0.npz", **data)
print(f"sweep done in {time.time() - t0:.0f}s -> {OUT / 'fd_eps0.npz'}")
for d, p in zip(DE_LIST, data["dP"]):
    print(f"  De={d:.3f}  Delta P={p:.5f}  series O(De^3)={an.pressure_drop(d, 3):.5f}")

if args.quick:
    sys.exit()

# ---- convergence study: the answer must not depend on numerical choices
rows = [("base: N=20, dZ=5e-3/1.5e-3, delta=0.01, L_out=0.5", dict(N=20))]
rows += [("N=28 (finer across)", dict(N=28)),
         ("dZ halved (finer along)", dict(N=20, dZ_fine=7.5e-4, dZ_coarse=2.5e-3)),
         ("delta=0.005 (sharper corners)", dict(N=20, delta=0.005, dZ_fine=7.5e-4)),
         ("delta=0.02 (rounder corners)", dict(N=20, delta=0.02)),
         ("L_out=1.0 (longer outlet)", dict(N=20, L_out=1.0))]
lines = ["Delta P of the eps -> 0 reference solver under refinement", "",
         f"{'case':52s} {'De=0.1':>9s} {'De=0.2':>9s} {'De=0.3':>9s}"]
for name, kw in rows:
    s, _ = fd.solve([0.1, 0.2, 0.3], **kw)
    lines.append(f"{name:52s} " + " ".join(f"{s[d]['dP']:9.5f}" for d in [0.1, 0.2, 0.3]))
    print(lines[-1])
(OUT / "convergence.txt").write_text("\n".join(lines) + "\n")
print(f"-> {OUT / 'convergence.txt'}")

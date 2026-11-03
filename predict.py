"""Ask the trained PINN for predictions from the command line.

Examples (run from this folder, after setup):
  python predict.py --list                                   which cases and De exist
  python predict.py --case eps0.02 --De 0.2 --dp            pressure drop
  python predict.py --case eps0.02 --De 0.2 --point 0.5 0.3 fields at Z = 0.5, Y = 0.3
  python predict.py --case eps0.02-any --De 0.17 --dp       any De, one network
  python predict.py --case eps0 --De 0.1 --csv fields.csv   whole field to a CSV file

Units are the paper's scaled units (see 03_code/oldroydb/predict.py).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "03_code"))
from oldroydb import predict as P  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
ap.add_argument("--list", action="store_true", help="show the available cases and De values")
ap.add_argument("--case", default="eps0.02", choices=list(P.CASES))
ap.add_argument("--De", type=float, default=0.2, help="Deborah number (0 to 0.3)")
ap.add_argument("--dp", action="store_true", help="print the pressure drop")
ap.add_argument("--point", type=float, nargs=2, metavar=("Z", "Y"),
                help="print every field at one point")
ap.add_argument("--csv", metavar="FILE", help="write the fields on a grid to a CSV file")
args = ap.parse_args()

if args.list:
    for c in P.available_cases():
        Des = P.trained_De(c)
        print(f"{c:12s} {P.CASES[c]['label']:36s} De: "
              + ("any value from 0 to 0.3" if Des is None else ", ".join(f"{d:g}" for d in Des)))
    sys.exit()

model, De, note = P.load(args.case, args.De)
print(f"case {args.case} ({P.CASES[args.case]['label']}), De = {De:g}")
if note:
    print(f"  note: {note}")

if args.dp or not (args.point or args.csv):
    direct, mom = P.pressure_drop(model, De)
    pc = P.PRESSURE_SCALE_PA[P.CASES[args.case]["eps"]]
    print(f"  pressure drop (scaled)        = {direct:.4f}   (momentum route {mom:.4f})")
    print(f"  in the paper's channel (p_c = {pc:g} Pa) = {direct * pc:.0f} Pa")

if args.point:
    Z, Y = args.point
    f = P.fields(model, De, Z, Y)
    if not f["inside"][0]:
        H = model.geom.H(np.clip(Z, model.geom.Z_min, model.geom.Z_max))
        sys.exit(f"  point is outside the channel (at Z = {Z:g} the wall is at Y = {H:.3f})")
    for k, name in P.FIELD_NAMES.items():
        print(f"  {name:28s} = {f[k][0]: .5f}")

if args.csv:
    ZZ, YY, f = P.grid(model, De)
    cols = ["Z", "Y"] + list(P.FIELD_NAMES)
    data = np.column_stack([ZZ.ravel(), YY.ravel()] + [f[k].ravel() for k in P.FIELD_NAMES])
    np.savetxt(args.csv, data, delimiter=",", header=",".join(cols), comments="", fmt="%.6g")
    print(f"  wrote {len(data)} points to {args.csv}")

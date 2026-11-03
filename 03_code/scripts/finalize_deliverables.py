"""Rebuild every deliverable from the saved results, in order.

  1. 04_results/summary.md              (summarize_results.py)
  2. the notebook, executed             (05_notebook/build_notebook.py + nbconvert)
  3. the HTML copy of the notebook      (nbconvert --to html)
  4. 04_results/figures/*.png           (export_notebook_figures.py)
  5. CLL731_OldroydB_PINN_bundle.zip    (make_bundle.py)

Run:  python finalize_deliverables.py      (about 2 minutes; trains nothing)
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
NB_DIR = ROOT / "05_notebook"
NB = "CLL731_OldroydB_PINN.ipynb"
SCRIPTS = Path(__file__).parent

steps = [
    ("summary", [PY, "summarize_results.py"], SCRIPTS),
    ("build notebook", [PY, "build_notebook.py"], NB_DIR),
    ("execute notebook", [PY, "-m", "jupyter", "nbconvert", "--execute", "--to", "notebook",
                          "--inplace", "--ExecutePreprocessor.timeout=900", NB], NB_DIR),
    ("export HTML", [PY, "-m", "jupyter", "nbconvert", "--to", "html", NB], NB_DIR),
    ("export figures", [PY, "export_notebook_figures.py"], SCRIPTS),
    ("zip bundle", [PY, "make_bundle.py"], SCRIPTS),
]
for name, cmd, cwd in steps:
    print(f"-- {name}", flush=True)
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:])
        sys.exit(f"step '{name}' failed")
    last = [l for l in (r.stdout + r.stderr).splitlines() if l.strip() and "WARNING" not in l]
    print("   " + (last[-1] if last else "ok"))
print("all deliverables rebuilt")

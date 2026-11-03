"""Save every numbered figure of the executed notebook as a PNG.

Each "**Figure N:**" caption belongs to the last image shown by the code cell
before it. Output: 04_results/figures/figNN_<first words of caption>.png

Run (after executing the notebook):  python export_notebook_figures.py
"""
import base64
import re
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[2]
NB = ROOT / "05_notebook" / "CLL731_OldroydB_PINN.ipynb"
OUT = ROOT / "04_results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
for old in OUT.glob("fig[0-9][0-9]_*.png"):        # start clean: figures follow the notebook
    old.unlink()

nb = nbformat.read(NB, 4)
last_png = None
saved = 0
for cell in nb.cells:
    if cell.cell_type == "code":
        for o in cell.get("outputs", []):
            if o.get("output_type") == "display_data" and "image/png" in o.get("data", {}):
                last_png = o["data"]["image/png"]
        continue
    m = re.search(r"\*\*Figure (\d+):\*\*\s*(.*)", cell.source)
    if m and last_png:
        words = re.sub(r"[^a-z0-9 ]", "", m.group(2).lower()).split()[:5]
        name = f"fig{int(m.group(1)):02d}_{'_'.join(words)}.png"
        (OUT / name).write_bytes(base64.b64decode(last_png))
        saved += 1
        last_png = None
print(f"saved {saved} figures to {OUT}")

"""Pack the project into one zip file that can be emailed or shared.

Leaves out the local Python environment, development runs, caches and
scratch files. The zip keeps the folder layout, so the notebook runs after
unzipping (see README.md, "Start here").

Run:  python make_bundle.py        -> CLL731_OldroydB_PINN_bundle.zip in the project folder
"""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "CLL731_OldroydB_PINN_bundle.zip"
SKIP_DIRS = {".venv", "__pycache__", ".ipynb_checkpoints", "dev_runs", "_figs",
             "notebook_demo", ".pytest_cache", ".claude", ".git"}
SKIP_FILES = {OUT.name, ".DS_Store"}
SKIP_SUFFIXES = {".running"}                      # lock files of runs in progress


def keep(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    if any(part in SKIP_DIRS for part in rel.parts):
        return False
    return (path.name not in SKIP_FILES and path.suffix not in SKIP_SUFFIXES
            and not path.name.startswith("_test_run"))


files = sorted(p for p in ROOT.rglob("*") if p.is_file() and keep(p))
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for f in files:
        z.write(f, Path(ROOT.name) / f.relative_to(ROOT))
print(f"{len(files)} files -> {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")

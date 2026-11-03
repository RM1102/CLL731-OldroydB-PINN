#!/usr/bin/env bash
# One-command setup and launcher for macOS and Linux.
#
#   bash setup_and_run.sh            set up (first time only), then show a menu
#   bash setup_and_run.sh gui        open the PINN explorer window
#   bash setup_and_run.sh notebook   open the notebook (full report with code)
#   bash setup_and_run.sh test       quick check that everything works (~10 s)
#   bash setup_and_run.sh retrain    retrain every network (about 4 hours!)
#
# The first run creates a private Python environment in .venv/ inside this
# folder and downloads the packages (about 1 GB, 5-15 minutes). Nothing is
# installed system-wide; delete the .venv folder to remove it all.
set -e
cd "$(dirname "$0")"

# ---- find a Python between 3.10 and 3.13 (the range PyTorch supports well)
PY=""
for c in python3.12 python3.11 python3.13 python3.10 python3; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c \
     'import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 13) else 1)' 2>/dev/null; then
    PY="$c"; break
  fi
done
if [ -z "$PY" ]; then
  echo "Python 3.10 to 3.13 was not found."
  echo "Install it from https://www.python.org/downloads/ and run this script again."
  exit 1
fi

# ---- first-time setup: private environment + packages
if [ ! -x .venv/bin/python ]; then
  echo "First-time setup with $($PY --version). This downloads about 1 GB (5-15 minutes)."
  "$PY" -m venv .venv
fi
if ! .venv/bin/python -c "import torch, matplotlib, scipy, jupyter, pytest" 2>/dev/null; then
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r 03_code/requirements.txt
fi
VPY=.venv/bin/python

run_test() {
  echo "Quick check (about 10 seconds) ..."
  "$VPY" -m pytest -q 03_code/tests
  "$VPY" gui.py --selftest gui_selftest.png
  "$VPY" predict.py --case eps0.02 --De 0.2 --dp
}

choice="${1:-}"
if [ -z "$choice" ]; then
  echo
  echo "What would you like to do?"
  echo "  1) Open the PINN explorer window (predictions for any case and De)"
  echo "  2) Open the notebook (the full report, with code and charts)"
  echo "  3) Quick check that everything works (about 10 seconds)"
  echo "  4) Retrain every network from scratch (about 4 hours on a laptop)"
  read -r -p "Type 1, 2, 3 or 4 and press Enter: " n
  case "$n" in 1) choice=gui ;; 2) choice=notebook ;; 3) choice=test ;; 4) choice=retrain ;; *) exit 0 ;; esac
fi

case "$choice" in
  gui)      "$VPY" gui.py ;;
  notebook) "$VPY" -m jupyter notebook 05_notebook/CLL731_OldroydB_PINN.ipynb ;;
  test)     run_test ;;
  retrain)
    echo "WARNING: retraining all 16 networks takes about 4 hours on a laptop CPU."
    echo "The trained networks already in 04_results/pinn/ are used if you skip this."
    read -r -p "Type YES to start: " ok
    [ "$ok" = "YES" ] || exit 0
    (cd 03_code/scripts && "../../$VPY" run_campaign.py --chain all && "../../$VPY" finalize_deliverables.py) ;;
  *) echo "Unknown option '$choice'. Use gui, notebook, test or retrain."; exit 1 ;;
esac

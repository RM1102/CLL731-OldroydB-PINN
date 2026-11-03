# The notebook

* `CLL731_OldroydB_PINN.html`: open in any web browser. Everything is already run; nothing to install.
* `CLL731_OldroydB_PINN.ipynb`: the same notebook, to run yourself.
* `build_notebook.py`: the script that writes the notebook (edit text here, then rebuild).
* `STORYLINE.md`: the section headings, which tell the story on their own.

## Running it on your own laptop (CPU only, about 2 minutes)

The notebook needs the whole project folder next to it: it imports the code from `03_code/` and loads results from `04_results/`.

```bash
cd "sem 9 atp project"
python3 -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r 03_code/requirements.txt
jupyter notebook 05_notebook/CLL731_OldroydB_PINN.ipynb
```

Then choose *Run → Run All Cells*. The trained networks are loaded from disk. Set `TRAIN_DEMO = True` in the first cell to also watch one small network train live (about 3 minutes). Full retraining uses the scripts in `03_code/scripts/` (several hours).

## Rebuilding after editing

```bash
cd 05_notebook
python build_notebook.py
jupyter nbconvert --execute --to notebook --inplace CLL731_OldroydB_PINN.ipynb
jupyter nbconvert --to html CLL731_OldroydB_PINN.ipynb
```

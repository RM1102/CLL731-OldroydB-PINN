# Results

Every file here is produced by a script in `03_code/scripts/`. Nothing is typed in by hand.

| Folder / file | What it holds | Made by |
|---|---|---|
| `analytical/sympy_check.txt` | each formula of the paper checked symbolically; ΔP₀…ΔP₃ for α = 4, β̃ = 0.4 | `derive_perturbation_sympy.py` |
| `analytical/newtonian_eps2.txt` | O(ε²) correction to the Newtonian pressure drop (extended lubrication) | `newtonian_eps2_correction.py` |
| `digitized/fig2_dP_vs_De.csv` | the paper's finite-element ΔP vs De (ε = 0.02 and 0.1), recovered from the PDF | `digitize_fe_data.py` |
| `digitized/fig7_*.csv`, `fig8_*.csv` | the same for ΔP vs α and vs η_p/η_s | `digitize_fe_data.py` |
| `reference/fd_eps0.npz` | our ε → 0 solver: all fields on the (Z, η) grid for 13 values of De | `run_reference_solver.py` |
| `reference/convergence.txt` | the solver's ΔP under grid, corner and outlet changes | `run_reference_solver.py` |
| `pinn/<eps>_De<De>/` | one trained PINN: `model.pt` (weights), `config.json` (settings), `history.json` (loss curves, run time) | `run_campaign.py` / `train_pinn.py` |
| `pinn/<eps>_param/` | PINN with De as an extra input (0 ≤ De ≤ 0.3) | `run_campaign.py --chain param` |
| `pinn/logs/` | the printed training log of every run | same |
| `pinn/dev_runs/` | development runs kept as a record of design changes (see `02_plan/project_plan.md`, section 9) | `train_pinn.py` |
| `figures/` | figures exported from the notebook | notebook |

In the CSV files, the last column (`*_centroid`) is the data value. The `*_boxcentre` column is kept only to document the marker-centre check.

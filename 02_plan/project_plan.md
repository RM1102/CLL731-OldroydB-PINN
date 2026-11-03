# Project plan: solving Boyko & Stone (2022) with a physics-informed neural network

**Course:** CLL731, IIT Delhi. **Status on 1 Nov 2026:** research, benchmarks, PINN training and notebook done.

**In brief.** We train a neural network to satisfy the Oldroyd-B flow equations in the paper's narrowing channel, using no simulation data. We then check its pressure drop and stress fields against three independent references: the paper's formulas, the paper's finite-element results, and our own numerical solver. The final product is one Jupyter notebook the professor can run on a laptop.

---

## 1. What we will deliver

| # | Deliverable | Where |
|---|---|---|
| 1 | Research notes: paper in plain words, literature review, code survey | `01_research/notes/` |
| 2 | This plan | `02_plan/` |
| 3 | Python package: geometry, analytical results, reference solver, PINN | `03_code/oldroydb/` |
| 4 | Reproducible scripts (SymPy check, FE digitizer, training runs) | `03_code/scripts/` |
| 5 | Results: trained weights, data files, figures | `04_results/` |
| 6 | **Main notebook + HTML copy**, runnable on CPU, with pre-trained weights | `05_notebook/` |

## 2. The approach in one picture

```
        paper's equations (2.14), scaled so every term is O(1)
                              |
        +---------------------+----------------------+
        |                     |                      |
  analytical series     reference solver        PINN (this project)
  (SymPy-verified)      (eps -> 0, global       (full 2-D eqs, any eps,
  valid for De<~0.1     Newton, any De)          trained on equations only)
        |                     |                      |
        +----------+----------+-----------+----------+
                   |                      |
          paper's FE data (recovered from the PDF vector graphics)
                   |
         comparison: Delta P(De), fields, error tables
```

## 3. Why these choices (research → decision)

| Decision | Reason | Source |
|---|---|---|
| Solve the paper's scaled equations (2.14) | Every term is O(1). The scaling is exact (checked by hand), so ε = 0 gives lubrication and finite ε gives full 2-D Stokes. | paper §2.1; `paper_notes.md` §3 |
| Wall-fitted coordinate η = Y/H(Z) ∈ [0, 1] | The wall becomes a fixed line, which makes exact wall conditions easy | Hinch et al. 2024; Housiadas 2025 |
| Stream function Ψ = Ψ₀(η) + η(1−η²)² N(Z, η²) | Mass conservation, no slip, no penetration, symmetry and unit flux hold **exactly** for any network output | Rao et al. 2020; LabelFree-DNN-Surrogate repo |
| Network inputs (Z, η²) for even fields; η·N for A_yz | Symmetry about the centreline is built in, not learned | literature review rec. 3 |
| Straight inlet and outlet sections, fully developed inflow stress imposed exactly | Matches the paper's FE set-up (appendix A). Stress needs a condition at inflow only. | Renardy 1988; paper app. A |
| Corners rounded over δ = 0.01 | The hyperbolic shape meets the straight sections with a slope jump; a smooth network cannot represent a jump. Our reference solver shows Δp changes by < 2×10⁻⁴ between δ = 0.005 and 0.02. | Housiadas 2025; our test |
| Outputs scaled by local lubrication sizes (A_zz ~ 18De/H⁴, A_yz ~ 3/H², ...) | The network outputs stay O(1) from inlet (H = 4) to outlet (H = 1), a 256-fold range | Hinch et al. 2024 |
| Separate small networks per field | Fields of very different shape do not compete for the same weights | ViscoelasticNet (Thakur et al. 2024) |
| Loss = equation residuals only, each divided by its local size | All boundary conditions are exact, so there are no loss weights to tune between BCs and PDEs | Wang et al. 2023 (expert's guide) |
| Adam, then L-BFGS in float64 on CPU | Adam finds the basin; L-BFGS polishes. MPS has no float64. | Rathore et al. 2024; code survey |
| Continuation in De (warm start) and a De-as-input network | Low-De solution is a good start for higher De; a De-input network gives the whole q–Δp curve from one model | Krishnapriyan et al. 2021; code survey |
| Δp computed two ways (pressure difference and integrated centreline momentum) | Agreement is a convergence check; the second route also splits Δp into the paper's four terms (eq. 5.3, fig. 3) | Sialmas & Housiadas 2025 |
| Global (all-stations) reference solver, not marching | Marching is ill-posed for De > 0 (we observed blow-up that worsens with refinement) | our finding, `paper_notes.md` §6 |

## 4. How each resource is used

| Resource | Used for |
|---|---|
| Boyko & Stone 2022 (main paper) | equations, parameters (α = 4, β̃ = 0.4, ε = 0.02/0.1, De ≤ 0.3), formulas, FE data (figs 2, 7, 8), field maps (figs 4–6) |
| Boyko & Stone 2021 (reciprocal theorem) | background for ΔP₃; we verified ΔP₃ independently |
| Hinch, Boyko & Stone 2024; Boyko, Hinch & Stone 2024 | inflow conditions, stress scales, wall layers ~0.04/De, exit-length guidance |
| Housiadas & Beris 2023/2024, Housiadas 2025 | expected slope-jump behaviour at the inlet corner; De ≈ 0.5 practical limit; Δp cross-check routes |
| Eaglw/PINN-tesi (GitHub) | closest existing Oldroyd-B PINN; reference for residual structure and Adam→L-BFGS staging (read, not copied) |
| LabelFree-DNN-Surrogate | hard wall conditions and parameter-as-input design |
| jaxpi, lu-group/pinn-sampling | ideas for loss balancing and residual-based resampling (ported, not imported) |
| rheoTool / COMSOL | not used: our recovered FE data plus our own solver already give an independent reference |

Full lists: `01_research/notes/literature_review.md` (56 references) and `github_and_tools_survey.md` (44 tools).

## 5. The PINN, concretely

**Domain.** Z ∈ [−0.5, 1.5] (straight inlet 0.5ℓ, contraction ℓ, straight outlet 0.5ℓ), η ∈ [0, 1].

**Unknown fields and how the network builds them.**

| Field | Built as | What is exact by construction |
|---|---|---|
| Ψ (gives Uz = ∂Ψ/∂Y, Uy = −∂Ψ/∂Z) | Ψ₀(η) + η(1−η²)² · B(Z) · c·N_ψ | continuity, no slip, no penetration, symmetry, flux = 1, fully developed at both ends |
| P | N_P(Z, η²) | symmetry |
| A_zz, A_yy, A_yz | A_inlet(η) + ((Z−Z_in)/L) · scale(Z) · N_A | inflow stress, symmetry (A_yz odd) |

Ψ₀ = (3η − η³)/2 is the cubic that meets the wall conditions; B(Z) vanishes at both ends.

**Loss.** Mean square of the five residuals (z-momentum, y-momentum, three conformation equations), each divided by its local lubrication size, at random collocation points that are resampled during training.

**Training stages.** (1) ε = 0 (lubrication limit), compared point-by-point with the reference solver. (2) ε = 0.02 and 0.1 (full 2-D), compared with the paper's FE pressure drops. (3) One De-as-input network (ε = 0.02) for the whole curve.

## 6. Validation matrix

| Test | Reference | Metric | Pass mark (our choice) |
|---|---|---|---|
| De = 0, ε = 0 | ΔP₀ = 0.99609, exact Newtonian fields | relative error | < 0.1 % |
| small De, ε = 0 | series to O(De³) | ΔP difference | within the O(De⁴) size |
| De = 0.05–0.3, ε = 0 | reference solver (fields + ΔP) | relative L2 error per field; ΔP error | < 1–2 % fields, < 0.5 % ΔP |
| ε = 0.02, 0.1, De = 0–0.3 | paper's FE (recovered) | ΔP error | < 1 % (ε = 0.02), < 2 % (ε = 0.1) |
| fields at ε = 0.02 | paper figs 4–6 | sign, location and size of peaks | qualitative |
| Δp two ways | itself | difference between routes | < 0.2 % |
| contributions ①–④ (eq. 5.3) | paper fig. 3 | trend with De | qualitative |

## 7. Risks and what we do about them

| Risk | Mitigation |
|---|---|
| Steep stresses near the outlet wall at De = 0.3 | local output scaling; extra points near the outlet; float64 L-BFGS |
| Corner layers (slope jump) | rounded corners (δ = 0.01) and dense points there; measured δ effect is tiny |
| Training too slow for the professor's laptop | notebook loads pre-trained weights by default; retraining is optional |
| PINN stalls at higher De | continuation from lower De; De-input network trained on [0, 0.3] only |
| Silent errors in our own references | SymPy check of every formula; solver checked against FE data (0.35 %) and against the series at small De |

## 8. Schedule (status on 1 Nov 2026)

| Step | Status |
|---|---|
| Read paper, verify formulas (SymPy) | done |
| Literature + code survey | done |
| Recover FE data from PDF | done |
| Reference solver (ε → 0, global Newton) | done; matches FE within 0.35 % |
| PINN code + ε = 0 validation | done; ΔP within 0.25 % of the reference solver |
| ε = 0.02 / 0.1 runs | done; within 0.68 % of the paper's FE pressure drops (1.05 % by the momentum route) |
| De-input network (ε = 0.02) | trained last (CPU lane) |
| Notebook + HTML, final checks | done after the last training run; results in `04_results/summary.md` |

## 9. What changed while building, and why

Each change below was forced by a measured problem. The runs that showed the problem are kept in `04_results/pinn/dev_runs/` with their logs.

| # | Problem we measured | Change | Effect |
|---|---|---|---|
| 1 | Marching the ε → 0 equations downstream blew up for every De > 0, and sooner on finer grids (ill-posed: the ∂T_zz/∂Z ↔ ∂U_z/∂Z loop acts like diffusion along Z) | solve all stations at once with Newton's method and an outflow condition | converges in 3–5 Newton steps; ΔP within 0.4 % of the paper's FE data |
| 2 | At ε = 0, PINN ΔP was 0.26 % off the reference although its loss was small; the network had let the pressure vary across the channel by ~0.007 to hide errors in other equations | pressure written as P = P_c(Z) + ε² P₁(Z, η²), so it is exactly uniform across the channel when ε = 0 | ΔP error 0.26 % → 0.02 % at De = 0.1 |
| 3 | Field errors concentrated in the two corner zones where the wall slope jumps | network also given H(Z) and a slope indicator (0 in straight parts, 1 in the contraction) as inputs | training loss 10× lower; ΔP unchanged (the error came from item 2) |
| 4 | First ε = 0.02 run: the two ΔP routes disagreed by 1.2 %, held-out loss 7× the training loss; residuals piled up at the inlet corner, where dividing by the tiny local pressure gradient (3/H³ = 0.047) inflated them about 4000-fold | for ε > 0: corners rounded over 0.03 instead of 0.01; y-momentum divided by its own size 3/H²; no residual divided by less than 10 % of its outlet size; L-BFGS in 3 rounds on fresh points | loss about 150× lower at the same step; corner rounding changes the reference ΔP by ≤ 0.03 % (`04_results/reference/corner_rounding.txt`) |
| 5 | Notebook flag `QUICK = False` promised a full retrain inside the notebook (hours) | replaced by `TRAIN_DEMO`, a 3-minute live training run; full retraining stays in the scripts | notebook always runs in minutes |
| 6 | Warm-started ε = 0 runs drifted to 0.2 % ΔP error and ~2 % stress error at De = 0.2 | tried a stronger schedule (6000 Adam steps at lr 10⁻³, 3 L-BFGS rounds) | no change in field errors (`dev_runs/test_eps0_De0.2_strongwarm`): the error is set by the outlet corner, not by the optimiser; kept the cheaper schedule |
| 7 | Training time (about 4 hours of runs on one laptop) | measured on the Apple M4: two CPU runs side by side give only ~70 % of one-at-a-time throughput (they share the 4 fast cores); the 6 efficiency cores are ~7× slower; the GPU (MPS, float32) is a separate resource | one CPU lane (ε = 0.02, De-input net) plus one GPU lane (ε = 0.1): CPU lane 77 → 120 ms/step, GPU lane 140 ms/step, total ~1.4× faster. Moving the GPU lane's dispatch thread to the efficiency cores was worse (~1.1×) |
| 8 | GPU has no float64 | ε = 0.1 runs use float32 for L-BFGS too (as in the original PINN papers) | labelled in the notebook; their held-out loss is reported with the others |

**What limits accuracy now.** Away from the corners the PINN fields agree with the reference solver to 0.01–0.5 %. Within about 0.05 of the outlet corner, where the wall slope changes over 1 % of the length and the stresses are largest, errors reach a few percent. Options not tried for lack of time: a separate sub-network for the corner zone (XPINN-style), or a smoother wall shape as in Hinch et al. (2024).


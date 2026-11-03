# Software and code survey: PINN for Oldroyd-B flow in a narrow hyperbolic contraction

Scope: code and tools that could help re-solve Boyko & Stone (2022, JFM 936 A23, arXiv 2109.06928) with a PyTorch PINN on an Apple M4 laptop. Papers are mentioned only where they have (or should have had) a code repository. The literature survey is separate.

How this was checked (1 Nov 2026):
- Every GitHub entry was confirmed with `gh api repos/OWNER/REPO`. Stars and "pushed" dates come from that call on 1 Nov 2026. READMEs and file lists were read through the API.
- No code was cloned, downloaded or run. Frameworks ("PyTorch", "JAX", ...) were read from READMEs or source file headers.
- Entries marked **(unreviewed)** are small repos (fewer than ~10 stars) whose correctness nobody has vetted. Use them as starting points, not as ground truth.

Two practical constraints for the M4 laptop apply throughout:
1. PyTorch's MPS backend does not support float64. PINN residuals need second derivatives, and L-BFGS fine-tuning usually needs double precision. Plan on CPU float64 for the final L-BFGS stage, and use MPS (float32) only for the Adam stage if it is actually faster for small MLPs.
2. JAX code (jaxpi, CausalPINNs, etc.) is GPU-oriented, and jaxpi's README says "Our code is GPU-only". Read it as a reference and port the ideas to PyTorch. Don't plan to run it.

---

## 1. General PINN frameworks and libraries

| Name | URL | Language/backend | Stars & last activity | What it does | How it helps THIS project |
|---|---|---|---|---|---|
| DeepXDE | https://github.com/lululxvi/deepxde | Python; backends TF1, TF2, PyTorch, JAX, Paddle | 4,494; pushed 2026-08-18 | Mature PINN library. CSG/polygon geometries, Dirichlet/Neumann/periodic BCs, hard constraints via approximate distance functions, residual-based adaptive sampling, Adam + L-BFGS. | Fits a laptop (PyTorch backend, CPU). Use it for a quick Newtonian (De = 0) contraction run to cross-check your own PyTorch code, and as a reference for its RAR and `output_transform` (hard-BC) patterns. Its abstractions make staged training or De-continuation awkward, so the main solver is better hand-written. |
| PINA | https://github.com/PINA-org/PINA | PyTorch + PyTorch Lightning | 800; pushed 2026-11-01 | SciML library for PINNs and neural operators, with a Problem/Condition API and a Lightning trainer. (`mathLab/PINA` is now a fork pointing here.) | Fits a laptop; Lightning has an MPS accelerator. Reasonable if you want a framework, but it has fewer PINN-specific training tricks built in than DeepXDE. Optional. |
| NVIDIA PhysicsNeMo (formerly Modulus) | https://github.com/NVIDIA/physicsnemo | PyTorch, CUDA-centred | 3,411; pushed 2026-10-30 | Large physics-ML framework. The symbolic PINN layer (`physicsnemo-sym`, https://github.com/NVIDIA/physicsnemo-sym) was archived in 2026 and moved into `nvidia-physicsnemo[sym]`, with breaking API changes (v2.0 migration guide). | Skip for this project: heavy, CUDA-oriented, and its API is in flux. Its parameterised-geometry examples are only worth reading. |
| jaxpi (incl. PirateNets) | https://github.com/PredictiveIntelligenceLab/jaxpi (PirateNets/SOAP: `pirate` branch, https://github.com/PredictiveIntelligenceLab/jaxpi/tree/pirate) | JAX; README says GPU-only | 446; pushed 2025-11-14 | Reference implementation of the Perdikaris group's training recipe: grad-norm and NTK loss weighting, causal training, random weight factorisation, Fourier features, PirateNets, SOAP (second-order optimiser). Examples include `ldc`, `stokes_cylinder`, `ns_steady_cylinder`, `backwards_step`. | Don't run it on the M4. Read the steady-flow examples and the weighting code, then port grad-norm weighting and random weight factorisation to PyTorch (each is a few dozen lines). The `ldc` example trains with a Reynolds-number curriculum (`training.Re = [100, 400, 1000]` in its README), which is the same idea as continuation in De. |
| PINNacle | https://github.com/i207M/PINNacle | Python, built on DeepXDE (PyTorch backend) | 459; pushed 2026-07-24 | NeurIPS 2024 benchmark: 20+ PDEs and about 10 methods (Adam+L-BFGS, LRA, NTK weighting, RAR, MultiAdam, gPINN, hp-VPINN, LAAF, FBPINN) in one codebase. | PyTorch-side versions of LRA/NTK weighting and RAR to copy from. Also a template for running a fair ablation of training tricks. |
| PINNs-Torch | https://github.com/rezaakb/pinns-torch | PyTorch + Lightning + Hydra | 962; pushed 2026-02-08 | PyTorch re-implementation of Raissi's PINN examples (incl. Navier–Stokes), with CUDA-graph/TorchScript speed-ups. | The speed-ups need CUDA. It is still a clean PyTorch reference for NS-type residuals with autograd. Moderate use. |
| TorchPhysics | https://github.com/Qewton-Labs/torchphysics (moved from boschresearch) | PyTorch | 470; pushed 2026-06-12 | PINNs, Deep Ritz, DeepONet, with explicit domain and parameter-space samplers. | Shows a clean way to sample (x, y, De) jointly for a parametric PINN. Optional. |
| NeuroDiffEq | https://github.com/NeuroDiffGym/neurodiffeq | PyTorch | 791; pushed 2026-04-22 | ODE/PDE solver that builds BCs exactly into the network output (reparameterisation). | Handy for a quick check of 1D lubrication ODEs (for example the leading-order velocity profile) with exact BCs. Not suited to the full 2D coupled system. |
| NeuralPDE.jl | https://github.com/SciML/NeuralPDE.jl | Julia | 1,237; pushed 2026-10-30 | Symbolic PDE (ModelingToolkit) to PINN. Well maintained. | Skip unless you switch to Julia. |

Not tabulated, checked and skipped: SciANN (https://github.com/ehsanhaghighat/sciann, Keras/TF, 368 stars, last push 2023-12-20; stale) and the original Raissi code (https://github.com/maziarraissi/PINNs, TF1, 6,234 stars; background only).

---

## 2. Neural-network code for viscoelastic, non-Newtonian and rheology problems

Public, reviewed PINN code for Oldroyd-B channel flows is scarce. The well-known papers in this area did not publish code (see the note under the table).

| Name | URL | Language/backend | Stars & last activity | What it does | How it helps THIS project |
|---|---|---|---|---|---|
| PINN-tesi (Oldroyd-B / PTT / Giesekus PINN) **(unreviewed)** | https://github.com/Eaglw/PINN-tesi | PyTorch | 0; created 2025-11, pushed 2026-10-28 | Italian MSc-thesis repo. 2D PINN for Oldroyd-B, PTT and Giesekus with a stream function ψ (exact incompressibility), three separate MLPs for ψ, p and τ, and staged training: Adam fp32 with pressure frozen, then Adam joint, then L-BFGS in float64. Includes inverse parameter estimation and COMSOL reference data (`.mph` and `.csv`) for a channel and a channel with a restriction (`COMSOL/Channel/`). | Closest public analogue to this project. Read `Viscoelastic_physics.py` (residuals and BCs) and `trainer.py` (Adam to L-BFGS, fp32 to fp64 switch). Check the equations yourself; the README is in Italian. |
| jax_rheology (Sunol, Brenner et al.) | https://github.com/asunol/jax_rheology | JAX (README says the CPU build works) | 2; pushed 2026-09-04 | Code for arXiv 2510.24673: differentiable CFD for generalised-Newtonian and viscoelastic flows, plus tensor-basis neural-network constitutive closures learned from flow data. | Not a PINN. A reference for writing Oldroyd-B/Giesekus stress equations in a differentiable framework. A possible "future work" direction: learn the constitutive law from the contraction flow. |
| RUDE (Lennon, McKinley, Swan) | https://github.com/krlennon/rude | Julia (Flux) + OpenFOAM v9 / rheoTool case files | 22; pushed 2022-12-23 | Rheological universal differential equations (PNAS 2023). Ships complete rheoTool setups for a 4:1 contraction with Giesekus and log-conformation Oldroyd-B (`blockMeshDict`, `fvSchemes`, `fvSolution`, `constitutiveProperties`), plus simulation output. | Use the `giesekus/OpenFOAM/Oldroyd-BLog` case as a ready template for your own rheoTool hyperbolic-contraction benchmark. Replace the mesh, keep the solver settings. |
| RhINNs (Jamali lab, Northeastern) | https://github.com/procf/RhINNs | TensorFlow 2 | 24; pushed 2024-04-08 | Rheology-informed NNs: constitutive-model selection from steady-shear data, thixotropic LAOS/start-up, multi-fidelity NN. | Limited: rheometric (0D/1D) problems, no flow fields. Cite for context; don't build on it. |
| RheOFormer (Jamali lab) **(unreviewed)** | https://github.com/procf/RheOFormer | PyTorch (JAX for 1D data generation) | 0; pushed 2026-05-08 | Transformer neural operator (OFormer) for viscoelastic flows. Its configs include `2d_channel.yaml` (a contraction channel) on irregular meshes. | Not a PINN. Possible comparison point in a discussion section. Check whether the 2D data are actually shipped before relying on it. |
| PINN-gPTT-Viscoelastic-Flow **(unreviewed)** | https://github.com/MhdPasandideh/PINN-gPTT-Viscoelastic-Flow | PyTorch | 0; created 2026-10-28 | PINN for generalised Phan-Thien–Tanner flow in a channel with an obstacle, compared with ANSYS (paper PDF included). | Very new. Glance at how the stress equations are written; low priority. |

Papers checked for code, none found:
- **ViscoelasticNet** (Thakur, Raissi, Ardekani; arXiv 2209.06972). TensorFlow; stenosis and cross-slot flows with rheoTool reference data; Oldroyd-B/Giesekus/PTT model selection. No code link in the paper and no GitHub repo found.
- **nn-PINNs** (Mahmoudabadbozchelou, Karniadakis, Jamali; Soft Matter 2022). No repo found, including in the group's GitHub org `procf`.
- **Suresh & Gupta**, "PINNs for viscoelastic fluid flows around a cylinder in a 2D channel" (arXiv 2609.07861, Sept 2026). Oldroyd-B, with a Cholesky-factored conformation tensor to keep it positive-definite, plus transfer learning. The arXiv page lists no code.

These are method references for the literature survey. The Cholesky parameterisation and staged training are worth copying.

---

## 3. Lubrication, thin-gap, stenosis and variable-geometry flows

| Name | URL | Language/backend | Stars & last activity | What it does | How it helps THIS project |
|---|---|---|---|---|---|
| LabelFree-DNN-Surrogate (Sun, Gao, Pan, Wang 2020) | https://github.com/Jianxun-Wang/LabelFree-DNN-Surrogate | PyTorch (plus TF version) | 100; pushed 2023-08-17 | Data-free parametric PINN surrogates: pipe flow with viscosity as a network input, and a parametric aneurysm/stenosis geometry. BCs are imposed exactly by multiplying the network output by distance-type functions. | Best single reference for two things this project needs: (a) exact no-slip on a curved wall y = h(z) and exact symmetry on y = 0; (b) feeding a physical parameter (here De) as an extra input to get one network for all De. |
| PINN-wss (Arzani, Wang, D'Souza 2021) | https://github.com/amir-cardiolab/PINN-wss | PyTorch | 79; pushed 2024-02-21 | PINN Navier–Stokes in 2D stenosis and aneurysm geometries, with data included and Torch-to-VTK export. | PyTorch NS residuals in a constricted channel, input scaling (the README warns about normalised vs physical coordinates), and computing wall quantities from a trained PINN. |
| PhyGeoNet (Gao, Sun, Wang 2021) | https://github.com/Jianxun-Wang/phygeonet | PyTorch | 91; pushed 2021-01-24 | Physics-informed CNN on irregular domains via an elliptic map to a rectangle; parametric geometries. | Borrow the idea, not the code (it is CNN-based): map the contraction to a rectangle with η = y/h(z). That is the natural lubrication coordinate, and it makes the wall and centreline BCs trivial to hard-code. |
| Hidden Fluid Mechanics (Raissi et al. 2020) | https://github.com/maziarraissi/HFM | TensorFlow 1 | 372; pushed 2023-01-30 | NS-informed inference of velocity and pressure in stenosis and aneurysm geometries from dye data. | Background only (TF1). |
| Reynolds_PINN_Notebook | https://github.com/gbdiazc/Reynolds_PINN_Notebook | Python/NumPy notebook (no DL library detected) | 5; pushed 2024-12-03 | PINN for the 1D Reynolds equation, ported from Almqvist's MATLAB example (Sánchez López & Díaz Cortés 2024). | A worked lubrication-PINN example. Useful for a first sanity test of the leading-order (Newtonian) q–Δp relation before the 2D model. |
| isothermal-squeeze-film-giesekus **(unreviewed)** | https://github.com/nigeldhlamini/isothermal-squeeze-film-giesekus | Python (with tests) | 0; pushed 2026-06-07 | Viscoelastic Reynolds equation from a Deborah-number perturbation expansion of the Giesekus model (squeeze-film bearings). | Not a PINN. Its structure (perturbation orders in De, closed-form corrections) resembles Boyko & Stone's low-De expansion, so it shows one way to code the analytical benchmark. |

---

## 4. Reference code for PINN training techniques

| Name | URL | Language/backend | Stars & last activity | What it does | How it helps THIS project |
|---|---|---|---|---|---|
| GradientPathologiesPINNs (learning-rate annealing) | https://github.com/PredictiveIntelligenceLab/GradientPathologiesPINNs | TensorFlow 1 | 273; pushed 2021-10-14 | Wang, Teng, Perdikaris 2021: loss weights set from gradient-norm ratios. The NTK-weighting companion is https://github.com/PredictiveIntelligenceLab/PINNsNTK (TF/JAX, 119 stars). | Read for the update formula, then implement in PyTorch. Six stress/momentum residuals plus BC and flow-rate losses have very different scales, so some weighting is needed. |
| SA-PINNs (self-adaptive weights) | https://github.com/levimcclenny/SA-PINNs | TensorFlow 2 / Keras | 301; pushed 2021-10-12 | Point-wise trainable loss weights updated by gradient ascent; includes an eager L-BFGS helper. | Alternative to grad-norm weighting when errors concentrate near the throat or outlet. |
| rba-pinns (residual-based attention) | https://github.com/soanagno/rba-pinns | JAX and PyTorch notebooks | 79; pushed 2025-09-11 | Point-wise weights from a running average of residuals, with no extra gradient computations. | Cheapest local-weighting option, and PyTorch notebooks exist. A good default for stress residuals that grow near the contraction exit. |
| ConFIG | https://github.com/tum-pbs/ConFIG | PyTorch (pip package) | 125; pushed 2026-06-01 | ICLR 2025: conflict-free gradient combination for multi-term PINN losses. | Drop-in alternative to hand-tuned weights if the PDE, BC and flow-rate losses pull in different directions. |
| pinn-sampling (RAD / RAR-D) | https://github.com/lu-group/pinn-sampling | DeepXDE | 107; pushed 2022-10-24 | Wu et al. 2023: comparison of uniform vs residual-based adaptive sampling (RAD, RAR-D). | Use RAD to put more collocation points near the narrow outlet, where velocity gradients and polymer stresses peak. |
| characterizing-pinns-failure-modes (curriculum) | https://github.com/a1k12/characterizing-pinns-failure-modes | PyTorch | 152; pushed 2021-11-18 | Krishnapriyan et al. 2021: curriculum regularisation, training at an easy parameter value first and then increasing it. | Template for continuation in De: train at De = 0 (Newtonian), then warm-start at De = 0.05, 0.1, ... 0.3. Matches how the COMSOL runs sweep λ. |
| opt_for_pinns | https://github.com/pratikrathore8/opt_for_pinns | PyTorch | 46; pushed 2025-03-04 | Rathore et al. 2024: Adam, then L-BFGS, then NysNewton-CG for PINNs; loss-landscape analysis. | Use their Adam-to-L-BFGS schedule. Run the L-BFGS stage on CPU in float64 (MPS has no float64). |
| torch.optim.LBFGS (docs) | https://docs.pytorch.org/docs/stable/generated/torch.optim.LBFGS.html | PyTorch built-in | n/a | Full-batch L-BFGS with optional strong-Wolfe line search. | The optimiser for the fine-tuning stage; use `line_search_fn="strong_wolfe"` and a fixed collocation set. |
| hPINN (hard constraints / augmented Lagrangian) | https://github.com/lululxvi/hpinn | DeepXDE 0.9.1 (old API) | 166; pushed 2021-11-21 | Lu et al. 2021: penalty and augmented-Lagrangian constraint enforcement, including a Stokes-flow example. | Reference for enforcing a global constraint such as a fixed flow rate q = ∫u dy at every z. |
| Fourier feature networks | https://github.com/tancik/fourier-feature-networks (PINN version: https://github.com/PredictiveIntelligenceLab/MultiscalePINNs, TF1) | JAX / TF1 | 1,376; pushed 2023-01-17 | Random Fourier input features to fight spectral bias. | Low priority: the lubrication-scale flow is smooth. Consider only if stress boundary layers at the wall are under-resolved at higher De. |

Also covered elsewhere: causal training (https://github.com/PredictiveIntelligenceLab/CausalPINNs, JAX, 222 stars) targets time-dependent problems and matters little for this steady flow. Physics-informed DeepONets (https://github.com/PredictiveIntelligenceLab/Physics-informed-DeepONets, JAX, 433 stars) are relevant only if you want a surrogate over De rather than a De-input PINN. jaxpi (section 1) collects most of these in one place.

---

## 5. Conventional CFD tools for an independent viscoelastic benchmark

| Name | URL | Language/backend | Stars & last activity | What it does | How it helps THIS project |
|---|---|---|---|---|---|
| rheoTool (OpenFOAM) | https://github.com/fppimenta/rheoTool | C++ on OpenFOAM v9 / v7 / foam-extend 4.0 | 213; pushed 2024-02-25 | Finite-volume toolbox for GNF and viscoelastic flows (Oldroyd-B, Giesekus, FENE-P, PTT, ...) with log-conformation and coupled solvers. OpenFOAM-9 `rheoFoam` tutorials include `Channel`, `Contraction41`, `CrossSlot`, `Cylinder`. The user guide PDF is in `doc/`. | Best free independent benchmark. Build a 2D half-channel `blockMesh` of H(Z) = α/((α−1)Z+1) with straight entrance/exit sections, as in Boyko & Stone's Fig. 9, and sweep De = 0–0.3 at Re ≈ 0. Start from the `Contraction41` tutorial or the RUDE case files. On macOS the usual route for OpenFOAM is Docker or a Linux VM. |
| COMSOL Multiphysics, Polymer Flow Module (commercial) | https://www.comsol.com/polymer-flow-module | Commercial FEM | n/a | Viscoelastic interface with Oldroyd-B, Giesekus, FENE-P. | This is what Boyko & Stone used (COMSOL 5.6). If IIT Delhi has a licence, it is the most direct way to reproduce their Table 2 runs exactly. PINN-tesi (section 2) ships COMSOL `.mph` files for an Oldroyd-B channel that could serve as a setup template. |
| FEniCSx (DOLFINx) | https://github.com/FEniCS/dolfinx; Stokes demo: https://docs.fenicsproject.org/dolfinx/main/python/demos/demo_stokes.html | Python / C++ | 1,242; pushed 2026-10-30 | General FEM; no official viscoelastic demo. | Write your own steady Oldroyd-B solver with P2 velocity, P1 pressure, P1 stress (the same elements as the paper's COMSOL runs), SUPG or DEVSS stabilisation and Newton, continuing in De. Medium effort, but it gives a fully open, scriptable benchmark that runs natively on the Mac via conda-forge. |
| Viscoelastic_fluid_flow_using_dolfinx **(unreviewed)** | https://github.com/SachinBM-CE/Viscoelastic_fluid_flow_using_dolfinx | DOLFINx notebooks | 0; pushed 2024-08-21 | Student notebooks: Oldroyd-B with SUPG in DOLFINx. | Starting point for the DOLFINx weak form; verify before trusting. |
| ATMackay/fenics **(unreviewed)** | https://github.com/ATMackay/fenics | Legacy FEniCS (dolfin) | 3; pushed 2024-02-11 | Oldroyd-B and related models in a journal bearing (eccentric cylinders) and a lid-driven cavity. | Reference for stabilised Oldroyd-B weak forms. Legacy syntax, so translate to DOLFINx. |
| Firedrake | https://github.com/firedrakeproject/firedrake | Python FEM | 681; pushed 2026-10-31 | Automated FEM. Example viscoelastic research code: https://github.com/cachja/rod-climbing (Johnson–Segalman–Giesekus, axisymmetric ALE). | Alternative to FEniCSx with similar effort. Pick whichever installs more easily. |
| Basilisk (log-conformation) | http://basilisk.fr/src/log-conform.h; test: http://basilisk.fr/src/test/poiseuille-oldroydb.c | C | n/a (not on GitHub) | Log-conformation solver (Oldroyd-B, FENE-P); tests include transient planar Poiseuille Oldroyd-B. 3D extension: https://github.com/comphy-lab/Viscoelastic3D. | The Poiseuille test checks the inlet state (fully developed Oldroyd-B channel flow). A contraction needs embedded boundaries and time-marching to steady state, so this is lower priority than rheoTool or FEniCSx. |
| Dedalus | https://github.com/DedalusProject/dedalus | Python, spectral | 713; pushed 2026-10-30 | Spectral PDE framework. Viscoelastic example (Kolmogorov flow): https://github.com/beccathomases/dedalus_KOL_ve. | Only sensible if you solve in the mapped rectangle (z, η = y/h(z)) with Chebyshev bases. Otherwise skip. |
| pyoomph | https://github.com/pyoomph/pyoomph | Python front-end to oomph-lib | 18; pushed 2026-10-04 | Multiphysics FEM with a viscoelastic equation module (`pyoomph/equations/viscoelastic.py`) and built-in parameter continuation. | Interesting for clean continuation in De; early-stage software by its own README. Optional. |
| Goma (Sandia) | https://github.com/goma/goma | C | 127; pushed 2026-10-26 | FEM for free-surface, non-Newtonian and viscoelastic flows. | Skip: heavy build, overkill here. |

No simple, trustworthy Python finite-difference Oldroyd-B contraction solver turned up on GitHub. The cheapest independent check is to re-implement the lubrication finite-difference scheme of Hinch, Boyko & Stone (2024) described in section 6. The authors report it runs in under a minute on a laptop.

---

## 6. Public code or data from Boyko, Stone and co-authors

Result: **no public code or data found** for any of the three papers. Searches covered GitHub (names, "hyperbolic contraction", "narrow channel viscoelastic"), Zenodo (only unrelated taxonomy records by a different Boyko), and the papers' end matter. The papers do, however, specify their numerics in enough detail to rebuild the benchmark.

| Paper | Where checked | Code/data statement | Numerics usable for this project |
|---|---|---|---|
| Boyko & Stone 2022, JFM 936 A23 (doi:10.1017/jfm.2022.67; arXiv 2109.06928) | Full arXiv text (local PDF) and Cambridge HTML (open access, CC BY) | None. Only Acknowledgements, Funding and Declaration of interests. | Appendix A: COMSOL 5.6 viscoelastic flow module; half domain with a symmetry BC on y = 0; straight entrance/exit sections of length ℓ; fully developed Poiseuille velocity **and** Poiseuille polymer stress imposed at the inlet; p = 0 at the exit on y = 0; Δp taken along the centreline between z = 0 and z = ℓ. P2 velocity, P1 pressure and stress; about 84,000 elements (ε = 0.02) and 17,000 (ε = 0.1); PARDISO, relative tolerance 1e-5. Table 2: α = 4, η0 = 1 Pa s, ηs = 0.6, ηp = 0.4 (β̃ = 0.4), ℓ = 5 mm, u_c = 5 mm/s, λ = 0–0.3 s (De = 0–0.3); Case I ε = 0.02 (Re ≈ 5e-7), Case II ε = 0.1 (Re ≈ 2.5e-6). This is a complete benchmark specification. |
| Hinch, Boyko & Stone 2024, JFM 988 A11 (doi:10.1017/jfm.2024.260) | Cambridge HTML, both halves of the text (open access, CC BY) | None found. Funding (BSF, NSF) and Declaration of interests only. | Own finite-difference lubrication solver: equispaced grid, second-order central differences except first-order upwind for streamwise advection, first-order forward time-stepping to steady state (δx = 0.0208, δη = 0.0333, δt = 1e-3). Reported to struggle beyond De = 1 and to take under a minute on a laptop. Its FE comparisons reuse the Boyko & Stone (2022) COMSOL code. |
| Boyko, Hinch & Stone 2024, JFM 988 A10 (doi:10.1017/jfm.2024.223), ultra-dilute limit, arbitrary De | Cambridge HTML, both halves (open access, CC BY) | None found. | Semi-analytical (closed forms plus MATLAB `cumtrapz` quadrature). FE comparisons again point to the Boyko & Stone COMSOL implementation. |

Practical consequence: the reference numbers for the PINN must come from (i) the analytical low-De expressions in the 2022 paper (ΔP0 to ΔP3), (ii) your own re-run of the Table 2 cases in rheoTool, FEniCSx or COMSOL, and (iii) optionally a re-implementation of the 2024 lubrication finite-difference scheme. Digitising the published figures is a fallback for spot checks only.

---

## Top picks for this project

1. **Eaglw/PINN-tesi** (https://github.com/Eaglw/PINN-tesi): read its Oldroyd-B residuals, stream-function formulation and Adam-to-L-BFGS(float64) staged trainer before writing your own. It is the closest existing code, but unreviewed.
2. **LabelFree-DNN-Surrogate** (https://github.com/Jianxun-Wang/LabelFree-DNN-Surrogate): copy the hard-BC construction (exact no-slip on y = h(z), symmetry on y = 0) and the "physical parameter as network input" design to make De an input.
3. **characterizing-pinns-failure-modes** (https://github.com/a1k12/characterizing-pinns-failure-modes): use its curriculum pattern for continuation from De = 0 to 0.3, warm-starting each step.
4. **rheoTool + RUDE case files** (https://github.com/fppimenta/rheoTool, https://github.com/krlennon/rude): build the independent finite-volume benchmark of Boyko & Stone's Table 2 cases from the `Contraction41` / RUDE Oldroyd-B-log setups. Use COMSOL instead if a licence is available.
5. **jaxpi** (https://github.com/PredictiveIntelligenceLab/jaxpi): treat it as the reference for grad-norm loss weighting, random weight factorisation and PirateNet blocks, and port the pieces you need to PyTorch.
6. **lu-group/pinn-sampling** (https://github.com/lu-group/pinn-sampling): implement RAD resampling to concentrate collocation points near the contraction outlet, where stresses peak.
7. **opt_for_pinns + torch.optim.LBFGS** (https://github.com/pratikrathore8/opt_for_pinns): follow its Adam-then-L-BFGS schedule, running L-BFGS on CPU in float64 because MPS has no float64.
8. **DeepXDE** (https://github.com/lululxvi/deepxde): quick independent PINN for the Newtonian (De = 0) contraction, to check your hand-written PyTorch code against a mature library before adding the polymer stress.

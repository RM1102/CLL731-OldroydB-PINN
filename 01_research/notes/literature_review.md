# Literature review: benchmarks and PINN methods for Oldroyd-B flow in a narrow contraction

Compiled 1 November 2026 for the CLL731 project (re-solving Boyko & Stone 2022, JFM 936 A23, with a physics-informed neural network).

## Introduction

The target problem is steady, inertialess, pressure-driven Oldroyd-B flow through a narrow, slowly varying 2D contraction (the hyperbolic shape of Boyko & Stone 2022), with the flow rate q fixed and the pressure drop Δp as the main output. Since 2022 this problem has become a small benchmark family. Low-Deborah-number lubrication results now exist to O(De^8) with Padé resummation, there are semi-analytical results for arbitrary De in the ultra-dilute limit, there is a high-De asymptotic theory, and there are several sets of 2D finite-element and pseudospectral solutions to compare against. Separately, PINN work on viscoelastic flows is still thin. Most published PINN studies are inverse problems (stress or model discovery from velocity data), and the few data-free forward solvers report training failures at moderate Weissenberg numbers. This note collects (A) the analytical and numerical benchmarks around the Boyko–Stone problem, (B) what PINN and related methods have managed for non-Newtonian and viscoelastic flows, (C) general PINN training practice relevant to a stiff coupled Stokes + constitutive system, and (D) the known numerical difficulties of Oldroyd-B. It ends with design recommendations. I do not summarise Boyko & Stone (2022) itself.

How things were checked: every reference below was confirmed on a real page (arXiv abstract page, publisher page, or Crossref metadata for the DOI). Where I could read only the abstract or a search-engine snippet, and not the full text, the entry says so. Formulas quoted from papers were read from publisher HTML through an automated text extractor, so they must be re-checked against the PDFs before being used as numerical benchmarks. The places where this matters most are flagged with "(verify)".

A note on notation. The papers below use different Deborah numbers and different viscosity-ratio parameters, and these have to be converted before any comparison:
- Boyko, Hinch & Stone (2024) use De = λu_c/ℓ, Wi = λu_c/h₀, De = εWi, and β̃ = μ_p/(μ_s + μ_p) (polymer fraction).
- Hinch, Boyko & Stone (2024) use c = μ_p/μ_s, so β̃ = c/(1 + c).
- Housiadas & Beris use De = λ*ℓ*/u_c* (the AIP page describes u_c* as twice the mean inlet velocity (verify)), η = η_p/(η_s + η_p), Λ = h₀/h_f (inlet-to-outlet half-height ratio), and a modified Deborah number De_m = 3(Λ − 1)De/4, which equals λ times the centreline extension rate of the Newtonian solution.
- Housiadas (2025) states that his O(De_m^8) series agrees with Boyko & Stone (2022) up to O(De^3) once the different De definitions are converted.

---

## A. Benchmarks from the Boyko–Stone line of work and related lubrication theory

### A1. Reciprocal-theorem q–Δp relations (Boyko, Stone, Christov and co-workers)

**Boyko & Stone (2021), Phys. Rev. Fluids 6, L081301.** [doi:10.1103/PhysRevFluids.6.L081301](https://doi.org/10.1103/PhysRevFluids.6.L081301)
This paper uses the Lorentz reciprocal theorem to write the q–Δp relation of a weakly non-Newtonian fluid in a narrow channel as an integral of the non-Newtonian stress evaluated on the Newtonian solution, so the first-order correction needs no non-Newtonian flow solve. For a second-order fluid the first correction can raise or lower Δp depending on geometry; for a Carreau fluid it always lowers Δp. Use for the PINN: it gives an independent way to compute Δp from a PINN velocity and stress field, as a volume integral of the polymer stress against the Newtonian kinematics, instead of reading Δp off the wall pressure. A disagreement between the two estimates is a useful error indicator.

**Boyko, Stone & Christov (2022), Phys. Rev. Fluids 7, L092201.** [doi:10.1103/PhysRevFluids.7.L092201](https://doi.org/10.1103/PhysRevFluids.7.L092201)
This applies the fluidic and elastic reciprocal theorems together to get a closed-form q–Δp relation for a deformable channel. Compliance lowers Δp and sidewall drag raises it. It is not directly useful as a benchmark here, since our walls are rigid, but it shows the reciprocal-theorem method is general.

**Boyko & Christov (2023), J. Non-Newtonian Fluid Mech. 313, 104990.** [doi:10.1016/j.jnnfm.2023.104990](https://doi.org/10.1016/j.jnnfm.2023.104990)
This treats Oldroyd-B flow in a slender channel with a deformable wall, using lubrication theory and an expansion in De, and gives a closed-form pressure to O(De). Viscoelasticity lowers Δp and wall deformation. It is not a benchmark for the rigid case, but the O(De) structure it reports is the same as in the rigid problem.

**Chun, Boyko, Christov & Feng (2024), Phys. Rev. Fluids 9, 043302.** [doi:10.1103/PhysRevFluids.9.043302](https://doi.org/10.1103/PhysRevFluids.9.043302) (the title says "relations", plural)
This combines experiments and theory for Newtonian and power-law (shear-thinning) fluids in deformable channels and tubes. It is relevant only if the project later adds shear-thinning.

**Review.** I could not find a Boyko-authored review titled "Non-Newtonian fluid mechanics in narrow geometries" or similar (searched Crossref, arXiv and general web, November 2026). The closest review I found is **Christov (2022), "Soft hydraulics: from Newtonian to complex fluid flows through compliant conduits", J. Phys.: Condens. Matter 34, 063001** ([doi:10.1088/1361-648X/ac327d](https://doi.org/10.1088/1361-648X/ac327d)), which covers lubrication q–Δp relations for complex fluids, but in compliant conduits.

### A2. High-De and arbitrary-De results (Hinch, Boyko, Stone and co-workers)

**Hinch, Boyko & Stone (2024), "Fast flow of an Oldroyd-B model fluid through a narrow slowly varying contraction", J. Fluid Mech. 988, A11.** [doi:10.1017/jfm.2024.260](https://doi.org/10.1017/jfm.2024.260)
This is the high-De companion to Boyko & Stone (2022). Lubrication theory is modified to keep the large normal stresses that appear when De = O(1). The lubrication equations are solved numerically: the flow is computed from the current stresses, and the Oldroyd-B equations are time-stepped to steady state in a mapped coordinate η = y/h. The results are compared with 2D finite-element solutions at ε = 0.02. The facts most useful for a PINN:
- Shape used: h = 1 − (1 − H⁻¹)x²(2 − x)² on 0 < x < 1, with straight sections before and after. This shape has zero slope at both ends and is **not** the hyperbolic shape of Boyko & Stone (2022). Contraction ratios H = h(0)/h(ℓ) = √2, 2, 2√2 and 4 were used, mostly H = 2, with c = 1 in most runs.
- Inlet condition, needed because the constitutive equation is hyperbolic: Poiseuille flow with the steady simple-shear microstructure, u = (3/2h)(1 − η²), A₂₂ = 1, A₁₂ = −3Deη/h², A₁₁ = 18De²η²/h⁴.
- The pressure drop is lower than for a Newtonian fluid with the same shear viscosity through a contraction, and higher through an expansion. Both changes become linear in De at high De. High-De fit (valid for De > 0.3 to 0.4): Δp ≈ Δp₁ + cΔp₂ − (9cDe/5)(H² − 1), where Δp₁ is the Newtonian solvent drop and Δp₂ = ∫₀¹ 3/h dx. Low-De form given in the same paper: Δp ≈ (1 + c)Δp₁ − (9/2)cDe(H⁴ − 1). For H = 4, Δp₁ = 48.2 and Δp₂ = 6.249 (Table 1 of the paper) (verify).
- Wall stress boundary layer of thickness δ ≈ 0.04/De, "universal" for De > 0.1. Stresses are largest there, with A₁₁ = O(De²H⁴) and A₁₂ = O(DeH²).
- For a contraction, much of the change in Δp occurs in the long exit channel, where the stresses relax. For a contraction followed by an expansion (a constriction), the Δp reduction plateaus at high De.
- No recirculation was found in these slowly varying geometries.

Benchmark value: this gives the high-De trend (linear decrease of Δp with De, slope −(9c/5)(H² − 1)) and the inlet stress profile. For the hyperbolic shape the slope has to be recomputed, or the PINN has to be run on Hinch's shape.

**Boyko, Hinch & Stone (2024), "Flow of an Oldroyd-B fluid in a slowly varying contraction: theoretical results for arbitrary values of Deborah number in the ultra-dilute limit", J. Fluid Mech. 988, A10.** [doi:10.1017/jfm.2024.223](https://doi.org/10.1017/jfm.2024.223)
For β̃ ≪ 1 the velocity stays Newtonian and parabolic, so the conformation tensor is one-way coupled to a known flow. This gives closed-form conformation fields, written as integrals in terms of De·U₀(Z,η) with U₀ = (3/2)(1 − η²)/H, and a q–Δp relation valid for any De and any shape H(Z). Pressure drop: ΔP = ΔP₀ + β̃ΔP₁ + O(β̃²), with ΔP₀ = 3∫₀¹dZ/H³. The O(β̃) term ΔP₁ is a quadrature of the closed-form stresses.
- Low De: ΔP₁ ≈ (9/2)De(1 − H_ℓ⁻⁴), where H_ℓ = h_ℓ/h₀ < 1. This is the same O(De) term as Hinch et al.
- High De: ΔP₁ ≈ −3∫dZ/H³ + 3∫dZ/H + (9/5)De(1 − H_ℓ⁻²). In the core, Ã₂₂ = H², Ã₁₂ = −3Deη and Ã₁₁ = 18De²η²/H²; near the walls there is an O(1/De) boundary layer in ζ = De(1 − η).
- Exit channel: low De, ΔP = 3L/H_ℓ³ − 1728β̃De³H″(1)/(35H_ℓ⁷); high De, ΔP = 3L/H_ℓ³ + (36/5)β̃De(H_ℓ⁻² − H_ℓ⁻⁴). The pressure gradient relaxes to 1% of its peak over a length of about (5.3 ± 0.5) × 3De/(2H_ℓ) (verify all three).
- Validation: COMSOL 2D finite elements (set up as in Boyko & Stone 2022) at H_ℓ = 0.5, β̃ = 0.05, ε = 0.02, described as "excellent" agreement. The authors note they could not reach De = 2 or 10 with finite elements.
- They also note that Oldroyd-B has no dissipation mechanism beyond what is built in, which may explain why its predicted Δp reduction disagrees with experiments on Boger fluids that show an increase.

Benchmark value: this is the most useful arbitrary-De check. Because it holds for any H(Z), it can be evaluated for the hyperbolic shape of Boyko & Stone (2022) by quadrature. It also gives a decoupled test problem: freeze the velocity at the Newtonian lubrication solution and ask the PINN to solve only the constitutive equation. The exact answer is then known in closed form.

**Mahapatra, Ruangkriengsin, Stone & Boyko (2025), J. Fluid Mech. 1009, A12.** [doi:10.1017/jfm.2025.142](https://doi.org/10.1017/jfm.2025.142), [arXiv:2409.08150](https://arxiv.org/abs/2409.08150)
Same framework with the FENE-CR model (finite extensibility). Gives analytical Δp to O(De⁴) at low De, all-De results in the ultra-dilute limit, and 2D simulations that agree. FENE-CR Δp is non-monotonic at high De (first decreasing, then increasing) but stays below the Newtonian value. Useful if the project wants a second constitutive model, since FENE-CR has bounded extension and avoids the Oldroyd-B blow-up at λε̇ = 1/2.

**Kedem, Mahapatra & Boyko (2026), Phys. Rev. Fluids 11, 043303.** [doi:10.1103/kyql-qp7b](https://doi.org/10.1103/kyql-qp7b), [arXiv:2510.07166](https://arxiv.org/abs/2510.07166)
Ultra-dilute Oldroyd-B in a contraction–expansion (constriction). Closed-form stresses and Δp for any De, checked against simulations. Δp plateaus at high De because the elastic normal stresses cancel and elastic shear stresses drive the reduction. Stresses relax over a much shorter exit length after a constriction than after a contraction.

### A3. High-order low-De series and "exact" lubrication solutions (Housiadas, Beris, Sialmas)

**Housiadas & Beris (2023), "Lubrication approximation of pressure-driven viscoelastic flow in a hyperbolic channel", Phys. Fluids 35, 123116.** [doi:10.1063/5.0183154](https://doi.org/10.1063/5.0183154)
This is the most directly comparable benchmark, because it uses the same kind of geometry. Shape: H(z) = 1 for z ≤ 0, H(z) = 1/((Λ − 1)z + 1) for 0 ≤ z ≤ 1, H = 1/Λ for z ≥ 1. The regular perturbation in De is carried to **eighth order**, in closed form, for UCM/Oldroyd-B (and also PTT, Giesekus and FENE-P, which differ only slightly within lubrication theory). Results:
- Newtonian: ΔΠ_N = (3/8)(1 + Λ)(1 + Λ²).
- ΔΠ/ΔΠ_N ≈ 1 + η Σ_{k=1}^{8} ΔΠ̂_k De^k, with (Oldroyd-B):
  - ΔΠ̂₁ = 3(1 − Λ)
  - ΔΠ̂₂ = (216/35)(1 − Λ)²
  - ΔΠ̂₃ = (36/35)(11 − η)(1 − Λ)³ (verify)
  - ΔΠ̂₄ = (27/13475)(9800 − 3351η + 840η²)(1 − Λ)⁴ (verify)
- The raw series converges only for very small De (about 0 < De < 0.02 for their Λ = 4 case). Diagonal Padé [M/M] approximants converge, and M = 2, 3, 4 are indistinguishable.
- Checked against high-order finite-difference and pseudospectral solutions of the lubrication equations (Λ = 4, η = 0.4; the field comparisons shown are at De = 0.2). I could not confirm from the visible text whether full 2D simulations were also run.
- The pressure drop falls as De, η or Λ increase.

**Housiadas & Beris (2024a), "Pressure-drop and Trouton ratio for Oldroyd-B fluids in hyperbolic converging channels", Phys. Fluids 36, 021702.** [doi:10.1063/5.0194278](https://doi.org/10.1063/5.0194278)
Shows that with De_m = 3(Λ − 1)De/4, ΔΠ/ΔΠ_N depends only on De_m and η (Λ drops out): ΔΠ/ΔΠ_N = 1 − η(4De_m) + (24/35)η(4De_m)² + …, to O(De_m⁸). This is consistent with the 2023 coefficients above. Padé [M/M] with M = 2, 3, 4 (5, 7 and 9 terms) are practically identical for 0 ≤ De_m ≤ 1. The Trouton-ratio series diverges near De_m ≈ 1/2, which is the Oldroyd-B extensional blow-up threshold (λε̇ = 1/2 on the centreline). The paper says the accelerated lubrication series stops tracking the exact Newtonian-kinematics solution for De_m above about 0.3.

Benchmark value of A3: closed-form Δp(De) to O(De⁸) for the hyperbolic channel, plus a recipe (Padé [4/4]) that is accurate to about De_m ≈ 1. This is the main quantitative target for a PINN q–Δp curve, after converting De definitions.

**Housiadas & Beris (2024d), "Pressure-driven viscoelastic flow in axisymmetric geometries with application to the hyperbolic pipe", J. Fluid Mech. 999, A7.** [doi:10.1017/jfm.2024.848](https://doi.org/10.1017/jfm.2024.848), [arXiv:2312.15031](https://arxiv.org/abs/2312.15031)
Gives exact general formulas for the average Δp in terms of the wall shear rate and the axial normal stress, then an O(De⁸) lubrication series with Padé, checked against pseudospectral solutions. This is the axisymmetric counterpart of the planar problem. Its formulas for Δp in terms of wall quantities are a model for how to extract Δp from a PINN.

**Housiadas (2025), "An exact solution of the lubrication equations for the Oldroyd-B model in a hyperbolic channel", arXiv:2503.20636** (preprint; I found no journal version). [arXiv:2503.20636](https://arxiv.org/abs/2503.20636)
I extracted the full text (PDF) and read the key passages, not every page. It gives a similarity solution of the planar lubrication equations, written with a streamfunction and modified stress components (τ_ss, τ_ns, τ_nn) that stay O(1) everywhere, unlike the Cartesian τ_zz = τ_ss/H⁴. It matches the exact wall solution but does not satisfy the inlet stress condition. Points relevant to a PINN:
- Because the hyperbolic shape has a slope jump at the inlet of the varying section (H′(0⁻) = 0, H′(0⁺) = −(Λ − 1)), the Cartesian vertical velocity and τ_yz, τ_yy jump there. The modified pressure gradient also jumps, from G(0⁻) = 3/2 to G(0⁺) = (3/2)(1 − ηDe_m) in that paper's De_m. This jump cannot be removed by any change of variables, and the pressure gradient becomes adverse at the inlet when ηDe_m > 1.
- The O(De_m⁸) series agrees with Housiadas & Beris (2024a) and with Boyko & Stone (2022) to O(De³).
- New pseudospectral lubrication simulations converged only for η = 0.4–0.5, 2 ≤ Λ ≤ 8 and 0 ≤ De ≤ 0.5. Beyond that, the fully implicit scheme failed or became unstable, "typically associated with the loss of positive definiteness of the conformation tensor".
- The paper argues that a high-De asymptotic analysis cannot hold near the walls, where the local residence time diverges. This is a disagreement with Hinch et al. (2024), who handle the walls with a boundary layer.

**Sialmas & Housiadas (2025), "An exact solution of the lubrication equations for the Oldroyd-B model in a hyperbolic pipe", J. Non-Newtonian Fluid Mech. 335, 105331** ([doi:10.1016/j.jnnfm.2024.105331](https://doi.org/10.1016/j.jnnfm.2024.105331)), and **Hinch (2026), "Approach to a similarity solution of the lubrication flow of an Oldroyd-B fluid through a hyperbolic pipe", J. Non-Newtonian Fluid Mech. 347, 105519** ([doi:10.1016/j.jnnfm.2025.105519](https://doi.org/10.1016/j.jnnfm.2025.105519)).
An SSRN preprint by Hinch is titled "Comment on An exact solution of the lubrication equations for the Oldroyd-B model in a hyperbolic pipe by Sialmas & Housiadas (2025)", and the JNNFM paper appears to be its published form (not confirmed). I have not read either. Going by its title, it looks at how the flow approaches that similarity solution from the inlet condition. Practical point: treat the "exact" similarity solutions as asymptotic states that are not reached at the inlet, and do not use them as pointwise benchmarks near the inlet.

**Sialmas & Housiadas (2025), "General evaluation of the pressure gradient for lubrication flows in varying channels with applications to linear and hyperbolic contractions", J. Non-Newtonian Fluid Mech. 346, 105487.** [doi:10.1016/j.jnnfm.2025.105487](https://doi.org/10.1016/j.jnnfm.2025.105487), [arXiv:2504.09563](https://arxiv.org/abs/2504.09563) (the arXiv title differs)
Derives four ways of computing the pressure gradient or average Δp for Oldroyd-B in a slowly varying channel: momentum balance, total force balance, a higher moment of the momentum balance with continuity, and mechanical energy. It proves the first two are identical and stresses the role of the exact wall solution. Use: the PINN's Δp should be computed by at least two of these routes. If they disagree, the PINN has not converged.

### A4. Other lubrication-level work

**Ahmed & Biancofiore (2021), J. Non-Newtonian Fluid Mech. 292, 104524** ([doi:10.1016/j.jnnfm.2021.104524](https://doi.org/10.1016/j.jnnfm.2021.104524)), and **Ahmed & Biancofiore (2023), J. Non-Newtonian Fluid Mech. 321, 105123** ([doi:10.1016/j.jnnfm.2023.105123](https://doi.org/10.1016/j.jnnfm.2023.105123)).
These derive a "viscoelastic Reynolds equation" for thin-film bearings (Oldroyd-B in 2021; FENE-CR and FENE-P in 2023) and validate it against direct simulations of sliders and journal bearings. They are tribology (sliding-wall) problems, not pressure-driven contractions, but they show the lubrication reduction holding up against DNS over a wide Wi range.

**Tavakol, Froehlicher, Holmes & Stone (2017), "Extended lubrication theory: improved estimates of flow in channels with variable geometry", Proc. R. Soc. A 473, 20170234.** [doi:10.1098/rspa.2017.0234](https://doi.org/10.1098/rspa.2017.0234)
Newtonian O(ε²) corrections to the lubrication q–Δp relation. Needed if the PINN solves the full 2D problem at finite ε (such as ε = 0.02–0.1) and is compared with lubrication formulas, since the Newtonian part of the discrepancy is O(ε²).

Not found: I found no recent paper by Datt or Mehta on viscoelastic lubrication in narrow channels or reciprocal theorems that fits this problem, and I have not cited any.

---

## B. PINNs and related physics-informed methods for non-Newtonian and viscoelastic flow

Summary of the state of the field: there is **no published PINN study of Oldroyd-B flow in a slowly varying contraction** that I could find. The forward, data-free viscoelastic PINN studies are recent (2026) and cover a lid-driven cavity and a cylinder in a channel. The more mature work is inverse: learning stress, parameters or the constitutive law from velocity data.

**Raissi, Perdikaris & Karniadakis (2019), J. Comput. Phys. 378, 686–707** ([doi:10.1016/j.jcp.2018.10.045](https://doi.org/10.1016/j.jcp.2018.10.045)). This is the base PINN formulation (soft PDE and BC residuals, automatic differentiation), listed here for completeness.

**Matos, Fernandes & Alves (2026), "Simulation of steady viscoelastic creeping flows with physics-informed neural networks", J. Non-Newtonian Fluid Mech. 350, 105637** (open access, CC BY). [doi:10.1016/j.jnnfm.2026.105637](https://doi.org/10.1016/j.jnnfm.2026.105637)
- Model and PDEs: steady 2D Stokes-like flow (Re = 0.01) with Oldroyd-B and linear PTT, in a lid-driven cavity. Forward problem, trained only on PDE residuals (no data).
- Reported result: steady velocity, pressure and polymer-stress fields predicted successfully.
- Search-engine summaries of the abstract also mention a comparison of a direct-stress output with an output that is the logarithm of the conformation tensor ("log(A)"), and tests at higher De. I could not open the full text (the publisher page blocked automated access), so the De range reached, the architectures and the failure cases are **unverified**. This is the closest existing work to what we are doing and should be read in full first.

**Suresh & Gupta (2026), "Physics-informed neural networks for viscoelastic fluid flows around a cylinder in a two-dimensional channel", arXiv:2609.07861** (preprint, not peer-reviewed). [arXiv:2609.07861](https://arxiv.org/abs/2609.07861)
- Model: Oldroyd-B, β = 0.5, Re = 5–25, Wi = λU/D = 1, 3, 5. Forward problem, but hybrid: about 10% of points carry rheoTool CFD data ("anchors").
- Tricks: outputs (u, v, p, L₁₁, L₂₁, L₂₂), with the conformation built as c = LLᵀ so it is positive-definite by construction. Inlet stress c = I (plug inflow). Adam (5k epochs) then L-BFGS (5k–10k iterations). 20k Latin-hypercube points. Transfer learning across Re.
- What failed: the pure physics-only PINN, tested only at Wi = 1, Re = 5, had test loss that diverged after about 15k epochs (overfitting to collocation points under L-BFGS). They also report spectral bias smoothing the wake and non-physical local minima. There is no ablation without the Cholesky factorisation, so its benefit is not isolated. Max errors against CFD are about 1–5 × 10⁻².
- Lesson for us: even at Wi = 1, data-free training was not robust in their hands. Their fixes (data anchoring, transfer learning) make the result a hybrid rather than an independent solver.

**Thakur, Raissi & Ardekani (2024), "ViscoelasticNet: A physics informed neural network framework for stress discovery and model selection", J. Non-Newtonian Fluid Mech. 330, 105265.** [doi:10.1016/j.jnnfm.2024.105265](https://doi.org/10.1016/j.jnnfm.2024.105265), [arXiv:2209.06972](https://arxiv.org/abs/2209.06972)
- Models: Oldroyd-B, Giesekus, linear PTT. Inverse problem: from velocity data (RheoTool simulations of a 2D stenosis and a cross-slot), it learns the stress field, model parameters and pressure, and selects the model.
- Weissenberg range: not reported. Relaxation times are 0.05–0.1 in their units.
- Tricks: separate networks for the streamfunction (velocity), stress (8 × 128) and pressure (8 × 64). Multistage training: train the stress network, freeze it, then train velocity and pressure on the momentum residual. Losses are normalised by the standard deviation of the reference data. They tried gradient-based weights and Lagrange multipliers and preferred a backward-Euler-based weighting. Adam with cosine annealing, about 2M + 0.8M iterations. No L-BFGS.
- What failed: peak wall stress in the stenosis was under-predicted in all cases. Sharp stress peaks at the cross-slot corners were smoothed by the single global network (the authors suggest domain decomposition). Pressure errors were larger because stress errors feed into the decoupled pressure solve (about 12% relative error in the cross-slot).

**Mahmoudabadbozchelou, Karniadakis & Jamali (2022), "nn-PINNs: Non-Newtonian physics-informed neural networks for complex fluid modeling", Soft Matter 18, 172–185.** [doi:10.1039/D1SM01298C](https://doi.org/10.1039/D1SM01298C)
Couples constitutive models to mass and momentum conservation in a PINN. Covers generalized-Newtonian models (power-law, Carreau–Yasuda, Herschel–Bulkley) and phenomenological models with memory and thixotropic timescales. Both forward and inverse, including recovery of velocity and stress with unknown boundary conditions from sparse velocity data. I could not open the full text to confirm geometries or elasticity levels.

**Mahmoudabadbozchelou & Jamali (2021), "Rheology-informed neural networks (RhINNs) for forward and inverse metamodelling of complex fluids", Sci. Rep. 11, 12015.** [doi:10.1038/s41598-021-91518-3](https://doi.org/10.1038/s41598-021-91518-3)
Solves the constitutive ODEs (a thixotropic elasto-visco-plastic model) for rheometric protocols, forward and inverse. It is not a spatial flow solver. Its relevance is that a constitutive-ODE PINN is easy and robust; the difficulty starts when the constitutive equation is coupled to a spatial flow.

**Lennon, McKinley & Swan (2023), "Scientific machine learning for modeling and simulating complex fluids", PNAS 120, e2304669120.** [doi:10.1073/pnas.2304669120](https://doi.org/10.1073/pnas.2304669120)
"Rheological universal differential equations": a UCM backbone plus a frame-invariant tensor neural network, trained on rheometric data and then used inside a multidimensional CFD solver. It is not a PINN flow solver, but it is a good reference for keeping learned stress closures frame-invariant and symmetric.

**Nieto Simavilla, Bonfanti, García-Beristain, Español & Ellero (2025), "Hammering at the entropy: a GENERIC-guided approach to learning polymeric rheological constitutive equations using PINNs", J. Fluid Mech. 1016, A11.** [doi:10.1017/jfm.2025.10325](https://doi.org/10.1017/jfm.2025.10325), [arXiv:2409.07545](https://arxiv.org/abs/2409.07545)
A PINN learns the entropic part of a GENERIC-consistent conformation-tensor equation, trained on analytical Oldroyd-B rheometric data or on cylinder-flow CFD. The learned model transfers to other geometries. This is constitutive learning (inverse), and the reference solutions in it stay within the analytically bounded range of Oldroyd-B.

**Li, Hua, Yu & Zuo (2026), "Physics-informed neural networks for steady power-law fluid flows around a square cylinder in a two-dimensional channel", Phys. Fluids 38, 043103.** [doi:10.1063/5.0322116](https://doi.org/10.1063/5.0322116)
A data-free forward PINN for a power-law fluid. Per the search-engine summary, Fourier feature embedding and re-initialisation strongly affect robustness, and re-initialisation produced a non-physical wake deflection at the highest Re. I read only the abstract-level summary.

**PINNs for the Reynolds (lubrication) equation.** **Almqvist (2021), "Fundamentals of physics-informed neural networks applied to solve the Reynolds boundary value problem", Lubricants 9(8), 82** ([doi:10.3390/lubricants9080082](https://doi.org/10.3390/lubricants9080082)) solves the 1D Reynolds equation and is a teaching-level example. There is follow-on work with cavitation (e.g. Rom 2023, Tribol. Int. 179, 108141). Relevance: if the project also solves the reduced lubrication equations, the Newtonian part is a 1D ODE that a PINN solves easily. The hard part is the 2D (x, η) Oldroyd-B lubrication system.

**Electro-osmotic and viscoelastic flow.** I found no PINN study of electro-osmotic flow of a viscoelastic fluid. The closest is Newtonian: **Merdasi, Ebrahimi, Yang & Kunz (2023), Chem. Eng. Process. 193, 109540** ([doi:10.1016/j.cep.2023.109540](https://doi.org/10.1016/j.cep.2023.109540)), on combined electro-osmotic and pressure-driven flow with mixing and heat transfer. Per the search snippet (not verified in full text), clustering collocation points in the electric double layer improved training markedly. That is the same lesson as for wall stress boundary layers.

**Other recent items, confirmed to exist but not read in detail:** Li, Xu & Ye (2026), "Numerical simulation of viscoelastic flows using physics-informed neural networks: R-PINN and adaptive IR-PINN", Int. J. Numer. Methods Heat Fluid Flow ([doi:10.1108/HFF-05-2026-0617](https://doi.org/10.1108/HFF-05-2026-0617)); and Lu, Jia, Aniruddha, An & Lee (2026), "Invariant guided PINN for fluid flow computation", arXiv:2604.08002 ([link](https://arxiv.org/abs/2604.08002)), which trains on subdomains first and then does a global correction for steady Oldroyd-B flow past a confined cylinder.

---

## C. PINN training practice relevant to a stiff coupled Stokes + Oldroyd-B system

Each entry gives the trick, then how it applies to our 2D steady Stokes + Oldroyd-B problem in a contracting channel.

1. **Gradient-norm loss balancing ("learning-rate annealing").** Wang, Teng & Perdikaris (2021), SIAM J. Sci. Comput. 43(5), A3055–A3081, [doi:10.1137/20M1318043](https://doi.org/10.1137/20M1318043). They trace failure to stiffness that makes back-propagated gradients of different loss terms very unequal, and set the loss weights from gradient statistics. The paper reports 50–100× accuracy gains on their test problems.
   *Use:* yes. Our loss has momentum residuals (O(1) after scaling), three constitutive residuals whose natural size grows like De and De², and boundary terms. Global weights updated every ~1000 steps with a moving average are cheap.

2. **NTK-based diagnosis and weighting.** Wang, Yu & Perdikaris (2022), J. Comput. Phys. 449, 110768, [doi:10.1016/j.jcp.2021.110768](https://doi.org/10.1016/j.jcp.2021.110768). Different loss terms converge at very different rates (NTK eigenvalue disparity). They propose weights from NTK traces.
   *Use:* as a diagnostic. If the constitutive residual stalls while the momentum residual drops, this is the explanation. Gradient-norm weights are cheaper and usually enough.

3. **"Expert's guide".** Wang, Sankaran, Wang & Perdikaris (2023), arXiv:2308.08468, [link](https://arxiv.org/abs/2308.08468). A practical recipe: non-dimensionalise inputs and outputs to O(1); tanh MLPs 3–6 layers × 128–512; modified MLP; random weight factorisation (μ = 1, σ = 0.1); Fourier features with σ ∈ [1, 10]; gradient-norm weights (update every 1000 steps, α = 0.9); random resampling of collocation points every step (batch ~4096); Adam, lr 1e-3, exponential decay; curriculum in the hard parameter (for example Re = 100 → 400 → 1000 → 3200).
   *Use:* adopt as the default configuration. Replace the Re curriculum with a De curriculum. Fourier features are probably unnecessary in X for a smooth contraction, but may help in η near the wall boundary layer.

4. **PirateNets.** Wang, Li, Chen & Perdikaris (2024), J. Mach. Learn. Res. 25(402), 1–51, [link](https://jmlr.org/papers/v25/24-0313.html). Adaptive residual connections initialised so the network starts shallow, which fixes poor trainability of deep PINN MLPs.
   *Use:* low priority. Our fields are smooth except at the wall layer and any geometric kinks, so a 4–6 layer MLP should do. Worth trying if deeper networks are needed for higher De.

5. **Causal training.** Wang, Sankaran & Perdikaris (2024), Comput. Methods Appl. Mech. Eng. 421, 116813, [doi:10.1016/j.cma.2024.116813](https://doi.org/10.1016/j.cma.2024.116813). Residuals at later times are down-weighted until earlier times are resolved.
   *Use:* not directly, since the problem is steady. There is an analogue, though, because the constitutive equation is hyperbolic along streamlines and information travels downstream from the inlet. Down-weighting residuals downstream until upstream residuals are small is a natural spatial version of this (my suggestion, not tested in the literature I found).

6. **Fourier feature embeddings.** Wang, Wang & Perdikaris (2021), Comput. Methods Appl. Mech. Eng. 384, 113938, [doi:10.1016/j.cma.2021.113938](https://doi.org/10.1016/j.cma.2021.113938). Random Fourier features counter spectral bias, and multi-scale versions handle multi-scale solutions.
   *Use:* optional. Spectral bias matters for the thin wall stress layer (δ ≈ 0.04/De) and at slope discontinuities of the geometry. A cheaper option is to give the network a boundary-layer-stretched input such as De(1 − η).

7. **Self-adaptive pointwise weights (SA-PINN).** McClenny & Braga-Neto (2023), J. Comput. Phys. 474, 111722, [doi:10.1016/j.jcp.2022.111722](https://doi.org/10.1016/j.jcp.2022.111722). Each collocation point gets a trainable weight raised by gradient ascent, which acts as a soft attention mask on hard regions.
   *Use:* reasonable for the near-wall and throat regions, but it adds parameters. Try it only after gradient-norm global weights and adaptive sampling.

8. **Residual-based adaptive sampling.** Wu, Zhu, Tan, Kartha & Lu (2023), Comput. Methods Appl. Mech. Eng. 403, 115671, [doi:10.1016/j.cma.2022.115671](https://doi.org/10.1016/j.cma.2022.115671). A comparison of 10 sampling methods over more than 6000 runs. Their adaptive methods, RAD and RAR-D, give the best accuracy per point.
   *Use:* yes. Combine RAD with a fixed share of points clustered near the walls and near the contraction inlet and outlet.

9. **Hard constraints for boundary conditions.** Sukumar & Srivastava (2022), Comput. Methods Appl. Mech. Eng. 389, 114333, [doi:10.1016/j.cma.2021.114333](https://doi.org/10.1016/j.cma.2021.114333): impose Dirichlet BCs exactly with distance functions. Lu, Pestourie, Yao, Wang, Verdugo & Johnson (2021), SIAM J. Sci. Comput. 43(6), B1105–B1132, [doi:10.1137/21M1397908](https://doi.org/10.1137/21M1397908): hPINN imposes PDE and design constraints with penalty and augmented-Lagrangian methods.
   *Use:* in the mapped coordinate η = y/h(x) the walls are η = ±1, so a distance function is just (1 − η²). No-slip, symmetry and the fixed flow rate can all be built into a streamfunction exactly (Recommendation 2). An augmented Lagrangian is an alternative for the integral flow-rate constraint if it is not hard-coded.

10. **Mixed-variable formulation.** Rao, Sun & Liu (2020), Theor. Appl. Mech. Lett. 10(3), 207–212, [doi:10.1016/j.taml.2020.01.039](https://doi.org/10.1016/j.taml.2020.01.039). Outputs (ψ, p, σ): the streamfunction gives exact continuity, and the momentum equation is written as ∇·σ = 0 with a separate constitutive residual, which lowers the derivative order. Trainability improved, and results were insensitive to the BC weight, unlike the (ψ, p) scheme.
   *Use:* strongly recommended. The Oldroyd-B problem is already in this form, since the polymer stress is a separate field. Write momentum as −∇p + ∇·(2μ_s D) + ∇·τ_p = 0 with ψ, p and τ_p (or the conformation) as outputs.

11. **Failure modes and curriculum regularisation.** Krishnapriyan, Gholami, Zhe, Kirby & Mahoney (2021), NeurIPS 34, [arXiv:2109.01050](https://arxiv.org/abs/2109.01050). They argue PINN failures come from an ill-conditioned loss landscape, not lack of expressivity. Fixes: a curriculum (start with an easy PDE parameter and increase it, warm-starting each stage) and sequence-to-sequence learning in time. They report 1–2 orders of magnitude lower error.
   *Use:* yes. Curriculum in De (and in the polymer fraction β̃), starting from the Newtonian solution.

12. **Adam then L-BFGS.** Rathore, Lei, Frangella, Lu & Udell (2024), ICML 2024, [arXiv:2402.01868](https://arxiv.org/abs/2402.01868). The ill-conditioning comes from the differential operators in the residual. Adam followed by L-BFGS beats either alone, and their second-order NysNewton-CG improves further.
   *Use:* yes. Run Adam first, then L-BFGS in float64 for the last digits. That matters because the elastic part of Δp is a correction on top of the Newtonian value: a few percent at small De, and about 12% at η = 0.4, De_m = 0.1 (from the first four terms of the Housiadas & Beris 2024a series). Its initial slope in De is a key comparison quantity. Watch for the overfitting that Suresh & Gupta saw with L-BFGS on fixed collocation points, and check on a held-out point set.

13. **Scaling and non-dimensionalisation.** Rohrhofer, Posch, Gößnitzer & Geiger (2023), IEEE Access 11, 86252–86261, [arXiv:2105.00862](https://arxiv.org/abs/2105.00862). Characteristic scales and equation coefficients rescale individual loss terms and shift the reachable trade-off between them; weights or reparametrisation can compensate. The expert's guide (item 3) also lists non-dimensionalisation as the first step.
   *Use:* essential here, because at ε = h₀/ℓ ≈ 0.02–0.1 the x and y scales differ by 10–50×, and stress components differ by factors of De and H⁴ (Section D).

14. **Domain decomposition.** Jagtap & Karniadakis (2020), "Extended physics-informed neural networks (XPINNs)", Commun. Comput. Phys. 28(5), 2002–2041, [doi:10.4208/cicp.OA-2020-0164](https://doi.org/10.4208/cicp.OA-2020-0164). Separate networks per subdomain, with interface residuals.
   *Use:* for the hyperbolic geometry with straight inlet and outlet sections, put subdomain interfaces at the slope discontinuities, where Housiadas (2025) shows the Cartesian v, τ_yy and τ_xy and the pressure gradient jump. ViscoelasticNet's authors also suggested domain decomposition for sharp stress peaks.

---

## D. Known numerical difficulties of Oldroyd-B that a PINN will meet

- **High-Weissenberg-number problem (HWNP).** Grid-based solvers lose convergence above a critical Wi that depends on geometry and mesh (Keunings 1986, J. Non-Newtonian Fluid Mech. 20, 209–226, [doi:10.1016/0377-0257(86)80022-2](https://doi.org/10.1016/0377-0257(86)80022-2); review in Alves, Oliveira & Pinho 2021, Annu. Rev. Fluid Mech. 53, 509–541, [doi:10.1146/annurev-fluid-010719-060107](https://doi.org/10.1146/annurev-fluid-010719-060107)). In this problem specifically, the finite-element runs of Boyko, Hinch & Stone (2024) did not reach De = 2, and the pseudospectral lubrication runs of Housiadas (2025) failed beyond De ≈ 0.5 (η = 0.4–0.5, Λ ≤ 8). A PINN will not escape the underlying physics: stresses grow quickly and develop steep layers.
- **Why the HWNP happens, and log-conformation.** Fattal & Kupferman (2004), J. Non-Newtonian Fluid Mech. 123, 281–285, [doi:10.1016/j.jnnfm.2004.08.008](https://doi.org/10.1016/j.jnnfm.2004.08.008), argue that polynomial approximations fail to represent the near-exponential profiles of the conformation tensor in regions dominated by stretching. Solving for Ψ = log A keeps A positive-definite and makes those profiles linear. The time-dependent implementation is in Fattal & Kupferman (2005), J. Non-Newtonian Fluid Mech. 126, 23–37, [doi:10.1016/j.jnnfm.2004.12.003](https://doi.org/10.1016/j.jnnfm.2004.12.003). For a PINN, outputting Ψ (three components in 2D) and forming A = exp(Ψ) in closed form for a symmetric 2×2 matrix is cheap. The evolution equation for Ψ is more complicated, though, so the residual can instead be written in terms of A = exp(Ψ). A Cholesky output A = LLᵀ (Suresh & Gupta 2026) is a simpler way to get positive-definiteness.
- **Unbounded extension in Oldroyd-B.** Steady extensional stress diverges when λε̇ ≥ 1/2 (Rallison & Hinch 1988, "Do we understand the physics in the constitutive equation?", J. Non-Newtonian Fluid Mech. 29, 37–55, [doi:10.1016/0377-0257(88)85049-3](https://doi.org/10.1016/0377-0257(88)85049-3)). In the hyperbolic channel the centreline extension rate of the Newtonian flow is constant, so De_m = λε̇. Above De_m ≈ 1/2 the centreline A₁₁ grows like a power of the stretch ratio instead of saturating. It stays finite only because the contraction has finite length. My estimate of its upper bound is the affine stretch, a Λ² increase over the inlet value; this is consistent with the high-De core solution Ã₁₁ = 18De²η²/H² of Boyko, Hinch & Stone (2024). The Trouton-ratio series of Housiadas & Beris (2024a) diverges at exactly this point. Expect the stress output range to grow sharply above De_m ≈ 0.5.
- **Stress boundary layers at the walls.** At high De the stresses are largest in a wall layer of thickness O(1/De) (about 0.04/De per Hinch, Boyko & Stone 2024; the variable is ζ = De(1 − η) in Boyko, Hinch & Stone 2024). A₁₁ near the wall scales like De²H⁴ in the contraction. With H = 4 that is a factor of 256 above inlet values (H⁴ = 256). ViscoelasticNet under-predicted peak wall stresses for this kind of reason. On the wall itself, u = 0, so the steady Oldroyd-B equation reduces to an algebraic relation between the stress and the local wall shear rate; for a straight wall, τ_xy = η_p γ̇_w, τ_xx = 2λη_p γ̇_w², τ_yy = 0. Housiadas & Beris build their Δp formulas on this "exact wall solution". It is a free pointwise check on PINN wall stresses.
- **Hyperbolic constitutive equation: inflow stress needed, nothing else.** For differential models, the stress must be prescribed on inflow boundaries and must not be prescribed elsewhere (Renardy 1988, Rocky Mountain J. Math. 18(2), 445–454, [doi:10.1216/rmj-1988-18-2-445](https://doi.org/10.1216/rmj-1988-18-2-445)). Use the fully developed Poiseuille stresses at the inlet (as in Hinch et al. 2024). Imposing stress at the outlet or walls over-determines the problem, and a soft-constraint PINN will then quietly trade PDE accuracy for BC accuracy.
- **Geometric slope discontinuities.** A hyperbolic section joined to straight channels has slope jumps. Housiadas (2025) shows these produce jumps in the Cartesian vertical velocity, τ_yy and τ_xy, and in the pressure gradient (Section A3). A single smooth tanh network will smear these jumps and spread error. A smooth sharp-corner-free shape (as in Hinch et al. 2024), curvilinear components, or subdomains avoid the problem. Sharp re-entrant corners (as in 4:1 contractions) also give stress singularities (Hinch 1993, J. Non-Newtonian Fluid Mech. 50, 161–171, [doi:10.1016/0377-0257(93)80029-B](https://doi.org/10.1016/0377-0257(93)80029-B)). Our geometry has no re-entrant corners, but this is the reason contraction benchmarks with sharp corners are much harder than ours.
- **Long exit relaxation.** Elastic stresses and the pressure gradient relax over a downstream distance proportional to De (Boyko, Hinch & Stone 2024, about 5.3 × 3De/(2H_ℓ) to reach 1%). Truncating the domain too early changes Δp.
- **Oldroyd-B against experiments.** Oldroyd-B predicts a reduction of Δp in a contraction, while experiments with Boger fluids often show an increase (noted by Boyko, Hinch & Stone 2024). The PINN reproduces the model, not the experiment.

---

## What this means for our PINN design

1. **Solve in lubrication-scaled, wall-fitted coordinates.** Use X = x/ℓ and η = y/h(X) ∈ [0, 1] (half-channel with a symmetry line), as Hinch, Boyko & Stone (2024) and Housiadas (2025) do. Scale velocity with q/(2h₀) and pressure with the Newtonian lubrication drop. Inputs and outputs then are O(1), and the walls become fixed lines. This follows the non-dimensionalisation advice of Wang et al. (2023) and Rohrhofer et al. (2023). With ε = 0.02–0.1, unscaled physical coordinates would give residuals that differ by factors of ε⁻¹–ε⁻².

2. **Hard-code kinematics through a streamfunction.** Take ψ(X, η) = (q/2)(3η − η³)/2 + η(1 − η²)²·N_ψ(X, η²). This satisfies continuity, symmetry (ψ odd in η), no-slip and no-penetration at η = 1 (ψ = q/2 and ∂ψ/∂η = 0 there), and the fixed flow rate exactly, for any network output. This is the mixed-variable idea of Rao et al. (2020) with distance-function hard BCs (Sukumar & Srivastava 2022). The Newtonian lubrication solution is the starting point, so the network only learns the correction.

3. **Output the polymer field in a scaled, positive-definite form.** Predict conformation components scaled by their local Poiseuille sizes (A₁₂ ~ De/H², A₁₁ ~ De²/H⁴, per Hinch et al. 2024), or Housiadas' modified components, which stay O(1). Build positive-definiteness in through a Cholesky factor (Suresh & Gupta 2026) or a log-conformation output (Fattal & Kupferman 2004; reportedly Matos et al. 2026). Enforce parity in η: A₁₁ and A₂₂ even, A₁₂ odd. Use separate subnetworks for ψ, p and polymer stress, as ViscoelasticNet does (Thakur et al. 2024).

4. **Impose only an inflow stress condition, and impose it exactly.** Include a short straight inlet section. Set the stresses there to the fully developed Oldroyd-B profile (A₂₂ = 1, A₁₂ = −3Deη, A₁₁ = 18De²η² at h = 1; Hinch et al. 2024), using a blending function so the inlet value is exact. Put no stress condition on walls or outlet (Renardy 1988).

5. **Make the exit channel long enough, or define Δp exactly as the benchmark does.** The elastic pressure gradient needs about 5.3 × 3De/(2H_ℓ) of exit length to relax (Boyko, Hinch & Stone 2024). Either include that length, or match Boyko & Stone's (2022) definition of Δp and domain exactly, before comparing curves.

6. **Treat geometry kinks deliberately.** If the hyperbolic section joins straight channels with a slope jump, use separate networks for inlet, contraction and outlet, joined by continuity of ψ, its normal derivative and the conformation tensor (XPINN-style; Jagtap & Karniadakis 2020). Allow the pressure gradient to jump at the joint, since Housiadas (2025) shows it must. Otherwise, use Hinch's curvilinear velocity components, in which the jump disappears.

7. **Balance losses and place points where the physics is hard.** Use gradient-norm global weights (Wang et al. 2021, 2023), then RAD resampling (Wu et al. 2023), with a fixed fraction of points clustered within a few multiples of 0.04/De of the wall and around the contraction inlet and outlet (Hinch et al. 2024). Consider SA-PINN pointwise weights (McClenny & Braga-Neto 2023) only if these are not enough.

8. **Use a De curriculum, and start with the decoupled ultra-dilute problem.** Stage 0: De = 0 (the Newtonian solution must reproduce ΔP₀ = 3∫dZ/H³). Stage 1: freeze the velocity at the Newtonian field and train only the stress network. The exact answer is the closed-form conformation of Boyko, Hinch & Stone (2024), which isolates constitutive-equation errors. Stage 2: full coupling at small β̃, then larger β̃ and De, warm-starting each stage (Krishnapriyan et al. 2021; Wang et al. 2023).

9. **Optimise with Adam, then L-BFGS in float64, and check on held-out points.** Follow Rathore et al. (2024) for the optimiser. The elastic part of Δp is a small correction to the Newtonian value, so single-precision Adam alone will probably not resolve it. Monitor residuals on a fresh point set, because Suresh & Gupta (2026) saw test loss diverge under L-BFGS.

10. **Compute Δp several ways and require them to agree.** Compute Δp from (a) the wall pressure difference, (b) the integrated momentum or force balance, (c) the mechanical-energy route (Sialmas & Housiadas 2025; Housiadas & Beris 2024d), and (d) the reciprocal-theorem integral (Boyko & Stone 2021). Also check wall stresses against the algebraic wall solution (Section D). If these disagree, the PINN has not converged, even when the loss looks small.

11. **Validate against benchmarks in order of difficulty.** (i) Newtonian lubrication, plus O(ε²) corrections if comparing with 2D at finite ε (Tavakol et al. 2017). (ii) Boyko & Stone (2022) low-De q–Δp. (iii) The O(De⁸) series of Housiadas & Beris (2023, 2024a) for the same hyperbolic shape, resummed with Padé [4/4], with De converted to De_m. (iv) The ultra-dilute all-De result of Boyko, Hinch & Stone (2024), evaluated by quadrature for the hyperbolic shape. (v) The high-De linear trend of Hinch et al. (2024): exactly, if the PINN is also run on their x²(2 − x)² shape; otherwise only qualitatively. (vi) Boyko & Stone's COMSOL data.

12. **Set expectations on the reachable De.** Conventional solvers for this exact problem stop at about De ≤ 0.5 (pseudospectral lubrication) or below De = 2 (finite elements). Data-free viscoelastic PINNs have struggled at Wi ≈ 1 (Suresh & Gupta 2026). Treat De_m up to about 0.5 as the core target, where the centreline stretch is still bounded (Rallison & Hinch 1988; Housiadas & Beris 2024a). Treat higher De as exploratory. If sparse simulation data are added, call the result a hybrid, not an independent solution.

---

## References

### A. Lubrication theory and benchmarks
1. Boyko, E. & Stone, H.A. 2021 Reciprocal theorem for calculating the flow rate–pressure drop relation for complex fluids in narrow geometries. *Phys. Rev. Fluids* 6, L081301. https://doi.org/10.1103/PhysRevFluids.6.L081301
2. Boyko, E. & Stone, H.A. 2022 Pressure-driven flow of the viscoelastic Oldroyd-B fluid in narrow non-uniform geometries: analytical results and comparison with simulations. *J. Fluid Mech.* 936, A23. https://doi.org/10.1017/jfm.2022.67 (arXiv:2109.06928)
3. Boyko, E., Stone, H.A. & Christov, I.C. 2022 Flow rate–pressure drop relation for deformable channels via fluidic and elastic reciprocal theorems. *Phys. Rev. Fluids* 7, L092201. https://doi.org/10.1103/PhysRevFluids.7.L092201 (arXiv:2203.14112)
4. Boyko, E. & Christov, I.C. 2023 Non-Newtonian fluid–structure interaction: flow of a viscoelastic Oldroyd-B fluid in a deformable channel. *J. Non-Newtonian Fluid Mech.* 313, 104990. https://doi.org/10.1016/j.jnnfm.2023.104990 (arXiv:2210.13268)
5. Chun, S., Boyko, E., Christov, I.C. & Feng, J. 2024 Flow rate–pressure drop relations for shear-thinning fluids in deformable configurations: theory and experiments. *Phys. Rev. Fluids* 9, 043302. https://doi.org/10.1103/PhysRevFluids.9.043302 (arXiv:2401.05513)
6. Christov, I.C. 2022 Soft hydraulics: from Newtonian to complex fluid flows through compliant conduits. *J. Phys.: Condens. Matter* 34, 063001. https://doi.org/10.1088/1361-648X/ac327d
7. Hinch, E.J., Boyko, E. & Stone, H.A. 2024 Fast flow of an Oldroyd-B model fluid through a narrow slowly varying contraction. *J. Fluid Mech.* 988, A11. https://doi.org/10.1017/jfm.2024.260
8. Boyko, E., Hinch, E.J. & Stone, H.A. 2024 Flow of an Oldroyd-B fluid in a slowly varying contraction: theoretical results for arbitrary values of Deborah number in the ultra-dilute limit. *J. Fluid Mech.* 988, A10. https://doi.org/10.1017/jfm.2024.223
9. Mahapatra, B., Ruangkriengsin, T., Stone, H.A. & Boyko, E. 2025 Viscoelastic fluid flow in a slowly varying planar contraction: the role of finite extensibility on the pressure drop. *J. Fluid Mech.* 1009, A12. https://doi.org/10.1017/jfm.2025.142 (arXiv:2409.08150)
10. Kedem, Y., Mahapatra, B. & Boyko, E. 2026 Viscoelastic flow of an Oldroyd-B fluid through a slowly varying contraction–expansion channel: pressure drop and elastic stress relaxation. *Phys. Rev. Fluids* 11, 043303. https://doi.org/10.1103/kyql-qp7b (arXiv:2510.07166)
11. Housiadas, K.D. & Beris, A.N. 2023 Lubrication approximation of pressure-driven viscoelastic flow in a hyperbolic channel. *Phys. Fluids* 35, 123116. https://doi.org/10.1063/5.0183154
12. Housiadas, K.D. & Beris, A.N. 2024a Pressure-drop and Trouton ratio for Oldroyd-B fluids in hyperbolic converging channels. *Phys. Fluids* 36, 021702. https://doi.org/10.1063/5.0194278
13. Housiadas, K.D. & Beris, A.N. 2024d Pressure-driven viscoelastic flow in axisymmetric geometries with application to the hyperbolic pipe. *J. Fluid Mech.* 999, A7. https://doi.org/10.1017/jfm.2024.848 (arXiv:2312.15031)
14. Housiadas, K.D. 2025 An exact solution of the lubrication equations for the Oldroyd-B model in a hyperbolic channel. arXiv:2503.20636 (preprint). https://arxiv.org/abs/2503.20636
15. Sialmas, P. & Housiadas, K.D. 2025 An exact solution of the lubrication equations for the Oldroyd-B model in a hyperbolic pipe. *J. Non-Newtonian Fluid Mech.* 335, 105331. https://doi.org/10.1016/j.jnnfm.2024.105331
16. Hinch, E.J. 2026 Approach to a similarity solution of the lubrication flow of an Oldroyd-B fluid through a hyperbolic pipe. *J. Non-Newtonian Fluid Mech.* 347, 105519. https://doi.org/10.1016/j.jnnfm.2025.105519 (not read; existence and metadata confirmed via Crossref)
17. Sialmas, P. & Housiadas, K.D. 2025 General evaluation of the pressure gradient for lubrication flows in varying channels with applications to linear and hyperbolic contractions. *J. Non-Newtonian Fluid Mech.* 346, 105487. https://doi.org/10.1016/j.jnnfm.2025.105487 (arXiv:2504.09563)
18. Ahmed, H. & Biancofiore, L. 2021 A new approach for modeling viscoelastic thin film lubrication. *J. Non-Newtonian Fluid Mech.* 292, 104524. https://doi.org/10.1016/j.jnnfm.2021.104524
19. Ahmed, H. & Biancofiore, L. 2023 Modeling polymeric lubricants with non-linear stress constitutive relations. *J. Non-Newtonian Fluid Mech.* 321, 105123. https://doi.org/10.1016/j.jnnfm.2023.105123
20. Tavakol, B., Froehlicher, G., Holmes, D.P. & Stone, H.A. 2017 Extended lubrication theory: improved estimates of flow in channels with variable geometry. *Proc. R. Soc. A* 473, 20170234. https://doi.org/10.1098/rspa.2017.0234

### B. Physics-informed learning for non-Newtonian and viscoelastic flow
21. Matos, J.X., Fernandes, C. & Alves, M.A. 2026 Simulation of steady viscoelastic creeping flows with physics-informed neural networks. *J. Non-Newtonian Fluid Mech.* 350, 105637. https://doi.org/10.1016/j.jnnfm.2026.105637 (full text not read)
22. Suresh, M. & Gupta, A. 2026 Physics-informed neural networks for viscoelastic fluid flows around a cylinder in a two-dimensional channel. arXiv:2609.07861 (preprint). https://arxiv.org/abs/2609.07861
23. Thakur, S., Raissi, M. & Ardekani, A.M. 2024 ViscoelasticNet: a physics informed neural network framework for stress discovery and model selection. *J. Non-Newtonian Fluid Mech.* 330, 105265. https://doi.org/10.1016/j.jnnfm.2024.105265 (arXiv:2209.06972)
24. Mahmoudabadbozchelou, M., Karniadakis, G.E. & Jamali, S. 2022 nn-PINNs: non-Newtonian physics-informed neural networks for complex fluid modeling. *Soft Matter* 18, 172–185. https://doi.org/10.1039/D1SM01298C
25. Mahmoudabadbozchelou, M. & Jamali, S. 2021 Rheology-informed neural networks (RhINNs) for forward and inverse metamodelling of complex fluids. *Sci. Rep.* 11, 12015. https://doi.org/10.1038/s41598-021-91518-3
26. Lennon, K.R., McKinley, G.H. & Swan, J.W. 2023 Scientific machine learning for modeling and simulating complex fluids. *Proc. Natl Acad. Sci. USA* 120, e2304669120. https://doi.org/10.1073/pnas.2304669120
27. Nieto Simavilla, D., Bonfanti, A., García-Beristain, I., Español, P. & Ellero, M. 2025 Hammering at the entropy: a GENERIC-guided approach to learning polymeric rheological constitutive equations using PINNs. *J. Fluid Mech.* 1016, A11. https://doi.org/10.1017/jfm.2025.10325 (arXiv:2409.07545)
28. Li, H., Hua, H., Yu, Z. & Zuo, Z. 2026 Physics-informed neural networks for steady power-law fluid flows around a square cylinder in a two-dimensional channel. *Phys. Fluids* 38, 043103. https://doi.org/10.1063/5.0322116 (abstract-level only)
29. Almqvist, A. 2021 Fundamentals of physics-informed neural networks applied to solve the Reynolds boundary value problem. *Lubricants* 9(8), 82. https://doi.org/10.3390/lubricants9080082
30. Rom, M. 2023 Physics-informed neural networks for the Reynolds equation with cavitation modeling. *Tribol. Int.* 179, 108141. https://doi.org/10.1016/j.triboint.2022.108141
31. Merdasi, A., Ebrahimi, S., Yang, X. & Kunz, R. 2023 Physics informed neural network application on mixing and heat transfer in combined electroosmotic–pressure driven flow. *Chem. Eng. Process.* 193, 109540. https://doi.org/10.1016/j.cep.2023.109540 (abstract-level only)
32. Li, X., Xu, W. & Ye, Z. 2026 Numerical simulation of viscoelastic flows using physics-informed neural networks: R-PINN and adaptive IR-PINN. *Int. J. Numer. Methods Heat Fluid Flow* (online 8 Oct 2026). https://doi.org/10.1108/HFF-05-2026-0617 (metadata only)
33. Lu, Z., Jia, J., Aniruddha, B., An, X. & Lee, Y.J. 2026 Invariant guided PINN for fluid flow computation. arXiv:2604.08002 (preprint). https://arxiv.org/abs/2604.08002 (abstract only)

### C. PINN methodology
34. Raissi, M., Perdikaris, P. & Karniadakis, G.E. 2019 Physics-informed neural networks: a deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *J. Comput. Phys.* 378, 686–707. https://doi.org/10.1016/j.jcp.2018.10.045
35. Wang, S., Teng, Y. & Perdikaris, P. 2021 Understanding and mitigating gradient flow pathologies in physics-informed neural networks. *SIAM J. Sci. Comput.* 43(5), A3055–A3081. https://doi.org/10.1137/20M1318043 (arXiv:2001.04536)
36. Wang, S., Yu, X. & Perdikaris, P. 2022 When and why PINNs fail to train: a neural tangent kernel perspective. *J. Comput. Phys.* 449, 110768. https://doi.org/10.1016/j.jcp.2021.110768 (arXiv:2007.14527)
37. Wang, S., Sankaran, S., Wang, H. & Perdikaris, P. 2023 An expert's guide to training physics-informed neural networks. arXiv:2308.08468. https://arxiv.org/abs/2308.08468
38. Wang, S., Li, B., Chen, Y. & Perdikaris, P. 2024 PirateNets: physics-informed deep learning with residual adaptive networks. *J. Mach. Learn. Res.* 25(402), 1–51. https://jmlr.org/papers/v25/24-0313.html (arXiv:2402.00326)
39. Wang, S., Sankaran, S. & Perdikaris, P. 2024 Respecting causality for training physics-informed neural networks. *Comput. Methods Appl. Mech. Eng.* 421, 116813. https://doi.org/10.1016/j.cma.2024.116813
40. Wang, S., Wang, H. & Perdikaris, P. 2021 On the eigenvector bias of Fourier feature networks: from regression to solving multi-scale PDEs with physics-informed neural networks. *Comput. Methods Appl. Mech. Eng.* 384, 113938. https://doi.org/10.1016/j.cma.2021.113938
41. McClenny, L.D. & Braga-Neto, U.M. 2023 Self-adaptive physics-informed neural networks. *J. Comput. Phys.* 474, 111722. https://doi.org/10.1016/j.jcp.2022.111722 (arXiv:2009.04544)
42. Wu, C., Zhu, M., Tan, Q., Kartha, Y. & Lu, L. 2023 A comprehensive study of non-adaptive and residual-based adaptive sampling for physics-informed neural networks. *Comput. Methods Appl. Mech. Eng.* 403, 115671. https://doi.org/10.1016/j.cma.2022.115671 (arXiv:2207.10289)
43. Lu, L., Pestourie, R., Yao, W., Wang, Z., Verdugo, F. & Johnson, S.G. 2021 Physics-informed neural networks with hard constraints for inverse design. *SIAM J. Sci. Comput.* 43(6), B1105–B1132. https://doi.org/10.1137/21M1397908 (arXiv:2102.04626)
44. Sukumar, N. & Srivastava, A. 2022 Exact imposition of boundary conditions with distance functions in physics-informed deep neural networks. *Comput. Methods Appl. Mech. Eng.* 389, 114333. https://doi.org/10.1016/j.cma.2021.114333
45. Krishnapriyan, A.S., Gholami, A., Zhe, S., Kirby, R.M. & Mahoney, M.W. 2021 Characterizing possible failure modes in physics-informed neural networks. *NeurIPS* 34. https://arxiv.org/abs/2109.01050
46. Rathore, P., Lei, W., Frangella, Z., Lu, L. & Udell, M. 2024 Challenges in training PINNs: a loss landscape perspective. *ICML 2024*. https://arxiv.org/abs/2402.01868
47. Rao, C., Sun, H. & Liu, Y. 2020 Physics-informed deep learning for incompressible laminar flows. *Theor. Appl. Mech. Lett.* 10(3), 207–212. https://doi.org/10.1016/j.taml.2020.01.039 (arXiv:2002.10558)
48. Rohrhofer, F.M., Posch, S., Gößnitzer, C. & Geiger, B.C. 2023 Data vs. physics: the apparent Pareto front of physics-informed neural networks. *IEEE Access* 11, 86252–86261. https://arxiv.org/abs/2105.00862
49. Jagtap, A.D. & Karniadakis, G.E. 2020 Extended physics-informed neural networks (XPINNs): a generalized space-time domain decomposition based deep learning framework for nonlinear partial differential equations. *Commun. Comput. Phys.* 28(5), 2002–2041. https://doi.org/10.4208/cicp.OA-2020-0164

### D. Viscoelastic numerics
50. Keunings, R. 1986 On the high Weissenberg number problem. *J. Non-Newtonian Fluid Mech.* 20, 209–226. https://doi.org/10.1016/0377-0257(86)80022-2
51. Rallison, J.M. & Hinch, E.J. 1988 Do we understand the physics in the constitutive equation? *J. Non-Newtonian Fluid Mech.* 29, 37–55. https://doi.org/10.1016/0377-0257(88)85049-3
52. Renardy, M. 1988 Inflow boundary conditions for steady flow of viscoelastic fluids with differential constitutive laws. *Rocky Mountain J. Math.* 18(2), 445–454. https://doi.org/10.1216/rmj-1988-18-2-445 (correction: 19(2), 561–562, 1989)
53. Hinch, E.J. 1993 The flow of an Oldroyd fluid around a sharp corner. *J. Non-Newtonian Fluid Mech.* 50, 161–171. https://doi.org/10.1016/0377-0257(93)80029-B
54. Fattal, R. & Kupferman, R. 2004 Constitutive laws for the matrix-logarithm of the conformation tensor. *J. Non-Newtonian Fluid Mech.* 123, 281–285. https://doi.org/10.1016/j.jnnfm.2004.08.008
55. Fattal, R. & Kupferman, R. 2005 Time-dependent simulation of viscoelastic flows at high Weissenberg number using the log-conformation representation. *J. Non-Newtonian Fluid Mech.* 126, 23–37. https://doi.org/10.1016/j.jnnfm.2004.12.003
56. Alves, M.A., Oliveira, P.J. & Pinho, F.T. 2021 Numerical methods for viscoelastic fluid flows. *Annu. Rev. Fluid Mech.* 53, 509–541. https://doi.org/10.1146/annurev-fluid-010719-060107

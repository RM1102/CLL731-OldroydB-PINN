# Paper notes: Boyko & Stone (2022), in plain words

**Paper.** E. Boyko and H. A. Stone, "Pressure-driven flow of the viscoelastic Oldroyd-B fluid in narrow non-uniform geometries: analytical results and comparison with simulations", *J. Fluid Mech.* 936, A23 (2022). [doi:10.1017/jfm.2022.67](https://doi.org/10.1017/jfm.2022.67), [arXiv:2109.06928](https://arxiv.org/abs/2109.06928). PDF: `../papers/`.

**In brief.** The paper asks how much pressure it takes to push a stretchy polymer liquid through a channel that narrows. It answers with formulas for low elasticity and checks them against computer simulations. Our project solves the same problem with a neural network that is trained only on the equations (a PINN).

---

## 1. The question

Think of squeezing honey that has long rubber-band-like molecules mixed in (a "Boger fluid") through a funnel that narrows from 4 mm to 1 mm. For plain honey, the pressure needed is fixed by its viscosity alone. With the polymer molecules, the liquid also remembers how it was stretched, a bit like dough springing back. The paper asks:

> For a fixed flow rate *q*, how does the pressure drop Δ*p* across a narrowing channel change because of this elasticity?

**Answer, in one line:** elasticity *lowers* the pressure drop, and the drop keeps falling as elasticity (the Deborah number) grows.

## 2. The set-up

- A 2-D channel of half-height *h(z)* and length ℓ, with *h* ≪ ℓ (narrow and slowly varying). Only the top half is solved; the centreline is a symmetry line.
- Shape used for all results, eq. (5.1), in scaled units: **H(Z) = α / ((α − 1)Z + 1)**, a hyperbolic contraction from H = α = 4 at the inlet (Z = 0) to H = 1 at the outlet (Z = 1).
- Slow flow: inertia is negligible (Reynolds number about 10⁻⁷).
- Fluid model: **Oldroyd-B**. The stress is a Newtonian solvent part plus a polymer part. The polymer part is described by a "conformation tensor" **A**, which measures the average stretch and orientation of the molecules. **A** relaxes back to its rest state over a time λ.

## 3. The equations (scaled form, eq. 2.14)

The authors scale every quantity by its natural size in a narrow channel (eq. 2.10). In these units every term is of order one, which is also what a neural network needs. With Z along the channel and Y across it:

```
continuity:     dUz/dZ + dUy/dY = 0
z-momentum:     dP/dZ = (1-b) [eps^2 d2Uz/dZ2 + d2Uz/dY2] + dTzz/dZ + dTyz/dY
y-momentum:     dP/dY = eps^2 { (1-b) [eps^2 d2Uy/dZ2 + d2Uy/dY2] + dTyz/dZ + dTyy/dY }
polymer stress: T = b * A   (b = beta tilde = eta_p / eta_0)
Azz: De (u.grad Azz - 2 dUz/dZ Azz - 2 dUz/dY Ayz) - 2 eps^2 dUz/dZ = -Azz
Ayy: De (u.grad Ayy - 2 dUy/dZ Ayz - 2 dUy/dY Ayy) - 2 dUy/dY       = -Ayy
Ayz: De (u.grad Ayz -   dUy/dZ Azz -   dUz/dY Ayy) - eps^2 dUy/dZ - dUz/dY = -Ayz
```

Here **A** is the paper's *scaled* conformation, "A tilde" in eq. (2.10). We checked by hand that this scaled system is exact, not an approximation: setting ε = 0 gives the lubrication limit, and keeping ε gives the full 2-D Stokes–Oldroyd-B problem.

Boundary conditions, eq. (2.15): no slip and no penetration on the wall Y = H(Z); symmetry on Y = 0; flux ∫₀ᴴ Uz dY = 1.

## 4. The three numbers that control everything

| Symbol | Meaning | Everyday reading | Paper values |
|---|---|---|---|
| ε = h_ℓ / ℓ | how narrow the channel is | a drinking straw vs a pipe | 0.02 (case I), 0.1 (case II) |
| De = λ u_c / ℓ | molecular memory time ÷ time to cross the contraction | does the dough spring back before it leaves the funnel? | 0 to 0.3 |
| β̃ = η_p / η₀ | share of viscosity from the polymer | how much rubber is in the honey | 0.4 |

Two more: α = h₀/h_ℓ = 4 (contraction ratio) and Wi = De/ε (shear-based Weissenberg number, up to 15).

## 5. What the paper derives

1. **Leading order (Newtonian-like), eqs (3.4)–(3.9).** Parabolic velocity Uz₀ = (3/2)(H² − Y²)/H³, and ΔP₀ = 3∫₀¹ dZ/H³.
2. **First order in De, eqs (3.10)–(3.16).** The velocity does *not* change at O(De). Only the stresses and the pressure do: **ΔP₁ = (9/2) β̃ (1/H(0)⁴ − 1/H(1)⁴)**, which depends only on the inlet and outlet heights.
3. **Second order, eqs (3.17)–(3.22).** The velocity first changes here, and ΔP₂ depends on the whole shape.
4. **Third order, via a reciprocal theorem, eqs (4.5)–(4.14).** ΔP₃ comes from lower-order fields only, without solving the O(De³) flow.
5. **Hyperbolic channel, closed form, eq. (5.2).** For α = 4 and β̃ = 0.4:

| Term | Formula | Value |
|---|---|---|
| ΔP₀ | (3/4)(1+α)(1+α²)/α³ | +0.996094 |
| ΔP₁ | (9/2) β̃ (1 − α⁴)/α⁴ | −1.792969 |
| ΔP₂ | (648/35) β̃ (1+α)(1+α²)(1−α)²/α⁵ | +5.532589 |
| ΔP₃ | (216/35) β̃ (11−β̃)(1+α)(1+α²)(1−α)³/α⁶ | −14.661362 |

The series ΔP = ΔP₀ + De ΔP₁ + De² ΔP₂ + De³ ΔP₃ alternates in sign, and its coefficients grow about threefold at each order. It is therefore only trustworthy for De ≲ 0.1.

6. **Simulations (appendix A).** COMSOL finite elements with straight inlet and outlet sections of length ℓ, fully developed inflow (velocity *and* polymer stress), and Δp measured on the centreline from z = 0 to z = ℓ. At ε = 0.02, the 3rd-order formula is within about 5 % of the simulations up to De = 0.2. At ε = 0.1 it under-predicts by up to about 9 %.
7. **Physics insight, eq. (5.3), fig. 3.** The pressure-drop reduction comes from the gradient of the polymer **shear** stress (term ④), not from the axial normal stress.

## 6. What we checked, and what we found

All checks are reproducible scripts in `03_code/scripts/`; logs are in `04_results/`.

| Check | Result | Where |
|---|---|---|
| Re-derived the whole low-De solution symbolically (SymPy) for a general shape H(Z) | Every printed formula confirmed, **except one sign** | `04_results/analytical/sympy_check.txt` |
| Sign in eq. (3.16b), arXiv v1 | Must be **A_yz,1 = −18 H′ Y(H² − 2Y²)/H⁷** (printed with +). The corrected sign is the one that reproduces the colour bars of fig. 5. | same |
| ΔP₃ computed directly from the O(De³) momentum balance, without the reciprocal theorem | Equal to eq. (4.14) and eq. (5.2d) to 6 digits | same |
| Recovered the FE data points from the vector graphics of the PDF | 31 points per curve in fig. 2, calibration error about 3×10⁻⁶ | `04_results/digitized/` |
| Our own lubrication-limit (ε → 0) solver vs recovered FE data at ε = 0.02 | Agree within 0.35 % for De = 0.1–0.3 | `03_code/oldroydb/lubrication_fd.py` |
| Newtonian 2-D correction (extended lubrication, our SymPy derivation) | ΔP = 255/256 + 3.743 ε²; gives 0.99759 at ε = 0.02, which matches the recovered FE value 0.9975 | `04_results/analytical/newtonian_eps2.txt` |
| Effect of rounding the corners (our solver, ε → 0) | ΔP changes by at most 0.03 % between rounding widths 0.005 and 0.03 | `04_results/reference/corner_rounding.txt` |
| Printed asymptotic curves in fig. 8(b) | Lie about 2 % above eq. (5.2); fig. 8(a) matches it to 10⁻⁵. Probably a plotting slip; it does not affect the FE points, which we calibrated from the axes. | digitizer log |

**A finding that shapes the method.** Marching the ε → 0 equations downstream, station by station, blows up for any De > 0, and it blows up sooner on finer grids. The cause is the axial normal-stress gradient ∂Tzz/∂Z: it changes the velocity, and the velocity gradient feeds back into Tzz. Together they behave like diffusion along the channel, so the problem needs information from the outlet as well as the inlet. We therefore solve every station at once. A PINN, which treats the whole domain together, is naturally suited to this.

## 7. What we take from the paper as benchmarks

1. **Closed-form ΔP(De)** to O(De³), eq. (5.2), for small De.
2. **Closed-form fields** to O(De) (velocity, all three conformation components) and Uz to O(De²).
3. **FE pressure drops** (fig. 2: ΔP vs De for ε = 0.02 and 0.1; fig. 7: ΔP vs α; fig. 8: ΔP vs η_p/η_s), recovered from the PDF.
4. **Qualitative field maps** (figs 4–6): T_zz is positive and T_yz negative, both largest on the wall at the outlet, and the axial velocity hardly changes with De.

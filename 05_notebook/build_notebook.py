"""Build the main notebook from text + code cells.

Run:   python build_notebook.py          (writes CLL731_OldroydB_PINN.ipynb)
Then:  jupyter nbconvert --execute --to notebook --inplace CLL731_OldroydB_PINN.ipynb
       jupyter nbconvert --to html CLL731_OldroydB_PINN.ipynb
"""
from pathlib import Path

import nbformat as nbf

import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "03_code"))
from oldroydb import analytical as an, compare, report as rp  # noqa: E402


def headline():
    """Numbers quoted in the conclusions, computed from 04_results at build time."""
    ref = compare.load_reference(ROOT / "04_results" / "reference" / "fd_eps0.npz")
    fe = rp.load_fe(ROOT)
    e_dir, e_mom = [], []
    for d, (m, _) in rp.load_runs(ROOT, 0.0).items():
        dp = m.pressure_drop(d)
        r = float(np.interp(d, ref["De"], ref["dP"]))
        e_dir.append(abs(dp["direct"] / r - 1))
        e_mom.append(abs(dp["momentum"] / r - 1))
    r03 = float(np.interp(0.3, ref["De"], ref["dP"]))
    _, rows = rp.reference_vs_fe(ref, fe, 0.02)
    fe_err, fe_mom = [], []
    for eps in (0.02, 0.1):
        for d, (m, _) in rp.load_runs(ROOT, eps).items():
            x, y = fe[eps]
            f = float(y[np.argmin(np.abs(x - d))])
            dp = m.pressure_drop(d)
            fe_err.append(abs(dp["direct"] / f - 1))
            fe_mom.append(abs(dp["momentum"] / f - 1))
    return {"lub_dir": f"{100 * max(e_dir):.2f}", "lub_mom": f"{100 * max(e_mom):.2f}",
            "formula_03": f"{100 * abs(an.pressure_drop(0.3, 3) / r03 - 1):.0f}",
            "ref_fe": f"{max(abs(r[3]) for r in rows):.2f}",
            "fe_max": f"{100 * max(fe_err):.2f}" if fe_err else "n/a",
            "fe_mom": f"{100 * max(fe_mom):.2f}" if fe_mom else "n/a"}


H = headline()
fill = lambda t: [t := t.replace(f"<<{k}>>", v) for k, v in H.items()][-1]
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(fill(s).strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

# =============================================================== title + brief
md(r"""
# Squeezing a stretchy liquid through a narrowing channel
### Solving Boyko & Stone (2022) with a physics-informed neural network

CLL731 course project · Rahul Arvind Masand (2022CH71476) · Department of Chemical Engineering, IIT Delhi · November 2026

---

## The problem, in brief

A polymer solution is pushed at a fixed flow rate through a flat (2-D) channel whose half-height falls from 4 to 1 along a hyperbola, $H(Z) = 4/(3Z+1)$. $Z$ runs along the channel (the narrowing part is $0 \le Z \le 1$), and $Y$ runs across it (0 on the centreline, $H(Z)$ at the wall). All quantities are in the scaled units of the paper.

The flow obeys a mass balance, a force balance and an equation for how the polymer molecules stretch (the Oldroyd-B model):

$$\frac{\partial U_z}{\partial Z}+\frac{\partial U_y}{\partial Y}=0,\qquad
\frac{\partial P}{\partial Z}=(1-\tilde\beta)\,\frac{\partial^2U_z}{\partial Y^2}+\frac{\partial T_{zz}}{\partial Z}+\frac{\partial T_{yz}}{\partial Y}+O(\epsilon^2),$$

$$\mathrm{De}\,\Big[(\mathbf{u}\cdot\nabla)\mathbf{A}-(\nabla\mathbf{u})^{\mathsf T}\!\cdot\mathbf{A}-\mathbf{A}\cdot\nabla\mathbf{u}\Big]-(\text{flow-driven source})=-\mathbf{A},\qquad \mathbf{T}=\tilde\beta\,\mathbf{A}.$$

On the wall the liquid does not move ($U_z=U_y=0$), the flow is symmetric about the centreline, and the flow rate is fixed.

**Wanted:** the pressure drop $\Delta P = P(Z{=}0) - P(Z{=}1)$ on the centreline, for elasticity $\mathrm{De}$ from 0 to 0.3.

**What we do:** train a neural network on these equations alone, with no solution data. **How we check it:** against the paper's formulas, the paper's computer simulations (recovered from its PDF), and our own independent numerical solver.
""")

md(r"""
## Order of the notebook

0. Set-up: run this cell first
1. The physics in plain words
2. Three references to check the network against
3. How the PINN works
4. Results
5. What we learned, what is left
6. Appendix: how to retrain, file map, references
""")

# =============================================================== 0. set-up
md(r"""
## 0. Set-up: run this cell first

**In brief.** This cell finds the project folder and loads our code. The trained networks are read from disk, so the whole notebook runs in about two minutes on a laptop CPU, with no GPU. Set `TRAIN_DEMO = True` to also watch one small network train live (about 3 extra minutes).
""")

code(r"""
TRAIN_DEMO = False                                   # True: also train a small PINN live
import sys, json, warnings
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch

ROOT = Path.cwd().resolve()                          # walk up until the project folder
while not (ROOT / "03_code" / "oldroydb").exists():
    if ROOT == ROOT.parent:
        raise FileNotFoundError("Open this notebook from inside the project folder.")
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "03_code"))

from oldroydb import analytical as an, compare, pinn, report as rp, plotting as pl
RES = ROOT / "04_results"
pl.style()                                           # one plotting style everywhere
torch.set_num_threads(4)
warnings.filterwarnings("ignore", category=UserWarning)
print(f"project folder : {ROOT}")
print(f"numpy {np.__version__} | torch {torch.__version__} | live demo: {TRAIN_DEMO}")
""")

# =============================================================== 1. physics
md(r"""
## 1. The physics in plain words

**In brief.** A polymer liquid behaves partly like a thick liquid and partly like a rubber band. Squeezing it through a narrowing channel stretches its molecules, and the stretched molecules change how much pressure the flow needs.

### 1.1 A polymer liquid remembers being stretched

Pour honey and it flows smoothly; its resistance comes only from viscosity. Now dissolve a little long-chain polymer in it. Each long molecule is like a tiny coiled rubber band. When the flow speeds up, as it must in a narrowing channel, the rubber bands are pulled straight along the flow. They pull back, and they take a while to coil up again.

You have seen this "memory" before: a string of egg white or saliva that stretches between your fingers, or slime that bounces if you throw it but flows if you leave it on a table. Engineers meet it when polymer melts are pushed through extrusion dies, when inks are jetted, and when protein drugs are pushed through a syringe needle.

The **Oldroyd-B model**\* pictures each molecule as two beads joined by a spring. The average stretch and direction of all the springs at a point is stored in a $2\times2$ matrix $\mathbf{A}$, the *conformation tensor*†. At rest the springs are coiled and point every which way. Flow stretches them, and they relax back over a time $\lambda$.

<small>\* Named after J. G. Oldroyd (1950). It is the simplest model with a constant viscosity and elasticity, and it describes "Boger fluids" (dilute polymer in a thick solvent) well.<br>† A tensor here is just a small matrix of numbers attached to each point. $A_{zz}$ measures stretch along the channel, $A_{yy}$ across it, and $A_{yz}$ the tilt.</small>
""")

md(r"""
### 1.2 The channel, and the three numbers that control the flow

**In brief.** The geometry is fixed (a 4:1 hyperbolic contraction with straight sections before and after, as in the paper's simulations). Three dimensionless numbers decide what happens.
""")

code(r"""
fig = rp.channel_sketch()                            # drawn to scale in Z and Y
plt.show()
""")

md(r"""
**Figure 1:** the channel in the paper's scaled units, with the straight inlet and outlet sections of its simulations. **Takeaway:** the half-height falls four-fold, so the flow speeds up four-fold and the molecules are stretched along the way.
""")

md(r"""
| Symbol | Meaning | Everyday reading | Values used |
|---|---|---|---|
| ε = h<sub>ℓ</sub>/ℓ | how narrow the channel is | a drinking straw ($\epsilon$ small) versus a short pipe | 0 (limit), 0.02, 0.1 |
| De = λu<sub>c</sub>/ℓ | memory time of the molecules ÷ time to cross the contraction | does the rubber band spring back before it leaves the funnel? | 0 to 0.3 |
| β̃ = η<sub>p</sub>/η<sub>0</sub> | share of the viscosity that comes from the polymer | how much rubber is in the honey | 0.4 |

Recall that the paper scales every quantity by its natural size in a narrow channel. In these units every term in the equations is of order one, which also suits a neural network.
""")

md(r"""
### 1.3 The equations we solve

**In brief.** Five unknown fields ($U_z, U_y, P$ and the three parts of $\mathbf{A}$), five equations, and conditions on the walls, the centreline and the inlet.

We use the paper's scaled equations (2.14), written with the polymer stress $\mathbf{T}=\tilde\beta\mathbf{A}$. They are exact. Setting $\epsilon = 0$ gives the lubrication limit\*, and keeping $\epsilon$ gives the full 2-D flow.

**Mass:** what flows in flows out.
$$\frac{\partial U_z}{\partial Z}+\frac{\partial U_y}{\partial Y}=0$$

**Force along the channel:** the pressure push balances the solvent's friction and the polymer's pull.
$$\frac{\partial P}{\partial Z}=(1-\tilde\beta)\Big(\epsilon^2\frac{\partial^2U_z}{\partial Z^2}+\frac{\partial^2U_z}{\partial Y^2}\Big)+\frac{\partial T_{zz}}{\partial Z}+\frac{\partial T_{yz}}{\partial Y}$$

**Force across the channel:** in a narrow channel the pressure is almost the same across it.
$$\frac{\partial P}{\partial Y}=\epsilon^2\Big[(1-\tilde\beta)\Big(\epsilon^2\frac{\partial^2U_y}{\partial Z^2}+\frac{\partial^2U_y}{\partial Y^2}\Big)+\frac{\partial T_{yz}}{\partial Z}+\frac{\partial T_{yy}}{\partial Y}\Big]$$

**Stretch of the molecules:** carried by the flow, stretched by velocity gradients, relaxing towards rest.
$$\mathrm{De}\Big(U_z\tfrac{\partial A_{zz}}{\partial Z}+U_y\tfrac{\partial A_{zz}}{\partial Y}-2\tfrac{\partial U_z}{\partial Z}A_{zz}-2\tfrac{\partial U_z}{\partial Y}A_{yz}\Big)-2\epsilon^2\tfrac{\partial U_z}{\partial Z}=-A_{zz}$$
$$\mathrm{De}\Big(U_z\tfrac{\partial A_{yy}}{\partial Z}+U_y\tfrac{\partial A_{yy}}{\partial Y}-2\tfrac{\partial U_y}{\partial Z}A_{yz}-2\tfrac{\partial U_y}{\partial Y}A_{yy}\Big)-2\tfrac{\partial U_y}{\partial Y}=-A_{yy}$$
$$\mathrm{De}\Big(U_z\tfrac{\partial A_{yz}}{\partial Z}+U_y\tfrac{\partial A_{yz}}{\partial Y}-\tfrac{\partial U_y}{\partial Z}A_{zz}-\tfrac{\partial U_z}{\partial Y}A_{yy}\Big)-\epsilon^2\tfrac{\partial U_y}{\partial Z}-\tfrac{\partial U_z}{\partial Y}=-A_{yz}$$

**Conditions.** Wall $Y=H(Z)$: $U_z=U_y=0$. Centreline $Y=0$: mirror symmetry. Flow rate: $\int_0^{H}U_z\,dY=1$. Inlet: fully developed flow and stress (as in the paper's simulations). The stress needs no condition at the walls or outlet, because molecules carry their stretch downstream with the flow.

<small>\* Lubrication limit: the channel is so long and thin that the flow at each $Z$ looks like flow between parallel plates. Bearings and thin films are classic examples.</small>
""")

# =============================================================== 2. references
md(r"""
## 2. Three references to check the network against

**In brief.** A neural network's training loss can go down while its answer is still wrong. So before we trust it, we build three independent references: the paper's formulas, the paper's simulations, and our own solver.

### 2.1 The paper's formulas, re-derived and checked

The paper expands the solution in powers of De (like a Taylor series). This gives $\Delta P=\Delta P_0+\mathrm{De}\,\Delta P_1+\mathrm{De}^2\Delta P_2+\mathrm{De}^3\Delta P_3$, with closed forms for the hyperbolic channel (eq. 5.2). We re-derived the whole expansion with SymPy\* for a general channel shape, and computed $\Delta P_3$ directly instead of through the paper's reciprocal theorem.

<small>\* SymPy: a free Python library for exact algebra and calculus. Script: `03_code/scripts/derive_perturbation_sympy.py`.</small>
""")

code(r"""
print((RES / "analytical" / "sympy_check.txt").read_text())     # OK = formula confirmed
""")

md(r"""
Every printed formula is confirmed except one sign: in arXiv eq. (3.16b) the first-order shear conformation must read $A_{yz,1}=-18H'Y(H^2-2Y^2)/H^7$. The corrected sign is the one that reproduces the colour bars of the paper's figure 5, so it is a typesetting slip, not an error in the paper's results.

The series has coefficients of alternating sign that grow about three-fold per order (+1.0, −1.8, +5.5, −14.7). It is therefore reliable only for small De, roughly De ≲ 0.1, and we need other references beyond that.

### 2.2 The paper's simulations, recovered from its PDF

**In brief.** The authors did not publish their simulation data, but the figures in the arXiv PDF are vector drawings, so each data marker is stored as exact coordinates. We read them out.

The axes are calibrated against the paper's own formula curves, which we know exactly (fitting error about $3\times10^{-6}$). Two checks confirm that a marker's data point is its centroid: the straight channel ($\alpha=1$, figure 7) gives $\Delta P=2.9999$ (exact value 3), and the Newtonian points of figures 2 and 8 agree with each other. Script: `03_code/scripts/digitize_fe_data.py`.
""")

code(r"""
fe = rp.load_fe(ROOT)                                # {eps: (De values, Delta P values)}
fig = rp.replot_paper_fig2(fe)                       # our re-plot of the paper's figure 2
plt.show()
""")

md(r"""
**Figure 2:** the paper's figure 2, re-plotted from the recovered data. **Takeaway:** the simulations (triangles) follow the formulas only at small De; the 3rd-order formula falls away above De ≈ 0.15.

### 2.3 Our own reference solver, and why marching fails

**In brief.** We wrote an independent solver for the lubrication limit ($\epsilon\to0$) that works at any De. It agrees with the paper's simulations to within <<ref_fe>> %.

*How it works.* Across the channel we use the mapped coordinate $\eta=Y/H(Z)$, so the wall is always at $\eta=1$, and represent each field by a polynomial on Chebyshev points\*. Along the channel we use finite differences. All equations at all stations are then solved together by Newton's method†. One solve takes about 5 seconds.

*Why not march downstream?* Since molecules carry their stretch downstream, it is tempting to solve station by station, like reading a book page by page. We tried this, and it blew up for every De > 0, sooner on finer grids. The reason is a hidden loop: the gradient of the axial stress, $\partial T_{zz}/\partial Z$, changes the velocity, and the velocity gradient feeds back into $T_{zz}$. Together they act like heat conduction along the channel. You cannot find the temperature along a metal rod by walking from the left end only; you also need the right end. So we solve the whole channel at once, with an outflow condition at the far end. The PINN does the same by design.

<small>\* Chebyshev points: grid points bunched near the two ends of an interval, which makes polynomial fits very accurate.<br>† Newton's method: guess, measure how wrong every equation is, correct all unknowns together using the slopes, repeat. It converges here in 3–5 rounds.</small>
""")

code(r"""
# our eps -> 0 solver at 13 values of De, then its grid / corner / outlet checks
ref = compare.load_reference(RES / "reference" / "fd_eps0.npz")
print((RES / "reference" / "convergence.txt").read_text())
fig, rows = rp.reference_vs_fe(ref, fe, eps=0.02)
plt.show()
rp.print_table(["De", "our solver", "paper FE", "diff (%)", "formula O(De3)"], rows,
               ["{:.3f}", "{:.4f}", "{:.4f}", "{:+.2f}", "{:.4f}"])
""")

md(r"""
**Figure 3:** pressure drop from our reference solver (orange line) and the paper's simulations at $\epsilon=0.02$ (triangles). **Takeaway:** the two independent calculations agree within <<ref_fe>> % over the whole range, while the 3rd-order formula is <<formula_03>> % low at De = 0.3.

The convergence table above shows that the answer does not depend on our numerical choices: finer grids across and along the channel, sharper or rounder corners, and a longer outlet all change $\Delta P$ by less than $2\times10^{-4}$. The small gap at De = 0 (−0.15 %) is a real 2-D effect: the paper's simulation has $\epsilon=0.02$, while our solver is at $\epsilon=0$.
""")

# =============================================================== 3. PINN
md(r"""
## 3. How the PINN works

**In brief.** A neural network is a flexible formula with thousands of adjustable numbers. We adjust them until the network's output satisfies the flow equations everywhere in the channel. It is never shown a solution.

### 3.1 A network graded on the equations, not on answers

Think of a student who is never shown the answer key. Instead, after each attempt, they are told how badly each equation is violated at a few thousand random points in the channel, and they improve. That is a PINN\*.

```
 (Z, Y) ──► network ──► stream function Ψ, pressure P, stretch A
                              │
              automatic differentiation† gives every derivative exactly
                              │
          equation residuals at random points ("how wrong is each equation?")
                              │
       loss = mean of squared residuals ──► optimiser adjusts the weights ──► repeat
```

<small>\* PINN: physics-informed neural network (Raissi, Perdikaris & Karniadakis, 2019).<br>† Automatic differentiation ("autograd"): the computer applies the chain rule to the network formula itself, so $\partial U_z/\partial Y$ is exact, not a finite-difference estimate.</small>

### 3.2 Boundary conditions built into the network

**In brief.** Instead of asking the network to *learn* that the wall does not move, we build the wall into the formula, like a stencil that can only draw inside the lines. Then the training only has to deal with the equations.

The network has one small sub-network per field. Each output $N$ is turned into a physical field by a fixed formula:

* **Velocity.** Stream function $\Psi=\tfrac12(3\eta-\eta^3)+\eta(1-\eta^2)^2\,c\,N_\psi(Z,\eta^2)$, with $U_z=\partial\Psi/\partial Y$ and $U_y=-\partial\Psi/\partial Z$.
  Exact for any weights: mass balance, no slip, no penetration, symmetry, flow rate = 1.
* **Pressure.** $P=N_P(Z)+\epsilon^2N_{P1}(Z,\eta^2)$.
  Exact: the pressure is uniform across the channel when $\epsilon=0$, as lubrication theory requires.
* **Polymer stretch.** $\mathbf{A}=\mathbf{A}_{\text{inlet}}(\eta)+(\text{ramp})\times(\text{local size})\times N_A$.
  Exact: the fully developed inlet stress, and $A_{yz}$ odd in $\eta$.

Here $\eta=Y/H(Z)$, so the wall is always at $\eta=1$. The stream function‡ $\Psi$ is the key: both velocities come from its slopes, so mass is conserved automatically. Feeding $\eta^2$ rather than $\eta$ makes every output mirror-symmetric automatically. The "local size" factors (for example $3/H^2$ for $A_{yz}$) keep the network's outputs near 1 both at the wide inlet and at the narrow outlet, where stresses are 256 times larger.

The next cell proves it. We take a network, scramble its last layer with random numbers so its output is nonsense, and check the conditions anyway.

<small>‡ Stream function: one function $\Psi(Z,Y)$ whose slopes give the two velocities. Lines of constant $\Psi$ are the paths the liquid follows, like contour lines on a map. The flow rate between two such lines is the difference of their $\Psi$ values, so setting $\Psi=0$ on the centreline and $\Psi=1$ on the wall fixes the flow rate at 1.</small>
""")

code(r"""
check = pinn.OldroydPINN(pinn.PINNConfig(eps=0.02, De=0.1, geo_features=1)).double()
rows = rp.constraint_check(check)                    # random weights, conditions still hold
rp.print_table(["Z", "Uz moved by", "flow rate", "|Uz| wall", "|Uy| wall", "|Uy| centre",
                "|Ayz| centre"], rows,
               ["{:+.2f}", "{:.2f}", "{:.12f}", "{:.1e}", "{:.1e}", "{:.1e}", "{:.1e}"])
""")

md(r"""
The second column shows the scrambled network really changes the flow: the velocity profile differs from the textbook parabola by this much. Yet the flow rate is 1 to twelve digits, and every wall and centreline value is zero. These conditions never appear in the loss.

### 3.3 Training: two optimisers and what it costs

**In brief.** Training adjusts tens of thousands of numbers (counted below) in two stages, Adam then L-BFGS, and takes 7–50 minutes per case on a laptop (table below).

* The loss is the mean squared residual of the equations at 4096–6144 random points, denser near the outlet, the corners and the wall. Fresh points are drawn every 50 steps, so the network cannot memorise them.
* Each residual is divided by its local natural size, so the wide inlet and the narrow outlet both count. For the full 2-D runs we never divide by less than 10 % of the outlet size; otherwise the slow inlet corner takes over the loss (our choice, found by testing).
* Adam\* (6000 steps for the first De of each series, 2000–3000 for the next ones) moves quickly towards a good region. L-BFGS† (900–1500 steps, in double precision, i.e. 16 significant digits instead of 7) then polishes the answer; for the full 2-D runs it restarts three times on fresh points.
* Continuation in De: each De starts from the trained network of the previous De, like climbing stairs one step at a time.
* Hardware: everything ran on one laptop (Apple M4). The $\epsilon = 0.1$ networks were trained on its GPU in single precision, alongside the CPU runs; this made the whole campaign about 1.4 times faster.
* Corners: the corners where the straight sections meet the hyperbola are rounded over 1 % of the length (3 % in the full 2-D runs). Our reference solver shows this changes $\Delta P$ by at most 0.03 %.

<small>\* Adam: a gradient-descent method that adapts its step size for each weight (Kingma & Ba, 2015).<br>† L-BFGS: a quasi-Newton method that also uses the curvature of the loss; slower per step but far more precise near the minimum.</small>
""")

code(r"""
runs0 = rp.load_runs(ROOT, 0.0)                      # lubrication limit, one network per De
runs002 = rp.load_runs(ROOT, 0.02)                   # full 2-D, paper's case I
runs01 = rp.load_runs(ROOT, 0.1)                     # full 2-D, paper's case II
param0, param002 = rp.load_param(ROOT, 0.0), rp.load_param(ROOT, 0.02)
De_show = 0.1 if 0.1 in runs0 else next(iter(runs0))
fig = rp.loss_history(runs0[De_show][1], title=f"Training, eps = 0, De = {De_show:g}")
plt.show()

rows = []                                            # one row per trained network
for e, runs in [(0.0, runs0), (0.02, runs002), (0.1, runs01)]:
    for d, (m, h) in runs.items():
        n_par = sum(p.numel() for p in m.parameters())
        rows.append((e, d, n_par, h["train_seconds"] / 60, h["loss"][-1], h["test_loss"]))
rp.print_table(["eps", "De", "weights", "minutes", "final loss", "held-out loss"], rows,
               ["{:g}", "{:.2f}", "{:,d}", "{:.1f}", "{:.1e}", "{:.1e}"])
""")

md(r"""
**Figure 4:** training loss for each equation. **Takeaway:** across all runs, Adam brings the loss down by two to four orders of magnitude and L-BFGS then lowers it a further 2 to 30 times. The loss on fresh, unseen points (table, last column) is close to the training loss, so the networks have not overfitted their training points.
""")

# =============================================================== 4. results
md(r"""
Want to see it happen? With `TRAIN_DEMO = True` the next cell trains a small network from scratch (fewer steps than the saved ones) and prints its progress.
""")

code(r"""
if TRAIN_DEMO:
    cfg = pinn.PINNConfig(eps=0.0, De=0.1, geo_features=1, adam_iters=1500, lbfgs_iters=200)
    demo, hist = pinn.train(cfg, RES / "pinn" / "notebook_demo", log_every=250)
    dp_ref = float(np.interp(0.1, ref["De"], ref["dP"]))
    print(f"demo network: Delta P = {demo.pressure_drop(0.1)['direct']:.4f}"
          f"   (reference solver {dp_ref:.4f})")
else:
    print("Skipped. Set TRAIN_DEMO = True in the set-up cell to watch a short training run.")
""")

md(r"""
## 4. Results

### 4.1 Sanity check: the Newtonian case (De = 0)

**In brief.** With no elasticity the exact answer is known, $\Delta P_0 = 3\int_0^1 dZ/H^3 = 255/256 = 0.99609$. The PINN must reproduce it.
""")

code(r"""
m, h = runs0[0.0]
dp = m.pressure_drop(0.0)                            # two independent routes to Delta P
print(f"PINN  Delta P (pressure difference)   = {dp['direct']:.5f}")
print(f"PINN  Delta P (integrated momentum)   = {dp['momentum']:.5f}")
print(f"exact Delta P0 = 255/256              = {255/256:.5f}")
print(f"relative error                        = {abs(dp['direct'] / (255/256) - 1):.1e}")
""")

md(r"""
### 4.2 Pressure drop falls as elasticity grows

**In brief.** In the lubrication limit the PINN reproduces our reference solver's pressure drop at every De we tried, including De = 0.2–0.3, where the paper's formulas fail.
""")

code(r"""
fig, rows = rp.dp_lubrication(ref, runs0, param0)
plt.show()
rp.print_table(["De", "PINN", "PINN mom.", "reference", "PINN err %", "mom. err %",
                "formula O3"], rows,
               ["{:.2f}", "{:.5f}", "{:.5f}", "{:.5f}", "{:+.3f}", "{:+.3f}", "{:.5f}"])
""")

md(r"""
**Figure 5:** pressure drop against De in the lubrication limit. **Takeaway:** the PINN points sit on the reference solver's curve. The two PINN routes to $\Delta P$ (pressure difference and integrated force balance) agree, which shows the network is internally consistent.

Why does elasticity *lower* the pressure drop? The molecules are stretched most near the outlet wall. Their tilted, stretched state pulls the liquid along, partly doing the work that pressure would otherwise do. Section 4.5 measures this directly.
""")

md(r"""
### 4.3 Where the polymer is stretched: field maps and errors

**In brief.** The PINN gets the whole flow field right, not just one number. Over most of the contraction its error is well below 1 %; it reaches a few percent only near the outlet corner.
""")

code(r"""
De_map = 0.3 if 0.3 in runs0 else max(runs0)
fig, errs = rp.field_comparison(runs0[De_map][0], compare.reference_at(ref, De_map), De_map)
plt.show()
rows = rp.error_table(ref, runs0)                    # relative L2 error, percent
rp.print_table(["De", "Uz", "Uy", "Azz", "Ayy", "Ayz", "worst, 0.05<Z<0.95"], rows,
               ["{:.2f}"] + ["{:.2f}"] * 6)
""")

md(r"""
**Figure 6:** PINN (left), reference solver (middle) and their difference (right) at the largest De. Stresses are shown as $T=\tilde\beta A$. **Takeaway:** the axial stress $T_{zz}$ is largest on the wall at the outlet, the shear stress $T_{yz}$ is negative and also largest there, and the PINN matches both. Its remaining errors sit at the corners ($Z\approx0$ and $Z\approx1$).

The table gives the relative error\* of each field in percent. The last column excludes the two corner zones (the first and last 5 % of the contraction) and shows that the errors there dominate. A dash for $A_{zz}$ at De = 0 means the reference field is exactly zero there, so a relative error has no meaning.

<small>\* Relative L2 error: $\sqrt{\sum(f_{\text{PINN}}-f_{\text{ref}})^2/\sum f_{\text{ref}}^2}$ over all reference grid points in $0\le Z\le1$.</small>
""")

code(r"""
De_err = [d for d in (0.1, 0.3) if d in runs0] or [max(runs0)]
fig = rp.error_along_channel(runs0, ref, De_err)     # local error at every station Z
plt.show()
""")

md(r"""
**Figure 7:** the PINN's local error at each position along the contraction. **Takeaway:** over most of the channel the error is 0.01–0.5 %. It reaches a few percent inside the two corner zones (shaded), where the wall slope changes within about 1 % of the length, and just upstream of the outlet corner, where the stresses are largest. The corners are the main limit of this PINN.
""")

md(r"""
### 4.4 Full 2-D flow against the paper's simulations

**In brief.** With $\epsilon>0$ the PINN solves the full 2-D equations, exactly what the paper's finite-element simulations solve. We compare with the recovered simulation data for both of the paper's channels.
""")

code(r"""
fig, rows = rp.dp_vs_fe(fe, {0.02: runs002, 0.1: runs01}, {0.02: param002})
plt.show()
rp.print_table(["eps", "De", "PINN", "PINN mom.", "paper FE", "PINN err %",
                "mom. err %", "formula err %"], rows,
               ["{:g}", "{:.2f}", "{:.4f}", "{:.4f}", "{:.4f}", "{:+.2f}", "{:+.2f}",
                "{:+.1f}"])
""")

md(r"""
**Figure 8:** PINN (circles) against the paper's finite-element simulations (triangles) for the narrow channel $\epsilon=0.02$ (left) and the wider one $\epsilon=0.1$ (right). **Takeaway:** compare the error columns of the table. The PINN's error against the simulations stays below about 1 %, while the 3rd-order formula's grows with De to over 20 %. At $\epsilon=0.1$ the formula is off even at De = 0, because it leaves out the 2-D effects that the PINN includes.
""")

md(r"""
A second check needs no simulation at all. At De = 0 the full 2-D pressure drop differs from the lubrication value by a small amount of size $\epsilon^2$. Extended lubrication theory\* gives that amount; we derived it with SymPy for this channel: $\Delta P \approx 255/256 + C\,\epsilon^2$.

<small>\* Tavakol, Froehlicher, Holmes & Stone, *Proc. R. Soc. A* 473 (2017). Script: `03_code/scripts/newtonian_eps2_correction.py`.</small>
""")

code(r"""
import re
txt = (RES / "analytical" / "newtonian_eps2.txt").read_text()
C = float(re.search(r"C = ([0-9.]+)", txt).group(1))         # from the SymPy derivation
print(f"extended lubrication: Delta P = 255/256 + C eps^2,  C = {C:.4f}\n")
for e, runs in [(0.02, runs002), (0.1, runs01)]:
    if 0.0 in runs:
        dp = runs[0.0][0].pressure_drop(0.0)["direct"]
        print(f"eps = {e:<4g}  PINN {dp:.5f}   extended lubrication {255/256 + C*e**2:.5f}"
              f"   paper FE {fe[e][1][0]:.5f}")
""")

code(r"""
fig = rp.stress_maps(runs002)                        # cf. the paper's figures 4, 5
plt.show()
rp.print_table(["De", "max Tzz PINN", "max Tzz ref.", "min Tyz PINN", "min Tyz ref."],
               rp.stress_peaks(runs002, ref), ["{:.1f}", "{:.2f}", "{:.2f}", "{:.2f}", "{:.2f}"])
""")

md(r"""
**Figure 9:** PINN polymer stresses at $\epsilon=0.02$ for De = 0.1, 0.2 and 0.3, the cases of the paper's figures 4(d–f) and 5(d–f). **Takeaway:** as in the paper, $T_{zz}$ is positive and $T_{yz}$ negative, both concentrated along the wall near the outlet, and both grow with De. The table compares the peak values with our reference solver at $\epsilon\to0$; the small differences are 2-D effects plus the smoothing of the outlet corner.
""")

md(r"""
### 4.5 Why the pressure drops: the shear-stress term does the work

**In brief.** Integrating the force balance along the centreline splits $\Delta P$ into four parts (paper eq. 5.3): ① and ③ are axial viscous and polymer stresses, ② is the solvent-like shear term, and ④ is the gradient of the polymer shear stress. The paper's key physical claim is that ④ causes the drop. We check it.
""")

code(r"""
fig = rp.contributions_figure(ref, runs0)            # paper eq. (5.3), centreline
plt.show()
""")

md(r"""
**Figure 10:** the four contributions against De. **Takeaway:** ② barely changes and ③ is zero on the centreline. The whole drop comes from ④, the gradient of the polymer shear stress, which confirms the paper's conclusion. The 3rd-order formula over-predicts ④ at larger De, consistent with the paper's figure 3.
""")

md(r"""
### 4.6 One network for every De: explore it yourself

**In brief.** One network that takes De as an extra input gives the whole pressure-drop curve, and a field map for any De, in a fraction of a second. Move the slider (in a running notebook; the HTML copy shows the starting value).
""")

code(r"""
if param002 is not None:
    rp.explorer(param002)                            # slider: 0 <= De <= 0.3, eps = 0.02
elif param0 is not None:
    rp.explorer(param0)
else:
    print("De-as-input network not trained yet (see Appendix A).")
""")

md(r"""
**Figure 11:** axial polymer stress from the single De-as-input network at three values of De, and the pressure-drop curve it gives. **Takeaway:** one trained network replaces a sweep of separate solves, and each new De takes a fraction of a second.
""")

# =============================================================== 5. conclusions
md(r"""
## 5. What we learned, what is left

**In brief.** A PINN trained only on the Oldroyd-B equations reproduces the pressure drop and stress fields of Boyko & Stone (2022), including the elastic regime where the paper's formulas break down.

**What worked**

1. The paper's analysis is correct; we confirmed every formula symbolically and found one sign typo in arXiv eq. (3.16b).
2. The paper's simulation data, recovered from the PDF, give a precise check. Our independent reference solver agrees with it to within <<ref_fe>> %.
3. In the lubrication limit the PINN's pressure drop is within <<lub_dir>> % of the reference solver for every De up to 0.3 (<<lub_mom>> % by the momentum route), where the 3rd-order formula is off by <<formula_03>> %. Against the paper's full 2-D simulations its largest error is <<fe_max>> % (<<fe_mom>> % by the momentum route). The two routes differ most at $\epsilon=0.1$, where the training loss is also highest.
4. Building the boundary conditions into the network (stream function, symmetric inputs, exact inflow) leaves only the equations in the loss, so no loss weights need tuning.
5. Splitting the pressure into a part along the channel plus an $\epsilon^2$ part across it removed a hidden 0.25 % bias. The network had been using cross-channel pressure differences to hide errors in the other equations.
6. Elasticity lowers the pressure drop because of the polymer shear-stress gradient (term ④), as the paper says.

**Limits (stated plainly)**

* The PINN is slower than the reference solver for one case (minutes against seconds). Its advantages are the mesh-free formulation, the exact constraints, and the De-as-input network that answers any De instantly once trained.
* Errors are largest in the two corner zones where the wall slope jumps. We rounded the corners over 1 % of the length; our solver shows this changes $\Delta P$ by less than $2\times10^{-4}$.
* We stopped at De = 0.3, the paper's range. Above De ≈ 0.5, stretching along the centreline becomes unbounded in the Oldroyd-B model, and other solvers also struggle there (Housiadas 2025).

**Next steps:** higher De with a log-conformation output; the FENE-CR model of Mahapatra et al. (2025); inverse problems, such as finding $\lambda$ from a measured pressure drop.
""")

md(r"""
## 6. Appendix

### A. How to retrain everything

Each command runs from `03_code/scripts/`. Times are for an Apple M4 laptop, CPU only.

| Step | Command | Time |
|---|---|---|
| check the paper's formulas | `python derive_perturbation_sympy.py` | 5 s |
| recover the paper's data | `python digitize_fe_data.py` | 2 s |
| reference solver sweep | `python run_reference_solver.py` | 3 min |
| PINNs, lubrication limit | `python run_campaign.py --chain eps0` | ~1 h |
| PINNs, ε = 0.02 | `python run_campaign.py --chain eps0.02` | ~1.5 h |
| PINNs, ε = 0.1 | `python run_campaign.py --chain eps0.1` | ~1.2 h |
| De-as-input network | `python run_campaign.py --chain param` | ~1 h |
| a second lane on the Mac GPU (runs alongside a CPU lane) | `python run_campaign.py --chain eps0.1 --gpu` | ~45 min |
| everything, one run after another | `python run_campaign.py --chain all` | ~4 h |
| zip the project to share it | `python make_bundle.py` | 5 s |
| one custom run | `python train_pinn.py --name test --eps 0 --De 0.1 --geo-features 1` | 10 min |
| quick tests | `python -m pytest -q 03_code/tests` (from the project folder) | 5 s |

### B. File map and references

* Paper notes, literature review (56 references), code survey: `01_research/notes/`
* Plan and design decisions: `02_plan/project_plan.md`
* Code: `03_code/oldroydb/` (`geometry`, `analytical`, `lubrication_fd`, `pinn`, `compare`, `report`, `plotting`)
* Results: `04_results/` (SymPy log, recovered data, reference solutions, trained networks, logs)

**Main references.** E. Boyko & H. A. Stone, *J. Fluid Mech.* 936, A23 (2022). E. Boyko & H. A. Stone, *Phys. Rev. Fluids* 6, L081301 (2021). E. J. Hinch, E. Boyko & H. A. Stone, *J. Fluid Mech.* 988, A11 (2024). K. D. Housiadas & A. N. Beris, *Phys. Fluids* 35, 123116 (2023). M. Raissi, P. Perdikaris & G. E. Karniadakis, *J. Comput. Phys.* 378, 686 (2019). S. Wang et al., "An expert's guide to training physics-informed neural networks", arXiv:2308.08468 (2023).

*Use of AI: the research survey, code and text of this project were developed with the help of an AI assistant (Claude), as stated in the project proposal.*
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).with_name("CLL731_OldroydB_PINN.ipynb")
nbf.write(nb, out)
print(f"wrote {out} ({len(cells)} cells)")

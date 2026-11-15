"""Figures for the term-paper report (06_submission/report/).

Rule for the report: every plot shows output of OUR code only. That is the
PINN, our reference solver, and our own evaluation of the low-De perturbation
formulas, which we re-derived with SymPy (03_code/scripts/derive_perturbation_sympy.py).
The finite-element points recovered from the journal article are NOT plotted;
the report compares with them in the text only, with a citation.

Run:  python make_report_figures.py      (about a minute; reads 04_results only)
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "03_code"))
from oldroydb import analytical as an, compare, plotting as pl, report as rp  # noqa: E402
from oldroydb.geometry import ChannelGeometry  # noqa: E402

OUT = Path(__file__).with_name("figures")
OUT.mkdir(exist_ok=True)
pl.style()
plt.rcParams.update({"savefig.dpi": 220})

RENAME = {
    "paper formula, O(De$^1$)": "low-De theory, O(De)",
    "paper formula, O(De$^2$)": "low-De theory, O(De$^2$)",
    "paper formula, O(De$^3$)": "low-De theory, O(De$^3$)",
    "② paper formula, O(De²)": "② low-De theory, O(De²)",
    "④ paper formula, O(De³)": "④ low-De theory, O(De³)",
}


def relabel(ax, **legend_kw):
    for ln in ax.get_lines():
        if ln.get_label() in RENAME:
            ln.set_label(RENAME[ln.get_label()])
    if ax.get_legend() is not None or legend_kw:
        ax.legend(**legend_kw)


def save(fig, name):
    for ax in fig.axes:                       # captions carry the titles
        for loc in ("left", "center", "right"):
            ax.set_title("", loc=loc)
    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print("  ", name)


ref = compare.load_reference(ROOT / "04_results" / "reference" / "fd_eps0.npz")
runs0 = rp.load_runs(ROOT, 0.0)
runs002 = rp.load_runs(ROOT, 0.02)
runs01 = rp.load_runs(ROOT, 0.1)
param002 = rp.load_param(ROOT, 0.02)

# ---- Fig. 1: geometry sketch (half channel solved; centreline = symmetry line)
g = ChannelGeometry(4.0, 0.5, 0.5, 0.0001)
Z = np.linspace(g.Z_min, g.Z_max, 800)
H = g.H(Z)
fig, ax = plt.subplots(figsize=(7.6, 3.4))
ax.fill_between(Z, -H, H, color="#e6eef9")
ax.fill_between(Z, 0, H, color="#cfdff4")
ax.plot(Z, H, color=pl.INK, lw=1.6)
ax.plot(Z, -H, color=pl.INK, lw=1.6)
ax.axhline(0, color=pl.THEORY, lw=0.9, ls=(0, (6, 3)))
for z0 in (0.0, 1.0):
    ax.axvline(z0, color=pl.THEORY, lw=0.7, ls=":")
ax.annotate("", xy=(-0.12, 2.0), xytext=(-0.45, 2.0),
            arrowprops=dict(arrowstyle="->", color=pl.PINN, lw=2))
ax.text(-0.285, 2.25, "flow rate $q$", ha="center", fontsize=9)
ax.text(0.5, 2.75, "wall $y = h(z)$: no slip, no penetration", fontsize=9)
ax.text(0.62, 0.12, "centreline $y=0$: symmetry", fontsize=9, color=pl.THEORY)
ax.text(0.42, -1.6, "$h(z)=\\dfrac{h_0\\,h_\\ell\\,\\ell}{(h_0-h_\\ell)z+h_\\ell\\,\\ell}$",
        fontsize=10, ha="center")
ax.annotate("", xy=(-0.42, 4.0), xytext=(-0.42, 0.0),
            arrowprops=dict(arrowstyle="<->", color=pl.INK, lw=0.9))
ax.text(-0.40, 1.0, " $h_0$", fontsize=10)
ax.annotate("", xy=(1.42, 1.0), xytext=(1.42, 0.0),
            arrowprops=dict(arrowstyle="<->", color=pl.INK, lw=0.9))
ax.text(1.45, 0.35, "$h_\\ell$", fontsize=10)
ax.annotate("", xy=(1.0, -5.0), xytext=(0.0, -5.0),
            arrowprops=dict(arrowstyle="<->", color=pl.INK, lw=0.9))
ax.text(0.5, -4.9, "contraction, length $\\ell$  ($\\Delta p$ measured over it on $y=0$)",
        ha="center", va="bottom", fontsize=9)
ax.annotate("", xy=(0.0, -5.0), xytext=(-0.5, -5.0),
            arrowprops=dict(arrowstyle="<->", color=pl.THEORY, lw=0.8))
ax.text(-0.25, -4.9, "inlet $0.5\\ell$", ha="center", va="bottom", fontsize=8.5)
ax.annotate("", xy=(1.5, -5.0), xytext=(1.0, -5.0),
            arrowprops=dict(arrowstyle="<->", color=pl.THEORY, lw=0.8))
ax.text(1.25, -4.9, "outlet $0.5\\ell$", ha="center", va="bottom", fontsize=8.5)
ax.text(-0.25, -2.6, "fully developed\nvelocity and stress", ha="center", fontsize=8.5)
ax.text(1.25, -2.6, "free outflow", ha="center", fontsize=8.5)
ax.text(0.0, -6.0, "$z=0$", ha="center", fontsize=9)
ax.text(1.0, -6.0, "$z=\\ell$", ha="center", fontsize=9)
ax.set_xlim(g.Z_min - 0.02, g.Z_max + 0.08)
ax.set_ylim(-6.2, 4.6)
ax.set_xticks([])
ax.set_yticks([])
for s in ax.spines.values():
    s.set_visible(False)
ax.grid(False)
save(fig, "r01_geometry.png")

# ---- Fig. 3: Delta P vs De, lubrication limit
fig, rows_lub = rp.dp_lubrication(ref, runs0)
relabel(fig.axes[0], loc="lower left", fontsize=8)
save(fig, "r03_dp_lubrication.png")

# ---- Fig. 4: field maps at De = 0.3, PINN vs reference
fig, errs = rp.field_comparison(runs0[0.3][0], compare.reference_at(ref, 0.3), 0.3,
                                names=("Uz", "Azz", "Ayz"))
fig.set_size_inches(13, 7.2)
for ax in fig.axes:                       # field_comparison puts its labels in titles
    ax.set_title(ax.get_title(), fontsize=10)
fig.savefig(OUT / "r04_fields_De03.png", bbox_inches="tight", pad_inches=0.04)
plt.close(fig)
print("   r04_fields_De03.png", {k: round(100 * v, 2) for k, v in errs.items()})

# ---- Fig. 5: local error along the channel
fig = rp.error_along_channel(runs0, ref, [0.1, 0.3])
for ax, d in zip(fig.axes, (0.1, 0.3)):
    ax.legend(ncol=5, fontsize=8, loc="upper center")
    ax.text(0.98, 0.04, f"De = {d:g}", transform=ax.transAxes, ha="right", fontsize=10)
save(fig, "r05_error_along.png")

# ---- Fig. 6: full 2-D Delta P vs De (no FE points)
De = np.linspace(0, 0.3, 200)
fig, axs = plt.subplots(1, 2, figsize=(10.4, 4.0), sharey=True)
rows_2d = []
newt = {0.02: 0.99759, 0.1: 1.03352}        # 255/256 + 3.743 eps^2, our derivation
for ax, (eps, runs) in zip(axs, ((0.02, runs002), (0.1, runs01))):
    ax.plot(De, an.pressure_drop(De, 3), color=pl.THEORY, lw=1.2, ls="--",
            label="low-De theory, O(De$^3$)")
    ax.plot(ref["De"], ref["dP"], color=pl.REF, lw=2, label="our reference solver ($\\epsilon\\to0$)")
    pts = []
    for d, (m, h) in runs.items():
        dp = m.pressure_drop(d)
        pts.append((d, dp["direct"], dp["momentum"]))
        rows_2d.append((eps, d, dp["direct"], dp["momentum"]))
    pts = np.array(pts)
    ax.plot(pts[:, 0], pts[:, 1], "o", color=pl.PINN, mec="white", ms=8, label="PINN (pressure difference)",
            zorder=5)
    ax.plot(pts[:, 0], pts[:, 2], "x", color=pl.INK, ms=7, label="PINN (momentum integral)", zorder=6)
    if eps == 0.02 and param002 is not None:
        Dg = np.linspace(0, 0.3, 31)
        ax.plot(Dg, [param002[0].pressure_drop(d)["direct"] for d in Dg], color=pl.PINN,
                lw=1.2, ls="-.", label="PINN with De as input")
    ax.plot([0], [newt[eps]], "D", color=pl.FE, ms=7, mec="white",
            label="Newtonian, $255/256 + 3.743\\epsilon^2$", zorder=7)
    ax.text(0.97, 0.95, f"$\\epsilon$ = {eps:g}", transform=ax.transAxes, ha="right", va="top",
            fontsize=11)
    ax.set_xlabel("Deborah number  De")
    ax.set_ylim(0.55, 1.06)
axs[0].set_ylabel("pressure drop  $\\Delta P$")
axs[0].legend(loc="lower left", fontsize=8)
save(fig, "r06_dp_full2d.png")

# ---- Fig. 7: polymer stress maps at eps = 0.02
fig = rp.stress_maps(runs002, (0.1, 0.3))
for ax in fig.axes[:4]:
    pass
txt = [ax.get_title() for ax in fig.axes]
fig.savefig(OUT / "r07_stress_eps002.png", bbox_inches="tight", pad_inches=0.04)
plt.close(fig)
print("   r07_stress_eps002.png", txt[:4])

# ---- Fig. 8: the four contributions to Delta P (eq. 5.3 of the article)
fig = rp.contributions_figure(ref, runs0)
relabel(fig.axes[0], ncol=2, fontsize=8, loc="center left")
save(fig, "r08_contributions.png")

# ---- Fig. 2: training history (eps = 0.02, De = 0.2)
hist = runs002[0.2][1]
fig = rp.loss_history(hist)
save(fig, "r02_loss_eps002_De02.png")

# ---- numbers quoted in the report
stress = rp.stress_peaks(runs002, ref)
contrib = rp.contributions_reference(ref)
with open(OUT / "numbers.txt", "w") as f:
    f.write("lubrication: De, PINN, PINN(mom), ref, err%, err_mom%, series3\n")
    for r in rows_lub:
        f.write("  " + "  ".join(f"{v:.4f}" for v in r) + "\n")
    f.write("full 2-D: eps, De, PINN, PINN(mom)\n")
    for r in rows_2d:
        f.write("  " + "  ".join(f"{v:.4f}" for v in r) + "\n")
    f.write("field errors De=0.3 (rel L2, %): " + str({k: round(100 * v, 2) for k, v in errs.items()}) + "\n")
    f.write("stress peaks eps=0.02: De, Tzz max PINN, Tzz max ref, Tyz min PINN, Tyz min ref\n")
    for r in stress:
        f.write("  " + "  ".join(f"{v:.4f}" for v in r) + "\n")
    f.write("contributions (reference): De, c1, c2, c3, c4\n")
    for r in contrib:
        f.write("  " + "  ".join(f"{v:.4f}" for v in r) + "\n")
    if param002 is not None:
        f.write("De-input network eps=0.02: De, dP\n")
        for d in (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3):
            f.write(f"  {d:.2f}  {param002[0].pressure_drop(d)['direct']:.4f}\n")
print("done")

"""Slide versions of the notebook figures, plus equation images.

Differences from the notebook figures: no baked-in titles (the slide title and
caption do that job) and larger text, so labels stay at least 11 pt when the
figure fills the right-hand column of a 13.33-inch slide.

Run:  python make_slide_figures.py      (about a minute; reads 04_results only)
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "03_code"))
from oldroydb import compare, plotting as pl, report as rp  # noqa: E402
from oldroydb.geometry import ChannelGeometry  # noqa: E402

OUT = Path(__file__).with_name("assets")
OUT.mkdir(exist_ok=True)
pl.style()
plt.rcParams.update({"font.size": 13, "axes.labelsize": 13, "legend.fontsize": 11,
                     "xtick.labelsize": 12, "ytick.labelsize": 12, "savefig.dpi": 220})


def save(fig, name, notitle=True):
    if notitle:                                   # clear titles in every position
        for ax in fig.axes:
            for loc in ("left", "center", "right"):
                ax.set_title("", loc=loc)
    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print("  ", name)


ref = compare.load_reference(ROOT / "04_results" / "reference" / "fd_eps0.npz")
fe = rp.load_fe(ROOT)
runs0 = rp.load_runs(ROOT, 0.0)
runs002 = rp.load_runs(ROOT, 0.02)
runs01 = rp.load_runs(ROOT, 0.1)
param002 = rp.load_param(ROOT, 0.02)

# ---- the channel (slide 3)
g = ChannelGeometry(4.0, 0.5, 0.5, 0.01)
Z = np.linspace(g.Z_min, g.Z_max, 600)
H = g.H(Z)
fig, ax = plt.subplots(figsize=(6.6, 3.3))
ax.fill_between(Z, -H, H, color="#e6eef9")
ax.plot(Z, H, color=pl.INK, lw=1.6)
ax.plot(Z, -H, color=pl.INK, lw=1.6)
ax.axhline(0, color=pl.THEORY, lw=0.8, ls=(0, (6, 3)))
for z0 in (0.0, 1.0):
    ax.axvline(z0, color=pl.THEORY, lw=0.8, ls=":")
ax.annotate("", xy=(-0.05, 2.6), xytext=(-0.45, 2.6),
            arrowprops=dict(arrowstyle="->", color=pl.PINN, lw=2))
ax.text(-0.25, 2.9, "fixed flow rate", ha="center", fontsize=12)
ax.text(0.5, 0.25, "H = 4 / (3Z + 1)", ha="center", fontsize=13)
ax.annotate("", xy=(1.0, -4.8), xytext=(0.0, -4.8),
            arrowprops=dict(arrowstyle="<->", color=pl.INK, lw=1.2))
ax.text(0.5, -4.65, "pressure drop ΔP measured here", ha="center", va="bottom", fontsize=12)
ax.set_xlim(g.Z_min, g.Z_max)
ax.set_ylim(-5.4, 4.3)
ax.set_xlabel("Z = z / ℓ  (along the channel)")
ax.set_ylabel("Y = y / h$_ℓ$")
ax.grid(False)
save(fig, "s_channel.png")

# ---- the paper's formula vs its simulations, eps = 0.02 (slide 6)
De = np.linspace(0, 0.3, 200)
fig, ax = plt.subplots(figsize=(6.6, 4.3))
pl.dp_vs_de(ax, theory=rp.series_curves(De), fe=fe[0.02])
ax.set_ylim(0.4, 1.05)
save(fig, "s_formula_vs_fe.png")

# ---- reference solver vs simulations (slide 7)
fig, _ = rp.reference_vs_fe(ref, fe, 0.02)
fig.set_size_inches(6.6, 4.3)
save(fig, "s_ref_vs_fe.png")

# ---- PINN, narrow-channel limit (slide 11)
fig, _ = rp.dp_lubrication(ref, runs0)
fig.set_size_inches(6.6, 4.3)
save(fig, "s_dp_lub.png")

# ---- where the error sits (slide 12)
fig = rp.error_along_channel(runs0, ref, [0.3])
fig.set_size_inches(6.6, 4.1)
fig.axes[0].legend(ncol=5, fontsize=11, loc="upper center")
save(fig, "s_err_along.png")

# ---- full 2-D vs simulations (slide 13, full width)
fig, _ = rp.dp_vs_fe(fe, {0.02: runs002, 0.1: runs01}, {0.02: param002})
fig.set_size_inches(12.0, 4.4)
for ax, e in zip(fig.axes, (0.02, 0.1)):
    ax.set_title(f"channel ε = {e:g}", loc="left", fontsize=13)
fig.axes[0].legend(loc="lower left", fontsize=11)
save(fig, "s_dp_vs_fe.png", notitle=False)

# ---- the four contributions (slide 14)
fig = rp.contributions_figure(ref, runs0)
fig.set_size_inches(6.6, 4.4)
fig.axes[0].legend(ncol=2, fontsize=10, loc="center left")
save(fig, "s_contrib.png")


# ---- equations as images
def eq(name, lines, size=21, width=9.0, line_h=0.62):
    fig = plt.figure(figsize=(width, line_h * len(lines) + 0.1))
    for i, s in enumerate(lines):
        fig.text(0.0, 1 - (i + 0.5) / len(lines), s, fontsize=size, va="center", ha="left")
    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=0.06, transparent=True, dpi=260)
    plt.close(fig)
    print("  ", name)


eq("eq_governing.png", [
    r"$\mathrm{mass:}\ \ \partial U_z/\partial Z + \partial U_y/\partial Y = 0$",
    r"$\mathrm{force\ along:}\ \ \partial P/\partial Z = (1-\beta)\,\partial^2 U_z/\partial Y^2"
    r" + \partial T_{zz}/\partial Z + \partial T_{yz}/\partial Y + O(\epsilon^2)$",
    r"$\mathrm{force\ across:}\ \ \partial P/\partial Y = O(\epsilon^2)$",
    r"$\mathrm{stretch:}\ \ \mathrm{De}\,[\,\mathrm{carried} - \mathrm{stretched}\,] = -\mathbf{A},"
    r"\quad \mathbf{T} = \beta\,\mathbf{A}$",
], size=19, width=8.0)
eq("eq_series.png", [
    r"$\Delta P \approx \Delta P_0 + \mathrm{De}\,\Delta P_1 + \mathrm{De}^2 \Delta P_2 + \mathrm{De}^3 \Delta P_3$",
    r"$\ \ \ \ \ = 0.996 - 1.793\,\mathrm{De} + 5.533\,\mathrm{De}^2 - 14.661\,\mathrm{De}^3$",
], size=20, width=7.5)
eq("eq_typo.png", [
    r"$\mathrm{printed:}\ \ A_{yz,1} = +18\,H'\,Y\,(H^2 - 2Y^2)/H^7$",
    r"$\mathrm{correct:}\ \ A_{yz,1} = -18\,H'\,Y\,(H^2 - 2Y^2)/H^7$",
], size=19, width=7.0)
eq("eq_psi.png", [
    r"$\Psi = \frac{1}{2}(3\eta - \eta^3) + \eta\,(1-\eta^2)^2\,c\,N_\psi(Z, \eta^2),\quad \eta = Y/H(Z)$",
    r"$U_z = \partial\Psi/\partial Y,\qquad U_y = -\partial\Psi/\partial Z$",
    r"$P = N_P(Z) + \epsilon^2 N_{P1}(Z, \eta^2)$",
], size=19, width=8.0, line_h=0.72)
eq("eq_full.png", [
    r"$\partial U_z/\partial Z + \partial U_y/\partial Y = 0$",
    r"$\partial P/\partial Z = (1-\beta)(\epsilon^2 \partial^2_Z U_z + \partial^2_Y U_z)"
    r" + \partial_Z T_{zz} + \partial_Y T_{yz}$",
    r"$\partial P/\partial Y = \epsilon^2\,[(1-\beta)(\epsilon^2 \partial^2_Z U_y + \partial^2_Y U_y)"
    r" + \partial_Z T_{yz} + \partial_Y T_{yy}]$",
    r"$\mathrm{De}(U_z\partial_Z A_{zz} + U_y\partial_Y A_{zz} - 2\partial_Z U_z A_{zz}"
    r" - 2\partial_Y U_z A_{yz}) - 2\epsilon^2\partial_Z U_z = -A_{zz}$",
    r"$\mathrm{De}(U_z\partial_Z A_{yy} + U_y\partial_Y A_{yy} - 2\partial_Z U_y A_{yz}"
    r" - 2\partial_Y U_y A_{yy}) - 2\partial_Y U_y = -A_{yy}$",
    r"$\mathrm{De}(U_z\partial_Z A_{yz} + U_y\partial_Y A_{yz} - \partial_Z U_y A_{zz}"
    r" - \partial_Y U_z A_{yy}) - \epsilon^2\partial_Z U_y - \partial_Y U_z = -A_{yz}$",
], size=17, width=11.0, line_h=0.62)
print("done")

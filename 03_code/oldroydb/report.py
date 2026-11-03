"""Figures and tables used by the notebook, kept here so notebook cells stay short.

Every function reads saved results (04_results/) and returns a figure or prints
a table. Nothing here trains anything.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from . import analytical as an
from . import compare
from . import plotting as pl
from .geometry import ChannelGeometry

_trapezoid = getattr(np, "trapezoid", None) or np.trapz   # numpy < 2 fallback


# ------------------------------------------------------------------ section 1
def channel_sketch(alpha=4.0, eps=0.02):
    """The geometry: straight inlet, hyperbolic contraction, straight outlet."""
    g = ChannelGeometry(alpha, 0.5, 0.5, 0.01)
    Z = np.linspace(g.Z_min, g.Z_max, 600)
    H = g.H(Z)
    fig, ax = plt.subplots(figsize=(8.5, 3.0))
    ax.fill_between(Z, -H, H, color="#e6eef9")
    ax.plot(Z, H, color=pl.INK, lw=1.4)
    ax.plot(Z, -H, color=pl.INK, lw=1.4)
    ax.axhline(0, color=pl.THEORY, lw=0.8, ls=(0, (6, 3)))
    for z0, txt in [(-0.25, "straight inlet\n(flow fully developed)"),
                    (0.5, "hyperbolic contraction\nH = 4 / (3Z + 1)"),
                    (1.25, "straight outlet")]:
        ax.text(z0, 0.0, txt, ha="center", va="center", fontsize=9, color=pl.INK,
                bbox=dict(fc="white", ec="none", alpha=0.8))
    for z0 in (0.0, 1.0):
        ax.axvline(z0, color=pl.THEORY, lw=0.8, ls=":")
    ax.annotate("", xy=(-0.05, 2.7), xytext=(-0.45, 2.7),
                arrowprops=dict(arrowstyle="->", color=pl.PINN, lw=1.6))
    ax.text(-0.25, 2.95, "flow rate q fixed", ha="center", fontsize=9, color=pl.INK)
    ax.text(0.0, -5.9, "Z = 0", ha="center", fontsize=9)
    ax.text(1.0, -5.9, "Z = 1", ha="center", fontsize=9)
    ax.annotate("", xy=(1.0, -4.9), xytext=(0.0, -4.9),
                arrowprops=dict(arrowstyle="<->", color=pl.INK, lw=1))
    ax.text(0.5, -4.8, "$\\Delta P$ is measured between these points, on the centreline",
            ha="center", va="bottom", fontsize=9)
    ax.set_xlim(g.Z_min, g.Z_max)
    ax.set_ylim(-6.1, 4.2)
    ax.set_xlabel("Z = z / l   (along the channel)")
    ax.set_ylabel("Y = y / h$_l$")
    ax.grid(False)
    ax.set_title(f"The channel in scaled units. The real one is about {1/eps:.0f} times "
                 "longer than tall at the outlet.", loc="left", fontsize=10)
    return fig


# ------------------------------------------------------------------ section 2
def load_fe(root, fig="fig2"):
    """Recovered FE points: {eps: (x, y)} using the marker centroid (see digitizer)."""
    name = {"fig2": "fig2_dP_vs_De.csv", "fig7": "fig7_dP_vs_alpha.csv",
            "fig8": "fig8_dP_vs_viscosity_ratio.csv"}[fig]
    d = np.loadtxt(Path(root) / "04_results" / "digitized" / name, delimiter=",", skiprows=1)
    return {e: (d[d[:, 0] == e, 1], d[d[:, 0] == e, 3]) for e in np.unique(d[:, 0])}


def series_curves(De):
    return (De, {n: an.pressure_drop(De, n) for n in (1, 2, 3)})


def replot_paper_fig2(fe):
    """Our re-plot of the paper's figure 2 from the recovered data."""
    De = np.linspace(0, 0.3, 200)
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, e in zip(axs, sorted(fe)):
        pl.dp_vs_de(ax, theory=series_curves(De), fe=fe[e],
                    title=f"$\\epsilon$ = {e:g}  ($\\alpha$ = 4, $\\tilde\\beta$ = 0.4)")
        ax.set_ylim(0.4, 1.05)
    axs[1].set_ylabel("")
    axs[1].get_legend().remove()
    return fig


def reference_vs_fe(ref, fe, eps=0.02):
    """Delta P of our eps -> 0 solver against the FE points and the series."""
    De = np.linspace(0, 0.3, 200)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    pl.dp_vs_de(ax, theory=series_curves(De), fe=fe[eps], ref=(ref["De"], ref["dP"]),
                title=f"Reference solver vs paper (FE at $\\epsilon$ = {eps:g})")
    ax.set_ylim(0.4, 1.05)
    rows = []
    for d, p in zip(ref["De"], ref["dP"]):
        i = np.argmin(np.abs(fe[eps][0] - d))
        if abs(fe[eps][0][i] - d) < 2e-3:
            rows.append((d, p, fe[eps][1][i], 100 * (p / fe[eps][1][i] - 1),
                         an.pressure_drop(d, 3)))
    return fig, rows


def print_table(header, rows, fmts):
    """Plain aligned text table."""
    widths = [max(len(h), 9) for h in header]
    cell = lambda f, v: "-" if isinstance(v, float) and np.isnan(v) else f.format(v)
    print("  ".join(h.rjust(w) for h, w in zip(header, widths)))
    for r in rows:
        print("  ".join(cell(f, v).rjust(w) for f, v, w in zip(fmts, r, widths)))


# ------------------------------------------------------------------ section 3
def constraint_check(model, De=0.1, n=400):
    """Show that the boundary conditions hold for ANY network weights.

    The weights of the last layer are filled with random numbers first, so the
    network output is far from the solution; the conditions still hold.
    """
    import torch
    torch.manual_seed(1)
    with torch.no_grad():
        for net in model.nets.values():
            net.net[-1].weight.normal_(0, 1.0)
    g = model.geom
    rows = []
    for Z0 in (-0.3, 0.0, 0.37, 0.8, 1.0, 1.3):
        H = float(g.H(np.array(Z0)))
        eta, w = np.polynomial.legendre.leggauss(n)
        eta = 0.5 * (eta + 1)
        w = 0.5 * w
        f = model.evaluate(np.full(n + 2, Z0), np.r_[eta * H, H, 0.0], De)
        flux = float(np.sum(w * f["Uz"][:n]) * H)
        moved = float(np.max(np.abs(f["Uz"][:n] - 1.5 * (1 - eta**2) / H)))  # vs parabola
        rows.append((Z0, moved, flux, abs(f["Uz"][n]), abs(f["Uy"][n]), abs(f["Uy"][n + 1]),
                     abs(f["Ayz"][n + 1])))
    return rows


def loss_history(hist, title=""):
    """Training loss per equation: Adam steps, then L-BFGS steps."""
    it = np.array(hist["it"])
    phase = np.array(hist["phase"])
    n_adam = it[phase == "adam"].max() + 1
    x = np.where(phase == "adam", it, n_adam + it)
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    keys = list(hist["parts"][0].keys())
    colours = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"]
    for k, c in zip(keys, colours):
        ax.semilogy(x, [p[k] for p in hist["parts"]], color=c, lw=1.5, label=k)
    ax.semilogy(x, hist["loss"], color=pl.INK, lw=2.2, label="total")
    ax.axvline(n_adam, color=pl.THEORY, ls=":", lw=1)
    ax.text(n_adam, ax.get_ylim()[1], "  L-BFGS starts", va="top", fontsize=9)
    ax.set_xlabel("training step")
    ax.set_ylabel("mean squared residual")
    ax.set_title(title, loc="left")
    ax.legend(ncol=3, fontsize=8)
    return fig


# ------------------------------------------------------------------ section 4
def field_comparison(model, ref_De, De, names=("Uz", "Azz", "Ayz", "Ayy"), beta_t=0.4):
    """Rows: fields. Columns: PINN, reference, PINN minus reference."""
    labels = {"Uz": "axial velocity $U_z$", "Uy": "cross velocity $U_y$",
              "Azz": "axial polymer stress $T_{zz}$", "Ayz": "polymer shear stress $T_{yz}$",
              "Ayy": "cross polymer stress $T_{yy}$"}
    errs, f, (ZZ, EE, YY) = compare.field_errors(model, ref_De, De, Z_window=(0.0, 1.0))
    sel = (ref_De["Z"] >= 0) & (ref_De["Z"] <= 1)
    fig, axs = plt.subplots(len(names), 3, figsize=(13, 2.6 * len(names)), squeeze=False)
    for i, k in enumerate(names):
        mult = 1.0 if k in ("Uz", "Uy") else beta_t        # stress T = beta_t * A
        a = mult * f[k].reshape(ZZ.shape)
        b = mult * ref_De[k][sel]
        signed = np.nanmin(b) < -1e-9
        vmax = np.nanmax(np.abs(b))
        vlim = (-vmax, vmax) if signed else (min(np.nanmin(b), 0), vmax)
        pl.field_map(axs[i, 0], ZZ, YY, a, title=f"PINN: {labels[k]}", signed=signed, vlim=vlim)
        pl.field_map(axs[i, 1], ZZ, YY, b, title="reference solver", signed=signed, vlim=vlim)
        e = a - b
        em = np.nanmax(np.abs(e))
        pl.field_map(axs[i, 2], ZZ, YY, e, title=f"difference (rel. L2 error {abs(errs[k]):.1%})",
                     signed=True, vlim=(-em, em))
        for ax in axs[i]:
            ax.set_xlabel("" if i < len(names) - 1 else "Z")
            ax.set_ylabel("Y")
    fig.tight_layout()
    return fig, errs


# ------------------------------------------------------------------ loading runs
def load_runs(root, eps):
    """All fixed-De PINN runs for one eps: {De: (model, history)}."""
    from . import pinn
    out = {}
    for d in sorted((Path(root) / "04_results" / "pinn").glob(f"eps{eps:g}_De*")):
        if (d / "model.pt").exists():
            De = float(d.name.split("_De")[1])
            out[De] = pinn.load(d)
    return dict(sorted(out.items()))


def load_param(root, eps):
    """The De-as-input PINN for one eps, or None if it has not been trained."""
    from . import pinn
    d = Path(root) / "04_results" / "pinn" / f"eps{eps:g}_param"
    return pinn.load(d) if (d / "model.pt").exists() else None


# ------------------------------------------------------------------ section 4.2
def dp_lubrication(ref, runs, param=None):
    """Delta P vs De in the lubrication limit: PINN vs reference vs series."""
    De = np.linspace(0, 0.3, 200)
    rows = []
    for d, (m, h) in runs.items():
        dp = m.pressure_drop(d)
        r = float(np.interp(d, ref["De"], ref["dP"]))
        rows.append((d, dp["direct"], dp["momentum"], r, 100 * (dp["direct"] / r - 1),
                     100 * (dp["momentum"] / r - 1), an.pressure_drop(d, 3)))
    fig, ax = plt.subplots(figsize=(6.6, 4.1))
    pl.dp_vs_de(ax, theory=series_curves(De), ref=(ref["De"], ref["dP"]),
                pinn=([r[0] for r in rows], [r[1] for r in rows]),
                title="Lubrication limit ($\\epsilon\\to0$): PINN against the references")
    if param is not None:
        Dg = np.linspace(0, 0.3, 31)
        ax.plot(Dg, [param[0].pressure_drop(d)["direct"] for d in Dg], color=pl.PINN,
                lw=1.2, ls="--", label="PINN with De as input (one network)")
        ax.legend(loc="best")
    ax.set_ylim(0.4, 1.05)
    return fig, rows


def error_table(ref, runs):
    """Relative L2 error of every field over the contraction, for every De."""
    rows = []
    for d, (m, h) in runs.items():
        e, _, _ = compare.field_errors(m, compare.reference_at(ref, d), d)
        e_mid, _, _ = compare.field_errors(m, compare.reference_at(ref, d), d, (0.05, 0.95))
        # a negative entry marks a field that is zero in the reference (A_zz at
        # De = 0): it has no relative error, so the table shows "-"
        rel = lambda v: 100 * v if v >= 0 else float("nan")
        rows.append((d,) + tuple(rel(e[k]) for k in compare.FIELDS)
                    + (max(rel(v) for v in e_mid.values() if v >= 0),))
    return rows


def error_along_channel(runs, ref, De_list=None):
    """Local error of each field at every station Z: ||PINN - ref|| / ||ref||
    across the channel at that Z, divided by the field's largest value over Z
    (so a near-zero field at the inlet does not look huge)."""
    labels = {"Uz": "$U_z$", "Uy": "$U_y$", "Azz": "$A_{zz}$", "Ayy": "$A_{yy}$",
              "Ayz": "$A_{yz}$"}
    colours = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"]
    De_list = De_list or [d for d in runs if d > 0][-1:]
    fig, axs = plt.subplots(1, len(De_list), figsize=(6.6 * len(De_list), 3.6), squeeze=False)
    for ax, d in zip(axs[0], De_list):
        r = compare.reference_at(ref, d)
        _, f, (ZZ, EE, YY) = compare.field_errors(runs[d][0], r, d)
        sel = (r["Z"] >= 0) & (r["Z"] <= 1)
        Z = r["Z"][sel]
        for k, c in zip(compare.FIELDS, colours):
            a = f[k].reshape(ZZ.shape)
            b = r[k][sel]
            scale = np.max(np.linalg.norm(b, axis=1))
            ax.semilogy(Z, np.linalg.norm(a - b, axis=1) / scale, color=c, lw=1.5,
                        label=labels[k])
        ax.axvspan(0, 0.05, color="#f1efe8", zorder=0)
        ax.axvspan(0.95, 1.0, color="#f1efe8", zorder=0)
        ax.set_xlabel("Z (shaded: corner zones)")
        ax.set_ylabel("local relative error")
        ax.set_title(f"Where the PINN's error sits, De = {d:g}", loc="left")
        ax.legend(ncol=5, fontsize=8, loc="upper center")
        ax.set_ylim(1e-5, 1)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ section 4.4
def dp_vs_fe(fe, runs_by_eps, params=None):
    """Full 2-D PINN (finite eps) against the paper's finite elements."""
    De = np.linspace(0, 0.3, 200)
    fig, axs = plt.subplots(1, len(runs_by_eps), figsize=(5.2 * len(runs_by_eps), 4.0),
                            sharey=True, squeeze=False)
    rows = []
    for ax, (eps, runs) in zip(axs[0], runs_by_eps.items()):
        pts = []
        for d, (m, h) in runs.items():
            dp = m.pressure_drop(d)
            x, y = fe[eps]
            i = int(np.argmin(np.abs(x - d)))
            pts.append((d, dp["direct"]))
            rows.append((eps, d, dp["direct"], dp["momentum"], y[i],
                         100 * (dp["direct"] / y[i] - 1), 100 * (dp["momentum"] / y[i] - 1),
                         100 * (an.pressure_drop(d, 3) / y[i] - 1)))
        pl.dp_vs_de(ax, theory=series_curves(De), fe=fe[eps],
                    pinn=([p[0] for p in pts], [p[1] for p in pts]),
                    title=f"Full 2-D, $\\epsilon$ = {eps:g}")
        if params and params.get(eps) is not None:
            Dg = np.linspace(0, 0.3, 31)
            ax.plot(Dg, [params[eps][0].pressure_drop(d)["direct"] for d in Dg],
                    color=pl.PINN, lw=1.2, ls="--", label="PINN, De as input")
            ax.legend(loc="lower left", fontsize=8)
        ax.set_ylim(0.4, 1.06)
    for ax in axs[0][1:]:
        ax.set_ylabel("")
        if ax.get_legend():
            ax.get_legend().remove()
    return fig, rows


def stress_maps(runs, De_list=(0.1, 0.2, 0.3), beta_t=0.4):
    """PINN maps of the axial and shear polymer stress (cf. paper figs 4 and 5)."""
    fig, axs = plt.subplots(len(De_list), 2, figsize=(11, 2.5 * len(De_list)), squeeze=False)
    for i, d in enumerate(De_list):
        if d not in runs:
            continue
        m = runs[d][0]
        Z = np.linspace(0, 1, 161)
        eta = np.linspace(0, 1, 41)
        ZZ, EE = np.meshgrid(Z, eta, indexing="ij")
        YY = EE * m.geom.H(ZZ)
        f = m.evaluate(ZZ.ravel(), YY.ravel(), d)
        Tzz = beta_t * f["Azz"].reshape(ZZ.shape)
        Tyz = beta_t * f["Ayz"].reshape(ZZ.shape)
        pl.field_map(axs[i, 0], ZZ, YY, Tzz, title=f"$T_{{zz}}$, De = {d:g}  (max {Tzz.max():.2f})")
        pl.field_map(axs[i, 1], ZZ, YY, Tyz, signed=True,
                     title=f"$T_{{yz}}$, De = {d:g}  (min {Tyz.min():.2f})")
    fig.tight_layout()
    return fig


def stress_peaks(runs, ref, beta_t=0.4):
    """Largest axial stress and most negative shear stress in 0 <= Z <= 1:
    PINN (this eps) next to the reference solver (eps -> 0)."""
    rows = []
    for d, (m, h) in runs.items():
        if d == 0:
            continue
        Z = np.linspace(0, 1, 201)
        eta = np.linspace(0, 1, 41)
        ZZ, EE = np.meshgrid(Z, eta, indexing="ij")
        f = m.evaluate(ZZ.ravel(), (EE * m.geom.H(ZZ)).ravel(), d)
        r = compare.reference_at(ref, d)
        sel = r["Z"] <= 1.0
        rows.append((d, beta_t * f["Azz"].max(), beta_t * r["Azz"][sel].max(),
                     beta_t * f["Ayz"].min(), beta_t * r["Ayz"][sel].min()))
    return rows


# ------------------------------------------------------------------ section 4.5
def contributions_reference(ref, beta_t=0.4):
    """The four terms of eq. (5.3) from the reference solution (eps -> 0)."""
    from .lubrication_fd import diff_matrix
    D = diff_matrix(ref["eta"])
    D2 = D @ D
    Z, H = ref["Z"], ref["H"]
    sel = (Z >= 0) & (Z <= 1)
    out = []
    for i, d in enumerate(ref["De"]):
        Uz_YY = (ref["Uz"][i] @ D2.T)[:, 0] / H**2
        Ayz_Y = (ref["Ayz"][i] @ D.T)[:, 0] / H
        c2 = -_trapezoid(Uz_YY[sel], Z[sel])
        c3 = beta_t * (np.interp(0, Z, ref["Azz"][i][:, 0]) - np.interp(1, Z, ref["Azz"][i][:, 0]))
        c4 = -beta_t * _trapezoid((Ayz_Y - Uz_YY)[sel], Z[sel])
        out.append((d, 0.0, c2, c3, c4))
    return np.array(out)


def contributions_theory(De, beta_t=0.4, alpha=4.0):
    """Terms 2 and 4 of eq. (5.3) from the paper's asymptotic solution.

    Term 2 = int_1^0 d2Uz/dY2 at Y = 0, with Uz = Uz0 + De^2 Uz2 (eqs 3.6, 3.20).
    On the centreline: d2Uz0/dY2 = -3/H^3 and, writing Uz2 = (Y^2 - H^2) g(Y),
    d2Uz2/dY2 = 2 g(0) - H^2 g''(0). Term 4 = Delta P (series) - term 2.
    """
    from scipy.integrate import quad
    from .geometry import hyperbolic_H, hyperbolic_dH, hyperbolic_d2H
    b = beta_t

    def uzz2(z):
        H, H1, H2 = hyperbolic_H(z, alpha), hyperbolic_dH(z, alpha), hyperbolic_d2H(z, alpha)
        G2 = -324 / 35 * b * (14 * H1**2 / H**7 - 3 * H2 / H**6)
        F0 = -50 * H**3 * H1**2 + 11 * H**4 * H2
        F2 = 20 * H * H1**2 - 8 * H**2 * H2
        g0 = 0.5 * G2 - 27 * b / 20 * F0 / H**10
        g2 = -27 * b / 20 * F2 / H**10
        return 2 * g0 - H**2 * g2

    c2_0 = quad(lambda z: -3 / hyperbolic_H(z, alpha) ** 3, 0, 1)[0]
    c2_2 = quad(uzz2, 0, 1)[0]
    c2 = -(c2_0 + De**2 * c2_2)
    return c2, an.pressure_drop(De, 2) - c2, an.pressure_drop(De, 3) - c2


def contributions_figure(ref, runs=None, beta_t=0.4):
    De = np.linspace(0, 0.3, 100)
    c2, c4_2, c4_3 = contributions_theory(De, beta_t)
    cr = contributions_reference(ref, beta_t)
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.plot(De, c2, color=pl.THEORY, lw=1.4, ls="-.", label="② paper formula, O(De²)")
    ax.plot(De, c4_3, color=pl.THEORY, lw=1.4, ls="--", label="④ paper formula, O(De³)")
    ax.plot(cr[:, 0], cr[:, 2], color=pl.REF, lw=2, label="② reference solver")
    ax.plot(cr[:, 0], cr[:, 4], color=pl.REF, lw=2, ls=(0, (4, 2)), label="④ reference solver")
    ax.plot(cr[:, 0], cr[:, 3], color=pl.REF, lw=1, ls=":", label="③ reference solver")
    if runs:
        pts = np.array([[d] + m.pressure_drop(d)["terms"] for d, (m, h) in runs.items()])
        ax.plot(pts[:, 0], pts[:, 2], "o", color=pl.PINN, mec="white", label="PINN ②")
        ax.plot(pts[:, 0], pts[:, 4], "s", color=pl.PINN, mec="white", label="PINN ④")
        ax.plot(pts[:, 0], pts[:, 3], "x", color=pl.PINN, label="PINN ③")
    ax.axhline(0, color=pl.GRID, lw=1)
    ax.set_xlabel("Deborah number  De")
    ax.set_ylabel("contribution to $\\Delta P$")
    ax.set_title("Where the pressure drop comes from (eq. 5.3, centreline)", loc="left")
    ax.legend(ncol=2, fontsize=8, loc="center left")
    return fig


# ------------------------------------------------------------------ section 4.6
def explorer(param, beta_t=0.4, interactive=True):
    """Static overview (always), then a De slider when ipywidgets is available.

    The static part keeps the HTML copy useful, since sliders need a live kernel.
    """
    m = param[0]
    Z = np.linspace(-0.2, 1.3, 151)
    eta = np.linspace(0, 1, 31)
    ZZ, EE = np.meshgrid(Z, eta, indexing="ij")
    YY = EE * m.geom.H(ZZ)
    Dg = np.linspace(0, 0.3, 31)
    dps = [m.pressure_drop(d)["direct"] for d in Dg]

    def stress(De):
        f = m.evaluate(ZZ.ravel(), YY.ravel(), De)
        return beta_t * f["Azz"].reshape(ZZ.shape), beta_t * f["Ayz"].reshape(ZZ.shape)

    # ---- static overview: T_zz at three De + the whole Delta P curve
    fig, axs = plt.subplots(1, 4, figsize=(16, 3.0),
                            gridspec_kw={"width_ratios": [1, 1, 1, 0.75]})
    for ax, d in zip(axs[:3], (0.0, 0.15, 0.3)):
        pl.field_map(ax, ZZ, YY, stress(d)[0], title=f"$T_{{zz}}$, De = {d:g}",
                     vlim=(0, 2.1), cbar=(d == 0.3))
    axs[3].plot(Dg, dps, color=pl.PINN)
    axs[3].set_title("$\\Delta P$ from one network", loc="left")
    axs[3].set_xlabel("De")
    fig.tight_layout()
    plt.show()

    def draw(De=0.2):
        Tzz, Tyz = stress(De)
        fig, axs = plt.subplots(1, 3, figsize=(15, 3.2),
                                gridspec_kw={"width_ratios": [1.2, 1.2, 0.8]})
        pl.field_map(axs[0], ZZ, YY, Tzz, title=f"$T_{{zz}}$ at De = {De:.3f}", vlim=(0, 2.1))
        pl.field_map(axs[1], ZZ, YY, Tyz, signed=True, title="$T_{yz}$", vlim=(-2.7, 2.7))
        axs[2].plot(Dg, dps, color=pl.PINN)
        dp = m.pressure_drop(De)["direct"]
        axs[2].plot([De], [dp], "o", color=pl.PINN, ms=9, mec="white")
        axs[2].set_title(f"$\\Delta P$ = {dp:.4f}", loc="left")
        axs[2].set_xlabel("De")
        fig.tight_layout()
        plt.show()

    if not interactive:
        return
    try:
        import ipywidgets as w
    except ImportError:
        print("Install ipywidgets for the De slider.")
        return
    w.interact(draw, De=w.FloatSlider(value=0.2, min=0.0, max=0.3, step=0.01,
                                      continuous_update=False, description="De"))

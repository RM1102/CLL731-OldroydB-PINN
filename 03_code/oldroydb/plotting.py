"""One visual language for every figure in the project.

The same colour always means the same method:
  PINN (this project)          blue    #2a78d6   circles / solid line
  reference solver (eps -> 0)  orange  #eb6834   solid line
  paper's finite elements      aqua    #1baf7a   triangles
  paper's formulas O(De^n)     grey    #52514e   dotted / dash-dot / dashed
Field maps use one-hue sequential maps for sizes and a red-blue map with a
neutral middle for signed quantities and errors.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

PINN = "#2a78d6"
REF = "#eb6834"
FE = "#1baf7a"
THEORY = "#52514e"
INK = "#0b0b0b"
GRID = "#d8d7d2"
ORDER_STYLE = {1: (0, (1, 1.5)), 2: "-.", 3: "--"}       # series order -> line style

SEQ = "Blues"
DIV = "RdBu_r"


def style():
    """Matplotlib defaults: quiet axes, light grid, readable fonts."""
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "axes.edgecolor": "#8a8984", "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "xtick.color": "#52514e", "ytick.color": "#52514e",
        "legend.frameon": False, "legend.fontsize": 9,
        "lines.linewidth": 2.0, "lines.markersize": 6,
    })


def dp_vs_de(ax, *, theory=None, ref=None, fe=None, pinn=None, title=None):
    """Delta P against De with every available method.

    theory : (De array, {order: Delta P array})
    ref    : (De array, Delta P array)            reference solver
    fe     : (De array, Delta P array)            paper's finite elements
    pinn   : (De array, Delta P array)            PINN points
    """
    if theory is not None:
        De, by_order = theory
        for n, dp in by_order.items():
            ax.plot(De, dp, color=THEORY, lw=1.4, ls=ORDER_STYLE[n],
                    label=f"paper formula, O(De$^{n}$)")
    if fe is not None:
        ax.plot(fe[0], fe[1], "^", color=FE, ms=6, mec="white", mew=0.6,
                label="paper finite elements")
    if ref is not None:
        ax.plot(ref[0], ref[1], "-", color=REF, lw=2, label="our reference solver ($\\epsilon\\to0$)")
    if pinn is not None:
        ax.plot(pinn[0], pinn[1], "o", color=PINN, ms=7, mec="white", mew=0.8,
                label="PINN", zorder=5)
    ax.set_xlabel("Deborah number  De")
    ax.set_ylabel("pressure drop  $\\Delta P$")
    if title:
        ax.set_title(title, loc="left")
    ax.legend(loc="best")


def field_map(ax, Z, Y, F, *, title="", signed=False, levels=21, cbar=True, vlim=None):
    """Filled contour of a field over the channel (Z along, Y across)."""
    if vlim is None:
        vmax = np.nanmax(np.abs(F)) if signed else np.nanmax(F)
        vmin = -vmax if signed else np.nanmin(F)
    else:
        vmin, vmax = vlim
    cs = ax.contourf(Z, Y, F, levels=np.linspace(vmin, vmax, levels),
                     cmap=DIV if signed else SEQ, extend="both")
    ax.plot(Z[:, -1], Y[:, -1], color=INK, lw=1.0)               # the wall
    ax.set_title(title, loc="left")
    ax.set_xlabel("Z = z / l")
    ax.set_ylabel("Y = y / h$_l$")
    ax.grid(False)
    if cbar:
        span = max(abs(vmin), abs(vmax), 1e-30)
        tick = mpl.ticker.FuncFormatter(                    # show round-off as a clean 0
            lambda x, _: "0" if abs(x) < 1e-6 * span else f"{x:.2g}")
        plt.colorbar(cs, ax=ax, shrink=0.9, pad=0.02, format=tick)
    return cs

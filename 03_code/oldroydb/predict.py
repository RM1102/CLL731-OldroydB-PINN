"""Use the trained PINNs to predict the flow. The code behind predict.py and gui.py.

A trained PINN is a formula: give it a position (Z, Y) and a Deborah number De
and it returns the velocity, pressure and polymer stress there, in a fraction
of a second. Nothing is solved again.

Cases (all with alpha = 4 and beta_t = 0.4, as in the paper):
  eps0         narrow-channel limit (eps -> 0), trained at De = 0, 0.05, ..., 0.3
  eps0.02      full 2-D, eps = 0.02 (paper's case I), trained at De = 0, 0.1, 0.2, 0.3
  eps0.1       full 2-D, eps = 0.1  (paper's case II), trained at De = 0, 0.1, 0.2, 0.3
  eps0.02-any  full 2-D, eps = 0.02, ONE network for any De between 0 and 0.3

Units are the paper's scaled units (eq. 2.10): Z = z / l runs along the channel
(0 to 1 across the contraction), Y = y / h_l across it (0 on the centreline,
H(Z) = 4 / (3Z + 1) on the wall). Stresses are T = beta_t * A, scaled as in the
paper's figures 4 and 5.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import pinn

ROOT = Path(__file__).resolve().parents[2]
PINN_DIR = ROOT / "04_results" / "pinn"

CASES = {
    "eps0": dict(label="narrow-channel limit (eps -> 0)", eps=0.0, run="eps0_De"),
    "eps0.02": dict(label="full 2-D, eps = 0.02", eps=0.02, run="eps0.02_De"),
    "eps0.1": dict(label="full 2-D, eps = 0.1", eps=0.1, run="eps0.1_De"),
    "eps0.02-any": dict(label="eps = 0.02, any De (one network)", eps=0.02, run="eps0.02_param"),
}
FIELD_NAMES = {"Uz": "axial velocity U_z", "Uy": "cross velocity U_y", "P": "pressure P",
               "Tzz": "axial polymer stress T_zz", "Tyz": "polymer shear stress T_yz",
               "Tyy": "cross polymer stress T_yy"}
# The paper's pressure scale p_c = eta_0 u_c l / h_l^2 (its table 2), used to quote
# a scaled pressure drop in pascals for the paper's two physical channels.
PRESSURE_SCALE_PA = {0.0: 2500.0, 0.02: 2500.0, 0.1: 100.0}

_cache: dict = {}


def trained_De(case):
    """The De values a case was trained at (None for the any-De network)."""
    if case.endswith("-any"):
        return None
    out = []
    for d in PINN_DIR.glob(CASES[case]["run"] + "*"):
        if (d / "model.pt").exists():
            out.append(float(d.name.split("_De")[1]))
    return sorted(out)


def available_cases():
    """Cases whose networks are present on disk."""
    ok = []
    for c in CASES:
        if c.endswith("-any"):
            if (PINN_DIR / CASES[c]["run"] / "model.pt").exists():
                ok.append(c)
        elif trained_De(c):
            ok.append(c)
    return ok


def load(case, De):
    """The network for a case and De. Returns (model, De used, note).

    Fixed-De cases use the nearest trained De (the note says so); the any-De
    network accepts any De in [0, 0.3].
    """
    if case not in CASES:
        raise ValueError(f"unknown case {case!r}; choose from {list(CASES)}")
    note = ""
    if case.endswith("-any"):
        De_used = float(np.clip(De, 0.0, 0.3))
        run = PINN_DIR / CASES[case]["run"]
        if De_used != De:
            note = f"De limited to the trained range: {De_used:g}"
    else:
        Des = trained_De(case)
        if not Des:
            raise FileNotFoundError(f"no trained networks for {case} in {PINN_DIR}")
        De_used = min(Des, key=lambda d: abs(d - De))
        if abs(De_used - De) > 1e-9:
            note = f"nearest trained De is {De_used:g} (asked for {De:g})"
        run = PINN_DIR / f"{CASES[case]['run']}{De_used:g}"
    if run not in _cache:
        _cache[run] = pinn.load(run)[0]
    return _cache[run], De_used, note


def fields(model, De, Z, Y):
    """Velocity, pressure and polymer stress at points (Z, Y).

    Points outside the channel come back as NaN. Pressure is given relative to
    its value on the centreline at the end of the contraction, P(Z=1, Y=0) = 0.
    """
    Z = np.atleast_1d(np.asarray(Z, float))
    Y = np.atleast_1d(np.asarray(Y, float))
    Z, Y = np.broadcast_arrays(Z, Y)
    g, b = model.geom, model.cfg.beta_t
    inside = (Z >= g.Z_min) & (Z <= g.Z_max) & (Y >= 0) & (Y <= g.H(np.clip(Z, g.Z_min, g.Z_max)))
    out = {k: np.full(Z.shape, np.nan) for k in FIELD_NAMES}
    if inside.any():
        f = model.evaluate(Z[inside], Y[inside], De)
        P0 = model.evaluate(np.array([1.0]), np.array([0.0]), De)["P"][0]
        vals = {"Uz": f["Uz"], "Uy": f["Uy"], "P": f["P"] - P0, "Tzz": b * f["Azz"],
                "Tyz": b * f["Ayz"], "Tyy": b * f["Ayy"]}
        for k, v in vals.items():
            out[k][inside] = v
    out["inside"] = inside
    return out


def pressure_drop(model, De):
    """Centreline pressure drop between Z = 0 and Z = 1, two ways (see notebook 4.2)."""
    dp = model.pressure_drop(De)
    return dp["direct"], dp["momentum"]


def grid(model, De, nz=121, ny=31, Z_range=(-0.2, 1.3)):
    """Fields on a grid that follows the channel wall. Returns (Z, Y, fields)."""
    Zs = np.linspace(*Z_range, nz)
    eta = np.linspace(0, 1, ny)
    ZZ, EE = np.meshgrid(Zs, eta, indexing="ij")
    YY = EE * model.geom.H(ZZ)
    f = fields(model, De, ZZ.ravel(), YY.ravel())
    return ZZ, YY, {k: v.reshape(ZZ.shape) for k, v in f.items() if k != "inside"}

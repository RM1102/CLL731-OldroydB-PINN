"""Compare a trained PINN with the reference solver and the analytical series."""
from __future__ import annotations

from pathlib import Path

import numpy as np

FIELDS = ["Uz", "Uy", "Azz", "Ayy", "Ayz"]


def load_reference(path):
    """Reference (eps -> 0) solutions saved by scripts/run_reference_solver.py."""
    d = np.load(path)
    return {k: d[k] for k in d.files}


def reference_at(ref, De):
    """Fields of the reference solution for one De (must be in the sweep)."""
    i = int(np.argmin(np.abs(ref["De"] - De)))
    if abs(ref["De"][i] - De) > 1e-9:
        raise ValueError(f"De={De} not in reference sweep {ref['De']}")
    out = {k: ref[k][i] for k in ["Uz", "Uy", "Azz", "Ayy", "Ayz", "P", "G"]}
    out.update(Z=ref["Z"], eta=ref["eta"], H=ref["H"], dP=float(ref["dP"][i]))
    return out


def field_errors(model, ref_De, De, Z_window=(0.0, 1.0)):
    """Relative L2 error of each field over the contraction, on the reference grid.

    Error of field f = ||f_PINN - f_ref|| / ||f_ref||, with the norm taken over
    all reference grid points with Z inside Z_window (trapezoid-free: the grid
    is the same for both, so a plain sum is a fair weighting).
    """
    Z, eta, H = ref_De["Z"], ref_De["eta"], ref_De["H"]
    sel = (Z >= Z_window[0]) & (Z <= Z_window[1])
    ZZ, EE = np.meshgrid(Z[sel], eta, indexing="ij")
    YY = EE * H[sel][:, None]
    f = model.evaluate(ZZ.ravel(), YY.ravel(), De)
    errs = {}
    for k in FIELDS:
        a = f[k].reshape(ZZ.shape)
        b = ref_De[k][sel]
        nb = np.linalg.norm(b)
        # a field that is zero in the reference (A_zz at De = 0) has no relative
        # error; report its RMS absolute error instead, marked with a minus sign
        errs[k] = (float(np.linalg.norm(a - b) / nb) if nb > 1e-8 * b.size
                   else -float(np.sqrt(np.mean((a - b) ** 2))))
    return errs, f, (ZZ, EE, YY)

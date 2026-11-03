"""Channel shape H(Z) for the hyperbolic contraction of Boyko & Stone (2022).

All lengths are in the paper's lubrication scaling (eq. 2.10):
    Z = z / l      (0 at the start of the contraction, 1 at its end)
    Y = y / h_l    (0 on the centreline, H(Z) on the wall)
so the outlet half-height is H = 1 and the inlet half-height is H = alpha.

Two shapes are provided:
  * `hyperbolic_H`   : the exact shape of eq. (5.1), defined on 0 <= Z <= 1.
  * `ChannelGeometry`: straight inlet pipe + hyperbolic contraction + straight
                       outlet pipe (the finite-element set-up of the paper's
                       figure 9). The two corners are rounded over a small
                       length `delta` so that H, H' and H'' are continuous,
                       which the PINN and the marching solver both need.
"""
from __future__ import annotations

import numpy as np

try:
    import torch
except ImportError:          # the analytical part works without torch
    torch = None


# ---------------------------------------------------------------- exact shape
def hyperbolic_H(Z, alpha=4.0):
    """Eq. (5.1): H(Z) = alpha / ((alpha - 1) Z + 1)."""
    return alpha / ((alpha - 1.0) * Z + 1.0)


def hyperbolic_dH(Z, alpha=4.0):
    """First derivative dH/dZ of eq. (5.1)."""
    return -alpha * (alpha - 1.0) / ((alpha - 1.0) * Z + 1.0) ** 2


def hyperbolic_d2H(Z, alpha=4.0):
    """Second derivative d2H/dZ2 of eq. (5.1)."""
    return 2.0 * alpha * (alpha - 1.0) ** 2 / ((alpha - 1.0) * Z + 1.0) ** 3


# ------------------------------------------------- smooth ramp used at corners
def _ramp(x, delta, xp):
    """C2 smooth version of max(x, 0).

    Equal to 0 for x <= -delta and to x for x >= +delta. In between it is the
    quartic (x + delta)^3 (3 delta - x) / (16 delta^3), which matches value,
    slope and curvature at both ends. Returns (f, f', f'').
    """
    if delta <= 0:
        f = xp.where(x > 0, x, 0.0 * x)
        return f, xp.where(x > 0, 1.0 + 0.0 * x, 0.0 * x), 0.0 * x
    inner = (xp.abs(x) < delta)
    xi = xp.clip(x, -delta, delta) if xp is np else torch.clamp(x, -delta, delta)
    a = xi + delta
    f_in = a ** 3 * (3 * delta - xi) / (16 * delta ** 3)
    d1_in = (3 * a ** 2 * (3 * delta - xi) - a ** 3) / (16 * delta ** 3)
    d2_in = (6 * a * (3 * delta - xi) - 6 * a ** 2) / (16 * delta ** 3)
    pos = x >= delta
    f = xp.where(inner, f_in, xp.where(pos, x, 0.0 * x))
    d1 = xp.where(inner, d1_in, xp.where(pos, 1.0 + 0.0 * x, 0.0 * x))
    d2 = xp.where(inner, d2_in, 0.0 * x)
    return f, d1, d2


class ChannelGeometry:
    """Straight inlet + hyperbolic contraction + straight outlet.

    Parameters
    ----------
    alpha : inlet-to-outlet height ratio h0 / h_l (paper uses 4).
    L_in, L_out : lengths of the straight inlet and outlet sections (in units
        of l). The paper's simulations use 1; 0.5 is plenty here (see plan).
    delta : half-width of the corner rounding (0 = sharp corners).
    """

    def __init__(self, alpha=4.0, L_in=0.5, L_out=0.5, delta=0.01):
        self.alpha = float(alpha)
        self.L_in = float(L_in)
        self.L_out = float(L_out)
        self.delta = float(delta)
        self.Z_min = -self.L_in
        self.Z_max = 1.0 + self.L_out

    # The contraction is the exact hyperbola evaluated at a "clamped" Z:
    # Zc = 1 - ramp(1 - ramp(Z)), which is Z inside (0, 1), 0 upstream and 1
    # downstream, with rounded corners of half-width delta.
    def _clamped(self, Z, xp):
        r1, r1p, r1pp = _ramp(Z, self.delta, xp)
        r2, r2p, r2pp = _ramp(1.0 - r1, self.delta, xp)
        Zc = 1.0 - r2
        Zc_p = r2p * r1p
        Zc_pp = -r2pp * r1p ** 2 + r2p * r1pp
        return Zc, Zc_p, Zc_pp

    def H_all(self, Z):
        """Return H, H', H'' at Z (numpy array or torch tensor)."""
        xp = np if isinstance(Z, (np.ndarray, float, int)) else torch
        Z = np.asarray(Z, dtype=float) if xp is np else Z
        Zc, Zc_p, Zc_pp = self._clamped(Z, xp)
        a = self.alpha
        H = a / ((a - 1.0) * Zc + 1.0)
        dH_dZc = -(a - 1.0) * H ** 2 / a
        d2H_dZc2 = 2.0 * (a - 1.0) ** 2 * H ** 3 / a ** 2
        H1 = dH_dZc * Zc_p
        H2 = d2H_dZc2 * Zc_p ** 2 + dH_dZc * Zc_pp
        return H, H1, H2

    def H(self, Z):
        return self.H_all(Z)[0]

    def describe(self):
        return (f"alpha={self.alpha}, straight inlet {self.L_in}, straight outlet "
                f"{self.L_out}, corner rounding delta={self.delta}")

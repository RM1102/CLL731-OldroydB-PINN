"""Closed-form low-De results of Boyko & Stone (2022), checked with SymPy.

Every formula here was re-derived symbolically by
`scripts/derive_perturbation_sympy.py` (log: 04_results/analytical/sympy_check.txt).
One sign differs from arXiv v1: eq. (3.16b) for A_yz,1 must carry -18 H',
not +18 H'. The corrected sign is the one that reproduces the paper's figure 5.

Notation (paper's lubrication scaling, eqs 2.10-2.13):
  Z, Y      : axial and cross-channel coordinates, wall at Y = H(Z)
  Uz, Uy    : velocities;  P : pressure
  Azz, Ayy, Ayz : scaled conformation tensor ("A tilde" in the paper).
                The scaled polymer stress is  T = beta_t * A.
  beta_t    : eta_p / eta_0 (polymer share of the viscosity), 0.4 in the paper
  De        : Deborah number  lambda * u_c / l
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import quad

from .geometry import hyperbolic_H, hyperbolic_dH, hyperbolic_d2H


# ------------------------------------------------- pressure drop, eqs (5.2a-d)
def dP_terms_hyperbolic(alpha=4.0, beta_t=0.4):
    """Coefficients DeltaP_0..DeltaP_3 for the hyperbolic channel, eq. (5.2)."""
    a, b = alpha, beta_t
    dP0 = 0.75 * (1 + a) * (1 + a**2) / a**3
    dP1 = 4.5 * b * (1 - a**4) / a**4
    dP2 = 648 / 35 * b * (1 + a) * (1 + a**2) * (1 - a)**2 / a**5
    dP3 = 216 / 35 * b * (11 - b) * (1 + a) * (1 + a**2) * (1 - a)**3 / a**6
    return np.array([dP0, dP1, dP2, dP3])


def dP_terms_general(H, dH, d2H, beta_t=0.4):
    """Coefficients for any shape H(Z) on 0 <= Z <= 1: eqs (3.9), (3.15),
    (3.22) and (4.14). H, dH, d2H are callables."""
    b = beta_t
    dP0 = 3 * quad(lambda z: H(z) ** -3, 0, 1, limit=200)[0]
    dP1 = 4.5 * b * (H(0) ** -4 - H(1) ** -4)
    dP2 = 324 / 35 * b * quad(lambda z: 14 * dH(z) ** 2 / H(z) ** 7
                              - 3 * d2H(z) / H(z) ** 6, 0, 1, limit=200)[0]
    dP3 = (648 * b * (9 - b) / 35 * (dH(0) ** 2 / H(0) ** 8 - dH(1) ** 2 / H(1) ** 8)
           - 216 * b * (8 - b) / 35 * (d2H(0) / H(0) ** 7 - d2H(1) / H(1) ** 7))
    return np.array([dP0, dP1, dP2, dP3])


def pressure_drop(De, order=3, alpha=4.0, beta_t=0.4):
    """Truncated series DeltaP = sum_{n<=order} De^n DeltaP_n (eq. 4.15)."""
    c = dP_terms_hyperbolic(alpha, beta_t)
    De = np.asarray(De, dtype=float)
    return sum(c[n] * De**n for n in range(order + 1))


# --------------------------------------------------------- fields, section 3
def fields(Y, Z, De, alpha=4.0, beta_t=0.4, order=1):
    """Asymptotic velocity and conformation fields in the hyperbolic channel.

    order=0 : Newtonian-like leading order (eqs 3.6-3.8)
    order=1 : adds the O(De) conformation corrections (eq. 3.16, sign-fixed)
    Velocity has no O(De) term (eq. 3.13); `order>=2` adds Uz2 (eq. 3.20).
    Returns a dict of arrays with the same shape as Y and Z.
    """
    H = hyperbolic_H(Z, alpha)
    Hp = hyperbolic_dH(Z, alpha)
    Hpp = hyperbolic_d2H(Z, alpha)
    b = beta_t

    Uz = 1.5 * (H**2 - Y**2) / H**3
    Uy = 1.5 * Hp * Y * (H**2 - Y**2) / H**4
    Azz = 0.0 * Y
    Ayy = 3 * Hp * (H**2 - 3 * Y**2) / H**4
    Ayz = -3 * Y / H**3
    dPdZ = -3 / H**3

    if order >= 1:
        Azz = Azz + De * 18 * Y**2 / H**6
        Ayz = Ayz - De * 18 * Hp * Y * (H**2 - 2 * Y**2) / H**7
        Ayy = Ayy + De * 4.5 * (4 * (H**2 - 2 * Y**2) ** 2 * Hp**2
                                - H * Hpp * (Y**2 - H**2) ** 2) / H**8
        dPdZ = dPdZ + De * 4.5 * b * (-4 * Hp / H**5)
    if order >= 2:
        G2 = -324 / 35 * b * (14 * Hp**2 / H**7 - 3 * Hpp / H**6)
        Uz2 = (Y**2 - H**2) * (0.5 * G2 - 27 * b / 20 * (
            10 * H * Hp**2 * (Y**2 - 5 * H**2)
            + Hpp * (Y**4 - 4 * (Y * H) ** 2 + 11 * H**4)) / H**10)
        Uz = Uz + De**2 * Uz2
        dPdZ = dPdZ + De**2 * G2

    return {"Uz": Uz, "Uy": Uy, "Azz": Azz, "Ayy": Ayy, "Ayz": Ayz,
            "Tzz": b * Azz, "Tyy": b * Ayy, "Tyz": b * Ayz, "dPdZ": dPdZ, "H": H}


def fully_developed(eta, H, De):
    """Exact Oldroyd-B solution in a straight channel of half-height H.

    Used as the inflow condition (paper, appendix A: fully developed
    Poiseuille velocity and the matching polymer stress at the entrance).
    eta = Y / H.  Returns Uz, Azz, Ayy, Ayz, dP/dZ.
    """
    gamma = -3 * eta / H**2                      # shear rate dUz/dY
    return {"Uz": 1.5 * (1 - eta**2) / H, "Azz": 2 * De * gamma**2,
            "Ayy": 0.0 * eta, "Ayz": gamma, "dPdZ": -3 / H**3 + 0.0 * eta}

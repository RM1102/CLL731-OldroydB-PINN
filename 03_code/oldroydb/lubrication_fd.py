"""Global finite-difference / spectral solver for the lubrication-limit
Oldroyd-B equations (eps -> 0).

Why this exists
---------------
The paper's 2-D finite-element data are not public, and its analytical
solution is valid only for small De. This solver gives an independent
numerical reference at any De, so the PINN can be checked field by field.

What it solves
--------------
Eqs (2.14a-f) of Boyko & Stone (2022) with eps = 0, written with the scaled
conformation A ("A tilde") and polymer stress T = beta_t * A:

  dUz/dZ + dUy/dY = 0
  dP/dZ  = (1 - beta_t) d2Uz/dY2 + beta_t (dAzz/dZ + dAyz/dY),   dP/dY = 0
  De (u.grad Azz - 2 dUz/dZ Azz - 2 dUz/dY Ayz)                 = -Azz
  De (u.grad Ayy - 2 dUy/dZ Ayz - 2 dUy/dY Ayy) - 2 dUy/dY      = -Ayy
  De (u.grad Ayz -   dUy/dZ Azz -   dUz/dY Ayy) -   dUz/dY      = -Ayz

with Uz = Uy = 0 on the wall, symmetry on the centreline, unit flux, fully
developed flow and stress at the inlet, and a zero-gradient outflow.

Why a global solve and not marching
-----------------------------------
It is tempting to march downstream station by station, since the stresses are
carried by the flow. That fails for De > 0: the axial normal-stress gradient
dAzz/dZ changes the velocity, and the velocity gradient dUz/dZ changes Azz.
Together they act like a diffusion term along Z, so the problem needs a
condition at the outlet as well as the inlet. Marching then blows up, and it
blows up sooner on finer grids, which is the signature of an ill-posed
initial-value problem. We therefore solve all stations at once with Newton's
method (as a boundary-value problem). A PINN does the same thing by design.

Discretisation
--------------
  * across the channel: mapped coordinate eta = Y / H(Z) on Chebyshev points
    (spectral accuracy, points cluster at the wall);
  * along the channel: stress advection uses 2nd-order upwind (backward)
    differences; velocity derivatives and the momentum force use 2nd-order
    central differences (the "diffusion-like" coupling);
  * Uy comes from the stream function Psi = H * int_0^eta Uz d eta, so mass is
    conserved exactly and the flux is 1 at every station;
  * Newton's method with a sparse Jacobian built by coloured finite
    differences, and continuation in De (0 -> target).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla
from numpy.polynomial import chebyshev as C

from .analytical import fully_developed
from .geometry import ChannelGeometry


# ----------------------------------------------------- Chebyshev machinery
def chebyshev_eta(N):
    """N+1 Chebyshev-Lobatto points on eta in [0, 1], ascending."""
    return 0.5 * (1.0 - np.cos(np.pi * np.arange(N + 1) / N))


def diff_matrix(x):
    """Barycentric differentiation matrix on arbitrary distinct nodes x."""
    dx = x[:, None] - x[None, :]
    np.fill_diagonal(dx, 1.0)
    w = 1.0 / np.prod(dx, axis=1)                  # barycentric weights
    D = (w[None, :] / w[:, None]) / dx
    np.fill_diagonal(D, 0.0)
    np.fill_diagonal(D, -D.sum(axis=1))
    return D


def integration_matrix(eta):
    """Matrix J with (J f)_j = int_0^{eta_j} f d eta for the interpolant of f."""
    n = len(eta)
    x = 2.0 * eta - 1.0                            # map to [-1, 1]
    J = np.zeros((n, n))
    for k in range(n):
        e = np.zeros(n)
        e[k] = 1.0
        c = C.chebfit(x, e, n - 1)                 # exact interpolation
        ci = C.chebint(c, lbnd=-1.0) * 0.5         # d eta = dx / 2
        J[:, k] = C.chebval(x, ci)
    return J


# --------------------------------------------- finite differences along Z
def central_matrix(Z):
    """Sparse 2nd-order first-derivative matrix (one-sided at both ends)."""
    M = len(Z)
    rows, cols, vals = [], [], []
    for i in range(M):
        if i == 0:
            idx = [0, 1, 2]
        elif i == M - 1:
            idx = [M - 3, M - 2, M - 1]
        else:
            idx = [i - 1, i, i + 1]
        w = _fd_weights(Z[idx], Z[i])
        rows += [i] * 3
        cols += idx
        vals += list(w)
    return sps.csr_matrix((vals, (rows, cols)), shape=(M, M))


def upwind_matrix(Z):
    """Sparse 2nd-order backward first-derivative matrix (row 0 left empty:
    the inlet values are prescribed)."""
    M = len(Z)
    rows, cols, vals = [], [], []
    for i in range(1, M):
        idx = [i - 1, i] if i == 1 else [i - 2, i - 1, i]
        w = _fd_weights(Z[idx], Z[i])
        rows += [i] * len(idx)
        cols += idx
        vals += list(w)
    return sps.csr_matrix((vals, (rows, cols)), shape=(M, M))


def _fd_weights(zs, z0):
    """Weights of the first derivative at z0 from values at points zs."""
    zs = np.asarray(zs, float)
    n = len(zs)
    V = np.vander(zs - z0, n, increasing=True).T   # row k: (zs - z0)^k
    rhs = np.zeros(n)
    rhs[1] = 1.0
    return np.linalg.solve(V, rhs)


def station_grid(geom, dZ_fine=1.5e-3, dZ_coarse=5e-3, Z_start=-0.06, Z_end=None):
    """Z stations: fine near the two rounded corners, coarser elsewhere."""
    d = max(geom.delta, 2e-3)
    Z_end = geom.Z_max if Z_end is None else Z_end
    w = max(4 * d, 0.03)                           # refined band around corners
    pieces = [np.arange(Z_start, -w, dZ_coarse),
              np.arange(-w, w, dZ_fine),
              np.arange(w, 1 - w, dZ_coarse),
              np.arange(1 - w, 1 + w, dZ_fine),
              np.arange(1 + w, Z_end, 2 * dZ_coarse), [Z_end]]
    Z = np.unique(np.round(np.concatenate(pieces + [[0.0, 1.0]]), 12))
    return Z[np.r_[True, np.diff(Z) > 1e-9]]


# ------------------------------------------------------------- the solver
class LubricationSolver:
    """Solve the eps -> 0 equations on the whole channel at once.

    Parameters
    ----------
    geom   : ChannelGeometry (shape, corner rounding, straight sections)
    beta_t : eta_p / eta_0
    N      : number of Chebyshev intervals across the half-channel
    """

    FIELDS = ["Uz", "Azz", "Ayy", "Ayz"]

    def __init__(self, geom: ChannelGeometry, beta_t=0.4, N=20, **grid_kw):
        self.geom, self.bt, self.N = geom, float(beta_t), N
        self.n = N + 1
        self.nv = 4 * self.n + 1                   # unknowns per station
        self.eta = chebyshev_eta(N)
        self.D = diff_matrix(self.eta)
        self.D2 = self.D @ self.D
        self.J = integration_matrix(self.eta)
        self.w = self.J[-1]                        # quadrature weights on [0, 1]
        self.Z = station_grid(geom, **grid_kw)
        self.M = len(self.Z)
        self.H, self.H1, self.H2 = geom.H_all(self.Z)
        self.Dc = central_matrix(self.Z)
        self.Du = upwind_matrix(self.Z)

    # ------------------------------------------------------ helpers
    def _dZ(self, Dmat, f):
        """Apply a Z-derivative matrix to f of shape (B, M, k)."""
        B, M, k = f.shape
        g = Dmat @ f.transpose(1, 0, 2).reshape(M, B * k)
        return np.asarray(g).reshape(M, B, k).transpose(1, 0, 2)

    def inlet_state(self, De):
        H0 = self.H[0]
        fd = fully_developed(self.eta, H0, De)
        return np.concatenate([fd["Uz"], fd["Azz"], fd["Ayy"], fd["Ayz"], [-3 / H0**3]])

    def unpack(self, X):
        n = self.n
        return {name: X[..., k * n:(k + 1) * n] for k, name in enumerate(self.FIELDS)} | \
               {"G": X[..., 4 * n]}

    # ------------------------------------------------------ residual
    def residual(self, X, De, extras=False):
        """All residuals for unknowns X of shape (B, M, nv)."""
        n, bt = self.n, self.bt
        H = self.H[None, :, None]
        H1 = self.H1[None, :, None]
        eta = self.eta[None, None, :]
        Uz, Azz, Ayy, Ayz = (X[..., k * n:(k + 1) * n] for k in range(4))
        G = X[..., 4 * n:4 * n + 1]

        d_eta = lambda f: f @ self.D.T             # d/d eta
        Phi = H * (Uz @ self.J.T)                  # stream function Psi
        Uy = -self._dZ(self.Dc, Phi) + eta * H1 * Uz
        Uz_e = d_eta(Uz)
        dUz_dY = Uz_e / H
        dUz_dZ = self._dZ(self.Dc, Uz) - eta * H1 / H * Uz_e
        dUy_dY = -dUz_dZ                           # continuity, exactly
        dUy_dZ = self._dZ(self.Dc, Uy) - eta * H1 / H * d_eta(Uy)

        def grads(A, Dmat):
            A_e = d_eta(A)
            return self._dZ(Dmat, A) - eta * H1 / H * A_e, A_e / H

        Azz_Z, Azz_Y = grads(Azz, self.Du)
        Ayy_Z, Ayy_Y = grads(Ayy, self.Du)
        Ayz_Z, Ayz_Y = grads(Ayz, self.Du)
        Azz_Zc, _ = grads(Azz, self.Dc)            # force term: central

        R_zz = De * (Uz * Azz_Z + Uy * Azz_Y - 2 * dUz_dZ * Azz - 2 * dUz_dY * Ayz) + Azz
        R_yy = (De * (Uz * Ayy_Z + Uy * Ayy_Y - 2 * dUy_dZ * Ayz - 2 * dUy_dY * Ayy)
                - 2 * dUy_dY + Ayy)
        R_yz = (De * (Uz * Ayz_Z + Uy * Ayz_Y - dUy_dZ * Azz - dUz_dY * Ayy)
                - dUz_dY + Ayz)
        R_m = -G + (1 - bt) * (Uz @ self.D2.T) / H**2 + bt * (Azz_Zc + Ayz_Y)
        R_m[..., 0] = Uz_e[..., 0]                 # symmetry at the centreline
        R_m[..., -1] = Uz[..., -1]                 # no slip at the wall
        R_q = H[..., 0:1] * (Uz @ self.w)[..., None] - 1.0   # unit flux

        R = np.concatenate([R_m, R_zz, R_yy, R_yz, R_q], axis=-1)
        R[:, 0, :] = X[:, 0, :] - self.inlet_state(De)[None, :]  # inlet: fully developed
        if extras:
            return R, {"Uy": Uy, "dUz_dZ": dUz_dZ, "dUz_dY": dUz_dY}
        return R

    # ------------------------------------------------------ Jacobian
    def jacobian(self, X, De, R0=None, ncol=7):
        """Sparse Jacobian dR/dX by coloured finite differences.

        Residuals at station i only involve stations i-3..i+3, so stations
        that are `ncol` = 7 apart can be perturbed in the same evaluation.
        """
        M, nv = self.M, self.nv
        if R0 is None:
            R0 = self.residual(X[None], De)[0]
        h = 1e-7 * (1.0 + np.abs(X))                       # step for each unknown
        reach = (ncol - 1) // 2
        k_all = np.arange(nv)
        rows, cols, vals = [], [], []
        for c in range(ncol):
            st = np.arange(c, M, ncol)                      # stations of this colour
            Xp = np.repeat(X[None], nv, axis=0)             # copy k perturbs unknown k
            Xp[k_all[:, None], st[None, :], k_all[:, None]] += h[st].T
            dR = self.residual(Xp, De) - R0[None]
            for s in st:
                lo, hi = max(0, s - reach), min(M, s + reach + 1)
                d = dR[:, lo:hi, :] / h[s][:, None, None]
                k_idx, r_st, r_idx = np.nonzero(d)
                rows.append((lo + r_st) * nv + r_idx)
                cols.append(s * nv + k_idx)
                vals.append(d[k_idx, r_st, r_idx])
        rows, cols, vals = map(np.concatenate, (rows, cols, vals))
        return sps.csc_matrix((vals, (rows, cols)), shape=(M * nv, M * nv))

    # ------------------------------------------------------ Newton
    def newton(self, X, De, tol=1e-8, max_it=15, verbose=False):
        for it in range(max_it):
            R = self.residual(X[None], De)[0]
            err = np.max(np.abs(R))
            if verbose:
                print(f"    De={De:.3f} it={it} max|R|={err:.2e}")
            if err < tol:
                return X, it
            Jm = self.jacobian(X, De, R)
            dx = spla.splu(Jm).solve(-R.ravel())
            X = X + dx.reshape(X.shape)
        raise RuntimeError(f"Newton did not converge at De={De} (max|R|={err:.2e})")

    def newtonian_guess(self):
        """Exact De = 0 lubrication fields on every station (start of continuation)."""
        X = np.zeros((self.M, self.nv))
        for i, H in enumerate(self.H):
            fd = fully_developed(self.eta, H, 0.0)
            X[i] = np.concatenate([fd["Uz"], fd["Azz"], fd["Ayy"], fd["Ayz"], [-3 / H**3]])
        return X

    def package(self, X, De):
        """Turn the unknown vector into a dict of fields plus Delta P."""
        R, ex = self.residual(X[None], De, extras=True)
        out = {k: v.copy() for k, v in self.unpack(X).items()}
        out["Uy"] = ex["Uy"][0]
        out["Z"], out["eta"], out["H"] = self.Z, self.eta, self.H
        G = out["G"]
        P = np.concatenate([[0.0], np.cumsum(0.5 * (G[1:] + G[:-1]) * np.diff(self.Z))])
        P -= np.interp(1.0, self.Z, P)
        out["P"] = P
        out["dP"] = float(np.interp(0.0, self.Z, P) - np.interp(1.0, self.Z, P))
        out["De"], out["beta_t"], out["max_residual"] = De, self.bt, float(np.abs(R).max())
        return out


def solve(De_list, beta_t=0.4, alpha=4.0, delta=0.01, N=20, L_out=0.5, dDe=0.05,
          verbose=False, **grid_kw):
    """Convenience wrapper: one continuation run that returns every De in De_list."""
    geom = ChannelGeometry(alpha=alpha, L_in=0.5, L_out=L_out, delta=delta)
    solver = LubricationSolver(geom, beta_t=beta_t, N=N, **grid_kw)
    De_list = sorted(set(float(d) for d in De_list))
    X = solver.newtonian_guess()
    out = {}
    De_prev = 0.0
    for De in De_list:
        steps = max(1, int(np.ceil((De - De_prev) / dDe - 1e-9)))
        for De_k in np.linspace(De_prev, De, steps + 1)[1:] if De > De_prev else [De]:
            X, _ = solver.newton(X, float(De_k), verbose=verbose)
        out[De] = solver.package(X, De)
        De_prev = De
    return out, solver

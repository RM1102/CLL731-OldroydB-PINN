"""Physics-informed neural network (PINN) for Oldroyd-B flow in a narrow
hyperbolic contraction (Boyko & Stone 2022).

The network never sees solution data. It is trained only to make the residuals
of the paper's scaled equations (2.14) small at random points in the channel.
All boundary conditions hold exactly because of how the outputs are built:

  stream function  Psi = Psi0(eta) + eta (1 - eta^2)^2 * B(Z) * c_psi * N_psi
      -> continuity, no slip, no penetration, centreline symmetry, unit flux,
         and fully developed flow at both ends of the domain
  pressure         P   = N_P(Z) + eps^2 * N_P1(Z, eta^2)
      -> exactly uniform across the channel when eps = 0 (lubrication), and a
         cross-channel variation of size eps^2 otherwise, as the scaling says
  conformation     A   = A_inlet(eta) + R(Z) * scale(Z) * N_A
      -> fully developed polymer stress at the inlet; A_yz odd in eta

Here eta = Y / H(Z) runs from 0 (centreline) to 1 (wall). Every network takes
(Z, eta^2) as input, so its output is automatically even in eta. Optionally De
is a third input, so one network covers a whole range of De.

Equations (stress form; T = beta_t * A, eps = h_l / l):
  r_z  = -P_Z + (1-b)(eps^2 Uz_ZZ + Uz_YY) + b (Azz_Z + Ayz_Y)
  r_y  = -P_Y + eps^2 [ (1-b)(eps^2 Uy_ZZ + Uy_YY) + b (Ayz_Z + Ayy_Y) ]
  r_zz = De (u.grad Azz - 2 Uz_Z Azz - 2 Uz_Y Ayz) - 2 eps^2 Uz_Z + Azz
  r_yy = De (u.grad Ayy - 2 Uy_Z Ayz - 2 Uy_Y Ayy) - 2 Uy_Y + Ayy
  r_yz = De (u.grad Ayz -   Uy_Z Azz -   Uz_Y Ayy) - eps^2 Uy_Z - Uz_Y + Ayz
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .geometry import ChannelGeometry


# ------------------------------------------------------------- settings
@dataclass
class PINNConfig:
    eps: float = 0.0                 # aspect ratio h_l / l (0 = lubrication limit)
    beta_t: float = 0.4              # eta_p / eta_0
    alpha: float = 4.0               # contraction ratio h_0 / h_l
    De: float = 0.1                  # Deborah number (fixed-De runs)
    De_range: tuple | None = None    # (De_min, De_max) -> De becomes a network input
    L_in: float = 0.5                # straight inlet length
    L_out: float = 0.5               # straight outlet length
    delta: float = 0.01              # corner rounding half-width
    width: int = 64                  # neurons per hidden layer
    depth: int = 4                   # hidden layers per field network
    c_psi: float = 0.1               # size of the stream-function correction
    De_floor: float = 0.05           # floor for the A_zz scale when De -> 0
    geo_features: int = 0            # 1 = also feed H(Z) and the slope indicator
    ymom_power: int = 3              # y-momentum divided by 3/H^power (2 = its own size)
    norm_floor: float = 0.0          # residual sizes never below this x outlet size
    lbfgs_rounds: int = 1            # L-BFGS restarts, each on fresh random points
    lbfgs_device: str = "cpu"        # "cpu" (float64) or "mps" (Mac GPU, float32 only)
    n_points: int = 4096             # collocation points per Adam step
    resample_every: int = 50         # draw fresh points this often
    adam_iters: int = 6000
    lr: float = 2e-3
    lr_final: float = 2e-5
    lbfgs_iters: int = 1500          # L-BFGS polishing steps (float64, CPU)
    lbfgs_points: int = 8192
    seed: int = 0
    device: str = "cpu"
    init_from: str | None = None     # warm start from a saved run directory
    tag: str = ""


# ------------------------------------------------------------- networks
class MLP(nn.Module):
    """Plain fully connected network with tanh activations."""

    def __init__(self, n_in, width, depth, n_out=1):
        super().__init__()
        layers, d = [], n_in
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.Tanh()]
            d = width
        layers += [nn.Linear(d, n_out)]
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.zeros_(self.net[-1].weight)      # start exactly at the base fields

    def forward(self, x):
        return self.net(x).squeeze(-1)


class OldroydPINN(nn.Module):
    FIELDS = ["psi", "p", "azz", "ayy", "ayz"]

    def __init__(self, cfg: PINNConfig):
        super().__init__()
        self.cfg = cfg
        self.geom = ChannelGeometry(cfg.alpha, cfg.L_in, cfg.L_out, cfg.delta)
        self.parametric = cfg.De_range is not None
        n_in = 2 + (1 if self.parametric else 0) + (2 if cfg.geo_features else 0)
        self.nets = nn.ModuleDict({k: MLP(n_in, cfg.width, cfg.depth)
                                   for k in ["psi", "azz", "ayy", "ayz"]})
        self.nets["p"] = MLP(n_in - 1, cfg.width, cfg.depth)    # no eta input
        if cfg.eps > 0:
            self.nets["p1"] = MLP(n_in, cfg.width, cfg.depth)   # O(eps^2) part

    # ------------------------------------------------------ inputs
    def features(self, Z, eta, De, H=None, H1=None):
        """Network inputs, each scaled to [-1, 1].

        Always: position along the channel and eta^2 (so outputs are even in eta).
        geo_features: local height H and the slope indicator
            sigma = -H' alpha / ((alpha - 1) H^2), which is 0 in the straight
            sections and 1 inside the hyperbolic contraction. With it the
            network can follow the sudden change of wall slope at the corners.
        parametric: De.
        """
        g, a = self.geom, self.cfg.alpha
        z = 2 * (Z - g.Z_min) / (g.Z_max - g.Z_min) - 1
        s = 2 * eta**2 - 1
        cols = [z, s]
        if self.cfg.geo_features:
            sigma = -H1 * a / ((a - 1) * H**2)
            cols += [2 * (H - 1) / (a - 1) - 1, 2 * sigma - 1]
        if self.parametric:
            lo, hi = self.cfg.De_range
            cols.append(2 * (De - lo) / max(hi - lo, 1e-12) - 1)
        return torch.stack(cols, dim=-1)

    # ------------------------------------------------------ fields
    def fields(self, Z, Y, De, need_eps_terms=True):
        """Velocity, pressure, conformation and all derivatives the residuals need.

        Z, Y : 1-D tensors that require grad (physical scaled coordinates)
        De   : scalar or 1-D tensor of the same length
        """
        cfg, g = self.cfg, self.geom
        De = torch.as_tensor(De, dtype=Z.dtype, device=Z.device).expand_as(Z)
        H, H1, _ = g.H_all(Z)
        eta = Y / H
        x = self.features(Z, eta, De, H, H1)

        # ---- stream function (exact kinematic conditions)
        ramp_in = 1 - torch.exp(-(Z - g.Z_min) / 0.1)
        ramp_out = 1 - torch.exp(-(g.Z_max - Z) / 0.1)
        psi0 = 0.5 * (3 * eta - eta**3)
        psi = psi0 + eta * (1 - eta**2) ** 2 * ramp_in * ramp_out * cfg.c_psi * self.nets["psi"](x)

        # ---- pressure: along-channel part + O(eps^2) cross-channel part
        x_along = torch.cat([x[:, :1], x[:, 2:]], dim=-1)      # drop the eta^2 column
        P = self.nets["p"](x_along)
        if cfg.eps > 0:
            P = P + cfg.eps**2 * self.nets["p1"](x)

        # ---- conformation (exact inflow, local lubrication scaling)
        H_in = cfg.alpha
        Azz_in = 18 * De * eta**2 / H_in**4
        Ayz_in = -3 * eta / H_in**2
        s_zz = 18 * torch.clamp(De, min=cfg.De_floor) / H**4
        s_yy = 3 * (cfg.alpha - 1) / cfg.alpha
        s_yz = 3 / H**2
        Azz = Azz_in + ramp_in * s_zz * self.nets["azz"](x)
        Ayy = ramp_in * s_yy * self.nets["ayy"](x)
        Ayz = Ayz_in + ramp_in * eta * s_yz * self.nets["ayz"](x)

        # ---- derivatives (autograd through the mapping eta = Y / H(Z))
        def d(f, wrt):
            g = torch.autograd.grad(f, wrt, torch.ones_like(f), create_graph=True,
                                    allow_unused=True)
            return tuple(torch.zeros_like(f) if gi is None else gi for gi in g)

        psi_Z, psi_Y = d(psi, (Z, Y))
        Uz, Uy = psi_Y, -psi_Z
        Uz_Z, Uz_Y = d(Uz, (Z, Y))
        (Uy_Z,) = d(Uy, (Z,))
        Uy_Y = -Uz_Z                                  # continuity holds exactly
        (Uz_YY,) = d(Uz_Y, (Y,))
        P_Z, P_Y = d(P, (Z, Y))
        Azz_Z, Azz_Y = d(Azz, (Z, Y))
        Ayy_Z, Ayy_Y = d(Ayy, (Z, Y))
        Ayz_Z, Ayz_Y = d(Ayz, (Z, Y))
        out = dict(H=H, eta=eta, De=De, psi=psi, Uz=Uz, Uy=Uy, P=P, Azz=Azz, Ayy=Ayy,
                   Ayz=Ayz, Uz_Z=Uz_Z, Uz_Y=Uz_Y, Uy_Z=Uy_Z, Uy_Y=Uy_Y, Uz_YY=Uz_YY,
                   P_Z=P_Z, P_Y=P_Y, Azz_Z=Azz_Z, Azz_Y=Azz_Y, Ayy_Z=Ayy_Z,
                   Ayy_Y=Ayy_Y, Ayz_Z=Ayz_Z, Ayz_Y=Ayz_Y)
        if cfg.eps > 0 and need_eps_terms:
            Uz_ZZ, Uz_ZY = d(Uz_Z, (Z, Y))
            (Uy_ZZ,) = d(Uy_Z, (Z,))
            out.update(Uz_ZZ=Uz_ZZ, Uy_YY=-Uz_ZY, Uy_ZZ=Uy_ZZ)
        return out

    # ------------------------------------------------------ residuals
    def residuals(self, f):
        """The five equation residuals, each divided by its local size."""
        cfg = self.cfg
        b, e2 = cfg.beta_t, cfg.eps**2
        H, De = f["H"], f["De"]
        Uz, Uy = f["Uz"], f["Uy"]
        lap_Uz = f["Uz_YY"] + (e2 * f["Uz_ZZ"] if e2 > 0 else 0.0)
        r_z = -f["P_Z"] + (1 - b) * lap_Uz + b * (f["Azz_Z"] + f["Ayz_Y"])
        if e2 > 0:                                    # y-momentum divided by eps^2
            lap_Uy = f["Uy_YY"] + e2 * f["Uy_ZZ"]
            r_y = -f["P_Y"] / e2 + (1 - b) * lap_Uy + b * (f["Ayz_Z"] + f["Ayy_Y"])
        Azz, Ayy, Ayz = f["Azz"], f["Ayy"], f["Ayz"]
        adv = lambda a: Uz * f[a + "_Z"] + Uy * f[a + "_Y"]
        r_zz = (De * (adv("Azz") - 2 * f["Uz_Z"] * Azz - 2 * f["Uz_Y"] * Ayz)
                - 2 * e2 * f["Uz_Z"] + Azz)
        r_yy = (De * (adv("Ayy") - 2 * f["Uy_Z"] * Ayz - 2 * f["Uy_Y"] * Ayy)
                - 2 * f["Uy_Y"] + Ayy)
        r_yz = (De * (adv("Ayz") - f["Uy_Z"] * Azz - f["Uz_Y"] * Ayy)
                - e2 * f["Uy_Z"] - f["Uz_Y"] + Ayz)
        with torch.no_grad():                         # local sizes (not trained)
            fl = cfg.norm_floor                       # floor = fl x the outlet (H = 1) size
            Dz = torch.clamp(De, min=cfg.De_floor)
            n_z = torch.clamp(3 / H**3, min=3 * fl)
            n_zz = torch.clamp(18 * Dz / H**4, min=18 * Dz * fl)
            n_yy = torch.full_like(H, 3 * (cfg.alpha - 1) / cfg.alpha)
            n_yz = torch.clamp(3 / H**2, min=3 * fl)
            n_y = torch.clamp(3 / H**cfg.ymom_power, min=3 * fl)
        r = {"z-mom": r_z / n_z, "Azz": r_zz / n_zz, "Ayy": r_yy / n_yy, "Ayz": r_yz / n_yz}
        if e2 > 0:                                    # with eps = 0, dP/dY = 0 exactly
            r["y-mom"] = r_y / n_y
        return r

    # ------------------------------------------------------ evaluation
    def evaluate(self, Z, Y, De, chunk=8192):
        """Fields (detached numpy) at arbitrary points; no training graph kept."""
        out = {}
        dev = next(self.parameters()).device
        dt = next(self.parameters()).dtype
        Z = np.asarray(Z, float).ravel()
        Y = np.asarray(Y, float).ravel()
        for i in range(0, len(Z), chunk):
            z = torch.tensor(Z[i:i + chunk], dtype=dt, device=dev, requires_grad=True)
            y = torch.tensor(Y[i:i + chunk], dtype=dt, device=dev, requires_grad=True)
            f = self.fields(z, y, De)
            r = self.residuals(f)
            keep = {k: v for k, v in f.items() if torch.is_tensor(v) and v.shape == z.shape}
            keep.update({"res_" + k: v for k, v in r.items()})
            for k, v in keep.items():
                out.setdefault(k, []).append(v.detach().cpu().numpy())
        return {k: np.concatenate(v) for k, v in out.items()}

    def pressure_drop(self, De, n=400):
        """Delta P = P(Z=0) - P(Z=1) on the centreline, two ways.

        'direct'   : difference of the pressure network output
        'momentum' : integral of the centreline z-momentum terms (eq. 5.3),
                     split into the paper's four contributions
        """
        b, e2 = self.cfg.beta_t, self.cfg.eps**2
        zq, wq = np.polynomial.legendre.leggauss(n)
        Zq = 0.5 * (zq + 1)                           # Gauss points on [0, 1]
        f = self.evaluate(np.r_[Zq, 0.0, 1.0], np.zeros(n + 2), De)
        g = {k: v[:n] for k, v in f.items()}
        e0 = {k: v[n] for k, v in f.items()}
        e1 = {k: v[n + 1] for k, v in f.items()}
        integ = lambda v: -0.5 * np.sum(wq * v)       # int_1^0 v dZ
        Uz_ZZ = g.get("Uz_ZZ", 0 * Zq)
        c1 = e2 * (e0["Uz_Z"] - e1["Uz_Z"])
        c2 = integ(g["Uz_YY"])
        c3 = b * ((e0["Azz"] - 2 * e2 * e0["Uz_Z"]) - (e1["Azz"] - 2 * e2 * e1["Uz_Z"]))
        # De*S_yz = A_yz - eps^2 Uy_Z - Uz_Y, and on the centreline Uy_ZY = -Uz_ZZ
        c4 = b * integ(g["Ayz_Y"] + e2 * Uz_ZZ - g["Uz_YY"])
        direct = float(e0["P"] - e1["P"])
        return {"direct": direct, "momentum": float(c1 + c2 + c3 + c4),
                "terms": [float(c1), float(c2), float(c3), float(c4)]}


# ------------------------------------------------------------- sampling
class Sampler:
    """Random collocation points, denser in the contraction, near the outlet
    (where stresses are largest), near the two corners and near the wall."""

    def __init__(self, model: OldroydPINN, seed=0):
        self.m = model
        self.rng = np.random.default_rng(seed)

    def draw(self, n):
        g, rng, cfg = self.m.geom, self.rng, self.m.cfg
        k = [int(0.10 * n), int(0.45 * n), int(0.20 * n), int(0.10 * n)]
        k.append(n - sum(k))
        Z = np.concatenate([
            rng.uniform(g.Z_min, 0.0, k[0]),                       # inlet section
            rng.uniform(0.0, 1.0, k[1]),                           # contraction
            1.0 - rng.uniform(0, 1, k[2]) ** 2,                    # towards outlet
            np.r_[rng.normal(0, 0.03, k[3] // 2), 1 + rng.normal(0, 0.03, k[3] - k[3] // 2)],
            rng.uniform(1.0, g.Z_max, k[4]),                       # outlet section
        ])
        Z = np.clip(Z, g.Z_min, g.Z_max)
        u = rng.uniform(0, 1, n)
        near_wall = rng.uniform(0, 1, n) < 0.3
        eta = np.where(near_wall, 1 - u**2, u)
        H = g.H(Z)
        Y = eta * H
        De = None
        if cfg.De_range is not None:
            De = rng.uniform(*cfg.De_range, n)
        return Z, Y, De


# ------------------------------------------------------------- training
def _loss(model, Z, Y, De):
    f = model.fields(Z, Y, De)
    r = model.residuals(f)
    parts = {k: torch.mean(v**2) for k, v in r.items()}
    return sum(parts.values()), parts


def _tensors(arrs, dtype, device):
    """Z and Y need gradients (derivatives are taken w.r.t. them); De does not."""
    Z, Y, De = arrs
    t = lambda a, g: torch.tensor(a, dtype=dtype, device=device, requires_grad=g)
    return t(Z, True), t(Y, True), (None if De is None else t(De, False))


def train(cfg: PINNConfig, out_dir, log_every=250, verbose=True):
    """Adam (float32) with resampling, then L-BFGS (float64, CPU). Saves the run."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    model = OldroydPINN(cfg)
    if cfg.init_from:                                # warm start (continuation in De)
        state = torch.load(Path(cfg.init_from) / "model.pt", map_location="cpu")
        missing, _ = model.load_state_dict(state, strict=False)
        if verbose and missing:
            print(f"warm start: new sub-networks start from zero: {sorted(set(k.split('.')[1] for k in missing))}")
    dev = torch.device(cfg.device)
    model = model.to(dev).float()
    sampler = Sampler(model, cfg.seed)
    De_fixed = None if cfg.De_range is not None else cfg.De
    hist = {"it": [], "loss": [], "parts": [], "time": [], "phase": []}
    t0 = time.time()

    def log(it, loss, parts, phase):
        hist["it"].append(it)
        hist["loss"].append(loss.item())
        hist["parts"].append({k: v.item() for k, v in parts.items()})
        hist["time"].append(time.time() - t0)
        hist["phase"].append(phase)
        if verbose:
            p = " ".join(f"{k}={v.item():.1e}" for k, v in parts.items())
            print(f"[{phase} {it:6d}] loss={loss.item():.3e}  {p}  ({time.time() - t0:.0f}s)",
                  flush=True)

    # ---- phase 1: Adam
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    decay = (cfg.lr_final / cfg.lr) ** (1 / max(cfg.adam_iters, 1))
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, decay)
    for it in range(cfg.adam_iters):
        if it % cfg.resample_every == 0:
            Z, Y, De = sampler.draw(cfg.n_points)
            Z, Y, De = _tensors([Z, Y, De], torch.float32, dev)
            De_use = De if De is not None else De_fixed
        opt.zero_grad()
        loss, parts = _loss(model, Z, Y, De_use)
        loss.backward()
        opt.step()
        sched.step()
        if it % log_every == 0 or it == cfg.adam_iters - 1:
            log(it, loss, parts, "adam")

    # ---- phase 2: L-BFGS in rounds on fresh points (fresh points stop the
    #      optimiser from fitting one fixed point set). float64 on the CPU by
    #      default; the Mac GPU (mps) has no float64, so there it runs in float32.
    if cfg.lbfgs_iters > 0:
        lb_dev = cfg.lbfgs_device
        lb_dt = torch.float64 if lb_dev == "cpu" else torch.float32
        model = model.to(lb_dev).to(lb_dt)
        state = {"n": 0}
        for _ in range(max(cfg.lbfgs_rounds, 1)):
            Z, Y, De = sampler.draw(cfg.lbfgs_points)
            Z, Y, De = _tensors([Z, Y, De], lb_dt, lb_dev)
            De_use = De if De is not None else De_fixed
            opt = torch.optim.LBFGS(model.parameters(), lr=1.0,
                                    max_iter=cfg.lbfgs_iters // max(cfg.lbfgs_rounds, 1),
                                    history_size=50, tolerance_grad=1e-12,
                                    tolerance_change=1e-15, line_search_fn="strong_wolfe")

            def closure():
                opt.zero_grad()
                loss, parts = _loss(model, Z, Y, De_use)
                loss.backward()
                if state["n"] % log_every == 0:
                    log(state["n"], loss, parts, "lbfgs")
                state["n"] += 1
                return loss

            opt.step(closure)
        loss, parts = _loss(model, Z, Y, De_use)
        log(state["n"], loss, parts, "lbfgs")
    model = model.to("cpu").double()

    # ---- held-out check on fresh points
    Z, Y, De = sampler.draw(cfg.lbfgs_points)
    Z, Y, De = _tensors([Z, Y, De], torch.float64, "cpu")
    loss, parts = _loss(model, Z, Y, De if De is not None else De_fixed)
    hist["test_loss"] = loss.item()
    hist["test_parts"] = {k: v.item() for k, v in parts.items()}
    hist["train_seconds"] = time.time() - t0

    torch.save(model.state_dict(), out_dir / "model.pt")
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    (out_dir / "history.json").write_text(json.dumps(hist))
    return model, hist


def load(run_dir, device="cpu"):
    """Load a trained run (config + weights) in float64."""
    run_dir = Path(run_dir)
    d = json.loads((run_dir / "config.json").read_text())
    if d.get("De_range") is not None:
        d["De_range"] = tuple(d["De_range"])
    cfg = PINNConfig(**d)
    model = OldroydPINN(cfg).double()
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location=device))
    model.eval()
    hist = json.loads((run_dir / "history.json").read_text())
    return model, hist

"""Recover the finite-element (FE) data points of Boyko & Stone (2022) from the
vector graphics of the arXiv PDF.

Why: the authors did not publish their simulation data. The figures in the
arXiv PDF are vector drawings, so each FE marker (a filled triangle) is stored
as exact coordinates. We read them with PyMuPDF and convert page coordinates
to data values using curves whose formulas we know exactly:
  * Fig. 2 (Delta P vs De):     the 1st-order line  DP0 + De*DP1  (eq. 5.2a-b)
  * Fig. 7 (Delta P vs alpha):  the Newtonian curve  DP0(alpha)   (eq. 5.2a)
  * Fig. 8 (Delta P/(1-b) vs eta_p/eta_s): the 1st-order line     (eq. 5.2a-b)
The calibration is checked against a second known curve; the check error is
printed and written next to the data.

Output: CSV files in 04_results/digitized/.
Run:    python digitize_fe_data.py
"""
from pathlib import Path

import numpy as np
import pymupdf

ROOT = Path(__file__).resolve().parents[2]
PDF = ROOT / "01_research/papers/Boyko_Stone_2022_JFM_OldroydB_narrow_channels_arXiv2109.06928.pdf"
OUT = ROOT / "04_results/digitized"

CYAN = (0.0, 0.68, 0.94)        # 1st-order asymptotic curve (dotted)
INK = (0.14, 0.12, 0.13)        # axes, FE markers, 2nd-order curve (solid)
RED = (0.93, 0.12, 0.14)        # 3rd-order curve (dashed)
GREY = (0.58, 0.59, 0.60)       # Newtonian curve (dash-dot), figure 7 only


def series(a, b, De, order):
    """Truncated Delta P series of eq. (5.2) for alpha = a, beta_t = b."""
    c = [0.75 * (1 + a) * (1 + a**2) / a**3,
         4.5 * b * (1 - a**4) / a**4,
         648 / 35 * b * (1 + a) * (1 + a**2) * (1 - a) ** 2 / a**5,
         216 / 35 * b * (11 - b) * (1 + a) * (1 + a**2) * (1 - a) ** 3 / a**6]
    return sum(c[n] * De**n for n in range(order + 1))


def close(c1, c2, tol=0.05):
    return c1 is not None and all(abs(x - y) < tol for x, y in zip(c1, c2))


def path_points(d):
    """All vertex coordinates of a drawing."""
    pts = []
    for it in d["items"]:
        if it[0] == "l":
            pts += [it[1], it[2]]
        elif it[0] == "c":
            pts += [it[1], it[4]]
    return np.array([(p.x, p.y) for p in pts])


def markers(d):
    """Filled triangles stored in one drawing (3 line items each).

    Returns both the bounding-box centre and the vertex centroid of every
    triangle; which one is the data point is decided by a check at De = 0.
    """
    lines = [it for it in d["items"] if it[0] == "l"]
    box, cen = [], []
    for k in range(0, len(lines) - 2, 3):
        pts = np.array([(p.x, p.y) for it in lines[k:k + 3] for p in it[1:2]])
        box.append([(pts[:, 0].min() + pts[:, 0].max()) / 2,
                    (pts[:, 1].min() + pts[:, 1].max()) / 2])
        cen.append(pts.mean(axis=0))
    return np.array(box), np.array(cen)


def panels(page):
    """Split drawings into left/right panels by x position."""
    ds = page.get_drawings()
    mid = page.rect.width / 2
    return ([d for d in ds if d["rect"].x1 < mid + 5],
            [d for d in ds if d["rect"].x0 > mid - 5])


def frame(ds):
    """Axes box: the widest and tallest ink-coloured stroke drawings."""
    strokes = [d["rect"] for d in ds if d.get("fill") is None and close(d.get("color"), INK)
               and len(d["items"]) < 30]
    x0 = min(r.x0 for r in strokes)
    x1 = max(r.x1 for r in strokes)
    y0 = min(r.y0 for r in strokes)
    y1 = max(r.y1 for r in strokes)
    return x0, x1, y0, y1


def curve(ds, colour, solid=None):
    """Vertices of the longest stroke drawing of a given colour."""
    cands = [d for d in ds if d.get("fill") is None and close(d.get("color"), colour)
             and len(d["items"]) > 30]
    if solid is not None:
        cands = [d for d in cands if (d.get("dashes", "[] 0").startswith("[] ")) == solid]
    return path_points(max(cands, key=lambda d: len(d["items"])))


def calibrate(ds, x_range, known):
    """Page -> data maps. x from the axes frame; y by least squares over every
    interior vertex of the curves in `known` (list of (colour, solid, f(x)))."""
    fx0, fx1, fy0, fy1 = frame(ds)
    sx = (x_range[1] - x_range[0]) / (fx1 - fx0)
    fx = lambda x: x_range[0] + sx * (np.asarray(x) - fx0)
    ys, vs = [], []
    for colour, solid, f in known:
        pts = curve(ds, colour, solid)
        keep = (pts[:, 1] > fy0 + 0.5) & (pts[:, 1] < fy1 - 0.5)   # drop clipped ends
        ys.append(pts[keep, 1])
        vs.append(f(fx(pts[keep, 0])))
    ys, vs = np.concatenate(ys), np.concatenate(vs)
    A = np.c_[ys, np.ones_like(ys)]
    (c, d), *_ = np.linalg.lstsq(A, vs, rcond=None)
    rms = float(np.sqrt(np.mean((A @ [c, d] - vs) ** 2)))
    return fx, (lambda y: c * np.asarray(y) + d), rms


def fe_points(ds, fx, fy):
    """Data values of the FE markers (both centre conventions)."""
    cand = [d for d in ds if d.get("fill") is not None and close(d["fill"], INK)
            and len(d["items"]) > 6]
    box, cen = markers(max(cand, key=lambda d: len(d["items"])))
    return fx(box[:, 0]), fy(box[:, 1]), fy(cen[:, 1])


def digitise(page, x_range, known_for_eps, eps_list, tag):
    rows = []
    for side, e in zip(panels(page), eps_list):
        fx, fy, rms = calibrate(side, x_range, known_for_eps(e))
        x, y_box, y_cen = fe_points(side, fx, fy)
        print(f"{tag} eps={e}: {len(x)} points, calibration rms {rms:.1e}, "
              f"first point x={x.min():.4f}  y(box)={y_box[np.argmin(x)]:.4f} "
              f"y(centroid)={y_cen[np.argmin(x)]:.4f}")
        rows += [(e, float(a), float(b), float(c)) for a, b, c in zip(x, y_box, y_cen)]
    return rows


def fig2(page):
    """Delta P vs De; alpha = 4, beta_t = 0.4."""
    known = lambda e: [(CYAN, False, lambda De: series(4, 0.4, De, 1)),
                       (INK, True, lambda De: series(4, 0.4, De, 2)),
                       (RED, False, lambda De: series(4, 0.4, De, 3))]
    return digitise(page, (0.0, 0.3), known, [0.02, 0.1], "fig2")


def fig7(page):
    """Delta P vs alpha; De = 0.2, beta_t = 0.4."""
    known = lambda e: [(GREY, False, lambda a: series(a, 0.4, 0.2, 0)),
                       (CYAN, False, lambda a: series(a, 0.4, 0.2, 1)),
                       (INK, True, lambda a: series(a, 0.4, 0.2, 2)),
                       (RED, False, lambda a: series(a, 0.4, 0.2, 3))]
    return digitise(page, (1.0, 10.0), known, [0.02, 0.1], "fig7")


def fig8(page):
    """Delta P/(1 - beta_t) vs r = eta_p/eta_s; De = 0.1, alpha = 4.

    In panel (b) the printed asymptotic curves sit about 2 % above eq. (5.2)
    (panel (a) matches it to 1e-5), so panel (b) is calibrated from its axes
    frame instead: the y-limits are 1 and 2.5 (tick labels). Applied to panel
    (a), frame and curve calibrations differ by up to 4e-3, which we take as the
    uncertainty of panel (b). Panel (a) keeps the more accurate curve fit.
    """
    g = lambda order: (lambda r: series(4, r / (1 + r), 0.1, order) * (1 + r))
    known = lambda e: [(CYAN, False, g(1)), (INK, True, g(2)), (RED, False, g(3))]
    rows = []
    for side, e in zip(panels(page), [0.02, 0.1]):
        fx0, fx1, fy0, fy1 = frame(side)
        fx = lambda x: 2.0 * (np.asarray(x) - fx0) / (fx1 - fx0)
        fy = lambda y: 1.0 + 1.5 * (fy1 - np.asarray(y)) / (fy1 - fy0)
        x, y_box, y_cen = fe_points(side, fx, fy)
        _, fy_curves, rms = calibrate(side, (0.0, 2.0), known(e))
        _, y_box_c, y_cen_c = fe_points(side, fx, fy_curves)
        print(f"fig8 eps={e}: {len(x)} points; frame vs curve calibration differ by "
              f"{np.max(np.abs(y_cen - y_cen_c)):.1e} (curve-fit rms {rms:.1e})")
        if e == 0.02:                              # curve fit is valid in panel (a)
            y_box, y_cen = y_box_c, y_cen_c
        rows += [(e, float(a), float(b), float(c)) for a, b, c in zip(x, y_box, y_cen)]
    return rows


def save(name, header, rows):
    OUT.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows)
    np.savetxt(OUT / name, np.array(rows), delimiter=",", header=header,
               fmt="%.6g", comments="")
    print(f"  -> {OUT / name}")


if __name__ == "__main__":
    doc = pymupdf.open(PDF)
    hdr = "eps,{x},{y}_FE_boxcentre,{y}_FE_centroid"
    save("fig2_dP_vs_De.csv", hdr.format(x="De", y="dP"), fig2(doc[15]))
    save("fig7_dP_vs_alpha.csv", hdr.format(x="alpha", y="dP"), fig7(doc[21]))
    save("fig8_dP_vs_viscosity_ratio.csv",
         hdr.format(x="etap_over_etas", y="dP_over_1mb"), fig8(doc[22]))

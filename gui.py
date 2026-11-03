"""A small window to explore the trained PINN (Windows and Mac).

Pick a case and a field, move the De slider, and type a point (Z, Y) to read
the predicted values there. Built only from matplotlib, so nothing extra is
needed beyond the project's requirements.

Run:  python gui.py               (or use setup_and_run.bat / setup_and_run.sh)
Test without a screen:  python gui.py --selftest gui_test.png
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "03_code"))
SELFTEST = "--selftest" in sys.argv

import matplotlib  # noqa: E402

if SELFTEST:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.widgets import Button, RadioButtons, Slider, TextBox  # noqa: E402

from oldroydb import analytical as an  # noqa: E402
from oldroydb import plotting as pl  # noqa: E402
from oldroydb import predict as P  # noqa: E402
from oldroydb.report import load_fe  # noqa: E402

CASES = P.available_cases()
if not CASES:
    sys.exit("No trained networks found in 04_results/pinn/.")
FIELDS = list(P.FIELD_NAMES)
SIGNED = {"Uy", "Tyz", "Tyy"}
FE = load_fe(ROOT)
state = {"case": "eps0.02" if "eps0.02" in CASES else CASES[0], "field": "Tzz", "De": 0.2}

pl.style()
fig = plt.figure(figsize=(13.5, 7.6))
fig.canvas.manager.set_window_title("Oldroyd-B PINN explorer (CLL731)") if hasattr(
    fig.canvas, "manager") and fig.canvas.manager else None
ax_map = fig.add_axes([0.05, 0.47, 0.50, 0.40])
ax_cb = fig.add_axes([0.56, 0.47, 0.012, 0.40])
ax_dp = fig.add_axes([0.66, 0.47, 0.32, 0.40])
fig.text(0.05, 0.955, "Oldroyd-B flow in a 4:1 hyperbolic contraction: PINN predictions",
         fontsize=13, weight="bold")
fig.text(0.05, 0.925, "Scaled units of Boyko & Stone (2022). Z along the channel, Y across it.",
         fontsize=9, color="#52514e")

ax_case = fig.add_axes([0.05, 0.05, 0.25, 0.27])
ax_field = fig.add_axes([0.32, 0.05, 0.15, 0.27])
ax_slider = fig.add_axes([0.56, 0.33, 0.36, 0.035])
ax_z = fig.add_axes([0.56, 0.24, 0.07, 0.045])
ax_y = fig.add_axes([0.67, 0.24, 0.07, 0.045])
ax_go = fig.add_axes([0.76, 0.24, 0.08, 0.045])
ax_save = fig.add_axes([0.86, 0.24, 0.10, 0.045])
ax_out = fig.add_axes([0.52, 0.02, 0.46, 0.19])
ax_out.axis("off")
for a in (ax_case, ax_field):
    a.set_facecolor("#f6f5f1")
ax_case.set_title("case", loc="left", fontsize=10)
ax_field.set_title("field", loc="left", fontsize=10)

r_case = RadioButtons(ax_case, [P.CASES[c]["label"] for c in CASES],
                      active=CASES.index(state["case"]))
r_field = RadioButtons(ax_field, [P.FIELD_NAMES[f] for f in FIELDS],
                       active=FIELDS.index(state["field"]))
s_De = Slider(ax_slider, "De", 0.0, 0.3, valinit=state["De"])
t_z = TextBox(ax_z, "Z ", initial="0.8")
t_y = TextBox(ax_y, "Y ", initial="0.5")
b_go = Button(ax_go, "Predict")
b_save = Button(ax_save, "Save picture")
out_text = ax_out.text(0, 1, "", va="top", family="monospace", fontsize=9)


def snap_slider():
    """Fixed-De cases only have a few trained values: snap the slider to them."""
    Des = P.trained_De(state["case"])
    s_De.valstep = None if Des is None else np.array(Des)
    if Des is not None:
        s_De.set_val(min(Des, key=lambda d: abs(d - s_De.val)))


def draw_map():
    model, De, note = P.load(state["case"], state["De"])
    ZZ, YY, f = P.grid(model, De, nz=101, ny=26)
    F = f[state["field"]]
    ax_map.clear()
    ax_cb.clear()
    signed = state["field"] in SIGNED
    vmax = np.nanmax(np.abs(F)) if signed else np.nanmax(F)
    vmin = -vmax if signed else min(np.nanmin(F), 0)
    cs = ax_map.contourf(ZZ, YY, F, levels=np.linspace(vmin, vmax + 1e-12, 21),
                         cmap=pl.DIV if signed else pl.SEQ)
    ax_map.plot(ZZ[:, -1], YY[:, -1], color=pl.INK, lw=1.2)
    span = max(abs(vmin), abs(vmax), 1e-30)                # print round-off as a clean 0
    fig.colorbar(cs, cax=ax_cb, format=matplotlib.ticker.FuncFormatter(
        lambda x, _: "0" if abs(x) < 1e-6 * span else f"{x:.2g}"))
    ax_map.set_title(f"{P.FIELD_NAMES[state['field']]},  De = {De:g}", loc="left")
    ax_map.set_xlabel("Z (along)")
    ax_map.set_ylabel("Y (across)")
    ax_map.grid(False)
    return model, De, note


def draw_dp(model, De):
    ax_dp.clear()
    Dg = np.linspace(0, 0.3, 61)
    ax_dp.plot(Dg, an.pressure_drop(Dg, 3), color=pl.THEORY, ls="--", lw=1.2,
               label="paper formula O(De$^3$)")
    eps = P.CASES[state["case"]]["eps"]
    if eps in FE:
        ax_dp.plot(*FE[eps], "^", color=pl.FE, ms=5, label="paper simulations")
    Des = P.trained_De(state["case"])
    if Des is None:                                   # one network: draw its curve
        Ds = np.linspace(0, 0.3, 16)
        ax_dp.plot(Ds, [P.pressure_drop(model, d)[0] for d in Ds], color=pl.PINN,
                   label="PINN (any De)")
    else:
        pts = [(d, P.pressure_drop(P.load(state["case"], d)[0], d)[0]) for d in Des]
        ax_dp.plot(*zip(*pts), "o", color=pl.PINN, mec="white", label="PINN")
    dp = P.pressure_drop(model, De)[0]
    ax_dp.plot([De], [dp], "o", ms=12, mfc="none", mec=pl.INK, mew=1.5)
    ax_dp.set_title(f"pressure drop  $\\Delta P$ = {dp:.4f}", loc="left")
    ax_dp.set_xlabel("De")
    ax_dp.set_ylim(0.4, 1.08)
    ax_dp.legend(fontsize=8, loc="lower left")
    return dp


def predict_point(event=None):
    model, De, _ = P.load(state["case"], state["De"])
    try:
        Z, Y = float(t_z.text), float(t_y.text)
    except ValueError:
        out_text.set_text("Type numbers for Z and Y.")
        fig.canvas.draw_idle()
        return
    f = P.fields(model, De, Z, Y)
    if not f["inside"][0]:
        H = model.geom.H(np.clip(Z, model.geom.Z_min, model.geom.Z_max))
        msg = f"(Z, Y) = ({Z:g}, {Y:g}) is outside the channel: at this Z the wall is at Y = {H:.3f}"
    else:
        lines = [f"prediction at Z = {Z:g}, Y = {Y:g}  (case {state['case']}, De = {De:g})"]
        lines += [f"  {P.FIELD_NAMES[k]:28s} {f[k][0]: .5f}" for k in FIELDS]
        msg = "\n".join(lines)
    out_text.set_text(msg)
    ax_map.plot([Z], [Y], "x", color="#e34948", ms=9, mew=2)
    fig.canvas.draw_idle()


def refresh(*_):
    model, De, note = draw_map()
    dp = draw_dp(model, De)
    pc = P.PRESSURE_SCALE_PA[P.CASES[state["case"]]["eps"]]
    out_text.set_text(f"case {state['case']}, De = {De:g}{'  (' + note + ')' if note else ''}\n"
                      f"pressure drop = {dp:.4f} (scaled) = {dp * pc:.0f} Pa in the paper's channel\n\n"
                      "Type a point (Z, Y) and press Predict to read every field there.")
    fig.canvas.draw_idle()


def on_case(label):
    state["case"] = CASES[[P.CASES[c]["label"] for c in CASES].index(label)]
    snap_slider()
    state["De"] = s_De.val
    refresh()


def on_field(label):
    state["field"] = FIELDS[[P.FIELD_NAMES[f] for f in FIELDS].index(label)]
    refresh()


def on_slider(val):
    if abs(val - state["De"]) > 1e-9:
        state["De"] = val
        refresh()


def on_save(event=None, path=None):
    path = path or str(ROOT / f"pinn_{state['case']}_{state['field']}_De{state['De']:.2f}.png")
    fig.savefig(path, dpi=150)
    out_text.set_text(f"saved {path}")
    fig.canvas.draw_idle()


r_case.on_clicked(on_case)
r_field.on_clicked(on_field)
s_De.on_changed(on_slider)
b_go.on_clicked(predict_point)
b_save.on_clicked(on_save)
snap_slider()
refresh()

if SELFTEST:                                          # exercise every control, then save
    out = sys.argv[sys.argv.index("--selftest") + 1] if len(sys.argv) > sys.argv.index(
        "--selftest") + 1 else "gui_selftest.png"
    for i in range(len(CASES)):                       # click through every case
        r_case.set_active(i)
    r_field.set_active(FIELDS.index("Tyz"))
    s_De.set_val(0.3)
    predict_point()
    on_save(path=out)
    print(f"GUI self-test passed: every case loaded; picture saved to {out}")
elif matplotlib.get_backend().lower() in ("agg", "pdf", "svg", "ps", "template"):
    on_save(path=str(ROOT / "gui_snapshot.png"))
    print("No window could be opened (no GUI backend). A picture was saved to gui_snapshot.png.\n"
          "On Windows, install Python from python.org (it includes the window toolkit, Tk).")
else:
    plt.show()

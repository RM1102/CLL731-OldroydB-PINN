"""Re-derive the low-Deborah-number solution of Boyko & Stone (2022) with SymPy.

Why: every analytical curve we compare the PINN against comes from this
solution, so we check each published formula instead of trusting the typeset
equations. The script solves eqs (2.14)-(2.16) order by order in De for a
general channel shape H(Z) (lubrication limit, eps -> 0), then
  1. checks the printed fields (3.6)-(3.8), (3.16), (3.20)-(3.22);
  2. computes Delta P_3 directly from the O(De^3) momentum balance and
     compares it with the reciprocal-theorem result (4.14) and with (5.2d).

Run:  python derive_perturbation_sympy.py   (takes about a minute)
Output is printed and also written to 04_results/analytical/sympy_check.txt
"""
from pathlib import Path

import sympy as sp

Y, Z, b = sp.symbols("Y Z beta_t", real=True)    # b = beta tilde = eta_p / eta_0
H = sp.Function("H")(Z)
Hp, Hpp = H.diff(Z), H.diff(Z, 2)

lines = []


def report(name, expr):
    ok = sp.simplify(expr) == 0
    lines.append(f"{'OK ' if ok else 'XX '} {name}")
    print(lines[-1])
    return ok


def momentum_solution(polymer_force, flux):
    """dP/dZ = d2U/dY2 + polymer_force(Y, Z).  Returns U(Y, Z), dP/dZ(Z)."""
    G = sp.Symbol("G")
    c1, c2 = sp.symbols("c1 c2")
    # integrate twice in Y: U'' = G - polymer_force
    U = sp.integrate(sp.integrate(G - polymer_force, (Y, 0, Y)), Y) + c1 * Y + c2
    eqs = [sp.Eq(U.diff(Y).subs(Y, 0), 0), sp.Eq(U.subs(Y, H), 0),
           sp.Eq(sp.integrate(U, (Y, 0, H)), flux)]
    sol = sp.solve(eqs, [G, c1, c2], dict=True)[0]
    return sp.simplify(U.subs(sol)), sp.simplify(sol[G])


def uy_from(Uz):
    """Continuity dUz/dZ + dUy/dY = 0 with Uy(0) = 0."""
    return sp.simplify(-sp.integrate(Uz.diff(Z), (Y, 0, Y)))


def S_terms(Uz, Uy, Azz, Ayy, Ayz):
    """S_zz and S_yz of eq. (2.16) (A here = scaled conformation, eps -> 0)."""
    Szz = -(Uz * Azz.diff(Z) + Uy * Azz.diff(Y) - 2 * Uz.diff(Z) * Azz
            - 2 * Uz.diff(Y) * Ayz)
    Syz = -(Uz * Ayz.diff(Z) + Uy * Ayz.diff(Y) - Uy.diff(Z) * Azz
            - Uz.diff(Y) * Ayy)
    return Szz, Syz


# ---------------------------------------------------------------- order De^0
Uz0, G0 = momentum_solution(0, 1)
Uy0 = uy_from(Uz0)
Azz0 = sp.Integer(0)
Ayy0 = 2 * Uy0.diff(Y)
Ayz0 = Uz0.diff(Y)
report("(3.5)  dP0/dZ = -3/H^3", G0 + 3 / H**3)
report("(3.6)  Uz0", Uz0 - sp.Rational(3, 2) * (H**2 - Y**2) / H**3)
report("(3.7)  Uy0", Uy0 - sp.Rational(3, 2) * Hp * Y * (H**2 - Y**2) / H**4)
report("(3.8)  Ayy0", Ayy0 - 3 * Hp * (-3 * Y**2 + H**2) / H**4)
report("(3.8)  Ayz0", Ayz0 + 3 * Y / H**3)

# ---------------------------------------------------------------- order De^1
# Constitutive equations (2.14d-f) at O(De): lower-order fields only.
Szz0, Syz0 = S_terms(Uz0, Uy0, Azz0, Ayy0, Ayz0)
Uz1, G1 = momentum_solution(b * (Szz0.diff(Z) + Syz0.diff(Y)), 0)
Uy1 = uy_from(Uz1)
Azz1 = sp.simplify(2 * Uz0.diff(Y) * Ayz0)                     # (3.10d)
Ayy1 = sp.simplify(-(Uz0 * Ayy0.diff(Z) + Uy0 * Ayy0.diff(Y)
                     - 2 * Uy0.diff(Z) * Ayz0 - 2 * Uy0.diff(Y) * Ayy0)
                   + 2 * Uy1.diff(Y))                            # (3.10e)
Ayz1 = sp.simplify(-(Uz0 * Ayz0.diff(Z) + Uy0 * Ayz0.diff(Y)
                     - Uy0.diff(Z) * Azz0 - Uz0.diff(Y) * Ayy0)
                   + Uz1.diff(Y))                                # (3.10f)
report("(3.14) dP1/dZ = 9/2 b d/dZ(1/H^4)", G1 - sp.Rational(9, 2) * b * sp.diff(1 / H**4, Z))
report("(3.13) Uz1 = 0", Uz1)
report("(3.16a) Azz1 = 18 Y^2/H^6", Azz1 - 18 * Y**2 / H**6)
printed_Ayz1 = 18 * Hp * Y * (H**2 - 2 * Y**2) / H**7
ok = report("(3.16b) Ayz1 as printed in arXiv v1 (+18 H' ...)", Ayz1 - printed_Ayz1)
if not ok:
    report("(3.16b) Ayz1 with corrected sign (-18 H' ...)", Ayz1 + printed_Ayz1)
printed_Ayy1 = sp.Rational(9, 2) * (4 * (-2 * Y**2 + H**2)**2 * Hp**2
                                    - H * Hpp * (Y**2 - H**2)**2) / H**8
report("(3.16c) Ayy1", Ayy1 - printed_Ayy1)

# ---------------------------------------------------------------- order De^2
Szz1, Syz1 = S_terms(Uz0, Uy0, Azz1, Ayy1, Ayz1)
# Cross terms with first-order velocity vanish because Uz1 = Uy1 = 0.
force2 = sp.simplify(b * (Szz1.diff(Z) + Syz1.diff(Y)))
printed_force2 = sp.Rational(81, 2) * b * (Y**2 - H**2) * (
    4 * H * Hp**2 + Hpp * (Y**2 - H**2)) / H**10
report("(3.19) polymer force at O(De^2)", force2 - printed_force2)
Uz2, G2 = momentum_solution(force2, 0)
Uy2 = uy_from(Uz2)
report("(3.21) dP2/dZ", G2 + sp.Rational(324, 35) * b * (14 * Hp**2 / H**7 - 3 * Hpp / H**6))
printed_Uz2 = (Y**2 - H**2) * (sp.Rational(1, 2) * G2 - sp.Rational(27, 20) * b * (
    10 * H * Hp**2 * (Y**2 - 5 * H**2) + Hpp * (Y**4 - 4 * (Y * H)**2 + 11 * H**4)) / H**10)
report("(3.20) Uz2", Uz2 - printed_Uz2)

# Conformation at O(De^2) from (3.17d-f)
Azz2 = sp.simplify(-(Uz0 * Azz1.diff(Z) + Uy0 * Azz1.diff(Y) - 2 * Uz0.diff(Z) * Azz1
                     - 2 * Uz0.diff(Y) * Ayz1))                  # (3.17d), uses Uz1 = 0
Ayy2 = sp.simplify(-(Uz0 * Ayy1.diff(Z) + Uy0 * Ayy1.diff(Y) - 2 * Uy0.diff(Z) * Ayz1
                     - 2 * Uy0.diff(Y) * Ayy1) + 2 * Uy2.diff(Y))
Ayz2 = sp.simplify(-(Uz0 * Ayz1.diff(Z) + Uy0 * Ayz1.diff(Y) - Uy0.diff(Z) * Azz1
                     - Uz0.diff(Y) * Ayy1) + Uz2.diff(Y))

# ---------------------------------------------------------------- order De^3
# Momentum at O(De^3): dP3/dZ = d2Uz3/dY2 + b (dSzz2/dZ + dSyz2/dY), where S at
# O(De^2) contains all products U_i * A_j with i + j = 2 (eq. 4.13).
Szz2 = -(Uz0 * Azz2.diff(Z) + Uy0 * Azz2.diff(Y) - 2 * Uz0.diff(Z) * Azz2
         - 2 * Uz0.diff(Y) * Ayz2
         + Uz2 * Azz0.diff(Z) + Uy2 * Azz0.diff(Y) - 2 * Uz2.diff(Z) * Azz0
         - 2 * Uz2.diff(Y) * Ayz0)
Syz2 = -(Uz0 * Ayz2.diff(Z) + Uy0 * Ayz2.diff(Y) - Uy0.diff(Z) * Azz2
         - Uz0.diff(Y) * Ayy2
         + Uz2 * Ayz0.diff(Z) + Uy2 * Ayz0.diff(Y) - Uy2.diff(Z) * Azz0
         - Uz2.diff(Y) * Ayy0)
force3 = b * (Szz2.diff(Z) + Syz2.diff(Y))
_, G3 = momentum_solution(sp.expand(force3), 0)

# Evaluate on the hyperbolic channel alpha = 4 and integrate in Z numerically.
alpha = sp.Integer(4)
Hh = alpha / ((alpha - 1) * Z + 1)
results = {}
for name, G in [("dP0", G0), ("dP1", G1), ("dP2", G2), ("dP3", G3)]:
    g = sp.simplify(G.subs(H, Hh).doit())
    g = sp.lambdify((Z, b), g, "mpmath")
    import mpmath as mp
    results[name] = lambda bb, g=g: -mp.quad(lambda z: g(z, bb), [0, 1])

bt = 0.4
a = 4.0
closed = {
    "dP0": 3 / 4 * (1 + a) * (1 + a**2) / a**3,
    "dP1": 9 / 2 * bt * (1 - a**4) / a**4,
    "dP2": 648 / 35 * bt * (1 + a) * (1 + a**2) * (1 - a)**2 / a**5,
    "dP3 (5.2d)": 216 / 35 * bt * (11 - bt) * (1 + a) * (1 + a**2) * (1 - a)**3 / a**6,
}
Hn = lambda z: a / ((a - 1) * z + 1)
H1n = lambda z: -a * (a - 1) / ((a - 1) * z + 1)**2
H2n = lambda z: 2 * a * (a - 1)**2 / ((a - 1) * z + 1)**3
closed["dP3 (4.14)"] = (648 * bt * (9 - bt) / 35 * (H1n(0)**2 / Hn(0)**8 - H1n(1)**2 / Hn(1)**8)
                        - 216 * bt * (8 - bt) / 35 * (H2n(0) / Hn(0)**7 - H2n(1) / Hn(1)**7))

lines.append("\nHyperbolic channel, alpha = 4, beta_t = 0.4:")
for k in ["dP0", "dP1", "dP2", "dP3"]:
    lines.append(f"  {k} from direct O(De^n) momentum balance = {float(results[k](bt)):+.6f}")
for k, v in closed.items():
    lines.append(f"  {k} closed form in paper            = {v:+.6f}")
print("\n".join(lines[-9:]))

out = Path(__file__).resolve().parents[2] / "04_results" / "analytical"
out.mkdir(parents=True, exist_ok=True)
(out / "sympy_check.txt").write_text("\n".join(lines) + "\n")
print(f"\nwritten to {out / 'sympy_check.txt'}")

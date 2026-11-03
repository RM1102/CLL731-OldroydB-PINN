"""O(eps^2) correction to the Newtonian pressure drop in a slowly varying channel.

Why: the paper's finite-element value at De = 0 is not exactly the lubrication
value 255/256, because the channel has a finite aspect ratio eps. Extended
lubrication theory (Tavakol et al. 2017, Proc. R. Soc. A 473, 20170234) gives
the first correction, of size eps^2. We derive it here for our scaled
equations, so that the full 2-D PINN at De = 0 can be checked against it.

Scaled Stokes equations (beta_t plays no role when De = 0):
  dP/dZ = eps^2 Uz_ZZ + Uz_YY,      dP/dY = eps^2 (eps^2 Uy_ZZ + Uy_YY)
Expand  Uz = Uz0 + eps^2 Uz1,  P = P0(Z) + eps^2 P1(Z, Y),  and keep O(eps^2).
"""
from pathlib import Path

import sympy as sp
from scipy.integrate import quad

Y, Z = sp.symbols("Y Z", real=True)
H = sp.Function("H")(Z)

Uz0 = sp.Rational(3, 2) * (H**2 - Y**2) / H**3
Uy0 = -sp.integrate(Uz0.diff(Z), (Y, 0, Y))
# y-momentum at O(eps^2): P1_Y = Uy0_YY  ->  P1 = p1(Z) + q(Z, Y)
q = sp.integrate(Uy0.diff(Y, 2), (Y, 0, Y))
# z-momentum at O(eps^2): p1'(Z) + q_Z = Uz0_ZZ + Uz1_YY, with Uz1(H) = 0,
# Uz1_Y(0) = 0 and zero extra flux. Solve for p1'(Z) = G1.
G1 = sp.Symbol("G1")
c1, c2 = sp.symbols("c1 c2")
rhs = G1 + q.diff(Z) - Uz0.diff(Z, 2)
Uz1 = sp.integrate(sp.integrate(rhs, (Y, 0, Y)), Y) + c1 * Y + c2
sol = sp.solve([Uz1.diff(Y).subs(Y, 0), Uz1.subs(Y, H), sp.integrate(Uz1, (Y, 0, H))],
               [G1, c1, c2], dict=True)[0]
G1 = sp.simplify(sol[G1])
print("dp1/dZ (centreline pressure gradient at O(eps^2)) =", G1)

# Delta P on the centreline between Z = 0 and 1: q(Z, 0) = 0, so only p1 counts.
alpha = 4
Hh = alpha / ((alpha - 1) * Z + 1)
g = sp.lambdify(Z, G1.subs(H, Hh).doit())
coef = -quad(g, 0, 1)[0]
lines = [f"dp1/dZ = {G1}",
         f"hyperbolic channel alpha = 4: Delta P = 255/256 + C eps^2 with C = {coef:.4f}",
         f"  eps = 0.02 -> {255/256 + coef * 0.02**2:.5f}",
         f"  eps = 0.1  -> {255/256 + coef * 0.1**2:.5f}",
         "(smooth-geometry result; the corners add effects the expansion does not see)"]
print("\n".join(lines))
out = Path(__file__).resolve().parents[2] / "04_results" / "analytical" / "newtonian_eps2.txt"
out.write_text("\n".join(lines) + "\n")

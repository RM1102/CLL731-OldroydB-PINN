"""Quick checks of the building blocks. Run:  python -m pytest -q 03_code/tests"""
import sys
from pathlib import Path

import numpy as np
import pytest

_trapezoid = getattr(np, "trapezoid", None) or np.trapz

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oldroydb import analytical as an  # noqa: E402
from oldroydb import lubrication_fd as fd  # noqa: E402
from oldroydb.geometry import (ChannelGeometry, hyperbolic_H, hyperbolic_dH,  # noqa: E402
                               hyperbolic_d2H)


def test_geometry_derivatives_match_finite_differences():
    g = ChannelGeometry(delta=0.02)
    Z = np.linspace(-0.4, 1.4, 20001)
    H, H1, H2 = g.H_all(Z)
    assert np.allclose(np.gradient(H, Z), H1, atol=1e-3 * np.abs(H1).max())
    # H''' jumps at the ends of each rounding zone, so allow a small relative error
    err = np.abs(np.gradient(H1, Z) - H2)[5:-5]
    assert err.max() < 2e-3 * np.abs(H2).max()


def test_geometry_is_exact_away_from_corners():
    g = ChannelGeometry(delta=0.01)
    Z = np.linspace(0.05, 0.95, 50)
    assert np.allclose(g.H(Z), hyperbolic_H(Z))
    assert np.allclose(g.H(np.array([-0.3, 1.3])), [4.0, 1.0])


def test_closed_form_equals_general_shape_formulas():
    general = an.dP_terms_general(hyperbolic_H, hyperbolic_dH, hyperbolic_d2H)
    assert np.allclose(general, an.dP_terms_hyperbolic(), rtol=1e-10)


def test_fully_developed_flux_is_one():
    eta = np.linspace(0, 1, 2001)
    for H in [1.0, 2.5, 4.0]:
        Uz = an.fully_developed(eta, H, 0.2)["Uz"]
        assert _trapezoid(Uz, eta * H) == pytest.approx(1.0, rel=1e-6)


def test_reference_solver_newtonian_and_first_order():
    sols, _ = fd.solve([0.0, 0.01], N=16, delta=0.005, dDe=0.01)
    dP0, dP1, dP2 = an.dP_terms_hyperbolic()[:3]
    assert sols[0.0]["dP"] == pytest.approx(dP0, rel=2e-4)
    slope = (sols[0.01]["dP"] - sols[0.0]["dP"]) / 0.01
    # slope = dP1 + 0.01 dP2 + O(1e-4); corner rounding adds about 1 %
    assert slope == pytest.approx(dP1 + 0.01 * dP2, rel=0.02)


def test_pinn_boundary_conditions_hold_for_random_weights():
    from oldroydb import pinn, report
    model = pinn.OldroydPINN(pinn.PINNConfig(eps=0.02, De=0.1, geo_features=1)).double()
    for Z0, moved, flux, uz_w, uy_w, uy_c, ayz_c in report.constraint_check(model):
        assert moved > 1e-3                      # the network really changes the flow
        assert flux == pytest.approx(1.0, abs=1e-12)
        assert max(uz_w, uy_w, uy_c, ayz_c) < 1e-12


def test_trained_pinn_matches_reference_solver():
    from oldroydb import compare, pinn
    root = Path(__file__).resolve().parents[2]
    run = root / "04_results" / "pinn" / "eps0_De0.1"
    if not (run / "model.pt").exists():
        pytest.skip("trained network not present")
    model, _ = pinn.load(run)
    ref = compare.load_reference(root / "04_results" / "reference" / "fd_eps0.npz")
    assert model.pressure_drop(0.1)["direct"] == pytest.approx(
        compare.reference_at(ref, 0.1)["dP"], rel=5e-3)

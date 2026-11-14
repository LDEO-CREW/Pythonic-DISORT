import numpy as np
from scipy.integrate import quad_vec
import PythonicDISORT
from math import pi
import pytest

# ============================================================================================================
# Test Problem NT: NT corrections, at the quadrature nodes and at arbitrary `mu` (`interpolate_mu=True`)
# ============================================================================================================

# Inhomogeneous 3-layer atmosphere with delta-M scaling and NT corrections applicable
tau_arr = np.array([2, 5, 8])
NLayers = len(tau_arr)
omega_arr = np.array([0.85, 0.75, 0.6])
NQuad = 8
g_per_layer = np.array([0.85, 0.6, 0.2])
Leg_coeffs_all = g_per_layer[:, None] ** np.arange(32)[None, :]
mu0 = 0.6
I0 = pi / mu0
phi0 = 0.9 * pi
f_arr = Leg_coeffs_all[:, NQuad]

# Evaluation points; the downward (negative) `mu` values are chosen away from
# `-mu0` and from the delta-scaled `-mu0` of the IMS correction
mu_eval = np.array([0.9, 0.7, 0.35, 0.15, -0.2, -0.45, -0.8, -0.95])
tau_eval = np.array([0.0, 0.5, 2.0, 3.7, 8.0])
phi_eval = np.array([0.0, 1.1, pi, 5.0])


def _pydisort(NT_cor, **kwargs):
    return PythonicDISORT.pydisort(
        tau_arr, omega_arr,
        NQuad,
        Leg_coeffs_all,
        mu0, I0, phi0,
        f_arr=f_arr,
        NT_cor=NT_cor,
        **kwargs,
    )

def test_NT_at_quad_nodes():
    """At the quadrature nodes, the corrections which are evaluated at the `mu` values passed to `u`
    (`interpolate_mu=True`) must agree with those which are evaluated at the quadrature nodes."""
    mu_arr, _, _, _, u_nodes = _pydisort(True)
    u = _pydisort(True, interpolate_mu=True)[4]

    for tau_derivative_order in (0, -1, 1):
        ref = u_nodes(tau_eval, phi_eval, tau_derivative_order)
        res = u(mu_arr, tau_eval, phi_eval, tau_derivative_order)
        assert np.max(np.abs(res - ref)) < 1e-12 * np.max(np.abs(ref))


def test_interp_NT_antiderivative():
    """The antiderivative branch of the corrections at evaluation points must
    integrate correctly (same idiom as the `I_test.py` problems)."""
    u = _pydisort(True, interpolate_mu=True)[4]

    end = tau_arr[-1]
    upper = tau_arr.astype(float)
    lower = np.insert(upper[:-1] + 1e-15, 0, 0)

    integral = quad_vec(lambda tau: u(mu_eval, tau, phi_eval), 0, end)[0]
    telescoped = (
        u(mu_eval, upper, phi_eval, -1)
        - u(mu_eval, lower, phi_eval, -1)
    ).sum(axis=1)
    assert np.allclose(integral, telescoped)


def test_NT_derivative():
    """The tau-derivative of the corrected intensity must match central finite differences of it,
    at the quadrature nodes and at arbitrary `mu` (`interpolate_mu=True`), including at the
    removable singularities of the corrections."""
    tau_interior = np.array([0.3, 1.1, 1.99, 2.5, 4.2, 5.01, 6.6, 7.9])
    h = 1e-5
    u_nodes = _pydisort(True)[4]
    derivative = u_nodes(tau_interior, phi_eval, 1)
    finite_diff = (u_nodes(tau_interior + h, phi_eval) - u_nodes(tau_interior - h, phi_eval)) / (2 * h)
    assert np.max(np.abs(derivative - finite_diff)) < 1e-7 * np.max(np.abs(derivative))

    u = _pydisort(True, interpolate_mu=True)[4]
    mu = np.append(mu_eval, -mu0)
    derivative = u(mu, tau_interior, phi_eval, 1)
    finite_diff = (u(mu, tau_interior + h, phi_eval) - u(mu, tau_interior - h, phi_eval)) / (2 * h)
    assert np.max(np.abs(derivative - finite_diff)) < 1e-7 * np.max(np.abs(derivative))


def test_NT_Fourier_error_passthrough():
    """`return_Fourier_error` must pass through the NT corrections: the intensities are
    unchanged and the Fourier error is that of the uncorrected `u`, both at the quadrature
    nodes and at arbitrary `mu` (`interpolate_mu=True`)."""
    u_true = _pydisort(True)[4]
    u_false = _pydisort(False)[4]
    u_true_at_mu = _pydisort(True, interpolate_mu=True)[4]
    u_false_at_mu = _pydisort(False, interpolate_mu=True)[4]

    for tau_derivative_order in (0, -1, 1):
        Fourier_error_ref = u_false(tau_eval, phi_eval, tau_derivative_order, True)[1]
        assert np.isscalar(Fourier_error_ref) and Fourier_error_ref >= 0

        vals, Fourier_error = u_true(tau_eval, phi_eval, tau_derivative_order, True)
        assert np.allclose(vals, u_true(tau_eval, phi_eval, tau_derivative_order))
        assert np.isclose(Fourier_error, Fourier_error_ref)

        Fourier_error_ref = u_false_at_mu(mu_eval, tau_eval, phi_eval, tau_derivative_order, True)[1]
        assert np.isscalar(Fourier_error_ref) and Fourier_error_ref >= 0

        vals, Fourier_error = u_true_at_mu(mu_eval, tau_eval, phi_eval, tau_derivative_order, True)
        assert np.allclose(vals, u_true_at_mu(mu_eval, tau_eval, phi_eval, tau_derivative_order))
        assert np.isclose(Fourier_error, Fourier_error_ref)


def test_interp_NT_singularity_continuity():
    """The NT corrections' removable singularities at mu = -mu0 (TMS) and at the
    IMS-scaled -mu0 are handled exactly by the phi-kernel formulations: evaluating
    exactly AT the singular angles must give finite values matching the limit of
    nearby evaluations to increasing accuracy."""
    u_interpol_eval = _pydisort(True, interpolate_mu=True)[4]

    # An IMS-scaled mu0: the whole-atmosphere average below equals the BOTTOM-boundary
    # set of `pydisort`'s per-boundary NT statics (which weight by layer optical
    # thickness following DISORT's `SECSCA`), so it is a removable-singularity candidate
    # for evaluation points in the last layer
    thickness_arr = np.diff(tau_arr, prepend=0)
    sum1 = np.sum(omega_arr * thickness_arr)
    omega_avg = sum1 / tau_arr[-1]
    sum2 = np.sum(f_arr * omega_arr * thickness_arr)
    f_avg = sum2 / sum1
    scaled_mu0 = mu0 / (1 - omega_avg * f_avg)

    for mu_sing in (-mu0, -scaled_mu0):
        if mu_sing < -1:
            continue
        vals_at = u_interpol_eval(np.array([mu_sing]), tau_eval, phi_eval)
        assert np.all(np.isfinite(vals_at))
        scale = np.max(np.abs(vals_at))
        for eps, rtol in ((1e-6, 1e-3), (1e-9, 1e-6), (1e-12, 1e-9)):
            vals_near = u_interpol_eval(np.array([mu_sing * (1 + eps)]), tau_eval, phi_eval)
            assert np.allclose(vals_near, vals_at, rtol=0, atol=rtol * scale)


def test_NT_tau_continuity_at_layer_boundaries():
    """The NT-corrected intensity must be continuous in `tau` across layer boundaries
    of an inhomogeneous atmosphere. This exercises the IMS boundary-blending scheme
    (the correction's per-boundary optical-property averages follow DISORT's `SECSCA`
    at the layer boundaries and are blended/interpolated linearly in `tau` within each layer)."""
    u_interpol_eval = _pydisort(True, interpolate_mu=True)[4]

    scale = np.max(np.abs(u_interpol_eval(mu_eval, tau_eval, phi_eval)))
    eps = 1e-11
    for tau_b in tau_arr[:-1]:
        above = u_interpol_eval(mu_eval, np.array([tau_b - eps]), phi_eval)
        below = u_interpol_eval(mu_eval, np.array([tau_b + eps]), phi_eval)
        assert np.allclose(above, below, rtol=0, atol=1e-8 * scale)


def test_NT_mu0_at_quadrature_node():
    """`pydisort` no longer forbids `mu0` from coinciding with a quadrature node when
    `NT_cor=True`; the corrected intensity must be finite there and continuous in `mu0`."""
    mu_q = PythonicDISORT.subroutines.Gauss_Legendre_quad(NQuad // 2)[0]
    mu0_sing = mu_q[2]

    def solve(m0):
        return PythonicDISORT.pydisort(
            tau_arr, omega_arr,
            NQuad,
            Leg_coeffs_all,
            m0, I0, phi0,
            f_arr=f_arr,
            NT_cor=True,
        )[4]

    u_at = solve(mu0_sing)(tau_eval, phi_eval)
    assert np.all(np.isfinite(u_at))
    u_near = solve(mu0_sing * (1 + 1e-9))(tau_eval, phi_eval)
    assert np.allclose(u_near, u_at, rtol=0, atol=1e-6 * np.max(np.abs(u_at)))

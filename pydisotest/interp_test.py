import numpy as np
from scipy.integrate import quad_vec
import PythonicDISORT
from PythonicDISORT.subroutines import interpolate
from math import pi
import pytest

# ============================================================================================================
# Test Problem interp: `subroutines.interpolate` and its NT corrections options (Addresses GitHub issue #47)
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

'''
def test_interp_NT_quad_equivalence():
    """`interpolate(u, NT_cor=True)` on an uncorrected `u` must reproduce
    the interpolation of a `pydisort` run with `NT_cor=True`."""
    u_true = _pydisort(True)[4]
    u_false = _pydisort(False)[4]

    u_interpol_ref = interpolate(u_true)
    u_interpol_quad = interpolate(u_false, NT_cor=True)

    for is_anti in (False, True):
        ref = u_interpol_ref(mu_eval, tau_eval, phi_eval, is_anti)
        res = u_interpol_quad(mu_eval, tau_eval, phi_eval, is_anti)
        assert np.allclose(res, ref)

    # The string flag is equivalent to the boolean flag
    u_interpol_quad_str = interpolate(u_false, NT_cor="quad")
    assert np.allclose(
        u_interpol_quad_str(mu_eval, tau_eval, phi_eval),
        u_interpol_ref(mu_eval, tau_eval, phi_eval),
    )

    # `NT_cor=True` on an already-corrected `u` is a no-op
    assert np.allclose(
        interpolate(u_true, NT_cor=True)(mu_eval, tau_eval, phi_eval),
        u_interpol_ref(mu_eval, tau_eval, phi_eval),
    )


def test_interp_NT_off_equivalence():
    """`interpolate(u, NT_cor=False)` on a corrected `u` must reproduce
    the interpolation of a `pydisort` run with `NT_cor=False`."""
    u_true = _pydisort(True)[4]
    u_false = _pydisort(False)[4]

    u_interpol_ref = interpolate(u_false)
    u_interpol_off = interpolate(u_true, NT_cor=False)

    for is_anti in (False, True):
        ref = u_interpol_ref(mu_eval, tau_eval, phi_eval, is_anti)
        res = u_interpol_off(mu_eval, tau_eval, phi_eval, is_anti)
        assert np.allclose(res, ref)

    assert np.allclose(
        interpolate(u_true, NT_cor="off")(mu_eval, tau_eval, phi_eval),
        u_interpol_ref(mu_eval, tau_eval, phi_eval),
    )


def test_interp_NT_eval_at_quad_nodes():
    """At the quadrature nodes the barycentric interpolation is exact, so the
    "eval" option must agree with the "quad" option (and with the corrected `u` itself)."""
    mu_arr, _, _, _, u_true = _pydisort(True)
    u_false = _pydisort(False)[4]

    u_interpol_eval = interpolate(u_false, NT_cor="eval")

    for is_anti in (False, True):
        ref = u_true(tau_eval, phi_eval, is_anti)
        res = u_interpol_eval(mu_arr, tau_eval, phi_eval, is_anti)
        assert np.allclose(res, ref)
'''


def test_interp_NT_eval_more_accurate_than_quad():
    """The corrections at evaluation points should be more accurate than interpolating
    the corrections from the quadrature points (the motivation for GitHub issue #47).
    A well-resolved NQuad=32 run serves as the reference."""
    u_false = _pydisort(False)[4]
    u_interpol_quad = interpolate(u_false, NT_cor="quad")
    u_interpol_eval = interpolate(u_false, NT_cor="eval")

    Leg_coeffs_more = g_per_layer[:, None] ** np.arange(64)[None, :]
    u_hires = PythonicDISORT.pydisort(
        tau_arr, omega_arr,
        32,
        Leg_coeffs_more,
        mu0, I0, phi0,
        f_arr=Leg_coeffs_more[:, 32],
        NT_cor=True,
    )[4]
    ref = interpolate(u_hires)(mu_eval, tau_eval, phi_eval)

    err_quad = np.max(np.abs(u_interpol_quad(mu_eval, tau_eval, phi_eval) - ref))
    err_eval = np.max(np.abs(u_interpol_eval(mu_eval, tau_eval, phi_eval) - ref))
    assert err_eval < err_quad


def test_interp_NT_antiderivative():
    """The antiderivative branch of the corrections at evaluation points must
    integrate correctly (same idiom as the `I_test.py` problems)."""
    u_false = _pydisort(False)[4]
    u_interpol_eval = interpolate(u_false, NT_cor="eval")
    u_interpol_quad = interpolate(u_false, NT_cor="quad")

    end = tau_arr[-1]
    upper = tau_arr.astype(float)
    lower = np.insert(upper[:-1] + 1e-15, 0, 0)

    for u_interpol in (u_interpol_eval, u_interpol_quad):
        integral = quad_vec(lambda tau: u_interpol(mu_eval, tau, phi_eval), 0, end)[0]
        telescoped = (
            u_interpol(mu_eval, upper, phi_eval, True)
            - u_interpol(mu_eval, lower, phi_eval, True)
        ).sum(axis=1)
        assert np.allclose(integral, telescoped)


def test_interp_NT_output_passthrough():
    """`return_Fourier_error` and `return_tau_arr` must pass through with the NT options."""
    u_false = _pydisort(False)[4]
    u_interpol_eval = interpolate(u_false, NT_cor="eval")

    vals_plain = u_interpol_eval(mu_eval, tau_eval, phi_eval)
    vals, Fourier_error, tau_arr_out = u_interpol_eval(
        mu_eval, tau_eval, phi_eval, False, True, True
    )
    assert np.allclose(vals, vals_plain)
    assert np.isscalar(Fourier_error) and Fourier_error >= 0
    assert np.array_equal(tau_arr_out, tau_arr)


def test_interp_NT_reduces_to_off_when_unavailable():
    """Without delta-M scaling (`f_arr=0`) the corrections are not applicable and all the
    NT options must reduce to "off". The string options ("quad" / "eval") must warn when
    this happens; the boolean options must stay silent, mirroring `pydisort`'s behavior."""
    import warnings

    u_nof = PythonicDISORT.pydisort(
        tau_arr, omega_arr,
        NQuad,
        Leg_coeffs_all[:, :NQuad],
        mu0, I0, phi0,
    )[4]
    ref = interpolate(u_nof)(mu_eval, tau_eval, phi_eval)
    for flag, should_warn in ((True, False), (False, False), ("off", False),
                              ("quad", True), ("eval", True)):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            u_interpol = interpolate(u_nof, NT_cor=flag)
        assert len(caught) == (1 if should_warn else 0), (flag, [str(c.message) for c in caught])
        assert np.allclose(u_interpol(mu_eval, tau_eval, phi_eval), ref)


'''
def test_interp_NT_input_checks():
    u_false = _pydisort(False)[4]
    u0 = _pydisort(False)[3]

    # Invalid flag value
    with pytest.raises(ValueError):
        interpolate(u_false, NT_cor="nonsense")

    # `u0` does not support the NT options
    with pytest.raises(ValueError):
        interpolate(u0, NT_cor=True)

    # The corrections at evaluation points require |mu| >= 1e-8 ...
    u_interpol_eval = interpolate(u_false, NT_cor="eval")
    with pytest.raises(ValueError):
        u_interpol_eval(np.array([0.5, 0]), tau_eval, phi_eval)
    # ... and `tau` within the atmosphere
    with pytest.raises(ValueError):
        u_interpol_eval(mu_eval, tau_arr[-1] + 1, phi_eval)
'''


def test_interp_NT_singularity_continuity():
    """The NT corrections' removable singularities at mu = -mu0 (TMS) and at the
    IMS-scaled -mu0 are handled exactly by the phi-kernel formulations: evaluating
    exactly AT the singular angles must give finite values matching the limit of
    nearby evaluations to increasing accuracy."""
    u_false = _pydisort(False)[4]
    u_interpol_eval = interpolate(u_false, NT_cor="eval")

    # The IMS-scaled mu0 (must mirror the formula in `pydisort`'s NT statics,
    # which weights by layer optical thickness following DISORT's `SECSCA`)
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

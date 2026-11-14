import numpy as np
from scipy.integrate import quad_vec
import PythonicDISORT
from PythonicDISORT.subroutines import (
    generate_s_poly_coeffs,
    blackbody_contrib_to_BCs,
    cache_BDRF_Fourier_modes,
    generate_diff_act_flux_funcs,
)
from math import pi
import pytest

# ============================================================================================================
# Test Problem IotS: intensities at arbitrary polar angles by the integration of the source function,
# i.e. the intensity functions of `pydisort` when `interpolate_mu=True`
# ============================================================================================================
# This subsumes test problem 10 of Stamnes' DISORT (`disotest.f90`), see `test_IotS_reproduces_quadrature_nodes`

rng = np.random.default_rng(0) # Seed the RNG
NLayers = 5
tau_arr = np.cumsum(rng.uniform(0.1, 2, NLayers))
omega_arr = rng.uniform(0.5, 0.99, NLayers)
omega_arr[2] = 0  # Non-scattering layer (no-scattering shortcut for every Fourier mode)
Leg_coeffs_all = rng.uniform(0.6, 0.9, NLayers)[:, None] ** np.arange(64)[None, :]
Leg_coeffs_all[3, 1:] = 0
Leg_coeffs_all[3, 2] = 0.1  # Rayleigh layer (no-scattering shortcut for Fourier modes >= 3)
NQuad = 16
b_neg = 0.1

mu_eval = np.array([-0.93, -0.45, -0.07, 0.05, 0.33, 0.97])
tau_eval = np.concatenate([[0], tau_arr, rng.uniform(0, tau_arr[-1], 6)])
phi_eval = np.array([0.0, 0.7, 2.5])


def BDRF_mode0(mu, neg_mup):
    return 0.1 + 0.05 * np.outer(mu, neg_mup)


def _pydisort(NQuad=NQuad, tau_arr=tau_arr, **kwargs):
    kwargs.setdefault("BDRF_Fourier_modes", [BDRF_mode0, 0.02])
    return PythonicDISORT.pydisort(
        tau_arr, omega_arr,
        NQuad,
        Leg_coeffs_all,
        0.45, 1.3, 1.0,
        f_arr=Leg_coeffs_all[:, NQuad],
        s_poly_coeffs=np.array([[1.0 + l, 0.3, -0.05] for l in range(NLayers)]),
        b_pos=0.3,
        b_neg=b_neg,
        **kwargs,
    )


def test_IotS_reproduces_quadrature_nodes():
    # At the quadrature nodes the integration of the source function reproduces the discrete ordinates solution,
    # and the NT corrections at those `mu` values are those at the quadrature nodes.
    # The same holds for the antiderivative and the derivative with respect to tau, and the fluxes are unaffected.
    # The second configuration has the features of test problem 10 of Stamnes' DISORT, which compares
    # USRANG = TRUE at the quadrature nodes against USRANG = FALSE: 4 streams, thick layers and a Lambertian surface.
    print()
    configs = ({}, {"NQuad": 4, "tau_arr": 5 * tau_arr, "BDRF_Fourier_modes": [0.5]})
    for config in configs:
        tau_eval_config = tau_eval * config.get("tau_arr", tau_arr)[-1] / tau_arr[-1]
        for NT_cor in (False, True):
            mu_arr, flux_up_nodes, flux_down_nodes, u0_nodes, u_nodes = _pydisort(NT_cor=NT_cor, **config)
            _, flux_up, flux_down, u0, u = _pydisort(NT_cor=NT_cor, interpolate_mu=True, **config)
            for tau_derivative_order in (0, -1, 1):
                ref = u_nodes(tau_eval_config, phi_eval, tau_derivative_order)
                res = u(mu_arr, tau_eval_config, phi_eval, tau_derivative_order)
                assert np.max(np.abs(res - ref)) < 1e-12 * np.max(np.abs(ref))
                ref0 = u0_nodes(tau_eval_config, tau_derivative_order)
                res0 = u0(mu_arr, tau_eval_config, tau_derivative_order)
                assert np.max(np.abs(res0 - ref0)) < 1e-12 * np.max(np.abs(ref0))
                assert np.array_equal(
                    flux_up(tau_eval_config, tau_derivative_order),
                    flux_up_nodes(tau_eval_config, tau_derivative_order),
                )
                assert np.array_equal(
                    flux_down(tau_eval_config, tau_derivative_order),
                    flux_down_nodes(tau_eval_config, tau_derivative_order),
                )


def test_IotS_boundary_conditions():
    # The boundary conditions hold at every `mu`, and `mu = 0` is the limit from below
    print()
    _, _, _, u0, u = _pydisort(interpolate_mu=True)
    mu_down = -np.linspace(0, 1, 9)
    assert np.allclose(u0(mu_down, 0), b_neg, rtol=1e-13, atol=0)
    assert np.allclose(u(mu_down, 0, phi_eval), b_neg, rtol=1e-13, atol=0)
    vals = u(np.array([0.0, -1e-12]), tau_eval[1:], phi_eval)
    assert np.allclose(vals[0], vals[1], rtol=1e-9, atol=0)


def test_IotS_antiderivative():
    # The tau-integral over each layer (telescoping idiom) against adaptive quadrature, at arbitrary mu
    print()
    for NT_cor in (False, True):
        u = _pydisort(NT_cor=NT_cor, interpolate_mu=True)[-1]
        phi = np.array([0.0, 1.3])
        tops = np.concatenate([[0], np.nextafter(tau_arr[:-1], np.inf)])
        for top, bottom in zip(tops, tau_arr):
            integral = u(mu_eval, bottom, phi, -1) - u(mu_eval, top, phi, -1)
            quad = quad_vec(lambda t: u(mu_eval, t, phi), top, bottom, epsabs=1e-13, epsrel=1e-12)[0]
            assert np.max(np.abs(integral - quad)) < 1e-9 * np.max(np.abs(quad))


def test_IotS_derivative():
    # The tau-derivative against central finite differences of the intensity, at arbitrary mu
    print()
    tau_interior = np.random.default_rng(1).uniform(0.01, tau_arr[-1] - 0.01, 12)
    h = 1e-5
    for NT_cor in (False, True):
        _, _, _, u0, u = _pydisort(NT_cor=NT_cor, interpolate_mu=True)
        derivative = u(mu_eval, tau_interior, phi_eval, 1)
        finite_diff = (u(mu_eval, tau_interior + h, phi_eval) - u(mu_eval, tau_interior - h, phi_eval)) / (2 * h)
        assert np.max(np.abs(derivative - finite_diff)) < 1e-7 * np.max(np.abs(derivative))
        derivative = u0(mu_eval, tau_interior, 1)
        finite_diff = (u0(mu_eval, tau_interior + h) - u0(mu_eval, tau_interior - h)) / (2 * h)
        assert np.max(np.abs(derivative - finite_diff)) < 1e-7 * np.max(np.abs(derivative))

'''
def test_IotS_call_patterns():
    # The values do not depend on which other points are evaluated in the same call or in earlier calls
    print()
    for NT_cor in (False, True):
        u = _pydisort(NT_cor=NT_cor, interpolate_mu=True)[-1]
        ref = _pydisort(NT_cor=NT_cor, interpolate_mu=True)[-1](mu_eval, tau_eval, phi_eval)
        scale = np.max(np.abs(ref))
        # One `tau` at a time, the top of the atmosphere first, then one `mu` at a time
        for j in range(len(tau_eval)):
            assert np.max(np.abs(u(mu_eval, tau_eval[j], phi_eval) - ref[:, j, :])) < 1e-12 * scale
        for i in range(len(mu_eval)):
            assert np.max(np.abs(u(mu_eval[i], tau_eval, phi_eval) - ref[i])) < 1e-12 * scale
        assert np.isclose(u(mu_eval[1], tau_eval[2], phi_eval[1]), ref[1, 2, 1], rtol=1e-12, atol=0)


def test_IotS_u0_and_only_flux():
    # `u0` is the azimuthal average of `u`; with `only_flux=True` it is still returned, and is the same
    print()
    mu_arr, flux_up, flux_down, u0, u = _pydisort(interpolate_mu=True)
    outputs = _pydisort(interpolate_mu=True, only_flux=True)
    assert len(outputs) == 4
    Nphi = 33
    phi = 2 * pi * np.arange(Nphi) / Nphi
    for tau_derivative_order in (0, -1, 1):
        ref = u0(mu_eval, tau_eval, tau_derivative_order)
        scale = np.max(np.abs(ref))
        average = np.mean(u(mu_eval, tau_eval, phi, tau_derivative_order), axis=-1)
        assert np.max(np.abs(average - ref)) < 1e-12 * scale
        assert np.max(np.abs(outputs[3](mu_eval, tau_eval, tau_derivative_order) - ref)) < 1e-12 * scale

    # The actinic fluxes are computed from `u0` at the quadrature nodes whether or not `interpolate_mu=True`
    u0_nodes = _pydisort()[3]
    for Fa, Fa_ref in zip(generate_diff_act_flux_funcs(u0), generate_diff_act_flux_funcs(u0_nodes)):
        assert np.array_equal(Fa(tau_eval), Fa_ref(tau_eval))
'''

def test_IotS_cached_BDRF():
    # Cached BDRF Fourier modes are evaluated at arbitrary `mu` like the modes that were cached
    print()

    def BDRF_mode1(mu, neg_mup):
        return 0.03 * np.outer(np.sqrt(1 - mu**2), np.sqrt(1 - neg_mup**2))

    BDRF_Fourier_modes = [BDRF_mode0, BDRF_mode1]
    ref = _pydisort(interpolate_mu=True, BDRF_Fourier_modes=BDRF_Fourier_modes)[-1](mu_eval, tau_eval, phi_eval)
    for mu0 in (0.45, 0):
        cached = cache_BDRF_Fourier_modes(NQuad // 2, BDRF_Fourier_modes, mu0)
        res = _pydisort(interpolate_mu=True, BDRF_Fourier_modes=cached)[-1](mu_eval, tau_eval, phi_eval)
        assert np.max(np.abs(res - ref)) < 1e-12 * np.max(np.abs(ref))


def test_IotS_Fourier_error_and_input_checks():
    print()
    u = _pydisort(interpolate_mu=True)[-1]
    for tau_derivative_order in (0, -1, 1):
        vals, Fourier_error = u(mu_eval, tau_eval, phi_eval, tau_derivative_order, True)
        assert np.array_equal(vals, u(mu_eval, tau_eval, phi_eval, tau_derivative_order))
        assert np.isscalar(Fourier_error) and Fourier_error >= 0
    # Fourier modes >= 1 vanish at mu = +-1
    assert u(np.array([1.0, -1.0]), tau_eval, phi_eval, 0, True)[1] < 1e-12

    with pytest.raises(ValueError):
        u(1.1, 0, 0)
    with pytest.raises(ValueError):
        u(0.5, tau_arr[-1] * 1.1, 0)
    # The tau-derivative requires `mu` values bounded away from 0
    with pytest.raises(ValueError):
        u(0.0, 0.5, 0, 1)
    # `True` was the antiderivative of the boolean flag which `tau_derivative_order` replaced; it must not be read as 1
    with pytest.raises(ValueError):
        u(0.5, 0.5, 0, True)
    with pytest.raises(ValueError):
        u(0.5, 0.5, 0, 2)


def test_IotS_vs_Stamnes_DISORT():
    # Stamnes' DISORT with USRANG = TRUE (single precision), delta-M scaling and NT corrections.
    # Stamnes' DISORT applies the IMS correction only within 10 degrees of the polar angle of the beam, whereas
    # PythonicDISORT applies it to every downward direction; the largest difference is just outside that window.
    print()
    results = np.load("Stamnes_results/IotS_test.npz")
    umu, phi = results["umu"], results["phi"]

    # One layer, Lambertian surface, direct beam
    Leg_coeffs = 0.75 ** np.arange(32)
    u = PythonicDISORT.pydisort(
        1.0, 0.9, 16, Leg_coeffs, 0.6, pi, 0.4,
        f_arr=Leg_coeffs[16],
        NT_cor=True,
        BDRF_Fourier_modes=[0.2],
        interpolate_mu=True,
    )[-1]
    uu = results["uuA"]
    I = u(umu, results["utauA"], phi)
    print("One layer: max difference / max intensity =", np.max(np.abs(I - uu)) / np.max(np.abs(uu)))
    assert np.max(np.abs(I - uu)) < 1e-4 * np.max(np.abs(uu))

    # Three layers, thermal emission, direct beam, Lambertian surface
    tau = np.array([0.5, 1.5, 4.0])
    moments = np.array([g ** np.arange(32) for g in (0.2, 0.7, 0.5)])
    u = PythonicDISORT.pydisort(
        tau, np.array([0.3, 0.8, 0.6]), 16, moments, 0.7, 50.0, 0.0,
        f_arr=moments[:, 16],
        NT_cor=True,
        BDRF_Fourier_modes=[0.1],
        s_poly_coeffs=generate_s_poly_coeffs(tau, np.array([220.0, 250.0, 270.0, 290.0]), 500.0, 700.0),
        b_pos=0.9 * blackbody_contrib_to_BCs(300.0, 500.0, 700.0),
        interpolate_mu=True,
    )[-1]
    uu = results["uuC"]
    I = u(umu, results["utauC"], phi)
    print("Three layers: max difference / max intensity =", np.max(np.abs(I - uu)) / np.max(np.abs(uu)))
    assert np.max(np.abs(I - uu)) < 1e-4 * np.max(np.abs(uu))

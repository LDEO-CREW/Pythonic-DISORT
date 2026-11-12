import numpy as np
from scipy.integrate import quad_vec
import PythonicDISORT
from PythonicDISORT.subroutines import interpolate, generate_s_poly_coeffs, blackbody_contrib_to_BCs
from math import pi

# ============================================================================================================
# Test Problem IotS: intensities at arbitrary polar angles by the integration of the source function,
# i.e. `subroutines.interpolate` applied to the outputs of `pydisort`
# ============================================================================================================

rng = np.random.default_rng(0)
NLayers = 5
tau_arr = np.cumsum(rng.uniform(0.1, 2, NLayers))
omega_arr = rng.uniform(0.5, 0.99, NLayers)
omega_arr[2] = 0  # Non-scattering layer (no-scattering shortcut for every Fourier mode)
Leg_coeffs_all = rng.uniform(0.6, 0.9, NLayers)[:, None] ** np.arange(64)[None, :]
Leg_coeffs_all[3, 1:] = 0
Leg_coeffs_all[3, 2] = 0.1  # Rayleigh layer (no-scattering shortcut for Fourier modes >= 3)
NQuad = 16


def BDRF_mode0(mu, neg_mup):
    return 0.1 + 0.05 * np.outer(mu, neg_mup)


def _pydisort(**kwargs):
    return PythonicDISORT.pydisort(
        tau_arr, omega_arr,
        NQuad,
        Leg_coeffs_all,
        0.45, 1.3, 1.0,
        f_arr=Leg_coeffs_all[:, NQuad],
        BDRF_Fourier_modes=[BDRF_mode0, 0.02],
        s_poly_coeffs=np.array([[1.0 + l, 0.3, -0.05] for l in range(NLayers)]),
        b_pos=0.3,
        b_neg=0.1,
        **kwargs,
    )


def test_IotS_reproduces_quadrature_nodes():
    # At the quadrature nodes the integration of the source function reproduces the discrete ordinates solution
    print()
    mu_arr, _, _, u0, u = _pydisort(NT_cor=True)
    u_star = u._NT_data["u_star"]
    tau = np.concatenate([[0], tau_arr, rng.uniform(0, tau_arr[-1], 6)])
    phi = np.array([0.0, 0.7, 2.5])
    I = interpolate(u, NT_cor=False)
    I_u0 = interpolate(u0)
    for is_antiderivative_wrt_tau in (False, True):
        ref = u_star(tau, phi, is_antiderivative_wrt_tau)
        assert np.max(np.abs(I(mu_arr, tau, phi, is_antiderivative_wrt_tau) - ref)) < 1e-12 * np.max(np.abs(ref))
        ref0 = u0(tau, is_antiderivative_wrt_tau)
        assert np.max(np.abs(I_u0(mu_arr, tau, is_antiderivative_wrt_tau) - ref0)) < 1e-12 * np.max(np.abs(ref0))


def test_IotS_antiderivative():
    # The tau-integral over each layer (telescoping idiom) against adaptive quadrature, at arbitrary mu
    print()
    u = _pydisort(NT_cor=True)[-1]
    mu = np.array([-0.93, -0.45, -0.07, 0.05, 0.33, 0.97])
    phi = np.array([0.0, 1.3])
    for NT_cor in (False, "eval"):
        I = interpolate(u, NT_cor=NT_cor)
        tops = np.concatenate([[0], np.nextafter(tau_arr[:-1], np.inf)])
        for top, bottom in zip(tops, tau_arr):
            integral = I(mu, bottom, phi, True) - I(mu, top, phi, True)
            quad = quad_vec(lambda t: I(mu, t, phi), top, bottom, epsabs=1e-13, epsrel=1e-12)[0]
            assert np.max(np.abs(integral - quad)) < 1e-9 * np.max(np.abs(quad))


def test_IotS_vs_Stamnes_DISORT():
    # Stamnes' DISORT with USRANG = TRUE (single precision); no delta-M scaling and so no NT corrections
    print()
    results = np.load("Stamnes_results/IotS_test.npz")
    umu, phi = results["umu"], results["phi"]

    # One layer, Lambertian surface, direct beam
    u = PythonicDISORT.pydisort(
        1.0, 0.9, 16, 0.75 ** np.arange(16), 0.6, pi, 0.4, BDRF_Fourier_modes=[0.2]
    )[-1]
    uu = results["uuA"]
    I = interpolate(u)(umu, results["utauA"], phi)
    print("One layer: max difference / max intensity =", np.max(np.abs(I - uu)) / np.max(np.abs(uu)))
    assert np.max(np.abs(I - uu)) < 1e-4 * np.max(np.abs(uu))

    # Three layers, thermal emission, direct beam, Lambertian surface
    tau = np.array([0.5, 1.5, 4.0])
    moments = np.array([g ** np.arange(16) for g in (0.2, 0.7, 0.5)])
    u = PythonicDISORT.pydisort(
        tau, np.array([0.3, 0.8, 0.6]), 16, moments, 0.7, 50.0, 0.0,
        BDRF_Fourier_modes=[0.1],
        s_poly_coeffs=generate_s_poly_coeffs(tau, np.array([220.0, 250.0, 270.0, 290.0]), 500.0, 700.0),
        b_pos=0.9 * blackbody_contrib_to_BCs(300.0, 500.0, 700.0),
    )[-1]
    uu = results["uuC"]
    I = interpolate(u)(umu, results["utauC"], phi)
    print("Three layers: max difference / max intensity =", np.max(np.abs(I - uu)) / np.max(np.abs(uu)))
    assert np.max(np.abs(I - uu)) < 1e-4 * np.max(np.abs(uu))

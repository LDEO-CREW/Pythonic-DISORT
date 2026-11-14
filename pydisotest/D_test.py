import numpy as np
import PythonicDISORT
from PythonicDISORT.subroutines import generate_diff_act_flux_funcs
from math import pi
import pytest

# ======================================================================================================
# Test Problem D: Derivative functionality (`tau_derivative_order = 1`)
# ======================================================================================================

# Inhomogeneous 3-layer atmosphere; the optical parameters are identical to those in test Id
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
s_poly_coeffs = np.array([[1, 2, 3], [2, 1, 4], [3, 2, 1]])

# Test points, none of which is on a layer boundary
tau_test_arr = np.array([0.02, 0.7, 1.95, 2.6, 4.1, 5.3, 6.6, 7.97])
phi_arr = np.array([0, 1.1, pi, 5])


def finite_difference(func, tau, h=1e-4):
    """Fourth-order central finite difference approximation of the derivative of `func` at `tau`."""
    return (-func(tau + 2 * h) + 8 * func(tau + h) - 8 * func(tau - h) + func(tau - 2 * h)) / (12 * h)


def compare_derivatives(flux_up, flux_down, u0, u):
    """Compares the tau-derivative of each output function against finite differences of that function."""
    for func in (
        flux_up,
        lambda tau, *args: flux_down(tau, *args)[0],
        lambda tau, *args: flux_down(tau, *args)[1],
        u0,
        lambda tau, *args: u(tau, phi_arr, *args),
        *generate_diff_act_flux_funcs(u0),
    ):
        derivative = func(tau_test_arr, 1)
        assert np.allclose(derivative, finite_difference(func, tau_test_arr), rtol=1e-6, atol=1e-8 * np.max(np.abs(derivative)))


def test_Da():
    print()
    print("################################################ Test Da ##################################################")
    print("########## Test differentiation with a beam source, delta-M scaling, NT corrections and BDRF ##############")
    print()

    for s_poly_coeffs_test in (np.array([[]]), s_poly_coeffs[:, :1], s_poly_coeffs[:, :2], s_poly_coeffs):
        outputs = PythonicDISORT.pydisort(
            tau_arr, omega_arr,
            NQuad,
            Leg_coeffs_all,
            mu0, I0, phi0,
            b_pos=1,
            b_neg=1,
            f_arr=f_arr,
            BDRF_Fourier_modes=[1],
            s_poly_coeffs=s_poly_coeffs_test,
            NT_cor=True,
        )[1:]
        compare_derivatives(*outputs)


def test_Db():
    print()
    print("################################################ Test Db ##################################################")
    print("################ Test differentiation without a beam source (boundary + isotropic sources) ################")
    print()

    outputs = PythonicDISORT.pydisort(
        tau_arr, omega_arr,
        NQuad,
        Leg_coeffs_all[:, :NQuad],
        0, 0, 0,
        b_pos=1 / pi,
        b_neg=1 / pi,
        s_poly_coeffs=s_poly_coeffs[:, :2],
    )[1:]
    compare_derivatives(*outputs)


def test_Dc():
    print()
    print("################################################ Test Dc ##################################################")
    print("################# Test the derivative of the net flux against the flux divergence identity ################")
    print()

    # Integrating the radiative transfer equation over all directions gives
    #     d(net upward flux) / d(tau) = (1 - omega) * (total actinic flux - 4 * pi * isotropic source),
    # whose right-hand side requires no derivative. The identity holds with and without delta-M scaling.
    tau = np.concatenate([[0], tau_test_arr, tau_arr])
    l = np.searchsorted(tau_arr, tau)
    for f_arr_test in (0, f_arr):
        _, flux_up, flux_down, u0 = PythonicDISORT.pydisort(
            tau_arr, omega_arr,
            NQuad,
            Leg_coeffs_all,
            mu0, I0, phi0,
            b_pos=1,
            b_neg=1,
            f_arr=f_arr_test,
            BDRF_Fourier_modes=[0.3],
            s_poly_coeffs=s_poly_coeffs,
            only_flux=True,
        )
        Fp_act, Fm_act = generate_diff_act_flux_funcs(u0)
        direct_act_flux = flux_down(tau)[1] / mu0
        iso_source = np.sum(s_poly_coeffs[l, :] * tau[:, None] ** np.arange(3)[None, :], axis=1)

        net_flux_derivative = flux_up(tau, 1) - np.sum(flux_down(tau, 1), axis=0)
        flux_divergence = (1 - omega_arr[l]) * (Fp_act(tau) + Fm_act(tau) + direct_act_flux - 4 * pi * iso_source)
        assert np.allclose(net_flux_derivative, flux_divergence, rtol=1e-11, atol=0)


def test_Dd():
    print()
    print("################################################ Test Dd ##################################################")
    print("######################## Test that the derivative inverts the antiderivative ##############################")
    print()

    mu_arr, flux_up, flux_down, u0, u = PythonicDISORT.pydisort(
        tau_arr, omega_arr,
        NQuad,
        Leg_coeffs_all,
        mu0, I0, phi0,
        b_pos=1,
        b_neg=1,
        f_arr=f_arr,
        BDRF_Fourier_modes=[1],
        s_poly_coeffs=s_poly_coeffs,
        NT_cor=True,
    )
    for func in (
        flux_up,
        lambda tau, *args: flux_down(tau, *args)[0],
        lambda tau, *args: flux_down(tau, *args)[1],
        u0,
        lambda tau, *args: u(tau, phi_arr, *args),
    ):
        antiderivative = lambda tau: func(tau, -1)
        assert np.allclose(finite_difference(antiderivative, tau_test_arr), func(tau_test_arr), rtol=1e-6)

    # The boolean flag which `tau_derivative_order` replaced is rejected rather than read as the order 1
    with pytest.raises(ValueError):
        flux_up(tau_test_arr, True)
    with pytest.raises(ValueError):
        u(tau_test_arr, phi_arr, 2)

import numpy as np
from scipy.integrate import quad, quad_vec
import PythonicDISORT
from PythonicDISORT.subroutines import cache_BDRF_Fourier_modes
from math import pi

# ======================================================================================================
# Test Problem I: Antiderivative / integration functionality
# ======================================================================================================

def test_Ia():
    print()
    print("################################################ Test Ia ##################################################")
    print("####################### Test integration when s(tau) (isotropic source) is constant #######################")
    print()
    
    # Except for `s_poly_coeffs`, the optical parameters are identical to those in test 11a
    
    ######################################### PYDISORT ARGUMENTS #######################################
    tau_arr = np.array([8/3, 16/3, 8]) # 3-layer homogeneous split
    NLayers = len(tau_arr)
    omega_arr = np.full(NLayers, 0.8)
    NQuad = 16
    Leg_coeffs_all = np.tile(0.75 ** np.arange(32), (NLayers, 1))
    mu0 = 0.6
    I0 = pi / mu0
    phi0 = 0.9 * pi

    # Optional (used)
    f_arr = np.repeat(Leg_coeffs_all[0, NQuad], NLayers)
    NT_cor = True
    b_neg=1
    b_pos=1
    BDRF_Fourier_modes=[1]
    s_poly_coeffs=np.tile(np.arange(4) + 1, (NLayers, 1))

    # Optional (unused)
    NLeg=None
    NFourier=None
    only_flux=False
    use_banded_solver_NLayers=10
    autograd_compatible=False

    ####################################################################################################

    # Test points
    Nphi = int((NQuad * pi) // 2) * 2 + 1
    phi_arr, full_weights_phi = PythonicDISORT.subroutines.Clenshaw_Curtis_quad(Nphi)

    # Call pydisort function
    flux_up, flux_down, u0, u = PythonicDISORT.pydisort(
        tau_arr, omega_arr,
        NQuad,
        Leg_coeffs_all,
        mu0, I0, phi0,
        b_pos=b_pos,
        b_neg=b_neg,
        f_arr=f_arr,
        BDRF_Fourier_modes=BDRF_Fourier_modes,
        s_poly_coeffs=s_poly_coeffs[:, :1],
        NT_cor=True,
    )[1:]
    
    end = tau_arr[-1]
    # Multi-layer integration via the documented antiderivative idiom (see `pydisort.py` docstring):
    # per-layer local antiderivatives telescope when sampled at each layer bottom minus each layer top (+eps to ensure correct layer).
    upper = tau_arr
    lower = np.insert(tau_arr[:-1] + 1e-15, 0, 0)
    assert np.allclose(quad_vec(lambda tau: u(tau, phi_arr), 0, end)[0], (u(upper, phi_arr, True) - u(lower, phi_arr, True)).sum(axis=1))
    assert np.allclose(quad_vec(u0, 0, end)[0], (u0(upper, True) - u0(lower, True)).sum(axis=-1))
    assert np.allclose(quad(flux_up, 0, end)[0], (flux_up(upper, True) - flux_up(lower, True)).sum())
    assert np.allclose(quad(lambda tau: flux_down(tau)[0], 0, end)[0], (flux_down(upper, True)[0] - flux_down(lower, True)[0]).sum())
    assert np.allclose(quad(lambda tau: flux_down(tau)[1], 0, end)[0], (flux_down(upper, True)[1] - flux_down(lower, True)[1]).sum())
    # ------------------------------------------------------------------------------------------------------------------


def test_Ib():
    print()
    print("################################################ Test Ib ##################################################")
    print("######################## Test integration when s(tau) (isotropic source) is linear ########################")
    print()
     
    ######################################### PYDISORT ARGUMENTS #######################################
    tau_arr = np.array([8/3, 16/3, 8]) # 3-layer homogeneous split
    NLayers = len(tau_arr)
    omega_arr = np.full(NLayers, 0.8)
    NQuad = 16
    Leg_coeffs_all = np.tile(0.75 ** np.arange(32), (NLayers, 1))
    mu0 = 0.6
    I0 = pi / mu0
    phi0 = 0.9 * pi

    # Optional (used)
    f_arr = np.repeat(Leg_coeffs_all[0, NQuad], NLayers)
    NT_cor = True
    b_neg=1
    b_pos=1
    BDRF_Fourier_modes=[1]
    s_poly_coeffs=np.tile(np.arange(3) + 1, (NLayers, 1))

    # Optional (unused)
    NLeg=None
    NFourier=None
    only_flux=False
    use_banded_solver_NLayers=10
    autograd_compatible=False

    ####################################################################################################

    # Test points
    Nphi = int((NQuad * pi) // 2) * 2 + 1
    phi_arr, full_weights_phi = PythonicDISORT.subroutines.Clenshaw_Curtis_quad(Nphi)

    # Call pydisort function
    flux_up, flux_down, u0, u = PythonicDISORT.pydisort(
        tau_arr, omega_arr,
        NQuad,
        Leg_coeffs_all,
        mu0, I0, phi0,
        b_pos=b_pos,
        b_neg=b_neg,
        f_arr=f_arr,
        BDRF_Fourier_modes=BDRF_Fourier_modes,
        s_poly_coeffs=s_poly_coeffs[:, :2],
        NT_cor=True,
    )[1:]
    
    end = tau_arr[-1]
    # Multi-layer integration via the documented antiderivative idiom.
    upper = tau_arr
    lower = np.insert(tau_arr[:-1] + 1e-15, 0, 0)
    assert np.allclose(quad_vec(lambda tau: u(tau, phi_arr), 0, end)[0], (u(upper, phi_arr, True) - u(lower, phi_arr, True)).sum(axis=1))
    assert np.allclose(quad_vec(u0, 0, end)[0], (u0(upper, True) - u0(lower, True)).sum(axis=-1))
    assert np.allclose(quad(flux_up, 0, end)[0], (flux_up(upper, True) - flux_up(lower, True)).sum())
    assert np.allclose(quad(lambda tau: flux_down(tau)[0], 0, end)[0], (flux_down(upper, True)[0] - flux_down(lower, True)[0]).sum())
    assert np.allclose(quad(lambda tau: flux_down(tau)[1], 0, end)[0], (flux_down(upper, True)[1] - flux_down(lower, True)[1]).sum())
    # ------------------------------------------------------------------------------------------------------------------


def test_Ic():
    print()
    print("################################################ Test Ic ##################################################")
    print("######################## Test integration when s(tau) (isotropic source) is cubic #########################")
    print()

    ######################################### PYDISORT ARGUMENTS #######################################
    tau_arr = np.array([8/3, 16/3, 8]) # 3-layer homogeneous split
    NLayers = len(tau_arr)
    omega_arr = np.full(NLayers, 0.8)
    NQuad = 16
    Leg_coeffs_all = np.tile(0.75 ** np.arange(32), (NLayers, 1))
    mu0 = 0.6
    I0 = pi / mu0
    phi0 = 0.9 * pi

    # Optional (used)
    f_arr = np.repeat(Leg_coeffs_all[0, NQuad], NLayers)
    NT_cor = True
    b_neg=1
    b_pos=1
    BDRF_Fourier_modes=[1]
    s_poly_coeffs=np.tile(np.arange(3) + 1, (NLayers, 1))

    # Optional (unused)
    NLeg=None
    NFourier=None
    only_flux=False
    use_banded_solver_NLayers=10
    autograd_compatible=False

    ####################################################################################################

    # Test points
    Nphi = int((NQuad * pi) // 2) * 2 + 1
    phi_arr, full_weights_phi = PythonicDISORT.subroutines.Clenshaw_Curtis_quad(Nphi)

    # Call pydisort function
    flux_up, flux_down, u0, u = PythonicDISORT.pydisort(
        tau_arr, omega_arr,
        NQuad,
        Leg_coeffs_all,
        mu0, I0, phi0,
        b_pos=b_pos,
        b_neg=b_neg,
        f_arr=f_arr,
        BDRF_Fourier_modes=BDRF_Fourier_modes,
        s_poly_coeffs=s_poly_coeffs,
        NT_cor=True,
    )[1:]

    end = tau_arr[-1]
    # Multi-layer integration via the documented antiderivative idiom.
    upper = tau_arr
    lower = np.insert(tau_arr[:-1] + 1e-15, 0, 0)
    assert np.allclose(quad_vec(lambda tau: u(tau, phi_arr), 0, end)[0], (u(upper, phi_arr, True) - u(lower, phi_arr, True)).sum(axis=1))
    assert np.allclose(quad_vec(u0, 0, end)[0], (u0(upper, True) - u0(lower, True)).sum(axis=-1))
    assert np.allclose(quad(flux_up, 0, end)[0], (flux_up(upper, True) - flux_up(lower, True)).sum())
    assert np.allclose(quad(lambda tau: flux_down(tau)[0], 0, end)[0], (flux_down(upper, True)[0] - flux_down(lower, True)[0]).sum())
    assert np.allclose(quad(lambda tau: flux_down(tau)[1], 0, end)[0], (flux_down(upper, True)[1] - flux_down(lower, True)[1]).sum())
    # ------------------------------------------------------------------------------------------------------------------


def test_Id(): # inhomogeneous 3-layer atmosphere with cubic source
    print()
    print("################################################ Test Id ##################################################")
    print("############### Test integration when s(tau) is cubic in an inhomogeneous 3-layer atmosphere ###############")
    print()

    ######################################### PYDISORT ARGUMENTS #######################################
    tau_arr = np.array([2, 5, 8]) # inhomogeneous layer thicknesses
    NLayers = len(tau_arr)
    omega_arr = np.array([0.85, 0.75, 0.6]) # per-layer single-scattering albedo
    NQuad = 8 # reduced from 16 to bound runtime cost of this added test
    g_per_layer = np.array([0.85, 0.6, 0.2]) # forward / mid / near-isotropic asymmetry
    Leg_coeffs_all = g_per_layer[:, None] ** np.arange(32)[None, :] # per-layer Henyey-Greenstein-like phase
    mu0 = 0.6
    I0 = pi / mu0
    phi0 = 0.9 * pi

    # Optional (used)
    f_arr = Leg_coeffs_all[:, NQuad] # per-layer truncation fraction
    NT_cor = True
    b_neg = 1
    b_pos = 1
    BDRF_Fourier_modes = [1]
    s_poly_coeffs = np.array([[1, 2, 3], [2, 1, 4], [3, 2, 1]]) # distinct cubic source coeffs per layer

    # Optional (unused)
    NLeg = None
    NFourier = None
    only_flux = False
    use_banded_solver_NLayers = 10
    autograd_compatible = False

    ####################################################################################################

    # Test points
    Nphi = int((NQuad * pi) // 2) * 2 + 1
    phi_arr, full_weights_phi = PythonicDISORT.subroutines.Clenshaw_Curtis_quad(Nphi)

    # Call pydisort function
    flux_up, flux_down, u0, u = PythonicDISORT.pydisort(
        tau_arr, omega_arr,
        NQuad,
        Leg_coeffs_all,
        mu0, I0, phi0,
        b_pos=b_pos,
        b_neg=b_neg,
        f_arr=f_arr,
        BDRF_Fourier_modes=BDRF_Fourier_modes,
        s_poly_coeffs=s_poly_coeffs,
        NT_cor=NT_cor,
    )[1:]

    end = tau_arr[-1]
    # Multi-layer integration via the documented antiderivative idiom.
    upper = tau_arr
    lower = np.insert(tau_arr[:-1] + 1e-15, 0, 0)
    assert np.allclose(quad_vec(lambda tau: u(tau, phi_arr), 0, end)[0], (u(upper, phi_arr, True) - u(lower, phi_arr, True)).sum(axis=1))
    assert np.allclose(quad_vec(u0, 0, end)[0], (u0(upper, True) - u0(lower, True)).sum(axis=-1))
    assert np.allclose(quad(flux_up, 0, end)[0], (flux_up(upper, True) - flux_up(lower, True)).sum())
    assert np.allclose(quad(lambda tau: flux_down(tau)[0], 0, end)[0], (flux_down(upper, True)[0] - flux_down(lower, True)[0]).sum())
    assert np.allclose(quad(lambda tau: flux_down(tau)[1], 0, end)[0], (flux_down(upper, True)[1] - flux_down(lower, True)[1]).sum())
    # ------------------------------------------------------------------------------------------------------------------
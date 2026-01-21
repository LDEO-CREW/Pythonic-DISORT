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
    
    # Except for `s_poly_coeffs`, the optical parameters are identical to those for `u_1layer` in test 11a
    
    ######################################### PYDISORT ARGUMENTS #######################################
    tau_arr = np.array([8])
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
        tau_arr[-1], omega_arr[0],
        NQuad,
        Leg_coeffs_all[0, :],
        mu0, I0, phi0,
        b_pos=b_pos,
        b_neg=b_neg,
        f_arr=f_arr[0],
        BDRF_Fourier_modes=BDRF_Fourier_modes,
        s_poly_coeffs=s_poly_coeffs[0, :1],
        NT_cor=True,
    )[1:]
    
    end = tau_arr[-1]
    assert np.allclose(quad_vec(lambda tau: u(tau, phi_arr), 0, end)[0], u(end, phi_arr, True) - u(0, phi_arr, True))
    assert np.allclose(quad_vec(u0, 0, end)[0], u0(end, True) - u0(0, True))
    assert np.allclose(quad(flux_up, 0, end)[0], flux_up(end, True) - flux_up(0, True))
    assert np.allclose(quad(lambda tau: flux_down(tau)[0], 0, end)[0], flux_down(end, True)[0] - flux_down(0, True)[0])
    assert np.allclose(quad(lambda tau: flux_down(tau)[1], 0, end)[0], flux_down(end, True)[1] - flux_down(0, True)[1])
    # ------------------------------------------------------------------------------------------------------------------
    
    
def test_Ib():
    print()
    print("################################################ Test Ib ##################################################")
    print("######################## Test integration when s(tau) (isotropic source) is linear ########################")
    print()
     
    ######################################### PYDISORT ARGUMENTS #######################################
    tau_arr = np.array([8])
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
        tau_arr[-1], omega_arr[0],
        NQuad,
        Leg_coeffs_all[0, :],
        mu0, I0, phi0,
        b_pos=b_pos,
        b_neg=b_neg,
        f_arr=f_arr[0],
        BDRF_Fourier_modes=BDRF_Fourier_modes,
        s_poly_coeffs=s_poly_coeffs[0, :2],
        NT_cor=True,
    )[1:]
    
    end = tau_arr[-1]
    assert np.allclose(quad_vec(lambda tau: u(tau, phi_arr), 0, end)[0], u(end, phi_arr, True) - u(0, phi_arr, True))
    assert np.allclose(quad_vec(u0, 0, end)[0], u0(end, True) - u0(0, True))
    assert np.allclose(quad(flux_up, 0, end)[0], flux_up(end, True) - flux_up(0, True))
    assert np.allclose(quad(lambda tau: flux_down(tau)[0], 0, end)[0], flux_down(end, True)[0] - flux_down(0, True)[0])
    assert np.allclose(quad(lambda tau: flux_down(tau)[1], 0, end)[0], flux_down(end, True)[1] - flux_down(0, True)[1])
    # ------------------------------------------------------------------------------------------------------------------
    
    
def test_Ic():
    print()
    print("################################################ Test Ic ##################################################")
    print("######################## Test integration when s(tau) (isotropic source) is cubic #########################")
    print()

    ######################################### PYDISORT ARGUMENTS #######################################
    tau_arr = np.array([8])
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
        tau_arr[-1], omega_arr[0],
        NQuad,
        Leg_coeffs_all[0, :],
        mu0, I0, phi0,
        b_pos=b_pos,
        b_neg=b_neg,
        f_arr=f_arr[0],
        BDRF_Fourier_modes=BDRF_Fourier_modes,
        s_poly_coeffs=s_poly_coeffs[0, :],
        NT_cor=True,
    )[1:]
    
    end = tau_arr[-1]
    assert np.allclose(quad_vec(lambda tau: u(tau, phi_arr), 0, end)[0], u(end, phi_arr, True) - u(0, phi_arr, True))
    assert np.allclose(quad_vec(u0, 0, end)[0], u0(end, True) - u0(0, True))
    assert np.allclose(quad(flux_up, 0, end)[0], flux_up(end, True) - flux_up(0, True))
    assert np.allclose(quad(lambda tau: flux_down(tau)[0], 0, end)[0], flux_down(end, True)[0] - flux_down(0, True)[0])
    assert np.allclose(quad(lambda tau: flux_down(tau)[1], 0, end)[0], flux_down(end, True)[1] - flux_down(0, True)[1])
    # ------------------------------------------------------------------------------------------------------------------
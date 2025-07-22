import numpy as np
import PythonicDISORT
from PythonicDISORT.subroutines import _compare
from math import pi

# ======================================================================================================
# Test Problem 11: Single-Layer vs. Multiple Layers
# ======================================================================================================

def test_11a():
    ######################################### PYDISORT ARGUMENTS #######################################
    tau_arr = np.arange(16) / 2 + 0.5
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
    BDRF_Fourier_modes=[lambda mu, neg_mup: np.full((len(mu), len(neg_mup)), 1)]
    s_poly_coeffs=np.tile(np.array([1, 1]), (NLayers, 1))

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
    Ntau = 100
    tau_test_arr = np.sort(np.random.random(Ntau) * tau_arr[-1])
    
    # Call pydisort function
    flux_up_1layer, flux_down_1layer, u0, u_1layer = PythonicDISORT.pydisort(
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

    flux_up_16layers, flux_down_16layers, u0, u_16layers = PythonicDISORT.pydisort(
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
    
    assert np.allclose(flux_up_1layer(tau_test_arr), flux_up_16layers(tau_test_arr))
    assert np.allclose(flux_down_1layer(tau_test_arr), flux_down_16layers(tau_test_arr))
    assert np.allclose(u_1layer(tau_test_arr, phi_arr), u_16layers(tau_test_arr, phi_arr))
    # --------------------------------------------------------------------------------------------------
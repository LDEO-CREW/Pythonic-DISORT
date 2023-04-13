import numpy as np
import PyDISORT
from PyDISORT.subroutines import _compare
from math import pi

# ======================================================================================================
# Test Problem 11: Single-Layer vs. Multiple Layers
# ======================================================================================================

def test_11a():
    ######################################### PYDISORT ARGUMENTS #######################################

    tau_arr = np.array([2, 4, 6, 8])
    omega_arr = np.array([0.9, 0.9, 0.9, 0.9])
    NQuad = 16
    Leg_coeffs_all = np.tile(0.75 ** np.arange(32), (4, 1))
    mu0 = 0.6
    I0 = pi / mu0
    phi0 = 0.9 * pi

    # Optional (used)
    f_arr = np.tile(Leg_coeffs_all[0, NQuad], 4)
    NT_cor = True
    b_neg=1
    b_pos=1
    Leg_coeffs_BDRF=np.array([0.1])
    s_poly_coeffs=np.array([[ 172311.79936609, -102511.44170512],
                           [ 172311.79936609, -102511.44170512],
                           [ 172311.79936609, -102511.44170512],
                           [ 172311.79936609, -102511.44170512]])

    # Optional (unused)
    NLeg=None
    NLoops=None
    only_flux=False
    use_sparse_NLayers=6

    ####################################################################################################
    
    # Test points
    Nphi = int((NQuad * pi) // 2) * 2 + 1  
    phi_arr, full_weights_phi = PyDISORT.subroutines.Clenshaw_Curtis_quad(Nphi)
    Ntau = 1000
    tau_test_arr = np.sort(np.random.random(Ntau) * tau_arr[-1])
    
    # Call PyDISORT
    flux_up_1layer, flux_down_1layer, u_1layer = PyDISORT.pydisort(
        tau_arr[-1], omega_arr[0],
        NQuad,
        Leg_coeffs_all[0, :],
        mu0, I0, phi0,
        b_pos=b_pos,
        b_neg=b_neg,
        f_arr=f_arr[0],
        s_poly_coeffs=s_poly_coeffs[0, :],
        NT_cor=True,
    )[1:4]

    flux_up_4layers, flux_down_4layers, u_4layers = PyDISORT.pydisort(
        tau_arr, omega_arr,
        NQuad,
        Leg_coeffs_all,
        mu0, I0, phi0,
        b_pos=b_pos,
        b_neg=b_neg,
        f_arr=f_arr,
        s_poly_coeffs=s_poly_coeffs,
        NT_cor=True,
    )[1:4]
    
    assert np.allclose(flux_up_1layer(tau_test_arr), flux_up_4layers(tau_test_arr))
    assert np.allclose(flux_down_1layer(tau_test_arr), flux_down_4layers(tau_test_arr))
    assert np.allclose(u_1layer(tau_test_arr, phi_arr), u_4layers(tau_test_arr, phi_arr))
    # --------------------------------------------------------------------------------------------------
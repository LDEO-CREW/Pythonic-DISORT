import numpy as np
import PyDISORT
from PyDISORT.subroutines import compare
from math import pi

def test_3a():
    ######################################### PYDISORT ARGUMENTS #######################################

    tau_arr = 1
    omega_arr = 0.99 # Instead of 1 as that special case has not been implemented
    NQuad = 16
    g = 0.75
    Leg_coeffs_all = g ** np.arange(32)
    mu0 = 0.5
    I0 = pi / mu0
    phi0 = 0

    # Optional (used)
    f_arr=Leg_coeffs_all[NQuad]
    NT_cor=True

    # Optional (unused)
    NLeg=None
    NLoops=None
    b_pos=0
    b_neg=0
    only_flux=False
    Leg_coeffs_BDRF=np.array([])
    s_poly_coeffs=np.array([[]])
    use_sparse_NLayers=6

    ####################################################################################################

    # Call PyDISORT
    mu_arr, flux_up, flux_down, u = PyDISORT.pydisort(
        tau_arr, omega_arr,
        NQuad,
        Leg_coeffs_all,
        mu0, I0, phi0,
        f_arr=f_arr,
        NT_cor=NT_cor
    )

    # Load results from version 4.0.99 of Stamnes' DISORT for comparison
    results = np.load("results/3a_test.npz")
    
    # Perform the comparisons
    (
        diff_flux_up,
        ratio_flux_up,
        diff_flux_down_diffuse,
        ratio_flux_down_diffuse,
        diff_flux_down_direct,
        ratio_flux_down_direct,
        diff,
        diff_ratio,
    ) = compare(results, mu_arr, flux_up, flux_down, u)
    
    assert np.max(ratio_flux_up) <= 1e-3 or np.max(diff_flux_up) <= 1e-3
    assert np.max(ratio_flux_down_diffuse) <= 1e-3 or np.max(diff_flux_down_diffuse) <= 1e-3
    assert np.max(ratio_flux_down_direct) <= 1e-3 or np.max(diff_flux_down_direct) <= 1e-3
    assert np.max(diff_ratio) <= 1e-2 or np.max(diff) <= 1e-2
    # --------------------------------------------------------------------------------------------------
    
    
def test_3b():
    ######################################### PYDISORT ARGUMENTS #######################################

    tau_arr = 8
    omega_arr = 0.99 # Instead of 1 as that special case has not been implemented
    NQuad = 16
    g = 0.75
    Leg_coeffs_all = g ** np.arange(32)
    mu0 = 0.5
    I0 = pi / mu0
    phi0 = 0

    # Optional (used)
    f_arr=Leg_coeffs_all[NQuad]
    NT_cor=True

    # Optional (unused)
    NLeg=None
    NLoops=None
    b_pos=0
    b_neg=0
    only_flux=False
    Leg_coeffs_BDRF=np.array([])
    s_poly_coeffs=np.array([[]])
    use_sparse_NLayers=6

    ####################################################################################################

    # Call PyDISORT
    mu_arr, flux_up, flux_down, u = PyDISORT.pydisort(
        tau_arr, omega_arr,
        NQuad,
        Leg_coeffs_all,
        mu0, I0, phi0,
        f_arr=f_arr,
        NT_cor=NT_cor
    )

    # Load results from version 4.0.99 of Stamnes' DISORT for comparison
    results = np.load("results/3b_test.npz")
    
    # Perform the comparisons
    (
        diff_flux_up,
        ratio_flux_up,
        diff_flux_down_diffuse,
        ratio_flux_down_diffuse,
        diff_flux_down_direct,
        ratio_flux_down_direct,
        diff,
        diff_ratio,
    ) = compare(results, mu_arr, flux_up, flux_down, u)
    
    assert np.max(ratio_flux_up) <= 1e-3 or np.max(diff_flux_up) <= 1e-3
    assert np.max(ratio_flux_down_diffuse) <= 1e-3 or np.max(diff_flux_down_diffuse) <= 1e-3
    assert np.max(ratio_flux_down_direct) <= 1e-3 or np.max(diff_flux_down_direct) <= 1e-3
    assert np.max(diff_ratio) <= 1e-2 or np.max(diff) <= 1e-2
    # --------------------------------------------------------------------------------------------------
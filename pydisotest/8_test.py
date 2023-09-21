import numpy as np
import PythonicDISORT
from PythonicDISORT.subroutines import _compare
from math import pi

# ======================================================================================================
# Test Problem 8:  Absorbing/Isotropic-Scattering Medium With Two Computational Layers
# ======================================================================================================

def test_8a():
    print()
    print("################################################ Test 8a ##################################################")
    print()
    ######################################### PYDISORT ARGUMENTS #######################################

    tau_arr = np.array([0.25, 0.5])
    omega_arr = np.array([0.5, 0.3])
    NQuad = 8
    Leg_coeffs_all = np.zeros((2, 9))
    Leg_coeffs_all[:, 0] = 1
    mu0 = 0
    I0 = 0
    phi0 = 0

    # Optional (used)
    b_neg = 1 / pi

    # Optional (unused)
    NLeg = None
    NLoops = None
    b_pos = 0
    only_flux = False
    f_arr = 0
    NT_cor = False
    BDRF_Fourier_modes = []
    s_poly_coeffs = np.array([[]])
    use_sparse_NLayers = 13

    ####################################################################################################

    # Call pydisort function
    mu_arr, flux_up, flux_down, u0, u = PythonicDISORT.pydisort(
        tau_arr, omega_arr,
        NQuad,
        Leg_coeffs_all,
        mu0, I0, phi0,
        b_neg=b_neg,
    )
    
    # mu_arr is arranged as it is for code efficiency and readability
    # For presentation purposes we re-arrange mu_arr from smallest to largest
    reorder_mu = np.argsort(mu_arr)
    mu_arr_RO = mu_arr[reorder_mu]

    # By default we do not compare intensities 10 degrees around the direct beam
    deg_around_beam_to_not_compare = 0 # Changed to 0 since this test problem has no direct beam
    mu_to_compare = (
        np.abs(np.arccos(np.abs(mu_arr_RO)) - np.arccos(mu0)) * 180 / pi
        > deg_around_beam_to_not_compare
    )
    

    
    # Load results from version 4.0.99 of Stamnes' DISORT for comparison
    results = np.load("Stamnes_results/8a_test.npz")
    
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
    ) = _compare(results, mu_to_compare, reorder_mu, flux_up, flux_down, u)
    
    assert np.max(ratio_flux_up[diff_flux_up > 1e-3], initial=0) < 1e-3
    assert np.max(ratio_flux_down_diffuse[diff_flux_down_diffuse > 1e-3], initial=0) < 1e-3
    assert np.max(ratio_flux_down_direct[diff_flux_down_direct > 1e-3], initial=0) < 1e-3
    assert np.max(diff_ratio[diff > 1e-3], initial=0) < 1e-2
    # --------------------------------------------------------------------------------------------------


def test_8b():
    print()
    print("################################################ Test 8b ##################################################")
    print()
    ######################################### PYDISORT ARGUMENTS #######################################

    tau_arr = np.array([0.25, 0.5])
    omega_arr = np.array([0.8, 0.95])
    NQuad = 8
    Leg_coeffs_all = np.zeros((2, 9))
    Leg_coeffs_all[:, 0] = 1
    mu0 = 0
    I0 = 0
    phi0 = 0

    # Optional (used)
    b_neg = 1 / pi

    # Optional (unused)
    NLeg = None
    NLoops = None
    b_pos = 0
    only_flux = False
    f_arr = 0
    NT_cor = False
    BDRF_Fourier_modes = []
    s_poly_coeffs = np.array([[]])
    use_sparse_NLayers = 13

    ####################################################################################################

    # Call pydisort function
    mu_arr, flux_up, flux_down, u0, u = PythonicDISORT.pydisort(
        tau_arr, omega_arr,
        NQuad,
        Leg_coeffs_all,
        mu0, I0, phi0,
        b_neg=b_neg,
    )
    
    # mu_arr is arranged as it is for code efficiency and readability
    # For presentation purposes we re-arrange mu_arr from smallest to largest
    reorder_mu = np.argsort(mu_arr)
    mu_arr_RO = mu_arr[reorder_mu]

    # By default we do not compare intensities 10 degrees around the direct beam
    deg_around_beam_to_not_compare = 0 # Changed to 0 since this test problem has no direct beam
    mu_to_compare = (
        np.abs(np.arccos(np.abs(mu_arr_RO)) - np.arccos(mu0)) * 180 / pi
        > deg_around_beam_to_not_compare
    )
    

    
    # Load results from version 4.0.99 of Stamnes' DISORT for comparison
    results = np.load("Stamnes_results/8b_test.npz")
    
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
    ) = _compare(results, mu_to_compare, reorder_mu, flux_up, flux_down, u)
    
    assert np.max(ratio_flux_up[diff_flux_up > 1e-3], initial=0) < 1e-3
    assert np.max(ratio_flux_down_diffuse[diff_flux_down_diffuse > 1e-3], initial=0) < 1e-3
    assert np.max(ratio_flux_down_direct[diff_flux_down_direct > 1e-3], initial=0) < 1e-3
    assert np.max(diff_ratio[diff > 1e-3], initial=0) < 1e-2
    # --------------------------------------------------------------------------------------------------


def test_8c():
    print()
    print("################################################ Test 8c ##################################################")
    print()
    ######################################### PYDISORT ARGUMENTS #######################################

    tau_arr = np.array([1, 3])
    omega_arr = np.array([0.8, 0.95])
    NQuad = 8
    Leg_coeffs_all = np.zeros((2, 9))
    Leg_coeffs_all[:, 0] = 1
    mu0 = 0
    I0 = 0
    phi0 = 0

    # Optional (used)
    b_neg = 1 / pi

    # Optional (unused)
    NLeg = None
    NLoops = None
    b_pos = 0
    only_flux = False
    f_arr = 0
    NT_cor = False
    BDRF_Fourier_modes = []
    s_poly_coeffs = np.array([[]])
    use_sparse_NLayers = 13

    ####################################################################################################

    # Call pydisort function
    mu_arr, flux_up, flux_down, u0, u = PythonicDISORT.pydisort(
        tau_arr, omega_arr,
        NQuad,
        Leg_coeffs_all,
        mu0, I0, phi0,
        b_neg=b_neg,
    )
    
    # mu_arr is arranged as it is for code efficiency and readability
    # For presentation purposes we re-arrange mu_arr from smallest to largest
    reorder_mu = np.argsort(mu_arr)
    mu_arr_RO = mu_arr[reorder_mu]

    # By default we do not compare intensities 10 degrees around the direct beam
    deg_around_beam_to_not_compare = 0 # Changed to 0 since this test problem has no direct beam
    mu_to_compare = (
        np.abs(np.arccos(np.abs(mu_arr_RO)) - np.arccos(mu0)) * 180 / pi
        > deg_around_beam_to_not_compare
    )
    

    
    # Load results from version 4.0.99 of Stamnes' DISORT for comparison
    results = np.load("Stamnes_results/8c_test.npz")
    
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
    ) = _compare(results, mu_to_compare, reorder_mu, flux_up, flux_down, u)
    
    assert np.max(ratio_flux_up[diff_flux_up > 1e-3], initial=0) < 1e-3
    assert np.max(ratio_flux_down_diffuse[diff_flux_down_diffuse > 1e-3], initial=0) < 1e-3
    assert np.max(ratio_flux_down_direct[diff_flux_down_direct > 1e-3], initial=0) < 1e-3
    assert np.max(diff_ratio[diff > 1e-3], initial=0) < 1e-2
    # --------------------------------------------------------------------------------------------------

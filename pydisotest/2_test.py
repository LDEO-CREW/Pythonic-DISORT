import numpy as np
import PythonicDISORT
from PythonicDISORT.subroutines import _compare
from math import pi

# ======================================================================================================
# Test Problem 2:  Rayleigh Scattering, Beam Source
# ======================================================================================================

def test_2a():
    print()
    print("################################################ Test 2a ##################################################")
    print()
    ######################################### PYDISORT ARGUMENTS #######################################

    tau_arr = 0.2
    omega_arr = 0.5
    NQuad = 16
    Leg_coeffs_all = np.zeros(17)
    Leg_coeffs_all[0] = 1
    Leg_coeffs_all[2] = 0.1
    mu0 = 0.080442
    I0 = pi
    phi0 = pi

    # Optional (used)

    # Optional (unused)
    NLeg=None
    NFourier=None
    b_pos=0
    b_neg=0
    only_flux=False
    f_arr=0
    NT_cor=False
    BDRF_Fourier_modes=[]
    s_poly_coeffs=np.array([[]])
    use_banded_solver_NLayers=10
    autograd_compatible=False

    ####################################################################################################

    # Call pydisort function
    mu_arr, flux_up, flux_down, u0, u = PythonicDISORT.pydisort(
        tau_arr, omega_arr,
        NQuad,
        Leg_coeffs_all,
        mu0, I0, phi0,
    )
    
    # mu_arr is arranged as it is for code efficiency and readability
    # For presentation purposes we re-arrange mu_arr from smallest to largest
    reorder_mu = np.argsort(mu_arr)
    mu_arr_RO = mu_arr[reorder_mu]

    # We may not want to compare intensities around the direct beam
    deg_around_beam_to_not_compare = 0
    mu_to_compare = (
        np.abs(np.arccos(np.abs(mu_arr_RO)) - np.arccos(mu0)) * 180 / pi
        > deg_around_beam_to_not_compare
    )
    
    # Load results from version 4.0.99 of Stamnes' DISORT for comparison
    results = np.load("Stamnes_results/2a_test.npz")
    
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
    
    
def test_2b():
    print()
    print("################################################ Test 2b ##################################################")
    print()
    ######################################### PYDISORT ARGUMENTS #######################################

    tau_arr = 0.2
    omega_arr = 1 - 1e-6 # Reduced from 1 because we have not implemented that special case
    NQuad = 16
    Leg_coeffs_all = np.zeros(17)
    Leg_coeffs_all[0] = 1
    Leg_coeffs_all[2] = 0.1
    mu0 = 0.080442
    I0 = pi
    phi0 = pi

    # Optional (used)

    # Optional (unused)
    NLeg=None
    NFourier=None
    b_pos=0
    b_neg=0
    only_flux=False
    f_arr=0
    NT_cor=False
    BDRF_Fourier_modes=[]
    s_poly_coeffs=np.array([[]])
    use_banded_solver_NLayers=10
    autograd_compatible=False

    ####################################################################################################

    # Call pydisort function
    mu_arr, flux_up, flux_down, u0, u = PythonicDISORT.pydisort(
        tau_arr, omega_arr,
        NQuad,
        Leg_coeffs_all,
        mu0, I0, phi0,
    )
    
    # mu_arr is arranged as it is for code efficiency and readability
    # For presentation purposes we re-arrange mu_arr from smallest to largest
    reorder_mu = np.argsort(mu_arr)
    mu_arr_RO = mu_arr[reorder_mu]

    # We may not want to compare intensities around the direct beam
    deg_around_beam_to_not_compare = 0
    mu_to_compare = (
        np.abs(np.arccos(np.abs(mu_arr_RO)) - np.arccos(mu0)) * 180 / pi
        > deg_around_beam_to_not_compare
    )
    
    
    # Load results from version 4.0.99 of Stamnes' DISORT for comparison
    results = np.load("Stamnes_results/2b_test.npz")
    
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
    

def test_2c():
    print()
    print("################################################ Test 2c ##################################################")
    print()
    ######################################### PYDISORT ARGUMENTS #######################################

    tau_arr = 5
    omega_arr = 0.5
    NQuad = 16
    Leg_coeffs_all = np.zeros(17)
    Leg_coeffs_all[0] = 1
    Leg_coeffs_all[2] = 0.1
    mu0 = 0.080442
    I0 = pi
    phi0 = pi

    # Optional (used)

    # Optional (unused)
    NLeg=None
    NFourier=None
    b_pos=0
    b_neg=0
    only_flux=False
    f_arr=0
    NT_cor=False
    BDRF_Fourier_modes=[]
    s_poly_coeffs=np.array([[]])
    use_banded_solver_NLayers=10
    autograd_compatible=False

    ####################################################################################################

    # Call pydisort function
    mu_arr, flux_up, flux_down, u0, u = PythonicDISORT.pydisort(
        tau_arr, omega_arr,
        NQuad,
        Leg_coeffs_all,
        mu0, I0, phi0,
    )
    
    # mu_arr is arranged as it is for code efficiency and readability
    # For presentation purposes we re-arrange mu_arr from smallest to largest
    reorder_mu = np.argsort(mu_arr)
    mu_arr_RO = mu_arr[reorder_mu]

    # We may not want to compare intensities around the direct beam
    deg_around_beam_to_not_compare = 0
    mu_to_compare = (
        np.abs(np.arccos(np.abs(mu_arr_RO)) - np.arccos(mu0)) * 180 / pi
        > deg_around_beam_to_not_compare
    )
    
    
    # Load results from version 4.0.99 of Stamnes' DISORT for comparison
    results = np.load("Stamnes_results/2c_test.npz")
    
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
    
    
def test_2d():
    print()
    print("################################################ Test 2d ##################################################")
    print()
    ######################################### PYDISORT ARGUMENTS #######################################

    tau_arr = 5
    omega_arr = 1 - 1e-6 # Reduced from 1 because we have not implemented that special case
    NQuad = 16
    Leg_coeffs_all = np.zeros(17)
    Leg_coeffs_all[0] = 1
    Leg_coeffs_all[2] = 0.1
    mu0 = 0.080442
    I0 = pi
    phi0 = pi

    # Optional (used)

    # Optional (unused)
    NLeg=None
    NFourier=None
    b_pos=0
    b_neg=0
    only_flux=False
    f_arr=0
    NT_cor=False
    BDRF_Fourier_modes=[]
    s_poly_coeffs=np.array([[]])
    use_banded_solver_NLayers=10
    autograd_compatible=False

    ####################################################################################################

    # Call pydisort function
    mu_arr, flux_up, flux_down, u0, u = PythonicDISORT.pydisort(
        tau_arr, omega_arr,
        NQuad,
        Leg_coeffs_all,
        mu0, I0, phi0,
    )
    
    # mu_arr is arranged as it is for code efficiency and readability
    # For presentation purposes we re-arrange mu_arr from smallest to largest
    reorder_mu = np.argsort(mu_arr)
    mu_arr_RO = mu_arr[reorder_mu]

    # We may not want to compare intensities around the direct beam
    deg_around_beam_to_not_compare = 0
    mu_to_compare = (
        np.abs(np.arccos(np.abs(mu_arr_RO)) - np.arccos(mu0)) * 180 / pi
        > deg_around_beam_to_not_compare
    )
    
    
    # Load results from version 4.0.99 of Stamnes' DISORT for comparison
    results = np.load("Stamnes_results/2d_test.npz")
    
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
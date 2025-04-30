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
    NFourier = None
    b_pos = 0
    only_flux = False
    f_arr = 0
    NT_cor = False
    BDRF_Fourier_modes = []
    s_poly_coeffs = np.array([[]])
    use_banded_solver_NLayers = 10

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
    NFourier = None
    b_pos = 0
    only_flux = False
    f_arr = 0
    NT_cor = False
    BDRF_Fourier_modes = []
    s_poly_coeffs = np.array([[]])
    use_banded_solver_NLayers = 10

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
    NFourier = None
    b_pos = 0
    only_flux = False
    f_arr = 0
    NT_cor = False
    BDRF_Fourier_modes = []
    s_poly_coeffs = np.array([[]])
    use_banded_solver_NLayers = 10

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


def test_8ARTS_A():
    print()
    print("################################################ Test 8ARTS_A ##################################################")
    print()
    from ARTS_data.inpydis import src, tau

    nv = len(src)
    pyth = np.empty((nv, 20, 8))

    for i in range(nv):
        mu_arr, flux_up, flux_down, u0, u = PythonicDISORT.pydisort(
            tau_arr=tau[i],
            omega_arr=tau[i] * 0,
            NQuad=8,
            Leg_coeffs_all=np.ones((len(tau[i]), 1)),
            I0=0.0, 
            mu0=0.0, 
            phi0=0.0,
            NLeg=1,
            NFourier=1,
            s_poly_coeffs=src[i] * 1e15,
        )

        pyth[i] = u(tau[i], 0.0).T

        
    # Unused optional arguments
    NLeg = None
    NFourier = None
    b_pos = 0
    b_neg = 0
    only_flux = False
    f_arr = 0
    NT_cor = False
    BDRF_Fourier_modes = []
    use_banded_solver_NLayers = 10
    autograd_compatible = False
    
    ARTS_results = np.load("Stamnes_results/8ARTS_A_test.npy")
    assert np.max(np.abs(pyth[:, -1, -1] - ARTS_results) / ARTS_results) < 1e-2
    # --------------------------------------------------------------------------------------------------


def test_8ARTS_B():
    print()
    print("################################################ Test 8ARTS_B ##################################################")
    print()
    from ARTS_data.pydisort_data import (
        optical_thicknesses,
        single_scattering_albedo,
        quadrature_dimension,
        legendre_coefficients,
        TEMPER,
    )
    from scipy.constants import speed_of_light

    freqs = [31.5e9, 165e9, 666e9]
    WVNM = np.array(freqs) / (100.0 * speed_of_light)
    WVNMHI = np.ones(len(freqs)) * 50000
    WVNMLO = np.zeros(len(freqs))

    for ifreq in range(len(freqs)):
        
        ######################################### PYDISORT ##############################################

        tau_arr = optical_thicknesses[ifreq]
        omega_arr = single_scattering_albedo[ifreq]
        NQuad = quadrature_dimension
        # Stamnes' DISORT needs an extra coefficient but by our settings it will not be used
        Leg_coeffs_all = np.hstack((legendre_coefficients[ifreq], np.zeros((len(tau_arr), 1))))
        mu0 = 0
        I0 = 0  # No direct beam
        phi0 = 0

        # Optional (used)
        s_poly_coeffs = PythonicDISORT.subroutines.generate_s_poly_coeffs(
            tau_arr, TEMPER, WVNMLO[ifreq], WVNMHI[ifreq], np.array(omega_arr)
        )
        b_pos = PythonicDISORT.subroutines.blackbody_contrib_to_BCs(
            np.mean(TEMPER), WVNMLO[ifreq], WVNMHI[ifreq]
        ) # Using an arbitrary temperature since surface temperature data is missing
        b_neg = PythonicDISORT.subroutines.blackbody_contrib_to_BCs(
            np.median(TEMPER), WVNMLO[ifreq], WVNMHI[ifreq]
        ) # Using an arbitrary temperature since upper boundary temperature data is missing
        
        # Call pydisort function
        mu_arr, flux_up, flux_down, u0, u = PythonicDISORT.pydisort(
            tau_arr, omega_arr,
            NQuad,
            Leg_coeffs_all,
            mu0, I0, phi0,
            b_pos=b_pos,
            b_neg=b_neg,
            s_poly_coeffs=s_poly_coeffs
        )
        
        #################################################################################################
        ######################################### SETUP FOR TESTS #######################################
        
        # Reorder mu_arr from smallest to largest
        reorder_mu = np.argsort(mu_arr)
        mu_arr_RO = mu_arr[reorder_mu]

        # We may not want to compare intensities around the direct beam
        deg_around_beam_to_not_compare = 0
        mu_to_compare = (
            np.abs(np.arccos(np.abs(mu_arr_RO)) - np.arccos(mu0)) * 180 / pi
            > deg_around_beam_to_not_compare
        )
        mu_test_arr_RO = mu_arr_RO[mu_to_compare]
        
        ######################################### COMPARE RESULTS #######################################
        #################################################################################################
        
        # Load saved results from Stamnes' DISORT
        results = np.load("Stamnes_results/8ARTS_B" + str(ifreq) + "_test.npz")
        
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
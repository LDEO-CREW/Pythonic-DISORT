import numpy as np
import PythonicDISORT
from PythonicDISORT.subroutines import _compare
from math import pi

# ======================================================================================================
# Test Problem 5:  Cloud C.1 Scattering, Beam Source (Test 5BDRF has Lambertian BDRF with albedo 1)
# ======================================================================================================

Leg_coeffs_ALL = np.array([1,
                           2.544,  3.883,  4.568,  5.235,  5.887,  6.457,  7.177,  7.859,
                           8.494,  9.286,  9.856, 10.615, 11.229, 11.851, 12.503, 13.058,
                           13.626, 14.209, 14.660, 15.231, 15.641, 16.126, 16.539, 16.934,
                           17.325, 17.673, 17.999, 18.329, 18.588, 18.885, 19.103, 19.345,
                           19.537, 19.721, 19.884, 20.024, 20.145, 20.251, 20.330, 20.401,
                           20.444, 20.477, 20.489, 20.483, 20.467, 20.427, 20.382, 20.310,
                           20.236, 20.136, 20.036, 19.909, 19.785, 19.632, 19.486, 19.311,
                           19.145, 18.949, 18.764, 18.551, 18.348, 18.119, 17.901, 17.659,
                           17.428, 17.174, 16.931, 16.668, 16.415, 16.144, 15.883, 15.606,
                           15.338, 15.058, 14.784, 14.501, 14.225, 13.941, 13.662, 13.378,
                           13.098, 12.816, 12.536, 12.257, 11.978, 11.703, 11.427, 11.156,
                           10.884, 10.618, 10.350, 10.090,  9.827,  9.574,  9.318,  9.072,
                           8.822, 8.584, 8.340, 8.110, 7.874, 7.652, 7.424, 7.211, 6.990,
                           6.785, 6.573, 6.377, 6.173, 5.986, 5.790, 5.612, 5.424, 5.255,
                           5.075, 4.915, 4.744, 4.592, 4.429, 4.285, 4.130, 3.994, 3.847,
                           3.719, 3.580, 3.459, 3.327, 3.214, 3.090, 2.983, 2.866, 2.766,
                           2.656, 2.562, 2.459, 2.372, 2.274, 2.193, 2.102, 2.025, 1.940,
                           1.869, 1.790, 1.723, 1.649, 1.588, 1.518, 1.461, 1.397, 1.344,
                           1.284, 1.235, 1.179, 1.134, 1.082, 1.040, 0.992, 0.954, 0.909,
                           0.873, 0.832, 0.799, 0.762, 0.731, 0.696, 0.668, 0.636, 0.610,
                           0.581, 0.557, 0.530, 0.508, 0.483, 0.463, 0.440, 0.422, 0.401,
                           0.384, 0.364, 0.349, 0.331, 0.317, 0.301, 0.288, 0.273, 0.262,
                           0.248, 0.238, 0.225, 0.215, 0.204, 0.195, 0.185, 0.177, 0.167,
                           0.160, 0.151, 0.145, 0.137, 0.131, 0.124, 0.118, 0.112, 0.107,
                           0.101, 0.097, 0.091, 0.087, 0.082, 0.079, 0.074, 0.071, 0.067,
                           0.064, 0.060, 0.057, 0.054, 0.052, 0.049, 0.047, 0.044, 0.042,
                           0.039, 0.038, 0.035, 0.034, 0.032, 0.030, 0.029, 0.027, 0.026,
                           0.024, 0.023, 0.022, 0.021, 0.020, 0.018, 0.018, 0.017, 0.016,
                           0.015, 0.014, 0.013, 0.013, 0.012, 0.011, 0.011, 0.010, 0.009,
                           0.009, 0.008, 0.008, 0.008, 0.007, 0.007, 0.006, 0.006, 0.006,
                           0.005, 0.005, 0.005, 0.005, 0.004, 0.004, 0.004, 0.004, 0.003,
                           0.003, 0.003, 0.003, 0.003, 0.003, 0.002, 0.002, 0.002, 0.002,
                           0.002, 0.002, 0.002, 0.002, 0.002, 0.001, 0.001, 0.001, 0.001,
                           0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001,
                           0.001, 0.001, 0.001, 0.001, 0.001])

def test_5a():
    print()
    print("################################################ Test 5a ##################################################")
    print()
    ######################################### PYDISORT ARGUMENTS #######################################

    tau_arr = 64
    omega_arr = 1 - 1e-6 # Reduced from 1 because we have not implemented that special case
    NQuad = 48
    Leg_coeffs_all = Leg_coeffs_ALL / (2 * np.arange(300) + 1)
    mu0 = 1
    I0 = pi
    phi0 = pi

    # Optional (used)
    f_arr = Leg_coeffs_all[NQuad]
    NT_cor = True

    # Optional (unused)
    NLeg=None
    NFourier=None
    b_pos=0
    b_neg=0
    only_flux=False
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
        f_arr=f_arr,
        NT_cor=NT_cor,
    )
    
    # mu_arr is arranged as it is for code efficiency and readability
    # For presentation purposes we re-arrange mu_arr from smallest to largest
    reorder_mu = np.argsort(mu_arr)
    mu_arr_RO = mu_arr[reorder_mu]

    # By default we do not compare intensities 10 degrees around the direct beam
    deg_around_beam_to_not_compare = 10  # This parameter changes the size of the region
    mu_to_compare = (
        np.abs(np.arccos(np.abs(mu_arr_RO)) - np.arccos(mu0)) * 180 / pi
        > deg_around_beam_to_not_compare
    )
    

    
    # Load results from version 4.0.99 of Stamnes' DISORT for comparison
    results = np.load("Stamnes_results/5a_test.npz")
    
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
    
    
def test_5b():
    print()
    print("################################################ Test 5b ##################################################")
    print()
    ######################################### PYDISORT ARGUMENTS #######################################

    tau_arr = 64
    omega_arr = 0.9
    NQuad = 48
    Leg_coeffs_all = Leg_coeffs_ALL / (2 * np.arange(300) + 1)
    mu0 = 1
    I0 = pi
    phi0 = pi

    # Optional (used)
    f_arr = Leg_coeffs_all[NQuad]
    NT_cor = True

    # Optional (unused)
    NLeg=None
    NFourier=None
    b_pos=0
    b_neg=0
    only_flux=False
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
        f_arr=f_arr,
        NT_cor=NT_cor,
    )
    
    # mu_arr is arranged as it is for code efficiency and readability
    # For presentation purposes we re-arrange mu_arr from smallest to largest
    reorder_mu = np.argsort(mu_arr)
    mu_arr_RO = mu_arr[reorder_mu]

    # By default we do not compare intensities 10 degrees around the direct beam
    deg_around_beam_to_not_compare = 10  # This parameter changes the size of the region
    mu_to_compare = (
        np.abs(np.arccos(np.abs(mu_arr_RO)) - np.arccos(mu0)) * 180 / pi
        > deg_around_beam_to_not_compare
    )
    

    
    # Load results from version 4.0.99 of Stamnes' DISORT for comparison
    results = np.load("Stamnes_results/5b_test.npz")
    
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
    
    
def test_5BDRF():
    print()
    print("################################################ Test 5BDRF ##################################################")
    print()
    ######################################### PYDISORT ARGUMENTS #######################################

    tau_arr = 64
    omega_arr = 1 - 1e-6 # Reduced from 1 because we have not implemented that special case
    NQuad = 48
    Leg_coeffs_all = Leg_coeffs_ALL / (2 * np.arange(300) + 1)
    mu0 = 1
    I0 = pi
    phi0 = pi

    # Optional (used)
    f_arr = Leg_coeffs_all[NQuad]
    NT_cor = True
    omega_s = 1
    BDRF_Fourier_modes=[lambda mu, neg_mup: np.full((len(mu), len(neg_mup)), omega_s)]

    # Optional (unused)
    NLeg=None
    NFourier=None
    b_pos=0
    b_neg=0
    only_flux=False
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
        f_arr=f_arr,
        NT_cor=NT_cor,
        BDRF_Fourier_modes=BDRF_Fourier_modes,
    )
    
    # mu_arr is arranged as it is for code efficiency and readability
    # For presentation purposes we re-arrange mu_arr from smallest to largest
    reorder_mu = np.argsort(mu_arr)
    mu_arr_RO = mu_arr[reorder_mu]

    # By default we do not compare intensities 10 degrees around the direct beam
    deg_around_beam_to_not_compare = 10  # This parameter changes the size of the region
    mu_to_compare = (
        np.abs(np.arccos(np.abs(mu_arr_RO)) - np.arccos(mu0)) * 180 / pi
        > deg_around_beam_to_not_compare
    )
    

    
    # Load results from version 4.0.99 of Stamnes' DISORT for comparison
    results = np.load("Stamnes_results/5BDRF_test.npz")
    
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
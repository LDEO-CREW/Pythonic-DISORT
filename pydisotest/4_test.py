import numpy as np
import PyDISORT
from PyDISORT.subroutines import _compare
from math import pi

# ======================================================================================================
# Test Problem 1:  Isotropic Scattering
# ======================================================================================================

def test_4b():
    print()
    print("################################################ Test 4b ##################################################")
    print()
    Leg_coeffs_ALL = np.array([1,
                               2.41260, 3.23047, 3.37296, 3.23150, 2.89350, 
                               2.49594, 2.11361, 1.74812, 1.44692, 1.17714,
                               0.96643, 0.78237, 0.64114, 0.51966, 0.42563,
                               0.34688, 0.28351, 0.23317, 0.18963, 0.15788,
                               0.12739, 0.10762, 0.08597, 0.07381, 0.05828,
                               0.05089, 0.03971, 0.03524, 0.02720, 0.02451,
                               0.01874, 0.01711, 0.01298, 0.01198, 0.00904,
                               0.00841, 0.00634, 0.00592, 0.00446, 0.00418,
                               0.00316, 0.00296, 0.00225, 0.00210, 0.00160,
                               0.00150, 0.00115, 0.00107, 0.00082, 0.00077,
                               0.00059, 0.00055, 0.00043, 0.00040, 0.00031,
                               0.00029, 0.00023, 0.00021, 0.00017, 0.00015,
                               0.00012, 0.00011, 0.00009, 0.00008, 0.00006,
                               0.00006, 0.00005, 0.00004, 0.00004, 0.00003,
                               0.00003, 0.00002, 0.00002, 0.00002, 0.00001,
                               0.00001, 0.00001, 0.00001, 0.00001, 0.00001,
                               0.00001, 0.0000])

    ######################################### PYDISORT ARGUMENTS #######################################

    tau_arr = 1
    omega_arr = 0.9
    NQuad = 32
    Leg_coeffs_all = Leg_coeffs_ALL[:33] / (2 * np.arange(33) + 1)
    mu0 = 1
    I0 = pi
    phi0 = 0

    # Optional (used)
    f_arr = Leg_coeffs_all[NQuad]
    NT_cor = True

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
        NT_cor=NT_cor,
    )
    
    # mu_arr is arranged as it is for code efficiency and readability
    # For presentation purposes we re-arrange mu_arr from smallest to largest
    reorder_mu = np.argsort(mu_arr)
    mu_arr_RO = mu_arr[reorder_mu]

    # mu_arr is arranged as it is for code efficiency and readability
    # For presentation purposes we re-arrange mu_arr from smallest to largest
    reorder_mu = np.argsort(mu_arr)
    mu_arr_RO = mu_arr[reorder_mu]

    # By default we do not compare intensities 1 degree around the direct beam
    # The size of the region can be changed using the parameter below
    mu_around_beam_to_not_compare = 0.1
    mu_to_compare = np.abs(np.abs(mu_arr_RO) - mu0) > mu_around_beam_to_not_compare
    mu_test_arr_RO = mu_arr_RO[mu_to_compare]
    
    # Load results from version 4.0.99 of Stamnes' DISORT for comparison
    results = np.load("Stamnes_results/4b_test.npz")
    
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
    
    assert np.max(ratio_flux_up) <= 1e-3 or np.max(diff_flux_up) <= 1e-2 / pi
    assert np.max(ratio_flux_down_diffuse) <= 1e-3 or np.max(diff_flux_down_diffuse) <= 1e-2 / pi
    assert np.max(ratio_flux_down_direct) <= 1e-3 or np.max(diff_flux_down_direct) <= 1e-2 / pi
    assert np.max(diff_ratio) <= 1e-2 or np.max(diff) <= 1e-2
    # --------------------------------------------------------------------------------------------------
    
    
def test_4c():
    print()
    print("################################################ Test 4c ##################################################")
    print()
    Leg_coeffs_ALL = np.array([1,
                               2.41260, 3.23047, 3.37296, 3.23150, 2.89350, 
                               2.49594, 2.11361, 1.74812, 1.44692, 1.17714,
                               0.96643, 0.78237, 0.64114, 0.51966, 0.42563,
                               0.34688, 0.28351, 0.23317, 0.18963, 0.15788,
                               0.12739, 0.10762, 0.08597, 0.07381, 0.05828,
                               0.05089, 0.03971, 0.03524, 0.02720, 0.02451,
                               0.01874, 0.01711, 0.01298, 0.01198, 0.00904,
                               0.00841, 0.00634, 0.00592, 0.00446, 0.00418,
                               0.00316, 0.00296, 0.00225, 0.00210, 0.00160,
                               0.00150, 0.00115, 0.00107, 0.00082, 0.00077,
                               0.00059, 0.00055, 0.00043, 0.00040, 0.00031,
                               0.00029, 0.00023, 0.00021, 0.00017, 0.00015,
                               0.00012, 0.00011, 0.00009, 0.00008, 0.00006,
                               0.00006, 0.00005, 0.00004, 0.00004, 0.00003,
                               0.00003, 0.00002, 0.00002, 0.00002, 0.00001,
                               0.00001, 0.00001, 0.00001, 0.00001, 0.00001,
                               0.00001, 0.0000])

    ######################################### PYDISORT ARGUMENTS #######################################

    tau_arr = 1
    omega_arr = 0.9
    NQuad = 32
    Leg_coeffs_all = Leg_coeffs_ALL[:33] / (2 * np.arange(33) + 1)
    mu0 = 0.5
    I0 = pi
    phi0 = 0

    # Optional (used)
    f_arr = Leg_coeffs_all[NQuad]
    NT_cor = True

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
        NT_cor=NT_cor,
    )
    
    # mu_arr is arranged as it is for code efficiency and readability
    # For presentation purposes we re-arrange mu_arr from smallest to largest
    reorder_mu = np.argsort(mu_arr)
    mu_arr_RO = mu_arr[reorder_mu]

    # By default we do not compare intensities 1 degree around the direct beam
    # The size of the region can be changed using the parameter below
    mu_around_beam_to_not_compare = 0.1
    mu_to_compare = np.abs(np.abs(mu_arr_RO) - mu0) > mu_around_beam_to_not_compare
    mu_test_arr_RO = mu_arr_RO[mu_to_compare]
    
    # Load results from version 4.0.99 of Stamnes' DISORT for comparison
    results = np.load("Stamnes_results/4c_test.npz")
    
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
    
    assert np.max(ratio_flux_up) <= 1e-3 or np.max(diff_flux_up) <= 1e-2 / pi
    assert np.max(ratio_flux_down_diffuse) <= 1e-3 or np.max(diff_flux_down_diffuse) <= 1e-2 / pi
    assert np.max(ratio_flux_down_direct) <= 1e-3 or np.max(diff_flux_down_direct) <= 1e-2 / pi
    assert np.max(diff_ratio) <= 1e-2 or np.max(diff) <= 1e-2
    # --------------------------------------------------------------------------------------------------
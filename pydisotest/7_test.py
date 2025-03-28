import numpy as np
import scipy as sc
import PythonicDISORT
from PythonicDISORT.subroutines import _compare
from math import pi

from PythonicDISORT.subroutines import generate_s_poly_coeffs
from PythonicDISORT.subroutines import blackbody_contrib_to_BCs
from PythonicDISORT.subroutines import generate_emissivity_from_BDRF

# ===========================================================================================================================
# Test Problem 7:  Absorption + Scattering + All Possible Sources, Lambertian and Hapke Surface Reflectivities (One Layer)
# ===========================================================================================================================

def test_7a():
    print()
    print("################################################ Test 7a ##################################################")
    print()
    ######################################### PYDISORT ARGUMENTS #######################################

    tau_arr = 1  # One layer of thickness 1 (medium-thick atmosphere)
    omega_arr = 0.1  # Very low scattering
    NQuad = 16  # 16 streams (8 quadrature nodes for each hemisphere)
    Leg_coeffs_all = 0.05 ** np.arange(NQuad + 1) # Henyey-Greenstein phase function with g = 0.05
    mu0 = 0  # No direct beam
    I0 = 0  # No direct beam
    phi0 = 0  # No direct beam

    # Optional (used)
    TEMPER = np.array([200, 300])
    WVNMLO = 300
    WVNMHI = 800
    # Emissivity is (1 - omega_arr) by Kirchoff's law of thermal radiation
    s_poly_coeffs=generate_s_poly_coeffs(tau_arr, TEMPER, WVNMLO, WVNMHI) * (1 - omega_arr)

    # Optional (unused)
    NLeg = None
    NFourier = None
    b_pos = 0
    b_neg = 0
    only_flux = False
    f_arr = 0
    NT_cor = False
    BDRF_Fourier_modes = []
    use_banded_solver_NLayers = 10
    autograd_compatible=False

    ####################################################################################################

    # Call pydisort function
    mu_arr, flux_up, flux_down, u0, u = PythonicDISORT.pydisort(
        tau_arr, omega_arr,
        NQuad,
        Leg_coeffs_all,
        mu0, I0, phi0,
        s_poly_coeffs=s_poly_coeffs,
    )
    
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

    
    # Load results from version 4.0.99 of Stamnes' DISORT for comparison
    results = np.load("Stamnes_results/7a_test.npz")
    
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


def test_7b():
    print()
    print("################################################ Test 7b ##################################################")
    print()
    ######################################### PYDISORT ARGUMENTS #######################################

    tau_arr = 100  # One layer of thickness 100 (Very thick atmosphere)
    omega_arr = 0.95  # High scattering
    NQuad = 16  # 16 streams (8 quadrature nodes for each hemisphere)
    Leg_coeffs_all = 0.75 ** np.arange(NQuad + 1) # Henyey-Greenstein phase function with g = 0.75
    mu0 = 0  # No direct beam
    I0 = 0  # No direct beam
    phi0 = 0  # No direct beam

    # Optional (used)
    TEMPER = np.array([200, 300])
    WVNMLO = 2702.99
    WVNMHI = 2703.01
    # Emissivity is (1 - omega_arr) by Kirchoff's law of thermal radiation
    s_poly_coeffs=generate_s_poly_coeffs(tau_arr, TEMPER, WVNMLO, WVNMHI) * (1 - omega_arr)

    # Optional (unused)
    NLeg = None
    NFourier = None
    b_pos = 0
    b_neg = 0
    only_flux = False
    f_arr = 0
    NT_cor = False
    BDRF_Fourier_modes = []
    use_banded_solver_NLayers = 10
    autograd_compatible=False

    ####################################################################################################

    # Call pydisort function
    mu_arr, flux_up, flux_down, u0, u = PythonicDISORT.pydisort(
        tau_arr, omega_arr,
        NQuad,
        Leg_coeffs_all,
        mu0, I0, phi0,
        s_poly_coeffs=s_poly_coeffs,
    )
    
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

    
    # Load results from version 4.0.99 of Stamnes' DISORT for comparison
    results = np.load("Stamnes_results/7b_test.npz")
    
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


def test_7c():
    print()
    print("################################################ Test 7c ##################################################")
    print()
    ######################################### PYDISORT ARGUMENTS #######################################

    tau_arr = 1  # One layer of thickness 1 (Medium-thick atmosphere)
    omega_arr = 0.5  # Low scattering
    NQuad = 12  # 12 streams (6 quadrature nodes for each hemisphere)
    Leg_coeffs_all = 0.8 ** np.arange(NQuad * 2) # Henyey-Greenstein phase function with g = 0.8
    mu0 = 0.5  # Cosine of solar zenith angle (directly downwards)
    I0 = 200  # Intensity of direct beam
    phi0 = 0  # Azimuthal angle of direct beam

    # Optional (used)
    TEMPER = np.array([300, 200])
    WVNMLO = 0
    WVNMHI = 80000
    BTEMP = 320
    TTEMP = 100
    # Emissivity is (1 - omega_arr) by Kirchoff's law of thermal radiation
    s_poly_coeffs=generate_s_poly_coeffs(tau_arr, TEMPER, WVNMLO, WVNMHI, epsrel=1e-15) * (1 - omega_arr)
    b_pos = blackbody_contrib_to_BCs(BTEMP, WVNMLO, WVNMHI, epsrel=1e-15) # Emissivity 1
    b_neg = blackbody_contrib_to_BCs(TTEMP, WVNMLO, WVNMHI, epsrel=1e-15) + 100 # Emissivity 1

    f_arr = Leg_coeffs_all[NQuad]
    NT_cor = True

    # Optional (unused)
    NLeg = None
    NFourier = None
    only_flux = False
    BDRF_Fourier_modes = []
    use_banded_solver_NLayers = 10
    autograd_compatible=False

    ####################################################################################################

    # Call pydisort function
    mu_arr, flux_up, flux_down, u0, u = PythonicDISORT.pydisort(
        tau_arr, omega_arr,
        NQuad,
        Leg_coeffs_all,
        mu0, I0, phi0,
        b_pos=b_pos,
        b_neg=b_neg,
        s_poly_coeffs=s_poly_coeffs,
        f_arr=f_arr,
        NT_cor=NT_cor,
    )
    
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

    
    # Load results from version 4.0.99 of Stamnes' DISORT for comparison
    results = np.load("Stamnes_results/7c_test.npz")
    
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
    
    
def test_7d():
    print()
    print("################################################ Test 7d ##################################################")
    print()
    ######################################### PYDISORT ARGUMENTS #######################################

    tau_arr = 1  # One layer of thickness 1 (Medium-thick atmosphere)
    omega_arr = 0.5  # Low scattering
    NQuad = 12  # 12 streams (6 quadrature nodes for each hemisphere)
    Leg_coeffs_all = 0.8 ** np.arange(NQuad * 2) # Henyey-Greenstein phase function with g = 0.8
    mu0 = 0.5  # Cosine of solar zenith angle (directly downwards)
    I0 = 200  # Intensity of direct beam
    phi0 = 0  # Azimuthal angle of direct beam

    # Optional (used)
    TEMPER = np.array([300, 200])
    WVNMLO = 0
    WVNMHI = 80000
    BTEMP = 320
    TTEMP = 100
    # Emissivity is (1 - omega_arr) by Kirchoff's law of thermal radiation
    s_poly_coeffs=generate_s_poly_coeffs(tau_arr, TEMPER, WVNMLO, WVNMHI, epsrel=1e-15) * (1 - omega_arr)
    b_neg = blackbody_contrib_to_BCs(TTEMP, WVNMLO, WVNMHI, epsrel=1e-15) + 100 # Emissivity 1
    omega_s = 1
    BDRF_Fourier_modes=[lambda mu, neg_mup: np.full((len(mu), len(neg_mup)), omega_s)]
    
    f_arr = Leg_coeffs_all[NQuad]
    NT_cor = True

    # Optional (unused)
    NLeg = None
    NFourier = None
    b_pos = 0
    only_flux = False
    use_banded_solver_NLayers = 10
    autograd_compatible=False

    ####################################################################################################

    # Call pydisort function
    mu_arr, flux_up, flux_down, u0, u = PythonicDISORT.pydisort(
        tau_arr, omega_arr,
        NQuad,
        Leg_coeffs_all,
        mu0, I0, phi0,
        b_neg=b_neg,
        s_poly_coeffs=s_poly_coeffs,
        BDRF_Fourier_modes=BDRF_Fourier_modes,
        f_arr=f_arr,
        NT_cor=NT_cor,
    )
    
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

    
    # Load results from version 4.0.99 of Stamnes' DISORT for comparison
    results = np.load("Stamnes_results/7d_test.npz")
    
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
    
    
def test_7e():
    print()
    print("################################################ Test 7e ##################################################")
    print()
    def Hapke(mu, neg_mup, dphi, B0, HH, W):
        cos_alpha = (mu[:, None] * neg_mup[None, :] - np.sqrt(1 - mu**2)[:, None] * np.sqrt(
            (1 - neg_mup**2)[None, :]
        ) * np.cos(dphi)).clip(min=-1, max=1)
        alpha = np.arccos(cos_alpha)

        P = 1 + cos_alpha / 2
        B = B0 * HH / (HH + np.tan(alpha / 2))

        gamma = np.sqrt(1 - W)
        H0 = ((1 + 2 * neg_mup) / (1 + 2 * neg_mup * gamma))[None, :]
        H = ((1 + 2 * mu) / (1 + 2 * mu * gamma))[:, None]

        return W / 4 / (mu[:, None] + neg_mup[None, :]) * ((1 + B) * P + H0 * H - 1)
    
    ######################################### PYDISORT ARGUMENTS #######################################

    tau_arr = 1  # One layer of thickness 1 (Medium-thick atmosphere)
    omega_arr = 0.5  # Low scattering
    NQuad = 12  # 12 streams (6 quadrature nodes for each hemisphere)
    Leg_coeffs_all = 0.8 ** np.arange(NQuad * 2) # Henyey-Greenstein phase function with g = 0.8
    mu0 = 0.5  # Cosine of solar zenith angle (directly downwards)
    I0 = 200  # Intensity of direct beam
    phi0 = 0  # Azimuthal angle of direct beam

    # Optional (used)
    B0, HH, W = 1, 0.06, 0.6
    BDRF_Fourier_modes = [
        lambda mu, neg_mup, m=m: (sc.integrate.quad_vec(
            lambda dphi: Hapke(mu, neg_mup, dphi, B0, HH, W) * np.cos(m * dphi),
            0,
            2 * pi,
        )[0] / ((1 + (m == 0)) * pi))
        for m in range(NQuad)
    ]
    TEMPER = np.array([300, 200])
    WVNMLO = 0
    WVNMHI = 80000
    BTEMP = 320
    TTEMP = 100
    # Emissivity is (1 - omega_arr) by Kirchoff's law of thermal radiation
    s_poly_coeffs=generate_s_poly_coeffs(tau_arr, TEMPER, WVNMLO, WVNMHI, epsrel=1e-15) * (1 - omega_arr)
    # The emissivity of the surface should be consistent with the BDRF 
    # in accordance with Kirchoff's law of thermal radiation
    emissivity = generate_emissivity_from_BDRF(NQuad // 2, BDRF_Fourier_modes[0])
    b_pos = emissivity * blackbody_contrib_to_BCs(BTEMP, WVNMLO, WVNMHI)
    b_neg = blackbody_contrib_to_BCs(TTEMP, WVNMLO, WVNMHI, epsrel=1e-15) + 100 # Emissivity 1
    
    f_arr = Leg_coeffs_all[NQuad]
    only_flux = True

    # Optional (unused)
    NLeg = None
    NFourier = None
    NT_cor = False
    use_banded_solver_NLayers = 10
    autograd_compatible=False

    ####################################################################################################

    # Call pydisort function
    mu_arr, flux_up, flux_down, u0 = PythonicDISORT.pydisort(
        tau_arr, omega_arr,
        NQuad,
        Leg_coeffs_all,
        mu0, I0, phi0,
        b_pos=b_pos,
        b_neg=b_neg,
        s_poly_coeffs=s_poly_coeffs,
        BDRF_Fourier_modes=BDRF_Fourier_modes,
        f_arr=f_arr,
        only_flux=only_flux,
    )
    
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

    
    # Load results from version 4.0.99 of Stamnes' DISORT for comparison
    results = np.load("Stamnes_results/7e_test.npz")
    
    # Perform the comparisons
    (
        diff_flux_up,
        ratio_flux_up,
        diff_flux_down_diffuse,
        ratio_flux_down_diffuse,
        diff_flux_down_direct,
        ratio_flux_down_direct,
        #diff,
        #diff_ratio,
    ) = _compare(results, mu_to_compare, reorder_mu, flux_up, flux_down)
    
    assert np.max(ratio_flux_up[diff_flux_up > 1e-3], initial=0) < 1e-3
    assert np.max(ratio_flux_down_diffuse[diff_flux_down_diffuse > 1e-3], initial=0) < 1e-3
    assert np.max(ratio_flux_down_direct[diff_flux_down_direct > 1e-3], initial=0) < 1e-3
    # --------------------------------------------------------------------------------------------------
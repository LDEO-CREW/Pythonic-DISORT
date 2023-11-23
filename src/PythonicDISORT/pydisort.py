from PythonicDISORT import subroutines
from PythonicDISORT._loop_and_assemble_results import _loop_and_assemble_results
import scipy as sc
from math import pi
from numpy.polynomial.legendre import Legendre
from scipy import integrate
try:
    import autograd.numpy as np
except ImportError:
    import numpy as np
    

def pydisort(
    tau_arr, omega_arr,
    NQuad,
    Leg_coeffs_all,
    mu0, I0, phi0,
    NLeg=None, 
    NLoops=None,
    b_pos=0, 
    b_neg=0,
    only_flux=False,
    f_arr=0, 
    NT_cor=False,
    BDRF_Fourier_modes=[],
    s_poly_coeffs=np.array([[]]),
    use_sparse_NLayers=13,
    n_jobs=1
):
    """Solves the 1D RTE for the fluxes, and optionally intensity,
    of a multi-layer atmosphere with the specified optical properties, boundary conditions
    and sources. Optionally performs delta-M scaling and NT corrections. 
    
        See https://pythonic-disort.readthedocs.io/en/latest/Pythonic-DISORT.html#1.-USER-INPUT-REQUIRED:-Choose-parameters
        for a more detailed explanation of each parameter.
        See https://pythonic-disort.readthedocs.io/en/latest/Pythonic-DISORT.html#2.-PythonicDISORT-modules-and-outputs
        for a more detailed explanation of each output.
        The notebook also has numerous examples of this function being called.

    Parameters
    ----------
    tau_arr : array
        Optical depth of the lower boundary of each atmospheric layer.
    omega_arr : array
        Single-scattering albedo of each atmospheric layer.
    NQuad : int
        Number of mu quadrature nodes.
    Leg_coeffs_all : ndarray
        All available unweighted phase function Legendre coefficients.
    mu0 : float
        Cosine of polar angle of the incident beam.
    I0 : float
        Intensity of the incident beam.
    phi0 : float
        Azimuthal angle of the incident beam.
    NLeg : optional, int
        Number of phase function Legendre coefficients.
    NLoops : optional, int
        Number of outermost loops to perform, also number of Fourier modes in the numerical solution.
    b_pos : optional, 2darray or float
        Dirichlet boundary condition for the upward direction.
    b_neg : optional, 2darray or float
        Dirichlet boundary condition for the downward direction.
    only_flux : optional, bool
        Do NOT compute the intensity function?
    f_arr : optional, array
        Fractional scattering into peak for each atmospheric layer.
    NT_cor : optional, bool
        Perform Nakajima-Tanaka intensity corrections?
    BDRF_Fourier_modes : optional, list
        BDRF Fourier modes, each a function with arguments mu, -mup of type array
        and which output has the same dimensions as the outer product of the two arrays.
    s_poly_coeffs : optional, array
        Polynomial coefficients of isotropic internal sources.
        Arrange coefficients from lowest order term to highest.
    use_sparse_NLayers : optional, int
        At or above how many atmospheric layers should SciPy's sparse matrix framework be used?
    n_jobs: optional, int
        Maximum number of concurrently running jobs during parallelization.
        This is exactly the `n_jobs` argument in `joblib.Parallel`, see 
        https://joblib.readthedocs.io/en/stable/generated/joblib.Parallel.html
        and https://joblib.readthedocs.io/en/stable/parallel.html.

    Returns
    -------
    array
        All mu (cosine of polar angle) quadrature nodes.
    function
        (Energetic) Flux function with argument tau (type: array) for positive (upward) mu values.
        Returns the diffuse flux magnitudes (type: array).
    function
        (Energetic) Flux function with argument tau (type: array) for negative (downward) mu values.
        Returns a tuple of the diffuse and direct flux magnitudes respectively (type: (array, array)).
    function
        Zeroth Fourier mode of the intensity with argument tau (type: array).
        Returns an ndarray with axes corresponding to variation with mu and tau respectively.
        This function is useful for calculating actinic fluxes and other quantities of interest,
        but reclassification of delta-scaled flux and other corrections must be done manually.
    function, optional
        Intensity function with arguments (tau, phi, return_Fourier_error=False) of types (array, array, bool).
        Returns an ndarray with axes corresponding to variation with mu, tau, phi respectively.
        The optional flag `return_Fourier_error` determines whether the function will also return
        the Cauchy / Fourier convergence evaluation (type: float) for the last Fourier term.

    """
    # Turn scalars into arrays
    # --------------------------------------------------------------------------------------------------------------------------
    tau_arr = np.atleast_1d(tau_arr)
    omega_arr = np.atleast_1d(omega_arr)
    Leg_coeffs_all = np.atleast_2d(Leg_coeffs_all)
    s_poly_coeffs = np.atleast_2d(s_poly_coeffs)
    f_arr = np.atleast_1d(f_arr)
    # --------------------------------------------------------------------------------------------------------------------------
    
    # Setup
    # --------------------------------------------------------------------------------------------------------------------------
    if NLeg is None:
        NLeg = NQuad
    if NLoops is None:
        NLoops = NQuad
    if np.all(b_pos == 0):
        b_pos = 0
    if np.all(b_neg == 0):
        b_neg = 0
    if np.all(s_poly_coeffs == 0):
        Nscoeffs = 0
    else:
        Nscoeffs = np.shape(s_poly_coeffs)[1]
    NLayers = len(tau_arr)
    scalar_b_pos = False
    scalar_b_neg = False
    thickness_arr = np.concatenate([[tau_arr[0]], np.diff(tau_arr)])
    NLeg_all = np.shape(Leg_coeffs_all)[1]
    N = NQuad // 2
    # --------------------------------------------------------------------------------------------------------------------------

    # Input checks
    # --------------------------------------------------------------------------------------------------------------------------
    # Optical depths and thickness must be positive
    assert np.all(tau_arr > 0)
    assert np.all(thickness_arr > 0)
    # Single-scattering albedo must be between 0 and 1, excluding 1
    assert np.all(omega_arr >= 0)
    assert np.all(omega_arr < 1)
    # There must be a positive number of Legendre coefficients each with magnitude <= 1
    # The user must supply at least as many phase function Legendre coefficients as intended for use
    assert NLeg > 0
    assert np.all(Leg_coeffs_all[:, 0] == 1)
    assert np.all(np.abs(Leg_coeffs_all) <= 1)
    assert np.all(np.array([f.__code__.co_argcount for f in BDRF_Fourier_modes]) == 2)
    assert NLeg <= NLeg_all
    # Ensure that the first dimension of the following inputs corresponds to the number of layers
    assert np.shape(Leg_coeffs_all)[0] == NLayers
    assert len(omega_arr) == NLayers
    if len(f_arr) != 1 or f_arr[0] != 0:
        assert len(f_arr) == NLayers
    if Nscoeffs > 0:
        assert np.shape(s_poly_coeffs)[0] == NLayers
    # Conditions on the number of quadrature angles (NQuad), Legendre coefficients (NLeg) and loops (NLoops)
    assert NQuad >= 2
    assert NQuad % 2 == 0
    assert NLoops > 0
    assert NLoops <= NLeg
    # Not strictly necessary but there will be tremendous inaccuracies if this is violated
    assert NQuad >= NLeg
    # We require principal angles and a downward incident beam
    assert I0 >= 0
    if I0 > 0:
        assert 0 < mu0 and mu0 <= 1
        assert 0 <= phi0 and phi0 < 2 * pi
    # Ensure that the BC inputs are of the correct shape
    if len(np.atleast_1d(b_pos)) == 1:
        scalar_b_pos = True
    else:
        assert np.shape(b_pos) == (N, NLoops)
    if len(np.atleast_1d(b_neg)) == 1:
        scalar_b_neg = True
    else:
        assert np.shape(b_neg) == (N, NLoops)
    # The fractional scattering must be between 0 and 1
    assert np.all(0 <= f_arr) and np.all(f_arr <= 1)
    # The minimum threshold is the minimum numbers of layers: 1
    assert(use_sparse_NLayers >= 1)
    # --------------------------------------------------------------------------------------------------------------------------
    
    # Some more setup
    # --------------------------------------------------------------------------------------------------------------------------
    NBDRF = len(BDRF_Fourier_modes)
    weighted_Leg_coeffs_all = (2 * np.arange(NLeg_all) + 1) * Leg_coeffs_all
    Leg_coeffs = Leg_coeffs_all[:, :NLeg]
    if (scalar_b_pos and b_pos == 0) and (scalar_b_neg and b_neg == 0) and Nscoeffs == 0 and I0 > 0:
        I0_orig = I0
        I0 = 1
    else:
        I0_orig = 1
    # --------------------------------------------------------------------------------------------------------------------------
    
    # Generation of Double Gauss-Legendre quadrature weights and points
    # --------------------------------------------------------------------------------------------------------------------------
    # For positive mu values (the weights are identical for both domains)
    mu_arr_pos, W = subroutines.Gauss_Legendre_quad(N)  # mu_arr_neg = -mu_arr_pos
    mu_arr = np.concatenate([mu_arr_pos, -mu_arr_pos])
    M_inv = 1 / mu_arr_pos
    
    # We do not allow mu0 to equal a quadrature / computational angle
    assert not np.any(np.abs(mu_arr_pos - mu0) < 1e-8)
    # --------------------------------------------------------------------------------------------------------------------------

    # Delta-M scaling; there is no scaling if f = 0
    # --------------------------------------------------------------------------------------------------------------------------
    if np.any(f_arr > 0):
        scale_tau = 1 - omega_arr * f_arr
        scaled_thickness_arr = scale_tau * thickness_arr
        scaled_tau_arr_with_0 = np.array(
            list(map(lambda l: np.sum(scaled_thickness_arr[:l]), range(NLayers + 1)))
        )
        weighted_scaled_Leg_coeffs = ((Leg_coeffs - f_arr[:, None]) / (1 - f_arr[:, None])) * (
            2 * np.arange(NLeg) + 1
        )[None, :]
        scaled_omega_arr = (1 - f_arr) / scale_tau * omega_arr
    else:
        # This is a shortcut to the same results
        scale_tau = np.ones(NLayers)
        scaled_tau_arr_with_0 = np.concatenate([[0], tau_arr])
        weighted_scaled_Leg_coeffs = Leg_coeffs * (2 * np.arange(NLeg) + 1)[None, :]
        scaled_omega_arr = omega_arr
    # --------------------------------------------------------------------------------------------------------------------------
    
    if NT_cor and not only_flux and I0 > 0 and np.any(f_arr > 0) and NLeg < NLeg_all:
        
        ############################### Perform NT corrections on the intensity but not the flux ###################################
        
        # Delta-M scaled solution; no further corrections to the flux
        flux_up, flux_down, u0, u_star = _loop_and_assemble_results(
            scaled_omega_arr,
            tau_arr,
            scaled_tau_arr_with_0,
            mu_arr_pos, mu_arr,
            M_inv, W,
            N, NQuad, NLeg, NLoops,
            NLayers, NBDRF,
            weighted_scaled_Leg_coeffs,
            BDRF_Fourier_modes,
            mu0, I0, I0_orig, phi0,
            b_pos, b_neg,
            scalar_b_pos, scalar_b_neg,
            s_poly_coeffs,
            Nscoeffs,
            scale_tau,
            False,
            use_sparse_NLayers,
            n_jobs
        )
        
        # TMS correction
        # --------------------------------------------------------------------------------------------------------------------------
        def TMS_correction(tau, phi):
            Ntau = len(tau)
            Nphi = len(phi)
            # Atmospheric layer indices
            l = np.argmax(tau[:, None] <= tau_arr[None, :], axis=1)
            scaled_tau_arr_l = scaled_tau_arr_with_0[l + 1]
            scaled_tau_arr_lm1 = scaled_tau_arr_with_0[l]

            # Delta-M scaling
            if np.any(scale_tau != np.ones(NLayers)):
                tau_dist_from_top = tau_arr[l] - tau
                scaled_tau_dist_from_top = tau_dist_from_top * scale_tau[l]
                scaled_tau = scaled_tau_arr_l - scaled_tau_dist_from_top
            else:
                scaled_tau = tau
            
            # mathscr_B
            # --------------------------------------------------------------------------------------------------------------------------
            nu = subroutines.atleast_2d_append(
                subroutines.calculate_nu(mu_arr, phi, -mu0, phi0)
            )
            p_true = np.concatenate(
                [
                    f(nu)[:, None, :]
                    # Iterates over the 0th axis
                    for f in map(Legendre, weighted_Leg_coeffs_all)
                ],
                axis=1,
            )
            p_trun = np.concatenate(
                [
                    f(nu)[:, None, :]
                    # Iterates over the 0th axis
                    for f in map(Legendre, weighted_scaled_Leg_coeffs)
                ],
                axis=1,
            )
            mathscr_B = (
                (scaled_omega_arr * I0)[None, :, None]
                / (4 * np.pi)
                * (mu0 / (mu0 + mu_arr))[:, None, None]
                * (p_true / (1 - f_arr[None, :, None]) - p_trun)
            )
            mathscr_B_pos = mathscr_B[:N, :, :]
            mathscr_B_neg = mathscr_B[N:, :, :]
            # --------------------------------------------------------------------------------------------------------------------------
            
            TMS_correction_pos = (
                np.exp(-scaled_tau / mu0)[None, :]
                - np.exp(
                    (scaled_tau - scaled_tau_arr_l)[None, :] / mu_arr_pos[:, None]
                    - scaled_tau_arr_l[None, :] / mu0
                )
            )
            TMS_correction_neg = (
                np.exp(-scaled_tau / mu0)[None, :]
                - np.exp(
                    (scaled_tau_arr_lm1 - scaled_tau)[None, :] / mu_arr_pos[:, None]
                    - scaled_tau_arr_lm1[None, :] / mu0
                )
            )
            
            # Contribution from other layers
            # --------------------------------------------------------------------------------------------------------------------------
            if NLayers > 1:

                def Contribution_from_layer(j):
                    contribution = np.zeros((NQuad, Ntau, Nphi))
                    pos_contribution_mask = l < j
                    neg_contribution_mask = l > j
                    scaled_tau_l_pos = scaled_tau[None, pos_contribution_mask]
                    scaled_tau_l_neg = scaled_tau[None, neg_contribution_mask]
                    if np.any(pos_contribution_mask):
                        contribution[:N, pos_contribution_mask, :] = (
                            mathscr_B_pos[:, [j], :]
                            * (
                                np.exp(
                                    (scaled_tau_l_pos - scaled_tau_arr_with_0[j])
                                    / mu_arr_pos[:, None]
                                    - scaled_tau_arr_with_0[j] / mu0
                                )
                                - np.exp(
                                    (scaled_tau_l_pos - scaled_tau_arr_with_0[j + 1])
                                    / mu_arr_pos[:, None]
                                    - scaled_tau_arr_with_0[j + 1] / mu0
                                )
                            )[:, :, None]
                        )
                    if np.any(neg_contribution_mask):
                        contribution[N:, neg_contribution_mask, :] = (
                            mathscr_B_neg[:, [j], :]
                            * (
                                np.exp(
                                    (scaled_tau_arr_with_0[j + 1] - scaled_tau_l_neg)
                                    / mu_arr_pos[:, None]
                                    - scaled_tau_arr_with_0[j + 1] / mu0
                                )
                                - np.exp(
                                    (scaled_tau_arr_with_0[j] - scaled_tau_l_neg)
                                    / mu_arr_pos[:, None]
                                    - scaled_tau_arr_with_0[j] / mu0
                                )
                            )[:, :, None]
                        )
                    return contribution

                Contribution_from_other_layers = np.sum(
                    list(map(Contribution_from_layer, range(NLayers))), axis=0
                )
                return (
                    mathscr_B[:, l, :]
                    * np.vstack((TMS_correction_pos, TMS_correction_neg))[:, :, None]
                    + Contribution_from_other_layers
                )
            # --------------------------------------------------------------------------------------------------------------------------
                
            else:
                return (
                    mathscr_B[:, l, :]
                    * np.vstack((TMS_correction_pos, TMS_correction_neg))[:, :, None]
                )
        # --------------------------------------------------------------------------------------------------------------------------

        # IMS correction
        # --------------------------------------------------------------------------------------------------------------------------
        sum1 = np.sum(omega_arr * tau_arr)
        omega_avg = sum1 / np.sum(tau_arr)
        sum2 = np.sum(f_arr * omega_arr * tau_arr)
        f_avg = sum2 / sum1
        Leg_coeffs_residue = Leg_coeffs_all.copy()
        Leg_coeffs_residue[:, :NLeg] = np.tile(f_arr, (NLeg, 1)).T
        Leg_coeffs_residue_avg = (
            np.sum(Leg_coeffs_residue * omega_arr[:, None] * tau_arr[:, None], axis=0)
            / sum2
        )
        scaled_mu0 = mu0 / (1 - omega_avg * f_avg)

        def IMS_correction(tau, phi):
            nu = subroutines.atleast_2d_append(
                subroutines.calculate_nu(-mu_arr_pos, phi, -mu0, phi0)
            )
            x = 1 / mu_arr_pos - 1 / scaled_mu0
            chi = (1 / (mu_arr_pos * scaled_mu0 * x))[:, None] * (
                (tau[None, :] - 1 / x[:, None]) * np.exp(-tau / scaled_mu0)[None, :]
                + np.exp(-tau[None, :] / mu_arr_pos[:, None]) / x[:, None]
            )
            return (
                I0
                / (4 * pi)
                * (omega_avg * f_avg) ** 2
                / (1 - omega_avg * f_avg)
                * Legendre(
                    (2 * np.arange(NLeg_all) + 1)
                    * (2 * Leg_coeffs_residue_avg - Leg_coeffs_residue_avg**2)
                )(nu)
            )[:, None, :] * chi[:, :, None]
        # --------------------------------------------------------------------------------------------------------------------------

        # The corrected intensity
        # --------------------------------------------------------------------------------------------------------------------------
        def u_corrected(tau, phi, return_Fourier_error=False):
            tau = np.atleast_1d(tau)
            phi = np.atleast_1d(phi)
            NT_corrections = TMS_correction(tau, phi)

            # We provide two options below, comment and uncomment as desired.
            # Option 2 is more computationally efficient but would prevent the use of autograd for testing.

            #NT_corrections = NT_corrections + np.concatenate(
            #    [np.zeros((N, len(tau), len(phi))), IMS_correction(tau, phi)], axis=0
            #)  # Option 1

            NT_corrections[N:, :, :] += IMS_correction(tau, phi)  # Option 2
               
            if return_Fourier_error:
                u_star_outputs = u_star(tau, phi, True)
                return (
                    u_star_outputs[0] + I0_orig * np.squeeze(NT_corrections),
                    u_star_outputs[1],
                )
            else:
                return u_star(tau, phi, False) + I0_orig * np.squeeze(NT_corrections)
        # --------------------------------------------------------------------------------------------------------------------------

        return mu_arr, flux_up, flux_down, u0, u_corrected
        
    else:
        return (mu_arr,) + _loop_and_assemble_results(
            scaled_omega_arr,
            tau_arr,
            scaled_tau_arr_with_0,
            mu_arr_pos, mu_arr,
            M_inv, W,
            N, NQuad, NLeg, NLoops,
            NLayers, NBDRF,
            weighted_scaled_Leg_coeffs,
            BDRF_Fourier_modes,
            mu0, I0, I0_orig, phi0,
            b_pos, b_neg,
            scalar_b_pos, scalar_b_neg,
            s_poly_coeffs,
            Nscoeffs,
            scale_tau,
            only_flux,
            use_sparse_NLayers,
            n_jobs
        )
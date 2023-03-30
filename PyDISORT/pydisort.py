from PyDISORT import subroutines, _loop_and_assemble_results
import scipy as sc
from math import pi
from numpy.polynomial.legendre import Legendre
from scipy import integrate
from inspect import signature
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
    Leg_coeffs_BDRF=[],
    mathscr_vs=None,
    use_sparse_NLayers=6
):
    """Solves the 1D RTE for the fluxes, and optionally intensity,
    of a multilayer atmosphere with the specified optical properties, boundary conditions
    and sources. Optionally performs delta-M scaling and NT corrections.


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
        Polar angle of the incident beam.
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
        Do NOT compute the intensity function?.
    f_arr : optional, array
        Fractional scattering into peak for each atmospheric layer.
    NT_cor : optional, bool
        Perform Nakajima-Tanaka intensity corrections?
    Leg_coeffs_BDRF : optional, array
        Unweighted BDRF Legendre coefficients.
    mathscr_vs : optional, function
        Particular solution corresponding to the other internal sources.
        Must have arguments (tau_arr, int, 2darray, array, 2darray) and output (2darray).
    use_sparse_NLayers : optional, int
        At or above how many atmospheric layers should SciPy's sparse matrix framework be used?

    Returns
    -------
    array
        All mu (cosine of polar angle) quadrature nodes.
    function
        Flux function with argument tau (type: array) for positive (upward) mu values.
        The tau inputs must be sorted in ascending order.
        Returns the diffuse flux magnitudes (type: array).
    function
        Flux function with argument tau (type: array)  for negative (downward) mu values.
        The tau inputs must be sorted in ascending order.
        Returns a tuple of the diffuse and direct flux magnitudes respectively (type: (array, array)).
    function, optional
        Intensity function with arguments (tau, phi, return_Fourier_error=False) of types (array, array, bool).
        The tau inputs must be sorted in ascending order.
        The first and primary output is an ndarray with axes corresponding to (mu, tau, phi) variation.
        The optional flag `return_Fourier_error` determines whether the function will also output
        the Cauchy / Fourier convergence evaluation (type: float) for the last Fourier term.

    """
    # Turn scalars to arrays
    # --------------------------------------------------------------------------------------------------------------------------
    tau_arr = np.atleast_1d(tau_arr)
    omega_arr = np.atleast_1d(omega_arr)
    Leg_coeffs_all = np.atleast_2d(Leg_coeffs_all)
    f_arr = np.atleast_1d(f_arr)
    # --------------------------------------------------------------------------------------------------------------------------

    # Some setup
    # --------------------------------------------------------------------------------------------------------------------------
    if NLeg == None:
        NLeg = NQuad
    if NLoops == None:
        NLoops = NQuad
    Leg_coeffs = Leg_coeffs_all[:, :NLeg]
    scalar_b_pos, scalar_b_neg = False, False
    mathscr_vs_callable = callable(mathscr_vs)
    thickness_arr = np.concatenate([[tau_arr[0]], np.diff(tau_arr)])
    NLayers = len(tau_arr)
    NBDRF = len(Leg_coeffs_BDRF)
    NLeg_all = np.shape(Leg_coeffs_all)[1]
    weighted_Leg_coeffs_BDRF = (2 * np.arange(NBDRF) + 1) * Leg_coeffs_BDRF
    weighted_Leg_coeffs_all = (2 * np.arange(NLeg_all) + 1) * Leg_coeffs_all
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
    assert np.all(np.abs(Leg_coeffs_all) <= 1)
    assert np.all(np.abs(Leg_coeffs_BDRF) <= 1)
    assert NLeg <= NLeg_all
    # Conditions on the number of quadrature angles (NQuad), Legendre coefficients (NLeg) and loops (NLoops)
    assert NQuad >= 2
    assert NQuad % 2 == 0
    assert NLoops > 0
    assert NLoops <= NLeg
    # Not strictly necessary but there will be tremendous inaccuracies if this is violated
    assert NQuad >= NLeg
    N = NQuad // 2
    # We require principal angles and a downward incident beam
    assert 0 < mu0 and mu0 < 1
    assert 0 <= phi0 and phi0 < 2 * pi
    # This ensures that the BC inputs are of the correct shape
    # Ensure that the BC inputs are of the correct shape
    if len(np.atleast_1d(b_pos)) == 1:
        scalar_b_pos = True
    else:
        assert np.shape(b_pos) == (N, NLoops)

    if len(np.atleast_1d(b_neg)) == 1:
        scalar_b_neg = True
    else:
        assert np.shape(b_neg) == (N, NLoops)
    # The fractional scattering must be between 0 and 1, excluding 1
    assert np.all(0 <= f_arr < 1)
    # This checks that mathscr_vs has the correct number of arguments and
    # that mathscr_vs(tau_BoA) = 0
    if mathscr_vs_callable:
        assert len(signature(mathscr_vs).parameters) == 6
        assert np.allclose(
            mathscr_vs(
                tau_arr[-1],
                tau_arr[-1],
                NQuad,
                np.random.random((NQuad, NQuad)),
                np.random.random(NQuad),
                np.random.random((NQuad, NQuad)),
            ),
            0,
            atol=1e-3, # Tolerance may need to be adjusted
        )
    # The minimum threshold is the minimum numbers of layers: 1
    assert(use_sparse_NLayers >= 1)
    # --------------------------------------------------------------------------------------------------------------------------
    
    # Generation of Double Gauss-Legendre quadrature weights and points
    # --------------------------------------------------------------------------------------------------------------------------
    # For positive mu values (the weights are identical for both domains)
    mu_arr_pos, weights_mu = PyDISORT.subroutines.Gauss_Legendre_quad(N)  # mu_arr_neg = -mu_arr_pos
    mu_arr = np.concatenate([mu_arr_pos, -mu_arr_pos])
    full_weights_mu = np.concatenate([weights_mu, weights_mu])
    # We do not allow mu0 to equal a quadrature / computational angle
    assert not np.any(np.isclose(mu_arr_pos, mu0))
    # Some more setup
    M_inv = 1 / mu_arr_pos
    W = weights_mu[None, :]
    # --------------------------------------------------------------------------------------------------------------------------

    # Delta-M scaling; there is no scaling if f = 0
    # --------------------------------------------------------------------------------------------------------------------------
    scale_tau = 1 - omega_arr * f_arr
    scaled_thickness_arr = scale_tau * thickness_arr
    scaled_tau_arr_with_0 = np.array(
        list(map(lambda l: np.sum(scaled_thickness_arr[:l]), range(NLayers + 1)))
    )
    weighted_Leg_coeffs = ((Leg_coeffs - f_arr[:, None]) / (1 - f_arr[:, None])) * (
        2 * np.arange(NLeg) + 1
    )[None, :]
    scaled_omega_arr = (1 - f_arr) / scale_tau * omega_arr
    # --------------------------------------------------------------------------------------------------------------------------
    
    if NT_cor and not only_flux and I0 != 0 and np.any(f_arr > 0) and NLeg < NLeg_all:
    
        ############################### Perform NT corrections on the intensity but not the flux ###################################
        
        # Delta-M scaled solution; no further corrections to the flux
        u_star, flux_up, flux_down = PyDISORT._loop_and_assemble_results(
            scaled_omega_arr,
            tau_arr,
            scaled_tau_arr_with_0,
            mu_arr_pos, mu_arr, weights_mu,
            M_inv, W,
            N, NQuad, NLeg, NLoops,
            NLayers, NBDRF,
            weighted_Leg_coeffs,
            weighted_Leg_coeffs_BDRF,
            mu0, I0, phi0,
            b_pos, b_neg,
            mathscr_vs,
            mathscr_vs_callable,
            scale_tau,
            False,
            use_sparse_NLayers
        )

        # NT (TMS) correction for the intensity
        # --------------------------------------------------------------------------------------------------------------------------
        def mathscr_B(phi, l_unique):
            nu = PyDISORT.subroutines.atleast_2d_append(
                PyDISORT.subroutines.calculate_nu(mu_arr, phi, -mu0, phi0)
            )

            p_true = np.concatenate(
                [
                    f(nu)[:, None, :]
                    for f in map(Legendre, iter(weighted_Leg_coeffs_all[l_unique, :]))
                ],
                axis=1,
            )
            p_trun = np.concatenate(
                [
                    f(nu)[:, None, :] 
                    for f in map(Legendre, iter(weighted_Leg_coeffs[l_unique, :]))
                ],
                axis=1,
            )

            return (
                (scaled_omega_arr[None, l_unique, None] * I0)
                / (4 * np.pi)
                * (mu0 / (mu0 + mu_arr))[:, None, None]
                * (p_true / (1 - f_arr[None, l_unique, None]) - p_trun)
            )

        def TMS_correction(tau, phi):
            # Atmospheric layer indices
            l = np.argmax(tau[:, None] <= tau_arr[None, :], axis=1)
            l_unique = np.unique(l)
            scaled_tau_arr_l = scaled_tau_arr_with_0[l + 1]
            scaled_tau_arr_with_0_l = scaled_tau_arr_with_0[l]

            # Delta-M scaling
            tau_dist_from_top = tau_arr[l] - tau
            scaled_tau_dist_from_top = tau_dist_from_top * scale_tau[l]
            scaled_tau = scaled_tau_arr_l - scaled_tau_dist_from_top

            TMS_correction_pos = np.exp(-scaled_tau / mu0)[None, :] - np.exp(
                (scaled_tau - scaled_tau_arr_l)[None, :] / mu_arr_pos[:, None]
                - scaled_tau_arr_l[None, :] / mu0
            )
            TMS_correction_neg = np.exp(-scaled_tau / mu0)[None, :] - np.exp(
                (scaled_tau_arr_with_0_l - scaled_tau)[None, :] / mu_arr_pos[:, None]
                - scaled_tau_arr_with_0_l[None, :] / mu0
            )

            return np.squeeze(
                mathscr_B(phi, l_unique)[:, l, :]
                * np.concatenate([TMS_correction_pos, TMS_correction_neg])[:, :, None],
                axis=0,
            )
        # --------------------------------------------------------------------------------------------------------------------------

        # NT (IMS) correction for the intensity
        # --------------------------------------------------------------------------------------------------------------------------
        sum1 = np.sum(omega_arr * tau_arr)
        omega_avg = sum1 / np.sum(tau_arr)
        sum2 = np.sum(f_arr * omega_arr * tau_arr)
        f_avg = sum2 / sum1
        Leg_coeffs_residue = Leg_coeffs_all.copy()
        Leg_coeffs_residue[:, :NLeg] = np.tile(f_arr, (1, NLeg))
        Leg_coeffs_residue_avg = (
            np.sum(Leg_coeffs_residue * omega_arr[:, None] * tau_arr[:, None], axis=0)
            / sum2
        )
        scaled_mu0 = mu0 / (1 - omega_avg * f_avg)

        def chi(tau):
            x = 1 / mu_arr_pos - 1 / scaled_mu0
            return (1 / (mu_arr_pos * scaled_mu0 * x))[:, None] * (
                (tau[None, :] - 1 / x[:, None]) * np.exp(-tau / scaled_mu0)[None, :]
                + np.exp(-tau[None, :] / mu_arr_pos[:, None]) / x[:, None]
            )

        def IMS_correction(tau, phi):
            nu = PyDISORT.subroutines.atleast_2d_append(
                PyDISORT.subroutines.calculate_nu(-mu_arr_pos, phi, -mu0, phi0)
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
            )[:, None, :] * chi(tau)[:, :, None]
        # --------------------------------------------------------------------------------------------------------------------------

        # The corrected intensity
        # --------------------------------------------------------------------------------------------------------------------------
        def u_corrected(tau, phi, return_Fourier_error=False):
            tau = np.atleast_1d(tau)
            if return_Fourier_error:
                u_star_outputs = u_star(tau, phi, True)
                return (
                    u_star_outputs[0] + TMS_correction(tau, phi) + IMS_correction(tau, phi),
                    u_star_outputs[1],
                )
            else:
                return (
                    u_star(tau, phi, False)
                    + TMS_correction(tau, phi)
                    + IMS_correction(tau, phi)
                )

        return mu_arr, flux_up, flux_down, u_corrected
        # --------------------------------------------------------------------------------------------------------------------------

    else:
        return (mu_arr,) + PyDISORT._loop_and_assemble_results(
            scaled_omega_arr,
            tau_arr,
            scaled_tau_arr_with_0,
            mu_arr_pos, mu_arr, weights_mu,
            M_inv, W,
            N, NQuad, NLeg, NLoops,
            NLayers, NBDRF,
            weighted_Leg_coeffs,
            weighted_Leg_coeffs_BDRF,
            mu0, I0, phi0,
            b_pos, b_neg,
            mathscr_vs,
            mathscr_vs_callable,
            scale_tau,
            only_flux,
            use_sparse_NLayers
        )
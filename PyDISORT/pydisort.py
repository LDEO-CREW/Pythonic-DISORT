from PyDISORT import subroutines, _basic_solver

try:
    import autograd.numpy as np
except ImportError:
    import numpy as np
import scipy as sc
from math import pi
from numpy.polynomial.legendre import Legendre
from scipy import integrate
from inspect import signature


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
    parfor_Fourier=False
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
        Must have arguments (tau_arr, int, 2darray, array, 2darray) and output (2darray)..
    parfor_Fourier : optional, bool
        Parallelize the for-loop over Fourier modes?

    Returns
    -------
    array
        All mu (cosine of polar angle) quadrature nodes.
    function
        Flux function with argument tau (type: array) for positive (upward) mu values.
        Returns the diffuse flux magnitudes (type: array).
    function
        Flux function with argument tau (type: array)  for negative (downward) mu values.
        Returns a tuple of the diffuse and direct flux magnitudes respectively (type: (array, array)).
    function, optional
        Intensity function with arguments (tau, phi) of types (array, array).
        The output is an ndarray with axes corresponding to (mu, tau, phi) variation.

    """
    tau_arr = np.atleast_1d(tau_arr)
    omega_arr = np.atleast_1d(omega_arr)
    Leg_coeffs_all = np.atleast_2d(Leg_coeffs_all)
    f_arr = np.atleast_1d(f_arr)

    if NLeg == None:
        NLeg = NQuad
    if NLoops == None:
        NLoops = NQuad
    Leg_coeffs = Leg_coeffs_all[:, :NLeg]
    NLeg_all = np.shape(Leg_coeffs_all)[1]
    NLayers = len(tau_arr)
    scalar_b_pos, scalar_b_neg = False, False
    mathscr_vs_callable = callable(mathscr_vs)

    # INPUT CHECKS
    # -----------------------------------------------------------
    # Optical depth must be positive
    assert np.all(tau_arr > 0)
    # Single-scattering albedo must be between 0 and 1, excluding 1
    assert np.all(omega_arr >= 0)
    assert np.all(omega_arr < 1)
    # There must be a positive number of Legendre coefficients each with magnitude <= 1
    # We must obviously supply more Legendre coefficients than we intend to use
    assert NLeg > 0
    assert np.all(np.abs(Leg_coeffs_all) <= 1)
    assert NLeg <= NLeg_all
    # Conditions on the number of quadrature angles (NQuad), Legendre coefficients (NLeg) and loops (NLoops)
    assert NQuad >= 2
    assert NQuad % 2 == 0
    assert NLoops > 0
    assert NLoops <= NLeg
    assert (
        NQuad >= NLeg
    )  # Not strictly necessary but there will be tremendous inaccuracies if this is violated
    N = NQuad // 2
    # We require principal angles, a downward incident beam and
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
    assert np.all(np.abs(Leg_coeffs_BDRF) <= 1)
    if mathscr_vs_callable:
        assert len(signature(mathscr_vs).parameters) == 5
        assert np.allclose(
            mathscr_vs(
                np.array([tau_arr[-1]]),
                NQuad,
                np.random.random((NQuad, NQuad)),
                np.random.random(NQuad),
                np.random.random((NQuad, NQuad)),
            ),
            0,
        )
    # -----------------------------------------------------------

    # For positive mu values (the weights are identical for both domains)
    mu_arr_pos, weights_mu = PyDISORT.subroutines.Gauss_Legendre_quad(N)  # mu_arr_neg = -mu_arr_pos
    mu_arr = np.concatenate((mu_arr_pos, -mu_arr_pos))
    full_weights_mu = np.concatenate((weights_mu, weights_mu))
    # We do not allow mu0 to equal a quadrature / computational angle
    assert not np.any(np.isclose(mu_arr_pos, mu0))
    
    NBDRF = len(Leg_coeffs_BDRF)
    weighted_Leg_coeffs_BDRF = (2 * np.arange(NBDRF) + 1) * Leg_coeffs_BDRF

    # Delta-M scaling; there is no scaling if f = 0
    scale_tau = 1 - omega_arr * f_arr
    weighted_Leg_coeffs = ((Leg_coeffs - f_arr[:, None]) / (1 - f_arr[:, None])) * (
        2 * np.arange(NLeg) + 1
    )[None, :]
    omega_arr *= (1 - f_arr) / scale_tau

    # Perform NT correction on the intensity but not the flux
    if NT_cor and not only_flux and I0 != 0 and np.any(f_arr > 0) and NLeg < NLeg_all:
        # Delta-M scaled solution; no further corrections to the flux
        u_star, flux_up, flux_down = PyDISORT._basic_solver(
            tau_arr, omega_arr,
            N, NQuad, NLeg, NLoops,
            weighted_Leg_coeffs,
            mu0, I0, phi0,
            b_pos, b_neg,
            False,
            NBDRF,
            weighted_Leg_coeffs_BDRF,
            mathscr_vs,
            parfor_Fourier,
            mu_arr_pos, weights_mu,
            scale_tau
        )

        # NT (TMS; IMS below) correction for the intensity
        def mathscr_B(phi, l_unique):
            nu = PyDISORT.subroutines.atleast_2d_append(
                PyDISORT.subroutines.calculate_nu(mu_arr, phi, -mu0, phi0)
            )

            p_true = np.concatenate(
                [
                    f(nu)[:, None, :]
                    for f in map(
                        Legendre,
                        iter(
                            (2 * np.arange(NLeg_all) + 1)[None, :]
                            * Leg_coeffs_all[l_unique, :]
                        ),
                    )
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

            return (omega_arr[None, l_unique, None] * I0) / (4 * np.pi) * (
                mu0 / (mu0 + mu_arr)
            )[:, None, None] * p_true / (1 - f_arr[None, l_unique, None]) - p_trun

        def TMS_correction(tau, phi):
            l = np.argmax(tau[:, None] <= tau_arr[None, :], axis=1)
            l_unique = np.unique(l)

            # delta-M scaling
            tau *= scale_tau[l]
            tau_arr_l = tau_arr[l] * scale_tau[l]

            TMS_correction_pos = np.exp(-tau / mu0)[None, :] - np.exp(
                (tau - tau_arr_l)[None, :] * (1 / mu_arr_pos)[:, None]
                - tau_arr_l[None, :] / mu0
            )
            TMS_correction_neg = np.exp(-tau / mu0)[None, :] - np.exp(
                tau[None, :] * (-1 / mu_arr_pos)[:, None]
            )

            return np.squeeze(
                mathscr_B(phi, l_unique)[:, l, :]
                * np.concatenate((TMS_correction_pos, TMS_correction_neg))[:, :, None]
            )

        # NT (IMS; TMS above) correction for the intensity
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
        scaled_mu0 /= 1 - omega_avg * f_avg

        def chi(tau):
            x = 1 / mu - 1 / scaled_mu0
            return (1 / (mu * scaled_mu0 * x))[:, None] * (
                (tau[None, :] - 1 / x[:, None]) * np.exp(-tau / scaled_mu0)[None, :]
                + np.exp(-tau[None, :] / mu[:, None]) / x[:, None]
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

        # The corrected intensity
        def u_corrected(tau, phi):
            tau = np.atleast_1d(tau)
            return u_star + TMS_correction(tau, phi) + IMS_correction(tau, phi)

        return mu_arr, flux_up, flux_down, u_corrected

    else:  # Do not perform NT corrections
        return (mu_arr,) + PyDISORT._basic_solver(
            tau_arr, omega_arr,
            N, NQuad, NLeg, NLoops,
            weighted_Leg_coeffs,
            mu0, I0, phi0,
            b_pos, b_neg,
            only_flux,
            NBDRF,
            weighted_Leg_coeffs_BDRF,
            mathscr_vs,
            parfor_Fourier,
            mu_arr_pos, weights_mu,
            scale_tau
        )
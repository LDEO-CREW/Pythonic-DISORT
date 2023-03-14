import PyDISORT
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
    """Full radiative transfer solver with corrections and input checks.

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
    mu_arr : array
        All mu (cosine of polar angle) quadrature nodes.
    flux_up : function
        Flux function with argument tau (array) for positive (upward) mu values.
    flux_down : function
        Flux function with argument tau (array) for negative (downward) mu values.
    u : function, optional
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
    assert NLeg <= np.shape(Leg_coeffs_all)[0]
    # Conditions on the number of quadrature angles (NQuad), Legendre coefficients (NLeg) and loops (NLoops)
    assert NQuad >= 2
    assert NQuad % 2 == 0
    assert NLoops > 0
    assert NLoops <= NLeg
    assert NQuad >= NLeg # Not strictly necessary but there will be tremendous inaccuracies if this is violated
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
    mu_arr_pos, weights_mu = PyDISORT.subroutines.Gauss_Legendre_quad(N) # mu_arr_neg = -mu_arr_pos
    mu_arr = np.concatenate((mu_arr_pos, -mu_arr_pos))
    full_weights_mu = np.concatenate((weights_mu, weights_mu))
    # We do not allow mu0 to equal a quadrature / computational angle
    assert not np.any(np.isclose(mu_arr_pos, mu0))
    
    # Delta-M scaling; there is no scaling if f = 0
    scale_tau = 1 - omega_arr * f_arr
    Leg_coeffs = ((Leg_coeffs - f_arr[:, None]) / (1 - f_arr[:, None])) * (2 * np.arange(NLeg)[None, :] + 1)
    omega_arr *= (1 - f_arr) / scale_tau
    #tau_arr *= scale_tau

    # Perform NT correction on the intensity but not the flux
    if NT_cor and not only_flux and I0 != 0 and np.any(f_arr > 0) and NLeg < np.shape(Leg_coeffs_all)[0]:
        # Delta-M scaled solution; no further corrections to the flux
        u_star, flux_up, flux_down = PyDISORT.basic_solver._basic_solver(
            tau_arr, omega_arr,
            N, NQuad, NLeg, NLoops,
            Leg_coeffs_all,
            I0, mu0, phi0,
            b_pos, b_neg,
            False,
            BDRF_Leg_coeffs,
            mu_arr_pos, weights_mu,
            scale_tau
        )

        # NT (TMS) correction for the intensity
        def mathscr_B(tau, phi):
            nu = PyDISORT.subroutines.atleast_2d_append(
                PyDISORT.subroutines.calculate_nu(mu_arr, phi, -mu0, phi0)
            )
            l = np.argmax(tau[:, None] <= tau_arr[None, :], axis=1)
            l_unique = np.unique(l)

            p_true = np.concatenate(
                [f(nu)[:, None, :] for f in map(Legendre, iter(Leg_coeffs_all[l_unique, :]))],
                axis=1,
            )
            p_trun = np.concatenate(
                [
                    f(nu)[:, None, :]
                    for f in map(Legendre, iter(Leg_coeffs[l_unique, :]))
                ],
                axis=1,
            )

            return (
                (omega_arr[None, l_unique, None] * I0)
                / (4 * np.pi)
                * (mu0 / (mu0 + mu_arr))[:, None, None]
                * p_true
                / (1 - f_arr[None, l_unique, None])
                - p_trun
            )[:, l, :]  # The tau variation is due to the different atmospheric layers
        
        def TMS_correction(tau, phi):
            tau = scale_tau * np.atleast_1d(tau)  # Delta-M scaling
            l = np.argmax(tau[:, None] <= tau_arr[None, :], axis=1)
            l_unique = np.unique(l)

            TMS_correction_pos = np.exp(-tau / mu0)[None, :] - np.exp(
                (tau - tau_arr[l])[None, :] * (1 / mu_arr_pos)[:, None] - tau_arr[None, l] / mu0
            )
            TMS_correction_neg = np.exp(-tau / mu0)[None, :] - np.exp(
                tau[None, :] * (-1 / mu_arr_pos)[:, None]
            )

            return np.squeeze(
                mathscr_B(tau, phi)
                * np.concatenate((TMS_correction_pos, TMS_correction_neg))[:, :, None]
            )
            
        # NT (IMS) correction for the intensity
        sum1 = np.sum(omega_arr * tau_arr)
        omega_avg = sum1 / np.sum(tau_arr)
        sum2 = np.sum(omega_arr * f_arr * tau_arr)
        f_avg = sum2 / sum1
        Leg_coeffs_residue = Leg_coeffs_all.copy()
        Leg_coeffs_residue[:, :NLeg] = np.tile(f_arr, (1, NLeg))
        Leg_coeffs_residue_avg = (
            np.sum(Leg_coeffs_residue * omega_arr[:, None] * tau_arr[:, None], axis=0) / sum2
        )
        scaled_mu0 = mu0 / (1 - omega_avg * f_avg)
        
        #def chi(tau):
            
        
        
        # The corrected intensity
        u_corrected = (
            lambda tau, phi: u_star(tau, phi) + TMS_correction(tau, phi)
        )
        return mu_arr, flux_up, flux_down, u_corrected

    else: # Do not perform NT corrections
        return (mu_arr,) + PyDISORT.basic_solver._basic_solver(
            b_pos, b_neg,
            only_flux,
            N, NQuad, NLeg, NLoops,
            Leg_coeffs,
            mu_arr_pos, weights_mu,
            tau0, w0,
            mu0, phi0, I0, 
            scale_tau,
        )
import PyDISORT
try:
    import autograd.numpy as np
except ImportError:
    import numpy as np
import scipy as sc
import warnings
import PyDISORT
from math import pi
from numpy.polynomial import legendre
from scipy import integrate
from inspect import signature

def pydisort(
    b_pos, b_neg, only_flux, NQuad, NLoops, Leg_coeffs, tau0, w0, mu0, phi0, I0, f=0, p_for_NT=None,
):
    """Full radiative transfer solver which performs corrections
    
    :Input:
     - *b_pos / neg* (float matrix) - Boundary conditions for the upward / downward directions
     - *only_flux* (boolean) - Flag for whether to compute the intensity function
     - *NQuad* (integer) - Number of mu quadrature points
     - *NLoops* (integer) - Number of loops, also number of Fourier modes in the numerical solution
     - *Leg_coeffs* (float vector) - Phase function Legendre coefficients
     - *tau0* (float) - Optical depth
     - *w0* (float) - Single-scattering albedo
     - *mu0* (float) - Polar angle of the direct beam
     - *phi0* (float) - Azimuthal angle of the direct beam
     - *I0* (float) - Intensity of the direct beam
     :Optional:
     - *f* (float) - fractional scattering into the forward peak for use in delta-M scaling
     - *p_for_NT* (function) - True phase function with cosine of scattering angle as vector argument
     
     
    :Output:
     - *mu_arr* (float vector) - All mu values
     - *flux_up* (function) - Flux function with argument tau for positive (upward) mu values
     - *flux_down* (function) - Flux function with argument tau for negative (downward) mu values
     :Optional:
     - *u* (function) - Intensity function with arguments (tau, phi); the output is in the order (mu, tau, phi)
    """
    NLeg = len(Leg_coeffs)
    
    # INPUT CHECKS
    # -----------------------------------------------------------
    # Optical depth must be positive
    assert tau0 > 0
    # Single-scattering albedo must be between 0 and 1, excluding 1
    assert w0 >= 0
    assert w0 < 1
    # There must be a positive number of Legendre coefficients each with magnitude <= 1
    assert NLeg > 0
    assert np.all(np.abs(Leg_coeffs) <= 1)
    # Conditions on the number of quadrature angles (NQuad), Legendre coefficients (NLeg) and loops (NLoops)
    assert NQuad >= 2
    assert NQuad % 2 == 0
    assert NLoops > 0
    assert NLoops <= NLeg
    assert NQuad >= NLeg # Not strictly necessary but there will be tremendous inaccuracies if this is violated
    N = NQuad // 2
    # The fractional scattering must be between 0 and 1, excluding 1
    assert 0 <= f < 1
    # We require principal angles and a downward beam
    assert 0 < mu0 and mu0 < 1
    assert 0 <= phi0 and phi0 < 2 * pi
    # This ensures that the BC inputs are of the correct shape
    if len(np.atleast_1d(b_pos)) == 1:
        b_pos = np.full((N, NLoops), b_pos)
    else:
        assert np.shape(b_pos) == (N, NLoops)
    if len(np.atleast_1d(b_neg)) == 1:
        b_neg = np.full((N, NLoops), b_pos)
    else:
        assert np.shape(b_neg) == (N, NLoops)
    # -----------------------------------------------------------

    # For positive mu values (the weights are identical for both domains)
    mu_arr_pos, weights_mu = legendre.leggauss(N)
    mu_arr_pos = PyDISORT.subroutines.transform_interval(mu_arr_pos, 0, 1)  # mu_arr_neg = -mu_arr_pos
    weights_mu = PyDISORT.subroutines.transform_weights(weights_mu, 0, 1)
    # We do not allow mu0 to equal a quadrature / computational angle
    assert not np.any(np.isclose(mu_arr_pos, mu0))

    # Delta-M scaling; there is no scaling if f = 0
    scale_tau = 1 - w0 * f
    Leg_coeffs = (Leg_coeffs - f) / (1 - f) * (2 * np.arange(NLeg) + 1)
    w0 *= (1 - f) / scale_tau
    tau0 *= scale_tau

    # Perform NT correction on the intensity but not the flux
    if callable(p_for_NT) and not only_flux:
        # Verify that the true phase function has one argument
        assert len(signature(p_for_NT).parameters) == 1
        # We check that the last Legendre coefficient in Leg_coeffs matches the inputed phase function
        assert np.isclose(
            (1 / 2)
            * integrate.quad(
                lambda nu: p_for_NT(nu) * sc.special.eval_legendre(NLeg - 1, nu), -1, 1
            )[0],
            Leg_coeffs[NLeg - 1] / (2 * (NLeg - 1) + 1) * (1 - f) + f,
        )
        
        # Delta-M scaled solution; no further corrections to the flux
        u_star, flux_up, flux_down = PyDISORT.basic_solver(
            b_pos, b_neg,
            False,
            N, NQuad, NLeg, NLoops,
            Leg_coeffs,
            mu_arr_pos, weights_mu,
            tau0, w0,
            mu0, phi0, I0, 
            scale_tau,
        )

        # NT corrections for the intensity
        def p_for_NT_muphi(mu, phi, mu_p, phi_p):
            nu = PyDISORT.subroutines.calculate_nu(mu, phi, mu_p, phi_p)
            return p_for_NT(nu)

        def tilde_u_star1(tau, phi):
            tau = scale_tau * np.atleast_1d(tau)  # Delta-M scaling
            tilde_mathcal_B = lambda tau, mu, phi: (
                ((w0 * I0) / (4 * np.pi * (1 - f)))
                * (mu0 / (mu0 + mu))[:, None, None]
                * PyDISORT.subroutines.atleast_2d_append(p_for_NT_muphi(mu, phi, -mu0, phi0))[
                    :, None, :
                ]
            )
            tilde_u_star1_pos = (
                tilde_mathcal_B(tau, mu_arr_pos, phi)
                * (
                    np.exp(-tau / mu0)[None, :]
                    - np.exp((tau - tau0)[None, :] * (1 / mu_arr_pos)[:, None] - tau0 / mu0)
                )[:, :, None]
            )
            tilde_u_star1_neg = (
                tilde_mathcal_B(tau, -mu_arr_pos, phi)
                * (
                    np.exp(-tau / mu0)[None, :]
                    - np.exp(tau[None, :] * (-1 / mu_arr_pos)[:, None])
                )[:, :, None]
            )
            return np.squeeze(np.concatenate((tilde_u_star1_pos, tilde_u_star1_neg), axis=0))


        def p_truncated(mu, phi, mu_p, phi_p):
            nu = PyDISORT.subroutines.calculate_nu(mu, phi, mu_p, phi_p)
            return legendre.Legendre(Leg_coeffs)(nu)


        def u_star1(tau, phi):
            tau = scale_tau * np.atleast_1d(tau)  # Delta-M scaling
            mathcal_B = lambda tau, mu, phi: (
                ((w0 * I0) / (4 * np.pi))
                * (mu0 / (mu0 + mu))[:, None, None]
                * PyDISORT.subroutines.atleast_2d_append(p_truncated(mu, phi, -mu0, phi0))[
                    :, None, :
                ]
            )
            u_star1_pos = (
                mathcal_B(tau, mu_arr_pos, phi)
                * (
                    np.exp(-tau / mu0)[None, :]
                    - np.exp((tau - tau0)[None, :] * (1 / mu_arr_pos)[:, None] - tau0 / mu0)
                )[:, :, None]
            )
            u_star1_neg = (
                mathcal_B(tau, -mu_arr_pos, phi)
                * (
                    np.exp(-tau / mu0)[None, :]
                    - np.exp(tau[None, :] * (-1 / mu_arr_pos)[:, None])
                )[:, :, None]
            )
            return np.squeeze(np.concatenate((u_star1_pos, u_star1_neg), axis=0))


        # The corrected intensity
        u_corrected = (
            lambda tau, phi: u_star(tau, phi) - u_star1(tau, phi) + tilde_u_star1(tau, phi)
        )
        return np.concatenate((mu_arr_pos, -mu_arr_pos)), u_corrected, flux_up, flux_down

    else: # Do not perform NT corrections
        return (np.concatenate((mu_arr_pos, -mu_arr_pos)),) + PyDISORT.basic_solver(
            b_pos, b_neg,
            only_flux,
            N, NQuad, NLeg, NLoops,
            Leg_coeffs,
            mu_arr_pos, weights_mu,
            tau0, w0,
            mu0, phi0, I0, 
            scale_tau,
        )
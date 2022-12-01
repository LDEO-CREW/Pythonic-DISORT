import numpy as np
import scipy as sc
from math import pi
from numpy.polynomial import legendre

def PyDISORT(
    b_pos, b_neg, only_flux, NQuad, NLeg, Leg_coeffs_full, tau0, w0, mu0, phi0, I0, f=0, p_for_NT_with_g=(None,),
):
    """Full radiative transfer solver which performs corrections
    
    :Input:
     - *b_pos / neg* (matrix) - Boundary conditions for the upward / downward directions
     - *only_flux* (boolean) - Flag for whether to compute the intensity function
     - *NQuad* (integer) - Number of mu quadrature points
     - *NLeg* (integer) - Number of Legendre coefficients to be used in the basic solver
     - *Leg_coeffs_full* (vector) - Long vector of weighted phase function Legendre coefficients
     - *tau0* (float) - Optical depth
     - *w0* (float) - Single-scattering albedo
     - *mu0* (float) - Polar angle of the direct beam
     - *phi0* (float) - Azimuthal angle of the direct beam
     - *I0* (float) - Intensity of the direct beam
     :Optional:
     - *f* (float) - fractional scattering into the forward peak for use in delta-M scaling
     - *p_for_NT_with_g* (tuple: (function, float)) - Tuple: (True phase function, asymmetry factor)
     
     
    :Output:
     - *mu_arr* (vector) - All mu values
     - *flux_up* (function) - Flux function with argument tau for positive (upward) mu values
     - *flux_down* (vector) - Flux function with argument tau for negative (downward) mu values
     :Optional:
     - *u* (function) - Intensity function with arguments (tau, phi); the output is in the order (mu, tau, phi)
    """
    Leg_coeffs = Leg_coeffs_full[:NLeg]
    
    # INPUT CHECKS
    # -----------------------------------------------------------
    # The fractional scattering into peak must be 0<= and <1
    assert 0 <= f < 1
    # We require principal angles and a downward beam
    assert 0 < mu0 and mu0 < 1
    assert 0 <= phi0 and phi0 < 2 * pi
    # We require NQuad to be >= 2, and even
    # NLeg must obviously be >0
    assert NQuad % 2 == 0
    assert NQuad >= 2
    assert NLeg > 0

    N = NQuad // 2
    # The following ensures that the BC inputs are of the correct shape
    if len(np.atleast_1d(b_pos)) == 1:
        b_pos = np.full((N, NLeg), b_pos)
    else:
        assert np.shape(b_pos) == (N, NLeg)
    if len(np.atleast_1d(b_neg)) == 1:
        b_neg = np.full((N, NLeg), b_neg)
    else:
        assert np.shape(b_neg) == (N, NLeg)
    # -----------------------------------------------------------

    # For positive mu values (the weights are identical for both domains)
    mu_arr_pos, weights_mu = legendre.leggauss(N)
    mu_arr_pos = transform_interval(mu_arr_pos, 0, 1)  # mu_arr_neg = -mu_arr_pos
    weights_mu = transform_weights(weights_mu, 0, 1)
    # We do not allow mu0 to equal a quadrature / computational angle
    assert not np.any(np.isclose(mu_arr_pos, mu0))

    # Delta-M scaling; there is no scaling if f==0
    scale_tau = 1 - w0 * f
    scale_beam = 1 + w0 * f
    Leg_coeffs = (Leg_coeffs - f) / (1 - f)
    w0 *= (1 - f) / scale_tau
    tau0 *= scale_tau

    # Perform NT correction on the intensity but not the flux
    if callable(p_for_NT_with_g[0]) and not only_flux:
        # Verify that the true phase function has two arguments
        assert len(signature(p_for_NT_with_g[0]).parameters) == 2
        
        p_for_NT = p_for_NT_with_g[0]
        g = p_for_NT_with_g[1]
        
        # The magnitude of the asymmetry factor must be <1
        assert np.abs(g) < 1
        # We check that the last Legendre coefficient in Leg_coeffs matches the inputed phase function
        assert np.isclose(
            ((2 * (NLeg - 1) + 1) / 2)
            * integrate.quad(
                lambda nu: p_for_NT(nu, g) * sc.special.eval_legendre(NLeg - 1, nu), -1, 1
            )[0],
            Leg_coeffs[NLeg - 1] * (1 - f) + f,
        )
        
        # Delta-M scaled solution; no further corrections to the flux
        u_star, flux_up, flux_down = basic_solver(
            b_pos, b_neg,
            False,
            N, NQuad, NLeg,
            Leg_coeffs,
            mu_arr_pos, weights_mu,
            tau0, w0,
            mu0, phi0, I0, 
            scale_tau, scale_beam,
        )

        # NT corrections for the intensity
        def p_for_NT_muphi(mu, phi, mu_p, phi_p, g):
            nu = calculate_nu(mu, phi, mu_p, phi_p)
            return p_for_NT(nu, g)

        def tilde_u_star1(tau, phi):
            tau = scale_tau * np.atleast_1d(tau) # Delta-M scaling
            tilde_mathcal_B = lambda tau, mu_arr, phi: (
                ((w0 * I0) / (4 * np.pi * (1 - f)))
                * np.exp(-tau / mu0)[None, :, None]
                * atleast_2d_back(p_for_NT_muphi(mu_arr, phi, -mu0, phi0, g))[:, None, :]
                / (mu_arr / mu0 + 1)[:, None, None]
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
            nu = calculate_nu(mu, phi, mu_p, phi_p)
            return legendre.Legendre(Leg_coeffs)(nu)

        def u_star1(tau, phi):
            tau = scale_tau * np.atleast_1d(tau) # Delta-M scaling
            mathcal_B = lambda tau, mu_arr, phi: (
                ((w0 * I0) / (4 * np.pi))
                * np.exp(-tau / mu0)[None, :, None]
                * atleast_2d_back(p_truncated(mu_arr, phi, -mu0, phi0))[:, None, :]
                / (mu_arr / mu0 + 1)[:, None, None]
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
        u_corrected = lambda tau, phi: u_star(tau, phi) - u_star1(tau, phi) + tilde_u_star1(tau, phi)
        return np.concatenate((mu_arr_pos, -mu_arr_pos)), u_corrected, flux_up, flux_down

    else: # Do not perform NT corrections
        return (np.concatenate((mu_arr_pos, -mu_arr_pos)),) + basic_solver(
            b_pos, b_neg,
            only_flux,
            N, NQuad, NLeg,
            Leg_coeffs,
            mu_arr_pos, weights_mu,
            tau0, w0,
            mu0, phi0, I0, 
            scale_tau, scale_beam,
        )
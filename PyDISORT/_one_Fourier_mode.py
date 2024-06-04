from PyDISORT import subroutines
import numpy as np
import scipy as sc
from math import pi

def _one_Fourier_mode(
    m,
    tau_arr, omega_arr,
    N, NQuad, NLeg,
    weighted_Leg_coeffs,
    mu0, I0, phi0,
    b_pos, b_neg,
    NLayers, NBDRF,
    weighted_Leg_coeffs_BDRF,
    mathscr_vs,
    mu_arr_pos, weights_mu,
    M_inv, W,
    scale_tau
):  
    """This function is wrapped and looped by the `_loop_and_assemble_results` function.
    It has many seemingly redundant arguments to maximize precomputation in the `pydisort` function.
    See the Jupyter Notebook, especially section 3, for documentation, explanation and derivation.
    The labels in this file reference labels in the Jupyter Notebook, especially section 3.
    
    """
    # Setup
    # --------------------------------------------------------------------------------------------------------------------------
    ells = np.arange(m, NLeg)
    degree_tile = np.tile(ells, (N, 1)).T
    fac = np.prod(ells[:, None] + np.arange(-m + 1, m + 1)[None, :], axis=-1)
    signs = np.empty(NLeg - m)
    signs[::2] = 1
    signs[1::2] = -1
    if m == 0:
        prefactor = omegal * I0 / (4 * pi)
    else:
        prefactor = omegal * I0 / (2 * pi)
        
    asso_leg_term_pos = sc.special.lpmv(m, degree_tile, mu_arr_pos)
    asso_leg_term_neg = asso_leg_term_pos * signs[:, None]
    asso_leg_term_mu0 = sc.special.lpmv(m, ells, -mu0)

    # Generate mathscr_D and mathscr_X (BDRF terms)
    # --------------------------------------------------------------------------------------------------------------------------
    # If h_\ell = 0 for all \ell \geq m, then there is no BDRF contribution
    if m < NBDRF:
        if m == 0:
            prefactor_mathscr = mu0 * I0 / pi
        else:
            prefactor_mathscr = mu0 * I0 * 2 / pi

        weighted_asso_Leg_coeffs_BDRF = (
            weighted_Leg_coeffs_BDRF[ells[: (NBDRF - m)]] / fac[: (NBDRF - m)]
        )

        mathscr_D_temp = (
            weighted_asso_Leg_coeffs_BDRF[None, :] * asso_leg_term_pos[:NBDRF, :].T * mu_arr_pos[:, None]
        )
        mathscr_D_neg = 2 * mathscr_D_temp @ asso_leg_term_neg[:NBDRF, :]
        weighted_mathscr_D_neg = mathscr_D_neg * W

        mathscr_X_temp = prefactor_mathscr * weighted_asso_Leg_coeffs_BDRF * asso_leg_term_mu0[:NBDRF]
        mathscr_X_pos = mathscr_X_temp @ asso_leg_term_pos[:NBDRF, :]
        
    # Loop over NLayers (denoted L in exposition) atmospheric layers
    # --------------------------------------------------------------------------------------------------------------------------    
    for l in range(NLayers):
        weighted_Leg_coeffs_l = weighted_Leg_coeffs[l]
        G_collect_l = np.empty((L, NQuad, NQuad))
        K_collect_l = np.empty((L, NQuad))
        B_collect_l = np.zeros((L, NQuad))
        if not np.allclose(weighted_Leg_coeffs_l, 0):
            # Generate mathscr_D and mathscr_X (BDRF terms)
            # --------------------------------------------------------------------------------------------------------------------------
            weighted_asso_Leg_coeffs_l = weighted_Leg_coeffs_l[ells] / fac
            
            D_temp = weighted_asso_Leg_coeffs_l[None, :] * asso_leg_term_pos.T
            D_pos = (omegal / 2) * D_temp @ asso_leg_term_pos
            D_neg = (omegal / 2) * D_temp @ asso_leg_term_neg

            X_temp = prefactor * weighted_asso_Leg_coeffs_l * asso_leg_term_mu0
            X_pos = X_temp @ asso_leg_term_pos
            X_neg = X_temp @ asso_leg_term_neg

            # Assemble the coefficient matrix and additional terms
            # --------------------------------------------------------------------------------------------------------------------------
            alpha = M_inv[:, None] * (D_pos * W - np.eye(N))
            beta = M_inv[:, None] * D_neg * W
            A = np.vstack((np.hstack((-alpha, -beta)), np.hstack((beta, alpha))))

            X_tilde = np.concatenate((-M_inv * X_pos, M_inv * X_neg))

            # Diagonalization of coefficient matrix
            # --------------------------------------------------------------------------------------------------------------------------
            K_squared, eigenvecs_GpG = np.linalg.eig((alpha - beta) @ (alpha + beta))

            # Eigenvalues arranged negative then positive, from largest to smallest magnitude
            K = np.concatenate((-np.sqrt(K_squared), np.sqrt(K_squared)))
            eigenvecs_GpG = np.hstack((eigenvecs_GpG, eigenvecs_GpG))
            eigenvecs_GmG = (alpha + beta) @ eigenvecs_GpG / K

            # Eigenvector matrix
            G_pos = (eigenvecs_GpG - eigenvecs_GmG) / 2
            G_neg = (eigenvecs_GpG + eigenvecs_GmG) / 2
            G = np.vstack((G_pos, G_neg))
            G_inv = np.linalg.inv(G)
            
            # Particular solution for the sunbeam source
            # --------------------------------------------------------------------------------------------------------------------------
            B_collect_l[l, :] = np.linalg.solve(-(np.eye(NQuad) / mu0 + A), X_tilde) # coefficient vector
        
        else:
            K = 1 / mu_arr
            G, G_inv = np.eye(NQuad), np.eye(NQuad)
        
        G_collect_l[l, :, :] = G
        K_collect_l[l, :] = K
    
    # Solve for coefficients of homogeneous solution
    # --------------------------------------------------------------------------------------------------------------------------  
    
            
            
                
                
        
        
    # Solve for the coefficients of the homogeneous solution
    B_pos, B_neg = B[:N], B[N:]
    K_tau0_neg = np.exp(eigenvals[:N] * tau0)

    LHS = np.vstack((G_neg, G_pos)) # LHS is a re-arranged COPY of matrix G
    LHS[:N, N:] *= K_tau0_neg[None, :]
    LHS[N:, :N] *= K_tau0_neg[None, :]
    RHS = np.concatenate((b_neg[:, m] - B_neg, b_pos[:, m] - B_pos * np.exp(-tau0 / mu0)))
    C = np.linalg.solve(LHS, RHS)

    if only_flux:
        return PyDISORT.subroutines.generate_flux_functions(
            mu0, I0, tau0,
            G_pos * C[None, :], G_neg * C[None, :],
            eigenvals, N,
            B_pos, B_neg,
            mu_arr_pos, weights_mu,
            scale_tau,
        )
    
    GC_collect[:, m, :] = G * C[None, :]
    eigenvals_collect[:, m] = eigenvals
    B_collect[:, m] = B
    
    # Construct the intensity function
    def u(tau, phi):
        tau = scale_tau * np.atleast_1d(tau) # Delta-M scaling
        phi = np.atleast_1d(phi)
        exponent = np.concatenate(
            (
                eigenvals_collect[:N, :, None] * tau[None, None, :],
                eigenvals_collect[N:, :, None] * (tau - tau0)[None, None, :],
            ),
            axis=0,
        )
        um = np.einsum(
            "imj, jmt -> imt", GC_collect, np.exp(exponent), optimize=True
        ) + B_collect[:, :, None] * np.exp(-tau[None, None, :] / mu0)
        return np.squeeze(
            np.einsum(
                "imt, mp -> itp",
                um,
                np.cos(np.arange(NLoops)[:, None] * (phi0 - phi)[None, :]),
                optimize=True,
            )
        )

    return PyDISORT.subroutines.generate_flux_functions(
        mu0, I0, tau0,
        GC_collect[:N, 0, :], GC_collect[N:, 0, :],
        eigenvals_collect[:, 0], N,
        B_collect[:N, 0], B_collect[N:, 0],
        mu_arr_pos, weights_mu,
        scale_tau,
    ) + (u, )
import numpy as np
import scipy as sc
from math import pi


def _diagonalize(
    NFourier,
    scaled_omega_arr,
    mu_arr_pos, mu_arr,
    M_inv, W,
    N, NQuad, NLeg,
    NLayers,
    weighted_scaled_Leg_coeffs,
    mu0, I0,
    there_is_beam_source,
    Nscoeffs,
    there_is_iso_source,
):
    """This function is wrapped by the `_assemble_solution_functions` function.
    It has many seemingly redundant arguments to maximize precomputation in the `pydisort` function.
    See the Jupyter Notebook, especially section 3, for documentation, explanation and derivation.
    The labels in this file reference labels in the Jupyter Notebook, especially sections 3 and 4.

    """
    ############################### Assemble system and diagonalize coefficient matrix #########################################
    
    # Initialization
    # --------------------------------------------------------------------------------------------------------------------------
    ells_all = np.arange(NLeg)
    ind = 0
    
    G_collect = np.empty((NFourier * NLayers, NQuad, NQuad))
    K_collect = np.empty((NFourier * NLayers, NQuad))
    alpha_arr = np.empty((NFourier * NLayers, N, N))
    beta_arr = np.empty((NFourier * NLayers, N, N))
    X_tilde_arr = np.empty((NFourier * NLayers, NQuad))
    no_shortcut_indices = []
    no_shortcut_indices_0 = []
    
    if there_is_beam_source:
        B_collect = np.zeros((NFourier * NLayers, NQuad))
    if there_is_iso_source:
        G_inv_collect_0 = np.empty((NLayers, NQuad, NQuad))
    # --------------------------------------------------------------------------------------------------------------------------

    # Loop over NFourier Fourier modes
    # These can easily be parallelized, but the speed-up is unlikely to be worth the overhead
    # --------------------------------------------------------------------------------------------------------------------------  
    for m in range(NFourier): 
        # Setup
        # --------------------------------------------------------------------------------------------------------------------------
        m_equals_0 = (m == 0)
        ells = ells_all[m:]
        degree_tile = np.tile(ells, (N, 1)).T
        fac = sc.special.poch(ells + m + 1, -2 * m)
        signs = np.ones(NLeg - m)
        signs[1::2] = -1

        asso_leg_term_pos = sc.special.lpmv(m, degree_tile, mu_arr_pos)
        asso_leg_term_neg = asso_leg_term_pos * signs[:, None]
        asso_leg_term_mu0 = sc.special.lpmv(m, ells, -mu0)
        # --------------------------------------------------------------------------------------------------------------------------

        # Loop over NLayers atmospheric layers
        # --------------------------------------------------------------------------------------------------------------------------          
        for l in range(NLayers):
            # More setup
            weighted_asso_Leg_coeffs_l = weighted_scaled_Leg_coeffs[l, :][ells] * fac
            scaled_omega_l = scaled_omega_arr[l]
            
            # We take precautions against overflow and underflow (this shouldn't happen though)
            if np.any(weighted_asso_Leg_coeffs_l > 0) and np.all(
                np.isfinite(asso_leg_term_pos)
            ):  
                # Generate mathscr_D and mathscr_X (BDRF terms)
                # --------------------------------------------------------------------------------------------------------------------------
                D_temp = weighted_asso_Leg_coeffs_l[None, :] * asso_leg_term_pos.T
                D_pos = (scaled_omega_l / 2) * D_temp @ asso_leg_term_pos
                D_neg = (scaled_omega_l / 2) * D_temp @ asso_leg_term_neg

                if there_is_beam_source:
                    X_temp = (
                        (scaled_omega_l * I0 * (2 - (m == 0)) / (4 * pi))
                        * weighted_asso_Leg_coeffs_l
                        * asso_leg_term_mu0
                    )
                    X_pos = X_temp @ asso_leg_term_pos
                    X_neg = X_temp @ asso_leg_term_neg
                # --------------------------------------------------------------------------------------------------------------------------

                # Assemble the coefficient matrix and additional terms
                # --------------------------------------------------------------------------------------------------------------------------
                alpha = M_inv[:, None] * (D_pos * W[None, :] - np.eye(N))
                beta = M_inv[:, None] * D_neg * W[None, :]
                if there_is_beam_source:
                    X_tilde_arr[ind, :] = np.concatenate([-M_inv * X_pos, M_inv * X_neg])
                # --------------------------------------------------------------------------------------------------------------------------
                
                alpha_arr[ind, :, :] = alpha
                beta_arr[ind, :, :] = beta
                no_shortcut_indices.append(ind)
                if there_is_iso_source and m_equals_0:
                    no_shortcut_indices_0.append(l)
                
            else:
                # This is a shortcut to the diagonalization results
                G = np.zeros((NQuad, NQuad))
                G[N:, :N] = np.eye(N)
                G[:N, N:] = np.eye(N)
                
                G_collect[ind, :, :] = G
                K_collect[ind, :] = -1 / mu_arr
                if there_is_iso_source and m_equals_0:
                    G_inv_collect_0[l, :, :] = G
            ind += 1
                
    if len(no_shortcut_indices) > 0:
    
        # Diagonalization of coefficient matrix
        # --------------------------------------------------------------------------------------------------------------------------
        alpha_arr = alpha_arr[: len(no_shortcut_indices), :, :]
        beta_arr = beta_arr[: len(no_shortcut_indices), :, :]

        K_squared_arr, eigenvecs_GpG_arr = np.linalg.eig(
            np.einsum(
                "lij, ljk -> lik", alpha_arr - beta_arr, alpha_arr + beta_arr, optimize=True
            ),
        )

        # Eigenvalues arranged negative then positive, from largest to smallest magnitude
        K_arr = np.concatenate((-np.sqrt(K_squared_arr), np.sqrt(K_squared_arr)), axis=1)
        eigenvecs_GpG_arr = np.concatenate((eigenvecs_GpG_arr, eigenvecs_GpG_arr), axis=2)
        eigenvecs_GmG_arr = (
            np.einsum(
                "lij, ljk -> lik", alpha_arr + beta_arr, eigenvecs_GpG_arr, optimize=True
            )
            / K_arr[:, None, :]
        )

        # Eigenvector matrix
        G_arr = np.concatenate(
            (
                (eigenvecs_GpG_arr - eigenvecs_GmG_arr) / 2,
                (eigenvecs_GpG_arr + eigenvecs_GmG_arr) / 2,
            ),
            axis=1,
        )
        G_inv_arr = np.linalg.inv(G_arr)
        
        G_collect[no_shortcut_indices, :, :] = G_arr
        K_collect[no_shortcut_indices, :] = K_arr
        if len(no_shortcut_indices_0) > 0:
            G_inv_collect_0[no_shortcut_indices_0, :, :] = G_inv_arr[: len(no_shortcut_indices_0), :, :]
        # --------------------------------------------------------------------------------------------------------------------------

        # Particular solution for the sunbeam source
        # --------------------------------------------------------------------------------------------------------------------------
        if there_is_beam_source:
            X_tilde_arr = X_tilde_arr[: len(no_shortcut_indices), :]
            B_collect[no_shortcut_indices, :] = np.einsum(
                "lij, ljk, lk -> li",
                -G_arr / (1 / mu0 + K_arr)[:, None, :],
                G_inv_arr,
                X_tilde_arr,
                optimize=True,
            )
        # --------------------------------------------------------------------------------------------------------------------------
        
    if there_is_beam_source:
        if there_is_iso_source:
            return (
                G_collect.reshape((NFourier, NLayers, NQuad, NQuad)),
                K_collect.reshape((NFourier, NLayers, NQuad)),
                B_collect.reshape((NFourier, NLayers, NQuad)),
                G_inv_collect_0,
            )
        else:
            return (
                G_collect.reshape((NFourier, NLayers, NQuad, NQuad)),
                K_collect.reshape((NFourier, NLayers, NQuad)),
                B_collect.reshape((NFourier, NLayers, NQuad)),
            )
    else:
        if there_is_iso_source:
            return (
                G_collect.reshape((NFourier, NLayers, NQuad, NQuad)),
                K_collect.reshape((NFourier, NLayers, NQuad)),
                G_inv_collect_0,
            )
        else:
            return (
                G_collect.reshape((NFourier, NLayers, NQuad, NQuad)), 
                K_collect.reshape((NFourier, NLayers, NQuad)),
            )
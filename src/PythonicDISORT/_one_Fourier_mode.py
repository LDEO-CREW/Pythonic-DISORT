from PythonicDISORT.subroutines import _mathscr_v
import numpy as np
import scipy as sc
from warnings import warn
from math import pi


def _one_Fourier_mode(
    m,
    scaled_omega_arr,
    tau_arr,
    scaled_tau_arr_with_0,
    mu_arr_pos, mu_arr,
    M_inv, W,
    N, NQuad, NLeg,
    NLayers, NBDRF,
    weighted_scaled_Leg_coeffs,
    BDRF_Fourier_modes,
    mu0, I0,
    b_pos, b_neg,
    scalar_b_pos, scalar_b_neg,
    s_poly_coeffs,
    Nscoeffs,
    use_sparse_NLayers
):
    """This function is wrapped and looped by the `_loop_and_assemble_results` function.
    It has many seemingly redundant arguments to maximize precomputation in the `pydisort` function.
    See the Jupyter Notebook, especially section 3, for documentation, explanation and derivation.
    The labels in this file reference labels in the Jupyter Notebook, especially sections 3 and 4.

    """
    # Setup
    # --------------------------------------------------------------------------------------------------------------------------
    ells = np.arange(m, NLeg)
    degree_tile = np.tile(ells, (N, 1)).T
    fac = sc.special.poch(ells + m + 1, -2 * m)
    signs = np.empty(NLeg - m)
    signs[::2] = 1
    signs[1::2] = -1

    asso_leg_term_pos = sc.special.lpmv(m, degree_tile, mu_arr_pos)
    asso_leg_term_neg = asso_leg_term_pos * signs[:, None]
    asso_leg_term_mu0 = sc.special.lpmv(m, ells, -mu0)
    # --------------------------------------------------------------------------------------------------------------------------
    
    # Generate mathscr_D and mathscr_X (BDRF terms)
    # --------------------------------------------------------------------------------------------------------------------------
    if m < NBDRF:
        mathscr_D_neg = (1 + (m == 0) * 1) * BDRF_Fourier_modes[m](mu_arr_pos, -mu_arr_pos)
        R = mathscr_D_neg * (mu_arr_pos * W)[None, :]

        if I0 > 0:
            mathscr_X_pos = (mu0 * I0 / pi) * BDRF_Fourier_modes[m](
                mu_arr_pos, -np.array([mu0])
            )[:, 0]
    # --------------------------------------------------------------------------------------------------------------------------

    # Loop over NLayers atmospheric layers
    # --------------------------------------------------------------------------------------------------------------------------
    if Nscoeffs > 0 and m == 0:
        G_inv_collect_0 = np.empty((NLayers, NQuad, NQuad))
    if I0 > 0:
        B_collect_m = np.zeros((NLayers, NQuad))
        
    G_collect_m = np.empty((NLayers, NQuad, NQuad))
    K_collect_m = np.empty((NLayers, NQuad))
    alpha_list = []
    beta_list = []
    X_tilde_list=[]
    if_indices = []

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

            if I0 > 0:
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
            if I0 > 0:
                X_tilde_list.append(np.concatenate([-M_inv * X_pos, M_inv * X_neg]))
            # --------------------------------------------------------------------------------------------------------------------------
            
            if_indices.append(l)
            alpha_list.append(alpha)
            beta_list.append(beta)
            
        else:
            # This is a shortcut to the diagonalization results
            G = np.zeros((NQuad, NQuad))
            G[N:, :N] = np.eye(N)
            G[:N, N:] = np.eye(N)
            
            G_collect_m[l, :, :] = G
            K_collect_m[l, :] = -1 / mu_arr
            if Nscoeffs > 0 and m == 0:
                G_inv_collect_0[l, :, :] = G
            
    if len(if_indices) > 0:       
    
        # Diagonalization of coefficient matrix
        # --------------------------------------------------------------------------------------------------------------------------
        alpha_list = np.atleast_3d(np.array(alpha_list))
        beta_list = np.atleast_3d(np.array(beta_list))

        K_squared_arr, eigenvecs_GpG_arr = np.linalg.eig(
            np.einsum(
                "lij, ljk -> lik", alpha_list - beta_list, alpha_list + beta_list, optimize=True
            ),
        )
        if m == 0:
            if np.any(K_squared_arr < 1e-9):  # Then |K| < 1e-3
                warn(
                    "Some single-scattering albedos are too close to 1. Results may be inaccurate."
                )

        # Eigenvalues arranged negative then positive, from largest to smallest magnitude
        K_arr = np.concatenate((-np.sqrt(K_squared_arr), np.sqrt(K_squared_arr)), axis=1)
        eigenvecs_GpG_arr = np.concatenate((eigenvecs_GpG_arr, eigenvecs_GpG_arr), axis=2)
        eigenvecs_GmG_arr = (
            np.einsum(
                "lij, ljk -> lik", alpha_list + beta_list, eigenvecs_GpG_arr, optimize=True
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
        
        G_collect_m[if_indices, :, :] = G_arr
        K_collect_m[if_indices, :] = K_arr
        if Nscoeffs > 0 and m == 0:
            G_inv_collect_0[if_indices, :, :] = G_inv_arr
        # --------------------------------------------------------------------------------------------------------------------------

        # Particular solution for the sunbeam source
        # --------------------------------------------------------------------------------------------------------------------------
        if I0 > 0:
            X_tilde_list = np.atleast_2d(np.array(X_tilde_list))
            B_collect_m[if_indices, :] = np.einsum(
                "lij, ljk, lk -> li",
                -G_arr / (1 / mu0 + K_arr)[:, None, :],
                G_inv_arr,
                X_tilde_list,
                optimize=True,
            )
        # --------------------------------------------------------------------------------------------------------------------------

    ################################## Solve for coefficients of homogeneous solution ##########################################
    
    # Assemble RHS
    # --------------------------------------------------------------------------------------------------------------------------
    # Ensure the BCs are of the correct shape
    if scalar_b_pos:
        if m == 0:
            b_pos_m = np.full(N, b_pos)
        else:
            b_pos_m = np.zeros(N)
    else:
        b_pos_m = b_pos[:, m]
    if scalar_b_neg:
        if m == 0:
            b_neg_m = np.full(N, b_neg)
        else:
            b_neg_m = np.zeros(N)
    else:
        b_neg_m = b_neg[:, m]
    
    # _mathscr_v_contribution
    if Nscoeffs > 0 and m == 0:
        _mathscr_v_contribution_top = -_mathscr_v(
                    np.array([0]), np.array([0]),
                    s_poly_coeffs[0:1, :],
                    Nscoeffs,
                    G_collect_m[0:1, N:, :],
                    K_collect_m[0:1, :],
                    G_inv_collect_0[0:1, :, :],
                    mu_arr,
                ).flatten()
    
        _mathscr_v_contribution_middle = np.array([])
        if NLayers > 1:
            indices = np.arange(NLayers - 1)
            _mathscr_v_contribution_middle = (
                _mathscr_v(
                    tau_arr[:-1],
                    indices,
                    s_poly_coeffs[:-1, :],
                    Nscoeffs,
                    G_collect_m[:-1, :, :],
                    K_collect_m[:-1, :],
                    G_inv_collect_0[:-1, :, :],
                    mu_arr,
                )
                - _mathscr_v(
                    tau_arr[:-1],
                    indices,
                    s_poly_coeffs[1:, :],
                    Nscoeffs,
                    G_collect_m[1:, :, :],
                    K_collect_m[1:, :],
                    G_inv_collect_0[1:, :, :],
                    mu_arr,
                )
            ).flatten()
        
        _mathscr_v_contribution_bottom = _mathscr_v(
                tau_arr[-1:], np.array([0]),
                s_poly_coeffs[-1:, :],
                Nscoeffs,
                G_collect_m[-1:, N:, :],
                K_collect_m[-1:, :],
                G_inv_collect_0[-1:, :, :],
                mu_arr).flatten()
        if NBDRF > 0:
            _mathscr_v_contribution_bottom = (
                R + np.eye(N)
            ) @ _mathscr_v_contribution_bottom
                
        _mathscr_v_contribution = np.concatenate(
            [
                _mathscr_v_contribution_top,
                _mathscr_v_contribution_middle,
                _mathscr_v_contribution_bottom,
            ]
        )   
    else:
        _mathscr_v_contribution = 0
    
    if I0 > 0:
        if m < NBDRF:
            BDRF_RHS_contribution = mathscr_X_pos + R @ B_collect_m[-1, N:]
        else:
            BDRF_RHS_contribution = 0
        
        RHS_middle = np.array([])
        if NLayers > 1:
            RHS_middle = np.array(
                [
                    (B_collect_m[l + 1, :] - B_collect_m[l, :])
                    * np.exp(-mu0 * scaled_tau_arr_with_0[l + 1])
                    for l in range(NLayers - 1)
                ]
            ).flatten()
        
        RHS = (
            np.concatenate(
                [
                    b_neg_m - B_collect_m[0, N:],
                    RHS_middle,
                    b_pos_m
                    + (BDRF_RHS_contribution - B_collect_m[-1, :N])
                    * np.exp(-scaled_tau_arr_with_0[-1] / mu0),
                ]
            )
            + _mathscr_v_contribution
        )
    else:
        RHS_middle = np.zeros(NQuad * (NLayers - 1))
        RHS = np.concatenate([b_neg_m, RHS_middle, b_pos_m]) + _mathscr_v_contribution
    # --------------------------------------------------------------------------------------------------------------------------
    
    # Assemble LHS
    # --------------------------------------------------------------------------------------------------------------------------
    LHS = np.zeros((NLayers * NQuad, NLayers * NQuad))

    G_0_nn = G_collect_m[0, N:, :N]
    G_0_np = G_collect_m[0, N:, N:]
    G_L_pn = G_collect_m[-1, :N, :N]
    G_L_nn = G_collect_m[-1, N:, :N]
    G_L_pp = G_collect_m[-1, :N, N:]
    G_L_np = G_collect_m[-1, N:, N:]
    E_Lm1L = np.exp(
        K_collect_m[-1, :N] * (scaled_tau_arr_with_0[-1] - scaled_tau_arr_with_0[-2])
    )
    if m < NBDRF:
        BDRF_LHS_contribution_neg = R @ G_L_nn
        BDRF_LHS_contribution_pos = R @ G_L_np
    else:
        BDRF_LHS_contribution_neg = 0
        BDRF_LHS_contribution_pos = 0

    # BCs for the entire atmosphere
    LHS[:N, :N] = G_0_nn
    LHS[:N, N : 2 * N] = (
        G_0_np
        * np.exp(K_collect_m[0, :N] * scaled_tau_arr_with_0[1])[None, :]
    )
    LHS[-N:, -2 * N : -N] = (G_L_pn - BDRF_LHS_contribution_neg) * E_Lm1L[None, :]
    LHS[-N:, -N:] = G_L_pp - BDRF_LHS_contribution_pos

    # Interlayer BCs / continuity BCs
    for l in range(NLayers - 1):
        G_l_pn = G_collect_m[l, :N, :N]
        G_l_nn = G_collect_m[l, N:, :N]
        G_l_ap = G_collect_m[l, :, N:]
        G_lp1_an = G_collect_m[l + 1, :, :N]
        G_lp1_pp = G_collect_m[l + 1, :N, N:]
        G_lp1_np = G_collect_m[l + 1, N:, N:]
        scaled_tau_arr_lm1 = scaled_tau_arr_with_0[l]
        scaled_tau_arr_l = scaled_tau_arr_with_0[l + 1]
        scaled_tau_arr_lp1 = scaled_tau_arr_with_0[l + 2]
        # Postive eigenvalues
        K_l_pos = K_collect_m[l, N:]
        K_lp1_pos = K_collect_m[l + 1, N:]

        E_lm1l = np.exp(K_l_pos * (scaled_tau_arr_lm1 - scaled_tau_arr_l))
        E_llp1 = np.exp(K_lp1_pos * (scaled_tau_arr_l - scaled_tau_arr_lp1))
        block_row = np.hstack(
            [
                np.vstack([G_l_pn * E_lm1l[None, :], G_l_nn * E_lm1l[None, :]]),
                G_l_ap,
                -G_lp1_an,
                -np.vstack([G_lp1_pp * E_llp1[None, :], G_lp1_np * E_llp1[None, :]]),
            ]
        )
        LHS[
            N + l * NQuad : N + (l + 1) * NQuad, l * NQuad : l * NQuad + 2 * NQuad
        ] = block_row
    # --------------------------------------------------------------------------------------------------------------------------
    
    # Solve the system and return computations
    # --------------------------------------------------------------------------------------------------------------------------
    if NLayers >= use_sparse_NLayers:
        C_m = sc.sparse.linalg.spsolve(sc.sparse.csr_matrix(LHS), RHS).reshape(NLayers, NQuad)
    else:
        C_m = np.linalg.solve(LHS, RHS).reshape(NLayers, NQuad)
    
    if I0 > 0:
        if Nscoeffs > 0 and m == 0:
            return G_collect_m, C_m, K_collect_m, B_collect_m, G_inv_collect_0
        else:
            GC_collect_m = G_collect_m * C_m[:, None, :]
            return GC_collect_m, K_collect_m, B_collect_m
    else:
        if Nscoeffs > 0 and m == 0:
            return G_collect_m, C_m, K_collect_m, G_inv_collect_0
        else:
            GC_collect_m = G_collect_m * C_m[:, None, :]
            return GC_collect_m, K_collect_m
    # --------------------------------------------------------------------------------------------------------------------------
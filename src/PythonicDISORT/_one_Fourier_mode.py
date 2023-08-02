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
    weighted_Leg_coeffs_BDRF,
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
    The labels in this file reference labels in the Jupyter Notebook, especially sections 3 and 5.

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
    # If h_\ell = 0 for all \ell \geq m, then there is no BDRF contribution
    # We take precautions against overflow and underflow
    if m < NBDRF and np.all(np.isfinite(asso_leg_term_pos[:NBDRF, :])):
        weighted_asso_Leg_coeffs_BDRF = (
            weighted_Leg_coeffs_BDRF[ells[: (NBDRF - m)]] * fac[: (NBDRF - m)]
        )
        mathscr_D_temp = (
            weighted_asso_Leg_coeffs_BDRF[None, :] * asso_leg_term_pos.T[:, :NBDRF]
        )
        mathscr_D_neg = 2 * mathscr_D_temp @ asso_leg_term_neg[:NBDRF, :]
        R = mathscr_D_neg * (mu_arr_pos * W)[None, :]

        if I0 > 0:
            mathscr_X_temp = (
                (mu0 * I0 * (2 - (m == 0)) / pi)
                * weighted_asso_Leg_coeffs_BDRF
                * asso_leg_term_mu0[:NBDRF]
            )
            mathscr_X_pos = mathscr_X_temp @ asso_leg_term_pos[:NBDRF, :]
    # --------------------------------------------------------------------------------------------------------------------------

    # Loop over NLayers atmospheric layers
    # --------------------------------------------------------------------------------------------------------------------------
    if Nscoeffs > 0 and m == 0:
        G_inv_collect_0 = np.empty((NLayers, NQuad, NQuad))
    G_collect_m = np.empty((NLayers, NQuad, NQuad))
    K_collect_m = np.empty((NLayers, NQuad))
    if I0 > 0:
        B_collect_m = np.zeros((NLayers, NQuad))

    for l in range(NLayers):
        # More setup
        weighted_asso_Leg_coeffs_l = weighted_scaled_Leg_coeffs[l, :][ells] * fac
        scaled_omega_l = scaled_omega_arr[l]
        
        # We take precautions against overflow and underflow
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
            A = np.vstack([np.hstack([-alpha, -beta]), np.hstack([beta, alpha])])
            if I0 > 0:
                X_tilde = np.concatenate([-M_inv * X_pos, M_inv * X_neg])
            # --------------------------------------------------------------------------------------------------------------------------

            # Diagonalization of coefficient matrix
            # --------------------------------------------------------------------------------------------------------------------------
            K_squared, eigenvecs_GpG = np.linalg.eig((alpha - beta) @ (alpha + beta))
            if m == 0:
                if np.any(np.isclose(K_squared, 0)):
                    warn(
                        "Single-scattering albedo for layer "
                        + str(l)
                        + " (counting from 0) is too close to 1. Results may be inaccurate."
                    )

            # Eigenvalues arranged negative then positive, from largest to smallest magnitude
            K = np.concatenate([-np.sqrt(K_squared), np.sqrt(K_squared)])
            eigenvecs_GpG = np.hstack([eigenvecs_GpG, eigenvecs_GpG])
            eigenvecs_GmG = (alpha + beta) @ eigenvecs_GpG / K

            # Eigenvector matrix
            G_pos = (eigenvecs_GpG - eigenvecs_GmG) / 2
            G_neg = (eigenvecs_GpG + eigenvecs_GmG) / 2
            G = np.vstack([G_pos, G_neg])
            # We only need G^{-1} in special cases
            if Nscoeffs > 0 and m == 0:
                G_inv = np.linalg.inv(G)
            # --------------------------------------------------------------------------------------------------------------------------

            # Particular solution for the sunbeam source
            # --------------------------------------------------------------------------------------------------------------------------
            if I0 > 0:
                if Nscoeffs > 0 and m == 0:
                    B = -G / (1 / mu0 + K)[None, :] @ G_inv @ X_tilde
                else:
                    # This method is more numerically stable
                    LHS = A.copy()
                    np.fill_diagonal(LHS, 1 / mu0 + np.diag(A))
                    B = np.linalg.solve(LHS, -X_tilde)

                B_collect_m[l, :] = B
            # --------------------------------------------------------------------------------------------------------------------------
        else:
            # This is a shortcut to the same results
            K = -1 / mu_arr
            G = np.zeros((NQuad, NQuad))
            G[N:, :N] = np.eye(N)
            G[:N, N:] = np.eye(N)
            if Nscoeffs > 0 and m == 0:
                G_inv = G

        G_collect_m[l, :, :] = G
        K_collect_m[l, :] = K
        if Nscoeffs > 0 and m == 0:
            G_inv_collect_0[l, :, :] = G_inv
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
    
    # If h_\ell = 0 for all \ell >= m, then there is no BDRF contribution
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
    dim = NLayers * NQuad
    if NLayers >= use_sparse_NLayers:
        LHS = sc.sparse.lil_matrix((dim, dim))
    else:
        LHS = np.zeros((dim, dim))

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
        C_m = sc.sparse.linalg.spsolve(LHS.asformat("csr"), RHS).reshape(NLayers, NQuad)
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
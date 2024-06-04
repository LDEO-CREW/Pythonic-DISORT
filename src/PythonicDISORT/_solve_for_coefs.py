from PythonicDISORT.subroutines import _mathscr_v
from PythonicDISORT.subroutines import _nd_slice_to_indexes

import numpy as np
import scipy as sc
from math import pi


def _solve_for_coefs(
    NFourier,
    G_collect,
    K_collect,
    B_collect,
    G_inv_collect_0,
    tau_arr,
    scaled_tau_arr_with_0,
    mu_arr_pos, mu_arr_pos_times_W, mu_arr,
    N, NQuad,
    NLayers, NBDRF,
    atmos_is_multilayered,
    BDRF_Fourier_modes,
    mu0, I0,
    there_is_beam_source,
    b_pos, b_neg,
    b_pos_is_scalar, b_neg_is_scalar,
    s_poly_coeffs,
    Nscoeffs,
    there_is_iso_source,
    use_sparse_NLayers,
):
    """This function is wrapped by the `_assemble_solution_functions` function.
    It has many seemingly redundant arguments to maximize precomputation in the `pydisort` function.
    See the Jupyter Notebook, especially section 3, for documentation, explanation and derivation.
    The labels in this file reference labels in the Jupyter Notebook, especially sections 3 and 4.

    """
    ################################## Solve for coefficients of homogeneous solution ##########################################
    
    GC_collect = np.empty((NFourier, NLayers, NQuad, NQuad))
    use_sparse_framework = NLayers >= use_sparse_NLayers
    
    # The following loops can easily be parallelized, but the speed-up is unlikely to be worth the overhead
    for m in range(NFourier):
        m_equals_0 = (m == 0)
        BDRF_bool = m < NBDRF
        
        G_collect_m = G_collect[m, :, :, :]
        K_collect_m = K_collect[m, :, :]
        if there_is_beam_source:
            B_collect_m = B_collect[m, :, :]
            
        # Generate mathscr_D and mathscr_X (BDRF terms)
        # --------------------------------------------------------------------------------------------------------------------------
        if BDRF_bool:
            mathscr_D_neg = (1 + m_equals_0 * 1) * BDRF_Fourier_modes[m](mu_arr_pos, -mu_arr_pos)
            R = mathscr_D_neg * mu_arr_pos_times_W[None, :]

            if there_is_beam_source:
                mathscr_X_pos = (mu0 * I0 / pi) * BDRF_Fourier_modes[m](
                    mu_arr_pos, -np.array([mu0])
                )[:, 0]
        # --------------------------------------------------------------------------------------------------------------------------
    
        # Assemble RHS
        # --------------------------------------------------------------------------------------------------------------------------
        # Ensure the BCs are of the correct shape
        if b_pos_is_scalar:
            if m_equals_0:
                b_pos_m = np.full(N, b_pos)
            else:
                b_pos_m = np.zeros(N)
        else:
            b_pos_m = b_pos[:, m]
        if b_neg_is_scalar:
            if m_equals_0:
                b_neg_m = np.full(N, b_neg)
            else:
                b_neg_m = np.zeros(N)
        else:
            b_neg_m = b_neg[:, m]
        
        # _mathscr_v_contribution
        if there_is_iso_source and m_equals_0:
            _mathscr_v_contribution_top = -_mathscr_v(
                        np.array([0]), np.array([0]),
                        s_poly_coeffs[0:1, :],
                        Nscoeffs,
                        G_collect_m[0:1, N:, :],
                        K_collect_m[0:1, :],
                        G_inv_collect_0[0:1, :, :],
                        mu_arr,
                    ).ravel()
        
            _mathscr_v_contribution_middle = np.array([])
            if atmos_is_multilayered:
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
                ).ravel()
            
            _mathscr_v_contribution_bottom = _mathscr_v(
                    tau_arr[-1:], np.array([0]),
                    s_poly_coeffs[-1:, :],
                    Nscoeffs,
                    G_collect_m[-1:, N:, :],
                    K_collect_m[-1:, :],
                    G_inv_collect_0[-1:, :, :],
                    mu_arr).ravel()
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
        
        if there_is_beam_source:
            if BDRF_bool:
                BDRF_RHS_contribution = mathscr_X_pos + R @ B_collect_m[-1, N:]
            else:
                BDRF_RHS_contribution = 0
            
            RHS_middle = np.array([])
            if atmos_is_multilayered:
                RHS_middle = np.array(
                    [
                        (B_collect_m[l + 1, :] - B_collect_m[l, :])
                        * np.exp(-mu0 * scaled_tau_arr_with_0[l + 1])
                        for l in range(NLayers - 1)
                    ]
                ).ravel()
            
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
        dim = NLayers * NQuad

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
        
        if use_sparse_framework:
            N_squared = N**2
            density = 4 * N_squared * (2 * NLayers - 1)
            row_indices = np.empty(density)
            col_indices = np.empty(density)
            block_rows = np.empty(density)
            
            block_row_len = 2 * NQuad**2
            block_row_ind = np.arange(block_row_len).reshape(NQuad, 2 * NQuad)
            block_row_sort = np.empty(block_row_len, dtype='int')
            
            block_row_sort[:N_squared] = block_row_ind[:N, :N].ravel()
            block_row_sort[N_squared : 2 * N_squared] = block_row_ind[N : 2 * N, :N].ravel()
            block_row_sort[2 * N_squared : 4 * N_squared] = block_row_ind[:2 * N, N : 2 * N].ravel()
            block_row_sort[4 * N_squared : 6 * N_squared] = block_row_ind[:2 * N, 2 * N : 3 * N].ravel()
            block_row_sort[6 * N_squared : 7 * N_squared] = block_row_ind[:N, 3 * N : 4 * N].ravel()
            block_row_sort[7 * N_squared : 8 * N_squared] = block_row_ind[N : 2 * N, 3 * N : 4 * N].ravel()

            # BCs for the entire atmosphere
            row_indices[:N_squared], col_indices[:N_squared] = _nd_slice_to_indexes(np.s_[:N, :N])
            block_rows[:N_squared] = G_0_nn.ravel()
            (
                row_indices[N_squared : 2 * N_squared],
                col_indices[N_squared : 2 * N_squared],
            ) = _nd_slice_to_indexes(np.s_[:N, N:NQuad])
            block_rows[N_squared : 2 * N_squared] = (
                G_0_np * np.exp(K_collect_m[0, :N] * scaled_tau_arr_with_0[1])[None, :]
            ).ravel()
            (
                row_indices[2 * N_squared : 3 * N_squared],
                col_indices[2 * N_squared : 3 * N_squared],
            ) = _nd_slice_to_indexes(np.s_[dim - N : dim, dim - NQuad : dim - N])
            block_rows[2 * N_squared : 3 * N_squared] = (
                (G_L_pn - BDRF_LHS_contribution_neg) * E_Lm1L[None, :]
            ).ravel()
            (
                row_indices[3 * N_squared : 4 * N_squared],
                col_indices[3 * N_squared : 4 * N_squared],
            ) = _nd_slice_to_indexes(np.s_[dim - N : dim, dim - N : dim])
            block_rows[3 * N_squared : 4 * N_squared] = (
                G_L_pp - BDRF_LHS_contribution_pos
            ).ravel()

            # Interlayer / continuity BCs
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
                
                row_indices_l, col_indices_l = _nd_slice_to_indexes(
                    np.s_[N + l * NQuad : N + (l + 1) * NQuad, l * NQuad : (l + 2) * NQuad]
                )
                row_indices[
                    4 * N_squared + l * block_row_len : 4 * N_squared + (l + 1) * block_row_len
                ] = row_indices_l[block_row_sort]
                col_indices[
                    4 * N_squared + l * block_row_len : 4 * N_squared + (l + 1) * block_row_len
                ] = col_indices_l[block_row_sort]
                
                starting_ind = 4 * N_squared + l * block_row_len
                block_rows[starting_ind : N_squared + starting_ind] = (G_l_pn * E_lm1l[None, :]).ravel()
                block_rows[starting_ind + N_squared : 2 * N_squared + starting_ind] = (G_l_nn * E_lm1l[None, :]).ravel()
                block_rows[starting_ind + 2 * N_squared : 4 * N_squared + starting_ind] = G_l_ap.ravel()
                block_rows[starting_ind + 4 * N_squared : 6 * N_squared + starting_ind] = -G_lp1_an.ravel()
                block_rows[starting_ind + 6 * N_squared : 7 * N_squared + starting_ind] = -(G_lp1_pp * E_llp1[None, :]).ravel()
                block_rows[starting_ind + 7 * N_squared : 8 * N_squared + starting_ind] = -(G_lp1_np * E_llp1[None, :]).ravel()
    
            LHS = sc.sparse.coo_matrix(
                (
                    block_rows,
                    (
                        row_indices,
                        col_indices,
                    ),
                ),
                shape=(dim, dim),
            )
        
            # Solve the system
            C_m = sc.sparse.linalg.spsolve(LHS.tocsr(), RHS).reshape(NLayers, NQuad)
        
        else:
            LHS = np.zeros((dim, dim))

            # BCs for the entire atmosphere
            LHS[:N, :N] = G_0_nn
            LHS[:N, N : NQuad] = (
                G_0_np
                * np.exp(K_collect_m[0, :N] * scaled_tau_arr_with_0[1])[None, :]
            )
            LHS[-N:, -NQuad : -N] = (G_L_pn - BDRF_LHS_contribution_neg) * E_Lm1L[None, :]
            LHS[-N:, -N:] = G_L_pp - BDRF_LHS_contribution_pos

            # Interlayer / continuity BCs
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
                
                starting_row = N + l * NQuad
                starting_col = l * NQuad
                LHS[starting_row : N + starting_row, starting_col : N + starting_col] = G_l_pn * E_lm1l[None, :]
                LHS[N + starting_row : 2 * N + starting_row, starting_col : N + starting_col] = G_l_nn * E_lm1l[None, :]
                LHS[starting_row : 2 * N + starting_row, N + starting_col : 2 * N + starting_col] = G_l_ap
                LHS[starting_row : 2 * N + starting_row, 2 * N + starting_col : 3 * N + starting_col] = -G_lp1_an
                LHS[starting_row : N + starting_row, 3 * N + starting_col : 4 * N + starting_col] = -G_lp1_pp * E_llp1[None, :]
                LHS[N + starting_row : 2 * N + starting_row, 3 * N + starting_col : 4 * N + starting_col] = -G_lp1_np * E_llp1[None, :]
            
            # Solve the system
            C_m = np.linalg.solve(LHS, RHS).reshape(NLayers, NQuad)
        # --------------------------------------------------------------------------------------------------------------------------

        GC_collect[m, :, :, :] = G_collect_m * C_m[:, None, :]
        
    return GC_collect
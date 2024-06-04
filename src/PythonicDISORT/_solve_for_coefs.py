from PythonicDISORT.subroutines import _mathscr_v
import numpy as np
import scipy as sc
from math import pi


def _solve_for_coefs(
    NLoops,
    G_collect,
    K_collect,
    B_collect,
    G_inv_collect_0,
    tau_arr,
    scaled_tau_arr_with_0,
    mu_arr_pos, mu_arr_pos_times_W, mu_arr,
    N, NQuad,
    NLayers, NBDRF,
    multilayer_bool,
    BDRF_Fourier_modes,
    mu0, I0,
    beam_source_bool,
    b_pos, b_neg,
    scalar_b_pos, scalar_b_neg,
    s_poly_coeffs,
    Nscoeffs,
    iso_source_bool,
    use_sparse_NLayers,
):
    """This function is wrapped by the `_assemble_solution_functions` function.
    It has many seemingly redundant arguments to maximize precomputation in the `pydisort` function.
    See the Jupyter Notebook, especially section 3, for documentation, explanation and derivation.
    The labels in this file reference labels in the Jupyter Notebook, especially sections 3 and 4.

    """
    ################################## Solve for coefficients of homogeneous solution ##########################################
    
    GC_collect = np.empty((NLoops, NLayers, NQuad, NQuad))
    use_sparse_bool = NLayers >= use_sparse_NLayers
    
    # The following loops can easily be parallelized, but the speed-up is unlikely to be worth the overhead
    for m in range(NLoops):
        m_equals_0_bool = (m == 0)
        BDRF_bool = m < NBDRF
        
        G_collect_m = G_collect[m, :, :, :]
        K_collect_m = K_collect[m, :, :]
        if beam_source_bool:
            B_collect_m = B_collect[m, :, :]
            
        # Generate mathscr_D and mathscr_X (BDRF terms)
        # --------------------------------------------------------------------------------------------------------------------------
        if BDRF_bool:
            mathscr_D_neg = (1 + m_equals_0_bool * 1) * BDRF_Fourier_modes[m](mu_arr_pos, -mu_arr_pos)
            R = mathscr_D_neg * mu_arr_pos_times_W[None, :]

            if beam_source_bool:
                mathscr_X_pos = (mu0 * I0 / pi) * BDRF_Fourier_modes[m](
                    mu_arr_pos, -np.array([mu0])
                )[:, 0]
        # --------------------------------------------------------------------------------------------------------------------------
    
        # Assemble RHS
        # --------------------------------------------------------------------------------------------------------------------------
        # Ensure the BCs are of the correct shape
        if scalar_b_pos:
            if m_equals_0_bool:
                b_pos_m = np.full(N, b_pos)
            else:
                b_pos_m = np.zeros(N)
        else:
            b_pos_m = b_pos[:, m]
        if scalar_b_neg:
            if m_equals_0_bool:
                b_neg_m = np.full(N, b_neg)
            else:
                b_neg_m = np.zeros(N)
        else:
            b_neg_m = b_neg[:, m]
        
        # _mathscr_v_contribution
        if iso_source_bool and m_equals_0_bool:
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
            if multilayer_bool:
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
        
        if beam_source_bool:
            if BDRF_bool:
                BDRF_RHS_contribution = mathscr_X_pos + R @ B_collect_m[-1, N:]
            else:
                BDRF_RHS_contribution = 0
            
            RHS_middle = np.array([])
            if multilayer_bool:
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
        if BDRF_bool:
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
        
        # Solve the system
        # --------------------------------------------------------------------------------------------------------------------------
        if use_sparse_bool:
            C_m = sc.sparse.linalg.spsolve(sc.sparse.csr_matrix(LHS), RHS).reshape(NLayers, NQuad)
        else:
            C_m = np.linalg.solve(LHS, RHS).reshape(NLayers, NQuad)
        # --------------------------------------------------------------------------------------------------------------------------

        GC_collect[m, :, :, :] = G_collect_m * C_m[:, None, :]
        
    return GC_collect
from PythonicDISORT.subroutines import _mathscr_v

import numpy as np
from numpy.lib.stride_tricks import as_strided
import scipy as sc
from math import pi


def _solve_for_coeffs(
    NFourier,                                 # Number of intensity Fourier modes
    G_collect,                                # Eigenvector matrices
    K_collect,                                # Eigenvalues
    B_collect,                                # Coefficients vectors for particular solutions
    G_inv_collect_0,                          # Inverse of eigenvector matrix for the 0th Fourier mode
    scaled_tau_arr_with_0,                    # Delta-scaled lower boundary of layers with 0 inserted in front
    mu_arr, mu_arr_pos, mu_arr_pos_times_W,   # Quadrature nodes for 1) both 2) upper hemispheres; 3) upper hemisphere quadrature nodes times weights
    N, NQuad,                                 # Number of 1) upper 2) both hemispheres quadrature nodes
    NLayers, NBDRF,                           # Number of 1) layers; 2) BDRF Fourier modes
    is_atmos_multilayered,                    # Is the atmosphere multilayered?
    BDRF_Fourier_modes,                       # BDRF Fourier modes
    mu0, I0,                                  # Properties of the direct beam
    there_is_beam_source,                     # Is there a beam source?
    b_pos, b_neg,                             # Dirichlet BCs
    b_pos_is_scalar, b_neg_is_scalar,         # Is each Dirichlet BCs scalar and so isotropic?
    b_pos_is_vector, b_neg_is_vector,         # Is each Dirichlet BCs vector?
    Nscoeffs,                                 # Number of isotropic source polynomial coefficients
    scaled_s_poly_coeffs,                     # Polynomial coefficients of isotropic source           
    there_is_iso_source,                      # Is there an isotropic source?
    use_banded_solver_NLayers,                # Number of layers above or equal which to use `scipy.linalg.solve_banded`
):
    """
    Uses the boundary conditions to solve for the unknown coefficients 
    of the general solution to the system of ordinary differential equations for each Fourier mode. 
    Returns the product of the coefficients and the eigenvectors.
    This function is wrapped by the `_assemble_intensity_and_fluxes` function.
    It has many seemingly redundant arguments to maximize precomputation in the `pydisort` function.
    See the Jupyter Notebook, especially section 3, for documentation, explanation and derivation.
    The labels in this file reference labels in the Jupyter Notebook, especially sections 3 and 4.

    Arguments of _solve_for_coeffs
    |            Variable            |                  Type / Shape                |
    | ------------------------------ | -------------------------------------------- |
    | `NFourier`                     | scalar                                       |
    | `G_collect`                    | `NFourier x NLayers x NQuad x NQuad`         |
    | `K_collect`                    | `NFourier x NLayers x NQuad`                 |
    | `B_collect`                    | `NFourier x NLayers x NQuad` or `None`       |
    | `G_inv_collect_0`              | `NLayers x NQuad x NQuad` or `None`          |
    | `scaled_tau_arr_with_0`        | `NLayers + 1`                                |
    | `mu_arr`                       | `NQuad`                                      |
    | `mu_arr_pos`                   | `NQuad/2`                                    |
    | `mu_arr_pos_times_W`           | `NQuad/2`                                    |
    | `N`                            | scalar                                       |
    | `NQuad`                        | scalar                                       |
    | `NLayers`                      | scalar                                       |
    | `NBDRF`                        | scalar                                       |
    | `is_atmos_multilayered`        | boolean                                      |
    | `BDRF_Fourier_modes`           | `NBDRF`                                      |
    | `mu0`                          | scalar                                       |
    | `I0`                           | scalar                                       |
    | `there_is_beam_source`         | boolean                                      |
    | `b_pos`                        | `NQuad/2 x NFourier` or `NQuad/2` or scalar  |
    | `b_neg`                        | `NQuad/2 x NFourier` or `NQuad/2` or scalar  | 
    | `b_pos_is_scalar`              | boolean                                      |
    | `b_neg_is_scalar`              | boolean                                      |
    | `b_pos_is_vector`              | boolean                                      |
    | `b_neg_is_vector`              | boolean                                      |
    | `Nscoeffs`                     | scalar                                       |
    | `scaled_s_poly_coeffs`         | `NLayers x Nscoeffs`                         |
    | `there_is_iso_source`          | boolean                                      |
    | `use_banded_solver_NLayers`    | scalar                                       |
    
    Notable internal variables of _solve_for_coeffs
    |   Variable   |                 Shape                |
    | ------------ | ------------------------------------ |
    | `GC_collect` | `NFourier x NLayers x NQuad x NQuad` |

    """
    ################################## Solve for coefficients of homogeneous solution ##########################################
    ############# Refer to section 3.6.2 (single-layer) and 4 (multi-layer) of the Comprehensive Documentation #################
    
    GC_collect = np.empty((NFourier, NLayers, NQuad, NQuad))
    use_banded_solver = (NLayers >= use_banded_solver_NLayers)
    dim = NLayers * NQuad
    
    i_bot = dim - N # equals `j_bot_right`
    j_bot_left = dim - NQuad
    if use_banded_solver:
        Nsupsubdiags = 3 * N - 1
        LHS_dof = np.zeros((6 * N - 1, dim), order="F") # diagonal ordered form
        s0, s1 = LHS_dof.strides  # bytes
        col_stride = s1 - s0  # moving +1 col and -1 row
        
        def _view_slanted(i0, j0, nrows):
            """
            Return a view into `LHS_dof` using slanted mapping.
            The function uses `numpy.lib.stride_tricks.as_strided` which is performant but dangerous, 
            which is why this function is not exposed in `PythonicDISORT.subroutines`.
            See https://numpy.org/devdocs/reference/generated/numpy.lib.stride_tricks.as_strided.html.
            
            Assume the following exist in the outer scope:
                `LHS_dof`, `Nsupsubdiags`, `s0`, `col_stride`
            """
            r0 = Nsupsubdiags + i0 - j0
            base = LHS_dof[r0:r0 + nrows, j0:j0 + N]
            return as_strided(base, shape=(nrows, N), strides=(s0, col_stride))
            
    else:
        LHS = np.zeros((dim, dim))
    
    # The following loops can easily be parallelized, but the speed-up is unlikely to be worth the overhead
    for m in range(NFourier):
        m_equals_0 = (m == 0)
        there_is_BDRF_mode = (NBDRF > m) 
        
        G_collect_m = G_collect[m, :, :, :]
        K_collect_m = K_collect[m, :, :]
        if there_is_beam_source:
            B_collect_m = B_collect[m, :, :]
            
        # Generate mathscr_D and mathscr_X (BDRF terms)
        # Just for this part, refer to section 3.4.2 of the Comprehensive Documentation 
        # --------------------------------------------------------------------------------------------------------------------------
        if there_is_BDRF_mode:
            BDRF_Fourier_modes_m = BDRF_Fourier_modes[m]
            if np.isscalar(BDRF_Fourier_modes_m):
                mathscr_D_neg = (1 + m_equals_0 * 1) * BDRF_Fourier_modes_m
                R = mathscr_D_neg * mu_arr_pos_times_W[None, :]
                if there_is_beam_source:
                    mathscr_X_pos = (mu0 * I0 / pi) * BDRF_Fourier_modes_m
            else:
                mathscr_D_neg = (1 + m_equals_0 * 1) * BDRF_Fourier_modes_m(mu_arr_pos, mu_arr_pos)
                R = mathscr_D_neg * mu_arr_pos_times_W[None, :]
                if there_is_beam_source:
                    mathscr_X_pos = (mu0 * I0 / pi) * BDRF_Fourier_modes_m(
                        mu_arr_pos, np.array([mu0])
                    )[:, 0]
        # --------------------------------------------------------------------------------------------------------------------------
    
    
        # Assemble RHS
        # --------------------------------------------------------------------------------------------------------------------------
        # Ensure the BCs are of the correct shape
        if b_pos_is_scalar and m_equals_0:
            b_pos_m = np.full(N, b_pos)
        elif b_pos_is_vector and m_equals_0:
            b_pos_m = b_pos
        elif (b_pos_is_scalar or b_pos_is_vector) and not m_equals_0:
            b_pos_m = np.zeros(N)
        else:
            b_pos_m = b_pos[:, m]
            
        if b_neg_is_scalar and m_equals_0:
            b_neg_m = np.full(N, b_neg)
        elif b_neg_is_vector and m_equals_0:
            b_neg_m = b_neg
        elif (b_neg_is_scalar or b_neg_is_vector) and not m_equals_0:
            b_neg_m = np.zeros(N)
        else:
            b_neg_m = b_neg[:, m]
        
        # _mathscr_v_contribution
        if m_equals_0 and there_is_iso_source:
            _mathscr_v_contribution_top = -_mathscr_v(
                        np.array([0]), 
                        np.array([0]),
                        Nscoeffs,
                        scaled_s_poly_coeffs[[0], :],
                        G_collect_m[[0], N:, :],
                        K_collect_m[[0], :],
                        G_inv_collect_0[[0], :, :],
                        mu_arr,
                    ).ravel()
        
            _mathscr_v_contribution_middle = np.array([])
            if is_atmos_multilayered:
                indices = np.arange(NLayers - 1)
                _mathscr_v_contribution_middle = (
                    _mathscr_v(
                        scaled_tau_arr_with_0[1:-1],
                        indices,
                        Nscoeffs,
                        scaled_s_poly_coeffs[indices + 1],
                        G_collect_m[indices + 1],
                        K_collect_m[indices + 1],
                        G_inv_collect_0[indices + 1],
                        mu_arr,
                    )
                    - _mathscr_v(
                        scaled_tau_arr_with_0[1:-1],
                        indices,
                        Nscoeffs,
                        scaled_s_poly_coeffs[indices],
                        G_collect_m[indices],
                        K_collect_m[indices],
                        G_inv_collect_0[indices],
                        mu_arr,
                    )
                ).ravel(order='F')
            
            _mathscr_v_contribution_bottom = -_mathscr_v(
                    scaled_tau_arr_with_0[[-1]], 
                    np.array([0]),
                    Nscoeffs,
                    scaled_s_poly_coeffs[[-1], :],
                    G_collect_m[[-1], :N, :],
                    K_collect_m[[-1], :],
                    G_inv_collect_0[[-1], :, :],
                    mu_arr
                ).ravel()
            if NBDRF > 0:
                _mathscr_v_contribution_bottom = (
                    _mathscr_v_contribution_bottom
                    + R
                    @ _mathscr_v(
                        scaled_tau_arr_with_0[[-1]],
                        np.array([0]),
                        Nscoeffs,
                        scaled_s_poly_coeffs[[-1], :],
                        G_collect_m[[-1], N:, :],
                        K_collect_m[[-1], :],
                        G_inv_collect_0[[-1], :, :],
                        mu_arr
                    ).ravel()
                )
                    
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
            RHS_middle = np.array([])
            if is_atmos_multilayered:
                l_range = np.arange(1, NLayers)
                RHS_middle = (
                    (B_collect_m[l_range, :] - B_collect_m[l_range - 1, :])
                    * np.exp(-scaled_tau_arr_with_0 / mu0)[l_range, None]
                ).ravel()
                
            if there_is_BDRF_mode:
                RHS = (
                    np.concatenate(
                        [
                            b_neg_m - B_collect_m[0, N:],
                            RHS_middle,
                            b_pos_m
                            + (mathscr_X_pos + R @ B_collect_m[-1, N:] - B_collect_m[-1, :N])
                            * np.exp(-scaled_tau_arr_with_0[-1] / mu0),
                        ]
                    )
                    + _mathscr_v_contribution
                )
            else:
                RHS = (
                    np.concatenate(
                        [
                            b_neg_m - B_collect_m[0, N:],
                            RHS_middle,
                            b_pos_m
                            - B_collect_m[-1, :N] * np.exp(-scaled_tau_arr_with_0[-1] / mu0),
                        ]
                    )
                    + _mathscr_v_contribution
                )
        else:
            RHS_middle = np.zeros(NQuad * (NLayers - 1))
            RHS = np.concatenate([b_neg_m, RHS_middle, b_pos_m]) + _mathscr_v_contribution
            
        # --------------------------------------------------------------------------------------------------------------------------
        
        
        # Assemble LHS and solve (much of this code is replicated in section 4 of the Comprehensive Documentation)
        # --------------------------------------------------------------------------------------------------------------------------
        G_0 = G_collect_m[0]
        G_0_nn = G_0[N:, :N]
        G_0_np = G_0[N:, N:]
        E_01 = np.exp(K_collect_m[0, :N] * scaled_tau_arr_with_0[1])[None, :]
        
        G_L = G_collect_m[-1]
        G_L_pn = G_L[:N, :N]
        G_L_nn = G_L[N:, :N]
        G_L_pp = G_L[:N, N:]
        G_L_np = G_L[N:, N:]
        E_Lm1L = np.exp(
            K_collect_m[-1, :N] * (scaled_tau_arr_with_0[-1] - scaled_tau_arr_with_0[-2])
        )[None, :]
        
        ################################## Use `sc.linalg.solve_banded` ##################################
        if use_banded_solver:      
             
            # ---------------- Top BC ----------------
            G_0 = G_collect_m[0]
            G_0_nn = G_0[N:, :N]     # (N,N)
            G_0_np = G_0[N:, N:]     # (N,N)

            _view_slanted(0, 0, N)[:] = G_0_nn
            _view_slanted(0, N, N)[:] = G_0_np * E_01

            # ---------------- Bottom BC ----------------
            if there_is_BDRF_mode:
                _view_slanted(i_bot, j_bot_left, N)[:] = (G_L_pn - R @ G_L_nn) * E_Lm1L
                _view_slanted(i_bot, i_bot, N)[:] = G_L_pp - R @ G_L_np
            else:
                _view_slanted(i_bot, j_bot_left, N)[:] = G_L_pn * E_Lm1L
                _view_slanted(i_bot, i_bot, N)[:] = G_L_pp

            # ---------------- Interlayer / continuity BCs ----------------
            for l in range(NLayers - 1):
                G_l = G_collect_m[l]
                G_l_pn = G_l[:N, :N]      # (N,N)
                G_l_nn = G_l[N:, :N]      # (N,N)
                G_l_ap = G_l[:, N:]       # (2N,N)

                G_lp1 = G_collect_m[l + 1]
                G_lp1_an = G_lp1[:, :N]   # (2N,N)
                G_lp1_pp = G_lp1[:N, N:]  # (N,N)
                G_lp1_np = G_lp1[N:, N:]  # (N,N)

                tau_lm1 = scaled_tau_arr_with_0[l]
                tau_l   = scaled_tau_arr_with_0[l + 1]
                tau_lp1 = scaled_tau_arr_with_0[l + 2]

                E_lm1l = np.exp(K_collect_m[l,     N:] * (tau_lm1 - tau_l))[None, :]    # (N,)
                E_llp1 = np.exp(K_collect_m[l + 1, N:] * (tau_l   - tau_lp1))[None, :]  # (N,)

                start_row = N + l * NQuad
                start_col = l * NQuad

                _view_slanted(start_row,     start_col,     N)[:] = G_l_pn * E_lm1l
                _view_slanted(start_row + N, start_col,     N)[:] = G_l_nn * E_lm1l
                _view_slanted(start_row,     start_col + N, NQuad)[:] = G_l_ap

                _view_slanted(start_row,     start_col + NQuad, NQuad)[:] = -G_lp1_an
                _view_slanted(start_row,     start_col + 3 * N, N)[:] = -G_lp1_pp * E_llp1
                _view_slanted(start_row + N, start_col + 3 * N, N)[:] = -G_lp1_np * E_llp1
        
            C_m = sc.linalg.solve_banded(
                (Nsupsubdiags, Nsupsubdiags),
                LHS_dof,
                RHS,
                True,
                True,
                False,
            )
        
        ##################################################################################################
        ##################################### Use `np.linalg.solve` ######################################
        else:
            # ---------------- Top BC ----------------
            LHS[:N, :N] = G_0_nn
            LHS[:N, N : NQuad] = G_0_np * E_01
            
            # ---------------- Bottom BC ----------------
            if there_is_BDRF_mode:
                LHS[-N:, -NQuad : -N] = (G_L_pn - R @ G_L_nn) * E_Lm1L
                LHS[-N:, -N:] = G_L_pp - R @ G_L_np
            else:
                LHS[-N:, -NQuad : -N] = G_L_pn * E_Lm1L
                LHS[-N:, -N:] = G_L_pp

            # ---------------- Interlayer / continuity BCs ----------------
            for l in range(NLayers - 1):
                G_l = G_collect_m[l]
                G_l_pn = G_l[:N, :N]
                G_l_nn = G_l[N:, :N]
                G_l_ap = G_l[:, N:]
                
                G_lp1 = G_collect_m[l + 1]
                G_lp1_an = G_lp1[:, :N]
                G_lp1_pp = G_lp1[:N, N:]
                G_lp1_np = G_lp1[N:, N:]
                
                scaled_tau_arr_lm1 = scaled_tau_arr_with_0[l]
                scaled_tau_arr_l = scaled_tau_arr_with_0[l + 1]
                scaled_tau_arr_lp1 = scaled_tau_arr_with_0[l + 2]
                
                # Postive eigenvalues
                E_lm1l = np.exp(K_collect_m[l, N:] * (scaled_tau_arr_lm1 - scaled_tau_arr_l))
                E_llp1 = np.exp(K_collect_m[l + 1, N:] * (scaled_tau_arr_l - scaled_tau_arr_lp1))
                
                start_row = N + l * NQuad
                start_col = l * NQuad
                
                LHS[start_row : N + start_row, start_col : N + start_col] = G_l_pn * E_lm1l
                LHS[N + start_row : NQuad + start_row, start_col : N + start_col] = G_l_nn * E_lm1l
                LHS[start_row : NQuad + start_row, N + start_col : NQuad + start_col] = G_l_ap
                
                LHS[start_row : NQuad + start_row, NQuad + start_col : 3 * N + start_col] = -G_lp1_an
                LHS[start_row : N + start_row, 3 * N + start_col : 4 * N + start_col] = -G_lp1_pp * E_llp1
                LHS[N + start_row : NQuad + start_row, 3 * N + start_col : 4 * N + start_col] = -G_lp1_np * E_llp1
                
            # Solve the system
            C_m = np.linalg.solve(LHS, RHS)
                
            ################################################################################################## 
            # --------------------------------------------------------------------------------------------------------------------------

        GC_collect[m, :, :, :] = G_collect_m * C_m.reshape(NLayers, NQuad)[:, None, :]
        
    return GC_collect